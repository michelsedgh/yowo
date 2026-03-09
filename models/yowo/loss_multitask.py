"""
Multi-Task Loss for Action Genome + Charades

Matches the YOWOMultiTask model:
- Object (0-35): CrossEntropy (exclusive)
- Actions: Focal Loss with IoU weighting (handles class imbalance)
- Relations: Focal Loss with IoU weighting (handles class imbalance)

UPDATED: Now uses Focal Loss instead of per-class weights for better
handling of class imbalance. Focal Loss automatically downweights easy/common
examples and upweights hard/rare examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class SigmoidFocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification (BCE-based).
    
    Focal Loss automatically handles class imbalance by:
    - Downweighting easy examples (high confidence correct predictions)
    - Upweighting hard examples (low confidence or wrong predictions)
    
    Formula: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Balance factor for positive/negative (default 0.25)
        gamma: Focusing parameter (default 2.0) 
               Higher gamma = more focus on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [N, C] - raw predictions (before sigmoid)
            targets: [N, C] - target labels (0 or 1, or IoU-weighted 0-1)
        Returns:
            loss: [N, C] or scalar depending on reduction
        """
        # Ensure targets are pure floats
        targets = targets.float()

        # 1. Use pure PyTorch C++ BCEWithLogits for supreme FP16 numerical stability
        ce_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=targets, reduction="none"
        )
        
        # 2. Extract probability (using sigmoid)
        p = torch.sigmoid(logits)
        
        # 3. Calculate focal term: (1 - p_t)^gamma
        #    p_t = p if target=1, else (1-p)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        
        # 4. CRITICAL FP16 FIX:
        # Clamp p_t to prevent (1 - p_t)^gamma from causing log(0)
        # underflows entirely deep in PyTorch autograd engine.
        p_t = p_t.clamp(min=1e-5, max=1.0 - 1e-5)
        
        # 5. Apply the Focal modulator
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        
        # 6. Alpha balancing (handle soft targets properly)
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class MultiTaskCriterion(object):
    """
    Multi-task loss for YOWOMultiTask model.
    
    Uses:
    - CrossEntropy for objects (exclusive class)
    - Focal Loss for actions (multi-label, handles imbalance)
    - Focal Loss for relations (multi-label, handles imbalance)
    
    CRITICAL: Targets are HARD (0/1). IoU quality is applied as a LOSS
    weight, NOT as target modification. Soft targets (target * iou) cause
    the model to converge to p=iou instead of p=1.0, which is wrong.
    
    Expects outputs with separate pred_obj, pred_act, pred_rel.
    """
    
    def __init__(self, args, img_size, num_classes=219, 
                 num_objects=36, num_actions=157, num_relations=26,
                 use_focal_loss=True):
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_relations = num_relations
        
        self.loss_conf_weight = args.loss_conf_weight
        self.loss_cls_weight = args.loss_cls_weight
        self.loss_reg_weight = args.loss_reg_weight
        self.use_focal_loss = use_focal_loss
        
        # Loss functions
        # Confidence loss: Focal Loss to prevent dominating other losses
        # gamma=2.5 matches action/relation focal losses for aggressive hard example focus
        # alpha=0.25 upweights rare foreground anchors (background vastly outnumbers fg)
        self.conf_lossf = SigmoidFocalLoss(alpha=0.25, gamma=2.5, reduction='none')
        print(f"  Using Focal Loss for confidence (gamma=2.5, alpha=0.25)")
        
        # Object class weights - sqrt(max_count / class_count) from actual eval data
        # Recalculated from E5 export: person=13528, table=2050, chair=1540, etc.
        # Formula: sqrt(max_samples / class_samples), capped at 8.0
        # max_samples = 13528 (person)
        obj_class_weights = torch.tensor([
            0.3,   # 0: person (13528 samples) - heavily downweight, dominates dataset
            6.98,  # 1: bag (278 samples)
            4.18,  # 2: bed (775 samples)
            5.29,  # 3: blanket (484 samples)
            3.89,  # 4: book (893 samples)
            7.14,  # 5: box (265 samples)
            8.0,   # 6: broom (70 samples) - capped
            2.96,  # 7: chair (1540 samples)
            8.0,   # 8: closetcabinet (est ~50) - capped
            6.08,  # 9: clothes (366 samples)
            8.0,   # 10: cupglassbottle (est ~200)
            4.59,  # 11: dish (642 samples)
            4.35,  # 12: door (714 samples)
            8.0,   # 13: doorknob (111 samples)
            4.56,  # 14: doorway (651 samples)
            5.40,  # 15: floor (464 samples)
            3.29,  # 16: food (1249 samples)
            8.0,   # 17: groceries (39 samples) - capped
            4.64,  # 18: laptop (629 samples)
            8.0,   # 19: light (2 samples) - capped
            8.0,   # 20: medicine (68 samples) - capped
            8.0,   # 21: mirror (181 samples)
            8.0,   # 22: papernotebook (est ~100) - capped
            8.0,   # 23: phonecamera (est ~150)
            8.0,   # 24: picture (101 samples) - capped
            8.0,   # 25: pillow (173 samples)
            8.0,   # 26: refrigerator (113 samples)
            6.45,  # 27: sandwich (325 samples)
            8.0,   # 28: shelf (181 samples)
            8.0,   # 29: shoe (149 samples)
            8.0,   # 30: sofacouch (est ~200)
            2.57,  # 31: table (2050 samples)
            8.0,   # 32: television (178 samples)
            8.0,   # 33: towel (163 samples)
            8.0,   # 34: vacuum (151 samples)
            8.0,   # 35: window (55 samples) - capped
        ], dtype=torch.float32)
        self.register_buffer_fn = None  # Will move to device when needed
        self.obj_class_weights = obj_class_weights
        self.obj_lossf = nn.CrossEntropyLoss(weight=obj_class_weights, reduction='none')
        print(f"  Using object class weights (rare classes upweighted 8-15x)")
        
        # Action class weights - loaded from smart_home config if available
        # These weights upweight rare actions (e.g., "awakening" 2.18x, "shoes" 2.45x)
        # Focal Loss handles easy/hard imbalance, class weights handle frequency imbalance
        self.act_class_weights = None
        try:
            import json
            import os
            config_path = os.path.join(os.path.dirname(__file__), '../../config/smart_home_final.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    smart_home_cfg = json.load(f)
                if 'action_class_weights' in smart_home_cfg and num_actions == smart_home_cfg.get('num_actions', 0):
                    self.act_class_weights = torch.tensor(smart_home_cfg['action_class_weights'], dtype=torch.float32)
                    print(f"  Using action class weights from smart_home config ({len(self.act_class_weights)} classes, range {self.act_class_weights.min():.2f}-{self.act_class_weights.max():.2f})")
        except Exception as e:
            print(f"  Warning: Could not load action class weights: {e}")
        
        # Focal Loss for actions and relations (handles class imbalance)
        if use_focal_loss:
            print("  Using Focal Loss (gamma=2.5, alpha=0.75) for actions and relations")
            # gamma=2.5: AGGRESSIVE focus on hard examples - prevents lazy predictions
            # The model was hedging bets (predicting ~0.4 for everything). gamma=2.5 forces
            # it to commit: confident wrong predictions get heavily penalized.
            # alpha=0.75 upweights positive samples (good for rare action classes)
            self.act_lossf = SigmoidFocalLoss(alpha=0.75, gamma=2.5, reduction='none')
            self.rel_lossf = SigmoidFocalLoss(alpha=0.75, gamma=2.5, reduction='none')
        else:
            print("  Using standard BCE for actions and relations")
            self.act_lossf = nn.BCEWithLogitsLoss(reduction='none')
            self.rel_lossf = nn.BCEWithLogitsLoss(reduction='none')
        
        # One-to-Many Matcher (standard, topk>1 for rich supervision)
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=args.center_sampling_radius,
            topk_candidate=args.topk_candicate
        )
        
        # One-to-One Matcher (topk=1 for NMS-free)
        self.matcher_o2o = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=args.center_sampling_radius,
            topk_candidate=1  # CRITICAL: topk=1 forces single anchor per object
        )

    def __call__(self, outputs, targets):
        # Check if dual-head (end2end) outputs
        if 'one2many' in outputs:
            # Dual-head mode: compute both losses
            o2m_loss = self._compute_loss(outputs['one2many'], targets, self.matcher)
            o2o_loss = self._compute_loss(outputs['one2one'], targets, self.matcher_o2o)
            
            # Combine losses (equal weighting like ultralytics)
            combined = {}
            for key in o2m_loss:
                if key == 'losses':
                    combined[key] = o2m_loss[key] + o2o_loss[key]
                else:
                    combined[key] = o2m_loss[key] + o2o_loss[key]
            
            # Add per-branch losses for monitoring
            combined['o2m_loss'] = o2m_loss['losses']
            combined['o2o_loss'] = o2o_loss['losses']
            
            return combined
        else:
            # Single-head mode (backward compatible)
            return self._compute_loss(outputs, targets, self.matcher)

    def _compute_loss(self, outputs, targets, matcher):
        """Compute loss for a single head (O2M or O2O)."""
        bs = outputs['pred_obj'][0].shape[0]
        device = outputs['pred_obj'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        
        # Concatenate predictions
        conf_preds = torch.cat(outputs['pred_conf'], dim=1)
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)
        act_preds = torch.cat(outputs['pred_act'], dim=1)
        rel_preds = torch.cat(outputs['pred_rel'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        
        # Combined cls for matcher - convert object logits to sigmoid-compatible
        # form. Object head uses CrossEntropy (softmax) but the matcher applies
        # sigmoid() to all logits. We convert: softmax → probs → inverse_sigmoid
        # so that sigmoid(inverse_sigmoid(softmax_prob)) = softmax_prob.
        with torch.no_grad():
            obj_probs_sm = F.softmax(obj_preds.detach(), dim=-1)
            obj_probs_sm = obj_probs_sm.clamp(1e-6, 1 - 1e-6)
            obj_logits_for_matcher = torch.log(obj_probs_sm / (1 - obj_probs_sm))
        cls_preds = torch.cat([obj_logits_for_matcher, act_preds, rel_preds], dim=-1)
        
        # Label assignment
        obj_targets = []
        act_targets = []
        rel_targets = []
        box_targets = []
        conf_targets = []
        fg_masks = []
        is_person_masks = []  # Track which matched GT is a Person (for action masking)

        for batch_idx in range(bs):
            tgt_labels_cpu = targets[batch_idx]["labels"]
            tgt_bboxes_cpu = targets[batch_idx]["boxes"]

            if tgt_labels_cpu.numel() == 0 or tgt_bboxes_cpu.numel() == 0 or not torch.any(tgt_bboxes_cpu != 0):
                num_anchors = sum([ab.shape[0] for ab in anchors])
                obj_target = conf_preds.new_zeros((0,), dtype=torch.long)
                act_target = conf_preds.new_zeros((0, self.num_actions))
                rel_target = conf_preds.new_zeros((0, self.num_relations))
                box_target = conf_preds.new_zeros((0, 4))
                conf_target = conf_preds.new_zeros((num_anchors, 1))
                fg_mask = conf_preds.new_zeros(num_anchors).bool()
                is_person_mask = conf_preds.new_zeros((0,)).bool()
            else:
                tgt_labels = tgt_labels_cpu.to(device)
                tgt_bboxes = tgt_bboxes_cpu.to(device)
                tgt_bboxes_scaled = tgt_bboxes * self.img_size
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious,
                    matched_gt_inds,
                    num_fg,
                ) = matcher(
                    fpn_strides=fpn_strides,
                    anchors=anchors,
                    pred_conf=conf_preds[batch_idx],
                    pred_cls=cls_preds[batch_idx],
                    pred_box=box_preds[batch_idx],
                    tgt_labels=tgt_labels,
                    tgt_bboxes=tgt_bboxes_scaled,
                )

                conf_target = fg_mask.unsqueeze(-1).float()
                box_target = tgt_bboxes_scaled[matched_gt_inds]
                
                # Split labels
                matched_labels = tgt_labels[matched_gt_inds]
                
                # Object: argmax (exclusive class)
                obj_target = matched_labels[:, :self.num_objects].argmax(dim=-1)
                
                # Actions & Relations: HARD (0/1) targets
                # IoU quality applied as loss WEIGHT, not target modification.
                # Soft targets (target * iou) cause BCE to converge to p=iou
                # instead of p=1.0, which is fundamentally wrong.
                act_target = matched_labels[:, self.num_objects:self.num_objects+self.num_actions]
                rel_target = matched_labels[:, self.num_objects+self.num_actions:]
                
                # Person mask: object class 0 is "person" - for action loss masking
                is_person_mask = (obj_target == 0)

            obj_targets.append(obj_target)
            act_targets.append(act_target)
            rel_targets.append(rel_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)
            is_person_masks.append(is_person_mask)

        # Concatenate
        obj_targets = torch.cat(obj_targets, dim=0)
        act_targets = torch.cat(act_targets, dim=0)
        rel_targets = torch.cat(rel_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        conf_targets = torch.cat(conf_targets, dim=0)
        fg_masks = torch.cat(fg_masks, dim=0)
        is_person_masks = torch.cat(is_person_masks, dim=0)  # [total_fg]
        
        num_fg = fg_masks.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fg)
        num_fg = (num_fg / get_world_size()).clamp(1.0)

        # Confidence loss
        loss_conf = self.conf_lossf(conf_preds.view(-1, 1), conf_targets)
        loss_conf = loss_conf.sum() / num_fg

        # Object loss (CrossEntropy - exclusive class, with class weights)
        matched_obj_preds = obj_preds.view(-1, self.num_objects)[fg_masks]
        if len(obj_targets) > 0:
            # Move class weights to correct device if needed
            if self.obj_class_weights.device != device:
                self.obj_class_weights = self.obj_class_weights.to(device)
                self.obj_lossf = nn.CrossEntropyLoss(weight=self.obj_class_weights, reduction='none')
            loss_obj = self.obj_lossf(matched_obj_preds, obj_targets)
            loss_obj = loss_obj.sum() / num_fg
            
            # OBJECT MARGIN LOSS: Enforce GT logit > max(non-GT logits) + margin
            # This is the softmax equivalent of gap loss for multi-label heads
            # Aggressive values to ensure strong class separation
            N = matched_obj_preds.shape[0]
            gt_logits = matched_obj_preds[torch.arange(N, device=device), obj_targets]  # [N]
            # Mask out GT class to find max non-GT logit
            obj_mask = torch.ones_like(matched_obj_preds, dtype=torch.bool)
            obj_mask[torch.arange(N, device=device), obj_targets] = False
            non_gt_logits = matched_obj_preds.masked_fill(~obj_mask, float('-inf'))
            max_non_gt = non_gt_logits.max(dim=1).values  # [N]
            # margin=1.0 in logit space ≈ 2.7x probability ratio (e^1.0)
            # weight=0.5 matches action gap loss weight for consistency
            obj_margin = 1.0
            obj_gap_violations = F.relu(obj_margin - (gt_logits - max_non_gt))  # [N]
            obj_gap_loss = obj_gap_violations.mean()
            loss_obj = loss_obj + 0.5 * obj_gap_loss
        else:
            loss_obj = torch.tensor(0.0, device=device)

        # Action loss (Focal/BCE - multi-label, PERSON-ONLY)
        # Uses class weights to upweight rare actions (e.g., "awakening" 2.18x)
        matched_act_preds = act_preds.view(-1, self.num_actions)[fg_masks]
        if len(act_targets) > 0 and is_person_masks.sum() > 0:
            person_act_preds = matched_act_preds[is_person_masks]
            person_act_targets = act_targets[is_person_masks]
            loss_act = self.act_lossf(person_act_preds, person_act_targets)  # [N, C]
            # Apply action class weights if available (upweights rare actions)
            if self.act_class_weights is not None:
                if self.act_class_weights.device != device:
                    self.act_class_weights = self.act_class_weights.to(device)
                loss_act = loss_act * self.act_class_weights.unsqueeze(0)  # [N, C] * [1, C]
            # CRITICAL FIX: Divide by PERSON fg count, not all fg
            # Action loss is only computed on person boxes, so normalization should match
            # Previously divided by num_fg (all boxes) which diluted the gradient by ~3x
            num_person_fg = max(is_person_masks.sum().float(), 1.0)
            loss_act = loss_act.sum() / num_person_fg
            
            # GAP LOSS (vectorized): Enforce margin between positive and negative predictions
            # Prevents "lazy model" predicting ~0.4 for everything
            pred_probs = torch.sigmoid(person_act_preds)  # [N, C]
            pos_mask = person_act_targets > 0.5  # [N, C]
            neg_mask = ~pos_mask
            
            # Per-class mean of predictions on positive vs negative samples
            pos_sum = (pred_probs * pos_mask.float()).sum(0)   # [C]
            pos_count = pos_mask.float().sum(0).clamp(min=1)   # [C]
            neg_sum = (pred_probs * neg_mask.float()).sum(0)   # [C]
            neg_count = neg_mask.float().sum(0).clamp(min=1)   # [C]
            mean_pos = pos_sum / pos_count  # [C]
            mean_neg = neg_sum / neg_count  # [C]
            
            # Only penalize classes with both pos and neg in this batch
            has_both = (pos_mask.any(0) & neg_mask.any(0))  # [C]
            act_margin = 0.15
            gap_violations = F.relu(act_margin - (mean_pos - mean_neg))  # [C]
            gap_loss_act = (gap_violations * has_both.float()).mean()
            loss_act = loss_act + 0.5 * gap_loss_act
        else:
            loss_act = torch.tensor(0.0, device=device)

        # Relation loss (Focal/BCE - multi-label) + gap enforcement
        matched_rel_preds = rel_preds.view(-1, self.num_relations)[fg_masks]
        if len(rel_targets) > 0:
            loss_rel = self.rel_lossf(matched_rel_preds, rel_targets)  # [N, C]
            loss_rel = loss_rel.sum() / num_fg
            
            # GAP LOSS for relations (same lazy model problem)
            rel_probs = torch.sigmoid(matched_rel_preds)  # [N, C]
            rel_pos_mask = rel_targets > 0.5
            rel_neg_mask = ~rel_pos_mask
            
            rel_pos_sum = (rel_probs * rel_pos_mask.float()).sum(0)
            rel_pos_count = rel_pos_mask.float().sum(0).clamp(min=1)
            rel_neg_sum = (rel_probs * rel_neg_mask.float()).sum(0)
            rel_neg_count = rel_neg_mask.float().sum(0).clamp(min=1)
            rel_mean_pos = rel_pos_sum / rel_pos_count
            rel_mean_neg = rel_neg_sum / rel_neg_count
            
            rel_has_both = (rel_pos_mask.any(0) & rel_neg_mask.any(0))
            rel_margin = 0.10  # Slightly lower margin for relations (already learning better)
            rel_gap_violations = F.relu(rel_margin - (rel_mean_pos - rel_mean_neg))
            gap_loss_rel = (rel_gap_violations * rel_has_both.float()).mean()
            loss_rel = loss_rel + 0.3 * gap_loss_rel
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # Box loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        if len(box_targets) > 0:
            ious = get_ious(matched_box_preds, box_targets, box_mode="xyxy", iou_type='giou')
            loss_box = (1.0 - ious).sum() / num_fg
        else:
            loss_box = torch.tensor(0.0, device=device)

        # Total
        loss_cls = loss_obj + loss_act + loss_rel
        losses = (
            self.loss_conf_weight * loss_conf +
            self.loss_cls_weight * loss_cls +
            self.loss_reg_weight * loss_box
        )

        return dict(
            loss_conf=loss_conf,
            loss_cls=loss_cls,
            loss_obj=loss_obj,
            loss_act=loss_act,
            loss_rel=loss_rel,
            loss_box=loss_box,
            losses=losses
        )


def build_multitask_criterion(args, img_size, num_classes=219,
                               num_objects=36, num_actions=157, num_relations=26,
                               use_focal_loss=True):
    """
    Build the multi-task criterion.
    
    Args:
        use_focal_loss: If True, use Focal Loss for actions/relations
                       (recommended, handles class imbalance automatically)
                       If False, use standard BCE
    """
    return MultiTaskCriterion(
        args, img_size, num_classes, 
        num_objects, num_actions, num_relations,
        use_focal_loss=use_focal_loss
    )

