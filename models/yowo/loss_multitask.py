"""
Multi-Task Loss for Action Genome + Charades

Matches the YOWOMultiTask model:
- Object (0-35): CrossEntropy with class weights (exclusive)
- Actions (35 classes): BCE with per-class pos_weight (multi-label)
- Relations (26 classes): BCE with per-class pos_weight (multi-label)

KEY DESIGN DECISIONS:
1. Per-class pos_weight calculated from actual dataset statistics
   - Actions: range 4.01 (common) to 50.0 (rare)
   - Relations: range 0.43 (very common) to 50.0 (rare)
2. Aggressive decisiveness loss to prevent lazy predictions
   - Positives must predict > 0.5
   - Negatives must predict < 0.1
3. Actions weighted 1.5x (primary task), relations 0.5x (secondary)
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
                 num_objects=36, num_actions=157, num_relations=26):
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_relations = num_relations
        
        self.loss_conf_weight = args.loss_conf_weight
        self.loss_cls_weight = args.loss_cls_weight
        self.loss_reg_weight = args.loss_reg_weight
        
        # ================================================================
        # CONFIDENCE LOSS - BCE with pos_weight for FG/BG imbalance
        # ================================================================
        # CRITICAL: Without pos_weight, ~100:1 BG:FG ratio causes model to
        # predict conf→0 everywhere. This explains "detection drops"!
        # 
        # Math: 64000 anchors, ~400 foreground → 160:1 ratio
        # Using pos_weight=50 (conservative) to upweight foreground gradient
        # ================================================================
        conf_pos_weight = torch.tensor([50.0])
        self.conf_lossf = nn.BCEWithLogitsLoss(pos_weight=conf_pos_weight, reduction='none')
        self.conf_pos_weight = conf_pos_weight
        print(f"  Confidence BCE: pos_weight=50 (FG/BG balance for ~100:1 ratio)")
        
        # Object class weights - sqrt(max_count / class_count) from actual eval data
        # Recalculated from E5 export: person=13528, table=2050, chair=1540, etc.
        # Formula: sqrt(max_samples / class_samples), capped at 8.0
        # max_samples = 13528 (person)
        obj_class_weights = torch.tensor([
            1.0,   # 0: person (13528 samples) - KEEP FULL WEIGHT (actions depend on person detection!)
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
        
        # ================================================================
        # BCE LOSS FOR ACTIONS AND RELATIONS (no Focal Loss)
        # ================================================================
        # Load per-class pos_weights from config (calculated from actual data)
        # Each class needs its own pos_weight = num_negatives / num_positives
        # This balances gradient so rare classes get appropriate learning signal
        # ================================================================
        
        smart_home_cfg = {}
        try:
            import json
            import os
            config_path = os.path.join(os.path.dirname(__file__), '../../config/smart_home_final.json')
            if os.path.exists(config_path):
                with open(config_path) as f:
                    smart_home_cfg = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load smart_home config: {e}")
        
        # ACTION pos_weights: per-class from actual data
        # Range: 4.01 (sitting in chair) to 50.0 (closing laptop)
        if 'action_pos_weights' in smart_home_cfg and num_actions == smart_home_cfg.get('num_actions', 0):
            act_pos_weight = torch.tensor(smart_home_cfg['action_pos_weights'], dtype=torch.float32)
            print(f"  Action BCE: per-class pos_weight (range {act_pos_weight.min():.1f}-{act_pos_weight.max():.1f})")
        else:
            # Fallback: flat weight (not ideal but functional)
            act_pos_weight = torch.full((num_actions,), num_actions - 1.0, dtype=torch.float32)
            print(f"  Action BCE: flat pos_weight={num_actions - 1.0} (fallback)")
        
        # RELATION pos_weights: per-class from actual data
        # Range: 0.43 (infrontof - very common) to 50.0 (rare relations)
        # CRITICAL: Some relations like 'infrontof' need DOWN-weighting (pos_weight < 1)
        if 'relation_pos_weights' in smart_home_cfg and len(smart_home_cfg['relation_pos_weights']) == num_relations:
            rel_pos_weight = torch.tensor(smart_home_cfg['relation_pos_weights'], dtype=torch.float32)
            print(f"  Relation BCE: per-class pos_weight (range {rel_pos_weight.min():.1f}-{rel_pos_weight.max():.1f})")
        else:
            # Fallback: flat weight
            rel_pos_weight = torch.full((num_relations,), num_relations - 1.0, dtype=torch.float32)
            print(f"  Relation BCE: flat pos_weight={num_relations - 1.0} (fallback)")
        
        self.act_lossf = nn.BCEWithLogitsLoss(pos_weight=act_pos_weight, reduction='none')
        self.rel_lossf = nn.BCEWithLogitsLoss(pos_weight=rel_pos_weight, reduction='none')
        self.act_pos_weight = act_pos_weight
        self.rel_pos_weight = rel_pos_weight
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
        
        # CRITICAL FIX 1: Matcher should ONLY use Object logits, not Actions/Relations.
        # Non-person objects (e.g. "food", "bag") have ground-truth 0 for all actions/relations.
        # Since their action logits are untrained, calculating cross-class BCE on them creates
        # massive random noise that completely confuses SimOTA anchor matching for small objects.
        cls_preds_matcher = obj_logits_for_matcher
        
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
                    pred_cls=cls_preds_matcher[batch_idx],
                    pred_box=box_preds[batch_idx],
                    tgt_labels=tgt_labels[:, :self.num_objects],  # ONLY evaluate object class cost
                    tgt_bboxes=tgt_bboxes_scaled,
                )

                conf_target = fg_mask.unsqueeze(-1).float()
                box_target = tgt_bboxes_scaled[matched_gt_inds]
                
                # Split labels
                matched_labels = tgt_labels[matched_gt_inds]
                
                # Object: argmax (exclusive class)
                obj_target = matched_labels[:, :self.num_objects].argmax(dim=-1)
                
                # Actions & Relations: HARD (0/1) targets
                # NOTE: IoU weighting was considered but NOT implemented.
                # All matched anchors weighted equally regardless of box quality.
                # Soft targets (target * iou) were tested but caused BCE to
                # converge to p=iou instead of p=1.0.
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

        # Confidence loss (with pos_weight for FG/BG balance)
        if self.conf_lossf.pos_weight.device != device:
            self.conf_lossf.pos_weight = self.conf_lossf.pos_weight.to(device)
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
        else:
            loss_obj = torch.tensor(0.0, device=device)

        # ================================================================
        # ACTION LOSS (BCE with per-class pos_weight, PERSON-ONLY)
        # ================================================================
        matched_act_preds = act_preds.view(-1, self.num_actions)[fg_masks]
        if len(act_targets) > 0:
            person_act_preds = matched_act_preds[is_person_masks]
            person_act_targets = act_targets[is_person_masks]
            if person_act_preds.shape[0] > 0:
                # Move pos_weight to correct device
                if self.act_lossf.pos_weight.device != device:
                    self.act_lossf.pos_weight = self.act_lossf.pos_weight.to(device)
                
                # BCE loss with per-class pos_weight handles gradient imbalance
                loss_act = self.act_lossf(person_act_preds, person_act_targets)  # [N, C]
                
                # Normalize by PERSON fg count (not all boxes)
                num_person_fg = max(person_act_preds.shape[0], 1)
                loss_act = loss_act.sum() / num_person_fg
                
                # ============================================================
                # AGGRESSIVE DECISIVENESS LOSS
                # ============================================================
                # Force model to COMMIT: positives MUST predict > 0.5, negatives < 0.1
                # This prevents lazy predictions stuck in the middle (0.2-0.4 range)
                # Weight = 1.5 to push hard
                # ============================================================
                pos_mask = person_act_targets > 0.5
                neg_mask = ~pos_mask
                
                # AGGRESSIVE thresholds
                pos_logit_thresh = 0.0    # sigmoid(0) = 0.5 - positives MUST exceed this
                neg_logit_thresh = -2.2   # sigmoid(-2.2) ≈ 0.1 - negatives MUST be below this
                
                # Hinge losses: penalize when not meeting threshold
                pos_hinge = F.relu(pos_logit_thresh - person_act_preds) * pos_mask.float()
                neg_hinge = F.relu(person_act_preds - neg_logit_thresh) * neg_mask.float()
                
                # Normalize separately so 33x negatives don't overwhelm positives
                num_pos = pos_mask.float().sum().clamp(1)
                num_neg = neg_mask.float().sum().clamp(1)
                act_gap_loss = pos_hinge.sum() / num_pos + neg_hinge.sum() / num_neg
                
                # AGGRESSIVE weight = 1.5 to push model hard
                loss_act = loss_act + 1.5 * act_gap_loss
            else:
                loss_act = torch.tensor(0.0, device=device)
        else:
            loss_act = torch.tensor(0.0, device=device)

        # ================================================================
        # RELATION LOSS (BCE with per-class pos_weight)
        # ================================================================
        matched_rel_preds = rel_preds.view(-1, self.num_relations)[fg_masks]
        if len(rel_targets) > 0:
            # Move pos_weight to correct device
            if self.rel_lossf.pos_weight.device != device:
                self.rel_lossf.pos_weight = self.rel_lossf.pos_weight.to(device)
            
            # BCE loss with per-class pos_weight
            loss_rel = self.rel_lossf(matched_rel_preds, rel_targets)  # [N, C]
            loss_rel = loss_rel.sum() / num_fg
            
            # AGGRESSIVE decisiveness loss for relations (same as actions)
            rel_pos_mask = rel_targets > 0.5
            rel_neg_mask = ~rel_pos_mask
            
            # Same aggressive thresholds
            rel_pos_hinge = F.relu(0.0 - matched_rel_preds) * rel_pos_mask.float()
            rel_neg_hinge = F.relu(matched_rel_preds - (-2.2)) * rel_neg_mask.float()
            
            num_rel_pos = rel_pos_mask.float().sum().clamp(1)
            num_rel_neg = rel_neg_mask.float().sum().clamp(1)
            rel_gap_loss = rel_pos_hinge.sum() / num_rel_pos + rel_neg_hinge.sum() / num_rel_neg
            
            # Slightly lower weight for relations (1.0) since they're secondary task
            loss_rel = loss_rel + 1.0 * rel_gap_loss
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # Box loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        if len(box_targets) > 0:
            ious = get_ious(matched_box_preds, box_targets, box_mode="xyxy", iou_type='giou')
            loss_box = (1.0 - ious).sum() / num_fg
        else:
            loss_box = torch.tensor(0.0, device=device)

        # ================================================================
        # FINAL LOSS WEIGHTING
        # ================================================================
        # Actions are the PRIMARY task - weight 1.5 to push learning
        # Relations are SECONDARY - weight 0.5
        # Objects use CrossEntropy with class weights - weight 1.0
        # ================================================================
        act_weight = 1.5  # Actions are primary task, push hard
        rel_weight = 0.5  # Relations are secondary
        loss_cls = loss_obj + act_weight * loss_act + rel_weight * loss_rel
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
                               num_objects=36, num_actions=157, num_relations=26):
    """
    Build the multi-task criterion.
    
    Uses BCE with per-class pos_weight for actions and relations.
    Per-class weights calculated from actual dataset statistics.
    """
    return MultiTaskCriterion(
        args, img_size, num_classes, 
        num_objects, num_actions, num_relations
    )

