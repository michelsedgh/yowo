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
        self.conf_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.obj_lossf = nn.CrossEntropyLoss(reduction='none')
        
        # Focal Loss for actions and relations (handles class imbalance)
        if use_focal_loss:
            print("  Using Focal Loss (gamma=1.0) for actions and relations")
            # gamma=1.0 for faster early learning (was 2.0, too aggressive for epoch 1-5)
            self.act_lossf = SigmoidFocalLoss(alpha=0.25, gamma=1.0, reduction='none')
            self.rel_lossf = SigmoidFocalLoss(alpha=0.25, gamma=1.0, reduction='none')
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
        iou_weights = []  # IoU quality weights (applied to LOSS, not targets)

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)
            tgt_bboxes_scaled = tgt_bboxes * self.img_size

            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                obj_target = conf_preds.new_zeros((0,), dtype=torch.long)
                act_target = conf_preds.new_zeros((0, self.num_actions))
                rel_target = conf_preds.new_zeros((0, self.num_relations))
                box_target = conf_preds.new_zeros((0, 4))
                conf_target = conf_preds.new_zeros((num_anchors, 1))
                fg_mask = conf_preds.new_zeros(num_anchors).bool()
                is_person_mask = conf_preds.new_zeros((0,)).bool()
                iou_weight = conf_preds.new_zeros((0, 1))
            else:
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
                
                # IoU quality weight: higher IoU = more trustworthy features = more loss weight
                iou_weight = pred_ious.clamp(min=0.0).unsqueeze(-1)  # [N, 1]

                
                # Person mask: object class 0 is "person" - for action loss masking
                is_person_mask = (obj_target == 0)

            obj_targets.append(obj_target)
            act_targets.append(act_target)
            rel_targets.append(rel_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)
            is_person_masks.append(is_person_mask)
            iou_weights.append(iou_weight)

        # Concatenate
        obj_targets = torch.cat(obj_targets, dim=0)
        act_targets = torch.cat(act_targets, dim=0)
        rel_targets = torch.cat(rel_targets, dim=0)
        box_targets = torch.cat(box_targets, dim=0)
        conf_targets = torch.cat(conf_targets, dim=0)
        fg_masks = torch.cat(fg_masks, dim=0)
        is_person_masks = torch.cat(is_person_masks, dim=0)  # [total_fg]
        iou_weights = torch.cat(iou_weights, dim=0)  # [total_fg, 1]
        
        num_fg = fg_masks.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_fg)
        num_fg = (num_fg / get_world_size()).clamp(1.0)

        # Confidence loss
        loss_conf = self.conf_lossf(conf_preds.view(-1, 1), conf_targets)
        loss_conf = loss_conf.sum() / num_fg

        # Object loss (CrossEntropy - exclusive class)
        matched_obj_preds = obj_preds.view(-1, self.num_objects)[fg_masks]
        if len(obj_targets) > 0:
            loss_obj = self.obj_lossf(matched_obj_preds, obj_targets)
            loss_obj = loss_obj.sum() / num_fg
        else:
            loss_obj = torch.tensor(0.0, device=device)

        # Action loss (Focal/BCE - multi-label, PERSON-ONLY)
        # CRITICAL FIX: Removed IoU weighting - it was killing 86% of gradients!
        # Hard targets (0/1) already encode correctness. IoU weighting prevents
        # the model from LEARNING to make better boxes by dampening gradients.
        matched_act_preds = act_preds.view(-1, self.num_actions)[fg_masks]
        if len(act_targets) > 0 and is_person_masks.sum() > 0:
            person_act_preds = matched_act_preds[is_person_masks]
            person_act_targets = act_targets[is_person_masks]
            loss_act = self.act_lossf(person_act_preds, person_act_targets)  # [N, C]
            loss_act = loss_act.sum() / num_fg  # No IoU weighting!
        else:
            loss_act = torch.tensor(0.0, device=device)

        # Relation loss (Focal/BCE - multi-label)
        # CRITICAL FIX: Removed IoU weighting for consistency with action loss
        # Relations are spatial relationships (e.g., "holding", "sitting on") that
        # should be learned even when box localization is imperfect early in training
        matched_rel_preds = rel_preds.view(-1, self.num_relations)[fg_masks]
        if len(rel_targets) > 0:
            loss_rel = self.rel_lossf(matched_rel_preds, rel_targets)  # [N, C]
            loss_rel = loss_rel.sum() / num_fg  # No IoU weighting!
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

