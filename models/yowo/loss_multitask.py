"""
Multi-Task Loss for YOWO with Action Genome + Charades

This module implements separate loss functions for:
1. Object Head (36 classes) - CrossEntropy loss (exclusive classification)
2. Action Head (157 classes) - BCE loss (multi-label, Person-only with masking)
3. Relation Head (26 classes) - BCE loss (multi-label)
4. Interaction Head (1 class) - BCE loss (is object being interacted with?)

Key Features:
- Action loss is MASKED for non-Person boxes (actions only apply to Person)
- Object loss uses CrossEntropy for mutually exclusive classification
- Relation/Action losses use BCE for multi-label classification
- Interaction loss helps filter background objects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class MultiTaskCriterion(object):
    """
    Multi-Task Loss Criterion for Action Genome + Charades.
    
    Computes separate losses for:
    - Confidence (objectness)
    - Object identity (CrossEntropy - exclusive)
    - Actions (BCE - multi-label, Person-only)
    - Relations (BCE - multi-label)
    - Box regression (GIoU)
    
    Note: Interaction loss removed - negative relation classes (notlookingat, notcontacting)
    already indicate no interaction.
    """
    
    # Indices of "negative" relations that don't count as real interaction
    # notlookingat=1, unsure=2, notcontacting=17
    NEGATIVE_RELATION_INDICES = {1, 2, 17}
    
    def __init__(self, args, img_size, num_objects=36, num_actions=157, num_relations=26):
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_relations = num_relations
        self.num_classes = num_objects + num_actions + num_relations
        self.img_size = img_size
        
        # Loss weights
        self.loss_conf_weight = args.loss_conf_weight
        self.loss_obj_weight = getattr(args, 'loss_obj_weight', 1.0)
        self.loss_act_weight = getattr(args, 'loss_act_weight', 1.0)
        self.loss_rel_weight = getattr(args, 'loss_rel_weight', 1.0)
        # Note: loss_interact_weight removed - interaction head no longer exists
        self.loss_reg_weight = args.loss_reg_weight

        # Loss functions
        self.conf_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.obj_lossf = nn.CrossEntropyLoss(reduction='none')  # For exclusive object classification
        # Use Focal Loss for actions to handle class imbalance (rare actions get more focus)
        from .loss import SigmoidFocalLoss
        self.act_lossf = SigmoidFocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        self.rel_lossf = nn.BCEWithLogitsLoss(reduction='none')  # For multi-label relations
        
        # Label smoothing for multi-label heads (actions, relations)
        # Helps prevent overconfidence and improves generalization
        # 0.0 = no smoothing (default), 0.1 = 10% smoothing
        self.label_smoothing = getattr(args, 'label_smoothing', 0.0)
            
        # Matcher (uses combined class representation for matching)
        self.matcher = SimOTA(
            num_classes=self.num_classes,
            center_sampling_radius=args.center_sampling_radius,
            topk_candidate=args.topk_candicate
        )
    
    def apply_label_smoothing(self, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Apply label smoothing to multi-hot labels.
        
        For multi-hot labels:
        - Positive labels: 1.0 -> 1.0 - smoothing + smoothing/num_classes
        - Negative labels: 0.0 -> smoothing/num_classes
        
        This prevents the model from becoming overconfident.
        """
        if self.label_smoothing <= 0:
            return labels
        
        smoothed = labels * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        return smoothed

    def __call__(self, outputs, targets):
        """
        Compute multi-task losses.
        
        Args:
            outputs: dict with keys:
                - pred_conf: List[Tensor] [B, M, 1]
                - pred_obj: List[Tensor] [B, M, 36]
                - pred_act: List[Tensor] [B, M, 157]
                - pred_rel: List[Tensor] [B, M, 26]
                - pred_interact: List[Tensor] [B, M, 1]
                - pred_box: List[Tensor] [B, M, 4]
                - anchors: List[Tensor] [M, 2]
                - strides: List[int]
            targets: List[dict] with keys:
                - boxes: [N, 4] normalized boxes
                - labels: [N, 219] multi-hot labels (will be split)
                
        Returns:
            loss_dict: dict with individual losses and total
        """
        bs = outputs['pred_obj'][0].shape[0]
        device = outputs['pred_obj'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        
        # Concatenate predictions across FPN levels
        conf_preds = torch.cat(outputs['pred_conf'], dim=1)      # [B, M_total, 1]
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)        # [B, M_total, 36]
        act_preds = torch.cat(outputs['pred_act'], dim=1)        # [B, M_total, 157]
        rel_preds = torch.cat(outputs['pred_rel'], dim=1)        # [B, M_total, 26]
        # Note: interact_preds removed - interaction head no longer exists
        box_preds = torch.cat(outputs['pred_box'], dim=1)        # [B, M_total, 4]
        
        # For matcher, we need combined cls_preds
        cls_preds_combined = torch.cat([obj_preds, act_preds, rel_preds], dim=-1)
        
        # Storage for targets
        obj_targets = []
        act_targets = []
        rel_targets = []
        box_targets = []
        conf_targets = []
        fg_masks = []
        is_person_masks = []  # Track which matched GT is a Person (for action masking)

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)  # [N, 219]
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)   # [N, 4]

            # Denormalize boxes
            tgt_bboxes_scaled = tgt_bboxes * self.img_size

            # Check for valid targets
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                # No valid GT
                obj_target = conf_preds.new_zeros((0,), dtype=torch.long)
                act_target = conf_preds.new_zeros((0, self.num_actions))
                rel_target = conf_preds.new_zeros((0, self.num_relations))
                box_target = conf_preds.new_zeros((0, 4))
                conf_target = conf_preds.new_zeros((num_anchors, 1))
                fg_mask = conf_preds.new_zeros(num_anchors).bool()
                is_person_mask = conf_preds.new_zeros((0,)).bool()
            else:
                # Run SimOTA matcher
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.matcher(
                    fpn_strides=fpn_strides,
                    anchors=anchors,
                    pred_conf=conf_preds[batch_idx],
                    pred_cls=cls_preds_combined[batch_idx],
                    pred_box=box_preds[batch_idx],
                    tgt_labels=tgt_labels,
                    tgt_bboxes=tgt_bboxes_scaled,
                )

                conf_target = fg_mask.unsqueeze(-1).float()
                box_target = tgt_bboxes_scaled[matched_gt_inds]
                
                # Split the matched GT labels into obj/act/rel
                matched_labels = tgt_labels[matched_gt_inds]  # [num_fg, 219]
                
                # Object labels: indices 0-35 (convert from one-hot to class index)
                obj_labels_onehot = matched_labels[:, :self.num_objects]
                obj_target = obj_labels_onehot.argmax(dim=-1)  # [num_fg] class indices
                
                # Action labels: indices 36-192 (keep as multi-hot)
                act_target = matched_labels[:, self.num_objects:self.num_objects+self.num_actions]
                
                # Relation labels: indices 193-218 (keep as multi-hot)
                rel_target = matched_labels[:, self.num_objects+self.num_actions:]
                
                # Person mask: object class 0 is "person"
                is_person_mask = (obj_target == 0)
                
                # Note: interact_target removed - relation classes include negatives

            obj_targets.append(obj_target)
            act_targets.append(act_target)
            rel_targets.append(rel_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)
            is_person_masks.append(is_person_mask)

        # Concatenate across batch
        obj_targets = torch.cat(obj_targets, dim=0)          # [total_fg]
        act_targets = torch.cat(act_targets, dim=0)          # [total_fg, 157]
        rel_targets = torch.cat(rel_targets, dim=0)          # [total_fg, 26]
        box_targets = torch.cat(box_targets, dim=0)          # [total_fg, 4]
        conf_targets = torch.cat(conf_targets, dim=0)        # [total_anchors, 1]
        fg_masks = torch.cat(fg_masks, dim=0)                # [total_anchors]
        is_person_masks = torch.cat(is_person_masks, dim=0)  # [total_fg]
        
        num_foregrounds = fg_masks.sum()
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)

        # ============ CONFIDENCE LOSS ============
        loss_conf = self.conf_lossf(conf_preds.view(-1, 1), conf_targets)
        loss_conf = loss_conf.sum() / num_foregrounds

        # ============ OBJECT LOSS (CrossEntropy) ============
        matched_obj_preds = obj_preds.view(-1, self.num_objects)[fg_masks]  # [num_fg, 36]
        if len(obj_targets) > 0:
            loss_obj = self.obj_lossf(matched_obj_preds, obj_targets)
            loss_obj = loss_obj.sum() / num_foregrounds
        else:
            loss_obj = torch.tensor(0.0, device=device)

        # ============ ACTION LOSS (BCE, Person-only) ============
        matched_act_preds = act_preds.view(-1, self.num_actions)[fg_masks]  # [num_fg, 157]
        if len(act_targets) > 0 and is_person_masks.sum() > 0:
            # Only compute action loss for Person boxes
            person_act_preds = matched_act_preds[is_person_masks]
            person_act_targets = act_targets[is_person_masks]
            # Apply label smoothing if enabled
            person_act_targets = self.apply_label_smoothing(person_act_targets, self.num_actions)
            loss_act = self.act_lossf(person_act_preds, person_act_targets)
            loss_act = loss_act.sum() / is_person_masks.sum().clamp(1.0)
        else:
            loss_act = torch.tensor(0.0, device=device)

        # ============ RELATION LOSS (BCE) ============
        matched_rel_preds = rel_preds.view(-1, self.num_relations)[fg_masks]  # [num_fg, 26]
        if len(rel_targets) > 0:
            # Apply label smoothing if enabled
            smoothed_rel_targets = self.apply_label_smoothing(rel_targets, self.num_relations)
            loss_rel = self.rel_lossf(matched_rel_preds, smoothed_rel_targets)
            loss_rel = loss_rel.sum() / num_foregrounds
        else:
            loss_rel = torch.tensor(0.0, device=device)

        # Note: Interaction loss removed - relation classes include negatives (notlookingat, notcontacting)

        # ============ BOX LOSS (GIoU) ============
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        if len(box_targets) > 0:
            ious = get_ious(matched_box_preds, box_targets, box_mode="xyxy", iou_type='giou')
            loss_box = (1.0 - ious).sum() / num_foregrounds
        else:
            loss_box = torch.tensor(0.0, device=device)

        # ============ TOTAL LOSS ============
        losses = (
            self.loss_conf_weight * loss_conf +
            self.loss_obj_weight * loss_obj +
            self.loss_act_weight * loss_act +
            self.loss_rel_weight * loss_rel +
            self.loss_reg_weight * loss_box
        )

        loss_dict = dict(
            loss_conf=loss_conf,
            loss_obj=loss_obj,
            loss_act=loss_act,
            loss_rel=loss_rel,
            loss_box=loss_box,
            losses=losses
        )

        return loss_dict


def build_multitask_criterion(args, img_size, num_objects=36, num_actions=157, num_relations=26):
    """Build the multi-task criterion."""
    criterion = MultiTaskCriterion(
        args, img_size, num_objects, num_actions, num_relations
    )
    return criterion


