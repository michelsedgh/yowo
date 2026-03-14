import torch
import torch.nn.functional as F
from utils.box_ops import *



# SimOTA
class SimOTA(object):
    def __init__(self, num_classes, center_sampling_radius, topk_candidate):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides, 
                 anchors, 
                 pred_conf, 
                 pred_cls, 
                 pred_box, 
                 tgt_labels,
                 tgt_bboxes):
        # [M,]
        strides = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)
        num_anchor = anchors.shape[0]        
        num_gt = len(tgt_labels)

        # positive candidates
        fg_mask, is_in_boxes_and_center = \
            self.get_in_boxes_info(
                tgt_bboxes,
                anchors,
                strides,
                num_anchor,
                num_gt
                )

        conf_preds_ = pred_conf[fg_mask]   # [Mp, 1]
        cls_preds_ = pred_cls[fg_mask]   # [Mp, C]
        box_preds_ = pred_box[fg_mask]   # [Mp, 4]
        num_in_boxes_anchor = box_preds_.shape[0]

        # [N, Mp]
        pair_wise_ious, _ = box_iou(tgt_bboxes, box_preds_)
        pair_wise_ious = torch.nan_to_num(pair_wise_ious, nan=0.0, posinf=0.0, neginf=0.0).clamp_(0.0, 1.0)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if len(tgt_labels.shape) == 1:
            gt_cls = F.one_hot(tgt_labels.long(), self.num_classes)
        elif len(tgt_labels.shape) == 2:
            gt_cls = tgt_labels

        # OPTIMIZED: Use expand instead of repeat (no memory allocation)
        # [N, C] -> [N, Mp, C] via expand (memory-efficient, no copy)
        gt_cls = gt_cls.float().unsqueeze(1).expand(-1, num_in_boxes_anchor, -1)  # [N, Mp, C]

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Use expand for broadcasting (memory-efficient, no copy)
            cls_preds_exp = cls_preds_.float().unsqueeze(0).expand(num_gt, -1, -1).sigmoid()  # [N, Mp, C]
            conf_preds_exp = conf_preds_.float().unsqueeze(0).expand(num_gt, -1, -1).sigmoid()  # [N, Mp, 1]
            score_preds_ = torch.sqrt(cls_preds_exp * conf_preds_exp)  # [N, Mp, C]
            score_preds_ = torch.nan_to_num(score_preds_, nan=0.5, posinf=1.0, neginf=0.0)
            score_preds_ = score_preds_.clamp(min=1e-7, max=1.0 - 1e-7)
            pair_wise_cls_loss = F.binary_cross_entropy(
                score_preds_, gt_cls, reduction="none"
            ).sum(-1) # [N, Mp]
        del score_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        ) # [N, Mp]

        (
            num_fg,
            gt_matched_classes,         # [num_fg,]
            pred_ious_this_matching,    # [num_fg,]
            matched_gt_inds,            # [num_fg,]
        ) = self.dynamic_k_matching(
            cost,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask
            )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
        )


    def get_in_boxes_info(
        self,
        gt_bboxes,   # [N, 4]
        anchors,     # [M, 2]
        strides,     # [M,]
        num_anchors, # M
        num_gt,      # N
        ):
        # OPTIMIZED: Use broadcasting instead of .repeat() to avoid large allocations
        # anchor center: [M, 2] -> [1, M] for broadcasting with [N, 1]
        x_centers = anchors[:, 0].unsqueeze(0)  # [1, M]
        y_centers = anchors[:, 1].unsqueeze(0)  # [1, M]

        # GT box coords: [N, 4] -> [N, 1] for broadcasting
        gt_l = gt_bboxes[:, 0:1]  # [N, 1]
        gt_t = gt_bboxes[:, 1:2]  # [N, 1]
        gt_r = gt_bboxes[:, 2:3]  # [N, 1]
        gt_b = gt_bboxes[:, 3:4]  # [N, 1]

        # Broadcasting: [1, M] - [N, 1] -> [N, M]
        b_l = x_centers - gt_l
        b_r = gt_r - x_centers
        b_t = y_centers - gt_t
        b_b = gt_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)  # [N, M, 4]

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        
        # Center sampling
        center_radius = self.center_sampling_radius
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5  # [N, 2]
        
        # [1, M] for broadcasting
        center_radius_ = center_radius * strides.unsqueeze(0)

        # GT center coords: [N, 1] for broadcasting
        gt_cx = gt_centers[:, 0:1]  # [N, 1]
        gt_cy = gt_centers[:, 1:2]  # [N, 1]

        # Broadcasting: [1, M] op [N, 1] -> [N, M]
        c_l = x_centers - (gt_cx - center_radius_)
        c_r = (gt_cx + center_radius_) - x_centers
        c_t = y_centers - (gt_cy - center_radius_)
        c_b = (gt_cy + center_radius_) - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)  # [N, M, 4]
        
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    
    def dynamic_k_matching(
        self, 
        cost, 
        pair_wise_ious, 
        gt_classes, 
        num_gt, 
        fg_mask
        ):
        # Dynamic K - OPTIMIZED: no CPU sync, no Python loop
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # [num_gt]
        
        # OPTIMIZED: Fully vectorized assignment - no CPU sync, no Python loop
        # Use max_k for all GTs, then mask out excess matches
        max_k = min(int(dynamic_ks.max()), n_candidate_k)  # Stay on GPU until int()
        if max_k > 0 and cost.numel() > 0:
            # Get top max_k indices for ALL GTs at once
            _, topk_indices = torch.topk(cost, k=max_k, dim=1, largest=False)  # [num_gt, max_k]
            
            # Create mask for valid k values per GT: [num_gt, max_k]
            k_range = torch.arange(max_k, device=cost.device).unsqueeze(0)  # [1, max_k]
            valid_mask = k_range < dynamic_ks.unsqueeze(1)  # [num_gt, max_k]
            
            # Vectorized scatter using advanced indexing
            # Create row indices matching the valid positions
            gt_indices = torch.arange(num_gt, device=cost.device).unsqueeze(1).expand(-1, max_k)  # [num_gt, max_k]
            
            # Flatten and mask to get only valid assignments
            valid_gt_idx = gt_indices[valid_mask]  # [num_valid]
            valid_anchor_idx = topk_indices[valid_mask]  # [num_valid]
            
            # Single scatter operation for all assignments
            matching_matrix[valid_gt_idx, valid_anchor_idx] = 1

        del topk_ious, dynamic_ks

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        