"""
YOWO Multi-Task Architecture for Action Genome + Charades

This module implements a four-head architecture that separates:
1. Object Head (36 classes) - Softmax/CE loss - "WHAT is it?"
2. Action Head (157 classes) - Sigmoid/BCE loss - "WHAT is it doing?" (Person-only)
3. Relation Head (26 classes) - Sigmoid/BCE loss - "HOW does it interact?"
4. Interaction Head (1 class) - Sigmoid/BCE loss - "Is this object being interacted with?"

The Interaction Head helps filter out background objects that are not being
interacted with by the person. This is crucial for understanding "what is
happening" in a scene.

This is semantically superior to the "jammed" 219-class approach because:
- Objects are mutually exclusive (Softmax enforces this)
- Actions only apply to Person boxes (masking prevents invalid learning)
- Each head can specialize on its feature requirements
- Interaction head filters non-interacted objects

Reference: Action Genome dataset (https://arxiv.org/abs/1912.06992)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone_2d
from ..backbone import build_backbone_3d
from .encoder import build_channel_encoder
from .head import build_head

from utils.nms import multiclass_nms


class YOWOMultiTask(nn.Module):
    """
    YOWO with Multi-Task Heads for Action Genome + Charades.
    
    Instead of a single cls_preds head with 219 classes, this model has:
    - obj_preds: Object identity (36 classes, Softmax)
    - act_preds: Actions (157 classes, Sigmoid, Person-only)
    - rel_preds: Relations (26 classes, Sigmoid)
    - interact_preds: Is object being interacted with? (1 class, Sigmoid)
    """
    
    def __init__(self, 
                 cfg,
                 device,
                 num_objects=36,
                 num_actions=157,
                 num_relations=26,
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 topk=50,
                 trainable=False):
        super(YOWOMultiTask, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.num_relations = num_relations
        self.num_classes = num_objects + num_actions + num_relations  # For compatibility
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk

        # ------------------ Network ---------------------
        ## 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            cfg, pretrained=cfg['pretrained_2d'] and trainable)
            
        ## 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            cfg, pretrained=cfg['pretrained_3d'] and trainable)

        ## cls channel encoder (shared for all classification heads)
        self.cls_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])
            
        ## reg channel encoder
        self.reg_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])

        ## head (shared feature processing before prediction)
        self.heads = nn.ModuleList(
            [build_head(cfg) for _ in range(len(cfg['stride']))]
        ) 

        ## Prediction layers
        head_dim = cfg['head_dim']
        
        # Confidence prediction (unchanged)
        self.conf_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(cfg['stride']))
            ]) 
        
        # Object prediction (36 classes - Softmax/CE)
        self.obj_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, self.num_objects, kernel_size=1)
                for _ in range(len(cfg['stride']))
            ]) 
        
        # Action prediction (157 classes - Sigmoid/BCE, Person-only)
        self.act_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, self.num_actions, kernel_size=1)
                for _ in range(len(cfg['stride']))
            ]) 
        
        # Relation prediction (26 classes - Sigmoid/BCE)
        self.rel_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, self.num_relations, kernel_size=1)
                for _ in range(len(cfg['stride']))
            ]) 
        
        # Interaction prediction (1 class - Sigmoid/BCE)
        # Predicts whether this object is being interacted with by a person
        self.interact_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(cfg['stride']))
            ]) 
        
        # Box regression (unchanged)
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                for _ in range(len(cfg['stride']))
            ])                 

        # init weights
        self.init_yowo()


    def init_yowo(self): 
        # Init batch norm
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        # Init bias for conf/cls predictions
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        
        # Confidence prediction bias
        for conf_pred in self.conf_preds:
            b = conf_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            conf_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        # Object prediction bias (will use Softmax, but init similarly)
        for obj_pred in self.obj_preds:
            b = obj_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        # Action prediction bias
        for act_pred in self.act_preds:
            b = act_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            act_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        # Relation prediction bias
        for rel_pred in self.rel_preds:
            b = rel_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            rel_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        # Interaction prediction bias
        for interact_pred in self.interact_preds:
            b = interact_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            interact_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def generate_anchors(self, fmp_size, stride):
        """Generate anchor points for a feature map."""
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= stride
        anchors = anchor_xy.to(self.device)
        return anchors
        

    def decode_boxes(self, anchors, pred_reg, stride):
        """Decode box predictions from anchor offsets."""
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        pred_box_wh = pred_reg[..., 2:].exp() * stride
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        return pred_box


    def post_process_multi_hot(self, conf_preds, obj_preds, act_preds, rel_preds, interact_preds, reg_preds, anchors):
        """
        Post-process predictions for inference (multi-hot output).
        
        Returns boxes with format: [x1, y1, x2, y2, conf, interact, obj_classes..., act_classes..., rel_classes...]
        """
        all_conf_preds = []
        all_obj_preds = []
        all_act_preds = []
        all_rel_preds = []
        all_interact_preds = []
        all_box_preds = []
        
        for level, (conf_pred_i, obj_pred_i, act_pred_i, rel_pred_i, interact_pred_i, reg_pred_i, anchors_i) in enumerate(
            zip(conf_preds, obj_preds, act_preds, rel_preds, interact_preds, reg_preds, anchors)):
            
            # Decode boxes
            box_pred_i = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])
            
            # Confidence (sigmoid)
            conf_pred_i = torch.sigmoid(conf_pred_i.squeeze(-1))
            
            # Object (softmax for exclusive classification)
            obj_pred_i = torch.softmax(obj_pred_i, dim=-1)
            
            # Actions, Relations, Interaction (sigmoid for multi-label / binary)
            act_pred_i = torch.sigmoid(act_pred_i)
            rel_pred_i = torch.sigmoid(rel_pred_i)
            interact_pred_i = torch.sigmoid(interact_pred_i.squeeze(-1))
            
            # Top-k filtering
            topk_conf_pred_i, topk_inds = torch.topk(conf_pred_i, min(self.topk, conf_pred_i.shape[0]))
            topk_obj_pred_i = obj_pred_i[topk_inds]
            topk_act_pred_i = act_pred_i[topk_inds]
            topk_rel_pred_i = rel_pred_i[topk_inds]
            topk_interact_pred_i = interact_pred_i[topk_inds]
            topk_box_pred_i = box_pred_i[topk_inds]
            
            # Threshold filtering
            keep = topk_conf_pred_i.gt(self.conf_thresh)
            topk_conf_pred_i = topk_conf_pred_i[keep]
            topk_obj_pred_i = topk_obj_pred_i[keep]
            topk_act_pred_i = topk_act_pred_i[keep]
            topk_rel_pred_i = topk_rel_pred_i[keep]
            topk_interact_pred_i = topk_interact_pred_i[keep]
            topk_box_pred_i = topk_box_pred_i[keep]
            
            all_conf_preds.append(topk_conf_pred_i)
            all_obj_preds.append(topk_obj_pred_i)
            all_act_preds.append(topk_act_pred_i)
            all_rel_preds.append(topk_rel_pred_i)
            all_interact_preds.append(topk_interact_pred_i)
            all_box_preds.append(topk_box_pred_i)
        
        # Concatenate across levels
        conf_preds = torch.cat(all_conf_preds, dim=0)
        obj_preds = torch.cat(all_obj_preds, dim=0)
        act_preds = torch.cat(all_act_preds, dim=0)
        rel_preds = torch.cat(all_rel_preds, dim=0)
        interact_preds = torch.cat(all_interact_preds, dim=0)
        box_preds = torch.cat(all_box_preds, dim=0)
        
        # Combine all class predictions for compatibility
        # Format: [obj_36, act_157, rel_26] = 219 dims
        cls_preds = torch.cat([obj_preds, act_preds, rel_preds], dim=-1)
        
        # To CPU/numpy
        scores = conf_preds.cpu().numpy()
        interact_scores = interact_preds.cpu().numpy()
        labels = cls_preds.cpu().numpy()
        bboxes = box_preds.cpu().numpy()
        
        # NMS using object class (first 36 dims)
        obj_labels = obj_preds.argmax(dim=-1).cpu().numpy()
        scores_for_nms, labels_for_nms, bboxes = multiclass_nms(
            scores, obj_labels, bboxes, self.nms_thresh, self.num_objects, False)
        
        # Reconstruct output with full labels
        if len(bboxes) > 0:
            surviving_mask = np.isin(
                np.arange(len(box_preds.cpu().numpy())), 
                np.where(np.isin(box_preds.cpu().numpy().sum(axis=1), bboxes.sum(axis=1)))[0]
            )
            if surviving_mask.sum() > 0:
                labels = labels[surviving_mask[:len(labels)]][:len(bboxes)]
                interact_scores = interact_scores[surviving_mask[:len(interact_scores)]][:len(bboxes)]
                scores_for_nms = scores_for_nms[:len(bboxes)]
        
        # Output: [x1, y1, x2, y2, conf, interact, classes...]
        # interact score is at index 5, classes start at index 6
        if len(bboxes) > 0:
            out_boxes = np.concatenate([
                bboxes,                          # [0:4] bbox
                scores_for_nms[..., None],       # [4] confidence
                interact_scores[..., None],      # [5] interaction score
                labels                           # [6:] classes
            ], axis=-1)
        else:
            out_boxes = np.zeros((0, 6 + self.num_classes))
        
        return out_boxes


    @torch.no_grad()
    def inference(self, video_clips):
        """Inference mode: returns post-processed detections."""
        B, _, _, img_h, img_w = video_clips.shape
        
        # Key frame
        key_frame = video_clips[:, :, -1, :, :]
        
        # 3D backbone
        feat_3d = self.backbone_3d(video_clips)
        
        # 2D backbone
        cls_feats, reg_feats = self.backbone_2d(key_frame)
        
        # Process each level
        all_conf_preds = []
        all_obj_preds = []
        all_act_preds = []
        all_rel_preds = []
        all_interact_preds = []
        all_reg_preds = []
        all_anchors = []
        
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            # Upsample 3D features
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            # Encode
            cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up)
            reg_feat = self.reg_channel_encoders[level](reg_feat, feat_3d_up)
            
            # Head
            cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)
            
            # Predictions
            conf_pred = self.conf_preds[level](reg_feat)
            obj_pred = self.obj_preds[level](cls_feat)
            act_pred = self.act_preds[level](cls_feat)
            rel_pred = self.rel_preds[level](cls_feat)
            interact_pred = self.interact_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)
            
            # Generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            # Reshape: [B, C, H, W] -> [B, H*W, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_objects)
            act_pred = act_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_actions)
            rel_pred = rel_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_relations)
            interact_pred = interact_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            
            all_conf_preds.append(conf_pred)
            all_obj_preds.append(obj_pred)
            all_act_preds.append(act_pred)
            all_rel_preds.append(rel_pred)
            all_interact_preds.append(interact_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)
        
        # Batch processing
        batch_bboxes = []
        for batch_idx in range(B):
            cur_conf_preds = [p[batch_idx] for p in all_conf_preds]
            cur_obj_preds = [p[batch_idx] for p in all_obj_preds]
            cur_act_preds = [p[batch_idx] for p in all_act_preds]
            cur_rel_preds = [p[batch_idx] for p in all_rel_preds]
            cur_interact_preds = [p[batch_idx] for p in all_interact_preds]
            cur_reg_preds = [p[batch_idx] for p in all_reg_preds]
            
            out_boxes = self.post_process_multi_hot(
                cur_conf_preds, cur_obj_preds, cur_act_preds, cur_rel_preds, 
                cur_interact_preds, cur_reg_preds, all_anchors)
            
            # Normalize boxes
            out_boxes[..., :4] /= max(img_h, img_w)
            out_boxes[..., :4] = out_boxes[..., :4].clip(0., 1.)
            
            batch_bboxes.append(out_boxes)
        
        return batch_bboxes


    def forward(self, video_clips):
        """
        Forward pass.
        
        Args:
            video_clips: [B, 3, T, H, W] video tensor
            
        Returns:
            Training mode: dict with predictions for loss computation
            Inference mode: list of detection boxes
        """
        if not self.trainable:
            return self.inference(video_clips)
        
        # Training mode
        key_frame = video_clips[:, :, -1, :, :]
        
        # 3D backbone
        feat_3d = self.backbone_3d(video_clips)
        
        # 2D backbone
        cls_feats, reg_feats = self.backbone_2d(key_frame)
        
        # Process each level
        all_conf_preds = []
        all_obj_preds = []
        all_act_preds = []
        all_rel_preds = []
        all_interact_preds = []
        all_box_preds = []
        all_anchors = []
        
        for level, (cls_feat, reg_feat) in enumerate(zip(cls_feats, reg_feats)):
            # Upsample 3D features
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            # Encode
            cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up)
            reg_feat = self.reg_channel_encoders[level](reg_feat, feat_3d_up)
            
            # Head
            cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)
            
            # Predictions
            conf_pred = self.conf_preds[level](reg_feat)
            obj_pred = self.obj_preds[level](cls_feat)
            act_pred = self.act_preds[level](cls_feat)
            rel_pred = self.rel_preds[level](cls_feat)
            interact_pred = self.interact_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)
            
            # Generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            # Reshape: [B, C, H, W] -> [B, H*W, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            act_pred = act_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            rel_pred = rel_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            interact_pred = interact_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            
            # Decode boxes
            box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])
            
            all_conf_preds.append(conf_pred)
            all_obj_preds.append(obj_pred)
            all_act_preds.append(act_pred)
            all_rel_preds.append(rel_pred)
            all_interact_preds.append(interact_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)
        
        # Output dict for loss computation
        outputs = {
            "pred_conf": all_conf_preds,        # List[Tensor] [B, M, 1]
            "pred_obj": all_obj_preds,          # List[Tensor] [B, M, 36]
            "pred_act": all_act_preds,          # List[Tensor] [B, M, 157]
            "pred_rel": all_rel_preds,          # List[Tensor] [B, M, 26]
            "pred_interact": all_interact_preds, # List[Tensor] [B, M, 1]
            "pred_box": all_box_preds,          # List[Tensor] [B, M, 4]
            "anchors": all_anchors,             # List[Tensor] [M, 2]
            "strides": self.stride              # List[int]
        }
        
        return outputs


