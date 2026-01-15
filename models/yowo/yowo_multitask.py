"""
YOWO Multi-Task for Action Genome + Charades

DEFINITIVE CASCADE: Object → Relation → Action

Why this order:
1. OBJECTS: "What is this?" - No dependencies
2. RELATIONS: "What is the relation between person and object?" - Needs objects
3. ACTIONS: "What is person doing?" - Needs objects AND relations

Example: "Person typing on laptop"
  - Object detection: person, laptop
  - Relation: person is "holding" laptop, "looking_at" laptop  
  - Action: "typing" (confident because holding + looking_at laptop)
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


class ObjectContext(nn.Module):
    """
    Spatial Object Context for relation prediction using cross-attention.
    
    Each position attends to object predictions at ALL positions,
    weighted by spatial relevance. This provides TRUE spatial context
    where each position knows what objects are nearby vs far away.
    
    Key improvements over original:
    - Spatial attention instead of global pooling
    - Residual connection preserves original features
    - GroupNorm for stability
    - ~50% fewer parameters
    """
    def __init__(self, feat_dim, num_objects=36, attn_dim=64):
        super().__init__()
        self.num_objects = num_objects
        self.attn_dim = attn_dim
        
        # Project features to low-dim for efficient attention
        self.query_proj = nn.Conv2d(feat_dim, attn_dim, kernel_size=1)
        
        # Project object predictions to keys
        self.key_proj = nn.Conv2d(num_objects, attn_dim, kernel_size=1)
        
        # Project object predictions to values  
        self.value_proj = nn.Conv2d(num_objects, feat_dim // 2, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(feat_dim // 2, feat_dim, kernel_size=1)
        
        # Normalization for stability
        self.norm = nn.GroupNorm(num_groups=32, num_channels=feat_dim)
        
        # Learnable attention temperature
        self.temperature = nn.Parameter(torch.ones(1) * (attn_dim ** -0.5))
    
    def forward(self, feat, obj_logits):
        """
        Args:
            feat: [B, C, H, W] - backbone features
            obj_logits: [B, 36, H, W] - object predictions (logits)
        Returns:
            [B, C, H, W] - features enriched with spatial object context
        """
        B, C, H, W = feat.shape
        N = H * W
        
        # Get object probabilities (DETACHED - don't affect object training)
        obj_probs = F.softmax(obj_logits, dim=1).detach()
        
        # Query: What context does each position need? (from features)
        Q = self.query_proj(feat)  # [B, attn_dim, H, W]
        Q = Q.view(B, self.attn_dim, N).permute(0, 2, 1)  # [B, N, attn_dim]
        
        # Key: What objects are at each position? (from predictions)
        K = self.key_proj(obj_probs)  # [B, attn_dim, H, W]
        K = K.view(B, self.attn_dim, N)  # [B, attn_dim, N]
        
        # Value: Object context to retrieve (from predictions)
        V = self.value_proj(obj_probs)  # [B, C//2, H, W]
        V = V.view(B, C // 2, N).permute(0, 2, 1)  # [B, N, C//2]
        
        # Attention: Each position attends to all positions
        attn = torch.bmm(Q, K) * self.temperature  # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum of values
        context = torch.bmm(attn, V)  # [B, N, C//2]
        context = context.permute(0, 2, 1).view(B, C // 2, H, W)
        
        # Project and add RESIDUAL
        out = self.out_proj(context)
        return self.norm(feat + out)


class RelationContext(nn.Module):
    """
    Spatial Relation Context for action prediction using cross-attention.
    
    Each position attends to combined object + relation predictions,
    weighted by spatial relevance and gated by person presence.
    
    Key improvements over original:
    - Spatial attention instead of global pooling
    - Residual connection preserves original features
    - GroupNorm for stability
    - Person gating preserved
    """
    def __init__(self, feat_dim, num_objects=36, num_relations=26, attn_dim=64):
        super().__init__()
        self.num_objects = num_objects
        self.num_relations = num_relations
        self.attn_dim = attn_dim
        
        # Query from features
        self.query_proj = nn.Conv2d(feat_dim, attn_dim, kernel_size=1)
        
        # Key from combined object + relation predictions
        self.key_proj = nn.Conv2d(num_objects + num_relations, attn_dim, kernel_size=1)
        
        # Value from combined predictions
        self.value_proj = nn.Conv2d(num_objects + num_relations, feat_dim // 2, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(feat_dim // 2, feat_dim, kernel_size=1)
        
        # Normalization for stability
        self.norm = nn.GroupNorm(num_groups=32, num_channels=feat_dim)
        
        # Learnable attention temperature
        self.temperature = nn.Parameter(torch.ones(1) * (attn_dim ** -0.5))
    
    def forward(self, feat, obj_logits, rel_logits):
        """
        Args:
            feat: [B, C, H, W] - backbone features
            obj_logits: [B, 36, H, W] - object predictions
            rel_logits: [B, 26, H, W] - relation predictions
        Returns:
            [B, C, H, W] - features enriched with spatial relation context
        """
        B, C, H, W = feat.shape
        N = H * W
        
        # Get probabilities (DETACHED)
        obj_probs = F.softmax(obj_logits, dim=1).detach()
        rel_probs = torch.sigmoid(rel_logits).detach()
        
        # Combine object and relation predictions
        combined = torch.cat([obj_probs, rel_probs], dim=1)  # [B, 62, H, W]
        
        # Person probability for gating
        person_prob = obj_probs[:, 0:1, :, :]  # [B, 1, H, W]
        
        # Query from features
        Q = self.query_proj(feat).view(B, self.attn_dim, N).permute(0, 2, 1)
        
        # Key and Value from combined predictions
        K = self.key_proj(combined).view(B, self.attn_dim, N)
        V = self.value_proj(combined).view(B, C // 2, N).permute(0, 2, 1)
        
        # Attention
        attn = torch.bmm(Q, K) * self.temperature  # [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # Weighted context
        context = torch.bmm(attn, V).permute(0, 2, 1).view(B, C // 2, H, W)
        
        # Soft gate by person presence
        # Ensures at least 30% context flows (for pure motion actions)
        # While person regions get full context bonus
        soft_gate = 0.3 + 0.7 * person_prob  # Range: [0.3, 1.0]
        context = context * soft_gate
        
        # Project and add RESIDUAL
        out = self.out_proj(context)
        return self.norm(feat + out)


class YOWOMultiTask(nn.Module):
    """
    YOWO Multi-Task with full cascade: Object → Relation → Action
    
    Architecture:
    1. Backbone (same as YOWO): 2D + 3D feature extraction
    2. Object Head: predict 36 object classes
    3. ObjectContext: enrich features with object positions
    4. Relation Head: predict 26 relation classes  
    5. RelationContext: enrich features with relation info
    6. Action Head: predict 157 action classes
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
        self.num_classes = num_objects + num_actions + num_relations
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk

        # ==================== BACKBONE ====================
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            cfg, pretrained=cfg['pretrained_2d'] and trainable)
            
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            cfg, pretrained=cfg['pretrained_3d'] and trainable)

        # ==================== ENCODER ====================
        self.cls_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])
            
        self.reg_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])

        # ==================== HEAD ====================
        self.heads = nn.ModuleList(
            [build_head(cfg) for _ in range(len(cfg['stride']))]
        ) 

        head_dim = cfg['head_dim']
        
        # ==================== CASCADE PREDICTIONS ====================
        
        # Confidence (same as original YOWO)
        self.conf_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # 1. Object Head (first in cascade)
        self.obj_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, num_objects, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # Object → Relation context
        self.obj_context = nn.ModuleList([
            ObjectContext(head_dim, num_objects)
            for _ in range(len(cfg['stride']))
        ])
        
        # 2. Relation Head (second in cascade)
        self.rel_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, num_relations, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # Relation → Action context
        self.rel_context = nn.ModuleList([
            RelationContext(head_dim, num_objects, num_relations)
            for _ in range(len(cfg['stride']))
        ])
        
        # 3. Action Head (last in cascade)
        self.act_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, num_actions, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # Box regression
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                for _ in range(len(cfg['stride']))])

        self.init_yowo()


    def init_yowo(self): 
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        
        for pred_list in [self.conf_preds, self.obj_preds, self.rel_preds, self.act_preds]:
            for pred in pred_list:
                b = pred.bias.view(1, -1)
                b.data.fill_(bias_value.item())
                pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def generate_anchors(self, fmp_size, stride):
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)], indexing='ij')
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= stride
        return anchor_xy.to(self.device)
        

    def decode_boxes(self, anchors, pred_reg, stride):
        pred_ctr_xy = anchors + pred_reg[..., :2] * stride
        pred_box_wh = pred_reg[..., 2:].exp() * stride
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        return torch.cat([pred_x1y1, pred_x2y2], dim=-1)


    def _forward_single_level(self, level, cls_feat, reg_feat, feat_3d_up):
        """Process single FPN level with full cascade."""
        # Encode
        cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up)
        reg_feat = self.reg_channel_encoders[level](reg_feat, feat_3d_up)
        
        # Head
        cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)
        
        # ===== CASCADE: Object → Relation → Action =====
        
        # 1. Object prediction
        obj_logits = self.obj_preds[level](cls_feat)
        
        # 2. Relation prediction (with object context)
        rel_feat = self.obj_context[level](cls_feat, obj_logits)
        rel_logits = self.rel_preds[level](rel_feat)
        
        # 3. Action prediction (with object + relation context)
        act_feat = self.rel_context[level](rel_feat, obj_logits, rel_logits)
        act_logits = self.act_preds[level](act_feat)
        
        # Confidence and box
        conf_logits = self.conf_preds[level](reg_feat)
        reg_pred = self.reg_preds[level](reg_feat)
        
        return conf_logits, obj_logits, rel_logits, act_logits, reg_pred


    def post_process(self, conf_preds, cls_preds, reg_preds, anchors):
        all_conf = []
        all_cls = []
        all_box = []
        
        for level, (conf_i, cls_i, reg_i, anc_i) in enumerate(
            zip(conf_preds, cls_preds, reg_preds, anchors)):
            
            box_i = self.decode_boxes(anc_i, reg_i, self.stride[level])
            conf_i = torch.sigmoid(conf_i.squeeze(-1))
            
            k = min(self.topk, conf_i.shape[0])
            topk_conf, topk_idx = torch.topk(conf_i, k)
            topk_cls = cls_i[topk_idx]
            topk_box = box_i[topk_idx]
            
            keep = topk_conf > self.conf_thresh
            all_conf.append(topk_conf[keep])
            all_cls.append(topk_cls[keep])
            all_box.append(topk_box[keep])
        
        conf = torch.cat(all_conf, dim=0)
        cls = torch.cat(all_cls, dim=0)
        box = torch.cat(all_box, dim=0)
        
        if len(conf) == 0:
            return np.zeros((0, 5 + self.num_classes))
        
        scores = conf.cpu().numpy()
        labels = cls.cpu().numpy()
        bboxes = box.cpu().numpy()
        
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)
        
        return np.concatenate([bboxes, scores[..., None], labels], axis=-1)


    @torch.no_grad()
    def inference(self, video_clips):
        B, _, _, img_h, img_w = video_clips.shape
        
        key_frame = video_clips[:, :, -1, :, :]
        feat_3d = self.backbone_3d(video_clips)
        cls_feats, reg_feats = self.backbone_2d(key_frame)
        
        all_conf = []
        all_cls = []
        all_reg = []
        all_anchors = []
        
        for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            conf_l, obj_l, rel_l, act_l, reg_l = \
                self._forward_single_level(level, cls_f, reg_f, feat_3d_up)
            
            fmp_size = conf_l.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            # Reshape
            conf_l = conf_l.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            obj_l = obj_l.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_objects)
            rel_l = rel_l.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_relations)
            act_l = act_l.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_actions)
            reg_l = reg_l.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            
            # Activations
            obj_l = F.softmax(obj_l, dim=-1)
            rel_l = torch.sigmoid(rel_l)
            act_l = torch.sigmoid(act_l)
            
            # Combined class output: [obj, act, rel] to match label format
            cls_l = torch.cat([obj_l, act_l, rel_l], dim=-1)
            
            all_conf.append(conf_l)
            all_cls.append(cls_l)
            all_reg.append(reg_l)
            all_anchors.append(anchors)
        
        batch_bboxes = []
        for b in range(B):
            out = self.post_process(
                [c[b] for c in all_conf],
                [c[b] for c in all_cls],
                [r[b] for r in all_reg],
                all_anchors
            )
            out[..., :4] /= max(img_h, img_w)
            out[..., :4] = np.clip(out[..., :4], 0., 1.)
            batch_bboxes.append(out)
        
        return batch_bboxes


    def forward(self, video_clips):
        if not self.trainable:
            return self.inference(video_clips)
        
        key_frame = video_clips[:, :, -1, :, :]
        feat_3d = self.backbone_3d(video_clips)
        cls_feats, reg_feats = self.backbone_2d(key_frame)
        
        all_conf = []
        all_obj = []
        all_rel = []
        all_act = []
        all_box = []
        all_anchors = []
        
        for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            conf_l, obj_l, rel_l, act_l, reg_l = \
                self._forward_single_level(level, cls_f, reg_f, feat_3d_up)
            
            fmp_size = conf_l.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            # Reshape
            conf_l = conf_l.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            obj_l = obj_l.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            rel_l = rel_l.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            act_l = act_l.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_l = reg_l.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            
            box_l = self.decode_boxes(anchors, reg_l, self.stride[level])
            
            all_conf.append(conf_l)
            all_obj.append(obj_l)
            all_rel.append(rel_l)
            all_act.append(act_l)
            all_box.append(box_l)
            all_anchors.append(anchors)
        
        return {
            "pred_conf": all_conf,
            "pred_obj": all_obj,
            "pred_act": all_act,
            "pred_rel": all_rel,
            "pred_box": all_box,
            "anchors": all_anchors,
            "strides": self.stride
        }
