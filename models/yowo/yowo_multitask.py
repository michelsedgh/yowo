"""
YOWO Multi-Task V2 for Action Genome + Charades

CRITICAL FIX: Use objectness confidence to gate attention!

The previous version had a fatal flaw:
- Object prediction is softmax over 36 classes (no background)
- obj_probs.max() is always high (~0.7) even at background positions
- Attention spread uniformly → no spatial grounding

This version fixes it by:
1. Passing confidence logits to context modules  
2. Using conf * obj_probs as "grounded object presence"
3. Only attending to positions where objects are ACTUALLY detected

DEFINITIVE CASCADE: Object → Relation → Action
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone_2d
from ..backbone import build_backbone_3d
from .encoder import build_channel_encoder
from .head import build_head

from utils.nms import multiclass_nms


class ObjectContextModuleV2(nn.Module):
    """
    FIXED: Object context with confidence-gated attention.
    
    Now receives conf_logits to know WHERE objects actually exist.
    Attention only focuses on positions with high objectness confidence.
    """
    def __init__(self, dim=256, num_classes=36, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query: from classification features
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Key: from CONFIDENCE-WEIGHTED object predictions
        # Input: obj_probs * conf (grounded predictions)
        self.k_proj = nn.Sequential(
            nn.Conv2d(num_classes, dim, kernel_size=1),
            nn.GELU(),
        )
        
        # Value: object information to aggregate
        self.v_proj = nn.Sequential(
            nn.Conv2d(num_classes, dim, kernel_size=1),
            nn.GELU(),
        )
        
        # Relative position bias
        max_size = 32
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2*max_size-1, 2*max_size-1))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(32, dim)
        
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
    
    def get_rel_pos_bias(self, H, W, device):
        """Generate relative position bias for H x W grid."""
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.reshape(2, -1)
        
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        
        rel_coords[:, :, 0] += self.rel_pos_bias.shape[1] // 2
        rel_coords[:, :, 1] += self.rel_pos_bias.shape[2] // 2
        rel_coords = rel_coords.clamp(0, self.rel_pos_bias.shape[1] - 1)
        
        rel_pos_idx = rel_coords[:, :, 0] * self.rel_pos_bias.shape[2] + rel_coords[:, :, 1]
        rel_pos_idx = rel_pos_idx.long()
        
        bias = self.rel_pos_bias.view(self.num_heads, -1)[:, rel_pos_idx.view(-1)]
        bias = bias.view(self.num_heads, H*W, H*W)
        
        return bias
    
    def forward(self, cls_feat, obj_logits, conf_logits, return_weights=False):
        """
        FIXED: Now uses confidence to weight object predictions.
        
        Args:
            cls_feat: [B, C, H, W] - features 
            obj_logits: [B, 36, H, W] - object class predictions (logits)
            conf_logits: [B, 1, H, W] - objectness confidence (logits)
        Returns:
            context_feat: [B, C, H, W] - features enriched with spatial object context
        """
        B, C, H, W = cls_feat.shape
        N = H * W
        
        # Convert to probabilities
        obj_probs = F.softmax(obj_logits, dim=1)  # [B, 36, H, W]
        conf = torch.sigmoid(conf_logits)  # [B, 1, H, W] - true objectness!
        
        # CRITICAL FIX: Weight object probs by confidence
        # Background positions (low conf) → all object probs suppressed
        # Object positions (high conf) → object probs preserved
        grounded_obj = obj_probs * conf  # [B, 36, H, W]
        
        # Project Q, K, V
        Q = self.q_proj(cls_feat)
        K = self.k_proj(grounded_obj)  # Now uses confidence-weighted predictions!
        V = self.v_proj(grounded_obj)
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Relative position bias
        rel_pos_bias = self.get_rel_pos_bias(H, W, cls_feat.device)
        attn = attn + rel_pos_bias.unsqueeze(0)
        
        # CRITICAL FIX: Use CONFIDENCE as presence bias (not obj_probs.max())
        # conf already tells us where objects are detected
        conf_flat = conf.view(B, 1, 1, N)  # [B, 1, 1, N]
        presence_bias = (conf_flat - 0.3) * 8.0  # Strong bias: conf>0.3 → attend, else suppress
        attn = attn + presence_bias
        
        # Softmax
        attn_weights = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Output projection + residual
        out = self.out_proj(out)
        out = self.norm(cls_feat + out)
        
        if return_weights:
            return out, attn_weights.mean(dim=1)
        return out


class SceneContextAttentionV2(nn.Module):
    """
    FIXED: Scene context with confidence-gated attention.
    
    For action prediction, attends to positions where objects AND relations
    are actually detected (not just raw softmax probabilities).
    """
    def __init__(self, dim=256, num_objects=36, num_relations=26, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_objects = num_objects
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query: from classification features
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Key/Value: from confidence-weighted object + relation predictions
        self.k_proj = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim, kernel_size=1),
            nn.GELU(),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim, kernel_size=1),
            nn.GELU(),
        )
        
        # Relative position bias
        max_size = 32
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, 2*max_size-1, 2*max_size-1))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.GroupNorm(32, dim)
        
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
    
    def get_rel_pos_bias(self, H, W, device):
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.reshape(2, -1)
        
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        
        rel_coords[:, :, 0] += self.rel_pos_bias.shape[1] // 2
        rel_coords[:, :, 1] += self.rel_pos_bias.shape[2] // 2
        rel_coords = rel_coords.clamp(0, self.rel_pos_bias.shape[1] - 1)
        
        rel_pos_idx = rel_coords[:, :, 0] * self.rel_pos_bias.shape[2] + rel_coords[:, :, 1]
        rel_pos_idx = rel_pos_idx.long()
        
        bias = self.rel_pos_bias.view(self.num_heads, -1)[:, rel_pos_idx.view(-1)]
        bias = bias.view(self.num_heads, H*W, H*W)
        
        return bias
    
    def forward(self, cls_feat, obj_logits, rel_logits, conf_logits, return_weights=False):
        """
        FIXED: Now uses confidence to weight context.
        
        Args:
            cls_feat: [B, C, H, W] - features
            obj_logits: [B, 36, H, W] - object predictions (logits)
            rel_logits: [B, 26, H, W] - relation predictions (logits)
            conf_logits: [B, 1, H, W] - objectness confidence (logits)
        Returns:
            context_feat: [B, C, H, W] - features with spatial object+relation context
        """
        B, C, H, W = cls_feat.shape
        N = H * W
        
        # Convert to probabilities
        obj_probs = F.softmax(obj_logits, dim=1)  # [B, 36, H, W]
        rel_probs = torch.sigmoid(rel_logits)     # [B, 26, H, W]
        conf = torch.sigmoid(conf_logits)         # [B, 1, H, W]
        
        # CRITICAL FIX: Weight by confidence
        grounded_obj = obj_probs * conf  # [B, 36, H, W]
        grounded_rel = rel_probs * conf  # [B, 26, H, W]
        
        # Combine grounded context
        context_probs = torch.cat([grounded_obj, grounded_rel], dim=1)  # [B, 62, H, W]
        
        # Project Q, K, V
        Q = self.q_proj(cls_feat)
        K = self.k_proj(context_probs)
        V = self.v_proj(context_probs)
        
        # Reshape for multi-head attention
        Q = Q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        # Attention scores
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Relative position bias
        rel_pos_bias = self.get_rel_pos_bias(H, W, cls_feat.device)
        attn = attn + rel_pos_bias.unsqueeze(0)
        
        # CRITICAL FIX: Use confidence as presence bias
        conf_flat = conf.view(B, 1, 1, N)
        presence_bias = (conf_flat - 0.3) * 8.0
        attn = attn + presence_bias
        
        # Softmax
        attn_weights = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Output projection + residual
        out = self.out_proj(out)
        out = self.norm(cls_feat + out)
        
        if return_weights:
            return out, attn_weights.mean(dim=1)
        return out


# Keep aliases for backward compatibility with old names
ObjectContextModule = ObjectContextModuleV2
SceneContextAttention = SceneContextAttentionV2  
ObjectContext = ObjectContextModuleV2
RelationContext = SceneContextAttentionV2


class YOWOMultiTaskV2(nn.Module):
    """
    YOWO Multi-Task V2 with FIXED cascade: Object → Relation → Action
    
    Key fix: Context modules now receive confidence logits to know
    WHERE objects are actually detected, not just WHAT class they might be.
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
                 trainable=False,
                 end2end=False):
        super(YOWOMultiTaskV2, self).__init__()
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
        self.end2end = end2end

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
        
        # ==================== PREDICTIONS ====================
        
        # Confidence (objectness - CRITICAL for attention gating!)
        self.conf_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 1, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # 1. Object Head
        self.obj_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, num_objects, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # Object → Relation context (V2 - uses confidence!)
        self.obj_context = nn.ModuleList([
            ObjectContextModuleV2(head_dim, num_objects)
            for _ in range(len(cfg['stride']))
        ])
        
        # 2. Relation Head
        self.rel_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, num_relations, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # Relation → Action context (V2 - uses confidence!)
        self.rel_context = nn.ModuleList([
            SceneContextAttentionV2(head_dim, num_objects, num_relations)
            for _ in range(len(cfg['stride']))
        ])
        
        # 3. Action Head
        self.act_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, num_actions, kernel_size=1)
                for _ in range(len(cfg['stride']))]) 
        
        # Box regression
        self.reg_preds = nn.ModuleList(
            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                for _ in range(len(cfg['stride']))])

        # ==================== ONE-TO-ONE HEADS (NMS-FREE) ====================
        if self.end2end:
            self.o2o_conf_preds = copy.deepcopy(self.conf_preds)
            self.o2o_obj_preds = copy.deepcopy(self.obj_preds)
            self.o2o_obj_context = copy.deepcopy(self.obj_context)
            self.o2o_rel_preds = copy.deepcopy(self.rel_preds)
            self.o2o_rel_context = copy.deepcopy(self.rel_context)
            self.o2o_act_preds = copy.deepcopy(self.act_preds)
            self.o2o_reg_preds = copy.deepcopy(self.reg_preds)
            print("End-to-End NMS-Free Mode: ENABLED (dual O2M + O2O heads)")

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


    def _forward_single_level(self, level, cls_feat, reg_feat, feat_3d_up, use_o2o=False):
        """
        FIXED: Cascade now passes confidence to context modules.
        
        This allows attention to focus ONLY on positions where objects
        are actually detected (high confidence), not background.
        """
        # Encode
        cls_feat = self.cls_channel_encoders[level](cls_feat, feat_3d_up)
        reg_feat = self.reg_channel_encoders[level](reg_feat, feat_3d_up)
        
        # Head
        cls_feat, reg_feat = self.heads[level](cls_feat, reg_feat)
        
        # Select prediction heads
        if use_o2o and self.end2end:
            conf_pred = self.o2o_conf_preds[level]
            obj_pred = self.o2o_obj_preds[level]
            obj_ctx = self.o2o_obj_context[level]
            rel_pred = self.o2o_rel_preds[level]
            rel_ctx = self.o2o_rel_context[level]
            act_pred = self.o2o_act_preds[level]
            reg_pred = self.o2o_reg_preds[level]
        else:
            conf_pred = self.conf_preds[level]
            obj_pred = self.obj_preds[level]
            obj_ctx = self.obj_context[level]
            rel_pred = self.rel_preds[level]
            rel_ctx = self.rel_context[level]
            act_pred = self.act_preds[level]
            reg_pred = self.reg_preds[level]
        
        # ===== FIXED CASCADE: Object → Relation → Action =====
        # Now passes confidence to context modules!
        
        # 0. Confidence prediction (EARLY - needed for context gating!)
        conf_logits = conf_pred(reg_feat)  # [B, 1, H, W]
        
        # 1. Object prediction
        obj_logits = obj_pred(cls_feat)
        
        # 2. Relation prediction (with CONFIDENCE-GATED object context)
        rel_feat = obj_ctx(cls_feat, obj_logits, conf_logits)  # V2: passes conf!
        rel_logits = rel_pred(rel_feat)
        
        # 3. Action prediction (with CONFIDENCE-GATED object + relation context)
        act_feat = rel_ctx(cls_feat, obj_logits, rel_logits, conf_logits)  # V2: passes conf!
        act_logits = act_pred(act_feat)
        
        # Box regression
        reg_output = reg_pred(reg_feat)
        
        return conf_logits, obj_logits, rel_logits, act_logits, reg_output


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
        
        use_o2o = self.end2end
        
        for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            conf_l, obj_l, rel_l, act_l, reg_l = \
                self._forward_single_level(level, cls_f, reg_f, feat_3d_up, use_o2o=use_o2o)
            
            fmp_size = conf_l.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            conf_l = conf_l.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            obj_l = obj_l.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_objects)
            rel_l = rel_l.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_relations)
            act_l = act_l.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_actions)
            reg_l = reg_l.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            
            obj_l = F.softmax(obj_l, dim=-1)
            rel_l = torch.sigmoid(rel_l)
            act_l = torch.sigmoid(act_l)
            
            cls_l = torch.cat([obj_l, act_l, rel_l], dim=-1)
            
            all_conf.append(conf_l)
            all_cls.append(cls_l)
            all_reg.append(reg_l)
            all_anchors.append(anchors)
        
        batch_bboxes = []
        for b in range(B):
            if self.end2end:
                out = self.post_process_nms_free(
                    [c[b] for c in all_conf],
                    [c[b] for c in all_cls],
                    [r[b] for r in all_reg],
                    all_anchors
                )
            else:
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


    def post_process_nms_free(self, conf_preds, cls_preds, reg_preds, anchors):
        """NMS-free post-processing for O2O head."""
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
        
        return np.concatenate([bboxes, scores[..., None], labels], axis=-1)


    def forward(self, video_clips):
        if not self.trainable:
            return self.inference(video_clips)
        
        key_frame = video_clips[:, :, -1, :, :]
        feat_3d = self.backbone_3d(video_clips)
        cls_feats, reg_feats = self.backbone_2d(key_frame)
        
        o2m_outputs = self._forward_all_levels(cls_feats, reg_feats, feat_3d, use_o2o=False)
        
        if self.end2end:
            cls_feats_detach = [f.detach() for f in cls_feats]
            reg_feats_detach = [f.detach() for f in reg_feats]
            feat_3d_detach = feat_3d.detach()
            
            o2o_outputs = self._forward_all_levels(
                cls_feats_detach, reg_feats_detach, feat_3d_detach, use_o2o=True
            )
            
            return {
                "one2many": o2m_outputs,
                "one2one": o2o_outputs,
            }
        
        return o2m_outputs

    def _forward_all_levels(self, cls_feats, reg_feats, feat_3d, use_o2o=False):
        """Forward through all FPN levels."""
        all_conf = []
        all_obj = []
        all_rel = []
        all_act = []
        all_box = []
        all_anchors = []
        
        for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            conf_l, obj_l, rel_l, act_l, reg_l = \
                self._forward_single_level(level, cls_f, reg_f, feat_3d_up, use_o2o=use_o2o)
            
            fmp_size = conf_l.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
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


# For backward compatibility - alias to old name
YOWOMultiTask = YOWOMultiTaskV2
