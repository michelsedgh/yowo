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


class ObjectContextModule(nn.Module):
    """
    Provides object context to the relation head.
    
    KEY FIX: Uses SOFTMAX-normalized predictions instead of raw logits.
    
    This ensures:
    1. Context values are in [0,1] range (probabilities)
    2. Semantic meaning: high probability = this object class is present
    3. Balanced magnitudes with features for proper fusion
    4. Gradients flow through softmax to object predictions
    """
    def __init__(self, dim=256, num_classes=36):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # Project object probabilities to feature dimension
        # Input is softmax probabilities [0,1], not raw logits
        self.context_proj = nn.Sequential(
            nn.Conv2d(num_classes, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # Initialize to preserve probability magnitudes
        for m in self.context_proj:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Fusion: combine features with object context
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # Initialize fusion to start as identity-like for features
        nn.init.xavier_uniform_(self.fusion[0].weight, gain=0.5)
        nn.init.xavier_uniform_(self.fusion[2].weight, gain=0.5)
        
        # GroupNorm for stable training
        self.norm = nn.GroupNorm(32, dim)
    
    def forward(self, cls_feat, pred_logits, return_weights=False):
        """
        Args:
            cls_feat: [B, C, H, W] - features 
            pred_logits: [B, num_classes, H, W] - object predictions (LOGITS)
        Returns:
            context_feat: [B, C, H, W] - features enriched with object context
        """
        # CRITICAL FIX: Convert logits to probabilities with softmax
        # Now obj_probs is in [0,1] with semantic meaning
        obj_probs = F.softmax(pred_logits, dim=1)  # [B, num_classes, H, W]
        
        # Project probabilities to context embedding
        obj_context = self.context_proj(obj_probs)  # [B, C, H, W]
        
        # Match context magnitude to features for balanced fusion
        feat_scale = cls_feat.abs().mean().clamp(min=0.01)
        ctx_scale = obj_context.abs().mean().clamp(min=0.01)
        obj_context = obj_context * (feat_scale / ctx_scale)
        
        # Concatenate and fuse
        combined = torch.cat([cls_feat, obj_context], dim=1)  # [B, 2C, H, W]
        delta = self.fusion(combined)  # [B, C, H, W]
        
        # Residual connection + normalization
        out = self.norm(cls_feat + delta)
        
        if return_weights:
            B, C, H, W = cls_feat.shape
            dummy_weights = torch.ones(B, H*W, H*W, device=cls_feat.device) / (H*W)
            return out, dummy_weights
        return out


class SceneContextAttention(nn.Module):
    """
    Cross-attention for action prediction using object+relation context.
    
    KEY FIX: Uses NORMALIZED predictions (softmax/sigmoid) instead of raw logits.
    
    This ensures:
    1. Object probs in [0,1] via softmax (exclusive classes)
    2. Relation probs in [0,1] via sigmoid (multi-label)
    3. Balanced Q/K magnitudes for meaningful attention
    4. Model learns WHERE to attend based on WHAT objects/relations exist
    """
    def __init__(self, dim=256, num_objects=36, num_relations=26, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_objects = num_objects
        self.num_relations = num_relations
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection: from features
        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Key projection: from normalized context predictions
        # Input is probabilities [0,1], dimensionality = num_objects + num_relations
        self.key_proj = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # Value projection: features + context for rich retrieval
        self.context_to_value = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.value_fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        # Output projection with small init for stable residual
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        
        # Position encoding (helps query be position-aware)
        max_size = 32
        self.pos_embed_h = nn.Parameter(torch.zeros(1, dim // 2, max_size, 1))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, dim // 2, 1, max_size))
        nn.init.normal_(self.pos_embed_h, std=0.02)
        nn.init.normal_(self.pos_embed_w, std=0.02)
        
        # Initialize projections for balanced magnitudes
        for module in [self.key_proj, self.context_to_value]:
            for m in module:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # GroupNorm for stable training
        self.norm = nn.GroupNorm(32, dim)
    
    def get_position_encoding(self, H, W, device):
        """Generate 2D position encoding."""
        pos_h = F.interpolate(
            self.pos_embed_h, size=(H, 1), mode='bilinear', align_corners=False
        )
        pos_w = F.interpolate(
            self.pos_embed_w, size=(1, W), mode='bilinear', align_corners=False
        )
        pos_h = pos_h.expand(-1, -1, -1, W)
        pos_w = pos_w.expand(-1, -1, H, -1)
        return torch.cat([pos_h, pos_w], dim=1)
    
    def forward(self, cls_feat, obj_pred, rel_pred, return_weights=False):
        """
        Args:
            cls_feat: [B, C, H, W] - features
            obj_pred: [B, 36, H, W] - object predictions (LOGITS)
            rel_pred: [B, 26, H, W] - relation predictions (LOGITS)
        Returns:
            context_feat: [B, C, H, W] - features enriched with scene context
        """
        B, C, H, W = cls_feat.shape
        N = H * W
        
        # CRITICAL FIX: Normalize predictions to probabilities
        # Objects: softmax (mutually exclusive)
        # Relations: sigmoid (multi-label)
        obj_probs = F.softmax(obj_pred, dim=1)  # [B, 36, H, W], sums to 1
        rel_probs = torch.sigmoid(rel_pred)     # [B, 26, H, W], each in [0,1]
        
        # Combine normalized predictions
        context_probs = torch.cat([obj_probs, rel_probs], dim=1)  # [B, 62, H, W]
        
        # Query: features + position encoding
        pos = self.get_position_encoding(H, W, cls_feat.device)
        Q = self.query_proj(cls_feat) + pos  # [B, C, H, W]
        
        # Key: from context probabilities (semantic content)
        K = self.key_proj(context_probs)  # [B, C, H, W]
        
        # Match Q and K magnitudes for balanced attention
        q_scale = Q.abs().mean().clamp(min=0.01)
        k_scale = K.abs().mean().clamp(min=0.01)
        K = K * (q_scale / k_scale)
        
        # Value: features + projected context
        context_features = self.context_to_value(context_probs)
        # Scale context to match features
        feat_scale = cls_feat.abs().mean().clamp(min=0.01)
        ctx_scale = context_features.abs().mean().clamp(min=0.01)
        context_features = context_features * (feat_scale / ctx_scale)
        V = self.value_fusion(torch.cat([cls_feat, context_features], dim=1))
        
        # Reshape for multi-head attention: [B, heads, N, head_dim]
        Q = Q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        # Attention: Q @ K^T / sqrt(d)
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        
        # Attend to values
        attended = attn_weights @ V
        
        # Reshape back
        attended = attended.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        attended = self.out_proj(attended)
        
        # Residual + norm
        out = self.norm(cls_feat + attended)
        
        if return_weights:
            return out, attn_weights.mean(dim=1)
        return out


# Aliases for backward compatibility
ObjectContext = ObjectContextModule
RelationContext = SceneContextAttention
ObjectCrossAttention = ObjectContextModule
ObjectRelationCrossAttention = SceneContextAttention
ObjectRelationContextModule = SceneContextAttention  # Old class name from 810e3af


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
    
    NEW: end2end mode enables dual-head for NMS-free inference.
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
                 end2end=False):  # NEW: Enable dual-head NMS-free mode
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
        self.end2end = end2end  # NMS-free mode

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

        # ==================== ONE-TO-ONE HEADS (NMS-FREE) ====================
        # Duplicate heads for O2O - trained with topk=1 matcher
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
        """Process single FPN level with full cascade.
        
        Args:
            use_o2o: If True, use One-to-One heads (for NMS-free inference)
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
        
        # ===== CASCADE: Object → Relation → Action =====
        
        # 1. Object prediction
        obj_logits = obj_pred(cls_feat)
        
        # 2. Relation prediction (with object context)
        rel_feat = obj_ctx(cls_feat, obj_logits)
        rel_logits = rel_pred(rel_feat)
        
        # 3. Action prediction (with object + relation context)
        # NOTE: Use cls_feat (original features) as base, like old architecture
        # The context module attends to obj+rel predictions for enrichment
        act_feat = rel_ctx(cls_feat, obj_logits, rel_logits)
        act_logits = act_pred(act_feat)
        
        # Confidence and box
        conf_logits = conf_pred(reg_feat)
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
        
        # Use O2O heads if end2end mode (NMS-free inference)
        use_o2o = self.end2end
        
        for level, (cls_f, reg_f) in enumerate(zip(cls_feats, reg_feats)):
            feat_3d_up = F.interpolate(feat_3d, scale_factor=2 ** (2 - level))
            
            conf_l, obj_l, rel_l, act_l, reg_l = \
                self._forward_single_level(level, cls_f, reg_f, feat_3d_up, use_o2o=use_o2o)
            
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
            if self.end2end:
                # NMS-FREE post-processing for O2O head
                out = self.post_process_nms_free(
                    [c[b] for c in all_conf],
                    [c[b] for c in all_cls],
                    [r[b] for r in all_reg],
                    all_anchors
                )
            else:
                # Standard post-processing with NMS
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
        """NMS-free post-processing for O2O head.
        
        O2O head was trained to output single predictions per object,
        so we just need confidence thresholding - NO NMS needed!
        """
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
        
        # NO NMS! Just return the predictions directly
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
        
        # ========== ONE-TO-MANY (O2M) FORWARD ==========
        o2m_outputs = self._forward_all_levels(cls_feats, reg_feats, feat_3d, use_o2o=False)
        
        if self.end2end:
            # ========== ONE-TO-ONE (O2O) FORWARD ==========
            # CRITICAL: Use DETACHED features to prevent gradient conflict!
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

