"""
YOWO Multi-Task Architecture for Action Genome + Charades - FIXED VERSION

ROOT CAUSES IDENTIFIED:

1. SceneContextAttention Position/Semantic Imbalance
   - Position encoding magnitude: ~0.4
   - Semantic Key projection: ~0.03 (8x weaker!)
   - Attention is dominated by position-to-position matching

2. Value Contains No Context
   - V = value_proj(cls_feat) has no object/relation info
   - Even perfect attention retrieves nothing useful

3. Feature Magnitude Dominance  
   - cls_feat ~0.8, context ~0.1 (4-8x imbalance)
   - Context signal is tiny perturbation

4. BatchNorm Squashing
   - Normalizes away subtle differences

FIXES APPLIED:

1. SceneContextAttention:
   - Position encoding ONLY in Query (not Key) - positions ask "what's around me"
   - Context is projected directly into Value (not just Key)
   - Learnable context_scale to boost semantic signal
   - Use LayerNorm instead of BatchNorm

2. ObjectContextModule:
   - Explicit context scaling to match feature magnitude
   - Use LayerNorm instead of BatchNorm
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


class FixedObjectContextModule(nn.Module):
    """
    Fixed Object Context Module for Relation predictions.
    
    FIXES:
    - Explicit context scaling to match feature magnitude
    - LayerNorm instead of BatchNorm to preserve signal differences
    - Stronger context projection initialization
    """
    def __init__(self, dim=256, num_classes=36):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        
        # Learnable scale for context - starts at 1.0
        self.context_scale = nn.Parameter(torch.ones(1))
        
        # Project object logits to feature dimension
        # Use larger initialization for stronger gradient signal
        self.context_proj = nn.Sequential(
            nn.Conv2d(num_classes, dim // 2, kernel_size=1),
            nn.GELU(),  # GELU instead of ReLU for smoother gradients
            nn.Conv2d(dim // 2, dim, kernel_size=1)
        )
        
        # Custom initialization for stronger context signal
        for m in self.context_proj:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=2.0)  # Larger gain
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Fusion layer: combine original features with object context
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # LayerNorm instead of BatchNorm - preserves signal differences
        self.norm = nn.GroupNorm(32, dim)  # GroupNorm works like LayerNorm per channel
    
    def forward(self, cls_feat, pred_logits, return_weights=False):
        """
        Args:
            cls_feat: [B, C, H, W] - features 
            pred_logits: [B, num_classes, H, W] - object predictions (logits)
        Returns:
            context_feat: [B, C, H, W] - features enriched with object context
        """
        # Project object logits to context embedding with scaling
        obj_context = self.context_proj(pred_logits)  # [B, C, H, W]
        obj_context = obj_context * self.context_scale  # Learnable scale
        
        # Normalize context to match cls_feat magnitude
        # This ensures context isn't drowned out
        if cls_feat.abs().mean() > 0:
            scale_factor = cls_feat.abs().mean() / (obj_context.abs().mean() + 1e-6)
            obj_context = obj_context * scale_factor.clamp(max=10.0)  # Cap to prevent explosion
        
        # Concatenate and fuse
        combined = torch.cat([cls_feat, obj_context], dim=1)  # [B, 2C, H, W]
        out = self.fusion(combined)  # [B, C, H, W]
        
        # Residual connection + normalization
        out = self.norm(cls_feat + out)
        
        if return_weights:
            # Return dummy weights for compatibility
            B, C, H, W = cls_feat.shape
            dummy_weights = torch.ones(B, H*W, H*W, device=cls_feat.device) / (H*W)
            return out, dummy_weights
        return out


class FixedSceneContextAttention(nn.Module):
    """
    Fixed Scene Context Attention for Action prediction.
    
    FIXES:
    1. Position encoding ONLY in Query (not Key)
       - Query: "I'm at position X, what context do I need?"
       - Key: "I contain [object, relation] information" (PURE semantic)
       
    2. Context in BOTH Key AND Value
       - Key guides WHERE to attend (based on semantic content)
       - Value provides WHAT to retrieve (context-enriched features)
       
    3. Stronger context projection with learnable scale
    
    4. LayerNorm for signal preservation
    """
    def __init__(self, dim=256, num_objects=36, num_relations=26, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query projection: Features + Position
        self.query_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # Key projection: ONLY semantic content (no position!)
        # This forces attention to focus on WHAT is there, not WHERE
        self.key_proj = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1)
        )
        
        # Value projection: Features + Context (the WHAT to retrieve)
        # This is the key fix - Value should carry semantic information
        self.context_to_value = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=1)
        )
        self.value_fusion = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        # === POSITION ENCODING (Query only) ===
        max_size = 32
        self.pos_embed_h = nn.Parameter(torch.zeros(1, dim // 2, max_size, 1))
        self.pos_embed_w = nn.Parameter(torch.zeros(1, dim // 2, 1, max_size))
        nn.init.normal_(self.pos_embed_h, std=0.5)  # Reduced from 1.0
        nn.init.normal_(self.pos_embed_w, std=0.5)
        
        # Learnable scales
        self.pos_scale = nn.Parameter(torch.ones(1) * 0.3)  # Reduced position importance
        self.context_scale = nn.Parameter(torch.ones(1) * 2.0)  # Boosted context importance
        
        # Stronger initialization for key/value projections
        for module in [self.key_proj, self.context_to_value]:
            for m in module:
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=2.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        self.norm = nn.GroupNorm(32, dim)  # GroupNorm instead of BatchNorm
    
    def get_position_encoding(self, H, W, device):
        """Generate 2D position encoding for the given spatial dimensions."""
        pos_h = F.interpolate(
            self.pos_embed_h, size=(H, 1), mode='bilinear', align_corners=False
        )
        pos_w = F.interpolate(
            self.pos_embed_w, size=(1, W), mode='bilinear', align_corners=False
        )
        
        pos_h = pos_h.expand(-1, -1, -1, W)
        pos_w = pos_w.expand(-1, -1, H, -1)
        pos = torch.cat([pos_h, pos_w], dim=1)
        
        return pos
    
    def forward(self, cls_feat, obj_pred, rel_pred, return_weights=False):
        """
        Args:
            cls_feat: [B, C, H, W] - features
            obj_pred: [B, 36, H, W] - object predictions (logits)
            rel_pred: [B, 26, H, W] - relation predictions (logits)
        Returns:
            context_feat: [B, C, H, W] - features enriched with scene context
        """
        B, C, H, W = cls_feat.shape
        N = H * W
        
        # Get position encoding for Query only
        pos = self.get_position_encoding(H, W, cls_feat.device)
        pos = pos * self.pos_scale
        
        # Combine object and relation predictions
        context_preds = torch.cat([obj_pred, rel_pred], dim=1)  # [B, 62, H, W]
        
        # === FIXED ATTENTION ===
        # Query: Features + Position (knows WHERE I am)
        Q = self.query_proj(cls_feat) + pos  # [B, C, H, W]
        
        # Key: ONLY semantic content, scaled up (knows WHAT is at each position)
        K = self.key_proj(context_preds) * self.context_scale  # [B, C, H, W]
        
        # Value: Features + Context (retrieves INFORMATION about objects/relations)
        context_features = self.context_to_value(context_preds) * self.context_scale
        V = self.value_fusion(torch.cat([cls_feat, context_features], dim=1))  # [B, C, H, W]
        
        # Reshape for multi-head attention: [B, heads, N, head_dim]
        Q = Q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        # Attention scores
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn, dim=-1)
        
        # Attend to values
        attended = attn_weights @ V
        
        # Reshape back to [B, C, H, W]
        attended = attended.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        attended = self.out_proj(attended)
        
        # Residual connection + normalization
        out = self.norm(cls_feat + attended)
        
        if return_weights:
            avg_weights = attn_weights.mean(dim=1)
            return out, avg_weights
        return out


# Backward compatibility aliases
ObjectContextModule = FixedObjectContextModule
SceneContextAttention = FixedSceneContextAttention
ObjectRelationContextModule = FixedSceneContextAttention
ObjectCrossAttention = FixedObjectContextModule
ObjectRelationCrossAttention = FixedSceneContextAttention


class YOWOMultiTask(nn.Module):
    """
    YOWO with Multi-Task Heads for Action Genome + Charades.
    
    Uses FIXED cross-attention modules that properly propagate context.
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

        # ------------------ Network ---------------------
        # 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            cfg, pretrained=cfg['pretrained_2d'] and trainable)
            
        # 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            cfg, pretrained=cfg['pretrained_3d'] and trainable)

        # cls channel encoder (shared for all classification heads)
        self.cls_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])
            
        # reg channel encoder
        self.reg_channel_encoders = nn.ModuleList(
            [build_channel_encoder(cfg, bk_dim_2d[i]+bk_dim_3d, cfg['head_dim'])
                for i in range(len(cfg['stride']))])

        # head (shared feature processing before prediction)
        self.heads = nn.ModuleList(
            [build_head(cfg) for _ in range(len(cfg['stride']))]
        ) 

        # Prediction layers
        head_dim = cfg['head_dim']
        
        # Confidence prediction
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
        
        # ============ FIXED CROSS-ATTENTION ============
        # Object → Relation → Action prediction chain
        
        # Context modules for RELATION head: sees what objects exist
        self.obj_cross_attn = nn.ModuleList([
            FixedObjectContextModule(
                dim=head_dim,
                num_classes=self.num_objects
            )
            for _ in range(len(cfg['stride']))
        ])
        
        # Scene Context Attention for ACTION head
        # FIXED: Properly propagates object/relation context
        self.obj_rel_cross_attn = nn.ModuleList([
            FixedSceneContextAttention(
                dim=head_dim,
                num_objects=self.num_objects,
                num_relations=self.num_relations,
                num_heads=4
            )
            for _ in range(len(cfg['stride']))
        ])
        
        # Box regression
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
        
        # Object prediction bias
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


    def post_process_multi_hot(self, conf_preds, obj_preds, act_preds, rel_preds, reg_preds, anchors):
        """Post-process predictions for inference (multi-hot output)."""
        all_conf_preds = []
        all_obj_preds = []
        all_act_preds = []
        all_rel_preds = []
        all_box_preds = []
        
        for level, (conf_pred_i, obj_pred_i, act_pred_i, rel_pred_i, reg_pred_i, anchors_i) in enumerate(
            zip(conf_preds, obj_preds, act_preds, rel_preds, reg_preds, anchors)):
            
            # Decode boxes
            box_pred_i = self.decode_boxes(anchors_i, reg_pred_i, self.stride[level])
            
            # Confidence (sigmoid)
            conf_pred_i = torch.sigmoid(conf_pred_i.squeeze(-1))
            
            # Object (softmax for exclusive classification)
            obj_pred_i = torch.softmax(obj_pred_i, dim=-1)
            
            # Actions, Relations (sigmoid for multi-label)
            act_pred_i = torch.sigmoid(act_pred_i)
            rel_pred_i = torch.sigmoid(rel_pred_i)
            
            # Top-k filtering
            topk_conf_pred_i, topk_inds = torch.topk(conf_pred_i, min(self.topk, conf_pred_i.shape[0]))
            topk_obj_pred_i = obj_pred_i[topk_inds]
            topk_act_pred_i = act_pred_i[topk_inds]
            topk_rel_pred_i = rel_pred_i[topk_inds]
            topk_box_pred_i = box_pred_i[topk_inds]
            
            # Threshold filtering
            keep = topk_conf_pred_i.gt(self.conf_thresh)
            topk_conf_pred_i = topk_conf_pred_i[keep]
            topk_obj_pred_i = topk_obj_pred_i[keep]
            topk_act_pred_i = topk_act_pred_i[keep]
            topk_rel_pred_i = topk_rel_pred_i[keep]
            topk_box_pred_i = topk_box_pred_i[keep]
            
            all_conf_preds.append(topk_conf_pred_i)
            all_obj_preds.append(topk_obj_pred_i)
            all_act_preds.append(topk_act_pred_i)
            all_rel_preds.append(topk_rel_pred_i)
            all_box_preds.append(topk_box_pred_i)
        
        # Concatenate across levels
        conf_preds = torch.cat(all_conf_preds, dim=0)
        obj_preds = torch.cat(all_obj_preds, dim=0)
        act_preds = torch.cat(all_act_preds, dim=0)
        rel_preds = torch.cat(all_rel_preds, dim=0)
        box_preds = torch.cat(all_box_preds, dim=0)
        
        # Combine all class predictions for compatibility
        cls_preds = torch.cat([obj_preds, act_preds, rel_preds], dim=-1)
        
        # To CPU/numpy
        scores = conf_preds.cpu().numpy()
        labels = cls_preds.cpu().numpy()
        bboxes = box_preds.cpu().numpy()
        
        # NMS using object class
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
                scores_for_nms = scores_for_nms[:len(bboxes)]
        
        if len(bboxes) > 0:
            out_boxes = np.concatenate([
                bboxes,
                scores_for_nms[..., None],
                labels
            ], axis=-1)
        else:
            out_boxes = np.zeros((0, 5 + self.num_classes))
        
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
            
            # ============ CASCADED PREDICTIONS ============
            conf_pred = self.conf_preds[level](reg_feat)
            obj_pred = self.obj_preds[level](cls_feat)
            
            rel_feat = self.obj_cross_attn[level](cls_feat, obj_pred)
            rel_pred = self.rel_preds[level](rel_feat)
            
            act_feat = self.obj_rel_cross_attn[level](cls_feat, obj_pred, rel_pred)
            act_pred = self.act_preds[level](act_feat)
            
            reg_pred = self.reg_preds[level](reg_feat)
            
            # Generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            # Reshape: [B, C, H, W] -> [B, H*W, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_objects)
            act_pred = act_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_actions)
            rel_pred = rel_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_relations)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            
            all_conf_preds.append(conf_pred)
            all_obj_preds.append(obj_pred)
            all_act_preds.append(act_pred)
            all_rel_preds.append(rel_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)
        
        # Batch processing
        batch_bboxes = []
        for batch_idx in range(B):
            cur_conf_preds = [p[batch_idx] for p in all_conf_preds]
            cur_obj_preds = [p[batch_idx] for p in all_obj_preds]
            cur_act_preds = [p[batch_idx] for p in all_act_preds]
            cur_rel_preds = [p[batch_idx] for p in all_rel_preds]
            cur_reg_preds = [p[batch_idx] for p in all_reg_preds]
            
            out_boxes = self.post_process_multi_hot(
                cur_conf_preds, cur_obj_preds, cur_act_preds, cur_rel_preds, 
                cur_reg_preds, all_anchors)
            
            # Normalize boxes
            out_boxes[..., :4] /= max(img_h, img_w)
            out_boxes[..., :4] = out_boxes[..., :4].clip(0., 1.)
            
            batch_bboxes.append(out_boxes)
        
        return batch_bboxes


    def forward(self, video_clips):
        """Forward pass."""
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
            
            # ============ CASCADED PREDICTIONS ============
            conf_pred = self.conf_preds[level](reg_feat)
            obj_pred = self.obj_preds[level](cls_feat)
            
            rel_feat = self.obj_cross_attn[level](cls_feat, obj_pred)
            rel_pred = self.rel_preds[level](rel_feat)
            
            act_feat = self.obj_rel_cross_attn[level](cls_feat, obj_pred, rel_pred)
            act_pred = self.act_preds[level](act_feat)
            
            reg_pred = self.reg_preds[level](reg_feat)
            
            # Generate anchors
            fmp_size = conf_pred.shape[-2:]
            anchors = self.generate_anchors(fmp_size, self.stride[level])
            
            # Reshape: [B, C, H, W] -> [B, H*W, C]
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            act_pred = act_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            rel_pred = rel_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
            
            # Decode boxes
            box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])
            
            all_conf_preds.append(conf_pred)
            all_obj_preds.append(obj_pred)
            all_act_preds.append(act_pred)
            all_rel_preds.append(rel_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)
        
        # Output dict for loss computation
        outputs = {
            "pred_conf": all_conf_preds,
            "pred_obj": all_obj_preds,
            "pred_act": all_act_preds,
            "pred_rel": all_rel_preds,
            "pred_box": all_box_preds,
            "anchors": all_anchors,
            "strides": self.stride
        }
        
        return outputs
