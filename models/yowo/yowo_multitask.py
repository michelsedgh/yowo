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


class GlobalSceneContext(nn.Module):
    """
    Global Scene Context for action prediction using object+relation information.
    
    KEY INSIGHT: Actions depend on GLOBAL scene understanding, not local attention.
    
    This module:
    1. Pools object predictions globally (max-pool) to get scene-level context
       - "Is there a laptop ANYWHERE in the scene?" -> Yes/No confidence
    2. Pools relation predictions globally 
       - "Is there a 'holding' relation ANYWHERE?" -> Yes/No confidence  
    3. Projects this global context to feature dimension
    4. Broadcasts to all positions and fuses with local features
    
    This way, the action prediction at EVERY position knows:
    - All objects present in the scene
    - All relations happening in the scene
    - And can predict actions accordingly
    
    Example: Person, laptop, couch detected + sitting_on, holding relations
    -> Action head knows: "typing on laptop", "sitting on couch" are likely
    """
    def __init__(self, dim=256, num_objects=36, num_relations=26, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_objects = num_objects
        self.num_relations = num_relations
        # num_heads kept for API compatibility but not used
        
        # Project global context (obj + rel) to feature dimension
        # Using a small MLP to learn the mapping
        self.context_proj = nn.Sequential(
            nn.Linear(num_objects + num_relations, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # Initialize with standard weights - let the model learn the right magnitude
        for m in self.context_proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)  # Standard gain, not reduced
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Fusion: combine local features with global context
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )
        
        # Initialize fusion with standard weights
        nn.init.xavier_uniform_(self.fusion[0].weight, gain=1.0)
        nn.init.xavier_uniform_(self.fusion[2].weight, gain=1.0)
        
        # Learnable scale for context contribution
        # Starts at 1.0, model can learn to adjust
        self.context_scale = nn.Parameter(torch.tensor(1.0))
        
        # GroupNorm for stable training
        self.norm = nn.GroupNorm(32, dim)
    
    def forward(self, cls_feat, obj_pred, rel_pred, return_weights=False):
        """
        Args:
            cls_feat: [B, C, H, W] - features
            obj_pred: [B, 36, H, W] - object predictions (LOGITS)
            rel_pred: [B, 26, H, W] - relation predictions (LOGITS)
        Returns:
            context_feat: [B, C, H, W] - features enriched with global scene context
        """
        B, C, H, W = cls_feat.shape
        
        # Normalize predictions to probabilities
        obj_probs = F.softmax(obj_pred, dim=1)  # [B, 36, H, W]
        rel_probs = torch.sigmoid(rel_pred)     # [B, 26, H, W]
        
        # GLOBAL POOLING: Get scene-level context from ALL positions
        # Max-pool across spatial dimensions (H, W)
        # This answers: "What objects/relations exist ANYWHERE in the scene?"
        obj_global = obj_probs.max(dim=-1)[0].max(dim=-1)[0]  # [B, 36]
        rel_global = rel_probs.max(dim=-1)[0].max(dim=-1)[0]  # [B, 26]
        
        # Combine object and relation global context
        global_context = torch.cat([obj_global, rel_global], dim=1)  # [B, 62]
        
        # Project to feature dimension
        context_embed = self.context_proj(global_context)  # [B, dim]
        
        # Broadcast to all spatial positions
        # [B, dim] -> [B, dim, 1, 1] -> [B, dim, H, W]
        context_broadcast = context_embed.unsqueeze(-1).unsqueeze(-1)
        context_broadcast = context_broadcast.expand(-1, -1, H, W)  # [B, C, H, W]
        
        # Scale context to match feature magnitudes
        feat_scale = cls_feat.abs().mean().clamp(min=0.01)
        ctx_scale = context_broadcast.abs().mean().clamp(min=0.01)
        context_scaled = context_broadcast * (feat_scale / ctx_scale)
        
        # Apply learnable context scale
        context_scaled = context_scaled * self.context_scale.abs().clamp(min=0.1, max=2.0)
        
        # Fuse local features with global context
        combined = torch.cat([cls_feat, context_scaled], dim=1)  # [B, 2C, H, W]
        delta = self.fusion(combined)  # [B, C, H, W]
        
        # Residual connection + normalization
        out = self.norm(cls_feat + delta)
        
        if return_weights:
            # Return dummy uniform weights for API compatibility
            # (No actual attention in this module)
            dummy_weights = torch.ones(B, H*W, H*W, device=cls_feat.device) / (H*W)
            return out, dummy_weights
        return out


# Keep original name as alias for backward compatibility
class SceneContextAttention(GlobalSceneContext):
    """Alias pointing to GlobalSceneContext for backward compatibility."""
    pass


# Keep old name as alias for compatibility
class ObjectRelationContextModule(SceneContextAttention):
    """Alias for backward compatibility."""
    pass


# Aliases for backward compatibility with existing code
ObjectCrossAttention = ObjectContextModule
ObjectRelationCrossAttention = ObjectRelationContextModule


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
        
        # ============ CASCADED CROSS-ATTENTION ============
        # Object → Relation → Action prediction chain
        
        # Context modules for RELATION head: sees what objects exist
        # "I'm predicting 'holding' relation - but holding WHAT?"
        self.obj_cross_attn = nn.ModuleList([
            ObjectCrossAttention(
                dim=head_dim,
                num_classes=self.num_objects
            )
            for _ in range(len(cfg['stride']))
        ])
        
        # Scene Context Attention for ACTION head: attends to ENTIRE scene
        # Learns which objects/relations are relevant for action prediction
        # Uses cross-attention: Key=obj+rel predictions, Value=features
        self.obj_rel_cross_attn = nn.ModuleList([
            SceneContextAttention(
                dim=head_dim,
                num_objects=self.num_objects,
                num_relations=self.num_relations,
                num_heads=4  # Multi-head attention for diverse patterns
            )
            for _ in range(len(cfg['stride']))
        ])
        
        # NOTE: Interaction head REMOVED - redundant with negative relation classes
        # (notlookingat, unsure, notcontacting already indicate no interaction)
        
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
        
        # NOTE: Interaction head removed - no longer needs bias init


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
        """
        Post-process predictions for inference (multi-hot output).
        
        Returns boxes with format: [x1, y1, x2, y2, conf, obj_classes..., act_classes..., rel_classes...]
        Note: interact_pred removed - relation classes include negative relations (notlookingat, notcontacting)
        """
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
        # Format: [obj_36, act_157, rel_26] = 219 dims
        cls_preds = torch.cat([obj_preds, act_preds, rel_preds], dim=-1)
        
        # To CPU/numpy
        scores = conf_preds.cpu().numpy()
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
                scores_for_nms = scores_for_nms[:len(bboxes)]
        
        # Output: [x1, y1, x2, y2, conf, classes...]
        # Classes start at index 5 (no separate interact score needed)
        if len(bboxes) > 0:
            out_boxes = np.concatenate([
                bboxes,                          # [0:4] bbox
                scores_for_nms[..., None],       # [4] confidence
                labels                           # [5:] classes (obj+act+rel = 219)
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
            # Step 1: Object prediction (no dependencies)
            conf_pred = self.conf_preds[level](reg_feat)
            obj_pred = self.obj_preds[level](cls_feat)
            
            # Step 2: Relation prediction (sees objects via cross-attention)
            rel_feat = self.obj_cross_attn[level](cls_feat, obj_pred)
            rel_pred = self.rel_preds[level](rel_feat)  # Object-aware!
            
            # Step 3: Action prediction (sees objects + relations via cross-attention)
            act_feat = self.obj_rel_cross_attn[level](cls_feat, obj_pred, rel_pred)
            act_pred = self.act_preds[level](act_feat)  # Object+Relation-aware!
            
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
            # Step 1: Object prediction (no dependencies)
            conf_pred = self.conf_preds[level](reg_feat)
            obj_pred = self.obj_preds[level](cls_feat)
            
            # Step 2: Relation prediction (sees objects via cross-attention)
            rel_feat = self.obj_cross_attn[level](cls_feat, obj_pred)
            rel_pred = self.rel_preds[level](rel_feat)  # Object-aware!
            
            # Step 3: Action prediction (sees objects + relations via cross-attention)
            act_feat = self.obj_rel_cross_attn[level](cls_feat, obj_pred, rel_pred)
            act_pred = self.act_preds[level](act_feat)  # Object+Relation-aware!
            
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
            "pred_conf": all_conf_preds,        # List[Tensor] [B, M, 1]
            "pred_obj": all_obj_preds,          # List[Tensor] [B, M, 36]
            "pred_act": all_act_preds,          # List[Tensor] [B, M, 157]
            "pred_rel": all_rel_preds,          # List[Tensor] [B, M, 26]
            "pred_box": all_box_preds,          # List[Tensor] [B, M, 4]
            "anchors": all_anchors,             # List[Tensor] [M, 2]
            "strides": self.stride              # List[int]
        }
        
        return outputs


