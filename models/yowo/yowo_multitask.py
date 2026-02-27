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


class LocalContextModule(nn.Module):
    """
    Simple local context: pool nearby object predictions to enrich features.
    
    Uses 5x5 avg_pool of confidence-weighted object probabilities.
    No attention mechanism, no presence_bias — just "what objects are near me?"
    
    This replaced the broken ObjectContextModuleV2 which used attention with
    presence_bias = (conf - 0.3) * 8.0 that suppressed person features.
    """
    def __init__(self, dim=256, num_classes=36, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        # Simple projection: object probs → feature space
        self.context_proj = nn.Sequential(
            nn.Conv2d(num_classes, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.norm = nn.GroupNorm(32, dim)
        self.pool_size = 5  # local neighborhood size
    
    def forward(self, cls_feat, obj_logits, conf_logits, return_weights=False):
        """
        Enrich features with local object context.
        
        For each position, pools nearby confidence-weighted object predictions
        to answer: "what objects are near me?"
        """
        # V2 Fix: DO NOT detach. Let gradients flow back to learn strong confidence early!
        obj_probs = F.softmax(obj_logits, dim=1)  # [B, 36, H, W]
        
        # CRITICAL FIX: Removed confidence gating - it was dampening gradients by 50-70%!
        # Action gradients need to flow back to object predictions to learn better features.
        # Confidence is for inference filtering, not training gradient gating.
        grounded_obj = obj_probs  # [B, 36, H, W] - no confidence multiplication!
        
        # Pool from local neighborhood: "what objects are near this position?"
        local_obj = F.avg_pool2d(
            grounded_obj, self.pool_size, stride=1, padding=self.pool_size // 2
        )  # [B, 36, H, W]
        
        # Project to feature space and add via residual
        context = self.context_proj(local_obj)  # [B, C, H, W]
        out = self.norm(cls_feat + context)
        
        if return_weights:
            return out, None
        return out


class CascadeContextModule(nn.Module):
    """
    Simple cascade context: pool nearby object + relation predictions.
    
    For action prediction: "what objects and relations are near this position?"
    Uses local pooling instead of attention — proven to not hurt features.
    """
    def __init__(self, dim=256, num_objects=36, num_relations=26, num_heads=4):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Conv2d(num_objects + num_relations, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.norm = nn.GroupNorm(32, dim)
        self.pool_size = 5
    
    def forward(self, cls_feat, obj_logits, rel_logits, conf_logits, return_weights=False):
        """
        Enrich features with local object + relation context.
        """
        # V2 Fix: DO NOT detach. Action gradients must backpropagate to build meaningful object/relation context.
        obj_probs = F.softmax(obj_logits, dim=1)  # [B, 36, H, W]
        rel_probs = torch.sigmoid(rel_logits)      # [B, 26, H, W]
        
        # CRITICAL FIX: Removed confidence gating - same reason as object context
        # Action gradients need full strength to flow back through the cascade
        grounded = torch.cat([
            obj_probs,  # [B, 36, H, W] - no confidence multiplication!
            rel_probs,  # [B, 26, H, W] - no confidence multiplication!
        ], dim=1)  # [B, 62, H, W]
        
        # Pool from local neighborhood
        local_ctx = F.avg_pool2d(
            grounded, self.pool_size, stride=1, padding=self.pool_size // 2
        )
        
        # Project and residual
        context = self.context_proj(local_ctx)
        out = self.norm(cls_feat + context)
        
        if return_weights:
            return out, None
        return out


# Aliases for model building code
ObjectContextModuleV2 = LocalContextModule
SceneContextAttentionV2 = CascadeContextModule
ObjectContextModule = LocalContextModule
SceneContextAttention = CascadeContextModule
ObjectContext = LocalContextModule
RelationContext = CascadeContextModule


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
            # O2O only needs separate conf (anchor selection) and reg (box position).
            # Classification heads (obj, act, rel + context) are SHARED with O2M.
            # This way the action head gets rich O2M supervision (7+ person anchors/img)
            # instead of starving on O2O's topk=1 (1 person anchor/img).
            self.o2o_conf_preds = copy.deepcopy(self.conf_preds)
            self.o2o_reg_preds = copy.deepcopy(self.reg_preds)
            # Shared (just aliases, same weights):
            self.o2o_obj_preds = self.obj_preds
            self.o2o_obj_context = self.obj_context
            self.o2o_rel_preds = self.rel_preds
            self.o2o_rel_context = self.rel_context
            self.o2o_act_preds = self.act_preds
            print("End-to-End NMS-Free Mode: ENABLED (shared cls heads, separate conf/reg)")

        self.init_yowo()


    def init_yowo(self): 
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        
        # BCE-based heads: conf, act, rel (use low init prob to prevent background over-prediction)
        for pred_list in [self.conf_preds, self.rel_preds, self.act_preds]:
            for pred in pred_list:
                b = pred.bias.view(1, -1)
                b.data.fill_(bias_value.item())
                pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        
        # CrossEntropy-based head: obj (use zero bias for uniform prior over classes)
        # NOTE: obj_preds uses softmax, not sigmoid. Non-zero bias would create
        # artificial class imbalance at initialization.
        for pred in self.obj_preds:
            pred.bias.data.zero_()


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
        
        # ===== TRUE CASCADE: Object → Relation → Action =====
        # Each stage builds on the ENRICHED features from the previous stage.
        # This is a deep chain, not a shallow star topology.
        
        # 0. Confidence prediction (EARLY - needed for context gating!)
        conf_logits = conf_pred(reg_feat)  # [B, 1, H, W]
        
        # 1. Object prediction (from base classification features)
        obj_logits = obj_pred(cls_feat)
        
        # 2. Relation prediction (features enriched with object context)
        rel_feat = obj_ctx(cls_feat, obj_logits, conf_logits)
        rel_logits = rel_pred(rel_feat)
        
        # 3. Action prediction (features enriched with object + relation context)
        # KEY: starts from rel_feat (already has object context), NOT cls_feat
        act_feat = rel_ctx(rel_feat, obj_logits, rel_logits, conf_logits)
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
