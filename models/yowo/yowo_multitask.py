"""
YOWO Multi-Task Architecture for Action Genome + Charades

This module implements a three-head architecture that separates:
1. Object Head (36 classes) - Softmax/CE loss - "WHAT is it?"
2. Relation Head (26 classes) - Sigmoid/BCE loss - "HOW does it interact?"
3. Action Head (157 classes) - Sigmoid/BCE loss - "WHAT is it doing?" (Person-only)

Key architectural features:
- Objects are mutually exclusive (Softmax enforces this)
- Actions only apply to Person boxes (masking prevents invalid learning)
- Each head can specialize on its feature requirements
- Cascaded context: Object → Relation → Action
- Location-aware global context for multi-person awareness
- Multi-scale temporal pooling (recent + overall) in X3D backbone

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
    3. **NEW**: Also captures WHERE each max value was detected (location hints)
       - Uses SOFT-ARGMAX for differentiable position extraction
       - Enables multi-person awareness: "laptop at (0.3,0.4), chair at (0.8,0.7)"
    4. Projects this global context to feature dimension
    5. Broadcasts to all positions and fuses with local features
    
    This way, the action prediction at EVERY position knows:
    - All objects present in the scene + WHERE they are
    - All relations happening in the scene + WHERE they occur
    - Model can learn to use or ignore position based on action type
    
    Example: Person A with laptop at left, Person B with chair at right
    -> Model learns position-aware action associations
    
    SOFT-ARGMAX: Instead of hard argmax (non-differentiable), we use softmax-weighted
    expected positions. This allows gradients to flow through position computations,
    enabling the model to directly learn position-action associations.
    """
    def __init__(self, dim=256, num_objects=36, num_relations=26, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_objects = num_objects
        self.num_relations = num_relations
        # num_heads kept for API compatibility but not used
        
        # Context input dimension:
        # - Object confidences: num_objects (36)
        # - Relation confidences: num_relations (26) 
        # - Object absolute locations (y, x): num_objects * 2 (72)
        # - Relation absolute locations (y, x): num_relations * 2 (52)
        # - Object relative-to-person (dy, dx): num_objects * 2 (72)  <- NEW!
        # - Relation relative-to-person (dy, dx): num_relations * 2 (52)  <- NEW!
        # Total: 36 + 26 + 72 + 52 + 72 + 52 = 310
        context_input_dim = (num_objects + num_relations) * 5  # conf + abs_y + abs_x + rel_y + rel_x
        
        # Project global context (obj + rel + locations) to feature dimension
        # Using a small MLP to learn the mapping
        self.context_proj = nn.Sequential(
            nn.Linear(context_input_dim, dim),
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
        
        # Temperature for soft-argmax (lower = sharper, closer to hard argmax)
        # 0.1 is a good balance: sharp enough to approximate argmax, but still differentiable
        self.temperature = 0.1
    
    def soft_argmax_2d(self, probs: torch.Tensor, H: int, W: int) -> tuple:
        """
        Compute soft (differentiable) argmax coordinates.
        
        Instead of returning the index of the max value (non-differentiable),
        this returns the expected position using softmax-weighted coordinates.
        
        Args:
            probs: [B, C, H*W] - probability distribution over spatial positions
            H, W: Spatial dimensions
            
        Returns:
            y, x: [B, C] - expected coordinates in [0, 1]
        """
        B, C, HW = probs.shape
        device = probs.device
        
        # Create normalized coordinate grids [0, 1]
        y_coords = torch.arange(H, device=device).float() / max(H - 1, 1)
        x_coords = torch.arange(W, device=device).float() / max(W - 1, 1)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        y_flat = y_grid.flatten()  # [HW]
        x_flat = x_grid.flatten()  # [HW]
        
        # Sharpen with temperature and normalize to get attention weights
        # Lower temperature = more peaked distribution (closer to argmax)
        weights = F.softmax(probs / self.temperature, dim=-1)  # [B, C, HW]
        
        # Compute expected (weighted average) coordinates
        y = (weights * y_flat).sum(dim=-1)  # [B, C]
        x = (weights * x_flat).sum(dim=-1)  # [B, C]
        
        return y, x
    
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
        
        # === GLOBAL CONFIDENCE VALUES ===
        # Max-pool across spatial dimensions (H, W)
        # This answers: "What objects/relations exist ANYWHERE in the scene?"
        obj_global = obj_probs.max(dim=-1)[0].max(dim=-1)[0]  # [B, 36]
        rel_global = rel_probs.max(dim=-1)[0].max(dim=-1)[0]  # [B, 26]
        
        # === LOCATION HINTS (DIFFERENTIABLE!) ===
        # For each class, WHERE is the expected position of that class?
        # Uses soft-argmax for differentiable position extraction
        
        # Flatten spatial dims for soft-argmax
        obj_flat = obj_probs.view(B, self.num_objects, -1)  # [B, 36, H*W]
        obj_y, obj_x = self.soft_argmax_2d(obj_flat, H, W)  # [B, 36] each
        
        # Same for relations
        rel_flat = rel_probs.view(B, self.num_relations, -1)  # [B, 26, H*W]
        rel_y, rel_x = self.soft_argmax_2d(rel_flat, H, W)  # [B, 26] each
        
        
        # === RELATIVE-TO-PERSON POSITIONS (NEW!) ===
        # Person is object class 0 - compute distances from person to all objects/relations
        # This directly answers: "How far is the laptop from the person?"
        person_y = obj_y[:, 0:1]  # [B, 1] - person's Y position
        person_x = obj_x[:, 0:1]  # [B, 1] - person's X position
        
        # Relative positions: (object_pos - person_pos)
        # Values in range [-1, 1]: negative = above/left of person, positive = below/right
        obj_rel_y = obj_y - person_y  # [B, 36] - each object's Y relative to person
        obj_rel_x = obj_x - person_x  # [B, 36] - each object's X relative to person
        rel_rel_y = rel_y - person_y  # [B, 26] - each relation's Y relative to person
        rel_rel_x = rel_x - person_x  # [B, 26] - each relation's X relative to person
        
        # === COMBINE EVERYTHING ===
        # Confidences (62) + Absolute positions (124) + Relative positions (124) = 310
        global_context = torch.cat([
            # What exists in the scene?
            obj_global,   # Object confidences (36)
            rel_global,   # Relation confidences (26)
            # WHERE are they in the frame? (absolute)
            obj_y,        # Object Y positions (36)
            obj_x,        # Object X positions (36)
            rel_y,        # Relation Y positions (26)
            rel_x,        # Relation X positions (26)
            # WHERE are they RELATIVE TO PERSON? (explicit relative)
            obj_rel_y,    # Object Y relative to person (36)
            obj_rel_x,    # Object X relative to person (36)
            rel_rel_y,    # Relation Y relative to person (26)
            rel_rel_x     # Relation X relative to person (26)
        ], dim=1)  # [B, 310]
        
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


class ActionObjectCooccurrence(nn.Module):
    """
    Explicit action-object co-occurrence modeling.
    
    This module learns which objects commonly co-occur with which actions:
    - "typing" strongly associated with "laptop", "keyboard"
    - "eating" strongly associated with "food", "fork", "spoon"
    - "sitting" strongly associated with "chair", "couch", "bed"
    
    The matrix is initialized small and learned from data.
    It provides an additional signal to the action head based on detected objects.
    
    Integration:
        Applied AFTER computing action logits, adding object-based priors.
    """
    
    def __init__(self, num_actions: int = 157, num_objects: int = 36, 
                 hidden_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.num_actions = num_actions
        self.num_objects = num_objects
        self.temperature = temperature
        
        # Learnable co-occurrence embeddings
        # Low-rank factorization: M = action_embed @ object_embed.T
        # More efficient than full 157x36 matrix
        self.action_embed = nn.Parameter(torch.randn(num_actions, hidden_dim) * 0.01)
        self.object_embed = nn.Parameter(torch.randn(num_objects, hidden_dim) * 0.01)
        
        # Scale factor for co-occurrence contribution (starts small, model learns to increase)
        self.cooc_scale = nn.Parameter(torch.tensor(0.1))
        
    def get_cooccurrence_matrix(self) -> torch.Tensor:
        """Get the full co-occurrence matrix (for visualization/analysis)."""
        return torch.matmul(self.action_embed, self.object_embed.T)  # [num_actions, num_objects]
    
    def forward(self, action_logits: torch.Tensor, object_probs: torch.Tensor) -> torch.Tensor:
        """
        Apply object-based action priors.
        
        Args:
            action_logits: [B, num_actions, H, W] - raw action predictions
            object_probs: [B, num_objects, H, W] - softmax object probabilities
            
        Returns:
            enhanced_logits: [B, num_actions, H, W] - action logits with object priors
        """
        B, A, H, W = action_logits.shape
        
        # Compute co-occurrence matrix
        cooc = torch.matmul(self.action_embed, self.object_embed.T)  # [A, O]
        
        # Global object presence (max-pool across space)
        obj_global = object_probs.max(dim=-1)[0].max(dim=-1)[0]  # [B, O]
        
        # Object-to-action contribution: which actions are likely given these objects?
        action_prior = torch.matmul(obj_global, cooc.T)  # [B, A]
        
        # Broadcast to spatial dimensions
        action_prior = action_prior.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, A, H, W]
        
        # Add scaled prior to logits
        enhanced_logits = action_logits + self.cooc_scale * action_prior
        
        return enhanced_logits


class YOWOMultiTask(nn.Module):
    """
    YOWO with Multi-Task Heads for Action Genome + Charades.
    
    Three prediction heads in a cascaded architecture:
    - obj_preds: Object identity (36 classes, Softmax)
    - rel_preds: Relations (26 classes, Sigmoid) - receives object context
    - act_preds: Actions (157 classes, Sigmoid, Person-only) - receives object+relation+location context
    
    Key features:
    - Location-aware global scene context for multi-person awareness
    - Multi-scale temporal pooling in X3D backbone
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
        
        # ImageNet normalization for 3D branch (X3D) - Kinetics standard
        self.register_buffer('pixel_mean', torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1, 1))

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
        
        # ============ ACTION-OBJECT CO-OCCURRENCE ============
        # Learns which actions commonly co-occur with which objects
        # E.g., "typing" -> "laptop", "eating" -> "food", "sitting" -> "chair"
        self.action_object_cooc = ActionObjectCooccurrence(
            num_actions=self.num_actions,
            num_objects=self.num_objects,
            hidden_dim=64,
            temperature=0.1
        )
        
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
        
        # 3D backbone (with ImageNet normalization)
        video_clips_3d = (video_clips - self.pixel_mean) / self.pixel_std
        feat_3d = self.backbone_3d(video_clips_3d)
        
        # Extract key frame (last frame of clip) for 2D backbone
        key_frame = video_clips[:, :, -1, :, :]
        
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
            
            # Step 4: Apply action-object co-occurrence priors
            obj_probs = F.softmax(obj_pred, dim=1)
            act_pred = self.action_object_cooc(act_pred, obj_probs)
            
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
        
        # 3D backbone (with ImageNet normalization)
        video_clips_3d = (video_clips - self.pixel_mean) / self.pixel_std
        feat_3d = self.backbone_3d(video_clips_3d)
        
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
            
            # Step 4: Apply action-object co-occurrence priors
            obj_probs = F.softmax(obj_pred, dim=1)
            act_pred = self.action_object_cooc(act_pred, obj_probs)
            
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


