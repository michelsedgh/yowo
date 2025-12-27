# YOWO Multi-Task Architecture Fix - Complete Report

**Date:** December 27, 2025

## Executive Summary

Your cross-attention mechanism had a fundamental design issue that caused **100% uniform attention** (no learning). This has been fixed by replacing the broken cross-attention with a **Global Scene Context** module that correctly propagates object and relation information to the action prediction head.

---

## The Problem

### Original Design (Broken)
```
SceneContextAttention (Cross-Attention):
  Q = projection of visual features + position encoding
  K = projection of object+relation predictions
  V = features enriched with context
  
  Attention = softmax(Q @ K^T / sqrt(d))
```

### Why It Didn't Work
1. **Q (Query)** came from visual features → knows WHERE it is, not WHAT it wants
2. **K (Key)** came from predictions → has semantic content but no feature correlation  
3. **Q @ K^T** = essentially **random dot products** because Q and K are from unrelated sources
4. **Result**: 100% uniform attention entropy - the model learned nothing from context!

### Evidence
```
Before Fix:
  Attention entropy: 100% (completely uniform)
  Feature difference: 0.007 (negligible)
  Different scenarios produced nearly identical outputs
```

---

## The Fix: Global Scene Context

### New Design
```python
GlobalSceneContext:
  1. Pool object predictions globally (max across positions)
     -> "Is there a laptop ANYWHERE?" -> confidence 0.95
  
  2. Pool relation predictions globally
     -> "Is there a 'sitting_on' ANYWHERE?" -> confidence 0.90
  
  3. Project [62-dim global context] -> [256-dim feature]
  
  4. Broadcast to ALL spatial positions
  
  5. Fuse with local features using residual connection
```

### Why This Works
- Every spatial position now "sees" the **entire scene's context**
- When predicting actions at position (3,3), the model knows:
  - There's a laptop somewhere in the scene
  - There's a "holding" relation happening
  - It can predict "typing on laptop" accordingly

### Evidence After Fix
```
After Fix:
  Empty vs Rich scene difference: 0.22 (meaningful!)
  Carrying bag vs Lying on bed: 0.18 (different contexts = different outputs)
  Global pooling verified: Same content at different positions = identical output
```

---

## Architecture Flow (Fixed)

```
Video Input [B, 3, T, H, W]
     │
     ├─────────────────────────────────────┐
     ▼                                     ▼
┌─────────────┐                 ┌─────────────────┐
│ YOLO11 2D   │                 │ X3D 3D Backbone │
│ (key frame) │                 │ (full clip)     │
└──────┬──────┘                 └────────┬────────┘
       │                                  │
       └──────────┬───────────────────────┘
                  ▼
         ┌───────────────┐
         │ Feature Fusion│
         │ (Channel Enc) │
         └───────┬───────┘
                 │
         ┌───────┴───────┐
         ▼               ▼
┌──────────────┐   ┌──────────────┐
│ Object Head  │   │ Conf/Reg Head│
│ (36 classes) │   └──────────────┘
└───────┬──────┘
        │
        │ obj_pred logits
        ▼
┌──────────────────────────────┐
│ ObjectContextModule          │◄── VERIFIED WORKING ✅
│ - Uses softmax(obj_pred)     │    (3x modification at object positions)
│ - Local context propagation  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Relation Head (26 classes)   │
└──────────────┬───────────────┘
               │
               │ rel_pred logits
               ▼
┌──────────────────────────────┐
│ GlobalSceneContext           │◄── FIXED! ✅
│ - Global max-pool obj+rel    │    (was broken cross-attention)
│ - Broadcast to all positions │
│ - Scene-level context        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Action Head (157 classes)    │◄── Now receives proper scene context!
└──────────────────────────────┘
```

---

## Files Modified

### `/home/michel/yowo/models/yowo/yowo_multitask.py`

1. **Replaced `SceneContextAttention`** with `GlobalSceneContext`
   - Old: Cross-attention with Q from features, K from predictions
   - New: Global pooling + broadcast + fusion

2. **Key implementation details:**
   ```python
   class GlobalSceneContext(nn.Module):
       def forward(self, cls_feat, obj_pred, rel_pred):
           # Normalize predictions (softmax/sigmoid)
           obj_probs = F.softmax(obj_pred, dim=1)
           rel_probs = torch.sigmoid(rel_pred)
           
           # GLOBAL POOLING: Scene-level context
           obj_global = obj_probs.max(dim=-1)[0].max(dim=-1)[0]
           rel_global = rel_probs.max(dim=-1)[0].max(dim=-1)[0]
           
           # Project and broadcast
           context_embed = self.context_proj(torch.cat([obj_global, rel_global], dim=1))
           context_broadcast = context_embed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
           
           # Fuse with residual
           combined = torch.cat([cls_feat, context_broadcast], dim=1)
           delta = self.fusion(combined)
           return self.norm(cls_feat + delta)
   ```

3. **Initialization:**
   - `context_scale = 1.0` (learnable, can increase/decrease)
   - Xavier uniform with gain=1.0 (standard, not reduced)

---

## Verification Tests Passed

| Test | Result | Details |
|------|--------|---------|
| Module builds | ✅ | 26.7M parameters |
| Forward pass | ✅ | Correct output shapes |
| Gradient flow | ✅ | All context modules receive gradients |
| Context sensitivity | ✅ | 0.22 diff for different scenes |
| Global pooling | ✅ | Same content at different positions → identical output |
| Training loop | ✅ | Loss decreases, parameters update |

---

## Training Recommendations

### 1. Checkpoint Compatibility
**IMPORTANT:** Your existing checkpoints (epoch 5, 10) have weights for the OLD `SceneContextAttention`. Loading them into the new `GlobalSceneContext` will fail due to different parameter names.

**Options:**
- A) Start fresh training with new architecture
- B) Manual weight transfer (skip context layers, they'll reinitialize)

### 2. Learning Rate
The context modules have fresh weights. Consider:
- Use the same LR schedule you had before
- The backbone weights are pretrained and stable
- Context modules will learn quickly (few parameters)

### 3. What to Monitor During Training
```
Key metrics to watch:
- loss_act: Should decrease steadily
- loss_obj: Already working, should stay stable
- loss_rel: Should be helped by object context

The context_scale parameter:
- Starts at 1.0
- If it increases → model needs MORE context
- If it decreases → model needs LESS context
- Check: model.obj_rel_cross_attn[0].context_scale
```

### 4. Expected Behavior
```
Early training (epochs 1-3):
  - Object predictions are uncertain
  - Global context is still noisy
  - Action head is learning from features mostly

Mid training (epochs 4-7):
  - Object predictions become confident
  - Global context becomes meaningful
  - Action head starts using context more

Late training (epochs 8+):
  - Context-action correlation should be strong
  - Actions like "typing" should correlate with "laptop" object
  - Actions like "sitting" should correlate with "chair" object
```

---

## Quick Verification Before Training

Run this to verify everything is ready:
```bash
cd /home/michel/yowo
conda activate yowov2
python -c "
from models.yowo.yowo_multitask import YOWOMultiTask, GlobalSceneContext
print('✅ Import successful')

# Quick sensitivity test
import torch
ctx = GlobalSceneContext(dim=256, num_objects=36, num_relations=26)
feat = torch.randn(1, 256, 7, 7)
obj = torch.zeros(1, 36, 7, 7)
obj[:, 0] = 10.0  # Strong person
rel = torch.zeros(1, 26, 7, 7)

out_with = ctx(feat.clone(), obj, rel)
out_without = ctx(feat.clone(), torch.zeros_like(obj), torch.zeros_like(rel))
diff = (out_with - out_without).abs().mean().item()

print(f'Context impact: {diff:.4f}')
print('✅ Ready to train!' if diff > 0.1 else '❌ Something wrong')
"
```

---

## Summary

| Issue | Status | Impact |
|-------|--------|--------|
| Cross-attention was uniform | ✅ FIXED | Actions now see global scene context |
| ObjectContextModule | ✅ Already Working | Relations see local object context |
| Gradient flow | ✅ Verified | All heads learn together |
| Semantic sensitivity | ✅ Verified | Different scenes → different outputs |

**Your architecture is now correctly set up to learn action-object-relation correlations!**
