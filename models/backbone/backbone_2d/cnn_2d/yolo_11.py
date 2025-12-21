import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLO11Backbone(nn.Module):
    """
    YOLO11 backbone that extracts multi-scale features for YOWO.
    
    The backbone returns features in the same format as FreeYOLO:
    (cls_feats, reg_feats) - but for YOLO11 we use the same features for both
    since YOLO11 doesn't have separate cls/reg heads at the backbone level.
    """
    def __init__(self, model_name='yolo11m.pt', pretrained=True):
        super().__init__()
        # Load the full YOLO11 model
        yolo = YOLO(model_name)
        self.model = yolo.model.model # Access the internal nn.ModuleList (Sequential-like)
        
        # We want features from P3 (stride 8), P4 (stride 16), P5 (stride 32)
        # Based on YOLO11m architecture:
        # P3 is Layer 4
        # P4 is Layer 6
        # P5 is Layer 10 (C2PSA)
        self.feature_indices = [4, 6, 10]
        
        # We must save layers needed by later layers (like Concat)
        # Ultralytics models have a 'save' attribute
        self.save = yolo.model.save
        
        # Ensure our feature indices are also saved
        for idx in self.feature_indices:
            if idx not in self.save:
                self.save.append(idx)
        self.save.sort()

    def forward(self, x):
        y = [] # outputs
        outputs = []
        for i, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                # m.f can be an int or a list of ints
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            x = m(x)  # run layer
            
            # save output if needed by later layers
            y.append(x if i in self.save else None)
            
            # capture our target features
            if i in self.feature_indices:
                outputs.append(x)
        
        # Return in the same format as FreeYOLO: (cls_feats, reg_feats)
        # For YOLO11 we use the same features for both cls and reg
        # The channel encoder in YOWO will handle the fusion with 3D features
        return outputs, outputs

def build_yolo_11(model_name='yolo11m.pt', pretrained=True):
    model = YOLO11Backbone(model_name, pretrained)
    
    # Let's do a dummy forward to be 100% sure of feat_dims
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        cls_feats, reg_feats = model(dummy)
        feat_dims = [f.shape[1] for f in cls_feats]
        
    return model, feat_dims
