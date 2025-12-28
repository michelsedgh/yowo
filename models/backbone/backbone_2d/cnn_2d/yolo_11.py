import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLO11Backbone(nn.Module):
    """
    YOLO11 backbone + FPN neck that extracts multi-scale features for YOWO.
    
    Uses the FPN (Feature Pyramid Network) outputs which are richer than raw
    backbone features because they include multi-scale feature fusion.
    
    YOLO11m Architecture:
        Layers 0-10: Backbone (CSPDarknet with C3k2 blocks)
        Layers 11-22: Neck (PANet with upsampling and concatenation)
        Layer 23: Detect head (not used for feature extraction)
    
    We extract from layers [16, 19, 22] which are the FPN outputs:
        - Layer 16: P3 features (stride 8, 256 channels) - small objects
        - Layer 19: P4 features (stride 16, 512 channels) - medium objects
        - Layer 22: P5 features (stride 32, 512 channels) - large objects
    
    These are the same features that YOLO11's Detect head uses.
    """
    def __init__(self, model_name='yolo11m.pt', pretrained=True):
        super().__init__()
        # Load the full YOLO11 model
        yolo = YOLO(model_name)
        
        # Get the model layers (nn.Sequential-like)
        self.model = yolo.model.model
        
        # FPN output layers - same as what Detect head uses
        # Layer 16: C3k2 output, stride 8, 256 channels (P3)
        # Layer 19: C3k2 output, stride 16, 512 channels (P4)
        # Layer 22: C3k2 output, stride 32, 512 channels (P5)
        self.feature_indices = [16, 19, 22]
        
        # Stop before Detect head (layer 23) - no need to run it
        self.stop_layer = 23
        
        # Save indices needed for intermediate computations (Concat, etc.)
        # From the model: [4, 6, 10, 13, 16, 19, 22]
        self.save = list(yolo.model.save)
        
        # Ensure our feature indices are saved
        for idx in self.feature_indices:
            if idx not in self.save:
                self.save.append(idx)
        self.save.sort()
        
        # IMPORTANT: Ultralytics loads with requires_grad=False by default
        # We need to explicitly enable gradients for fine-tuning
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through YOLO11 backbone + FPN neck.
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            (cls_feats, reg_feats): Tuple of feature lists, each containing
                                    3 tensors at strides [8, 16, 32]
        """
        y = []  # Saved intermediate outputs
        outputs = []  # FPN features to return
        
        for i, m in enumerate(self.model):
            # Stop before Detect head
            if i >= self.stop_layer:
                break
                
            # Handle layer input (from previous layer or concatenation)
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            
            # Run the layer
            x = m(x)
            
            # Save output if needed by later layers
            y.append(x if i in self.save else None)
            
            # Capture FPN outputs
            if i in self.feature_indices:
                outputs.append(x)
        
        # Return in the same format as FreeYOLO: (cls_feats, reg_feats)
        # For YOLO11 we use the same features for both
        return outputs, outputs


def build_yolo_11(model_name='yolo11m.pt', pretrained=True):
    """
    Build YOLO11 backbone for feature extraction.
    
    Args:
        model_name: YOLO11 model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, etc.)
        pretrained: Whether to use pretrained weights (always True for YOLO)
        
    Returns:
        model: YOLO11Backbone instance
        feat_dims: List of feature dimensions [256, 512, 512] for yolo11m
    """
    model = YOLO11Backbone(model_name, pretrained)
    
    # Run a dummy forward pass to get feature dimensions
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
        cls_feats, reg_feats = model(dummy)
        feat_dims = [f.shape[1] for f in cls_feats]
    
    print(f"YOLO11 feature dimensions: {feat_dims}")
    print(f"YOLO11 feature strides: [8, 16, 32]")
        
    return model, feat_dims
