import torch
import torch.nn as nn
from torchvision import models # Needed for the base classes of backbones

class HybridModel(nn.Module):
    def __init__(self, backbone1, backbone2, num_classes): # Removed magnification-related arguments
        super().__init__()
        # --- Backbones ---
        self.effnet_feature_extractor = backbone1.features
        self.backbone2 = backbone2
        vit_original_head = self.backbone2.heads
        self.backbone2.heads = nn.Identity()

        # --- Dimensionalities ---
        if isinstance(backbone1.classifier, nn.Sequential) and \
           len(backbone1.classifier) > 1 and \
           isinstance(backbone1.classifier[1], nn.Linear):
            effnet_out_dim = backbone1.classifier[1].in_features
        elif isinstance(backbone1.classifier, nn.Linear):
             effnet_out_dim = backbone1.classifier.in_features
        else:
            raise AttributeError("Could not determine EfficientNet output features.")

        if hasattr(vit_original_head, 'head') and isinstance(vit_original_head.head, nn.Linear):
            vit_out_dim = vit_original_head.head.in_features
        else:
            # Default for vit_b_16 if structure is different or head was already Identity
            vit_out_dim = 768
        
        print(f"HybridModel (API version) - EffNet out: {effnet_out_dim}, ViT out: {vit_out_dim}")

        # Input to fusion is just EffNet + ViT
        fusion_input_dim = effnet_out_dim + vit_out_dim
        print(f"Fusion block input dimension: {fusion_input_dim}")

        fusion_hidden_dim = 1024 # As per your successful model's fusion block
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim), # Expects 2048 if effnet_out=1280, vit_out=768
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_hidden_dim),
            nn.Dropout(0.6),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim)
        )
        print(f"Fusion block output dimension: {fusion_hidden_dim}")

        classifier_hidden_dim = 512 # As per your successful model's classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_hidden_dim),
            nn.Linear(fusion_hidden_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(classifier_hidden_dim, num_classes)
        )
        print(f"Classifier output classes: {num_classes}")

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        print("HybridModel (API version - no magnification embedding) initialized.")

    def forward(self, x_image): # Only takes x_image as input
        effnet_features = self.effnet_feature_extractor(x_image)
        effnet_pooled = self.adaptive_pool(effnet_features)
        effnet_flat = torch.flatten(effnet_pooled, 1)

        if hasattr(self.backbone2, 'forward_features'):
            vit_output = self.backbone2.forward_features(x_image)
        else:
            x2_processed = self.backbone2._process_input(x_image)
            n = x2_processed.shape[0]
            batch_class_token = self.backbone2.class_token.expand(n, -1, -1)
            x2_tokens = torch.cat([batch_class_token, x2_processed], dim=1)
            x2_encoded = self.backbone2.encoder(x2_tokens)
            vit_output = x2_encoded

        if vit_output.ndim == 3:
            vit_cls_token = vit_output[:, 0]
        else:
            vit_cls_token = vit_output

        # No magnification embedding to concatenate
        combined_features = torch.cat([effnet_flat, vit_cls_token], dim=1)
        
        fused_features = self.fusion(combined_features)
        output_logits = self.classifier(fused_features)
        return output_logits