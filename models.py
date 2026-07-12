"""
models.py
=========
All model architectures used in the study, plus MODEL_REGISTRY: a single
canonical mapping from model_key -> factory function, imported by
train.py, evaluate.py, and visualize.py so the set of models being
compared can never silently diverge between scripts.

FIXES APPLIED vs. the original codebase:
- The former "SoftAttentionNet" used an ADDITIVE interaction
  (raw_features + raw_features * attn_map) but was informally described
  as "CBAM" in the surrounding narrative. That is not what CBAM actually
  does (CBAM uses channel attention via pooled-MLP + spatial attention
  via concatenated avg/max pooling, applied MULTIPLICATIVELY, with no
  sparsity regularization). Two separate, honestly-labeled models are
  now provided:
    * CBAMResNet       -> a faithful implementation of Woo et al. [6],
                          used as the literal "Generation 2: Soft
                          Intrinsic Attention" literature baseline.
                          Trained WITHOUT sparsity regularization.
    * AdditiveAttnAblation -> the original additive-interaction model,
                          relabeled as an internal ablation. It IS trained
                          with the same L1+TV pressure as GAB-Net, so it
                          isolates the effect of "multiplicative hard
                          gating" vs. "additive soft enhancement" while
                          holding regularization constant (Section 5.4-A
                          of the thesis).
- Every model implements a consistent forward() signature:
      forward(x) -> (steering_pred, attention_map_or_None)
  and every model that IS a pure black box (PilotNet, VanillaResNet)
  implements get_target_layer() for Grad-CAM hooking.
"""

import torch
import torch.nn as nn
from torchvision import models

from config import Config


# ==================================================================
# 1. INDUSTRY BASELINE: NVIDIA PilotNet (pure black box)
# ==================================================================
class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
        )
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
            flattened_size = self.conv_layers(dummy).view(1, -1).size(1)

        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        features = self.conv_layers(x)
        x = self.flatten(features)
        steering = self.fc_layers(x)
        return steering.squeeze(-1), None

    def get_target_layer(self):
        """Last conv layer (pre-ReLU module index -2) for Grad-CAM."""
        return self.conv_layers[-2]


# ==================================================================
# 2. ARCHITECTURAL CONTROL: Vanilla ResNet-18 (pure black box)
# ==================================================================
class VanillaResNet(nn.Module):
    """Isolates the impact of the gating module by holding the backbone
    architecture identical to GAB-Net/CBAM/ablation, minus any attention."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # -> [B,512,1,1]
        self.fc = nn.Linear(resnet.fc.in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), None

    def get_target_layer(self):
        """Final BasicBlock of layer4, for Grad-CAM."""
        return self.backbone[7][-1]


# ==================================================================
# 3. LITERATURE BASELINE: Faithful CBAM (Woo et al. [6])
#    Generation 2: Soft Intrinsic Attention. Multiplicative, but with NO
#    sparsity/TV regularization -- this is what distinguishes it from
#    GAB-Net, not the interaction type.
# ==================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAMBlock(nn.Module):
    """Sequential channel-then-spatial attention, both MULTIPLICATIVE,
    exactly as specified in Woo et al. [6]."""

    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_attn(x)
        spatial_mask = self.spatial_attn(x)
        x = x * spatial_mask
        return x, spatial_mask  # spatial_mask exposed for visualization


class CBAMResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B,512,7,7]
        self.cbam = CBAMBlock(channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        raw_features = self.backbone(x)
        refined_features, spatial_mask = self.cbam(raw_features)
        x = self.avgpool(refined_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        # spatial_mask returned for qualitative visualization only; CBAM
        # is NOT trained with L1/TV regularization (see Config.REGULARIZED_MODEL_KEYS).
        return steering.squeeze(-1), spatial_mask


# ==================================================================
# 4. PROPOSED SOLUTION: GAB-Net (multiplicative hard-gated bottleneck)
# ==================================================================
class GABNet(nn.Module):
    """
    Gated Attention Bottleneck Network. A single 1x1 conv + Sigmoid
    produces a spatial mask M that MULTIPLICATIVELY gates the 512-channel
    feature tensor before pooling, trained jointly with MSE + L1 (sparsity)
    + TV (continuity) regularization.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B,512,7,7]

        self.attention_branch = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        raw_features = self.backbone(x)
        attn_map = self.attention_branch(raw_features)

        # THE STRICT SPATIAL INFORMATION BOTTLENECK.
        gated_features = raw_features * attn_map

        x = self.avgpool(gated_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), attn_map

    def get_target_layer(self):
        """Not used for Grad-CAM (GAB-Net is intrinsic); kept for API symmetry."""
        return self.backbone[-1]


# ==================================================================
# 5. ABLATION: Additive-interaction attention (isolates gating type)
# ==================================================================
class AdditiveAttnAblation(nn.Module):
    """
    Internal ablation ONLY. Same backbone + same single-channel Sigmoid
    attention branch as GAB-Net, same L1+TV regularization pressure, but
    the interaction is ADDITIVE (raw_features + raw_features * attn_map)
    instead of strictly multiplicative. This isolates whether the
    "hard-like" multiplicative gate -- not just the presence of sparsity
    regularization -- is responsible for GAB-Net's causal-alignment gains.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.attention_branch = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        raw_features = self.backbone(x)
        attn_map = self.attention_branch(raw_features)

        # ADDITIVE interaction: background noise is enhanced, never zeroed.
        soft_features = raw_features + (raw_features * attn_map)

        x = self.avgpool(soft_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), attn_map

    def get_target_layer(self):
        return self.backbone[-1]


# ==================================================================
# 6. MODEL REGISTRY (single source of truth for "which models exist")
# ==================================================================
MODEL_REGISTRY = {
    "pilotnet": PilotNet,
    "resnet18": VanillaResNet,
    "cbam": CBAMResNet,
    "soft_attn": AdditiveAttnAblation,
    "gabnet": GABNet,
}


def build_model(model_key):
    if model_key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model_key '{model_key}'. Valid keys: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_key]()
