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
        return self.conv_layers[-2]

# ==================================================================
# 2. ARCHITECTURAL CONTROL: Vanilla ResNet-18 (pure black box)
# ==================================================================
class VanillaResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.fc = nn.Linear(resnet.fc.in_features, 1)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), None
    def get_target_layer(self):
        return self.backbone[7][-1]

# ==================================================================
# 3. LITERATURE BASELINE: Faithful CBAM (Woo et al.)
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
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
    def forward(self, x):
        x = x * self.channel_attn(x)
        spatial_mask = self.spatial_attn(x)
        x = x * spatial_mask
        return x, spatial_mask 

class CBAMResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cbam = CBAMBlock(channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
    def forward(self, x):
        raw_features = self.backbone(x)
        refined_features, spatial_mask = self.cbam(raw_features)
        x = self.avgpool(refined_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), spatial_mask

# ==================================================================
# 4. ABLATION: Additive-interaction attention
# ==================================================================
class AdditiveAttnAblation(nn.Module):
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
        soft_features = raw_features + (raw_features * attn_map)
        x = self.avgpool(soft_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), attn_map
    def get_target_layer(self):
        return self.backbone[-1]

# ==================================================================
# 5. PROPOSED SOLUTION: GAB-Net (Dual-Gated Information Bottleneck)
# ==================================================================
import torch
import torch.nn as nn
import torchvision.models as models

# class GABNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])  

#         # --- Channel Gate ---
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.channel_mlp = nn.Sequential(
#             nn.Conv2d(512, 32, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(32, 512, kernel_size=1, bias=False)
#         )
#         self.channel_sigmoid = nn.Sigmoid()
        
#         # --- Spatial Gate (Compressed) ---
#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.spatial_sigmoid = nn.Sigmoid()

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # NEW: Dropout layer added here (p=0.5 means 50% probability)
#         self.dropout = nn.Dropout(p=0.5) 
        
#         self.fc = nn.Linear(512, 1)
        
#         self.last_c_mask = None 

#     def forward(self, x):
#         raw_features = self.backbone(x)
        
#         # --- Channel Gate ---
#         avg_out = self.channel_mlp(self.avg_pool(raw_features))
#         max_out = self.channel_mlp(self.max_pool(raw_features))
#         c_mask = self.channel_sigmoid(avg_out + max_out)
        
#         self.last_c_mask = c_mask 
#         f_c = raw_features * c_mask
        
#         # --- Spatial Gate (Compressed) ---
#         s_avg_out = torch.mean(f_c, dim=1, keepdim=True)
#         s_max_out, _ = torch.max(f_c, dim=1, keepdim=True)
#         s_concat = torch.cat([s_avg_out, s_max_out], dim=1)
        
#         s_mask = self.spatial_sigmoid(self.spatial_conv(s_concat))
#         gated_features = f_c * s_mask

#         x = self.avgpool(gated_features)
#         x = torch.flatten(x, 1)
        
#         # NEW: Apply Dropout right before the final Linear layer
#         x = self.dropout(x) 
        
#         steering = self.fc(x)
#         return steering.squeeze(-1), s_mask

#     def get_target_layer(self):
#         return self.backbone[-1]

# class GABNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])  

#         # --- Channel Gate ---
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         self.channel_mlp = nn.Sequential(
#             nn.Conv2d(512, 32, kernel_size=1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(32, 512, kernel_size=1, bias=False)
#         )
#         self.channel_sigmoid = nn.Sigmoid()
        
#         # --- Spatial Gate (Compressed) ---
#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
#         self.spatial_sigmoid = nn.Sigmoid()

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # FIX 1: Dropout completely removed to prevent spurious backup learning.
        
#         self.fc = nn.Linear(512, 1)
        
#         self.last_c_mask = None 

#     def forward(self, x):
#         raw_features = self.backbone(x)
        
#         # --- Channel Gate ---
#         avg_out = self.channel_mlp(self.avg_pool(raw_features))
#         max_out = self.channel_mlp(self.max_pool(raw_features))
#         c_mask = self.channel_sigmoid(avg_out + max_out)
        
#         self.last_c_mask = c_mask 
#         f_c = raw_features * c_mask
        
#         # --- Spatial Gate (Compressed) ---
#         s_avg_out = torch.mean(f_c, dim=1, keepdim=True)
#         s_max_out, _ = torch.max(f_c, dim=1, keepdim=True)
#         s_concat = torch.cat([s_avg_out, s_max_out], dim=1)
        
#         s_mask = self.spatial_sigmoid(self.spatial_conv(s_concat))
        
#         # The physical gating happens here
#         gated_features = f_c * s_mask

#         x = self.avgpool(gated_features)
#         x = torch.flatten(x, 1)
        
#         # FIX 1 (cont.): No Dropout applied here before the final layer.
        
#         steering = self.fc(x)
        
#         # FIX 2: We must return raw_features alongside steering and s_mask.
#         # This allows the custom loss function to penalize any features 
#         # that activate outside the s_mask.
#         return steering.squeeze(-1), s_mask, raw_features

#     def get_target_layer(self):
#         # Grad-CAM targets this layer. The custom loss will now force 
#         # this layer to align mathematically with your s_mask.
#         return self.backbone[-1]
# # ==================================================================
# # 6. MODEL REGISTRY
# # ==================================================================
# MODEL_REGISTRY = {
#     "pilotnet": PilotNet,
#     "resnet18": VanillaResNet,
#     "cbam": CBAMResNet,
#     "soft_attn": AdditiveAttnAblation,
#     "gabnet": GABNet,
# }

# def build_model(model_key):
#     if model_key not in MODEL_REGISTRY:
#         raise KeyError(f"Unknown model_key '{model_key}'. Valid keys: {list(MODEL_REGISTRY.keys())}")
#     return MODEL_REGISTRY[model_key]()




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
        return self.conv_layers[-2]

# ==================================================================
# 2. ARCHITECTURAL CONTROL: Vanilla ResNet-18 (pure black box)
# ==================================================================
class VanillaResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) 
        self.fc = nn.Linear(resnet.fc.in_features, 1)
    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), None
    def get_target_layer(self):
        return self.backbone[7][-1]

# ==================================================================
# 3. LITERATURE BASELINE: Faithful CBAM (Woo et al.)
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
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))

class CBAMBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
    def forward(self, x):
        x = x * self.channel_attn(x)
        spatial_mask = self.spatial_attn(x)
        x = x * spatial_mask
        return x, spatial_mask 

class CBAMResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cbam = CBAMBlock(channels=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)
    def forward(self, x):
        raw_features = self.backbone(x)
        refined_features, spatial_mask = self.cbam(raw_features)
        x = self.avgpool(refined_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), spatial_mask

# ==================================================================
# 4. ABLATION: Additive-interaction attention
# ==================================================================
class AdditiveAttnAblation(nn.Module):
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
        soft_features = raw_features + (raw_features * attn_map)
        x = self.avgpool(soft_features)
        x = torch.flatten(x, 1)
        steering = self.fc(x)
        return steering.squeeze(-1), attn_map
    def get_target_layer(self):
        return self.backbone[-1]

# ==================================================================
# 5. PROPOSED SOLUTION: GAB-Net (Dual-Gated Information Bottleneck)
# ==================================================================
class GABNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  

        # --- Channel Gate ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 512, kernel_size=1, bias=False)
        )
        # LOCKED: Hard Sigmoid for true mathematical zeros
        self.channel_sigmoid = nn.Hardsigmoid()
        
        # --- Spatial Gate ---
        # LOCKED: 512 input channels (No Compression), 3x3 kernel (Goldilocks)
        self.spatial_conv = nn.Conv2d(512, 1, kernel_size=3, padding=1, bias=False)
        # LOCKED: Hard Sigmoid for true mathematical zeros
        self.spatial_sigmoid = nn.Hardsigmoid()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # LOCKED: No Dropout applied.
        
        self.fc = nn.Linear(512, 1)
        
        self.last_c_mask = None 

    def forward(self, x):
        raw_features = self.backbone(x)
        
        # --- Channel Gate ---
        avg_out = self.channel_mlp(self.avg_pool(raw_features))
        max_out = self.channel_mlp(self.max_pool(raw_features))
        c_mask = self.channel_sigmoid(avg_out + max_out)
        
        self.last_c_mask = c_mask 
        f_c = raw_features * c_mask
        
        # --- Spatial Gate (Full 512 Channels) ---
        # Direct 512-channel input into the 3x3 convolution
        s_mask = self.spatial_sigmoid(self.spatial_conv(f_c))
        
        # The physical gating happens here
        gated_features = f_c * s_mask

        x = self.avgpool(gated_features)
        x = torch.flatten(x, 1)
        
        steering = self.fc(x)
        
        # Returning raw_features for AGFLR (Leakage Loss) compatibility
        return steering.squeeze(-1), s_mask, raw_features

    def get_target_layer(self):
        return self.backbone[-1]

# ==================================================================
# 6. MODEL REGISTRY
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
        raise KeyError(f"Unknown model_key '{model_key}'. Valid keys: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]()