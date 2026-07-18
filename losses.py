# import torch
# import torch.nn as nn

# from config import Config

# def compute_loss(model, pred, target, attn_map, current_epoch, apply_regularization):
#     mse_loss = nn.MSELoss()(pred, target)

#     if attn_map is None or not apply_regularization:
#         return mse_loss, mse_loss.item(), 0.0, 0.0, 0.0

#     warmed_up = current_epoch > Config.WARMUP_EPOCHS
#     current_l1_lambda = Config.LAMBDA_SPARSITY if warmed_up else 0.0
#     current_tv_lambda = Config.LAMBDA_TV if warmed_up else 0.0
#     current_c_lambda = getattr(Config, 'LAMBDA_CHANNEL', 0.005) if warmed_up else 0.0

#     l1_spatial = torch.mean(torch.abs(attn_map))

#     tv_loss = torch.mean(torch.abs(attn_map[:, :, :-1, :] - attn_map[:, :, 1:, :])) + \
#               torch.mean(torch.abs(attn_map[:, :, :, :-1] - attn_map[:, :, :, 1:]))
              
#     l1_channel = 0.0
#     if hasattr(model, 'last_c_mask') and model.last_c_mask is not None:
#         l1_channel = torch.mean(torch.abs(model.last_c_mask))

#     total_loss = mse_loss + (current_l1_lambda * l1_spatial) + (current_tv_lambda * tv_loss) + (current_c_lambda * l1_channel)

#     return total_loss, mse_loss.item(), l1_spatial.item(), tv_loss.item(), (l1_channel.item() if isinstance(l1_channel, torch.Tensor) else 0.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

def compute_loss(model, pred, target, attn_map, raw_features, current_epoch, apply_regularization):
    # 1. Base Regression Loss
    mse_loss = nn.MSELoss()(pred, target)

    # 2. Early Exit (if no regularization requested)
    if attn_map is None or not apply_regularization:
        # Note: Added an extra 0.0 at the end to account for the new leakage return value
        return mse_loss, mse_loss.item(), 0.0, 0.0, 0.0, 0.0

    # 3. Warmup Logic (Allows the model to learn basic steering before enforcing rules)
    warmed_up = current_epoch > Config.WARMUP_EPOCHS
    current_l1_lambda = Config.LAMBDA_SPARSITY if warmed_up else 0.0
    current_tv_lambda = Config.LAMBDA_TV if warmed_up else 0.0
    current_c_lambda = getattr(Config, 'LAMBDA_CHANNEL', 0.005) if warmed_up else 0.0
    
    # NEW: Fetch the leakage lambda from config, default to 0.05
    current_leakage_lambda = getattr(Config, 'LAMBDA_LEAKAGE', 0.05) if warmed_up else 0.0 

    # 4. Existing Mask Generation Losses (Keeps the mask clean and concise)
    l1_spatial = torch.mean(torch.abs(attn_map))

    tv_loss = torch.mean(torch.abs(attn_map[:, :, :-1, :] - attn_map[:, :, 1:, :])) + \
              torch.mean(torch.abs(attn_map[:, :, :, :-1] - attn_map[:, :, :, 1:]))
              
    l1_channel = 0.0
    if hasattr(model, 'last_c_mask') and model.last_c_mask is not None:
        l1_channel = torch.mean(torch.abs(model.last_c_mask))

    # 5. NEW: Attention-Gradient Consistency Loss (The Anti-Leakage Novelty)
    # This forces the ResNet backbone's raw features to align perfectly with your s_mask
    loss_leakage = 0.0
    if raw_features is not None:
        target_mask = attn_map.detach()
        feature_activation = torch.mean(raw_features, dim=1, keepdim=True)
        inverse_mask = 1.0 - target_mask
        loss_leakage_tensor = torch.mean(torch.abs(feature_activation * inverse_mask))
        loss_leakage = loss_leakage_tensor
    else:
        loss_leakage_tensor = torch.tensor(0.0) # Safe fallback

    # 6. Total Combined Loss
    total_loss = mse_loss + \
                 (current_l1_lambda * l1_spatial) + \
                 (current_tv_lambda * tv_loss) + \
                 (current_c_lambda * l1_channel) + \
                 (current_leakage_lambda * loss_leakage)

    return (
        total_loss, 
        mse_loss.item(), 
        l1_spatial.item(), 
        tv_loss.item(), 
        (l1_channel.item() if isinstance(l1_channel, torch.Tensor) else 0.0),
        (loss_leakage_tensor.item() if isinstance(loss_leakage_tensor, torch.Tensor) else 0.0)
    )