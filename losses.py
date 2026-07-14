import torch
import torch.nn as nn

from config import Config

def compute_loss(model, pred, target, attn_map, current_epoch, apply_regularization):
    mse_loss = nn.MSELoss()(pred, target)

    if attn_map is None or not apply_regularization:
        return mse_loss, mse_loss.item(), 0.0, 0.0, 0.0

    warmed_up = current_epoch > Config.WARMUP_EPOCHS
    current_l1_lambda = Config.LAMBDA_SPARSITY if warmed_up else 0.0
    current_tv_lambda = Config.LAMBDA_TV if warmed_up else 0.0
    current_c_lambda = getattr(Config, 'LAMBDA_CHANNEL', 0.005) if warmed_up else 0.0

    l1_spatial = torch.mean(torch.abs(attn_map))

    tv_loss = torch.mean(torch.abs(attn_map[:, :, :-1, :] - attn_map[:, :, 1:, :])) + \
              torch.mean(torch.abs(attn_map[:, :, :, :-1] - attn_map[:, :, :, 1:]))
              
    l1_channel = 0.0
    if hasattr(model, 'last_c_mask') and model.last_c_mask is not None:
        l1_channel = torch.mean(torch.abs(model.last_c_mask))

    total_loss = mse_loss + (current_l1_lambda * l1_spatial) + (current_tv_lambda * tv_loss) + (current_c_lambda * l1_channel)

    return total_loss, mse_loss.item(), l1_spatial.item(), tv_loss.item(), (l1_channel.item() if isinstance(l1_channel, torch.Tensor) else 0.0)