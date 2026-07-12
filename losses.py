"""
losses.py
=========
The dual-objective (MSE + L1 sparsity + TV continuity) loss used for
regularized models, per Chapter 6.9 / Figure 6.2 of the thesis.

FIX APPLIED vs. the original codebase:
- The thesis (Fig. 6.2, Sec 6.9) states that BOTH lambda_1 (L1) AND
  lambda_2 (TV) are held at zero during the warm-up period. The original
  train.py only gated the L1 term; the TV term was applied at full
  strength from epoch 1, contradicting the documented design. Both terms
  are now gated identically by the warm-up schedule.
- Regularization is now applied based on an explicit `apply_regularization`
  flag passed in by the caller (train.py looks this up from
  Config.REGULARIZED_MODEL_KEYS), instead of implicitly regularizing
  "any model that returns a non-None attention map" -- which would have
  silently applied sparsity pressure to CBAM as well, contradicting how
  CBAM is actually used in the literature.
"""

import torch
import torch.nn as nn

from config import Config


def compute_loss(pred, target, attn_map, current_epoch, apply_regularization):
    """
    Returns (total_loss, mse_value, l1_value, tv_value).

    apply_regularization: bool
        Whether this model_key is in Config.REGULARIZED_MODEL_KEYS.
        If False (or attn_map is None), only MSE is used.
    """
    mse_loss = nn.MSELoss()(pred, target)

    if attn_map is None or not apply_regularization:
        return mse_loss, mse_loss.item(), 0.0, 0.0

    # --- SPARSITY + CONTINUITY WARM-UP (both gated identically) ---
    warmed_up = current_epoch > Config.WARMUP_EPOCHS
    current_l1_lambda = Config.LAMBDA_SPARSITY if warmed_up else 0.0
    current_tv_lambda = Config.LAMBDA_TV if warmed_up else 0.0

    l1_loss = torch.mean(torch.abs(attn_map))

    tv_loss = torch.mean(torch.abs(attn_map[:, :, :-1, :] - attn_map[:, :, 1:, :])) + \
              torch.mean(torch.abs(attn_map[:, :, :, :-1] - attn_map[:, :, :, 1:]))

    total_loss = mse_loss + (current_l1_lambda * l1_loss) + (current_tv_lambda * tv_loss)

    return total_loss, mse_loss.item(), l1_loss.item(), tv_loss.item()
