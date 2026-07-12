"""
utils.py
========
Shared helper utilities used across training, evaluation, and
visualization scripts.

FIXES APPLIED vs. the original codebase:
- ResearchLogger now ALWAYS logs tv_loss (previously computed in train.py
  but silently dropped -- there was no way to plot a TV-loss-over-epochs
  curve for the thesis).
- resize_heatmap_to_image() is the ONE place any explanation heatmap gets
  resized to the input image resolution. evaluate.py now calls this for
  EVERY model (including Grad-CAM baselines), fixing the shape mismatch
  bug that previously crashed / silently corrupted IoU + perturbation
  results for PilotNet and VanillaResNet.
- seed_worker() is passed to every DataLoader with num_workers > 0 so
  NumPy's RNG state is not identically inherited by forked worker
  processes (previously augmentations could be duplicated across
  workers, quietly reducing effective data diversity).
- Checkpoint save/load now goes through Config.checkpoint_path(), the
  single canonical path builder, instead of building filenames ad hoc.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import cv2


# ==================================================================
# 1. TRAINING LOGGER
# ==================================================================
class ResearchLogger:
    """Logs per-epoch metrics to CSV in real time (safe against crashes)."""

    def __init__(self, model_name, seed, log_dir):
        self.model_name = model_name
        self.seed = seed
        self.log_file = os.path.join(log_dir, f"{model_name}_seed_{seed}.csv")
        self.history = []

    def log_epoch(self, epoch, train_mse, val_mse, sparsity_l1=0.0, tv_loss=0.0):
        entry = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "sparsity_l1": sparsity_l1,
            "tv_loss": tv_loss,
        }
        self.history.append(entry)
        pd.DataFrame(self.history).to_csv(self.log_file, index=False)


# ==================================================================
# 2. STATISTICAL SIGNIFICANCE (multi-seed aggregation)
# ==================================================================
def calculate_statistical_significance(metrics_list, confidence=0.95):
    """
    Mean, sample std-dev, and t-distribution confidence interval.
    t-distribution is used because n (number of seeds) is small.
    """
    metrics_list = np.asarray(metrics_list, dtype=float)
    n = len(metrics_list)
    mean = np.mean(metrics_list)

    if n < 2:
        return {
            "mean": mean, "std_dev": 0.0,
            "ci_lower": mean, "ci_upper": mean,
            "n": n,
            "formatted": f"{mean:.4f} (n={n}, CI unavailable)",
        }

    std_dev = np.std(metrics_list, ddof=1)
    std_err = stats.sem(metrics_list)
    h = std_err * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    return {
        "mean": mean,
        "std_dev": std_dev,
        "ci_lower": mean - h,
        "ci_upper": mean + h,
        "n": n,
        "formatted": f"{mean:.4f} +/- {std_dev:.4f} "
                      f"(95% CI: [{mean - h:.4f}, {mean + h:.4f}], n={n})",
    }


def summarize_multiseed_results(results_df, group_col="Model", value_col="Best_Val_MSE"):
    """
    Given a long-format results dataframe with one row per (model, seed),
    returns a dataframe with mean/std/CI per model.
    """
    rows = []
    for model_name, group in results_df.groupby(group_col):
        stats_dict = calculate_statistical_significance(group[value_col].tolist())
        rows.append({
            "Model": model_name,
            "Mean": stats_dict["mean"],
            "Std": stats_dict["std_dev"],
            "CI_Lower": stats_dict["ci_lower"],
            "CI_Upper": stats_dict["ci_upper"],
            "N": stats_dict["n"],
            "Formatted": stats_dict["formatted"],
        })
    return pd.DataFrame(rows)


# ==================================================================
# 3. MODEL CAPACITY / CHECKPOINTING
# ==================================================================
def count_parameters(model, model_name):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CAPACITY] {model_name}: {total_params:,} trainable parameters")
    return total_params


def save_checkpoint(model, model_key, seed, config, is_best=False):
    """Saves weights using the single canonical path builder in Config."""
    path = config.checkpoint_path(model_key, seed, best=is_best)
    torch.save(model.state_dict(), path)
    if is_best:
        print(f"[CKPT] New best model saved: {path}")
    return path


def load_checkpoint_into(model, model_key, seed, config, best=True, map_location=None):
    """Loads weights using the single canonical path builder in Config."""
    path = config.checkpoint_path(model_key, seed, best=best)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found for model_key='{model_key}', seed={seed}: {path}\n"
            f"Did training complete for this model/seed combination?"
        )
    map_location = map_location or config.DEVICE
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(state_dict)
    return model


# ==================================================================
# 4. HEATMAP HANDLING (the critical shape-mismatch fix)
# ==================================================================
def resize_heatmap_to_image(heatmap, image_size):
    """
    Resizes ANY 2D explanation heatmap (Grad-CAM output at conv-layer
    resolution, or an intrinsic attention map at feature-map resolution)
    to the full input image resolution.

    This function MUST be called on every heatmap before it is compared
    against an image-resolution road prior, or used to mask an
    image-resolution tensor in the perturbation test. Skipping this step
    for Grad-CAM baselines was the root cause of the shape-mismatch bug
    in the original evaluate_science.py.
    """
    heatmap = np.asarray(heatmap, dtype=np.float32)
    h, w = image_size
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    return heatmap


def normalize_heatmap(heatmap, eps=1e-8):
    """Min-max normalize a heatmap to [0, 1]."""
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + eps)
    return heatmap


# ==================================================================
# 5. DATALOADER WORKER SEEDING
# ==================================================================
def seed_worker(worker_id):
    """
    Passed as worker_init_fn to DataLoader so each forked worker process
    gets a distinct, deterministic NumPy/random seed derived from
    torch's initial seed, instead of silently inheriting identical RNG
    state across workers.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    import random as _random
    _random.seed(worker_seed)
