"""
train.py
========
Multi-seed training engine for the full model comparison
(PilotNet, VanillaResNet, CBAMResNet, AdditiveAttnAblation, GAB-Net).

FIXES APPLIED vs. the original codebase:
- Uses Config.MODEL_KEYS / Config.checkpoint_path() for ALL checkpoint
  I/O, so evaluate.py / visualize.py can never fail to find a checkpoint
  due to a naming mismatch again.
- Uses the persisted train/val split (dataset.get_dataloaders), so the
  held-out validation set is identical across every model and every seed.
- The redundant "call get_dataloaders() once before the seed loop, then
  again inside it" pattern is removed -- the split is fixed, so it is
  loaded exactly once per seed (for the shuffling generator) and reused
  for that seed's full model sweep.
- TV loss is now logged every epoch (previously computed but discarded).
- Regularization is applied strictly according to Config.REGULARIZED_MODEL_KEYS,
  not "any model with a non-None attn_map".
- After all seeds finish, utils.calculate_statistical_significance is
  actually invoked (it existed in the original utils.py but was never
  called anywhere) to produce a mean +/- std / 95% CI summary table.
"""

import os
import pandas as pd

from config import Config
from utils import ResearchLogger, save_checkpoint, count_parameters, summarize_multiseed_results
from dataset import get_dataloaders
from models import MODEL_REGISTRY, build_model
from losses import compute_loss


def train_single_model(model_key, model, train_loader, val_loader, seed):
    model_name = Config.MODEL_KEYS[model_key]
    print(f"\n=== Training: {model_name} (seed={seed}) ===")
    model = model.to(Config.DEVICE)
    count_parameters(model, model_name)

    apply_regularization = model_key in Config.REGULARIZED_MODEL_KEYS

    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    logger = ResearchLogger(model_name, seed, Config.LOG_DIR)

    best_val_mse = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_mse_sum, train_l1_sum, train_tv_sum = 0.0, 0.0, 0.0

        for images, angles in train_loader:
            images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)

            optimizer.zero_grad()
            preds, attn_map = model(images)
            loss, mse, l1, tv = compute_loss(preds, angles, attn_map, epoch, apply_regularization)
            loss.backward()
            optimizer.step()

            train_mse_sum += mse
            train_l1_sum += l1
            train_tv_sum += tv

        n_train_batches = len(train_loader)
        avg_train_mse = train_mse_sum / n_train_batches
        avg_train_l1 = train_l1_sum / n_train_batches
        avg_train_tv = train_tv_sum / n_train_batches

        # ---- Validate ----
        model.eval()
        val_mse_sum = 0.0
        import torch
        with torch.no_grad():
            for images, angles in val_loader:
                images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
                preds, attn_map = model(images)
                _, mse, _, _ = compute_loss(preds, angles, attn_map, epoch, apply_regularization)
                val_mse_sum += mse

        avg_val_mse = val_mse_sum / len(val_loader)
        scheduler.step(avg_val_mse)

        logger.log_epoch(epoch, avg_train_mse, avg_val_mse, avg_train_l1, avg_train_tv)
        print(
            f"Epoch {epoch:02d}/{Config.EPOCHS} | "
            f"Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f} | "
            f"L1: {avg_train_l1:.4f} | TV: {avg_train_tv:.4f}"
        )

        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            save_checkpoint(model, model_key, seed, Config, is_best=True)

    save_checkpoint(model, model_key, seed, Config, is_best=False)
    print(f"[DONE] {model_name} (seed={seed}) | Best Val MSE: {best_val_mse:.4f}")
    return best_val_mse


def resolve_dataset_paths():
    dataset_root = os.path.dirname(Config.DEFAULT_DRIVING_LOG)
    csv_path = Config.DEFAULT_DRIVING_LOG
    img_dir = Config.resolve_img_dir(dataset_root)
    return csv_path, img_dir


def run_full_experiment(model_keys=None):
    """
    Trains every model in `model_keys` (default: ALL registered models)
    across every seed in Config.SEEDS, then writes a per-run CSV and a
    multi-seed statistical summary CSV.
    """
    model_keys = model_keys or list(MODEL_REGISTRY.keys())
    csv_path, img_dir = resolve_dataset_paths()

    print(f"[DATA] driving_log.csv -> {csv_path}")
    print(f"[DATA] image directory -> {img_dir}")

    results_summary = []

    for seed in Config.SEEDS:
        Config.set_global_seed(seed)
        train_loader, val_loader = get_dataloaders(csv_path, img_dir, seed=seed)

        for model_key in model_keys:
            net = build_model(model_key)
            best_mse = train_single_model(model_key, net, train_loader, val_loader, seed)
            results_summary.append({
                "Model": Config.MODEL_KEYS[model_key],
                "ModelKey": model_key,
                "Seed": seed,
                "Best_Val_MSE": best_mse,
            })

    results_df = pd.DataFrame(results_summary)
    per_run_path = os.path.join(Config.LOG_DIR, "research_per_run_results.csv")
    results_df.to_csv(per_run_path, index=False)
    print(f"\n[SAVED] Per-run results -> {per_run_path}")

    summary_df = summarize_multiseed_results(results_df, group_col="Model", value_col="Best_Val_MSE")
    summary_path = os.path.join(Config.LOG_DIR, "research_statistical_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Multi-seed statistical summary -> {summary_path}\n")
    print(summary_df.to_string(index=False))

    return results_df, summary_df


if __name__ == "__main__":
    run_full_experiment()
