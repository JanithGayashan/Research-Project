"""
evaluate.py
===========
Quantitative Phase 1 (visual/localization faithfulness proxy) and
Phase 2 (driving accuracy + sparsity) scientific evaluation.

FIXES APPLIED vs. the original evaluate_science.py:
1. Grad-CAM heatmaps for the black-box baselines (PilotNet, VanillaResNet)
   are now resized to full image resolution via
   utils.resize_heatmap_to_image() BEFORE being used for IoU or
   perturbation -- previously they were left at native conv-layer
   resolution (~21x21 or 7x7) while being compared/indexed against
   224x224 arrays, which either crashes or silently corrupts results.
2. Checkpoints are loaded via Config.checkpoint_path() / utils.load_checkpoint_into(),
   eliminating the filename-mismatch bug that caused a FileNotFoundError
   on the very first model in the original script.
3. Evaluation uses ONLY the persisted, held-out validation split
   (dataset.load_eval_dataset) -- no independent re-sampling of the raw
   CSV, so there is no possibility of evaluating on frames the model saw
   during training.
4. ALL FIVE models are evaluated (PilotNet, VanillaResNet, CBAMResNet,
   AdditiveAttnAblation, GAB-Net) -- the original script omitted the
   soft-attention ablation entirely, which is precisely the comparison
   that most directly tests the thesis's central "multiplicative vs.
   soft attention" hypothesis.
5. Results are aggregated across ALL seeds in Config.EVAL_SEEDS (mean +/-
   std), not just a single "representative" seed.
6. register_backward_hook -> register_full_backward_hook (the former is
   deprecated and can behave unexpectedly on non-leaf modules in newer
   PyTorch versions).
7. The heuristic road-prior used for IoU is clearly documented as a
   PROXY, with a pointer to dataset.export_manual_validation_sample()
   for sanity-checking it against a small hand-labeled sample before it
   is relied upon for a quantitative claim in the thesis.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader

from config import Config
from dataset import load_eval_dataset
from models import build_model, MODEL_REGISTRY
from utils import (
    load_checkpoint_into,
    resize_heatmap_to_image,
    normalize_heatmap,
    calculate_statistical_significance,
)


# ==================================================================
# 1. GRAD-CAM WRAPPER (fixed hook + always-resized output)
# ==================================================================
class GradCAM:
    """Post-hoc explanation wrapper for black-box baselines. generate()
    ALWAYS returns a heatmap already resized to Config.IMAGE_SIZE."""

    def __init__(self, model, target_layer, image_size=None):
        self.model = model
        self.target_layer = target_layer
        self.image_size = image_size or Config.IMAGE_SIZE
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor):
        self.model.zero_grad()
        output, _ = self.model(input_tensor)
        # Backprop the scalar steering prediction (adapted Grad-CAM for
        # regression: substitutes the scalar output in place of a class
        # logit -- see methodology notes).
        output.sum().backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()
        cam = normalize_heatmap(cam)

        # CRITICAL FIX: always resize to full image resolution.
        cam = resize_heatmap_to_image(cam, self.image_size)
        return cam

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()


def get_intrinsic_heatmap(attn_map_tensor, image_size=None):
    """Converts an intrinsic attention map tensor (e.g. GAB-Net's [1,1,7,7])
    into a normalized, full-resolution heatmap using the same resize path
    as the Grad-CAM baselines, so all models are compared on equal footing."""
    image_size = image_size or Config.IMAGE_SIZE
    heatmap = attn_map_tensor.squeeze().detach().cpu().numpy()
    heatmap = normalize_heatmap(heatmap)
    heatmap = resize_heatmap_to_image(heatmap, image_size)
    return heatmap


# ==================================================================
# 2. HEURISTIC ROAD PRIOR (PROXY ground truth for IoU -- see caveat)
# ==================================================================
def get_heuristic_road_prior(image_tensor):
    """
    PROXY road-region estimate: bottom-60%-of-frame mask AND Canny edges.
    This is NOT verified ground truth. Before relying on the resulting
    IoU numbers as a thesis claim, sanity-check this heuristic against a
    small hand-labeled sample -- see dataset.export_manual_validation_sample().
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    h, w = edges.shape
    mask = np.zeros_like(edges)
    mask[int(h * 0.4):, :] = 1

    road_prior = (edges > 0) & (mask > 0)
    return road_prior.astype(np.float32)


def calculate_iou(mask_a, mask_b, threshold=0.5):
    bin_a = (mask_a > threshold).astype(bool)
    bin_b = (mask_b > threshold).astype(bool)
    intersection = np.logical_and(bin_a, bin_b).sum()
    union = np.logical_or(bin_a, bin_b).sum()
    return float(intersection / (union + 1e-8))


# ==================================================================
# 3. PATCH PERTURBATION TEST
# ==================================================================
def apply_patch_perturbation(image_tensor, heatmap, budget, patch_size=None):
    """Masks the top-`budget` fraction of the image using contiguous
    patches aligned to `heatmap`, which MUST already be at full image
    resolution (enforced by callers using resize_heatmap_to_image)."""
    patch_size = patch_size or Config.PATCH_SIZE
    assert heatmap.shape[0] == image_tensor.shape[1], (
        "Heatmap/image resolution mismatch -- did you forget to call "
        "resize_heatmap_to_image()?"
    )

    threshold = np.percentile(heatmap, 100 * (1 - budget))
    mask = heatmap >= threshold

    perturbed_img = image_tensor.clone()
    h, w = heatmap.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            if np.mean(mask[i:i + patch_size, j:j + patch_size]) > 0.5:
                perturbed_img[:, i:i + patch_size, j:j + patch_size] = 0

    return perturbed_img


# ==================================================================
# 4. PER-SEED EVALUATION OF A SINGLE MODEL
# ==================================================================
def evaluate_model_for_seed(model_key, seed, loader, image_size=None):
    image_size = image_size or Config.IMAGE_SIZE
    model = build_model(model_key).to(Config.DEVICE)
    load_checkpoint_into(model, model_key, seed, Config, best=True)
    model.eval()

    is_blackbox = model_key in Config.BLACKBOX_MODEL_KEYS
    explainer = None
    if is_blackbox:
        explainer = GradCAM(model, model.get_target_layer(), image_size=image_size)

    total_iou = 0.0
    total_mse = 0.0
    total_mask_sum = 0.0
    n_mask_samples = 0
    mse_degradation = {b: 0.0 for b in Config.MASKING_BUDGETS}
    n = 0

    for img, steer in loader:
        img, steer = img.to(Config.DEVICE), steer.to(Config.DEVICE)
        n += img.size(0)

        with torch.no_grad():
            pred, attn_map = model(img)
            total_mse += nn.MSELoss(reduction="sum")(pred.view(-1), steer.view(-1)).item()

        # Only batch_size == 1 is supported below for per-sample heatmaps.
        if is_blackbox:
            heatmap = explainer.generate(img)
        else:
            heatmap = get_intrinsic_heatmap(attn_map, image_size=image_size)
            total_mask_sum += float(attn_map.mean().item())
            n_mask_samples += 1

        road_prior = get_heuristic_road_prior(img.squeeze(0))
        total_iou += calculate_iou(heatmap, road_prior)

        for budget in Config.MASKING_BUDGETS:
            perturbed = apply_patch_perturbation(img.squeeze(0), heatmap, budget)
            with torch.no_grad():
                pred_p, _ = model(perturbed.unsqueeze(0).to(Config.DEVICE))
                mse_degradation[budget] += nn.MSELoss()(pred_p.view(-1), steer.view(-1)).item()

    if explainer:
        explainer.remove_hooks()

    n_batches = len(loader)
    result = {
        "Model": Config.MODEL_KEYS[model_key],
        "ModelKey": model_key,
        "Seed": seed,
        "MSE": total_mse / n,
        "Mean_IoU": total_iou / n_batches,
        "Mean_Mask_Activation": (total_mask_sum / n_mask_samples) if n_mask_samples else float("nan"),
    }
    for budget in Config.MASKING_BUDGETS:
        result[f"MSE_Drop_{int(budget*100)}pct"] = mse_degradation[budget] / n_batches

    return result


# ==================================================================
# 5. EVALUATION EXECUTOR (all models, all seeds, aggregated)
# ==================================================================
def run_scientific_evaluation(model_keys=None, n_samples=None):
    model_keys = model_keys or list(MODEL_REGISTRY.keys())

    dataset_root = os.path.dirname(Config.DEFAULT_DRIVING_LOG)
    img_dir = Config.resolve_img_dir(dataset_root)
    val_df, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, img_dir)

    if n_samples is not None and n_samples < len(val_dataset):
        # Deterministic subsample of the HELD-OUT set (not the raw CSV).
        rng = np.random.RandomState(Config.VAL_SPLIT_SEED)
        idx = rng.choice(len(val_dataset), size=n_samples, replace=False)
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, idx.tolist())

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"[EVAL] Using {len(val_dataset)} held-out validation frames.")

    per_seed_rows = []
    for model_key in model_keys:
        for seed in Config.EVAL_SEEDS:
            print(f"[EVAL] {Config.MODEL_KEYS[model_key]} | seed={seed}")
            try:
                row = evaluate_model_for_seed(model_key, seed, loader)
                per_seed_rows.append(row)
            except FileNotFoundError as e:
                print(f"[EVAL] SKIPPED ({e})")

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_path = os.path.join(Config.LOG_DIR, "scientific_evaluation_per_seed.csv")
    per_seed_df.to_csv(per_seed_path, index=False)
    print(f"[SAVED] Per-seed metrics -> {per_seed_path}")

    # Aggregate mean +/- std / 95% CI across seeds, per model, per metric.
    metric_cols = [c for c in per_seed_df.columns if c not in ("Model", "ModelKey", "Seed")]
    summary_rows = []
    for model_name, group in per_seed_df.groupby("Model"):
        row = {"Model": model_name, "N_Seeds": len(group)}
        for col in metric_cols:
            values = group[col].dropna().tolist()
            if len(values) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_std"] = float("nan")
                continue
            stats_dict = calculate_statistical_significance(values)
            row[f"{col}_mean"] = stats_dict["mean"]
            row[f"{col}_std"] = stats_dict["std_dev"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(Config.LOG_DIR, "scientific_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Multi-seed evaluation summary -> {summary_path}\n")
    print(summary_df.to_string(index=False))

    return per_seed_df, summary_df


if __name__ == "__main__":
    # n_samples=200 for a quick run; set to None to evaluate the FULL
    # held-out validation set for the final thesis numbers.
    run_scientific_evaluation(n_samples=200)
