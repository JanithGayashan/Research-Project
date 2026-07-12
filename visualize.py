"""
visualize.py
============
Generates the Phase 1 qualitative "Glass Box" comparison figures:
Raw Input | PilotNet + Grad-CAM (post-hoc) | CBAM (soft intrinsic) |
GAB-Net (constrained intrinsic).

FIXES APPLIED vs. the original codebase:
- `import pandas as pd` moved to the top of the file (previously scoped
  inside `if __name__ == "__main__":`, which raised NameError whenever
  generate_thesis_figures() was called from anywhere else, e.g. a
  notebook or a combined evaluation driver script).
- Checkpoints loaded via Config.checkpoint_path() / utils.load_checkpoint_into(),
  matching whatever train.py actually wrote to disk.
- Frames are no longer selected by hard-coded raw CSV row indices (which
  silently break if the dataset changes size, and aren't guaranteed to
  be held-out). Frames are instead selected programmatically from the
  PERSISTED validation set by steering-angle bin (straight / left turn /
  right turn), guaranteeing both a leakage-free source and a stable,
  reproducible selection regardless of dataset changes.
- The comparison now includes the CBAM baseline (the literal "Generation
  2: Soft Intrinsic Attention" model) in addition to PilotNet+Grad-CAM
  and GAB-Net, giving a genuine 3-baseline visual argument instead of a
  2-way comparison.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import Config
from dataset import load_eval_dataset
from models import build_model
from utils import load_checkpoint_into, resize_heatmap_to_image, normalize_heatmap
from evaluate import GradCAM, get_intrinsic_heatmap


def denormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_top_k_threshold(heatmap, k_percent=None):
    k_percent = k_percent or Config.TOPK_VIS_PERCENT
    threshold = np.percentile(heatmap, 100 - k_percent)
    return np.where(heatmap >= threshold, heatmap, 0)


def create_overlay(img, heatmap, alpha=0.4):
    heatmap_norm = np.uint8(255 * normalize_heatmap(heatmap))
    heatmap_color = cv2_apply_jet(heatmap_norm)
    blended = (img.astype(np.float32) * (1 - alpha) + heatmap_color.astype(np.float32) * alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)


def cv2_apply_jet(heatmap_uint8):
    import cv2
    color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def select_representative_frames(val_df):
    """
    Selects one representative frame each for straight / left-turn /
    right-turn steering, deterministically, from the persisted held-out
    validation dataframe. Falls back gracefully if a bin is empty.
    """
    steering = val_df["steering"].astype(float)
    bins = {
        "straight": val_df[steering.abs() < 0.05],
        "left_turn": val_df[steering > 0.15],
        "right_turn": val_df[steering < -0.15],
    }
    selected = {}
    for label, subset in bins.items():
        if len(subset) == 0:
            continue
        # Deterministic pick: the median-index row of the bin.
        chosen_row = subset.iloc[len(subset) // 2]
        selected[label] = val_df.index.get_loc(chosen_row.name)
    return selected


def generate_thesis_figures(seed=None, k_percent=None):
    seed = seed if seed is not None else Config.SEEDS[0]
    k_percent = k_percent or Config.TOPK_VIS_PERCENT
    Config.set_global_seed(seed)

    dataset_root = os.path.dirname(Config.DEFAULT_DRIVING_LOG)
    img_dir = Config.resolve_img_dir(dataset_root)
    val_df, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, img_dir)

    frame_selection = select_representative_frames(val_df)
    if not frame_selection:
        raise RuntimeError("No frames matched the straight/left/right steering bins.")

    print(f"[VIS] Selected frames: {frame_selection}")

    # --- Load models ---
    pilot_net = build_model("pilotnet").to(Config.DEVICE)
    cbam_net = build_model("cbam").to(Config.DEVICE)
    gab_net = build_model("gabnet").to(Config.DEVICE)

    load_checkpoint_into(pilot_net, "pilotnet", seed, Config)
    load_checkpoint_into(cbam_net, "cbam", seed, Config)
    load_checkpoint_into(gab_net, "gabnet", seed, Config)
    pilot_net.eval(); cbam_net.eval(); gab_net.eval()

    explainer = GradCAM(pilot_net, pilot_net.get_target_layer())

    saved_paths = []
    for label, idx in frame_selection.items():
        print(f"[VIS] Rendering frame '{label}' (val_index={idx})")
        img_tensor, steer_true = val_dataset[idx]
        input_batch = img_tensor.unsqueeze(0).to(Config.DEVICE)

        # A. Post-hoc baseline (PilotNet + Grad-CAM)
        pilot_angle, _ = pilot_net(input_batch)
        pilot_heatmap = explainer.generate(input_batch)
        pilot_heatmap = apply_top_k_threshold(pilot_heatmap, k_percent)

        # B. Soft intrinsic baseline (CBAM)
        cbam_angle, cbam_attn = cbam_net(input_batch)
        cbam_heatmap = get_intrinsic_heatmap(cbam_attn)
        cbam_heatmap = apply_top_k_threshold(cbam_heatmap, k_percent)

        # C. Proposed constrained model (GAB-Net)
        gab_angle, gab_attn = gab_net(input_batch)
        gab_heatmap = get_intrinsic_heatmap(gab_attn)
        gab_heatmap = apply_top_k_threshold(gab_heatmap, k_percent)

        raw_img = denormalize(img_tensor)
        pilot_overlay = create_overlay(raw_img, pilot_heatmap)
        cbam_overlay = create_overlay(raw_img, cbam_heatmap)
        gab_overlay = create_overlay(raw_img, gab_heatmap)

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))

        axes[0].imshow(raw_img)
        axes[0].set_title(f"Raw Input ({label})\nGT steering: {steer_true:.2f}")
        axes[0].axis("off")

        axes[1].imshow(pilot_overlay)
        axes[1].set_title(f"PilotNet + Grad-CAM (Post-Hoc)\nPred: {pilot_angle.item():.2f}")
        axes[1].axis("off")

        axes[2].imshow(cbam_overlay)
        axes[2].set_title(f"CBAM (Soft Intrinsic)\nPred: {cbam_angle.item():.2f}")
        axes[2].axis("off")

        axes[3].imshow(gab_overlay)
        axes[3].set_title(f"GAB-Net (Constrained Intrinsic)\nPred: {gab_angle.item():.2f}")
        axes[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(Config.RESULTS_DIR, f"comparison_{label}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        saved_paths.append(save_path)
        print(f"[VIS] Saved -> {save_path}")

    explainer.remove_hooks()
    return saved_paths


if __name__ == "__main__":
    generate_thesis_figures()
