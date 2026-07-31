"""
evaluate_thesis_stats.py
========================
Comprehensive statistical evaluation for the thesis.
Calculates Driving Metrics (MAE, RMSE), XAI Metrics (Sparsity, Deletion AUC),
and performs paired Wilcoxon statistical tests with Cohen's d effect sizes.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.stats as stats
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from models import build_model
from dataset import load_eval_dataset

# ==================================================================
# CONFIGURATION
# ==================================================================
N_TEST_SAMPLES = 200 # Increased for statistical rigor
DELETION_STEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPARSITY_THRESHOLD = 1e-2 

# --- Helper: Grad-CAM for Black-Box Models ---
class HookBypassGradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            last_conv.register_forward_hook(self.save_fmaps)
            last_conv.register_full_backward_hook(self.save_grads)

    def save_fmaps(self, module, input, output): self.feature_maps = output
    def save_grads(self, module, grad_input, grad_output): self.gradients = grad_output[0]

    def generate(self, x):
        self.model.zero_grad()
        outputs = self.model(x)
        preds = outputs[0] if isinstance(outputs, tuple) else outputs
        torch.abs(preds).backward(retain_graph=True)
        
        if self.feature_maps is None or self.gradients is None:
            return torch.ones((x.size(2), x.size(3))).to(x.device)

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        import torch.nn.functional as F
        cam_resized = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return cam_resized.squeeze(0).squeeze(0)

def cohens_d(group1, group2):
    """Calculate Cohen's d for effect size."""
    diff = group1 - group2
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

def run_statistical_evaluation():
    print(f"\n{'='*60}\n🚀 STARTING FULL THESIS STATISTICAL EVALUATION\n{'='*60}\n")
    eval_models = ["resnet18", "soft_attn", "gabnet"]
    id_img_dir = Config.resolve_img_dir(os.path.dirname(Config.DEFAULT_DRIVING_LOG))
    _, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, id_img_dir)
    
    # Filter for turns to get meaningful XAI data
    val_df = val_dataset.data
    turn_indices = [i for i, angle in enumerate(val_df['steering']) if abs(angle) > 0.05]
    np.random.seed(Config.VAL_SPLIT_SEED)
    subset_indices = np.random.choice(turn_indices, min(N_TEST_SAMPLES, len(turn_indices)), replace=False)
    test_loader = DataLoader(Subset(val_dataset, subset_indices), batch_size=1, shuffle=False)
    
    # Data structure to hold per-image results across all seeds
    model_results = {key: {"mae": [], "rmse": [], "sparsity": [], "auc": []} for key in eval_models}
    
    mean_color = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(Config.DEVICE)

    for model_key in eval_models:
        model_name = Config.MODEL_KEYS[model_key]
        is_blackbox = (model_key == "resnet18")
        
        print(f"\n🔍 Evaluating {model_name}...")
        
        for seed in Config.SEEDS:
            ckpt = Config.checkpoint_path(model_key, seed, best=False) # Loading _last.pth per previous discussion
            if not os.path.exists(ckpt): continue
            
            model = build_model(model_key).to(Config.DEVICE)
            model.load_state_dict(torch.load(ckpt, map_location=Config.DEVICE, weights_only=True))
            model.eval()
            
            grad_cam = HookBypassGradCAM(model) if is_blackbox else None
            
            # Evaluate every single image
            for images, angles in tqdm(test_loader, desc=f"Seed {seed}", leave=False):
                images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
                img = images[0].unsqueeze(0)
                target = angles[0].item()
                
                # 1. Base Prediction (Driving Metrics)
                with torch.no_grad():
                    out_orig = model(img)
                    pred_orig = out_orig[0].item() if isinstance(out_orig, tuple) else out_orig.item()
                
                error = abs(pred_orig - target)
                model_results[model_key]["mae"].append(error)
                model_results[model_key]["rmse"].append(error ** 2)
                
                # 2. Extract Mask (Sparsity)
                if is_blackbox:
                    img.requires_grad = True
                    mask = grad_cam.generate(img)
                    img.requires_grad = False
                    sparsity = 0.0
                else:
                    with torch.no_grad():
                        outputs = model(img)
                        mask = outputs[1].squeeze()
                    sparsity = torch.mean((mask < SPARSITY_THRESHOLD).float()).item()
                    
                model_results[model_key]["sparsity"].append(sparsity)
                
                # 3. Iterative Deletion (Faithfulness AUC)
                mask_np = cv2.resize(mask.cpu().detach().numpy(), (img.size(3), img.size(2)))
                sorted_indices = np.argsort(mask_np.flatten())[::-1]
                total_pixels = len(mask_np.flatten())
                
                diffs_for_image = []
                for step in DELETION_STEPS:
                    num_to_delete = int(step * total_pixels)
                    ruined_img = img.clone().detach()
                    
                    if num_to_delete > 0:
                        y, x = np.unravel_index(sorted_indices[:num_to_delete], (img.size(2), img.size(3)))
                        ruined_img[0, :, y, x] = 0 # Blank out pixels
                    
                    with torch.no_grad():
                        outs = model(ruined_img)
                        pred_del = outs[0].item() if isinstance(outs, tuple) else outs.item()
                        
                    # FRIEND'S FIX: Difference from original prediction, not ground truth
                    diffs_for_image.append(abs(pred_orig - pred_del)) 
                    
                auc = np.trapz(diffs_for_image, DELETION_STEPS)
                model_results[model_key]["auc"].append(auc)

    # ==========================================
    # STATISTICS AND FORMATTING
    # ==========================================
    print("\n\n📊 THESIS RESULTS: METRICS & STATISTICAL SIGNIFICANCE")
    print("-" * 115)
    print(f"{'Model':<30} | {'MAE (↓)':<15} | {'RMSE (↓)':<15} | {'Sparsity (↑)':<15} | {'Deletion AUC (↑)':<15}")
    print("-" * 115)
    
    # Pre-calculate GAB-Net arrays for paired tests
    gab_mae = np.array(model_results["gabnet"]["mae"])
    gab_auc = np.array(model_results["gabnet"]["auc"])
    
    for model_key in eval_models:
        name = Config.MODEL_KEYS[model_key]
        
        # Means and Std Devs
        mae_arr = np.array(model_results[model_key]["mae"])
        rmse_val = np.sqrt(np.mean(model_results[model_key]["rmse"])) # RMSE is sqrt of mean squared errors
        rmse_std = np.std(np.sqrt(np.array(model_results[model_key]["rmse"])))
        spars_arr = np.array(model_results[model_key]["sparsity"])
        auc_arr = np.array(model_results[model_key]["auc"])
        
        mae_str = f"{np.mean(mae_arr):.4f} ± {np.std(mae_arr):.4f}"
        rmse_str = f"{rmse_val:.4f} ± {rmse_std:.4f}"
        spars_str = f"{np.mean(spars_arr):.1%} ± {np.std(spars_arr):.1%}"
        auc_str = f"{np.mean(auc_arr):.4f} ± {np.std(auc_arr):.4f}"
        
        print(f"{name:<30} | {mae_str:<15} | {rmse_str:<15} | {spars_str:<15} | {auc_str:<15}")
        
    print("-" * 115)
    
    # Wilcoxon Tests against GAB-Net
    print("\n🔬 WILCOXON SIGNED-RANK TEST (Base Models vs. Proposed GAB-Net)")
    for model_key in ["resnet18", "soft_attn"]:
        name = Config.MODEL_KEYS[model_key]
        
        # Test MAE (Driving Performance)
        stat, p_mae = stats.wilcoxon(np.array(model_results[model_key]["mae"]), gab_mae)
        d_mae = cohens_d(np.array(model_results[model_key]["mae"]), gab_mae)
        
        # Test AUC (Explainability)
        stat, p_auc = stats.wilcoxon(np.array(model_results[model_key]["auc"]), gab_auc)
        d_auc = cohens_d(np.array(model_results[model_key]["auc"]), gab_auc)
        
        print(f"\n{name} vs GAB-Net:")
        print(f"  - MAE Difference: p-value = {p_mae:.2e} | Effect Size (d) = {d_mae:.2f}")
        print(f"  - AUC Difference: p-value = {p_auc:.2e} | Effect Size (d) = {d_auc:.2f}")

if __name__ == "__main__":
    run_statistical_evaluation()