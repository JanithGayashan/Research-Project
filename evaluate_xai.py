"""
evaluate_xai.py
===============
Pillar 2 of the Evaluation Framework: Explainability & Fairness.

This script runs two quantitative tests:
1. Channel Sparsity Ratio (L0 Norm): Measures the percentage of dead channels 
   feeding into the final steering decision.
2. Deletion AUC: Iteratively deletes the pixels the model claims are important
   and measures the degradation in steering accuracy.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Config
from models import build_model
from dataset import load_eval_dataset

# ==================================================================
# CONFIGURATION FOR XAI
# ==================================================================
# Limit samples because Deletion AUC requires dozens of forward passes per image
N_XAI_SAMPLES = 50  
DELETION_STEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPARSITY_THRESHOLD = 1e-4  # Anything below this is considered a "dead" channel
# ==================================================================

# --- Helper: Simple Grad-CAM for the ResNet Baseline ---
class HookBypassGradCAM:
    """Dynamically finds the last Conv2d layer to generate Grad-CAM for black-box models."""
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Find the last Conv2d layer dynamically
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                
        if last_conv is not None:
            last_conv.register_forward_hook(self.save_fmaps)
            last_conv.register_full_backward_hook(self.save_grads)

    def save_fmaps(self, module, input, output):
        self.feature_maps = output

    def save_grads(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, x):
        self.model.zero_grad()
        preds, _ = self.model(x)
        preds.backward(retain_graph=True)
        
        if self.feature_maps is None or self.gradients is None:
            return torch.ones((x.size(2), x.size(3))).to(x.device) # Fallback blank mask

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        import torch.nn.functional as F
        cam_resized = F.interpolate(cam, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return cam_resized.squeeze(0).squeeze(0)

# --------------------------------------------------------

def calculate_sparsity(model, dataloader):
    """
    Hooks into the input of the final fully-connected layer to count
    how many feature channels have been permanently gated to zero.
    """
    model.eval()
    activations = []
    
    # Hook into the final Linear layer
    def hook_fn(module, inp, out):
        activations.append(inp[0].detach().cpu().numpy())
        
    # Find the final fc layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(hook_fn)
            
    with torch.no_grad():
        for i, (images, _) in enumerate(dataloader):
            images = images.to(Config.DEVICE)
            model(images)
            if i > 5: break # Only need a few batches to measure channel activity
            
    handle.remove()
    
    # Average the activations across the batches
    act_matrix = np.concatenate(activations, axis=0)
    mean_activations = np.mean(np.abs(act_matrix), axis=0)
    
    dead_channels = np.sum(mean_activations < SPARSITY_THRESHOLD)
    total_channels = len(mean_activations)
    sparsity_ratio = dead_channels / total_channels
    
    return sparsity_ratio, dead_channels, total_channels


def calculate_deletion_auc(model, dataloader, is_blackbox=False):
    """
    Implements the RISE Deletion metric.
    Deletes the top X% of pixels identified by the mask and measures MSE degradation.
    """
    model.eval()
    if is_blackbox:
        grad_cam = HookBypassGradCAM(model)
        
    mse_per_step = {step: [] for step in DELETION_STEPS}
    
    mean_color = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(Config.DEVICE)
    
    for images, angles in tqdm(dataloader, desc="Calculating Deletion AUC"):
        images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
        
        for j in range(images.size(0)):
            img = images[j].unsqueeze(0)
            target = angles[j]
            
            # 1. Get the Explanation Mask
            if is_blackbox:
                # Require gradients for Grad-CAM
                img.requires_grad = True
                mask = grad_cam.generate(img)
                img.requires_grad = False
            else:
                with torch.no_grad():
                    _, mask = model(img)
                mask = mask.squeeze()
                
            mask_np = mask.cpu().detach().numpy()
            
            # Resize mask to exactly match image dimensions
            import cv2
            mask_resized = cv2.resize(mask_np, (img.size(3), img.size(2)))
            
            # 2. Sort pixels by importance
            flattened_mask = mask_resized.flatten()
            sorted_indices = np.argsort(flattened_mask)[::-1] # Highest to lowest
            
            total_pixels = len(flattened_mask)
            
            # 3. Iteratively Delete and Measure
            for step in DELETION_STEPS:
                num_to_delete = int(step * total_pixels)
                
                # Create a copy of the image to ruin
                ruined_img = img.clone().detach()
                
                if num_to_delete > 0:
                    pixels_to_delete = sorted_indices[:num_to_delete]
                    y_coords = pixels_to_delete // img.size(3)
                    x_coords = pixels_to_delete % img.size(3)
                    
                    ruined_img[0, :, y_coords, x_coords] = mean_color[0, :, 0, 0]
                
                with torch.no_grad():
                    new_pred, _ = model(ruined_img)
                    
                error = (new_pred.item() - target.item()) ** 2
                mse_per_step[step].append(error)
                
    # Average the errors
    mean_mse_per_step = {step: np.mean(errors) for step, errors in mse_per_step.items()}
    
    # Calculate Area Under Curve (AUC) using Trapezoidal rule
    y_values = list(mean_mse_per_step.values())
    auc = np.trapz(y_values, DELETION_STEPS)
    
    return mean_mse_per_step, auc


def run_xai_evaluation():
    print(f"\\n{'='*50}")
    print("🚀 STARTING PILLAR 2: XAI & FAIRNESS EVALUATION")
    print(f"{'='*50}\\n")
    
    eval_models = ["resnet18", "soft_attn", "gabnet"]
    
    # Load validation data
    id_img_dir = os.path.join(os.path.dirname(Config.DEFAULT_DRIVING_LOG), "IMG")
    _, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, id_img_dir)
    
    np.random.seed(Config.VAL_SPLIT_SEED)
    subset_indices = np.random.choice(len(val_dataset), min(N_XAI_SAMPLES, len(val_dataset)), replace=False)
    xai_dataset = Subset(val_dataset, subset_indices)
    xai_loader = DataLoader(xai_dataset, batch_size=1, shuffle=False)
    
    sparsity_results = []
    deletion_curves = []
    
    for model_key in eval_models:
        model_name = Config.MODEL_KEYS[model_key]
        is_blackbox = (model_key == "resnet18")
        
        # Average results across all seeds for maximum statistical rigor
        seed_aucs = []
        seed_sparsities = []
        
        for seed in Config.SEEDS:
            ckpt_path = Config.checkpoint_path(model_key, seed, best=True)
            if not os.path.exists(ckpt_path):
                continue
                
            print(f"\\n🔍 Analyzing {model_name} (Seed: {seed})...")
            
            model = build_model(model_key).to(Config.DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
            
            # METHOD 3: Sparsity
            sparsity_ratio, dead, total = calculate_sparsity(model, xai_loader)
            seed_sparsities.append(sparsity_ratio)
            print(f"   => Channel Sparsity: {sparsity_ratio:.1%} ({dead}/{total} dead channels)")
            
            # METHOD 4: Deletion AUC
            curve, auc = calculate_deletion_auc(model, xai_loader, is_blackbox)
            seed_aucs.append(auc)
            print(f"   => Deletion AUC: {auc:.4f}")
            
            # Log the curve for plotting later
            curve["Model"] = model_name
            curve["Seed"] = seed
            curve["AUC"] = auc
            deletion_curves.append(curve)
            
        sparsity_results.append({
            "Model": model_name,
            "Mean_Sparsity": np.mean(seed_sparsities),
            "Mean_AUC": np.mean(seed_aucs)
        })

    # Save Results
    df_sparsity = pd.DataFrame(sparsity_results)
    df_curves = pd.DataFrame(deletion_curves)
    
    sparsity_path = os.path.join(Config.LOG_DIR, "xai_summary_results.csv")
    curves_path = os.path.join(Config.LOG_DIR, "xai_deletion_curves.csv")
    
    df_sparsity.to_csv(sparsity_path, index=False)
    df_curves.to_csv(curves_path, index=False)
    
    print(f"\\n✅ XAI Evaluation Complete! Logs saved to {Config.LOG_DIR}")
    
    print("\\n📊 THESIS SUMMARY: XAI FAIRNESS")
    print("-" * 65)
    print(df_sparsity.to_string(index=False))
    print("-" * 65)
    print("* Higher Sparsity means a tighter Information Bottleneck.")
    print("* Higher AUC means the Explanation Mask is more faithful to the actual logic.")

if __name__ == "__main__":
    run_xai_evaluation()