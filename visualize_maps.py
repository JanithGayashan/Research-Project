"""
visualize_maps.py
=================
Generates high-resolution, side-by-side heatmap overlays to visually 
compare Grad-CAM (Vanilla ResNet) against the intrinsic spatial masks 
of Soft Attention and GAB-Net.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

from config import Config
from models import build_model
from dataset import load_eval_dataset

# ==================================================================
# CONFIGURATION
# ==================================================================
NUM_IMAGES_TO_GENERATE = 5
SEED_TO_TEST = 42 # We will use the weights from Seed 42 for the visual comparison
# ==================================================================

class GradCAM:
    """Dynamically hooks into the last convolutional layer to generate heatmaps for ResNet18."""
    def __init__(self, model):
        self.model = model
        self.feature_map = None
        self.gradient = None
        
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                
        if last_conv is not None:
            last_conv.register_forward_hook(self.save_fmap)
            last_conv.register_full_backward_hook(self.save_grad)

    def save_fmap(self, module, inp, out):
        self.feature_map = out

    def save_grad(self, module, grad_in, grad_out):
        self.gradient = grad_out[0]

    def generate(self, x):
        self.model.zero_grad()
        x.requires_grad = True
        pred, _ = self.model(x)
        pred.backward(retain_graph=True)
        
        weights = torch.mean(self.gradient, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8) # Normalize between 0 and 1
        return pred, cam.detach()


def overlay_heatmap(img_np, mask_tensor):
    """Resizes the tensor mask, applies a colormap, and blends it with the original image."""
    mask_np = mask_tensor.squeeze().cpu().numpy()
    
    # Resize mask to match image dimensions
    mask_resized = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
    
    # Normalize mask strictly between 0 and 1 for coloring
    mask_resized = mask_resized - np.min(mask_resized)
    mask_max = np.max(mask_resized)
    if mask_max > 0:
        mask_resized = mask_resized / mask_max
        
    # Convert to 8-bit heatmap using JET colormap
    heatmap = np.uint8(255 * mask_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend with original image (0.6 image, 0.4 heatmap)
    superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
    return superimposed_img


def load_model(model_key, seed):
    """Helper to quickly load a trained model."""
    model = build_model(model_key).to(Config.DEVICE)
    ckpt_path = Config.checkpoint_path(model_key, seed, best=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Weights for {model_key} not found at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()
    return model


def run_visualizations():
    print(f"\n{'='*50}")
    print("🎨 GENERATING XAI VISUALIZATIONS")
    print(f"{'='*50}\n")
    
    out_dir = os.path.join(Config.LOG_DIR, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data and filter for sharp turns
    id_img_dir = os.path.join(os.path.dirname(Config.DEFAULT_DRIVING_LOG), "IMG")
    val_df, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, id_img_dir)
    
    # Find indices where the human was steering significantly (curves)
    turn_indices = [i for i, angle in enumerate(val_df['steering']) if abs(angle) > 0.15]
    np.random.seed(Config.VAL_SPLIT_SEED)
    subset_indices = np.random.choice(turn_indices, min(NUM_IMAGES_TO_GENERATE, len(turn_indices)), replace=False)
    
    loader = DataLoader(Subset(val_dataset, subset_indices), batch_size=1, shuffle=False)
    
    # 2. Load all three models simultaneously
    print(f"📥 Loading trained models (Seed {SEED_TO_TEST})...")
    model_resnet = load_model("resnet18", SEED_TO_TEST)
    grad_cam_resnet = GradCAM(model_resnet)
    
    model_soft = load_model("soft_attn", SEED_TO_TEST)
    model_gab = load_model("gabnet", SEED_TO_TEST)
    
    # Variables for un-normalizing the image back to visual RGB
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 3. Generate Maps
    for i, (image, target_angle) in enumerate(loader):
        print(f"📸 Processing Image {i+1}/{NUM_IMAGES_TO_GENERATE} (Target Steering: {target_angle.item():.3f})...")
        image = image.to(Config.DEVICE)
        
        # Recover visual RGB image
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_unnorm = np.clip((img_np * std) + mean, 0.0, 1.0)
        img_rgb_8bit = np.uint8(img_unnorm * 255)

        # -- Get Predictions & Masks --
        # ResNet18 (Grad-CAM)
        pred_res, mask_res = grad_cam_resnet.generate(image)
        
        with torch.no_grad():
            # Soft Attention
            pred_soft, mask_soft = model_soft(image)
            # GAB-Net
            pred_gab, mask_gab = model_gab(image)

        # -- Generate Overlays --
        vis_res = overlay_heatmap(img_rgb_8bit, mask_res)
        vis_soft = overlay_heatmap(img_rgb_8bit, mask_soft)
        vis_gab = overlay_heatmap(img_rgb_8bit, mask_gab)
        
        # -- Plotting --
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        plt.subplots_adjust(wspace=0.05)
        
        # Original
        axes[0].imshow(img_rgb_8bit)
        axes[0].set_title(f"Original Frame\nHuman Steer: {target_angle.item():.3f}", fontsize=14)
        axes[0].axis('off')
        
        # ResNet18
        axes[1].imshow(vis_res)
        axes[1].set_title(f"Vanilla ResNet-18 (Grad-CAM)\nPred: {pred_res.item():.3f}", fontsize=14)
        axes[1].axis('off')
        
        # Soft Attention
        axes[2].imshow(vis_soft)
        axes[2].set_title(f"Soft Attention (Intrinsic)\nPred: {pred_soft.item():.3f}", fontsize=14)
        axes[2].axis('off')
        
        # GAB-Net
        axes[3].imshow(vis_gab)
        axes[3].set_title(f"Proposed GAB-Net (Intrinsic)\nPred: {pred_gab.item():.3f}", fontsize=14)
        axes[3].axis('off')
        
        # Save Figure
        save_file = os.path.join(out_dir, f"heatmap_comparison_{i+1}.png")
        plt.savefig(save_file, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print(f"\n✅ Visualizations saved successfully to: {out_dir}")

if __name__ == "__main__":
    run_visualizations()