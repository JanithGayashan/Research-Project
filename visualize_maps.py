"""
visualize_maps.py
=================
Generates high-resolution, side-by-side heatmap overlays.
Now includes a 5th column to prove the "unfaithfulness" of Grad-CAM
by running it on GAB-Net and comparing it against GAB-Net's true intrinsic mask.
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
SEED_TO_TEST = 42
# ==================================================================

class GradCAM:
    """Dynamically hooks into the last convolutional layer to generate heatmaps."""
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
        # pred, _ = self.model(x)
        # pred.backward(retain_graph=True)
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            pred = outputs[0]  # The first item is ALWAYS the steering prediction
        else:
            pred = outputs     # Fallback for models that just return 1 item (like ResNet18)
            
        pred.backward(retain_graph=True)
        
        weights = torch.mean(self.gradient, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        return pred, cam.detach()


def overlay_heatmap(img_np, mask_tensor):
    """Resizes the tensor mask, applies a colormap, and blends it with the original image."""
    mask_np = mask_tensor.squeeze().cpu().numpy()
    
    mask_resized = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]))
    
    mask_resized = mask_resized - np.min(mask_resized)
    mask_max = np.max(mask_resized)
    if mask_max > 0:
        mask_resized = mask_resized / mask_max
        
    heatmap = np.uint8(255 * mask_resized)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
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
    print("🎨 GENERATING XAI VISUALIZATIONS (5 COLUMNS)")
    print(f"{'='*50}\n")
    
    out_dir = os.path.join(Config.LOG_DIR, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data and filter for sharp turns
    id_img_dir = os.path.join(os.path.dirname(Config.DEFAULT_DRIVING_LOG), "IMG")
    val_df, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, id_img_dir)
    
    turn_indices = [i for i, angle in enumerate(val_df['steering']) if abs(angle) > 0.15]
    # np.random.seed(Config.VAL_SPLIT_SEED)
    subset_indices = np.random.choice(turn_indices, min(NUM_IMAGES_TO_GENERATE, len(turn_indices)), replace=False)
    
    loader = DataLoader(Subset(val_dataset, subset_indices), batch_size=1, shuffle=False)
    
    # 2. Load all models
    print(f"📥 Loading trained models (Seed {SEED_TO_TEST})...")
    model_resnet = load_model("resnet18", SEED_TO_TEST)
    grad_cam_resnet = GradCAM(model_resnet)
    
    model_soft = load_model("soft_attn", SEED_TO_TEST)
    model_gab = load_model("gabnet", SEED_TO_TEST)
    
    # NEW: Hook Grad-CAM directly into GAB-Net!
    grad_cam_gab = GradCAM(model_gab)
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # 3. Generate Maps
    for i, (image, target_angle) in enumerate(loader):
        print(f"📸 Processing Image {i+1}/{NUM_IMAGES_TO_GENERATE} (Target Steering: {target_angle.item():.3f})...")
        image = image.to(Config.DEVICE)
        
        img_np = image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img_unnorm = np.clip((img_np * std) + mean, 0.0, 1.0)
        img_rgb_8bit = np.uint8(img_unnorm * 255)

        # -- Get Predictions & Masks --
        # 1. ResNet18 (Grad-CAM)
        pred_res, mask_res = grad_cam_resnet.generate(image)
        
        # 2. GAB-Net (Grad-CAM)
        pred_gab_cam, mask_gab_cam = grad_cam_gab.generate(image)
        
        # with torch.no_grad():
        #     # 3. Soft Attention (Intrinsic)
        #     pred_soft, mask_soft = model_soft(image)
        #     # 4. GAB-Net (Intrinsic)
        #     pred_gab, mask_gab_intrinsic = model_gab(image)

        with torch.no_grad():
            # 3. Soft Attention (Intrinsic) - Still returns 2 items
            pred_soft, mask_soft = model_soft(image)
            
            # 4. GAB-Net (Intrinsic) - Now returns 3 items
            gab_outputs = model_gab(image)
            pred_gab = gab_outputs[0]
            mask_gab_intrinsic = gab_outputs[1]

        # -- Generate Overlays --
        vis_res = overlay_heatmap(img_rgb_8bit, mask_res)
        vis_soft = overlay_heatmap(img_rgb_8bit, mask_soft)
        vis_gab_intrinsic = overlay_heatmap(img_rgb_8bit, mask_gab_intrinsic)
        vis_gab_cam = overlay_heatmap(img_rgb_8bit, mask_gab_cam)
        
        # -- Plotting --
        # Increased figsize to 30 to comfortably fit 5 images
        fig, axes = plt.subplots(1, 5, figsize=(30, 5))
        plt.subplots_adjust(wspace=0.05)
        
        axes[0].imshow(img_rgb_8bit)
        axes[0].set_title(f"Original Frame\nHuman: {target_angle.item():.3f}", fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(vis_res)
        axes[1].set_title(f"Vanilla ResNet (Grad-CAM)\nPred: {pred_res.item():.3f}", fontsize=14)
        axes[1].axis('off')
        
        axes[2].imshow(vis_soft)
        axes[2].set_title(f"Soft Attention (Intrinsic)\nPred: {pred_soft.item():.3f}", fontsize=14)
        axes[2].axis('off')
        
        axes[3].imshow(vis_gab_intrinsic)
        axes[3].set_title(f"GAB-Net (Intrinsic Mask)\nPred: {pred_gab.item():.3f}", fontsize=14)
        axes[3].axis('off')
        
        # NEW: The "Unfaithful" Grad-CAM approximation of GAB-Net
        axes[4].imshow(vis_gab_cam)
        axes[4].set_title(f"GAB-Net (Grad-CAM)\nPred: {pred_gab_cam.item():.3f}", fontsize=14)
        axes[4].axis('off')
        
        save_file = os.path.join(out_dir, f"heatmap_comparison_{i+1}.png")
        plt.savefig(save_file, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
    print(f"\n✅ 5-Column Visualizations saved successfully to: {out_dir}")

if __name__ == "__main__":
    run_visualizations()