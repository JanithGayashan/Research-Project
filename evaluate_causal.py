"""
evaluate_causal.py
==================
Pillar 1 of the Evaluation Framework: Causal Confusion & Domain Shift.

This script runs two quantitative tests:
1. Targeted Causal Intervention: Blurs the non-road background and measures steering variance.
   (A model with zero information leakage will have near-zero variance).
2. Out-of-Distribution (OoD) Shift: Measures Δ MSE on a completely unseen track.

Execute this script after train.py has successfully saved the .pth weights.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from config import Config
from models import build_model
from dataset import load_eval_dataset

# ==================================================================
# CONFIGURATION FOR EVALUATION
# ==================================================================
# If you have a second dataset (e.g., Jungle Track or Night Driving), put the path here.
# If None, the script will gracefully skip the OoD test and only run the Intervention test.
OOD_CSV_PATH = None 

# How many validation images to use for the blur intervention test (saves time).
N_INTERVENTION_SAMPLES = 200 
# ==================================================================


def apply_background_intervention(img_tensor, blur_kernel=(51, 51)):
    """
    Takes a normalized PyTorch image tensor, un-normalizes it, applies a heavy 
    Gaussian blur to everything EXCEPT the central driving lane (the heuristic prior),
    and returns the re-normalized PyTorch tensor.
    """
    # 1. Un-normalize back to visual RGB space
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    img_np = img_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img_unnorm = (img_np * std) + mean
    img_unnorm = np.clip(img_unnorm, 0.0, 1.0)
    
    # 2. Apply heavy intervention (Blur) to destroy background features
    img_blurred = cv2.GaussianBlur(img_unnorm, blur_kernel, 0)
    
    # 3. Create Heuristic Road Prior Mask (A trapezoid representing the ego-lane)
    H, W = img_unnorm.shape[:2]
    mask = np.zeros((H, W, 1), dtype=np.float32)
    
    # Typical lane polygon: bottom left, bottom right, middle right, middle left
    pts = np.array([
        [int(W * 0.10), H],                 # Bottom Left
        [int(W * 0.90), H],                 # Bottom Right
        [int(W * 0.55), int(H * 0.50)],     # Horizon Right
        [int(W * 0.45), int(H * 0.50)]      # Horizon Left
    ], np.int32)
    
    cv2.fillPoly(mask, [pts], (1.0,))
    
    # 4. Composite: Sharp road + Blurred background
    img_intervened = (img_unnorm * mask) + (img_blurred * (1.0 - mask))
    
    # 5. Re-normalize for the neural network
    img_intervened = (img_intervened - mean) / std
    intervened_tensor = torch.tensor(img_intervened).permute(2, 0, 1).unsqueeze(0).float()
    
    return intervened_tensor.to(Config.DEVICE)


def test_targeted_intervention(model, dataloader):
    """
    Evaluates the model's steering predictions before and after the background is blurred.
    Returns the Mean Absolute Error (Variance) caused by the background change.
    """
    model.eval()
    variance_sum = 0.0
    valid_samples = 0
    
    with torch.no_grad():
        for i, (images, angles) in enumerate(dataloader):
            images = images.to(Config.DEVICE)
            
            for j in range(images.size(0)):
                single_img = images[j].unsqueeze(0)
                
                # Predict on original clean image
                pred_orig, _ = model(single_img)
                
                # Predict on intervened (background blurred) image
                single_intervened = apply_background_intervention(single_img)
                pred_blur, _ = model(single_intervened)
                
                # Measure how much the steering changed just because the sky/trees blurred
                diff = abs(pred_orig.item() - pred_blur.item())
                variance_sum += diff
                valid_samples += 1
                
    mean_variance = variance_sum / valid_samples if valid_samples > 0 else 0.0
    return mean_variance


def test_ood_shift(model, id_loader, ood_loader):
    """
    Measures the MSE on In-Distribution (ID) vs Out-of-Distribution (OoD) data.
    """
    model.eval()
    mse_loss = torch.nn.MSELoss()
    
    def get_mse(loader):
        total_loss = 0.0
        with torch.no_grad():
            for images, angles in loader:
                images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
                preds, _ = model(images)
                total_loss += mse_loss(preds, angles).item() * images.size(0)
        return total_loss / len(loader.dataset)
        
    id_mse = get_mse(id_loader)
    ood_mse = get_mse(ood_loader) if ood_loader else None
    
    delta_mse = (ood_mse - id_mse) if ood_mse is not None else None
    return id_mse, ood_mse, delta_mse


def run_causal_evaluation():
    print(f"\\n{'='*50}")
    print("🚀 STARTING PILLAR 1: CAUSAL CONFUSION EVALUATION")
    print(f"{'='*50}\\n")
    
    # Only evaluate the models we actually trained to save time
    eval_models = ["resnet18", "soft_attn", "gabnet"]
    
    # 1. Load the Standard Validation Dataset (In-Distribution)
    # 1. Load the Standard Validation Dataset (In-Distribution)
    id_img_dir = os.path.join(os.path.dirname(Config.DEFAULT_DRIVING_LOG), "IMG")
    _, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, id_img_dir)
    
    # Create a small subset for the Intervention test so it runs quickly
    np.random.seed(Config.VAL_SPLIT_SEED)
    subset_indices = np.random.choice(len(val_dataset), min(N_INTERVENTION_SAMPLES, len(val_dataset)), replace=False)
    intervention_dataset = Subset(val_dataset, subset_indices)
    intervention_loader = DataLoader(intervention_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    id_full_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 2. Load the OoD Dataset (if provided)
    ood_loader = None
    if OOD_CSV_PATH and os.path.exists(OOD_CSV_PATH):
        print(f"🌍 Found Out-of-Distribution Dataset at {OOD_CSV_PATH}")
        ood_img_dir = os.path.join(os.path.dirname(OOD_CSV_PATH), "IMG")
        _, ood_dataset = load_eval_dataset(OOD_CSV_PATH, ood_img_dir)
        ood_loader = DataLoader(ood_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    else:
        print("⚠️ No OoD dataset provided (or path invalid). Skipping Method 1 (Domain Shift).")
        
    results = []
    
    # 3. Execution Loop
    for model_key in eval_models:
        model_name = Config.MODEL_KEYS[model_key]
        
        for seed in Config.SEEDS:
            ckpt_path = Config.checkpoint_path(model_key, seed, best=True)
            if not os.path.exists(ckpt_path):
                print(f"⏭️ Skipping {model_name} (Seed {seed}) - No weights found.")
                continue
                
            print(f"\\n🔍 Evaluating {model_name} (Seed: {seed})...")
            
            # Load Model
            model = build_model(model_key).to(Config.DEVICE)
            model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
            model.eval()
            
            # METHOD 1: Domain Shift
            id_mse, ood_mse, delta_mse = test_ood_shift(model, id_full_loader, ood_loader)
            
            # METHOD 2: Targeted Causal Intervention (Background Blur)
            steering_variance = test_targeted_intervention(model, intervention_loader)
            
            print(f"   => Background Blur Steering Variance: {steering_variance:.4f}")
            if delta_mse is not None:
                print(f"   => Δ MSE (Domain Shift): {delta_mse:.4f} (ID: {id_mse:.4f} -> OoD: {ood_mse:.4f})")
            
            results.append({
                "Model": model_name,
                "Seed": seed,
                "Intervention_Variance": steering_variance,
                "ID_MSE": id_mse,
                "OoD_MSE": ood_mse,
                "Delta_MSE": delta_mse
            })
            
    # 4. Save and Summarize Results
    if not results:
        print("\\n❌ No models were evaluated. Did you run train.py first?")
        return
        
    df = pd.DataFrame(results)
    save_path = os.path.join(Config.LOG_DIR, "causal_evaluation_results.csv")
    df.to_csv(save_path, index=False)
    
    print(f"\\n✅ Causal Evaluation Complete! Saved detailed logs to {save_path}")
    
    # Print Academic Summary
    print("\\n📊 THESIS SUMMARY: CAUSAL ROBUSTNESS (MEAN ACROSS SEEDS)")
    print("-" * 75)
    summary = df.groupby("Model").agg({
        "Intervention_Variance": ["mean", "std"],
        "Delta_MSE": ["mean"] if ood_loader else []
    }).reset_index()
    
    print(summary.to_string())
    print("-" * 75)
    print("* Lower 'Intervention_Variance' means the model successfully ignored background noise.")

if __name__ == "__main__":
    run_causal_evaluation()