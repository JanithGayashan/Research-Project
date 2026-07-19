"""
evaluate_sparsity.py
====================
Calculates the L0 Activation Sparsity (Zero-Pixel Proof) for the 
final convolutional feature maps across different models.
Proves that GAB-Net suppresses background noise to zero.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from config import Config
from models import build_model
from dataset import load_eval_dataset

# ==================================================================
# CONFIGURATION
# ==================================================================
SEED_TO_TEST = 42
SPARSITY_THRESHOLD = 1e-4  # Anything below this is considered a "0"
BATCH_SIZE = 32            # Process in batches for speed
# ==================================================================

class FeatureExtractor:
    """Automatically hooks into the last Conv2d layer to extract raw feature maps."""
    def __init__(self, model):
        self.model = model
        self.feature_map = None
        self.hook_handle = None
        
        # Find the last convolutional layer
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
                
        if last_conv is not None:
            self.hook_handle = last_conv.register_forward_hook(self.save_fmap)
        else:
            print("Warning: No Conv2d layer found in the model.")

    def save_fmap(self, module, inp, out):
        self.feature_map = out
        
    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()


def calculate_activation_sparsity(feature_map, threshold):
    """Calculates the percentage of pixels suppressed below the threshold."""
    total_pixels = feature_map.numel()
    # Count pixels where the absolute value is strictly less than the threshold
    suppressed_pixels = torch.sum(torch.abs(feature_map) < threshold).item()
    return suppressed_pixels / total_pixels


def load_model(model_key, seed):
    """Helper to quickly load a trained model."""
    model = build_model(model_key).to(Config.DEVICE)
    ckpt_path = Config.checkpoint_path(model_key, seed, best=True)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Weights for {model_key} not found at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE))
    model.eval()
    return model


def run_sparsity_evaluation():
    print(f"\n{'='*60}")
    print("📊 CALCULATING ACTIVATION SPARSITY (ZERO-PIXEL PROOF)")
    print(f"{'='*60}\n")
    
    # 1. Load the Validation Dataset
    id_img_dir = os.path.join(os.path.dirname(Config.DEFAULT_DRIVING_LOG), "IMG")
    val_df, val_dataset = load_eval_dataset(Config.DEFAULT_DRIVING_LOG, id_img_dir)
    loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Models
    print(f"📥 Loading trained models (Seed {SEED_TO_TEST})...")
    models = {
        "ResNet18 (Baseline)": load_model("resnet18", SEED_TO_TEST),
        "Soft Attention": load_model("soft_attn", SEED_TO_TEST),
        "GAB-Net (Ours)": load_model("gabnet", SEED_TO_TEST)
    }
    
    # 3. Setup Hooks and Tracking Dictionaries
    extractors = {name: FeatureExtractor(model) for name, model in models.items()}
    sparsity_results = {name: [] for name in models.keys()}
    
    # 4. Evaluate the Dataset
    print(f"🚀 Evaluating {len(val_dataset)} validation images...\n")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.to(Config.DEVICE)
            
            for name, model in models.items():
                # Forward pass
                _ = model(images)
                
                # Grab the raw extracted feature map
                raw_fmap = extractors[name].feature_map
                
                # NEW: Apply ReLU to simulate the standard post-activation state
                activated_fmap = F.relu(raw_fmap)
                
                # Calculate sparsity on the activated feature map
                batch_sparsity = calculate_activation_sparsity(activated_fmap, SPARSITY_THRESHOLD)
                sparsity_results[name].append(batch_sparsity)
                
            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"   Processed batch {i + 1}/{len(loader)}...")

    # 5. Clean up hooks
    for ext in extractors.values():
        ext.remove_hook()

    # 6. Calculate and Print Final Results
    print(f"\n{'='*60}")
    print(f"{'MODEL':<25} | {'AVERAGE SPARSITY (%)':<20}")
    print(f"{'-'*60}")
    
    csv_data = []
    for name, sparsities in sparsity_results.items():
        avg_sparsity = np.mean(sparsities) * 100  # Convert to percentage
        print(f"{name:<25} | {avg_sparsity:>19.2f}%")
        csv_data.append({"Model": name, "Average Sparsity (%)": avg_sparsity})
        
    print(f"{'='*60}\n")
    
    # 7. Save to Google Drive
    df = pd.DataFrame(csv_data)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    csv_path = os.path.join(Config.LOG_DIR, "sparsity_evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"💾 Results successfully saved to Drive at: {csv_path}")
    print("✅ Evaluation Complete.")

if __name__ == "__main__":
    run_sparsity_evaluation()