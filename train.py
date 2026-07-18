# import os
# import pandas as pd
# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# from config import Config
# from utils import ResearchLogger, save_checkpoint, count_parameters, summarize_multiseed_results
# from dataset import get_dataloaders
# from models import MODEL_REGISTRY, build_model
# from losses import compute_loss

# def train_single_model(model_key, model, train_loader, val_loader, seed):
#     model_name = Config.MODEL_KEYS[model_key]
#     print(f"\n=== Training: {model_name} (seed={seed}) ===")
#     model = model.to(Config.DEVICE)
#     count_parameters(model, model_name)

#     apply_regularization = model_key in Config.REGULARIZED_MODEL_KEYS

#     optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
#     scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
#     logger = ResearchLogger(model_name, seed, Config.LOG_DIR)

#     best_val_mse = float("inf")

#     for epoch in range(1, Config.EPOCHS + 1):
#         model.train()
#         train_mse_sum, train_l1_s_sum, train_tv_sum, train_l1_c_sum = 0.0, 0.0, 0.0, 0.0

#         for images, angles in train_loader:
#             images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)

#             optimizer.zero_grad()
#             preds, attn_map = model(images)
            
#             loss, mse, l1_s, tv, l1_c = compute_loss(model, preds, angles, attn_map, epoch, apply_regularization)
            
#             loss.backward()
#             optimizer.step()

#             train_mse_sum += mse
#             train_l1_s_sum += l1_s
#             train_tv_sum += tv
#             train_l1_c_sum += l1_c

#         n_train_batches = len(train_loader)
#         avg_train_mse = train_mse_sum / n_train_batches
#         avg_train_l1 = train_l1_s_sum / n_train_batches
#         avg_train_tv = train_tv_sum / n_train_batches

#         model.eval()
#         val_mse_sum = 0.0
#         with torch.no_grad():
#             for images, angles in val_loader:
#                 images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
#                 preds, attn_map = model(images)
#                 _, mse, _, _, _ = compute_loss(model, preds, angles, attn_map, epoch, apply_regularization)
#                 val_mse_sum += mse

#         avg_val_mse = val_mse_sum / len(val_loader)
#         scheduler.step(avg_val_mse)

#         logger.log_epoch(epoch, avg_train_mse, avg_val_mse, avg_train_l1, avg_train_tv)
#         print(
#             f"Epoch {epoch:02d}/{Config.EPOCHS} | "
#             f"Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f} | "
#             f"L1_Spatial: {avg_train_l1:.4f} | TV: {avg_train_tv:.4f}"
#         )

#         if avg_val_mse < best_val_mse:
#             best_val_mse = avg_val_mse
#             save_checkpoint(model, model_key, seed, Config, is_best=True)

#     save_checkpoint(model, model_key, seed, Config, is_best=False)
#     print(f"[DONE] {model_name} (seed={seed}) | Best Val MSE: {best_val_mse:.4f}")
#     return best_val_mse

# def resolve_dataset_paths():
#     dataset_root = os.path.dirname(Config.DEFAULT_DRIVING_LOG)
#     csv_path = Config.DEFAULT_DRIVING_LOG
#     img_dir = Config.resolve_img_dir(dataset_root)
#     return csv_path, img_dir

# def run_full_experiment(model_keys=None):
#     # This explicitly restricts it to the 3 models you want to train
#     model_keys = model_keys or ["resnet18", "soft_attn", "gabnet"]
    
#     csv_path, img_dir = resolve_dataset_paths()

#     print(f"[DATA] driving_log.csv -> {csv_path}")
#     print(f"[DATA] image directory -> {img_dir}")

#     results_summary = []

#     for seed in Config.SEEDS:
#         Config.set_global_seed(seed)
#         train_loader, val_loader = get_dataloaders(csv_path, img_dir, seed=seed)

#         for model_key in model_keys:
#             net = build_model(model_key)
#             best_mse = train_single_model(model_key, net, train_loader, val_loader, seed)
#             results_summary.append({
#                 "Model": Config.MODEL_KEYS[model_key],
#                 "ModelKey": model_key,
#                 "Seed": seed,
#                 "Best_Val_MSE": best_mse,
#             })

#     results_df = pd.DataFrame(results_summary)
#     per_run_path = os.path.join(Config.LOG_DIR, "research_per_run_results.csv")
#     results_df.to_csv(per_run_path, index=False)
#     print(f"\n[SAVED] Per-run results -> {per_run_path}")

#     summary_df = summarize_multiseed_results(results_df, group_col="Model", value_col="Best_Val_MSE")
#     summary_path = os.path.join(Config.LOG_DIR, "research_statistical_summary.csv")
#     summary_df.to_csv(summary_path, index=False)
#     print(f"[SAVED] Multi-seed statistical summary -> {summary_path}\n")
#     print(summary_df.to_string(index=False))

#     return results_df, summary_df

# if __name__ == "__main__":
#     from config import Config
    
#     # 1. Override the seeds list to ONLY use 42 for quick testing
#     Config.SEEDS = [42]
    
#     # 2. Restrict the training to ONLY GAB-Net
#     run_full_experiment(model_keys=["gabnet"])

import os
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    logger = ResearchLogger(model_name, seed, Config.LOG_DIR)

    best_val_mse = float("inf")

    for epoch in range(1, Config.EPOCHS + 1):
        model.train()
        train_mse_sum, train_l1_s_sum, train_tv_sum, train_l1_c_sum, train_leakage_sum = 0.0, 0.0, 0.0, 0.0, 0.0

        for images, angles in train_loader:
            images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)

            optimizer.zero_grad()
            
            # --- DYNAMIC MODEL UNPACKING ---
            outputs = model(images)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                preds, attn_map, raw_features = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                preds, attn_map = outputs
                raw_features = None
            else:
                preds = outputs
                attn_map = None
                raw_features = None
            
            # --- UPDATED LOSS CALCULATION ---
            loss, mse, l1_s, tv, l1_c, leakage = compute_loss(
                model, preds, angles, attn_map, raw_features, epoch, apply_regularization
            )
            
            loss.backward()
            optimizer.step()

            train_mse_sum += mse
            train_l1_s_sum += l1_s
            train_tv_sum += tv
            train_l1_c_sum += l1_c
            train_leakage_sum += leakage

        n_train_batches = len(train_loader)
        avg_train_mse = train_mse_sum / n_train_batches
        avg_train_l1 = train_l1_s_sum / n_train_batches
        avg_train_tv = train_tv_sum / n_train_batches
        avg_train_leakage = train_leakage_sum / n_train_batches

        model.eval()
        val_mse_sum = 0.0
        with torch.no_grad():
            for images, angles in val_loader:
                images, angles = images.to(Config.DEVICE), angles.to(Config.DEVICE)
                
                # --- DYNAMIC EVAL UNPACKING ---
                outputs = model(images)
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    preds, attn_map, raw_features = outputs
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    preds, attn_map = outputs
                    raw_features = None
                else:
                    preds, attn_map, raw_features = outputs, None, None
                    
                _, mse, _, _, _, _ = compute_loss(
                    model, preds, angles, attn_map, raw_features, epoch, apply_regularization
                )
                val_mse_sum += mse

        avg_val_mse = val_mse_sum / len(val_loader)
        scheduler.step(avg_val_mse)

        # Assuming logger.log_epoch doesn't track leakage yet, you can leave it as is 
        # or update utils.py later if you want it logged to WandB/Tensorboard
        logger.log_epoch(epoch, avg_train_mse, avg_val_mse, avg_train_l1, avg_train_tv)
        
        # --- UPDATED PRINT STATEMENT ---
        print(
            f"Epoch {epoch:02d}/{Config.EPOCHS} | "
            f"Train MSE: {avg_train_mse:.4f} | Val MSE: {avg_val_mse:.4f} | "
            f"L1_Spatial: {avg_train_l1:.4f} | TV: {avg_train_tv:.4f} | Leakage: {avg_train_leakage:.4f}"
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
    # This explicitly restricts it to the 3 models you want to train
    model_keys = model_keys or ["resnet18", "soft_attn", "gabnet"]
    
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
    from config import Config
    
    # 1. Override the seeds list to ONLY use 42 for quick testing
    Config.SEEDS = [42]
    
    # 2. Restrict the training to ONLY GAB-Net
    run_full_experiment(model_keys=["gabnet"])