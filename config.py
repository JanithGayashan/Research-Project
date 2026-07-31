# """
# config.py
# =========
# Single source of truth for the GAB-Net research project.
# """

# import os
# import random
# import numpy as np
# import torch


# class Config:
#     PROJECT_NAME = "GAB-Net_Explainable_Autonomous_Driving"
#     STUDENT_ID = "215525P"

#     # CANONICAL MODEL NAMING
#     MODEL_KEYS = {
#         "pilotnet": "NVIDIA_PilotNet",
#         "resnet18": "Vanilla_ResNet18",
#         "cbam": "CBAM_ResNet18",
#         "soft_attn": "Additive_SoftAttention_Ablation",
#         "gabnet": "Proposed_GAB_Net",
#     }

#     BLACKBOX_MODEL_KEYS = {"pilotnet", "resnet18"}
#     INTRINSIC_MODEL_KEYS = {"cbam", "soft_attn", "gabnet"}

#     # REGULARIZATION TARGETS
#     REGULARIZED_MODEL_KEYS = {"gabnet", "soft_attn"}

#     # STATISTICAL RIGOR
#     SEEDS = [42, 1337, 2026]
#     VAL_SPLIT_SEED = 2025

#     # FAIRNESS BLOCK
#     IMAGE_SIZE = (224, 224) 
#     BATCH_SIZE = 32
#     LEARNING_RATE = 1e-4
#     EPOCHS = 30
#     OPTIMIZER = "Adam"
#     NUM_WORKERS = 2

#     # GAB-NET HYPERPARAMETERS (UPDATED)
#     LAMBDA_SPARSITY = 0.01     # Spatial L1 penalty weight
#     LAMBDA_CHANNEL = 0.005     # Channel L1 penalty weight (NEW for Dual-Gate)
#     LAMBDA_TV = 0.001          # Total Variation penalty weight
#     LAMBDA_LEAKAGE = 0.05
#     WARMUP_EPOCHS = 5          

#     # DATASET SETTINGS
#     STEERING_OFFSET = 0.20        
#     TRAIN_VAL_SPLIT = 0.8          
#     DOWNSAMPLE_ZERO_FRAC = 0.30    

#     # EVALUATION SETTINGS
#     PATCH_SIZE = 16
#     MASKING_BUDGETS = [0.05, 0.10, 0.20]   
#     TOPK_VIS_PERCENT = 15                  
#     EVAL_SEEDS = SEEDS                     
#     N_MANUAL_VALIDATION_SAMPLES = 30       

#     # CLOSED-LOOP SETTINGS
#     TELEMETRY_PORT = 4567
#     STEERING_GAIN = 1.0
#     DEFAULT_THROTTLE = 0.15

#     # SYSTEM & PATHS
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     DATA_DIR = os.path.join(BASE_DIR, "data")
#     SPLIT_DIR = os.path.join(DATA_DIR, "splits")
#     LOG_DIR = os.path.join(BASE_DIR, "logs")
#     CHECKPOINT_DIR = os.path.join(BASE_DIR, "models")
#     RESULTS_DIR = os.path.join(BASE_DIR, "results")
#     MANUAL_VAL_DIR = os.path.join(RESULTS_DIR, "manual_iou_validation")
#     TELEMETRY_LOG_DIR = os.path.join(RESULTS_DIR, "closed_loop_logs")

#     TRAIN_SPLIT_CSV = os.path.join(SPLIT_DIR, "train_split.csv")
#     VAL_SPLIT_CSV = os.path.join(SPLIT_DIR, "val_split.csv")

#     DEFAULT_DRIVING_LOG = os.path.join(DATA_DIR, "self_driving_car_dataset_make", "driving_log.csv")
#     DEFAULT_IMG_DIR_CANDIDATES = ("IMG", "img")

#     @staticmethod
#     def resolve_img_dir(dataset_root):
#         for candidate in Config.DEFAULT_IMG_DIR_CANDIDATES:
#             candidate_path = os.path.join(dataset_root, candidate)
#             if os.path.isdir(candidate_path):
#                 return candidate_path
#         return dataset_root

#     @staticmethod
#     def checkpoint_path(model_key, seed, best=True):
#         prefix = Config.MODEL_KEYS[model_key]
#         suffix = "best" if best else "last"
#         filename = f"{prefix}_seed_{seed}_{suffix}.pth"
#         return os.path.join(Config.CHECKPOINT_DIR, filename)

#     @staticmethod
#     def set_global_seed(seed):
#         random.seed(seed)
#         os.environ["PYTHONHASHSEED"] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         if torch.cuda.is_available():
#             torch.cuda.manual_seed(seed)
#             torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         print(f"[SEED] Locked to {seed} | Device: {Config.DEVICE}")

# for _d in (Config.DATA_DIR, Config.SPLIT_DIR, Config.LOG_DIR, Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.MANUAL_VAL_DIR, Config.TELEMETRY_LOG_DIR):
#     os.makedirs(_d, exist_ok=True)

"""
config.py
=========
Single source of truth for the GAB-Net research project.
"""

import os
import random
import numpy as np
import torch


class Config:
    PROJECT_NAME = "GAB-Net_Explainable_Autonomous_Driving"
    STUDENT_ID = "215525P"

    # CANONICAL MODEL NAMING
    MODEL_KEYS = {
        "pilotnet": "NVIDIA_PilotNet",
        "resnet18": "Vanilla_ResNet18",
        "cbam": "CBAM_ResNet18",
        "soft_attn": "Additive_SoftAttention_Ablation",
        "gabnet": "Proposed_GAB_Net",
    }

    BLACKBOX_MODEL_KEYS = {"pilotnet", "resnet18"}
    INTRINSIC_MODEL_KEYS = {"cbam", "soft_attn", "gabnet"}

    # REGULARIZATION TARGETS
    REGULARIZED_MODEL_KEYS = {"gabnet", "soft_attn"}

    # STATISTICAL RIGOR
    SEEDS = [42, 1337, 2026]
    VAL_SPLIT_SEED = 2025

    # FAIRNESS BLOCK
    IMAGE_SIZE = (224, 224) 
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    OPTIMIZER = "Adam"
    NUM_WORKERS = 2

    # GAB-NET HYPERPARAMETERS
    LAMBDA_SPARSITY = 0.01     # Spatial L1 penalty weight
    LAMBDA_CHANNEL = 0.005     # Channel L1 penalty weight
    LAMBDA_TV = 0.001          # Total Variation penalty weight
    LAMBDA_LEAKAGE = 0.005      # Attention-Guided Feature Leakage Regularization (AGFLR)
    WARMUP_EPOCHS = 1          

    # DATASET SETTINGS
    STEERING_OFFSET = 0.20        
    TRAIN_VAL_SPLIT = 0.8          
    DOWNSAMPLE_ZERO_FRAC = 0.30    

    # EVALUATION SETTINGS
    PATCH_SIZE = 16
    MASKING_BUDGETS = [0.05, 0.10, 0.20]   
    TOPK_VIS_PERCENT = 15                  
    EVAL_SEEDS = SEEDS                     
    N_MANUAL_VALIDATION_SAMPLES = 30       

    # CLOSED-LOOP SETTINGS
    TELEMETRY_PORT = 4567
    STEERING_GAIN = 1.0
    DEFAULT_THROTTLE = 0.15

    # SYSTEM & PATHS
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    SPLIT_DIR = os.path.join(DATA_DIR, "splits")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    CHECKPOINT_DIR = os.path.join(BASE_DIR, "models")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    MANUAL_VAL_DIR = os.path.join(RESULTS_DIR, "manual_iou_validation")
    TELEMETRY_LOG_DIR = os.path.join(RESULTS_DIR, "closed_loop_logs")

    TRAIN_SPLIT_CSV = os.path.join(SPLIT_DIR, "train_split.csv")
    VAL_SPLIT_CSV = os.path.join(SPLIT_DIR, "val_split.csv")

    DEFAULT_DRIVING_LOG = os.path.join(DATA_DIR, "self_driving_car_dataset_make", "driving_log.csv")
    DEFAULT_IMG_DIR_CANDIDATES = ("IMG", "img")

    @staticmethod
    def resolve_img_dir(dataset_root):
        for candidate in Config.DEFAULT_IMG_DIR_CANDIDATES:
            candidate_path = os.path.join(dataset_root, candidate)
            if os.path.isdir(candidate_path):
                return candidate_path
        return dataset_root

    @staticmethod
    def checkpoint_path(model_key, seed, best=True):
        prefix = Config.MODEL_KEYS[model_key]
        suffix = "best" if best else "last"
        filename = f"{prefix}_seed_{seed}_{suffix}.pth"
        return os.path.join(Config.CHECKPOINT_DIR, filename)

    @staticmethod
    def set_global_seed(seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[SEED] Locked to {seed} | Device: {Config.DEVICE}")

for _d in (Config.DATA_DIR, Config.SPLIT_DIR, Config.LOG_DIR, Config.CHECKPOINT_DIR, Config.RESULTS_DIR, Config.MANUAL_VAL_DIR, Config.TELEMETRY_LOG_DIR):
    os.makedirs(_d, exist_ok=True)