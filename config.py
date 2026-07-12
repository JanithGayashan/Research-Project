"""
config.py
=========
Single source of truth for the GAB-Net research project.

FIXES APPLIED vs. the original codebase:
- MODEL_KEYS is now the ONE canonical mapping from a short internal key
  (e.g. "gabnet") to the checkpoint-filename prefix (e.g. "Proposed_GAB_Net").
  train.py, evaluate.py, visualize.py and drive.py all import this same
  dict, so a checkpoint saved by train.py can never fail to be found by
  evaluate.py again.
- VAL_SPLIT_SEED is now separate from the training SEEDS. The train/val
  split is fixed ONCE (independent of which training seed is running) and
  persisted to disk, so every model/seed combination is evaluated on the
  exact same held-out data (no leakage, no drift between runs).
- REGULARIZED_MODEL_KEYS explicitly documents which models receive the
  L1 + TV sparsity pressure during training (only the models whose
  research narrative calls for it), instead of silently regularizing
  "any model that happens to return a non-None attention map".
"""

import os
import random
import numpy as np
import torch


class Config:
    # ==========================================================
    # 1. RESEARCH IDENTIFICATION
    # ==========================================================
    PROJECT_NAME = "GAB-Net_Explainable_Autonomous_Driving"
    STUDENT_ID = "215525P"

    # ==========================================================
    # 2. CANONICAL MODEL NAMING (single source of truth)
    # ==========================================================
    # key -> checkpoint filename prefix. Used EVERYWHERE a checkpoint is
    # saved or loaded, so naming can never drift between scripts again.
    MODEL_KEYS = {
        "pilotnet": "NVIDIA_PilotNet",
        "resnet18": "Vanilla_ResNet18",
        "cbam": "CBAM_ResNet18",
        "soft_attn": "Additive_SoftAttention_Ablation",
        "gabnet": "Proposed_GAB_Net",
    }

    # Models whose forward pass returns an attention map that is a
    # POST-HOC diagnostic only (no Grad-CAM needed at eval time because
    # they are intrinsic), vs. models that are pure black boxes and
    # require Grad-CAM as their explanation method.
    BLACKBOX_MODEL_KEYS = {"pilotnet", "resnet18"}
    INTRINSIC_MODEL_KEYS = {"cbam", "soft_attn", "gabnet"}

    # Only these models are trained with the L1 (sparsity) + TV
    # (continuity) penalty. GAB-Net is the proposed constrained model.
    # soft_attn is an ablation that isolates "multiplicative vs additive
    # gating" while holding regularization pressure CONSTANT, so it is
    # regularized identically to GAB-Net on purpose.
    # cbam is trained WITHOUT sparsity regularization because that is how
    # CBAM is actually used in the literature (Woo et al. [6]) -- it is
    # the faithful "Generation 2: Soft Intrinsic Attention" baseline.
    REGULARIZED_MODEL_KEYS = {"gabnet", "soft_attn"}

    # ==========================================================
    # 3. STATISTICAL RIGOR (Reproducibility)
    # ==========================================================
    # Training seeds -> report Mean +/- Std / 95% CI across these 3 runs.
    SEEDS = [42, 1337, 2026]

    # The train/val SPLIT is fixed independently of the training seed
    # above. This guarantees every model/seed is evaluated on the exact
    # same held-out frames.
    VAL_SPLIT_SEED = 2025

    # ==========================================================
    # 4. FAIRNESS BLOCK (identical across all models)
    # ==========================================================
    IMAGE_SIZE = (224, 224)   # (H, W) - standard ResNet input
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    OPTIMIZER = "Adam"
    NUM_WORKERS = 2

    # ==========================================================
    # 5. GAB-NET / ABLATION HYPERPARAMETERS
    # ==========================================================
    LAMBDA_SPARSITY = 0.01     # L1 penalty weight
    LAMBDA_TV = 0.001          # Total Variation penalty weight
    WARMUP_EPOCHS = 5          # Both L1 AND TV are held at 0 during warm-up

    # ==========================================================
    # 6. DATASET SETTINGS
    # ==========================================================
    STEERING_OFFSET = 0.20        # Left/right camera recovery offset
    TRAIN_VAL_SPLIT = 0.8          # 80% train / 20% val
    DOWNSAMPLE_ZERO_FRAC = 0.30    # Keep 30% of zero-steering frames

    # ==========================================================
    # 7. EVALUATION SETTINGS
    # ==========================================================
    PATCH_SIZE = 16
    MASKING_BUDGETS = [0.05, 0.10, 0.20]   # Top 5% / 10% / 20% perturbation
    TOPK_VIS_PERCENT = 15                  # For qualitative figures
    EVAL_SEEDS = SEEDS                     # Evaluate & aggregate over ALL seeds
    N_MANUAL_VALIDATION_SAMPLES = 30       # For IoU heuristic sanity-check

    # ==========================================================
    # 8. CLOSED-LOOP / SIMULATOR SETTINGS
    # ==========================================================
    TELEMETRY_PORT = 4567
    # Applied IDENTICALLY to every model during closed-loop testing so
    # that no single model gets an unfair, undocumented advantage.
    STEERING_GAIN = 1.0
    DEFAULT_THROTTLE = 0.15

    # ==========================================================
    # 9. SYSTEM & PATHS
    # ==========================================================
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

    DEFAULT_DRIVING_LOG = os.path.join(
        DATA_DIR, "self_driving_car_dataset_make", "driving_log.csv"
    )
    DEFAULT_IMG_DIR_CANDIDATES = ("IMG", "img")

    @staticmethod
    def resolve_img_dir(dataset_root):
        """Handles the IMG vs img folder-casing inconsistency robustly."""
        for candidate in Config.DEFAULT_IMG_DIR_CANDIDATES:
            candidate_path = os.path.join(dataset_root, candidate)
            if os.path.isdir(candidate_path):
                return candidate_path
        return dataset_root

    @staticmethod
    def checkpoint_path(model_key, seed, best=True):
        """
        Canonical checkpoint path builder. This is the ONLY place that
        should ever construct a checkpoint filename, guaranteeing that
        train.py / evaluate.py / visualize.py / drive.py can never diverge.
        """
        prefix = Config.MODEL_KEYS[model_key]
        suffix = "best" if best else "last"
        filename = f"{prefix}_seed_{seed}_{suffix}.pth"
        return os.path.join(Config.CHECKPOINT_DIR, filename)

    @staticmethod
    def set_global_seed(seed):
        """Enforces deterministic behaviour for academic reproducibility."""
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


# Create required directories on import.
for _d in (
    Config.DATA_DIR,
    Config.SPLIT_DIR,
    Config.LOG_DIR,
    Config.CHECKPOINT_DIR,
    Config.RESULTS_DIR,
    Config.MANUAL_VAL_DIR,
    Config.TELEMETRY_LOG_DIR,
):
    os.makedirs(_d, exist_ok=True)
