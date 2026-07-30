# """
# dataset.py
# ==========
# Data loading, cleaning, splitting, and the PyTorch Dataset class.

# FIXES APPLIED vs. the original codebase:
# - The train/val split is now built ONCE (keyed on Config.VAL_SPLIT_SEED,
#   independent of the training seed) and PERSISTED to disk as
#   train_split.csv / val_split.csv. Every script (train.py, evaluate.py,
#   visualize.py) that needs data now loads the persisted split instead of
#   re-sampling the raw CSV independently. This closes the train/test
#   leakage hole where evaluate_science.py and visualize.py previously drew
#   their own samples from the full dataset with no guarantee those frames
#   were held out during training.
# - A single robust CSV loader (load_driving_log) is used everywhere,
#   instead of evaluate_science.py / visualize.py using a naive
#   pd.read_csv(..., names=[...]) that silently misinterprets a real
#   header row as a data row.
# - Left AND right camera image paths are existence-checked, not just the
#   center camera.
# - get_dataloaders() now accepts a `seed` used ONLY to seed the shuffling
#   generator (so training order differs meaningfully per seed) while the
#   underlying train/val split itself stays fixed across seeds.
# - A DataLoader worker_init_fn is wired in to avoid duplicated
#   augmentation streams across worker processes.
# - export_manual_validation_sample() is added as a lightweight tool to
#   support sanity-checking the heuristic road-prior used for IoU scoring
#   in evaluate.py (see evaluate.py docstring for details).
# """

# import os
# import cv2
# import numpy as np
# import pandas as pd
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image

# from config import Config
# from utils import seed_worker

# CSV_COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


# # ==================================================================
# # 1. ROBUST CSV LOADING (single source of truth)
# # ==================================================================
# def load_driving_log(csv_path):
#     """
#     Loads a Udacity-style driving log CSV whether or not it has a header
#     row, and normalizes it to the 7 standard columns.
#     """
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"Driving log CSV not found: {csv_path}")

#     raw = pd.read_csv(csv_path, header=None)
#     first_cell = str(raw.iloc[0, 0]).lower()

#     if ".jpg" in first_cell or "img" in first_cell or "center" not in first_cell:
#         # First row is a data row (no header present).
#         df = raw.iloc[:, :len(CSV_COLUMNS)].copy()
#         df.columns = CSV_COLUMNS
#     else:
#         # First row looks like a header; reload with header inference.
#         df = pd.read_csv(csv_path)
#         df = df.iloc[:, :len(CSV_COLUMNS)].copy()
#         df.columns = CSV_COLUMNS

#     df["steering"] = df["steering"].astype(float)
#     return df


# def _resolve_filename(path_value):
#     return os.path.basename(str(path_value).strip().replace("\\", "/"))


# def _path_exists(img_dir, path_value):
#     if pd.isna(path_value) or str(path_value).strip() == "":
#         return False
#     return os.path.exists(os.path.join(img_dir, _resolve_filename(path_value)))


# # ==================================================================
# # 2. PERSISTED, LEAKAGE-FREE TRAIN/VAL SPLIT
# # ==================================================================
# def build_or_load_split(csv_path, img_dir, force_rebuild=False):
#     """
#     Returns (train_df, val_df). Builds and persists the split on first
#     call; every subsequent call (from any script) loads the same
#     persisted split, guaranteeing no leakage between training and
#     evaluation across the whole project.
#     """
#     if (
#         not force_rebuild
#         and os.path.exists(Config.TRAIN_SPLIT_CSV)
#         and os.path.exists(Config.VAL_SPLIT_CSV)
#     ):
#         train_df = pd.read_csv(Config.TRAIN_SPLIT_CSV)
#         val_df = pd.read_csv(Config.VAL_SPLIT_CSV)
#         print(
#             f"[SPLIT] Loaded persisted split: "
#             f"{len(train_df)} train / {len(val_df)} val"
#         )
#         return train_df, val_df

#     print("[SPLIT] No persisted split found (or force_rebuild=True). Building...")
#     df = load_driving_log(csv_path)

#     # Validate existence of center image (mandatory) and flag left/right.
#     df["center_exists"] = df["center"].apply(lambda p: _path_exists(img_dir, p))
#     df["left_exists"] = df["left"].apply(lambda p: _path_exists(img_dir, p))
#     df["right_exists"] = df["right"].apply(lambda p: _path_exists(img_dir, p))

#     n_before = len(df)
#     df = df[df["center_exists"]].copy()
#     n_after = len(df)
#     if n_after == 0:
#         raise RuntimeError(
#             f"No valid center-camera images found under {img_dir}. "
#             "Check the dataset folder / CSV path."
#         )
#     if n_after < n_before:
#         print(f"[SPLIT] Dropped {n_before - n_after} rows with missing center image.")

#     n_missing_side = int((~df["left_exists"]).sum() + (~df["right_exists"]).sum())
#     if n_missing_side > 0:
#         print(
#             f"[SPLIT] WARNING: {n_missing_side} left/right camera references are "
#             "missing on disk. These rows are kept (center image is valid); "
#             "recovery-camera sampling will skip missing files at load time."
#         )

#     # Balance the steering distribution (downsample the straight-driving spike).
#     zero_steer = df[df["steering"] == 0]
#     non_zero = df[df["steering"] != 0]
#     keep_zeros = zero_steer.sample(
#         frac=Config.DOWNSAMPLE_ZERO_FRAC, random_state=Config.VAL_SPLIT_SEED
#     )
#     balanced_df = pd.concat([keep_zeros, non_zero]).sample(
#         frac=1.0, random_state=Config.VAL_SPLIT_SEED
#     ).reset_index(drop=True)

#     # Fixed, seed-independent split.
#     split_idx = int(len(balanced_df) * Config.TRAIN_VAL_SPLIT)
#     train_df = balanced_df.iloc[:split_idx].reset_index(drop=True)
#     val_df = balanced_df.iloc[split_idx:].reset_index(drop=True)

#     train_df.to_csv(Config.TRAIN_SPLIT_CSV, index=False)
#     val_df.to_csv(Config.VAL_SPLIT_CSV, index=False)

#     print(
#         f"[SPLIT] Built and persisted new split "
#         f"({Config.TRAIN_VAL_SPLIT*100:.0f}/{(1-Config.TRAIN_VAL_SPLIT)*100:.0f}): "
#         f"{len(train_df)} train / {len(val_df)} val -> saved to {Config.SPLIT_DIR}"
#     )
#     return train_df, val_df


# # ==================================================================
# # 3. PYTORCH DATASET
# # ==================================================================
# class DrivingDataset(Dataset):
#     """
#     mode='train' -> stochastic camera sampling, flip, and brightness
#                     augmentation.
#     mode='eval'  -> fully deterministic: center camera only, no
#                     augmentation. Used for BOTH validation-during-training
#                     and all post-hoc evaluation/visualization scripts.
#     """

#     def __init__(self, dataframe, img_dir, mode="train"):
#         assert mode in ("train", "eval"), "mode must be 'train' or 'eval'"
#         self.data = dataframe.reset_index(drop=True)
#         self.img_dir = img_dir
#         self.mode = mode

#         self.base_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225]),
#         ])

#     def __len__(self):
#         return len(self.data)

#     def _augment_brightness(self, image):
#         hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
#         ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
#         hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
#         return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

#     def _select_camera(self, row):
#         """Randomly picks center/left/right, falling back to center if
#         the chosen side-camera file is missing on disk."""
#         steering = float(row["steering"])
#         choice = np.random.choice(["center", "left", "right"])

#         if choice != "center":
#             candidate_path = row[choice]
#             if pd.isna(candidate_path) or not _path_exists(self.img_dir, candidate_path):
#                 choice = "center"  # graceful fallback

#         if choice == "left":
#             steering += Config.STEERING_OFFSET
#         elif choice == "right":
#             steering -= Config.STEERING_OFFSET
#         return choice, steering

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]

#         if self.mode == "train":
#             camera_choice, steering = self._select_camera(row)
#         else:
#             camera_choice, steering = "center", float(row["steering"])

#         filename = _resolve_filename(row[camera_choice])
#         img_path = os.path.join(self.img_dir, filename)
#         image = cv2.imread(img_path)

#         if image is None:
#             # Corrupted/missing file -> try the next sample deterministically.
#             return self.__getitem__((idx + 1) % len(self.data))

#         image = cv2.resize(image, Config.IMAGE_SIZE)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # Sky/background is intentionally NOT cropped: GAB-Net must learn
#         # to suppress it via the learned spatial bottleneck.

#         if self.mode == "train":
#             if np.random.rand() > 0.5:
#                 image = cv2.flip(image, 1)
#                 steering = -steering
#             if np.random.rand() > 0.5:
#                 image = self._augment_brightness(image)

#         steering = float(np.clip(steering, -1.0, 1.0))

#         pil_image = Image.fromarray(image)
#         img_tensor = self.base_transform(pil_image)
#         steer_tensor = torch.tensor(steering, dtype=torch.float32)

#         return img_tensor, steer_tensor


# # ==================================================================
# # 4. DATALOADER FACTORY
# # ==================================================================
# def get_dataloaders(csv_path, img_dir, seed, force_rebuild_split=False):
#     """
#     Returns (train_loader, val_loader) built from the PERSISTED split.
#     `seed` controls only the shuffling/generator behaviour of the
#     training loader; the underlying train/val row assignment is fixed
#     (see build_or_load_split).
#     """
#     train_df, val_df = build_or_load_split(csv_path, img_dir, force_rebuild=force_rebuild_split)

#     train_dataset = DrivingDataset(train_df, img_dir, mode="train")
#     val_dataset = DrivingDataset(val_df, img_dir, mode="eval")

#     generator = torch.Generator()
#     generator.manual_seed(seed)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=Config.NUM_WORKERS,
#         worker_init_fn=seed_worker,
#         generator=generator,
#         persistent_workers=Config.NUM_WORKERS > 0,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=Config.NUM_WORKERS,
#         worker_init_fn=seed_worker,
#         persistent_workers=Config.NUM_WORKERS > 0,
#     )

#     return train_loader, val_loader


# def load_eval_dataset(csv_path, img_dir):
#     """
#     Convenience loader used by evaluate.py / visualize.py: returns ONLY
#     the persisted, held-out validation dataframe + a matching
#     deterministic Dataset. Never builds a new split as a side effect
#     unless one genuinely doesn't exist yet.
#     """
#     _, val_df = build_or_load_split(csv_path, img_dir, force_rebuild=False)
#     return val_df, DrivingDataset(val_df, img_dir, mode="eval")


# # ==================================================================
# # 5. MANUAL VALIDATION SAMPLE EXPORT (for the IoU heuristic sanity check)
# # ==================================================================
# def export_manual_validation_sample(csv_path, img_dir, n=None, out_dir=None, seed=None):
#     """
#     Saves n raw validation frames as PNGs plus a CSV template with an
#     empty `human_road_mask_path` column. This does NOT auto-annotate
#     anything -- it is a tool to let a human quickly hand-label a small
#     sample of frames (e.g. in an image editor / labeling tool) so the
#     Canny-edge + bottom-half heuristic road prior used in evaluate.py can
#     be sanity-checked against real ground truth before being trusted for
#     a quantitative IoU claim in the thesis.
#     """
#     n = n or Config.N_MANUAL_VALIDATION_SAMPLES
#     out_dir = out_dir or Config.MANUAL_VAL_DIR
#     seed = seed if seed is not None else Config.VAL_SPLIT_SEED
#     os.makedirs(out_dir, exist_ok=True)

#     val_df, val_dataset = load_eval_dataset(csv_path, img_dir)
#     rng = np.random.RandomState(seed)
#     sample_indices = rng.choice(len(val_dataset), size=min(n, len(val_dataset)), replace=False)

#     records = []
#     for i, idx in enumerate(sample_indices):
#         img_tensor, steer = val_dataset[idx]
#         img = img_tensor.permute(1, 2, 0).numpy()
#         img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
#         img = np.clip(img, 0, 255).astype(np.uint8)

#         out_path = os.path.join(out_dir, f"manual_sample_{i:03d}.png")
#         cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

#         records.append({
#             "sample_id": i,
#             "val_index": int(idx),
#             "steering_gt": float(steer),
#             "frame_path": out_path,
#             "human_road_mask_path": "",  # to be filled in manually
#         })

#     manifest_path = os.path.join(out_dir, "manual_validation_manifest.csv")
#     pd.DataFrame(records).to_csv(manifest_path, index=False)
#     print(f"[MANUAL-IOU] Exported {len(records)} frames + manifest to {out_dir}")
#     return manifest_path


"""
dataset.py
==========
Data loading, cleaning, splitting, and the PyTorch Dataset class.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from config import Config
from utils import seed_worker

CSV_COLUMNS = ["center", "left", "right", "steering", "throttle", "brake", "speed"]


# ==================================================================
# 1. ROBUST CSV LOADING (single source of truth)
# ==================================================================
def load_driving_log(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Driving log CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, header=None)
    first_cell = str(raw.iloc[0, 0]).lower()

    if ".jpg" in first_cell or "img" in first_cell or "center" not in first_cell:
        df = raw.iloc[:, :len(CSV_COLUMNS)].copy()
        df.columns = CSV_COLUMNS
    else:
        df = pd.read_csv(csv_path)
        df = df.iloc[:, :len(CSV_COLUMNS)].copy()
        df.columns = CSV_COLUMNS

    df["steering"] = df["steering"].astype(float)
    return df


def _resolve_filename(path_value):
    return os.path.basename(str(path_value).strip().replace("\\", "/"))


def _path_exists(img_dir, path_value):
    if pd.isna(path_value) or str(path_value).strip() == "":
        return False
    return os.path.exists(os.path.join(img_dir, _resolve_filename(path_value)))


# ==================================================================
# 2. PERSISTED, LEAKAGE-FREE TRAIN/VAL SPLIT
# ==================================================================
def build_or_load_split(csv_path, img_dir, force_rebuild=False):
    if (
        not force_rebuild
        and os.path.exists(Config.TRAIN_SPLIT_CSV)
        and os.path.exists(Config.VAL_SPLIT_CSV)
    ):
        train_df = pd.read_csv(Config.TRAIN_SPLIT_CSV)
        val_df = pd.read_csv(Config.VAL_SPLIT_CSV)
        print(
            f"[SPLIT] Loaded persisted split: "
            f"{len(train_df)} train / {len(val_df)} val"
        )
        return train_df, val_df

    print("[SPLIT] No persisted split found (or force_rebuild=True). Building...")
    df = load_driving_log(csv_path)

    df["center_exists"] = df["center"].apply(lambda p: _path_exists(img_dir, p))
    df["left_exists"] = df["left"].apply(lambda p: _path_exists(img_dir, p))
    df["right_exists"] = df["right"].apply(lambda p: _path_exists(img_dir, p))

    n_before = len(df)
    df = df[df["center_exists"]].copy()
    n_after = len(df)
    if n_after == 0:
        raise RuntimeError(
            f"No valid center-camera images found under {img_dir}. "
            "Check the dataset folder / CSV path."
        )
    if n_after < n_before:
        print(f"[SPLIT] Dropped {n_before - n_after} rows with missing center image.")

    n_missing_side = int((~df["left_exists"]).sum() + (~df["right_exists"]).sum())
    if n_missing_side > 0:
        print(
            f"[SPLIT] WARNING: {n_missing_side} left/right camera references are "
            "missing on disk. These rows are kept (center image is valid); "
            "recovery-camera sampling will skip missing files at load time."
        )

    zero_steer = df[df["steering"] == 0]
    non_zero = df[df["steering"] != 0]
    keep_zeros = zero_steer.sample(
        frac=Config.DOWNSAMPLE_ZERO_FRAC, random_state=Config.VAL_SPLIT_SEED
    )
    balanced_df = pd.concat([keep_zeros, non_zero]).sample(
        frac=1.0, random_state=Config.VAL_SPLIT_SEED
    ).reset_index(drop=True)

    split_idx = int(len(balanced_df) * Config.TRAIN_VAL_SPLIT)
    train_df = balanced_df.iloc[:split_idx].reset_index(drop=True)
    val_df = balanced_df.iloc[split_idx:].reset_index(drop=True)

    train_df.to_csv(Config.TRAIN_SPLIT_CSV, index=False)
    val_df.to_csv(Config.VAL_SPLIT_CSV, index=False)

    print(
        f"[SPLIT] Built and persisted new split "
        f"({Config.TRAIN_VAL_SPLIT*100:.0f}/{(1-Config.TRAIN_VAL_SPLIT)*100:.0f}): "
        f"{len(train_df)} train / {len(val_df)} val -> saved to {Config.SPLIT_DIR}"
    )
    return train_df, val_df


# ==================================================================
# 3. PYTORCH DATASET
# ==================================================================
class DrivingDataset(Dataset):
    def __init__(self, dataframe, img_dir, mode="train"):
        assert mode in ("train", "eval"), "mode must be 'train' or 'eval'"
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.mode = mode

        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def _augment_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def _select_camera(self, row):
        steering = float(row["steering"])
        choice = np.random.choice(["center", "left", "right"])

        if choice != "center":
            candidate_path = row[choice]
            if pd.isna(candidate_path) or not _path_exists(self.img_dir, candidate_path):
                choice = "center"  

        if choice == "left":
            steering += Config.STEERING_OFFSET
        elif choice == "right":
            steering -= Config.STEERING_OFFSET
        return choice, steering

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        if self.mode == "train":
            camera_choice, steering = self._select_camera(row)
        else:
            camera_choice, steering = "center", float(row["steering"])

        filename = _resolve_filename(row[camera_choice])
        img_path = os.path.join(self.img_dir, filename)
        image = cv2.imread(img_path)

        if image is None:
            return self.__getitem__((idx + 1) % len(self.data))

        image = cv2.resize(image, Config.IMAGE_SIZE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == "train":
            if np.random.rand() > 0.5:
                image = cv2.flip(image, 1)
                steering = -steering
            if np.random.rand() > 0.5:
                image = self._augment_brightness(image)

        steering = float(np.clip(steering, -1.0, 1.0))

        pil_image = Image.fromarray(image)
        img_tensor = self.base_transform(pil_image)
        steer_tensor = torch.tensor(steering, dtype=torch.float32)

        return img_tensor, steer_tensor


# ==================================================================
# 4. DATALOADER FACTORY
# ==================================================================
def get_dataloaders(csv_path, img_dir, seed, force_rebuild_split=False):
    train_df, val_df = build_or_load_split(csv_path, img_dir, force_rebuild=force_rebuild_split)

    train_dataset = DrivingDataset(train_df, img_dir, mode="train")
    val_dataset = DrivingDataset(val_df, img_dir, mode="eval")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=Config.NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        worker_init_fn=seed_worker,
        persistent_workers=Config.NUM_WORKERS > 0,
    )

    return train_loader, val_loader


def load_eval_dataset(csv_path, img_dir):
    _, val_df = build_or_load_split(csv_path, img_dir, force_rebuild=False)
    return val_df, DrivingDataset(val_df, img_dir, mode="eval")


# ==================================================================
# 5. MANUAL VALIDATION SAMPLE EXPORT
# ==================================================================
def export_manual_validation_sample(csv_path, img_dir, n=None, out_dir=None, seed=None):
    n = n or Config.N_MANUAL_VALIDATION_SAMPLES
    out_dir = out_dir or Config.MANUAL_VAL_DIR
    seed = seed if seed is not None else Config.VAL_SPLIT_SEED
    os.makedirs(out_dir, exist_ok=True)

    val_df, val_dataset = load_eval_dataset(csv_path, img_dir)
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(len(val_dataset), size=min(n, len(val_dataset)), replace=False)

    records = []
    for i, idx in enumerate(sample_indices):
        img_tensor, steer = val_dataset[idx]
        img = img_tensor.permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        img = np.clip(img, 0, 255).astype(np.uint8)

        out_path = os.path.join(out_dir, f"manual_sample_{i:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        records.append({
            "sample_id": i,
            "val_index": int(idx),
            "steering_gt": float(steer),
            "frame_path": out_path,
            "human_road_mask_path": "",  
        })

    manifest_path = os.path.join(out_dir, "manual_validation_manifest.csv")
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    print(f"[MANUAL-IOU] Exported {len(records)} frames + manifest to {out_dir}")
    return manifest_path