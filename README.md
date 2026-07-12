# GAB-Net Research Codebase (Fixed & Restructured)

Implementation of the Gated Attention Bottleneck Network (GAB-Net) study,
rebuilt from scratch to fix the correctness and methodology issues found
in the original codebase. Every fix is documented directly in the
docstring of the file where it applies — read those first if you're
diffing against the old version.

## Project layout

```
config.py                 Single source of truth: paths, hyperparameters,
                           seeds, and the canonical model-name registry
                           (fixes the checkpoint filename-mismatch bug).
utils.py                  Logging, checkpoint I/O, heatmap resizing
                           (fixes the Grad-CAM shape-mismatch bug),
                           multi-seed statistics, DataLoader worker seeding.
dataset.py                Robust CSV loading, a PERSISTED leakage-free
                           train/val split (fixes the data-leakage bug),
                           the Dataset class, and a manual-IoU-validation
                           export tool.
models.py                 PilotNet, VanillaResNet, a faithful CBAM
                           baseline, the AdditiveAttnAblation ablation,
                           and GAB-Net -- all behind one MODEL_REGISTRY.
losses.py                 The MSE + L1 + TV composite loss, with warm-up
                           now correctly gating BOTH L1 and TV.
train.py                  Multi-seed training over all 5 models, with a
                           final mean/std/95% CI statistical summary.
evaluate.py                Phase 1/2 quantitative evaluation: Grad-CAM
                           (fixed), IoU vs. a heuristic road prior,
                           patch-perturbation testing, aggregated across
                           seeds, on the held-out split only.
visualize.py               Phase 1 qualitative "Glass Box" figures:
                           Raw | PilotNet+Grad-CAM | CBAM | GAB-Net.
drive.py                   Closed-loop simulator bridge (any of the 5
                           models, selectable via --model), with
                           per-frame telemetry logging for Phase 3.
closed_loop_analysis.py    Aggregates drive.py's telemetry logs into a
                           Phase 3 robustness/distribution-shift table.
requirements.txt           Pinned dependency list.
```

## Setup

```bash
pip install -r requirements.txt
```

Place your Udacity-format dataset at:
```
data/self_driving_car_dataset_make/driving_log.csv
data/self_driving_car_dataset_make/IMG/  (or img/)
```

## Running each phase

**1. Train everything (5 models x 3 seeds):**
```bash
python train.py
```
Produces `logs/research_per_run_results.csv` and
`logs/research_statistical_summary.csv` (mean +/- std / 95% CI per model).
The train/val split is built once and persisted under `data/splits/` —
every later script reuses exactly that split, so nothing is ever
evaluated on training data.

**2. Quantitative evaluation (Phase 1 proxy metrics + Phase 2):**
```bash
python evaluate.py
```
Edit `n_samples` in `__main__` (or pass `n_samples=None`) to run on the
full held-out set for final thesis numbers. Before trusting the IoU
numbers as a thesis claim, sanity-check the heuristic road prior:
```bash
python -c "from dataset import export_manual_validation_sample; from config import Config; export_manual_validation_sample(Config.DEFAULT_DRIVING_LOG, Config.resolve_img_dir('data/self_driving_car_dataset_make'))"
```
This exports a small sample of frames for you to hand-annotate and
compare against the automatic heuristic.

**3. Qualitative figures:**
```bash
python visualize.py
```
Saves `results/comparison_straight.png`, `comparison_left_turn.png`,
`comparison_right_turn.png`.

**4. Closed-loop robustness (Phase 3):**
Open the Udacity simulator, select a track, enter Autonomous Mode, then:
```bash
python drive.py --model gabnet --seed 42 --track_label training_track
python drive.py --model gabnet --seed 42 --track_label jungle_track
python drive.py --model pilotnet --seed 42 --track_label jungle_track
```
Each run logs telemetry to `results/closed_loop_logs/`. After collecting
runs for the models/tracks you want to compare:
```bash
python closed_loop_analysis.py
```
Note: the base simulator telemetry protocol used here does not expose a
genuine cross-track-error field. The analysis script reports steering
stability (frame-to-frame jerk) and an early-stop/crash proxy instead,
and will automatically use a real `cte` field if your simulator fork
provides one (see `closed_loop_analysis.py` docstring).

## Key methodological notes carried over from the thesis

- Only `gabnet` and `soft_attn` are trained with the L1 + TV sparsity
  penalty (`Config.REGULARIZED_MODEL_KEYS`). `cbam` is trained without
  it, matching how CBAM is actually used in the literature.
- The warm-up schedule (`Config.WARMUP_EPOCHS`) holds both L1 and TV at
  zero for the first N epochs, matching Figure 6.2 of the thesis exactly.
- `soft_attn` (AdditiveAttnAblation) exists specifically to isolate
  "multiplicative hard gating" from "sparsity regularization pressure" —
  it has the latter but not the former, so any gap between it and
  GAB-Net is attributable to the gating mechanism itself.
