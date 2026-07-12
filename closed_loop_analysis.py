"""
closed_loop_analysis.py
========================
Aggregates the per-frame telemetry CSVs produced by drive.py (one file
per model/seed/track run) into the Phase 3 robustness comparison table.

WHY THIS FILE EXISTS:
The original codebase's drive.py only displayed a live cv2 window and
printed to stdout -- it produced no persisted, analyzable data, so there
was no code path at all to generate the quantitative Phase-3 numbers
(cross-track error, steering stability) that the thesis's evaluation
plan promises. drive.py now logs every frame to CSV; this script turns
those logs into the comparison table.

IMPORTANT LIMITATION (documented, not hidden):
The standard Udacity Self-Driving Car Simulator telemetry payload does
NOT include a genuine cross-track-error (CTE) field in the base protocol
used here. If your specific simulator build exposes a `cte` field, wire
it through TelemetryRecorder.log() in drive.py and this script will use
it directly (see `has_true_cte` below). Otherwise, this script reports:
  - Steering Stability: standard deviation of frame-to-frame steering
    delta ("jerk"), a legitimate proxy for how erratically the policy is
    steering, independent of whether a true CTE signal is available.
  - Disengagement/Crash proxy: whether the run's log ends abruptly
    (fewer frames than a `min_expected_frames` threshold), which is a
    weak but real signal of the vehicle going off-road or the manual
    tester stopping the run early.
A true CTE-based robustness number requires either a simulator fork that
reports it, or an independent lane-position estimator run over the
recorded frames -- flagged here as a documented follow-up, not silently
approximated as something it is not.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from config import Config

FILENAME_PATTERN = re.compile(
    r"^(?P<model>.+)_seed_(?P<seed>\d+)_(?P<track>.+)_(?P<timestamp>\d+)\.csv$"
)


def _parse_run_metadata(filepath):
    fname = os.path.basename(filepath)
    match = FILENAME_PATTERN.match(fname)
    if not match:
        return None
    return match.groupdict()


def analyze_single_run(filepath, min_expected_frames=200):
    df = pd.read_csv(filepath)
    meta = _parse_run_metadata(filepath)
    if meta is None or len(df) == 0:
        return None

    steering = df["steering_angle"].to_numpy()
    jerk = np.diff(steering)
    steering_stability_std = float(np.std(jerk)) if len(jerk) > 0 else float("nan")

    has_true_cte = "cte" in df.columns
    mean_abs_cte = float(df["cte"].abs().mean()) if has_true_cte else float("nan")

    likely_early_stop = len(df) < min_expected_frames

    return {
        "Model": meta["model"],
        "Seed": int(meta["seed"]),
        "Track": meta["track"],
        "N_Frames": len(df),
        "Steering_Stability_Std": steering_stability_std,
        "Mean_Abs_CTE": mean_abs_cte,
        "Has_True_CTE": has_true_cte,
        "Likely_Early_Stop_Or_Crash": likely_early_stop,
        "Mean_Speed": float(df["speed"].mean()),
        "File": os.path.basename(filepath),
    }


def run_closed_loop_analysis(log_dir=None, min_expected_frames=200):
    log_dir = log_dir or Config.TELEMETRY_LOG_DIR
    files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))

    if not files:
        print(f"[CLOSED-LOOP] No telemetry CSVs found in {log_dir}. "
              f"Run drive.py for each model/track combination first.")
        return pd.DataFrame()

    rows = [analyze_single_run(f, min_expected_frames) for f in files]
    rows = [r for r in rows if r is not None]
    result_df = pd.DataFrame(rows)

    out_path = os.path.join(Config.RESULTS_DIR, "closed_loop_robustness_summary.csv")
    result_df.to_csv(out_path, index=False)
    print(f"[SAVED] Closed-loop robustness summary -> {out_path}\n")
    print(result_df.to_string(index=False))

    if not result_df["Has_True_CTE"].any():
        print(
            "\n[NOTE] No run in this batch reported a true 'cte' field. "
            "Robustness comparison above relies on Steering_Stability_Std "
            "and Likely_Early_Stop_Or_Crash as documented proxies, not a "
            "verified cross-track-error metric."
        )

    return result_df


if __name__ == "__main__":
    run_closed_loop_analysis()
