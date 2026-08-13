"""Evaluation metrics: MAE, RMSE, Pearson r, SNR -- the metric suite
mandated in Questions 1, 4, and 5 -- computed per-window and aggregated,
with the POS baseline evaluated under the identical protocol as required
by Question 4.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats

from pos_baseline import hr_from_pulse, snr_db


def compute_metrics(pred_hr: np.ndarray, gt_hr: np.ndarray) -> dict:
    pred_hr = np.asarray(pred_hr, dtype=np.float64)
    gt_hr = np.asarray(gt_hr, dtype=np.float64)
    valid = np.isfinite(pred_hr) & np.isfinite(gt_hr)
    pred_hr, gt_hr = pred_hr[valid], gt_hr[valid]

    if len(pred_hr) < 2:
        return {"n": len(pred_hr), "mae": float("nan"), "rmse": float("nan"), "pearson_r": float("nan")}

    err = pred_hr - gt_hr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if np.std(pred_hr) < 1e-8 or np.std(gt_hr) < 1e-8:
        r = float("nan")
    else:
        r, _ = sstats.pearsonr(pred_hr, gt_hr)
    return {"n": int(len(pred_hr)), "mae": mae, "rmse": rmse, "pearson_r": float(r)}


def evaluate_windows(windows: list[dict], model_pred_fn, fps: float) -> dict:
    """windows: list of dicts with keys 'hr_bpm' (gt) and whatever
    `model_pred_fn` needs. model_pred_fn(window) -> (pred_pulse, pred_hr).
    Returns metrics dict plus raw arrays for downstream statistical analysis.
    """
    pred_hrs, gt_hrs, snrs = [], [], []
    for w in windows:
        pulse, pred_hr = model_pred_fn(w)
        pred_hrs.append(pred_hr)
        gt_hrs.append(w["hr_bpm"])
        snrs.append(snr_db(pulse, fps, w["hr_bpm"]))

    metrics = compute_metrics(np.array(pred_hrs), np.array(gt_hrs))
    metrics["mean_snr_db"] = float(np.nanmean(snrs))
    metrics["pred_hr"] = np.array(pred_hrs)
    metrics["gt_hr"] = np.array(gt_hrs)
    return metrics
