"""Statistical analysis for Question 5: bootstrap confidence intervals,
paired significance testing between PICA-Net and the POS baseline (as
Chen et al., 2026, do for TDA-Phys vs. PhysFormer), and Bland-Altman
agreement analysis.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sstats


def bootstrap_ci(values: np.ndarray, statistic=np.mean, n_boot: int = 2000, ci: float = 0.95, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), (float("nan"), float("nan"))
    boot_stats = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    lo = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return float(statistic(values)), (float(lo), float(hi))


def paired_error_test(pred_a_hr: np.ndarray, pred_b_hr: np.ndarray, gt_hr: np.ndarray):
    """Paired t-test on absolute errors of two methods (A = PICA-Net,
    B = POS baseline) evaluated on the same windows."""
    err_a = np.abs(np.asarray(pred_a_hr) - np.asarray(gt_hr))
    err_b = np.abs(np.asarray(pred_b_hr) - np.asarray(gt_hr))
    valid = np.isfinite(err_a) & np.isfinite(err_b)
    err_a, err_b = err_a[valid], err_b[valid]
    if len(err_a) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "n": len(err_a)}
    t_stat, p_value = sstats.ttest_rel(err_a, err_b)
    return {"t_stat": float(t_stat), "p_value": float(p_value), "n": int(len(err_a)),
            "mean_abs_err_a": float(err_a.mean()), "mean_abs_err_b": float(err_b.mean())}


def bland_altman_stats(pred_hr: np.ndarray, gt_hr: np.ndarray):
    pred_hr = np.asarray(pred_hr, dtype=np.float64)
    gt_hr = np.asarray(gt_hr, dtype=np.float64)
    valid = np.isfinite(pred_hr) & np.isfinite(gt_hr)
    pred_hr, gt_hr = pred_hr[valid], gt_hr[valid]
    diff = pred_hr - gt_hr
    mean = (pred_hr + gt_hr) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std())
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd
    return {
        "bias": bias, "sd": sd,
        "loa_lower": float(loa_lower), "loa_upper": float(loa_upper),
        "mean_hr": mean, "diff": diff,
    }
