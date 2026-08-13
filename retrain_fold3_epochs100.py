"""Retrain the best fold (fold 3: test=subject9,subject1) from the Question 5
deep-improvement pass with more epochs (100 instead of 50), to check whether
additional training time improves on the previously reported best result
(test MAE = 5.41 bpm). Same architecture, hyperparameters, and fold split as
run_cv.py -- only EPOCHS changes. Written to a separate output directory so
the existing best-fold result and checkpoints are not overwritten until the
new result is compared against it.
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch

from model import PICANet
from train import WindowDataset, train_model
from evaluate import compute_metrics
from complexity import full_complexity_report
from stats_analysis import bootstrap_ci, bland_altman_stats
from run_cv import predict_ensemble, CACHE_DIR, WINDOW_FRAMES, DROPOUT_P, WEIGHT_DECAY, \
    BATCH_SIZE, DEVICE, ENSEMBLE_SEEDS

EPOCHS = 100
OUT_DIR = "../final-report/q5_v2_assets/fold3_epoch100"

# same fold-3 split used in cv_summary.json (train/val/test subjects), so this
# is a direct like-for-like comparison against the existing best-fold result.
TRAIN_SUBJ = ["subject11", "subject5", "subject12", "subject3", "subject8", "subject10"]
VAL_SUBJ = ["subject14", "subject4"]
TEST_SUBJ = ["subject9", "subject1"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ds = WindowDataset(TRAIN_SUBJ, CACHE_DIR, augment=True)
    val_ds = WindowDataset(VAL_SUBJ, CACHE_DIR)
    test_ds = WindowDataset(TEST_SUBJ, CACHE_DIR)
    print(f"windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_hr = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    mean_hr_hz = float(train_hr.mean() / 60.0)

    models = []
    for seed in ENSEMBLE_SEEDS:
        print(f"\n-- training seed={seed}, epochs={EPOCHS} --")
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = PICANet(out_frames=WINDOW_FRAMES, base_freq_hz=mean_hr_hz, freq_range_hz=1.0,
                         dropout_p=DROPOUT_P)
        model, history = train_model(model, train_ds, val_ds, device=DEVICE, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
                                      verbose=True, diagnostic_every=25)
        models.append(model)
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"fold3_epoch100_seed{seed}.pt"))

    print("\n-- ensembling predictions --")
    train_pred, train_gt = predict_ensemble(models, train_ds, DEVICE)
    val_pred, val_gt = predict_ensemble(models, val_ds, DEVICE)
    test_pred, test_gt = predict_ensemble(models, test_ds, DEVICE)

    train_metrics = compute_metrics(train_pred, train_gt)
    val_metrics = compute_metrics(val_pred, val_gt)
    test_metrics = compute_metrics(test_pred, test_gt)

    print("\nTRAIN:", train_metrics)
    print("VAL  :", val_metrics)
    print("TEST :", test_metrics)

    from scipy import stats as scipy_stats
    r_val, p_val = scipy_stats.pearsonr(test_pred, test_gt)
    mae_mean, mae_ci = bootstrap_ci(np.abs(test_pred - test_gt))
    ba = bland_altman_stats(test_pred, test_gt)

    example = train_ds[0]
    complexity = full_complexity_report(
        models[0], (example[0].unsqueeze(0), example[1].unsqueeze(0)), device=DEVICE
    )

    result = {
        "epochs": EPOCHS,
        "train_subjects": TRAIN_SUBJ, "val_subjects": VAL_SUBJ, "test_subjects": TEST_SUBJ,
        "train_metrics": train_metrics, "val_metrics": val_metrics, "test_metrics": test_metrics,
        "test_pearson_significance": {"r": float(r_val), "p": float(p_val), "n": int(len(test_pred))},
        "test_mae_bootstrap_95ci": {"mean": mae_mean, "ci": mae_ci},
        "test_bland_altman": {k: v for k, v in ba.items() if k not in ("mean_hr", "diff")},
        "complexity": complexity,
    }
    with open(os.path.join(OUT_DIR, "fold3_epoch100_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)

    print("\n=== COMPARISON vs. 50-epoch result (test MAE=5.41, RMSE=6.81, r=0.321) ===")
    print(f"100-epoch TEST: MAE={test_metrics['mae']:.2f} RMSE={test_metrics['rmse']:.2f} r={test_metrics['pearson_r']:.3f}")
    print(f"\nDone. Results written to {OUT_DIR}/fold3_epoch100_summary.json")


if __name__ == "__main__":
    main()
