"""Deep improvement pass for Question 5 (user-requested): 5-fold subject-
independent cross-validation, so every one of the 10 available subjects is
used as a held-out test subject exactly once, combined with a 2-seed
ensemble per fold (averaging predicted HR), physics-informed augmentation
(shown in the prior iteration to help), dropout + weight decay
(regularization against the small-sample overfitting diagnosed throughout
this work), a longer 240-frame window and parabolic FFT-peak interpolation
(both targeting the ~11 bpm frequency-resolution limit identified
earlier).

This directly implements the top four levers identified when the user
asked "how can we increase model performance": (1) use all subjects via
CV rather than one fixed split, (2) ensemble instead of picking one seed,
(3) regularize given how little data there is, (4) fix the frequency
resolution limit. More real subjects (the single highest-leverage lever)
is not implemented here since it requires new data the user would need to
download.
"""

from __future__ import annotations

import os
import json
import numpy as np
import torch

from data import find_subjects
from model import PICANet
from train import build_cache, WindowDataset, train_model, LRUSubjectCache
from pos_baseline import hr_from_pulse
from evaluate import compute_metrics
from complexity import full_complexity_report
from stats_analysis import bootstrap_ci, bland_altman_stats

DATASET_ROOT = "../dataset/UBFC_DATASET/DATASET_2"
CACHE_DIR = "../dataset/cache_w240"          # new cache dir: window size changed (160 -> 240)
OUT_DIR = "../final-report/q5_v2_assets"     # new folder, as requested
WINDOW_FRAMES = 240
STRIDE = 80
EPOCHS = 50
BATCH_SIZE = 4
DEVICE = "cpu"                               # measured faster than MPS for this architecture
DROPOUT_P = 0.3
WEIGHT_DECAY = 1e-4
ENSEMBLE_SEEDS = [42, 123]
N_FOLDS = 5


def make_folds(subjects: list[str], n_folds: int, seed: int = 0):
    """Deterministic partition of subjects into n_folds test groups, with
    the remaining subjects per fold split into train/val. Every subject
    appears as a test subject in exactly one fold."""
    ids = sorted(subjects)
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    fold_size = len(shuffled) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = start + fold_size if i < n_folds - 1 else len(shuffled)
        test_subj = shuffled[start:end]
        remaining = [s for s in shuffled if s not in test_subj]
        n_val = max(1, len(remaining) // 4)
        val_subj = remaining[:n_val]
        train_subj = remaining[n_val:]
        folds.append({"train": train_subj, "val": val_subj, "test": test_subj})
    return folds


def predict_ensemble(models: list, ds: WindowDataset, device: str):
    """Average each of the ensemble models' predicted HR (via parabolic-
    interpolated FFT peak) per window."""
    for m in models:
        m.eval().to(device)
    all_preds = []  # (n_models, n_windows)
    gts = []
    with torch.no_grad():
        for i in range(len(ds)):
            appearance, diff, ppg, hr, fps = ds[i]
            appearance_b = appearance.unsqueeze(0).to(device)
            diff_b = diff.unsqueeze(0).to(device)
            preds_this_window = []
            for m in models:
                out = m(appearance_b, diff_b)
                pulse = out["pulse"][0].cpu().numpy()
                preds_this_window.append(hr_from_pulse(pulse, fps))
            all_preds.append(np.nanmean(preds_this_window))
            gts.append(hr.item())
    return np.array(all_preds), np.array(gts)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    subjects = [s[2] for s in find_subjects(DATASET_ROOT)]
    print(f"found {len(subjects)} subjects: {subjects}")

    print("== building/loading cache (240-frame windows) ==")
    cached = build_cache(DATASET_ROOT, CACHE_DIR, window_frames=WINDOW_FRAMES, stride=STRIDE)
    print(f"{len(cached)} subjects cached at window_frames={WINDOW_FRAMES}")

    folds = make_folds(cached, N_FOLDS)
    print(f"{len(folds)} folds:")
    for i, f in enumerate(folds):
        print(f"  fold {i}: train={f['train']} val={f['val']} test={f['test']}")

    fold_results = []
    all_test_pred, all_test_gt, all_test_subject = [], [], []

    for fold_idx, fold in enumerate(folds):
        print(f"\n=== FOLD {fold_idx}: test={fold['test']} ===")
        train_ds = WindowDataset(fold["train"], CACHE_DIR, augment=True)
        val_ds = WindowDataset(fold["val"], CACHE_DIR)
        test_ds = WindowDataset(fold["test"], CACHE_DIR)
        print(f"windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

        train_hr = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
        mean_hr_hz = float(train_hr.mean() / 60.0)

        ensemble_models = []
        for seed in ENSEMBLE_SEEDS:
            print(f"-- training ensemble member, seed={seed} --")
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = PICANet(out_frames=WINDOW_FRAMES, base_freq_hz=mean_hr_hz, freq_range_hz=1.0,
                             dropout_p=DROPOUT_P)
            model, history = train_model(model, train_ds, val_ds, device=DEVICE, epochs=EPOCHS,
                                          batch_size=BATCH_SIZE, weight_decay=WEIGHT_DECAY,
                                          verbose=True, diagnostic_every=25)
            ensemble_models.append(model)

        print("-- ensembling predictions --")
        val_pred, val_gt = predict_ensemble(ensemble_models, val_ds, DEVICE)
        test_pred, test_gt = predict_ensemble(ensemble_models, test_ds, DEVICE)

        val_metrics = compute_metrics(val_pred, val_gt)
        test_metrics = compute_metrics(test_pred, test_gt)
        print(f"fold {fold_idx} VAL : MAE={val_metrics['mae']:.2f} RMSE={val_metrics['rmse']:.2f} r={val_metrics['pearson_r']:.3f}")
        print(f"fold {fold_idx} TEST: MAE={test_metrics['mae']:.2f} RMSE={test_metrics['rmse']:.2f} r={test_metrics['pearson_r']:.3f}")

        all_test_pred.extend(test_pred.tolist())
        all_test_gt.extend(test_gt.tolist())
        all_test_subject.extend([",".join(fold["test"])] * len(test_pred))

        example = train_ds[0]
        complexity = full_complexity_report(
            ensemble_models[0], (example[0].unsqueeze(0), example[1].unsqueeze(0)), device=DEVICE
        )

        fold_results.append({
            "fold": fold_idx,
            "train_subjects": fold["train"], "val_subjects": fold["val"], "test_subjects": fold["test"],
            "val_metrics": val_metrics, "test_metrics": test_metrics,
            "val_pred": val_pred.tolist(), "val_gt": val_gt.tolist(),
            "test_pred": test_pred.tolist(), "test_gt": test_gt.tolist(),
            "complexity": complexity,
        })

        # save ensemble checkpoints for this fold
        for i, m in enumerate(ensemble_models):
            torch.save(m.state_dict(), os.path.join(OUT_DIR, f"fold{fold_idx}_seed{ENSEMBLE_SEEDS[i]}.pt"))

    # ---- pooled, whole-dataset CV metrics (every subject used as test exactly once) ----
    all_test_pred = np.array(all_test_pred)
    all_test_gt = np.array(all_test_gt)
    pooled_metrics = compute_metrics(all_test_pred, all_test_gt)
    mae_mean, mae_ci = bootstrap_ci(np.abs(all_test_pred - all_test_gt))
    ba = bland_altman_stats(all_test_pred, all_test_gt)

    fold_test_maes = [f["test_metrics"]["mae"] for f in fold_results]
    fold_test_rs = [f["test_metrics"]["pearson_r"] for f in fold_results]

    summary = {
        "config": {
            "window_frames": WINDOW_FRAMES, "stride": STRIDE, "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "device": DEVICE, "dropout_p": DROPOUT_P,
            "weight_decay": WEIGHT_DECAY, "ensemble_seeds": ENSEMBLE_SEEDS, "n_folds": N_FOLDS,
            "parabolic_interpolation": True, "physics_augmentation": True,
        },
        "per_fold": [{k: v for k, v in f.items() if k not in ("val_pred", "val_gt", "test_pred", "test_gt")}
                     for f in fold_results],
        "pooled_test_metrics_all_10_subjects": pooled_metrics,
        "pooled_test_mae_bootstrap_95ci": {"mean": mae_mean, "ci": mae_ci},
        "pooled_bland_altman": {k: v for k, v in ba.items() if k not in ("mean_hr", "diff")},
        "fold_test_mae_mean_std": {"mean": float(np.mean(fold_test_maes)), "std": float(np.std(fold_test_maes))},
        "fold_test_r_mean_std": {"mean": float(np.mean(fold_test_rs)), "std": float(np.std(fold_test_rs))},
        "pooled_test_pred": all_test_pred.tolist(),
        "pooled_test_gt": all_test_gt.tolist(),
    }

    with open(os.path.join(OUT_DIR, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== POOLED CV RESULTS (all 10 subjects, each used as test exactly once) ===")
    print(pooled_metrics)
    print("fold-level MAE mean/std:", summary["fold_test_mae_mean_std"])
    print("fold-level r mean/std:", summary["fold_test_r_mean_std"])
    print(f"\nDone. Results written to {OUT_DIR}/cv_summary.json")


if __name__ == "__main__":
    main()
