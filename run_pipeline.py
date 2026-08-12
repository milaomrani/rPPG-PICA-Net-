"""End-to-end Question 5 pipeline: cache -> split -> train PICA-Net ->
evaluate PICA-Net + POS baseline on train/val/test -> complexity profile
-> statistical analysis -> plots -> markdown report.

Usage:
    python run_pipeline.py --dataset_root ../dataset/UBFC_DATASET/DATASET_2 \
                            --out_dir ../final-report/q5_assets \
                            --epochs 15
"""

from __future__ import annotations

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import find_subjects
from model import PICANet
from train import (build_cache, subject_split, make_window_index, temporal_split_indices,
                    WindowDataset, LRUSubjectCache, train_model)
from pos_baseline import pos_algorithm, hr_from_pulse
from evaluate import compute_metrics
from complexity import full_complexity_report
from stats_analysis import bootstrap_ci, paired_error_test, bland_altman_stats


def device_str():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def predict_pica_batch(model, ds: WindowDataset, indices: list[int], device: str, batch_size=8):
    model.eval().to(device)
    preds_hr, gt_hr, sids = [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            appearances, diffs, hrs, fpss = [], [], [], []
            for idx in batch_idx:
                appearance, diff, ppg, hr, fps = ds[idx]
                appearances.append(appearance)
                diffs.append(diff)
                hrs.append(hr.item())
                fpss.append(fps)
                sids.append(ds.index[idx][0])
            appearance_b = torch.stack(appearances).to(device)
            diff_b = torch.stack(diffs).to(device)
            out = model(appearance_b, diff_b)
            pulses = out["pulse"].cpu().numpy()
            for p, fps, hr_gt in zip(pulses, fpss, hrs):
                pred_hr = hr_from_pulse(p, fps)
                preds_hr.append(pred_hr)
                gt_hr.append(hr_gt)
    return np.array(preds_hr), np.array(gt_hr), sids


def predict_pos_batch(cache: LRUSubjectCache, index: list[tuple]):
    preds_hr, gt_hr = [], []
    for sid, i in index:
        entry = cache.get(sid)
        rgb_trace = entry["rgb_trace"][i]
        fps = float(entry["fps"])
        pulse = pos_algorithm(rgb_trace, fps=fps)
        pred_hr = hr_from_pulse(pulse, fps)
        preds_hr.append(pred_hr)
        gt_hr.append(float(entry["hr_bpm"][i]))
    return np.array(preds_hr), np.array(gt_hr)


def summarize_split(name, pica_pred, pica_gt, pos_pred, pos_gt):
    pica_metrics = compute_metrics(pica_pred, pica_gt)
    pos_metrics = compute_metrics(pos_pred, pos_gt)
    print(f"[{name}] PICA-Net: MAE={pica_metrics['mae']:.2f} RMSE={pica_metrics['rmse']:.2f} "
          f"r={pica_metrics['pearson_r']:.3f} (n={pica_metrics['n']})")
    print(f"[{name}] POS      : MAE={pos_metrics['mae']:.2f} RMSE={pos_metrics['rmse']:.2f} "
          f"r={pos_metrics['pearson_r']:.3f} (n={pos_metrics['n']})")
    return pica_metrics, pos_metrics


def make_plots(out_dir, history, test_pica_pred, test_pica_gt, test_pos_pred, test_pos_gt):
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(5, 4))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("epoch"); plt.ylabel("composite loss"); plt.legend(); plt.title("PICA-Net training curve")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150); plt.close()

    plt.figure(figsize=(5, 5))
    lims = [30, 180]
    plt.scatter(test_pica_gt, test_pica_pred, alpha=0.6, label="PICA-Net")
    plt.scatter(test_pos_gt, test_pos_pred, alpha=0.6, label="POS", marker="x")
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Ground-truth HR (bpm)"); plt.ylabel("Predicted HR (bpm)")
    plt.legend(); plt.title("Test-set predictions vs. ground truth")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "scatter_test.png"), dpi=150); plt.close()

    ba = bland_altman_stats(test_pica_pred, test_pica_gt)
    plt.figure(figsize=(5, 4))
    plt.scatter(ba["mean_hr"], ba["diff"], alpha=0.6)
    plt.axhline(ba["bias"], color="k", linestyle="-")
    plt.axhline(ba["loa_upper"], color="r", linestyle="--")
    plt.axhline(ba["loa_lower"], color="r", linestyle="--")
    plt.xlabel("Mean of predicted & ground-truth HR (bpm)")
    plt.ylabel("Predicted - Ground truth (bpm)")
    plt.title("Bland-Altman: PICA-Net vs. ground truth (test set)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "bland_altman_pica.png"), dpi=150); plt.close()

    return ba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="../dataset/UBFC_DATASET/DATASET_2")
    parser.add_argument("--cache_dir", type=str, default="../dataset/cache")
    parser.add_argument("--out_dir", type=str, default="../final-report/q5_assets")
    parser.add_argument("--window_frames", type=int, default=160)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--subject_independent_min", type=int, default=5,
                         help="minimum subjects required to use subject-independent splitting")
    parser.add_argument("--physics_augment", action="store_true",
                         help="apply physics-informed, dichromatic-model-consistent perturbation "
                              "augmentation (physics_augment.py) to the training set on the fly "
                              "(Question 2/4). Off by default so results stay comparable to earlier runs.")
    parser.add_argument("--device", type=str, default=None,
                         help="override auto-detected device (cpu/mps/cuda). PICA-Net's sequential "
                              "per-timestep temporal head does not parallelize well on MPS for a "
                              "model this small -- CPU is often faster in practice; benchmark both.")
    parser.add_argument("--seed", type=int, default=42,
                         help="global random seed (numpy + torch) for reproducibility across runs.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if args.device else device_str()
    print(f"device: {device}, seed: {args.seed}")

    print("== Locating subjects ==")
    subjects = find_subjects(args.dataset_root)
    print(f"found {len(subjects)} subjects: {[s[2] for s in subjects]}")
    if len(subjects) == 0:
        raise SystemExit(f"No subjects found under {args.dataset_root}. "
                          f"Expected DATASET_2-style subject folders with vid.avi + ground_truth.txt.")

    print("== Building / loading cache ==")
    cached_subjects = build_cache(args.dataset_root, args.cache_dir,
                                   window_frames=args.window_frames, stride=args.stride,
                                   max_frames=args.max_frames)
    n_subjects = len(cached_subjects)
    print(f"{n_subjects} subjects cached")

    pilot_mode = n_subjects < args.subject_independent_min
    split_info = {"pilot_mode": pilot_mode, "n_subjects": n_subjects}

    if not pilot_mode:
        train_ids, val_ids, test_ids = subject_split(cached_subjects)
        train_ds = WindowDataset(train_ids, args.cache_dir, augment=args.physics_augment)
        val_ds = WindowDataset(val_ids, args.cache_dir)
        test_ds = WindowDataset(test_ids, args.cache_dir)
        split_info.update({"train_subjects": train_ids, "val_subjects": val_ids, "test_subjects": test_ids})
        print(f"SUBJECT-INDEPENDENT split -> train {train_ids} / val {val_ids} / test {test_ids}")
    else:
        print(f"PILOT MODE ({n_subjects} subject(s) available, need >= {args.subject_independent_min} "
              f"for subject-independent splitting per Question 4). Falling back to a chronological, "
              f"within-subject split. Results below are a code-correctness / preliminary demonstration, "
              f"NOT a claim of cross-subject generalization.")
        full_index = make_window_index(cached_subjects, args.cache_dir)
        train_idx, val_idx, test_idx = temporal_split_indices(full_index)
        train_ds = WindowDataset(cached_subjects, args.cache_dir, explicit_index=train_idx, augment=args.physics_augment)
        val_ds = WindowDataset(cached_subjects, args.cache_dir, explicit_index=val_idx)
        test_ds = WindowDataset(cached_subjects, args.cache_dir, explicit_index=test_idx)
        split_info.update({"train_windows": len(train_idx), "val_windows": len(val_idx), "test_windows": len(test_idx)})
    print(f"physics-informed augmentation on training set: {args.physics_augment}")

    print(f"windows: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_hr_bpm = np.array([train_ds[i][3].item() for i in range(len(train_ds))])
    mean_hr_hz = float(train_hr_bpm.mean() / 60.0)
    print(f"training-set mean HR: {train_hr_bpm.mean():.1f} bpm ({mean_hr_hz:.3f} Hz) "
          f"-> used to initialize PICA-Net's temporal-head base frequency")

    print("== Training PICA-Net ==")
    model = PICANet(out_frames=args.window_frames, base_freq_hz=mean_hr_hz, freq_range_hz=1.0)
    model, history = train_model(model, train_ds, val_ds, device=device,
                                  epochs=args.epochs, batch_size=args.batch_size)

    print("== Evaluating PICA-Net + POS baseline on all splits ==")
    results = {}
    shared_cache = LRUSubjectCache(args.cache_dir, max_resident=len(cached_subjects) + 1)
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        pica_pred, pica_gt, _ = predict_pica_batch(model, ds, list(range(len(ds))), device, args.batch_size)
        pos_pred, pos_gt = predict_pos_batch(shared_cache, ds.index)
        pica_metrics, pos_metrics = summarize_split(name, pica_pred, pica_gt, pos_pred, pos_gt)
        results[name] = {
            "pica": pica_metrics, "pos": pos_metrics,
            "pica_pred": pica_pred.tolist(), "pica_gt": pica_gt.tolist(),
            "pos_pred": pos_pred.tolist(), "pos_gt": pos_gt.tolist(),
        }

    print("== Computational complexity ==")
    example = train_ds[0]
    appearance_ex = example[0].unsqueeze(0)
    diff_ex = example[1].unsqueeze(0)
    complexity = full_complexity_report(model, (appearance_ex, diff_ex), device=device)
    print(complexity)

    print("== Statistical analysis (test split) ==")
    test = results["test"]
    mae_mean, mae_ci = bootstrap_ci(np.abs(np.array(test["pica_pred"]) - np.array(test["pica_gt"])))
    r_mean, r_ci = bootstrap_ci(np.array(test["pica_pred"]), statistic=lambda x: np.corrcoef(x, test["pica_gt"])[0, 1]) \
        if len(test["pica_gt"]) > 2 else (float("nan"), (float("nan"), float("nan")))
    paired = paired_error_test(test["pica_pred"], test["pos_pred"], test["pica_gt"])
    print(f"PICA-Net test MAE bootstrap 95% CI: {mae_mean:.2f} {mae_ci}")
    print(f"Paired t-test (|err| PICA-Net vs POS): t={paired['t_stat']:.3f} p={paired['p_value']:.4f}")

    print("== Plots ==")
    ba = make_plots(args.out_dir, history, np.array(test["pica_pred"]), np.array(test["pica_gt"]),
                     np.array(test["pos_pred"]), np.array(test["pos_gt"]))

    summary = {
        "split_info": split_info,
        "results": results,  # includes per-window pica/pos pred+gt arrays for full transparency
        "complexity": complexity,
        "bootstrap_test_mae_95ci": {"mean": mae_mean, "ci": mae_ci},
        "paired_ttest_pica_vs_pos": paired,
        "bland_altman_pica_test": {k: v for k, v in ba.items() if k not in ("mean_hr", "diff")},
        "history": history,
        "device": device,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    model_path = os.path.join(args.out_dir, "pica_net.pt")
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "split_info": split_info}, model_path)

    print(f"\nDone. Results written to {args.out_dir}/summary.json, model checkpoint at {model_path}, and plots.")
    return summary


if __name__ == "__main__":
    main()
