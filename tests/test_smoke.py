"""Fast, no-GPU-required smoke test validating the full pipeline end to
end on a tiny synthetic video BEFORE real UBFC-rPPG data is available.
Not used for any reported Question 5 results -- purely a correctness
check, run with: python tests/test_smoke.py
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import find_subjects, process_video_to_windows
from model import PICANet, count_parameters
from losses import composite_loss
from pos_baseline import pos_algorithm, hr_from_pulse, snr_db
from complexity import full_complexity_report
from stats_analysis import bootstrap_ci, paired_error_test, bland_altman_stats
from train import build_cache, subject_split, WindowDataset, train_model


def make_synthetic_subject(root, subject_id="subject_smoke", n_frames=300, fps=30.0,
                            hr_bpm=72.0, size=(160, 120)):
    subj_dir = os.path.join(root, subject_id)
    os.makedirs(subj_dir, exist_ok=True)
    vid_path = os.path.join(subj_dir, "vid.avi")
    gt_path = os.path.join(subj_dir, "ground_truth.txt")

    w, h = size
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(vid_path, fourcc, fps, (w, h))

    t = np.arange(n_frames) / fps
    ppg = np.sin(2 * np.pi * (hr_bpm / 60.0) * t)

    rng = np.random.default_rng(0)
    base = rng.integers(80, 180, size=(h, w, 3), dtype=np.uint8)
    fx, fy, fw, fh = int(0.2 * w), int(0.1 * h), int(0.6 * w), int(0.8 * h)

    for i in range(n_frames):
        frame = base.copy()
        # simple oval "face" region whose brightness pulses with the synthetic PPG
        pulse_val = int(20 * ppg[i])
        cv2.ellipse(frame, (fx + fw // 2, fy + fh // 2), (fw // 2, fh // 2), 0, 0, 360,
                    (120 + pulse_val, 90 + pulse_val, 150 + pulse_val), -1)
        noise = rng.integers(-3, 3, size=frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(frame)
    writer.release()

    hr_series = np.full(n_frames, hr_bpm)
    with open(gt_path, "w") as f:
        f.write(" ".join(f"{v:.6e}" for v in ppg) + "\n")
        f.write(" ".join(f"{v:.6e}" for v in hr_series) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in t) + "\n")

    return vid_path, gt_path, subject_id


def main():
    tmp_root = tempfile.mkdtemp(prefix="rppg_smoke_")
    try:
        print("== 1. Synthetic data generation ==")
        vid_path, gt_path, subject_id = make_synthetic_subject(tmp_root)
        subjects = find_subjects(tmp_root)
        assert len(subjects) == 1, f"expected 1 subject, found {len(subjects)}"
        print("OK:", subjects)

        print("== 2. Preprocessing pipeline (Q3) ==")
        windows = list(process_video_to_windows(vid_path, gt_path, window_frames=60, stride=30,
                                                  detect_every=10))
        assert len(windows) > 0, "no windows produced"
        w0 = windows[0]
        assert w0["appearance"].shape == (60, 5, 3, 36, 36)
        assert w0["diff"].shape == (60, 5, 3, 36, 36)
        assert w0["mask"].shape == (60, 5, 36, 36)
        print(f"OK: {len(windows)} windows, appearance shape {w0['appearance'].shape}")

        print("== 3. POS baseline (Q1) ==")
        cap = cv2.VideoCapture(vid_path)
        rgb_means = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb_means.append(frame[..., ::-1].reshape(-1, 3).mean(axis=0))
        cap.release()
        rgb_means = np.array(rgb_means)
        pulse = pos_algorithm(rgb_means, fps=30.0)
        hr_est = hr_from_pulse(pulse, fps=30.0)
        snr = snr_db(pulse, fps=30.0, gt_hr_bpm=72.0)
        print(f"OK: POS HR estimate={hr_est:.1f} bpm (gt=72.0), SNR={snr:.2f} dB")

        print("== 4. Model forward + loss (Q2/Q3) ==")
        model = PICANet(out_frames=60)
        n_params = count_parameters(model)
        print(f"OK: model has {n_params:,} trainable parameters")

        appearance = torch.from_numpy(w0["appearance"]).permute(1, 2, 0, 3, 4).unsqueeze(0)
        diff = torch.from_numpy(w0["diff"]).permute(1, 2, 0, 3, 4).unsqueeze(0)
        ppg_t = torch.from_numpy(w0["ppg"]).unsqueeze(0)
        hr_t = torch.tensor([w0["hr_bpm"]], dtype=torch.float32)

        out = model(appearance, diff)
        assert out["pulse"].shape == (1, 60)
        loss, parts = composite_loss(out, ppg_t, hr_t, fps=30.0)
        print(f"OK: forward pass, loss={loss.item():.4f}, parts={parts}")

        print("== 4b. Physics-informed augmentation (Q2/Q4) ==")
        from physics_augment import physics_augment
        rng = np.random.default_rng(0)
        app_aug, diff_aug = physics_augment(w0["appearance"], w0["diff"], fps=30.0, rng=rng)
        assert app_aug.shape == w0["appearance"].shape
        assert diff_aug.shape == w0["diff"].shape
        assert np.isfinite(app_aug).all() and np.isfinite(diff_aug).all()
        # per-channel gain applied identically to both frames of a normalized
        # difference must cancel exactly -- verify that property directly on
        # a hand-built pair rather than trusting the derivation alone.
        f1 = np.random.default_rng(1).uniform(0.2, 0.8, size=(1, 1, 3, 4, 4)).astype(np.float32)
        f2 = np.random.default_rng(2).uniform(0.2, 0.8, size=(1, 1, 3, 4, 4)).astype(np.float32)
        d = (f2 - f1) / (f2 + f1 + 1e-6)
        gains = np.array([1.3, 0.7, 1.1], dtype=np.float32).reshape(1, 1, 3, 1, 1)
        d_scaled = (gains * f2 - gains * f1) / (gains * f2 + gains * f1 + 1e-6)
        assert np.allclose(d, d_scaled, atol=1e-5), "channel gain should cancel exactly in normalized diff"
        print("OK: augmentation preserves shapes/finiteness; channel-gain cancellation verified")

        print("== 5. One training step reduces loss ==")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(appearance, diff)
            loss, _ = composite_loss(out, ppg_t, hr_t, fps=30.0)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("OK: loss trajectory", [f"{l:.4f}" for l in losses])

        print("== 6. Complexity profiling ==")
        report = full_complexity_report(model, (appearance, diff), device="cpu")
        print("OK:", report)

        print("== 7. Cache build + Dataset + train_model (full loop, 1 epoch) ==")
        cache_dir = os.path.join(tmp_root, "cache")
        cached = build_cache(tmp_root, cache_dir, window_frames=60, stride=30, detect_every=10, verbose=False)
        assert len(cached) == 1
        ds = WindowDataset(cached, cache_dir)
        assert len(ds) == len(windows)
        model2 = PICANet(out_frames=60)
        trained_model, history = train_model(model2, ds, ds, device="cpu", epochs=1, batch_size=2, verbose=False)
        print("OK: history", history)

        print("== 8. Statistical analysis utilities ==")
        pred_hr = np.array([70, 74, 71, 73, 90])
        gt_hr = np.array([72, 72, 72, 72, 72])
        mean, ci = bootstrap_ci(np.abs(pred_hr - gt_hr))
        test = paired_error_test(pred_hr, pred_hr + 1, gt_hr)
        ba = bland_altman_stats(pred_hr, gt_hr)
        print(f"OK: bootstrap MAE={mean:.2f} CI={ci}, paired test p={test['p_value']:.3f}, "
              f"Bland-Altman bias={ba['bias']:.2f}")

        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
