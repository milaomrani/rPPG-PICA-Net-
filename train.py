"""Caching, dataset assembly, subject-independent splitting, and the
training loop for PICA-Net (Question 5), following the train/validation/
test strategy defined in Question 4 (subject independence is mandatory;
full skin-tone/lighting/motion stratification requires metadata that
UBFC-rPPG does not provide, so this scoped-down implementation performs
subject-independent splitting only -- see the Q5 report for the explicit
discussion of this limitation).
"""

from __future__ import annotations

import os
import collections
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from data import find_subjects, process_video_to_windows

CACHE_DTYPE = np.float16


def build_cache(dataset_root: str, cache_dir: str, window_frames=160, stride=80,
                 detect_every=15, max_frames=None, verbose=True):
    os.makedirs(cache_dir, exist_ok=True)
    subjects = find_subjects(dataset_root)
    cached_subjects = []
    for vid_path, gt_path, subject_id in subjects:
        cache_path = os.path.join(cache_dir, f"{subject_id}.npz")
        if not os.path.exists(cache_path):
            if verbose:
                print(f"[cache] processing {subject_id} ...")
            windows = list(process_video_to_windows(
                vid_path, gt_path, window_frames=window_frames, stride=stride,
                detect_every=detect_every, max_frames=max_frames,
            ))
            if len(windows) == 0:
                if verbose:
                    print(f"[cache] {subject_id}: too short, skipped")
                continue
            appearance = np.stack([w["appearance"] for w in windows]).astype(CACHE_DTYPE)
            diff = np.stack([w["diff"] for w in windows]).astype(CACHE_DTYPE)
            mask = np.stack([w["mask"] for w in windows]).astype(CACHE_DTYPE)
            hr_bpm = np.array([w["hr_bpm"] for w in windows], dtype=np.float32)
            ppg = np.stack([w["ppg"] for w in windows]).astype(np.float32)
            rgb_trace = np.stack([w["rgb_trace"] for w in windows]).astype(np.float32)
            fps = windows[0]["fps"]
            np.savez_compressed(cache_path, appearance=appearance, diff=diff, mask=mask,
                                 hr_bpm=hr_bpm, ppg=ppg, rgb_trace=rgb_trace, fps=fps)
            if verbose:
                print(f"[cache] {subject_id}: {len(windows)} windows cached")
        else:
            if verbose:
                print(f"[cache] {subject_id}: cache hit")
        cached_subjects.append(subject_id)
    return cached_subjects


def make_window_index(subject_ids: list[str], cache_dir: str):
    """Flat (subject_id, window_idx) index, for temporal (within-subject)
    splitting when too few subjects are available for a subject-independent
    split (Question 5 pilot-data fallback -- see report caveat)."""
    index = []
    for sid in subject_ids:
        path = os.path.join(cache_dir, f"{sid}.npz")
        if not os.path.exists(path):
            continue
        n_windows = np.load(path)["hr_bpm"].shape[0]
        index += [(sid, i) for i in range(n_windows)]
    return index


def temporal_split_indices(index: list[tuple], train_frac=0.6, val_frac=0.2):
    """Chronological split of a single subject's (or few subjects')
    windows: NOT subject-independent, used only as a pilot/smoke-level
    fallback when fewer than ~5 subjects are available."""
    n = len(index)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n))) if n - n_train > 1 else 0
    return index[:n_train], index[n_train:n_train + n_val], index[n_train + n_val:]


def subject_split(subject_ids: list[str], train_frac=0.6, val_frac=0.2, seed=0):
    ids = sorted(subject_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n))) if n - n_train > 1 else 0
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    if len(test_ids) == 0 and len(val_ids) > 0:
        test_ids = [val_ids.pop()]
    return train_ids, val_ids, test_ids


class LRUSubjectCache:
    def __init__(self, cache_dir: str, max_resident: int = 6):
        self.cache_dir = cache_dir
        self.max_resident = max_resident
        self._data = collections.OrderedDict()

    def get(self, subject_id: str):
        if subject_id in self._data:
            self._data.move_to_end(subject_id)
            return self._data[subject_id]
        path = os.path.join(self.cache_dir, f"{subject_id}.npz")
        npz = np.load(path)
        entry = {k: npz[k] for k in npz.files}
        self._data[subject_id] = entry
        if len(self._data) > self.max_resident:
            self._data.popitem(last=False)
        return entry


class WindowDataset(Dataset):
    """Flat index over (subject_id, window_idx) pairs, backed by an
    LRU-cached per-subject npz store to bound memory use."""

    def __init__(self, subject_ids: list[str], cache_dir: str, max_resident: int = 6,
                 explicit_index: list[tuple] | None = None, augment: bool = False, seed: int = 0):
        self.cache_dir = cache_dir
        self.cache = LRUSubjectCache(cache_dir, max_resident=max_resident)
        # Physics-informed augmentation (Question 2/4): applied on-the-fly,
        # freshly resampled on every access, to training windows only.
        self.augment = augment
        self._rng = np.random.default_rng(seed)
        if explicit_index is not None:
            self.index = list(explicit_index)
        else:
            self.index = []
            for sid in subject_ids:
                path = os.path.join(cache_dir, f"{sid}.npz")
                if not os.path.exists(path):
                    continue
                n_windows = np.load(path)["hr_bpm"].shape[0]
                for i in range(n_windows):
                    self.index.append((sid, i))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, i = self.index[idx]
        entry = self.cache.get(sid)
        appearance_np = entry["appearance"][i].astype(np.float32)  # (T,R,3,H,W)
        diff_np = entry["diff"][i].astype(np.float32)

        if self.augment:
            from physics_augment import physics_augment
            appearance_np, diff_np = physics_augment(appearance_np, diff_np, float(entry["fps"]), self._rng)

        appearance = torch.from_numpy(appearance_np)
        diff = torch.from_numpy(diff_np)
        mask = torch.from_numpy(entry["mask"][i].astype(np.float32))
        appearance = appearance * mask.unsqueeze(2)  # cheap deterministic skin pre-filter (Q3)
        diff = diff * mask.unsqueeze(2)

        # rearrange (T, R, C, H, W) -> (R, C, T, H, W) for Conv3d
        appearance = appearance.permute(1, 2, 0, 3, 4)
        diff = diff.permute(1, 2, 0, 3, 4)

        ppg = torch.from_numpy(entry["ppg"][i])
        hr = torch.tensor(entry["hr_bpm"][i], dtype=torch.float32)
        fps = float(entry["fps"])
        return appearance, diff, ppg, hr, fps


def _val_hr_diagnostic(model, val_loader, device):
    """Quick per-epoch diagnostic: predicted-vs-true HR correlation and MAE
    on the validation set, using the current model weights. Tracked
    separately from the training loss because Question 5's report found
    that loss alone can look fine while the model has collapsed to a
    near-constant, non-discriminative prediction -- correlation is the
    metric that actually exposes that failure mode during training."""
    from pos_baseline import hr_from_pulse

    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for appearance, diff, ppg, hr, fps in val_loader:
            appearance, diff = appearance.to(device), diff.to(device)
            out = model(appearance, diff)
            pulses = out["pulse"].cpu().numpy()
            for p, f, h in zip(pulses, fps.tolist(), hr.tolist()):
                preds.append(hr_from_pulse(p, f))
                gts.append(h)
    preds, gts = np.array(preds), np.array(gts)
    mae = float(np.mean(np.abs(preds - gts)))
    r = float(np.corrcoef(preds, gts)[0, 1]) if np.std(preds) > 1e-6 else float("nan")
    return mae, r


def train_model(model, train_ds, val_ds, device="cpu", epochs=15, batch_size=8, lr=1e-3,
                 verbose=True, diagnostic_every=10, weight_decay=0.0):
    from losses import composite_loss

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model = model.to(device)

    history = {"train_loss": [], "val_loss": [], "val_hr_mae": [], "val_hr_r": []}
    best_val = float("inf")
    best_state = None

    fps_ref = None
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for appearance, diff, ppg, hr, fps in train_loader:
            fps_ref = float(fps[0])
            appearance, diff, ppg, hr = appearance.to(device), diff.to(device), ppg.to(device), hr.to(device)
            optimizer.zero_grad()
            out = model(appearance, diff)
            loss, _ = composite_loss(out, ppg, hr, fps_ref)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for appearance, diff, ppg, hr, fps in val_loader:
                appearance, diff, ppg, hr = appearance.to(device), diff.to(device), ppg.to(device), hr.to(device)
                out = model(appearance, diff)
                loss, _ = composite_loss(out, ppg, hr, float(fps[0]))
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        do_diag = (epoch == 0) or ((epoch + 1) % diagnostic_every == 0) or (epoch == epochs - 1)
        if do_diag:
            val_hr_mae, val_hr_r = _val_hr_diagnostic(model, val_loader, device)
            history["val_hr_mae"].append((epoch, val_hr_mae))
            history["val_hr_r"].append((epoch, val_hr_r))
            if verbose:
                print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"val_hr_mae={val_hr_mae:.2f}  val_hr_r={val_hr_r:.3f}")
        elif verbose:
            print(f"epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
