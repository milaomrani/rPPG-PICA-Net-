"""Data loading and preprocessing for UBFC-rPPG, implementing the input
pipeline designed in Question 3: landmark-anchored ROI extraction,
temporal-difference computation, skin-segmentation masking, and windowing.
"""

from __future__ import annotations

import os
import glob
import numpy as np
import cv2

ROI_LAYOUT = {
    # relative (x, y, w, h) offsets within the detected face bounding box,
    # heuristic anatomical placement (forehead / cheeks / nose / chin) used
    # in the absence of a full facial-landmark model, as noted in Q3.
    "forehead": (0.30, 0.05, 0.40, 0.18),
    "left_cheek": (0.12, 0.45, 0.28, 0.28),
    "right_cheek": (0.60, 0.45, 0.28, 0.28),
    "nose": (0.38, 0.35, 0.24, 0.25),
    "chin": (0.32, 0.75, 0.36, 0.18),
}
REGION_NAMES = list(ROI_LAYOUT.keys())
ROI_SIZE = 36  # per Chen & McDuff (2018): small, fixed resolution trade-off


class FaceTracker:
    """Lightweight face detector + hold-last-box tracker.

    Runs the (comparatively expensive) Haar-cascade detector only every
    `detect_every` frames and reuses/interpolates the last known box in
    between, following the efficiency strategy discussed in Question 3
    (Liu et al., 2023, dynamic-detection scheme).
    """

    def __init__(self, detect_every: int = 15):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.detect_every = detect_every
        self._last_box = None
        self._frame_idx = 0

    def __call__(self, frame_bgr: np.ndarray):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        need_detect = (self._frame_idx % self.detect_every == 0) or (self._last_box is None)
        if need_detect:
            faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) > 0:
                # largest face
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                self._last_box = tuple(faces[0])
        self._frame_idx += 1
        return self._last_box  # (x, y, w, h) or None


def extract_regions(frame_bgr: np.ndarray, face_box) -> dict:
    """Crop and resize the five anatomical ROIs from a face bounding box.
    Returns dict[region_name] -> (ROI_SIZE, ROI_SIZE, 3) BGR uint8 array,
    or None for every region if no face box is available.
    """
    if face_box is None:
        return {name: None for name in REGION_NAMES}

    x, y, w, h = face_box
    H, W = frame_bgr.shape[:2]
    out = {}
    for name, (rx, ry, rw, rh) in ROI_LAYOUT.items():
        cx0 = int(x + rx * w)
        cy0 = int(y + ry * h)
        cw = max(2, int(rw * w))
        ch = max(2, int(rh * h))
        cx1 = min(W, cx0 + cw)
        cy1 = min(H, cy0 + ch)
        cx0 = max(0, cx0)
        cy0 = max(0, cy0)
        if cx1 <= cx0 or cy1 <= cy0:
            out[name] = None
            continue
        crop = frame_bgr[cy0:cy1, cx0:cx1]
        crop = cv2.resize(crop, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)
        out[name] = crop
    return out


def skin_mask(bgr_roi: np.ndarray) -> np.ndarray:
    """Cheap, deterministic skin-probability mask via YCrCb thresholding
    (the fast pre-filter discussed in Q3, complementary to the network's
    learned cross-attention reliability weighting)."""
    ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper).astype(np.float32) / 255.0
    return mask  # (ROI_SIZE, ROI_SIZE) in [0, 1]


def parse_ubfc_dataset2_gt(path: str):
    """Parse DATASET_2-style ground_truth.txt: three whitespace-separated
    rows -> (ppg_wave, hr_bpm, time_seconds)."""
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    ppg = np.array([float(v) for v in lines[0].split()])
    hr = np.array([float(v) for v in lines[1].split()])
    t = np.array([float(v) for v in lines[2].split()])
    return ppg, hr, t


def find_subjects(dataset_root: str):
    """Locate every (video_path, gt_path, subject_id) pair under
    DATASET_2-style subject folders. DATASET_1's gtdump.xmp format is
    intentionally not parsed here (different sensor/format, minority of
    the literature uses it); this can be extended if needed."""
    subjects = []
    for gt_path in glob.glob(os.path.join(dataset_root, "**", "ground_truth.txt"), recursive=True):
        folder = os.path.dirname(gt_path)
        vid_path = os.path.join(folder, "vid.avi")
        if os.path.exists(vid_path):
            subject_id = os.path.basename(folder)
            subjects.append((vid_path, gt_path, subject_id))
    subjects.sort(key=lambda t: t[2])
    return subjects


def process_video_to_windows(
    video_path: str,
    gt_path: str,
    window_frames: int = 160,
    stride: int = 80,
    detect_every: int = 15,
    max_frames: int | None = None,
):
    """Full Question-3 preprocessing pipeline for one video.

    Yields dicts with:
      appearance : (window_frames, n_regions, 3, ROI_SIZE, ROI_SIZE) float32, normalized
      diff       : (window_frames, n_regions, 3, ROI_SIZE, ROI_SIZE) float32, temporal difference
      mask       : (window_frames, n_regions, ROI_SIZE, ROI_SIZE) float32, skin mask
      hr_bpm     : scalar ground-truth HR averaged over the window
      ppg        : (window_frames,) ground-truth PPG waveform resampled to frame times
      fps        : float
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    ppg_gt, hr_gt, t_gt = parse_ubfc_dataset2_gt(gt_path)

    tracker = FaceTracker(detect_every=detect_every)
    n_regions = len(REGION_NAMES)

    raw_frames = []  # list of dict(region -> HxWx3 float32 in [0,1])
    masks_frames = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        regions = extract_regions(frame, tracker(frame))
        raw = {}
        masks = {}
        for name in REGION_NAMES:
            roi = regions[name]
            if roi is None:
                raw[name] = np.zeros((ROI_SIZE, ROI_SIZE, 3), dtype=np.float32)
                masks[name] = np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.float32)
            else:
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                raw[name] = rgb
                masks[name] = skin_mask(roi)
        raw_frames.append(raw)
        masks_frames.append(masks)
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break
    cap.release()

    n_total = len(raw_frames)
    if n_total < window_frames + 1:
        return

    # Video frame timestamps assumed uniform at `fps`.
    frame_times = np.arange(n_total) / fps
    ppg_interp = np.interp(frame_times, t_gt, ppg_gt)

    # Some UBFC-rPPG ground-truth files contain sensor-dropout placeholder
    # values (observed as HR = 1.0 bpm for extended stretches, e.g. in
    # subject11) rather than true readings. Filter to the physiologically
    # plausible adult HR band before interpolating, so dropout periods are
    # bridged from neighboring valid samples instead of corrupting the
    # window-averaged HR label.
    valid_hr = (hr_gt >= 30) & (hr_gt <= 220)
    if valid_hr.sum() < 2:
        return
    hr_interp = np.interp(frame_times, t_gt[valid_hr], hr_gt[valid_hr])

    # Stack per-region arrays: (n_total, n_regions, 3, H, W)
    appearance_full = np.zeros((n_total, n_regions, 3, ROI_SIZE, ROI_SIZE), dtype=np.float32)
    mask_full = np.zeros((n_total, n_regions, ROI_SIZE, ROI_SIZE), dtype=np.float32)
    for i in range(n_total):
        for r, name in enumerate(REGION_NAMES):
            appearance_full[i, r] = np.transpose(raw_frames[i][name], (2, 0, 1))
            mask_full[i, r] = masks_frames[i][name]

    # Temporally-normalized frame difference (Chen & McDuff, 2018, Eq. 11),
    # computed per region per channel.
    eps = 1e-6
    diff_full = np.zeros_like(appearance_full)
    diff_full[:-1] = (appearance_full[1:] - appearance_full[:-1]) / (
        appearance_full[1:] + appearance_full[:-1] + eps
    )
    diff_full[-1] = diff_full[-2]

    for start in range(0, n_total - window_frames + 1, stride):
        end = start + window_frames
        appearance_win = appearance_full[start:end]
        diff_win = diff_full[start:end]
        mask_win = mask_full[start:end]

        # remove the dominant stationary term by dividing by the temporal mean
        mean_app = appearance_win.mean(axis=0, keepdims=True) + eps
        appearance_norm = appearance_win / mean_app

        # raw (unnormalized), skin-masked spatial-average RGB trace across
        # all five regions, for the traditional POS baseline (Question 1),
        # which expects a single whole-face-style average per frame.
        mask_win_exp = mask_win[:, :, None, :, :]  # (T, R, 1, H, W)
        weighted = appearance_win * mask_win_exp
        pixel_count = mask_win_exp.sum(axis=(1, 3, 4)) + eps  # (T, 1) broadcastable below
        rgb_trace = weighted.sum(axis=(1, 3, 4)) / pixel_count  # (T, 3)

        yield {
            "appearance": appearance_norm.astype(np.float32),
            "diff": diff_win.astype(np.float32),
            "mask": mask_win.astype(np.float32),
            "hr_bpm": float(hr_interp[start:end].mean()),
            "ppg": ppg_interp[start:end].astype(np.float32),
            "rgb_trace": rgb_trace.astype(np.float32),
            "fps": float(fps),
        }
