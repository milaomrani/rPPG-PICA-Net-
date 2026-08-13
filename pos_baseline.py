"""Traditional POS algorithm (Wang, den Brinker, Stuijk & de Haan, 2017).

Reference baseline used throughout Questions 1-4: a training-free,
physiologically-grounded projection method that any proposed deep network
must be benchmarked against, especially cross-dataset.
"""

import numpy as np


def pos_algorithm(rgb_signal: np.ndarray, fps: float, window_seconds: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin pulse extraction.

    Parameters
    ----------
    rgb_signal : (N, 3) array of spatially-averaged RGB values per frame, ordered [R, G, B].
    fps : video frame rate.
    window_seconds : sliding window length (Wang et al. use l=32 @ 20fps = 1.6s).

    Returns
    -------
    H : (N,) array, the reconstructed pulse signal (Algorithm 1 in Wang et al., 2017).
    """
    rgb_signal = np.asarray(rgb_signal, dtype=np.float64)
    n = rgb_signal.shape[0]
    l = max(2, int(round(window_seconds * fps)))

    H = np.zeros(n)
    proj = np.array([[0.0, 1.0, -1.0],
                      [-2.0, 1.0, 1.0]])

    for start in range(0, n - l + 1):
        window = rgb_signal[start:start + l]  # (l, 3)
        mu = window.mean(axis=0)
        mu[mu == 0] = 1e-8
        C_n = window / mu  # temporal normalization (Eq. 12 in Wang et al., 2017)

        S = C_n @ proj.T  # (l, 2) -> S1, S2
        S1, S2 = S[:, 0], S[:, 1]
        std1, std2 = S1.std(), S2.std()
        alpha = std1 / std2 if std2 > 1e-8 else 0.0
        h = S1 + alpha * S2  # Eq. 34
        h = h - h.mean()

        H[start:start + l] += h  # overlap-add

    return H


def hr_from_pulse(pulse: np.ndarray, fps: float, band_hz=(0.7, 3.0), parabolic: bool = True) -> float:
    """Estimate heart rate (bpm) from a pulse waveform via FFT peak search
    restricted to the physiologically plausible band, as used throughout
    Questions 1-4 (Wang et al., 2017; Liu et al., 2023).

    With `parabolic=True` (default), the discrete FFT peak is refined via
    quadratic interpolation using its two neighboring bins, which relaxes
    the fps/n_frames bin-width resolution limit identified in Question 5
    (e.g. ~11 bpm/bin at 160 frames @ 29.5 fps) to a small fraction of a
    bin, at negligible extra cost (a handful of scalar operations).
    """
    pulse = np.asarray(pulse, dtype=np.float64)
    n = len(pulse)
    if n < 8:
        return float("nan")
    pulse = pulse - pulse.mean()
    window = np.hanning(n)
    spec = np.fft.rfft(pulse * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(spec) ** 2

    lo, hi = band_hz
    band_mask = (freqs >= lo) & (freqs <= hi)
    band_idx = np.where(band_mask)[0]
    if len(band_idx) == 0:
        return float("nan")
    k = band_idx[np.argmax(power[band_idx])]
    peak_freq = freqs[k]

    if parabolic and 0 < k < len(power) - 1:
        p_left, p_center, p_right = power[k - 1], power[k], power[k + 1]
        denom = (p_left - 2 * p_center + p_right)
        if abs(denom) > 1e-12:
            delta = 0.5 * (p_left - p_right) / denom
            delta = float(np.clip(delta, -1.0, 1.0))  # guard against pathological curvature
            bin_width = freqs[1] - freqs[0]
            peak_freq = peak_freq + delta * bin_width

    return float(peak_freq * 60.0)


def snr_db(pulse: np.ndarray, fps: float, gt_hr_bpm: float, band_hz=(0.7, 4.0)) -> float:
    """Signal-to-noise ratio (dB) around the first two harmonics of the
    ground-truth HR, following Wang et al. (2017) / Liu et al. (2023)."""
    pulse = np.asarray(pulse, dtype=np.float64)
    n = len(pulse)
    if n < 8 or not np.isfinite(gt_hr_bpm):
        return float("nan")
    pulse = pulse - pulse.mean()
    spec = np.fft.rfft(pulse * np.hanning(n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    power = np.abs(spec) ** 2

    gt_hz = gt_hr_bpm / 60.0
    harmonics = [gt_hz, 2 * gt_hz]
    tol = 0.15  # Hz tolerance window around each harmonic

    signal_mask = np.zeros_like(freqs, dtype=bool)
    for h in harmonics:
        signal_mask |= (freqs >= h - tol) & (freqs <= h + tol)

    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    noise_mask = band_mask & (~signal_mask)

    signal_power = power[signal_mask].sum()
    noise_power = power[noise_mask].sum()
    if noise_power <= 0 or signal_power <= 0:
        return float("nan")
    return float(10.0 * np.log10(signal_power / noise_power))
