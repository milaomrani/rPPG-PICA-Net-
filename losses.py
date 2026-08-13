"""Composite loss for PICA-Net (Question 2/3): waveform fidelity,
frequency-domain plausibility, and physics-consistency, mirroring the
loss design discussed for PhysFormer/PhysFormer++ and TDA-Phys in
Question 1, extended with an explicit physics-consistency term tied to
PICA-Net's phase/frequency dynamical state.
"""

import torch
import torch.nn.functional as F


def neg_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation, averaged over the batch."""
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred ** 2).sum(dim=1) * (target ** 2).sum(dim=1) + 1e-8)
    r = num / den
    return (1 - r).mean()


def frequency_band_loss(pred: torch.Tensor, hr_bpm: torch.Tensor, fps: float, band_hz=(0.7, 3.0)) -> torch.Tensor:
    """Cross-entropy-style loss encouraging the power spectrum of `pred`
    to peak at the ground-truth HR frequency bin, restricted to the
    physiologically plausible band (Debnath & Kim, 2025; Question 3)."""
    B, T = pred.shape
    device = pred.device
    window = torch.hann_window(T, device=device)
    spec = torch.fft.rfft(pred * window, dim=1)
    power = (spec.real ** 2 + spec.imag ** 2)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fps).to(device)

    band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    band_freqs = freqs[band_mask]
    band_power = power[:, band_mask]

    log_probs = F.log_softmax(band_power, dim=1)

    target_hz = (hr_bpm / 60.0).unsqueeze(1)  # (B,1)
    # soft target: Gaussian around the true frequency bin (label-distribution loss)
    sigma = 0.1
    target_dist = torch.exp(-((band_freqs.unsqueeze(0) - target_hz) ** 2) / (2 * sigma ** 2))
    target_dist = target_dist / (target_dist.sum(dim=1, keepdim=True) + 1e-8)

    loss = -(target_dist * log_probs).sum(dim=1).mean()
    return loss


def physics_consistency_loss(freq_hz: torch.Tensor, max_accel_hz_per_step: float = 0.05) -> torch.Tensor:
    """Penalizes implausibly abrupt changes in the instantaneous frequency
    trace predicted by the physics-informed temporal head, reflecting the
    prior that heart rate cannot change discontinuously (Question 2)."""
    d_freq = freq_hz[:, 1:] - freq_hz[:, :-1]
    excess = F.relu(d_freq.abs() - max_accel_hz_per_step)
    return (excess ** 2).mean()


def composite_loss(outputs: dict, target_ppg: torch.Tensor, target_hr_bpm: torch.Tensor, fps: float,
                    w_pearson=1.0, w_freq=2.0, w_physics=0.1):
    # w_freq raised from 0.5 to 2.0: this is the loss term that most directly
    # supervises each window's own true HR bin (a per-window cross-entropy
    # against the ground-truth frequency), and is therefore the most direct
    # available signal against the near-constant-prediction collapse
    # diagnosed in Question 5's report -- raising its weight relative to the
    # waveform-shape (Pearson) and smoothness (physics) terms is intended to
    # penalize that collapse more strongly during training.
    pred = outputs["pulse"]
    l_pearson = neg_pearson_loss(pred, target_ppg)
    l_freq = frequency_band_loss(pred, target_hr_bpm, fps)
    l_phys = physics_consistency_loss(outputs["freq_hz"])
    total = w_pearson * l_pearson + w_freq * l_freq + w_physics * l_phys
    return total, {"pearson": l_pearson.item(), "freq": l_freq.item(), "physics": l_phys.item()}
