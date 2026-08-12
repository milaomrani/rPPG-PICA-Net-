"""Physics-informed data augmentation (Question 2's "physics-informed
training" pillar and Question 4's synthetic-augmentation strategy),
implemented here as label-preserving, dichromatic-model-consistent
perturbations of real training windows rather than full synthetic face
synthesis (which would require an image-animation pipeline, out of scope
for this implementation -- see the Q5 report for this explicit scoping
decision).

Grounded in the skin-reflection model used throughout Questions 1-4
(Wang et al., 2017): C_k(t) = I(t) * (u_c*c0 + u_s*s(t) + u_p*p(t)) + v_n(t).
The augmentation below perturbs exactly the nuisance terms this model
identifies as non-physiological -- the stationary skin-reflection color
u_c*c0 (simulated skin-tone/illuminant variation) and the intensity/
specular terms I(t), u_s*s(t) (simulated lighting drift and motion-
induced specular noise) -- while leaving the pulsatile term u_p*p(t)
(and therefore the ground-truth heart-rate label) untouched, and while
keeping every injected perturbation band-limited to frequencies well
below the physiological band (0.7-3 Hz) so it cannot mimic or corrupt
genuine cardiac-frequency content.
"""

from __future__ import annotations

import numpy as np


def _smooth_low_freq_trajectory(n_frames: int, fps: float, rng: np.random.Generator,
                                 max_freq_hz: float = 0.3, n_components: int = 3) -> np.ndarray:
    """A smooth random trajectory built from a few sinusoids strictly below
    the physiological band, representing slow illumination drift or
    motion-induced specular variation rather than anything resembling a
    pulse."""
    t = np.arange(n_frames) / fps
    traj = np.zeros(n_frames)
    for _ in range(n_components):
        freq = rng.uniform(0.02, max_freq_hz)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.3, 1.0)
        traj += amp * np.sin(2 * np.pi * freq * t + phase)
    traj /= (np.abs(traj).max() + 1e-8)  # normalize to [-1, 1]
    return traj


def physics_augment(appearance: np.ndarray, diff: np.ndarray, fps: float,
                     rng: np.random.Generator | None = None,
                     channel_gain_range=(0.85, 1.15),
                     illumination_amplitude=0.15,
                     specular_amplitude=0.03) -> tuple[np.ndarray, np.ndarray]:
    """Apply one physics-informed perturbation to a single (appearance, diff)
    window pair. Shapes: (T, R, 3, H, W). Does not touch the ground-truth
    HR/PPG labels, since nothing here is a function of the pulsatile term.

    - Per-channel gain simulates a different skin-tone/illuminant color
      balance (u_c). This cancels exactly in a normalized frame difference
      (c*f2 - c*f1)/(c*f2 + c*f1) = (f2-f1)/(f2+f1) for a per-channel
      constant c, so it is applied to `appearance` only -- a direct,
      checkable consequence of the same normalization argument Chen and
      McDuff (2018) use to justify differencing in the first place.
    - A slow (<0.3 Hz), multiplicative illumination trajectory simulates
      lighting drift (I(t)).
    - A slow, small additive perturbation to `diff` simulates motion-
      induced specular noise (u_s*s(t)), which is spectrally neutral
      (added identically across channels) per the dichromatic model.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = appearance.shape[0]
    appearance_aug = appearance.copy()
    diff_aug = diff.copy()

    # Skin-tone / illuminant color-balance perturbation (appearance only).
    gains = rng.uniform(channel_gain_range[0], channel_gain_range[1], size=3).astype(np.float32)
    appearance_aug *= gains[None, None, :, None, None]

    # Slow illumination-drift trajectory (appearance only, multiplicative).
    illum = _smooth_low_freq_trajectory(T, fps, rng)
    illum_factor = (1.0 + illumination_amplitude * illum).astype(np.float32)
    appearance_aug *= illum_factor[:, None, None, None, None]

    # Slow, spectrally-neutral specular-like perturbation (diff only, additive).
    specular = _smooth_low_freq_trajectory(T, fps, rng)
    specular_term = (specular_amplitude * specular).astype(np.float32)
    diff_aug += specular_term[:, None, None, None, None]

    appearance_aug = np.clip(appearance_aug, 0.0, None)
    return appearance_aug.astype(np.float32), diff_aug.astype(np.float32)
