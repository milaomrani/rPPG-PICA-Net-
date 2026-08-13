"""PICA-Net (Physics-Informed Cross-Attention Network), the architecture
proposed in Question 2, implemented here in a computationally tractable
form for real training/evaluation in Question 5.

Pipeline per region: depthwise-separable 3D conv encoder -> temporal
tokens. Cross-attention fusion across the five regions -> fused tokens.
Physics-informed temporal head: an explicit phase/frequency/amplitude
recurrence reads the fused tokens out into a pulse waveform, so that
occlusion of all regions simultaneously can be handled by propagating the
dynamical state instead of collapsing to an unconstrained prediction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise 3D conv + pointwise 1x1x1 conv: the efficient
    spatio-temporal building block discussed in Questions 2-3."""

    def __init__(self, in_ch, out_ch, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_ch, in_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class RegionEncoder(nn.Module):
    """Shared-weight spatio-temporal encoder applied independently to each
    of the five anatomical regions (weight sharing reflects that the same
    physiological pulse-extraction function should apply regardless of
    which region is being observed).

    Input: (B, 6, T, H, W) -- 3 appearance + 3 temporal-difference channels.
    Output: (B, T', D) per-region temporal token sequence.
    """

    def __init__(self, in_channels=6, base_ch=16, embed_dim=64, dropout_p=0.0):
        super().__init__()
        self.block1 = DepthwiseSeparableConv3d(in_channels, base_ch, stride=(1, 2, 2))
        self.block2 = DepthwiseSeparableConv3d(base_ch, base_ch * 2, stride=(2, 2, 2))
        self.block3 = DepthwiseSeparableConv3d(base_ch * 2, embed_dim, stride=(2, 2, 2))
        # Channel-wise (3D) dropout between conv blocks: regularizes against
        # overfitting to the handful of real subjects available (Question 5's
        # "increase model performance" follow-up), by randomly dropping
        # entire feature channels rather than individual spatio-temporal
        # positions, which better matches how Conv3d features are reused.
        self.drop2 = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.drop3 = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.drop2(x)
        x = self.block3(x)
        x = self.drop3(x)
        x = x.mean(dim=(-2, -1))          # (B, D, T') -- manual spatial mean (MPS-safe,
                                           # unlike nn.AdaptiveAvgPool3d which lacks an MPS kernel)
        return x.transpose(1, 2)          # (B, T', D)


class CrossAttentionFusion(nn.Module):
    """Learned, per-timestep reliability weighting across the five region
    tokens (Question 2), replacing the static spatial average used by
    every traditional method (POS, CHROM, PBV)."""

    def __init__(self, embed_dim=64):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, region_tokens):
        # region_tokens: (B, R, T', D)
        query = region_tokens.mean(dim=1, keepdim=True)  # (B, 1, T', D) global context
        q = self.q_proj(query)
        k = self.k_proj(region_tokens)
        v = self.v_proj(region_tokens)

        # attention over the region axis, per timestep
        attn_logits = (q * k).sum(-1) * self.scale  # (B, R, T')
        attn = torch.softmax(attn_logits, dim=1)     # softmax over regions
        fused = (attn.unsqueeze(-1) * v).sum(dim=1)  # (B, T', D)
        return fused, attn  # attn used both as diagnostic and confidence signal


class PhysicsInformedTemporalHead(nn.Module):
    """Explicit phase/frequency/amplitude dynamical read-out (Question 2).

    At each reduced timestep t, predicts a frequency deviation and an
    amplitude from the fused visual token, integrates phase, and reads
    out p_t = a_t * sin(phi_t). When confidence (attention certainty) is
    low, the recurrence still runs on its own dynamics -- the visual
    correction term is simply small/absent, which is what lets the model
    degrade gracefully rather than collapse under occlusion.
    """

    def __init__(self, embed_dim=64, base_freq_hz=1.2, freq_range_hz=1.0, dt=1.0 / 7.5, dropout_p=0.0):
        super().__init__()
        self.freq_head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.Tanh(), nn.Dropout(dropout_p), nn.Linear(32, 1)
        )
        self.amp_head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.Tanh(), nn.Dropout(dropout_p), nn.Linear(32, 1)
        )
        # Learnable base frequency, initialized from the training population's
        # mean HR (passed in by the caller) rather than a fixed generic 72 bpm
        # prior, so the model starts near the right answer and gradient
        # descent only needs to learn the per-window deviation from it.
        self.base_freq = nn.Parameter(torch.tensor(float(base_freq_hz)))
        # Deviation range widened from +/-0.6 Hz (36 bpm) to +/-1.0 Hz (60 bpm)
        # so the reachable output band comfortably covers a wider real-world
        # HR range around the learned base frequency.
        self.freq_range_hz = freq_range_hz
        self.dt = dt  # seconds per reduced timestep (depends on temporal downsampling)

    def forward(self, fused_tokens):
        # fused_tokens: (B, T', D)
        B, T, D = fused_tokens.shape
        device = fused_tokens.device

        d_freq = self.freq_head(fused_tokens).squeeze(-1)          # (B, T') Hz deviation
        amp = 0.5 + 0.5 * torch.sigmoid(self.amp_head(fused_tokens).squeeze(-1))  # (B, T') in (0.5,1)
        freq = self.base_freq + self.freq_range_hz * torch.tanh(d_freq)  # keep within a plausible band

        phase = torch.zeros(B, device=device)
        phases = []
        for t in range(T):
            phase = phase + 2 * math.pi * freq[:, t] * self.dt
            phases.append(phase)
        phase_seq = torch.stack(phases, dim=1)  # (B, T')

        pulse = amp * torch.sin(phase_seq)      # (B, T')
        return pulse, freq, amp


class PICANet(nn.Module):
    """Top-level model: 5 shared-weight region encoders -> cross-attention
    fusion -> physics-informed temporal head -> upsample to frame rate."""

    def __init__(self, n_regions=5, embed_dim=64, out_frames=160, base_freq_hz=1.2, freq_range_hz=1.0,
                 dropout_p=0.0):
        super().__init__()
        self.n_regions = n_regions
        self.encoder = RegionEncoder(in_channels=6, embed_dim=embed_dim, dropout_p=dropout_p)
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)
        self.temporal_head = PhysicsInformedTemporalHead(
            embed_dim=embed_dim, base_freq_hz=base_freq_hz, freq_range_hz=freq_range_hz, dropout_p=dropout_p
        )
        self.out_frames = out_frames

    def forward(self, appearance, diff):
        # appearance, diff: (B, R, 3, T, H, W)
        B, R, C, T, H, W = appearance.shape
        x = torch.cat([appearance, diff], dim=2)  # (B, R, 6, T, H, W)
        x = x.view(B * R, 6, T, H, W)
        tokens = self.encoder(x)                  # (B*R, T', D)
        Tp, D = tokens.shape[1], tokens.shape[2]
        tokens = tokens.view(B, R, Tp, D)

        fused, attn = self.fusion(tokens)          # (B, T', D), (B, R, T')
        pulse_low, freq, amp = self.temporal_head(fused)  # (B, T')

        pulse = F.interpolate(
            pulse_low.unsqueeze(1), size=self.out_frames, mode="linear", align_corners=False
        ).squeeze(1)  # (B, out_frames)

        return {
            "pulse": pulse,
            "attn": attn,
            "freq_hz": freq,
            "amp": amp,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
