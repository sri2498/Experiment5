# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""CFAR-YOLO modules for SAR (Synthetic Aperture Radar) vessel detection.

This file implements the two novel components of CFAR-YOLOv11, a resource-optimised detector for dark-vessel
detection in polarimetric SAR imagery:

1. ``RobustCFARGate`` — a differentiable, parameterised robust CFAR (Constant False Alarm Rate) attention gate.
   Classical CA-CFAR estimates local sea-clutter statistics in a sliding background window surrounding a guard
   window and thresholds the cell under test against them. CA-CFAR is notoriously fragile in multi-target and
   near-shore scenes (the "capture"/masking effect), which robust variants such as truncated-statistics CFAR
   (TS-CFAR) address by censoring high-intensity outliers from the clutter estimate. ``RobustCFARGate``
   re-formulates TS-CFAR as a fully differentiable feature-space operator with a learnable per-channel detection
   threshold, so the false-alarm operating point is optimised end-to-end with the detector instead of being
   hand-tuned. The module is built exclusively from box filters (average pooling) and adds only 3 parameters per
   channel and zero convolutions, preserving the lightweight budget of YOLOv11n.

2. ``DCABlock`` / ``C3k2DCA`` — a Dense Context-Aware redesign of the YOLOv11 C3k2 block (the successor of the
   YOLOv8 C2f block that the original proposal modified). Small vessels in cluttered/near-shore backgrounds are
   only separable from clutter when local scattering evidence is combined with a wider spatial context. Each
   ``DCABlock`` therefore aggregates a multi-dilation depthwise pyramid (dilations 1/2/3 → effective receptive
   fields 3/5/7 at negligible cost), fuses it with a pointwise convolution and re-weights channels with a
   global-context (squeeze-excitation style) gate. ``C3k2DCA`` keeps the CSP split/dense-concatenation topology
   of C3k2, so every intermediate DCA output is densely aggregated into the block output.

References:
    - Robust/truncated-statistics CFAR: Tao, Anfinsen & Brekke, "Robust CFAR detector based on truncated
      statistics in multiple-target situations", IEEE TGRS 54(1), 2016.
    - HRSID benchmark: Wei et al., "HRSID: A high-resolution SAR images dataset for ship detection and instance
      segmentation", IEEE Access, 2020.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C2f
from .conv import Conv

__all__ = ("RobustCFARGate", "DCABlock", "C3k2DCA")


class RobustCFARGate(nn.Module):
    """Differentiable robust (truncated-statistics) CFAR gate with a learnable detection threshold.

    For every spatial location (cell under test) the module estimates background clutter statistics over a
    hollow annulus: a ``k_bg`` x ``k_bg`` background window minus a centred ``k_guard`` x ``k_guard`` guard
    window that excludes the target's own energy. Robustness to interfering targets inside the background
    window (the multi-target masking effect of plain CA-CFAR) is obtained with a two-pass truncated-statistics
    scheme: a first pass estimates (mu0, sigma0), the feature map is softly censored at mu0 + t_trunc * sigma0,
    and a second pass re-estimates the clutter statistics from the censored map. The detection statistic
    z = (x - mu) / sigma is converted into a soft detection decision g = sigmoid(alpha * (z - tau)) where alpha
    (slope) and tau (threshold, in clutter standard deviations) are learnable per channel — a "parameterised
    CFAR" whose false-alarm operating point is learned jointly with the detector. Salient (target-like)
    responses are residually amplified: y = x * (1 + gamma * g).

    The operator is built purely from reflect-padded box filters, adds 3 * C parameters and no convolutions,
    and degrades gracefully to identity when gamma → 0.

    Args:
        c1 (int): Number of input (= output) channels.
        k_bg (int): Odd background window size. Default 11.
        k_guard (int): Odd guard window size, strictly smaller than ``k_bg``. Default 5.
        t_trunc (float): Truncation depth in clutter standard deviations for the censoring pass. Default 2.0.
    """

    def __init__(self, c1: int, k_bg: int = 11, k_guard: int = 5, t_trunc: float = 2.0):
        """Initialize the robust CFAR gate."""
        super().__init__()
        assert k_bg % 2 == 1 and k_guard % 2 == 1, "CFAR windows must be odd-sized"
        assert k_bg > k_guard, "background window must be larger than guard window"
        self.k_bg, self.k_guard, self.t_trunc = k_bg, k_guard, float(t_trunc)
        self.alpha = nn.Parameter(torch.ones(1, c1, 1, 1))  # gate slope
        self.tau = nn.Parameter(2.0 * torch.ones(1, c1, 1, 1))  # detection threshold in clutter std units
        self.gamma = nn.Parameter(torch.ones(1, c1, 1, 1))  # residual amplification strength

    @staticmethod
    def _window_mean(x: torch.Tensor, k: int) -> torch.Tensor:
        """Sliding-window mean with reflect padding so window population is constant at image borders."""
        p = k // 2
        return F.avg_pool2d(F.pad(x, (p, p, p, p), mode="reflect"), k, stride=1)

    def _annulus_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and standard deviation over the background annulus (background minus guard window)."""
        nb, ng = self.k_bg**2, self.k_guard**2
        inv = 1.0 / (nb - ng)
        mu = (self._window_mean(x, self.k_bg) * nb - self._window_mean(x, self.k_guard) * ng) * inv
        ex2 = (self._window_mean(x * x, self.k_bg) * nb - self._window_mean(x * x, self.k_guard) * ng) * inv
        var = (ex2 - mu * mu).clamp_min(0.0)
        return mu, (var + 1e-6).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the robust CFAR gate; statistics are computed in float32 for AMP stability."""
        xf = x.float()
        mu0, sig0 = self._annulus_stats(xf)  # pass 1: raw clutter statistics
        cap = mu0 + self.t_trunc * sig0
        xt = xf - F.softplus(xf - cap)  # soft truncation, smooth approximation of min(x, cap)
        mu, sig = self._annulus_stats(xt)  # pass 2: outlier-censored clutter statistics
        z = (xf - mu) / sig  # CFAR detection statistic
        g = torch.sigmoid(self.alpha.float() * (z - self.tau.float()))  # soft detection decision
        return (xf * (1.0 + self.gamma.float() * g)).to(x.dtype)


class DCABlock(nn.Module):
    """Dense Context-Aware residual block for clutter-robust feature extraction.

    A lightweight replacement for the Bottleneck inside C3k2: three parallel depthwise 3x3 convolutions with
    dilations 1, 2 and 3 sample local scattering evidence and progressively wider sea/land context, a pointwise
    convolution fuses the pyramid, and a global-context channel gate (squeeze-excitation style, computed from
    the fused response) suppresses channels dominated by clutter. A residual connection preserves the identity
    path.

    Args:
        c (int): Number of input (= output) channels.
        shortcut (bool): Whether to add the residual connection. Default True.
        r (int): Channel reduction ratio of the global-context gate. Default 4.
    """

    def __init__(self, c: int, shortcut: bool = True, r: int = 4):
        """Initialize the Dense Context-Aware block."""
        super().__init__()
        self.dw1 = Conv(c, c, 3, 1, g=c, d=1)  # local evidence
        self.dw2 = Conv(c, c, 3, 1, g=c, d=2)  # mid-range context
        self.dw3 = Conv(c, c, 3, 1, g=c, d=3)  # wide clutter context
        self.pw = Conv(3 * c, c, 1, 1)  # pyramid fusion
        ch = max(c // r, 8)
        self.fc1 = nn.Conv2d(c, ch, 1)
        self.fc2 = nn.Conv2d(ch, c, 1)
        self.act = nn.SiLU()
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fuse the multi-dilation pyramid, apply the global-context gate and the residual connection."""
        y = self.pw(torch.cat((self.dw1(x), self.dw2(x), self.dw3(x)), 1))
        w = torch.sigmoid(self.fc2(self.act(self.fc1(F.adaptive_avg_pool2d(y, 1)))))
        y = y * w
        return x + y if self.add else y


class C3k2DCA(C2f):
    """C3k2 variant whose inner blocks are Dense Context-Aware blocks (CFAR-YOLO feature extractor).

    Keeps the CSP split and dense concatenation of every intermediate output (the C2f topology shared by C3k2),
    so all DCA outputs are densely aggregated — the "Dense" in Dense Context-Aware. Mirrors the C3k2 signature:
    the fourth YAML argument (``dca2``, analogous to ``c3k``) stacks two DCA blocks per repeat for the deeper
    m/l/x scales.

    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of DCA blocks.
        dca2 (bool): Stack two DCA blocks per repeat (auto-enabled for m/l/x scales, like C3k2's ``c3k``).
        e (float): Hidden-channel expansion ratio.
        g (int): Unused, kept for C3k2 signature compatibility.
        shortcut (bool): Whether DCA blocks use residual connections.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        dca2: bool = False,
        e: float = 0.5,
        g: int = 1,
        shortcut: bool = True,
    ):
        """Initialize C3k2DCA with Dense Context-Aware inner blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(DCABlock(self.c, shortcut), DCABlock(self.c, shortcut))
            if dca2
            else DCABlock(self.c, shortcut)
            for _ in range(n)
        )
