# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Classical robust (truncated-statistics) CFAR reference implementation for SAR image chips.

This is the image-domain counterpart of the differentiable ``RobustCFARGate`` used inside CFAR-YOLOv11
(ultralytics/nn/modules/cfar.py). It is provided for analysis and paper figures: it visualises the clutter
statistics, the CFAR detection statistic z = (x - mu)/sigma and the resulting candidate mask on real SAR chips,
demonstrating why a learnable, feature-space robust CFAR is a sound inductive bias for dark-vessel detection.

Algorithm (two-pass truncated-statistics CA-CFAR, cf. Tao et al., IEEE TGRS 2016):
    1. Estimate local clutter mean/std over a background window minus a guard window (hollow annulus).
    2. Censor pixels above mu0 + t_trunc * sigma0 (removes interfering targets from the clutter estimate).
    3. Re-estimate clutter statistics from the censored image and threshold z >= tau.

Example:
    python research/cfar_demo.py --image chip.jpg --out cfar_fig.png --tau 5.0
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np


def _annulus_stats(img: np.ndarray, k_bg: int, k_guard: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean/std over the hollow annulus between the background and guard windows (box filters)."""
    nb, ng = k_bg * k_bg, k_guard * k_guard
    mean_bg = cv2.boxFilter(img, -1, (k_bg, k_bg), borderType=cv2.BORDER_REFLECT)
    mean_gd = cv2.boxFilter(img, -1, (k_guard, k_guard), borderType=cv2.BORDER_REFLECT)
    ex2_bg = cv2.boxFilter(img * img, -1, (k_bg, k_bg), borderType=cv2.BORDER_REFLECT)
    ex2_gd = cv2.boxFilter(img * img, -1, (k_guard, k_guard), borderType=cv2.BORDER_REFLECT)
    mu = (mean_bg * nb - mean_gd * ng) / (nb - ng)
    ex2 = (ex2_bg * nb - ex2_gd * ng) / (nb - ng)
    var = np.clip(ex2 - mu * mu, 0.0, None)
    return mu, np.sqrt(var + 1e-6)


def robust_cfar(
    img: np.ndarray, k_bg: int = 41, k_guard: int = 21, t_trunc: float = 2.0, tau: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Run two-pass truncated-statistics CFAR.

    Args:
        img: Grayscale SAR chip, any dtype (converted to float32).
        k_bg: Background window size (odd).
        k_guard: Guard window size (odd, < k_bg).
        t_trunc: Truncation depth in clutter standard deviations.
        tau: Detection threshold on the z statistic.

    Returns:
        z: CFAR detection statistic map (float32).
        mask: Binary candidate mask (uint8, {0, 255}).
    """
    x = img.astype(np.float32)
    mu0, sig0 = _annulus_stats(x, k_bg, k_guard)  # pass 1
    xt = np.minimum(x, mu0 + t_trunc * sig0)  # censor interfering targets
    mu, sig = _annulus_stats(xt, k_bg, k_guard)  # pass 2 on truncated image
    z = (x - mu) / sig
    mask = ((z >= tau) * 255).astype(np.uint8)
    return z, mask


def main():
    """CLI entry point: save a 3-panel figure (input | z statistic | candidate overlay)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="input SAR chip (grayscale)")
    ap.add_argument("--out", default="cfar_demo.png", help="output figure path")
    ap.add_argument("--k-bg", type=int, default=41)
    ap.add_argument("--k-guard", type=int, default=21)
    ap.add_argument("--t-trunc", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=5.0)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"could not read {args.image}"
    z, mask = robust_cfar(img, args.k_bg, args.k_guard, args.t_trunc, args.tau)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    overlay[mask > 0] = (255, 64, 64)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (im, title, kw) in zip(
        axes,
        [
            (img, "SAR chip", {"cmap": "gray"}),
            (np.clip(z, 0, 2 * args.tau), "robust CFAR statistic z", {"cmap": "inferno"}),
            (overlay, f"candidates (z ≥ {args.tau})", {}),
        ],
    ):
        ax.imshow(im, **kw)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
