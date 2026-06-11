#!/usr/bin/env python3
"""Manual-landmark registration of the profilometry scan onto the CMO reconstruction.

Automated cross-modal registration of the noisy DIS-stereo relief onto the clean
profilometry relief plateaus (no reliable repeatable features). This tool lets a
human supply the alignment: click the same N letter landmarks (corners/junctions
of the "EN" lettering) first on the profilometry image, then on the CMO
reconstruction, in the **same order**. It fits a homography
(profilo -> reconstruction pixel canvas), reports the relief correlation it
achieves, saves the transform, and writes an overlay for visual checking.

Run it locally with an interactive backend (so the click windows open)::

    rtk .venv/bin/python examples/notebooks/register_profilo_landmarks.py \
        --profilo profilo/coin_Profilo.npy

Tips: pick 6-10 well-spread points (the four outer corners of E and N, the
mid-bar junctions). Press <enter> when done with each image. Re-run to redo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path("docs/assets/pycaso_real_data")


def _plane_normalise(z: np.ndarray) -> np.ndarray:
    """Subtract the best-fit tilt plane from a 2-D height map (ignoring NaNs)."""
    h, w = z.shape
    ys, xs = np.mgrid[0:h, 0:w]
    m = np.isfinite(z)
    design = np.column_stack([xs[m], ys[m], np.ones(int(m.sum()))])
    coeffs, *_ = np.linalg.lstsq(design, z[m], rcond=None)
    return z - (coeffs[0] * xs + coeffs[1] * ys + coeffs[2])


def _cmo_relief_canvas() -> np.ndarray:
    """Rasterise the committed CMO 26p specimen relief onto the ROI canvas."""
    corr = np.load(ASSETS / "specimen_correspondences.npz")
    u = corr["uL"].astype(int).clip(0, 2047)
    v = corr["vL"].astype(int).clip(0, 2047)
    rec = np.load(ASSETS / "specimen_reconstruction_cmo26.npz")
    valid = np.asarray(rec["valid"], dtype=bool)
    canvas = np.full((2048, 2048), np.nan)
    canvas[v[valid], u[valid]] = rec["Z"][valid]
    return _plane_normalise(canvas[300:1700, 300:1700])  # ROI [300, 1700)


def _collect(image: np.ndarray, title: str) -> np.ndarray:
    """Show a height map and return the clicked (x, y) points as an array."""
    lo, hi = np.nanpercentile(image[np.isfinite(image)], [2, 98])
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(image, cmap="viridis", vmin=lo, vmax=hi)
    ax.set_title(title)
    pts = plt.ginput(n=-1, timeout=0)
    plt.close(fig)
    return np.asarray(pts, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profilo", default="profilo/coin_Profilo.npy")
    parser.add_argument("--profilo-rows", type=int, default=1850,
                        help="keep the top rows of the profilo (coin region)")
    parser.add_argument("--out", default="/tmp/profilo_landmarks.npz")
    args = parser.parse_args()

    prof = _plane_normalise(np.load(args.profilo)[: args.profilo_rows, :].astype(float))
    recon = _cmo_relief_canvas()

    print("Click landmarks on the PROFILO, then <enter>...")
    src = _collect(prof, "PROFILO — click EN landmarks, then <enter>")
    print(f"  {len(src)} points")
    print("Click the SAME landmarks, SAME order, on the CMO RECONSTRUCTION...")
    dst = _collect(recon, "CMO RECONSTRUCTION — same landmarks, same order")
    print(f"  {len(dst)} points")

    if len(src) != len(dst) or len(src) < 4:
        raise SystemExit("need >=4 matched points and equal counts on both images")

    import cv2

    homography, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    warped = cv2.warpPerspective(prof.astype(np.float32), homography, recon.shape[::-1])
    mask = np.isfinite(warped) & np.isfinite(recon) & (warped != 0)
    cc = float(np.corrcoef(warped[mask], recon[mask])[0, 1])
    print(f"\nlandmark homography: relief correlation cc = {cc:.3f} "
          f"(automated plateau ~0.55)")

    np.savez(args.out, homography=homography, src=src, dst=dst, cc=cc)
    print(f"saved transform -> {args.out}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    lo, hi = np.nanpercentile(recon[np.isfinite(recon)], [2, 98])
    axes[0].imshow(recon, cmap="viridis", vmin=lo, vmax=hi)
    axes[0].set_title("CMO reconstruction")
    axes[1].imshow(warped, cmap="viridis", vmin=lo, vmax=hi)
    axes[1].set_title(f"profilo warped onto it (cc={cc:.3f})")
    for ax in axes:
        ax.axis("off")
    fig.savefig("/tmp/profilo_landmark_overlay.png", dpi=110, bbox_inches="tight")
    print("overlay -> /tmp/profilo_landmark_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
