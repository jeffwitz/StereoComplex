#!/usr/bin/env python3
"""Zernike vs CMO rigid-gauge removal (Figure 10 of the CMO paper).

After applying the Kabsch SE(3) that best aligns the Zernike rayfield
reconstruction onto the CMO 26p surface, plus a best-fit affine plane
removal on both, the 3-D Z residual collapses to the surface relief
(Case A in the manuscript).

The figure has 2 rows × 4 columns:

- Row 0: CMO Z plane-normalised, Zernike Z (after SE(3) + plane-norm),
  dZ raw, dZ plane-normalised.
- Row 1: dZ before SE(3) (dominated by the affine ramp), the fitted
  affine plane, the residual histogram before/after, and a textual
  case-A summary.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure10_zernike_cmo_rigid_removed/``. Emits both
PDF (paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

MANIFEST = Path("docs/assets/cmo_paper/figure10_zernike_cmo_rigid_removed/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "zernike_cmo_rigid_removed"

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def _resolve(manifest_root: Path, rel: str) -> Path:
    return (manifest_root / rel).resolve()


def _to_2d_canvas(values: np.ndarray, uL: np.ndarray, vL: np.ndarray,
                  image_size: tuple[int, int]) -> np.ndarray:
    h, w = image_size
    canvas = np.full((h, w), np.nan, dtype=np.float64)
    u = np.clip(uL.astype(int), 0, w - 1)
    v = np.clip(vL.astype(int), 0, h - 1)
    canvas[v, u] = values
    return canvas


def _apply_se3_to_grid(Z: np.ndarray, valid: np.ndarray,
                       R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply the Kabsch SE(3) to a ``(H, W)`` Z grid indexed by ``(j, i)``.

    The original Kabsch was fitted on points ``[j, i, Z]`` in image-pixel
    coordinates with a sign flip on the final Z (mirror about the image
    plane), so we reproduce that convention here.
    """
    out = np.full_like(Z, np.nan)
    H, W = Z.shape
    ys, xs = np.where(valid & np.isfinite(Z))
    pts = np.column_stack([xs, ys, Z[ys, xs]])
    transformed = pts @ R.T + t
    out[ys, xs] = -transformed[:, 2]
    return out


def _plane_normalise(Z: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(Z - plane, plane)`` where ``plane`` is the LSQ fit on ``Z``."""
    H, W = Z.shape
    ys, xs = np.where(valid & np.isfinite(Z))
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    a, b, c = np.linalg.lstsq(A, Z[ys, xs], rcond=None)[0]
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    plane = a * jj + b * ii + c
    return Z - plane, plane


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    corr = np.load(_resolve(manifest_root, manifest["correspondences"]))
    cmo = np.load(_resolve(manifest_root, manifest["cmo_reconstruction"]))
    zer = np.load(_resolve(manifest_root, manifest["zernike_reconstruction"]))
    rigid = json.loads(
        _resolve(manifest_root, manifest["rigid_comparison"]).read_text(encoding="utf-8")
    )

    image_size = tuple(int(x) for x in corr["image_size"])
    uL = np.asarray(corr["uL"], dtype=np.float64)
    vL = np.asarray(corr["vL"], dtype=np.float64)

    Z_cmo = _to_2d_canvas(cmo["Z"], uL, vL, image_size)
    Z_zer = _to_2d_canvas(zer["Z"], uL, vL, image_size)
    V_cmo = _to_2d_canvas(cmo["valid"].astype(float), uL, vL, image_size) > 0.5

    x0, x1, y0, y1 = manifest["roi_crop_px"]
    Zc = Z_cmo[y0:y1, x0:x1]
    Zz_raw = Z_zer[y0:y1, x0:x1]
    Vc = V_cmo[y0:y1, x0:x1]

    R = Rotation.from_rotvec(rigid["kabsch_se3"]["rotation_vec"]).as_matrix()
    t = np.asarray(rigid["kabsch_se3"]["translation"], dtype=np.float64)
    Zz_se3 = _apply_se3_to_grid(Zz_raw, Vc & np.isfinite(Zz_raw), R, t)

    valid2 = Vc & np.isfinite(Zc) & np.isfinite(Zz_se3)
    Zc_norm, _ = _plane_normalise(Zc, valid2)
    Zz_norm, Zz_plane = _plane_normalise(Zz_se3, valid2)

    dZ_raw = Zz_se3 - Zc
    dZ_norm = Zz_norm - Zc_norm
    dZ_before = Zz_raw - Zc

    z_p_lo, z_p_hi = np.nanpercentile(Zz_norm[valid2], manifest["z_percentile_clip"])
    p99 = manifest["dz_percentile_clip"]
    vdz_raw = float(np.nanpercentile(np.abs(dZ_raw[valid2]), p99))
    vdz_norm = float(np.nanpercentile(np.abs(dZ_norm[valid2]), p99))
    vdz_pre = float(np.nanpercentile(np.abs(dZ_before[valid2]), p99))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    im = axes[0, 0].imshow(Zc_norm, cmap="viridis", vmin=z_p_lo, vmax=z_p_hi)
    axes[0, 0].set_title(
        f"CMO 26p Z (plane-norm)\nstd={np.nanstd(Zc_norm[valid2]):.4f} mm"
    )
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(Zz_norm, cmap="viridis", vmin=z_p_lo, vmax=z_p_hi)
    axes[0, 1].set_title(
        f"Zernike 57p Z (SE3+plane-norm)\nstd={np.nanstd(Zz_norm[valid2]):.4f} mm"
    )
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(dZ_raw, cmap="RdBu_r", vmin=-vdz_raw, vmax=vdz_raw)
    axes[0, 2].set_title(
        f"dZ raw\nmed={np.nanmedian(np.abs(dZ_raw[valid2])):.3f} mm"
    )
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[0, 3].imshow(dZ_norm, cmap="RdBu_r", vmin=-vdz_norm, vmax=vdz_norm)
    axes[0, 3].set_title(
        f"dZ plane-norm\nmed={np.nanmedian(np.abs(dZ_norm[valid2])):.3f} mm"
    )
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = axes[1, 0].imshow(dZ_before, cmap="RdBu_r", vmin=-vdz_pre, vmax=vdz_pre)
    axes[1, 0].set_title(
        "dZ before SE(3)\n"
        f"R²={rigid['affine_plane']['before_se3']['r2']:.3f}"
    )
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(Zz_plane, cmap="viridis")
    axes[1, 1].set_title("Affine plane")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    dPr = np.abs(dZ_before[valid2])
    dPo = np.abs(dZ_norm[valid2])
    axes[1, 2].hist(dPr, bins=100, alpha=0.6, color="red",
                    label=f"before (med={np.median(dPr):.2f})")
    axes[1, 2].hist(dPo, bins=100, alpha=0.6, color="green",
                    label=f"after (med={np.median(dPo):.3f})")
    axes[1, 2].set_xlabel("3D residual [mm]")
    axes[1, 2].set_xlim(0, manifest["residual_histogram_xmax_mm"])
    axes[1, 2].legend(fontsize=8)

    scale_ratio = float(np.nanstd(Zz_norm[valid2]) / np.nanstd(Zc_norm[valid2]))
    summary = (
        f"Global frame change\n"
        f"{rigid['kabsch_se3']['rotation_angle_deg']:.1f}° rotation\n"
        f"Residual: {rigid['kabsch_se3']['median_dP_before_mm']:.1f} "
        f"-> {np.median(dPo):.2f} mm\n"
        f"Plane R²={rigid['affine_plane']['before_se3']['r2']:.3f}"
    )
    axes[1, 3].text(0.5, 0.8, "CASE A", ha="center", va="center",
                    fontsize=20, fontweight="bold", transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.5, 0.55, summary, ha="center", va="center",
                    fontsize=10, transform=axes[1, 3].transAxes)
    axes[1, 3].text(0.5, 0.15, f"Zern amp / CMO amp = {scale_ratio:.2f}x",
                    ha="center", va="center", fontsize=11,
                    fontweight="bold", color="darkblue",
                    transform=axes[1, 3].transAxes)
    axes[1, 3].axis("off")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    render(manifest, MANIFEST.parent, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
