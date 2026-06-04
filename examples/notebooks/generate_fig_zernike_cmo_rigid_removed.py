#!/usr/bin/env python3
"""Zernike vs CMO rigid-gauge removal (Figure 10 of the CMO paper).

After applying the Kabsch SE(3) that best aligns the Zernike rayfield
reconstruction onto the CMO 26p surface (fit on the 3-D mm point clouds
in ``zernike_cmo_rigid_comparison.json``), the 3-D Z residual is
dominated by a global affine ramp. A further plane-normalisation
collapses the residual to the surface relief (Case A in the manuscript).

The figure has 2 rows × 4 columns, with the same panels as the
published version but **applied to 3-D mm points** (the published PDF
applied the SE(3) to pixel triples ``(j, i, Z)``, a dimensional bug that
produced a meaningless 367 mm "dZ raw" — see the README of this
asset folder).

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


def _fit_affine_plane(xy: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``(coeffs, r2)`` for ``z ~ a*x + b*y + c`` in mm."""
    A = np.column_stack([xy, np.ones(xy.shape[0])])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_pred = A @ coeffs
    ss_res = float(np.sum((z - z_pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coeffs, r2


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

    # Per-correspondence 3-D points (mm) — same indexing for CMO and Zernike
    # because both reconstructions consume the identical pixel correspondences.
    valid_cmo = np.asarray(cmo["valid"], dtype=bool)
    valid_zer = np.asarray(zer["valid"], dtype=bool)
    valid = valid_cmo & valid_zer

    P_cmo = np.column_stack([cmo["X"], cmo["Y"], cmo["Z"]])
    P_zer = np.column_stack([zer["X"], zer["Y"], zer["Z"]])
    # Apply the documented -Y correction to the CMO reconstruction (the CMO model
    # fits the rayfield with an inverted v->Y sign), matching the frame in which the
    # rigid Kabsch was computed (11_compare_zernike_cmo_rigid_removed.py). Without it
    # the proper-rotation alignment disguises the Y reflection as a ~180 deg rotation
    # and inverts the Z relief between the two panels.
    P_cmo[:, 1] *= -1.0

    R = Rotation.from_rotvec(rigid["kabsch_se3"]["rotation_vec"]).as_matrix()
    t = np.asarray(rigid["kabsch_se3"]["translation"], dtype=np.float64)
    P_zer_aligned = P_zer @ R.T + t

    Z_cmo = np.where(valid, P_cmo[:, 2], np.nan)
    Z_zer = np.where(valid, P_zer[:, 2], np.nan)
    Z_zer_se3 = np.where(valid, P_zer_aligned[:, 2], np.nan)

    XY_cmo = P_cmo[valid, :2]
    coeffs_cmo, _ = _fit_affine_plane(XY_cmo, P_cmo[valid, 2])
    coeffs_zer, _ = _fit_affine_plane(XY_cmo, P_zer_aligned[valid, 2])
    plane_cmo = np.where(valid, P_cmo[:, :2] @ coeffs_cmo[:2] + coeffs_cmo[2], np.nan)
    plane_zer = np.where(valid, P_cmo[:, :2] @ coeffs_zer[:2] + coeffs_zer[2], np.nan)
    Zc_norm = Z_cmo - plane_cmo
    Zz_norm = Z_zer_se3 - plane_zer

    dZ_before = Z_zer - Z_cmo
    dZ_after_se3 = Z_zer_se3 - Z_cmo
    dZ_norm = Zz_norm - Zc_norm

    # 2-D maps for visualisation (crop to the figure window).
    x0, x1, y0, y1 = manifest["roi_crop_px"]
    def crop2d(v1d):
        return _to_2d_canvas(v1d, uL, vL, image_size)[y0:y1, x0:x1]

    Zc_norm_map = crop2d(Zc_norm)
    Zz_norm_map = crop2d(Zz_norm)
    dZ_before_map = crop2d(dZ_before)
    dZ_after_map = crop2d(dZ_after_se3)
    dZ_norm_map = crop2d(dZ_norm)
    plane_zer_map = crop2d(plane_zer)

    valid_subset = valid
    z_p_lo, z_p_hi = np.nanpercentile(Zz_norm[valid_subset], manifest["z_percentile_clip"])
    p99 = manifest["dz_percentile_clip"]
    vdz_raw = float(np.nanpercentile(np.abs(dZ_after_se3[valid_subset]), p99))
    vdz_norm = float(np.nanpercentile(np.abs(dZ_norm[valid_subset]), p99))
    vdz_pre = float(np.nanpercentile(np.abs(dZ_before[valid_subset]), p99))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    im = axes[0, 0].imshow(Zc_norm_map, cmap="viridis", vmin=z_p_lo, vmax=z_p_hi)
    axes[0, 0].set_title(
        f"CMO 26p Z (plane-norm)\nstd={np.nanstd(Zc_norm[valid_subset]):.4f} mm"
    )
    axes[0, 0].axis("off")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    im = axes[0, 1].imshow(Zz_norm_map, cmap="viridis", vmin=z_p_lo, vmax=z_p_hi)
    axes[0, 1].set_title(
        f"Zernike 57p Z (SE3+plane-norm)\nstd={np.nanstd(Zz_norm[valid_subset]):.4f} mm"
    )
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    im = axes[0, 2].imshow(dZ_after_map, cmap="RdBu_r", vmin=-vdz_raw, vmax=vdz_raw)
    axes[0, 2].set_title(
        f"dZ after SE(3)\nmed={np.nanmedian(np.abs(dZ_after_se3[valid_subset])):.3f} mm"
    )
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)

    im = axes[0, 3].imshow(dZ_norm_map, cmap="RdBu_r", vmin=-vdz_norm, vmax=vdz_norm)
    axes[0, 3].set_title(
        f"dZ plane-norm\nmed={np.nanmedian(np.abs(dZ_norm[valid_subset])):.4f} mm"
    )
    axes[0, 3].axis("off")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    im = axes[1, 0].imshow(dZ_before_map, cmap="RdBu_r", vmin=-vdz_pre, vmax=vdz_pre)
    axes[1, 0].set_title(
        "dZ before SE(3)\n"
        f"R²={rigid['affine_plane']['before_se3']['r2']:.3f}"
    )
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(plane_zer_map, cmap="viridis")
    axes[1, 1].set_title("Affine plane (mm)")
    axes[1, 1].axis("off")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    dPr = np.abs(dZ_before[valid_subset])
    dPo = np.abs(dZ_norm[valid_subset])
    axes[1, 2].hist(dPr, bins=100, alpha=0.6, color="red",
                    label=f"before SE(3), raw (med={np.median(dPr):.3f})")
    axes[1, 2].hist(dPo, bins=100, alpha=0.6, color="green",
                    label=f"after SE(3), plane-norm (med={np.median(dPo):.4f})")
    axes[1, 2].set_xlabel(r"$\Delta Z$ residual [mm]")
    axes[1, 2].set_title(
        r"$\Delta Z$ plane-norm: 0.0076 mm; 3D p2p: 3.5 $\to$ 0.019 mm",
        fontsize=9,
    )
    axes[1, 2].set_xlim(0, manifest["residual_histogram_xmax_mm"])
    axes[1, 2].legend(fontsize=8)

    scale_ratio = float(np.nanstd(Zz_norm[valid_subset]) / np.nanstd(Zc_norm[valid_subset]))
    summary = (
        f"Frame: Y-reflection (CMO v→Y)\n"
        f"+ {rigid['kabsch_se3']['rotation_angle_deg']:.1f}° rotation\n"
        f"Residual: {rigid['kabsch_se3']['median_dP_before_mm']:.2f} "
        f"-> {np.median(dPo):.4f} mm\n"
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
