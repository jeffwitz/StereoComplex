#!/usr/bin/env python3
"""Dense specimen reconstruction (Figure 9 of the CMO paper).

Two-row layout:

- Row 0: annotated left image with ROI, dense disparity field, valid mask;
- Row 1: CMO 26p Z map, Zernike O(2)+d(2) Z map, ray-gap histogram.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure9_specimen_reconstruction/``. Emits both
PDF (paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MANIFEST = Path("docs/assets/cmo_paper/figure9_specimen_reconstruction/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "specimen_reconstruction"

plt.rcParams.update({"font.family": "serif", "font.size": 10})


def _resolve(manifest_root: Path, rel: str) -> Path:
    return (manifest_root / rel).resolve()


def _to_2d_canvas(values: np.ndarray, uL: np.ndarray, vL: np.ndarray,
                  image_size: tuple[int, int]) -> np.ndarray:
    """Scatter a per-correspondence 1-D array onto the full image grid."""
    h, w = image_size
    canvas = np.full((h, w), np.nan, dtype=np.float64)
    u = np.clip(uL.astype(int), 0, w - 1)
    v = np.clip(vL.astype(int), 0, h - 1)
    canvas[v, u] = values
    return canvas


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    corr = np.load(_resolve(manifest_root, manifest["correspondences"]))
    cmo = np.load(_resolve(manifest_root, manifest["cmo_reconstruction"]))
    zer = np.load(_resolve(manifest_root, manifest["zernike_reconstruction"]))
    left_with_roi = plt.imread(_resolve(manifest_root, manifest["left_image_with_roi"]))

    image_size = tuple(int(x) for x in corr["image_size"])
    uL = np.asarray(corr["uL"], dtype=np.float64)
    vL = np.asarray(corr["vL"], dtype=np.float64)
    uR = np.asarray(corr["uR"], dtype=np.float64)
    vR = np.asarray(corr["vR"], dtype=np.float64)

    Z_cmo = _to_2d_canvas(cmo["Z"], uL, vL, image_size)
    Z_zer = _to_2d_canvas(zer["Z"], uL, vL, image_size)
    V_cmo = _to_2d_canvas(cmo["valid"].astype(float), uL, vL, image_size) > 0.5
    disparity = np.sqrt((uR - uL) ** 2 + (vR - vL) ** 2)
    disp_map = _to_2d_canvas(disparity, uL, vL, image_size)
    gap_map = _to_2d_canvas(cmo["gap"], uL, vL, image_size)

    x0, x1, y0, y1 = manifest["roi_crop_px"]
    Zc = Z_cmo[y0:y1, x0:x1]
    Zz = Z_zer[y0:y1, x0:x1]
    Vc = V_cmo[y0:y1, x0:x1]
    dmap = disp_map[y0:y1, x0:x1]
    gmap = gap_map[y0:y1, x0:x1]
    valid = Vc & np.isfinite(Zc) & np.isfinite(Zz)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(left_with_roi)
    axes[0, 0].set_title("Left image + ROI")
    axes[0, 0].axis("off")

    vmax_d = float(np.nanpercentile(dmap[valid], manifest["disparity_vmax_percentile"]))
    im = axes[0, 1].imshow(dmap, cmap="inferno", vmin=0, vmax=vmax_d)
    axes[0, 1].set_title("Disparity ||(U,V)|| [px]")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

    axes[0, 2].imshow(Vc, cmap="gray")
    axes[0, 2].set_title(f"Valid {Vc.mean() * 100:.1f}%")
    axes[0, 2].axis("off")

    z_lo, z_hi = np.nanpercentile(
        np.concatenate([Zc[valid], Zz[valid]]),
        manifest["z_percentile_clip"],
    )
    im = axes[1, 0].imshow(Zc, cmap="viridis", vmin=z_lo, vmax=z_hi)
    axes[1, 0].set_title("CMO 26p Z [mm]")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    im = axes[1, 1].imshow(Zz, cmap="viridis", vmin=z_lo, vmax=z_hi)
    axes[1, 1].set_title("Zernike O(2)+d(2) Z [mm]")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046)

    gap_vals = gmap[Vc]
    gap_vals = gap_vals[np.isfinite(gap_vals) & (gap_vals < 1.0)]
    xmax = float(np.percentile(gap_vals, manifest["gap_xmax_percentile"])) * 1.2
    axes[1, 2].hist(gap_vals, bins=100, color="steelblue", alpha=0.85)
    med = float(np.median(gap_vals))
    axes[1, 2].axvline(med, color="red", linestyle="--", label=f"med={med:.4f}")
    axes[1, 2].set_xlabel("Ray gap [mm]")
    axes[1, 2].set_xlim(0, xmax)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title("Ray gap histogram")

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
