#!/usr/bin/env python3
"""3-D sub-pupil reconstruction from the Zernike rayfield (Figure 4).

Fits (or reads from cache) the constrained Zernike rayfield at the
order selected in §3.1 of the paper (`O=0, d=2`, 57 modes), then reads
the centre-pixel sub-pupils `O_L`, `O_R` of both channels. The figure
draws those sub-pupils in 3-D, the chief rays toward the working
plane, the baseline, and the working-plane intersection points.

This matches the caption of Figure 4 in `manuscript.tex` (line 252),
which explicitly states that the sub-pupils are read from the
**Zernike rayfield**, not from the CMO 26-parameter model. The
descriptors quoted in the paper (`b ≈ 24.9 mm`, `WD ≈ 64.7 mm`,
half-angle ≈ 11°) match the `O=0, d=2` row of
`zernike_order_sweep.json`.

Two-mode generator:

- default — reads the cached `zernike_rayfield_canonical.npz` and
  re-renders the figure;
- ``--recompute`` — re-fits the rayfield (~15 s), refreshes the cache,
  then renders.

Emits both PDF (paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "src")

MANIFEST = Path("docs/assets/cmo_paper/figure4_subpupil_3d/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "subpupil_3d"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def _fit_rayfield(manifest: dict, manifest_root: Path) -> dict:
    """Re-run the constrained Zernike rayfield fit and return its coefficients."""
    import cv2  # noqa: PLC0415
    from stereocomplex.benchmarks.charuco_observation_simulator import (  # noqa: PLC0415
        CharucoObservationSet,
    )
    from stereocomplex.benchmarks.rayfield_from_observations import (  # noqa: PLC0415
        fit_constrained_zernike_rayfield,
    )

    state = np.load((manifest_root / manifest["intermediate_state"]).resolve())
    W, H = (int(x) for x in state["image_size"])
    fx_ref = float(manifest["fx_reference_px"])
    cx, cy = manifest["principal_point_px"]
    K = np.array([[fx_ref, 0, cx], [0, fx_ref, cy], [0, 0, 1]], dtype=np.float64)
    n_frames = int(manifest["n_frames"])
    n_corners = int(manifest["n_corners"])

    left_pixels = state["left_pixels"]
    right_pixels = state["right_pixels"]
    obj_pts = state["obj_pts"]

    print(f"  fitting Zernike rayfield (max_order_d={manifest['max_order_d']}, "
          f"max_nfev={manifest['max_nfev']})...")
    t0 = time.time()

    rvecs, tvecs = [], []
    for pi in range(n_frames):
        ok, rv, tv = cv2.solvePnP(
            obj_pts.astype(np.float32),
            left_pixels[pi].astype(np.float32),
            K.astype(np.float32),
            np.zeros(5, dtype=np.float32),
        )
        rvecs.append(rv.ravel().astype(np.float64) if ok else np.zeros(3))
        tvecs.append(tv.ravel().astype(np.float64) if ok else np.array([0.0, 0.0, 65.0]))

    obs = CharucoObservationSet(
        object_points_mm=obj_pts,
        pose_rvecs=np.array(rvecs),
        pose_tvecs=np.array(tvecs),
        left_pixels=[left_pixels[i] for i in range(n_frames)],
        right_pixels=[right_pixels[i] for i in range(n_frames)],
        point_indices=[np.arange(n_corners, dtype=int) for _ in range(n_frames)],
        noise_std_px=0.0,
        image_size=(W, H),
    )
    lf, rf, *_ = fit_constrained_zernike_rayfield(
        obs, image_size=(W, H), K_left=K, K_right=K.copy(),
        max_order_d=int(manifest["max_order_d"]),
        max_nfev=int(manifest["max_nfev"]),
        origin_reg_weight=0.0,
    )
    print(f"    done in {time.time() - t0:.1f}s")

    return {
        "image_size": np.array([W, H], dtype=np.int64),
        "K": K,
        "left_origin_coeffs": np.asarray(lf.origin_coeffs, dtype=np.float64),
        "left_direction_coeffs": np.asarray(lf.direction_coeffs, dtype=np.float64),
        "right_origin_coeffs": np.asarray(rf.origin_coeffs, dtype=np.float64),
        "right_direction_coeffs": np.asarray(rf.direction_coeffs, dtype=np.float64),
        "max_order_d": np.int64(manifest["max_order_d"]),
    }


def _save_cache(data: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"  cached {cache_path}")


def _load_cache(cache_path: Path) -> dict:
    npz = np.load(cache_path)
    return {k: npz[k] for k in npz.files}


def _build_rayfields(data: dict):
    from stereocomplex.rayfields.zernike_origin_field import (  # noqa: PLC0415
        ZernikeOriginFieldConfig,
        ZernikeRayField,
        ZernikeRayFieldCoefficients,
    )
    image_size = tuple(int(x) for x in data["image_size"])
    K = np.asarray(data["K"], dtype=np.float64)
    config = ZernikeOriginFieldConfig(image_size=image_size, max_order=int(data["max_order_d"]))
    lf = ZernikeRayField(K=K, config=config, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=np.asarray(data["left_origin_coeffs"], dtype=np.float64),
        direction_coeffs=np.asarray(data["left_direction_coeffs"], dtype=np.float64),
    ))
    rf = ZernikeRayField(K=K, config=config, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=np.asarray(data["right_origin_coeffs"], dtype=np.float64),
        direction_coeffs=np.asarray(data["right_direction_coeffs"], dtype=np.float64),
    ))
    return lf, rf, image_size


def _closest_point_midpoint(O1, d1, O2, d2):  # noqa: E741
    n = np.cross(d1, d2)
    nn = float(np.dot(n, n))
    if nn < 1e-12:
        return None
    w = O2 - O1
    t1 = float(np.dot(np.cross(d2, n), w) / nn)
    t2 = float(np.dot(np.cross(d1, n), w) / nn)
    return (O1 + t1 * d1 + O2 + t2 * d2) / 2.0


def render(manifest: dict, data: dict, out_dir: Path) -> None:
    lf, rf, image_size = _build_rayfields(data)
    W, H = image_size

    OL_arr, dL_arr = lf.ray(np.array([W / 2]), np.array([H / 2]))
    OR_arr, dR_arr = rf.ray(np.array([W / 2]), np.array([H / 2]))
    OcL, dcL = OL_arr[0], dL_arr[0]
    OcR, dcR = OR_arr[0], dR_arr[0]

    b = float(np.linalg.norm(OcR - OcL))
    cos_a = float(np.clip(dcL @ dcR, -1.0, 1.0))
    full_angle_deg = float(np.degrees(np.arccos(cos_a)))
    half_angle_deg = full_angle_deg / 2.0
    bmid = (OcL + OcR) / 2.0
    P_cross = _closest_point_midpoint(OcL, dcL, OcR, dcR)
    WD_geom = float(np.linalg.norm(P_cross - bmid)) if P_cross is not None else float("nan")

    # Paper-consistent WD from mean pose Z (not the geometric crossing distance)
    is_path = Path(manifest["intermediate_state"])
    if not is_path.is_absolute():
        is_path = (MANIFEST.parent / is_path).resolve()
    chk = np.load(str(is_path))
    WD_paper = float(np.mean(np.abs(np.asarray(chk["opt_t"], dtype=np.float64)[:, 2])))

    print(f"  centre-pixel descriptors (Zernike rayfield, max_order_d="
          f"{int(data['max_order_d'])}):")
    print(f"    OL = ({OcL[0]:+.3f}, {OcL[1]:+.3f}, {OcL[2]:+.3f}) mm")
    print(f"    OR = ({OcR[0]:+.3f}, {OcR[1]:+.3f}, {OcR[2]:+.3f}) mm")
    print(f"    baseline b = {b:.3f} mm")
    print(f"    working distance WD (geom) = {WD_geom:.3f} mm")
    print(f"    working distance WD (paper, from mean pose Z) = {WD_paper:.3f} mm")
    print(f"    full convergence angle = {full_angle_deg:.3f} deg "
          f"(half = {half_angle_deg:.3f} deg)")

    fig = plt.figure(figsize=tuple(manifest["figure_size"]))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*OcL, c="#2a55c7", s=200, marker="o",
               edgecolors="black", linewidth=0.5,
               label=f"O$_L$ ({OcL[0]:+.2f}, {OcL[1]:+.2f}, {OcL[2]:+.2f}) mm")
    ax.scatter(*OcR, c="#c4332a", s=200, marker="o",
               edgecolors="black", linewidth=0.5,
               label=f"O$_R$ ({OcR[0]:+.2f}, {OcR[1]:+.2f}, {OcR[2]:+.2f}) mm")

    ray_len = float(manifest["chief_ray_length_mm"])
    ray_L_end = OcL + ray_len * dcL
    ray_R_end = OcR + ray_len * dcR
    ax.plot([OcL[0], ray_L_end[0]], [OcL[1], ray_L_end[1]], [OcL[2], ray_L_end[2]],
            color="#2a55c7", lw=1.5, alpha=0.7, label="Chief ray L")
    ax.plot([OcR[0], ray_R_end[0]], [OcR[1], ray_R_end[1]], [OcR[2], ray_R_end[2]],
            color="#c4332a", lw=1.5, alpha=0.7, label="Chief ray R")

    if P_cross is not None:
        ax.scatter(*P_cross, c="green", s=140, marker="x", linewidth=1.5,
                   label=f"Chief-ray crossing (WD={WD_paper:.2f} mm)")

    ax.plot([OcL[0], OcR[0]], [OcL[1], OcR[1]], [OcL[2], OcR[2]],
            "k--", lw=2, alpha=0.6)
    ax.text(bmid[0], bmid[1], bmid[2] - 2, f"b = {b:.2f} mm",
            fontsize=12, ha="center", fontweight="bold")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        "3D sub-pupil reconstruction from Zernike rayfield\n"
        f"b={b:.2f} mm, WD={WD_paper:.2f} mm, half-angle "
        f"$\\theta_{{1/2}}$={half_angle_deg:.2f}°"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.view_init(elev=manifest["view_init"]["elev"],
                 azim=manifest["view_init"]["azim"])

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="re-fit the Zernike rayfield (~15 s) and refresh the cache")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_root = MANIFEST.parent
    cache_path = manifest_root / manifest["cache_file"]

    if args.recompute or not cache_path.is_file():
        data = _fit_rayfield(manifest, manifest_root)
        _save_cache(data, cache_path)
    else:
        print(f"using cached rayfield: {cache_path}")
        data = _load_cache(cache_path)

    render(manifest, data, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
