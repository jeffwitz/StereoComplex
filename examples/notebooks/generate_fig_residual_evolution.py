#!/usr/bin/env python3
"""Residual evolution across model stages (Figure 6 of the CMO paper).

Three-panel direction-error heatmap (Δd in degrees) between the Zernike
rayfield reference and each candidate model on a 31×31 pixel grid:
Perspective CMO → Telecentric CMO → CMO+SE(3).

Two-mode generator:

- default: reads the cached 31×31 maps from
  ``docs/assets/cmo_paper/figure6_residual_evolution/residual_evolution_data.npz``
  and re-renders the figure (~1 s);
- ``--recompute``: re-runs the Zernike rayfield BA, the Telecentric CMO
  fit, and the CMO+SE(3) fit (~2–3 min), writes the cache, then renders.

Emits both PDF (paper) and PNG (docs) in one run.
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

MANIFEST = Path("docs/assets/cmo_paper/figure6_residual_evolution/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "residual_evolution"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def _compute(manifest: dict, manifest_root: Path) -> dict:
    """Run the full BA pipeline to produce the 31×31 direction-error maps."""
    import cv2  # noqa: PLC0415  (only needed in the slow path)
    from scipy.optimize import least_squares  # noqa: PLC0415
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    from stereocomplex.benchmarks.charuco_observation_simulator import (  # noqa: PLC0415
        CharucoObservationSet,
    )
    from stereocomplex.benchmarks.rayfield_from_observations import (  # noqa: PLC0415
        fit_constrained_zernike_rayfield,
    )
    from stereocomplex.physics.cmo_physical import (  # noqa: PLC0415
        CMOTelecentricStereoModel,
        _normalize,
    )

    inter = np.load((manifest_root / manifest["intermediate_state"]).resolve())
    W, H = (int(x) for x in inter["image_size"])
    FX = float(inter["FX"])
    K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)
    n_frames = 10
    grid_size = int(manifest["grid_size"])
    z_planes = [float(z) for z in manifest["z_planes_mm"]]
    max_nfev = int(manifest["max_nfev_per_stage"])

    left_pixels = inter["left_pixels"]
    right_pixels = inter["right_pixels"]
    obj_pts = inter["obj_pts"]
    opt_t = inter["opt_t"]

    print("  Stage 0: Zernike rayfield fit...")
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
        point_indices=[np.arange(165, dtype=int) for _ in range(n_frames)],
        noise_std_px=0.0,
        image_size=(W, H),
    )
    lf, rf, *_ = fit_constrained_zernike_rayfield(
        obs, image_size=(W, H), K_left=K, K_right=K.copy(),
        max_order_d=2, max_nfev=max_nfev, origin_reg_weight=0.0,
    )
    print(f"    done in {time.time() - t0:.1f}s")

    u_grid, v_grid = np.meshgrid(
        np.linspace(0, W - 1, grid_size), np.linspace(0, H - 1, grid_size)
    )
    uf, vf = u_grid.ravel(), v_grid.ravel()
    OzL, dzL = lf.ray(uf, vf)

    OcL, _ = lf.ray(np.array([1024.0]), np.array([1024.0]))
    OcR, _ = rf.ray(np.array([1024.0]), np.array([1024.0]))
    b_est = float(np.linalg.norm(OcR[0] - OcL[0]))
    WD_est = float(np.mean([float(opt_t[i][2]) for i in range(n_frames)]))
    z_p = float((abs(OcL[0, 2]) + abs(OcR[0, 2])) / 2.0)
    f_obj_est = WD_est - z_p

    def angle_map(d_model: np.ndarray) -> np.ndarray:
        dot = np.clip(np.sum(dzL * d_model, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(dot)).reshape(grid_size, grid_size)

    print("  Stage 1: Perspective CMO...")
    pp = 0.0055 / f_obj_est
    dL_persp = _normalize(np.column_stack([
        (uf - 1024) * pp - b_est / (2 * f_obj_est),
        (vf - 1024) * pp,
        np.ones_like(uf),
    ]))
    r1 = angle_map(dL_persp)

    print("  Stage 2: Telecentric CMO fit...")
    theta_fixed = float(np.arctan2(b_est / 2, f_obj_est))
    x0_tel = np.array([
        f_obj_est, WD_est, b_est, 1024.0, 1024.0, f_obj_est,
        theta_fixed, lf.ray(np.array([1024.0]), np.array([1024.0]))[1][0, 1],
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float64)
    lo_tel = np.array([1., 1., 0., 0., 0., 20., 0., -0.3,
                       -10., -10., -10., -10., -10., -10.])
    hi_tel = np.array([500., 1000., 200., 2048., 2048., 200., 0.5, 0.3,
                       10., 10., 10., 10., 10., 10.])

    def res_tel(x):
        m = CMOTelecentricStereoModel.from_parameter_vector(
            x, pixel_pitch_mm=0.0055, image_size=(W, H)
        )
        OL, dL = m.ray(uf, vf, "left")
        OR, dR = m.ray(uf, vf, "right")
        OzR, dzR = rf.ray(uf, vf)
        blocks = []
        for z in z_planes:
            for O_a, d_a, O_r, d_r in [(OL, dL, OzL, dzL), (OR, dR, OzR, dzR)]:
                tz_ref = (z - O_r[:, 2]) / d_r[:, 2]
                P_ref = O_r + tz_ref[:, None] * d_r
                tz_mod = (z - O_a[:, 2]) / d_a[:, 2]
                P_mod = O_a + tz_mod[:, None] * d_a
                blocks.append((P_ref - P_mod).reshape(-1))
        return np.concatenate(blocks)

    sol_tel = least_squares(
        res_tel, x0=x0_tel, bounds=(lo_tel, hi_tel),
        loss="huber", max_nfev=max_nfev, xtol=1e-10,
    )
    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        sol_tel.x[:14], pixel_pitch_mm=0.0055, image_size=(W, H)
    )
    _, dL_tel = m_tel.ray(uf, vf, "left")
    r2 = angle_map(dL_tel)

    print("  Stage 3: CMO + SE(3) fit...")
    def apply_se3(O, d, rv, t):  # noqa: E741
        R = Rotation.from_rotvec(rv).as_matrix()
        return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)

    arm_lo = np.concatenate([np.full(3, -0.08), np.full(3, -3.0),
                             np.full(3, -0.08), np.full(3, -3.0)])
    arm_hi = np.concatenate([np.full(3, 0.08), np.full(3, 3.0),
                             np.full(3, 0.08), np.full(3, 3.0)])

    def res_se3(x):
        m = CMOTelecentricStereoModel.from_parameter_vector(
            x[:14], pixel_pitch_mm=0.0055, image_size=(W, H)
        )
        OL, dL = m.ray(uf, vf, "left")
        OR, dR = m.ray(uf, vf, "right")
        OL_a, dL_a = apply_se3(OL, dL, x[14:17], x[17:20])
        OR_a, dR_a = apply_se3(OR, dR, x[20:23], x[23:26])
        OzR, dzR = rf.ray(uf, vf)
        blocks = []
        for z in z_planes:
            for O_a, d_a, O_r, d_r in [(OL_a, dL_a, OzL, dzL), (OR_a, dR_a, OzR, dzR)]:
                tz_ref = (z - O_r[:, 2]) / d_r[:, 2]
                P_ref = O_r + tz_ref[:, None] * d_r
                tz_mod = (z - O_a[:, 2]) / d_a[:, 2]
                P_mod = O_a + tz_mod[:, None] * d_a
                blocks.append((P_ref - P_mod).reshape(-1))
        return np.concatenate(blocks)

    sol_se3 = least_squares(
        res_se3,
        x0=np.concatenate([sol_tel.x[:14], np.zeros(12)]),
        bounds=(np.concatenate([lo_tel, arm_lo]),
                np.concatenate([hi_tel, arm_hi])),
        loss="huber", max_nfev=max_nfev, xtol=1e-10,
    )
    m_se3 = CMOTelecentricStereoModel.from_parameter_vector(
        sol_se3.x[:14], pixel_pitch_mm=0.0055, image_size=(W, H)
    )
    OL_s, dL_s = m_se3.ray(uf, vf, "left")
    _, dL_se3 = apply_se3(OL_s, dL_s, sol_se3.x[14:17], sol_se3.x[17:20])
    r3 = angle_map(dL_se3)

    return {
        "W": W, "H": H,
        "grid_size": grid_size,
        "perspective_deg": r1,
        "telecentric_deg": r2,
        "cmo_se3_deg": r3,
        "rms_perspective_deg": float(np.sqrt(np.mean(r1**2))),
        "rms_telecentric_deg": float(np.sqrt(np.mean(r2**2))),
        "rms_cmo_se3_deg": float(np.sqrt(np.mean(r3**2))),
    }


def _plot(data: dict, out_dir: Path) -> None:
    W = int(data["W"])
    H = int(data["H"])
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    panels = [
        (axes[0], data["perspective_deg"], "(a) Perspective CMO",
         0, 30, data["rms_perspective_deg"], ".1f", "°"),
        (axes[1], data["telecentric_deg"], "(b) Telecentric CMO",
         0, 0.5, data["rms_telecentric_deg"], ".3f", "°"),
        (axes[2], data["cmo_se3_deg"], "(c) CMO + SE(3)",
         0, 0.01, data["rms_cmo_se3_deg"], ".3f", "°"),
    ]
    for ax, arr, title, vmin, vmax, rms, fmt, unit in panels:
        im = ax.imshow(arr, origin="lower", cmap="hot",
                       vmin=vmin, vmax=vmax,
                       extent=[0, W, 0, H], aspect="auto")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Direction error (deg)", fontsize=10)
        ax.text(0.98, 0.02, f"RMS = {rms:{fmt}}{unit}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle(
        r"Residual-guided model construction: $\Delta d$ (deg) vs Zernike rayfield",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def _load_cache(cache_path: Path) -> dict:
    npz = np.load(cache_path)
    return {k: npz[k].item() if npz[k].ndim == 0 else npz[k] for k in npz.files}


def _save_cache(data: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **data)
    print(f"  cached {cache_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                        help="force a full BA pipeline rerun (~2–3 min)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_root = MANIFEST.parent
    cache_path = manifest_root / manifest["cache_file"]

    if args.recompute or not cache_path.is_file():
        data = _compute(manifest, manifest_root)
        _save_cache(data, cache_path)
    else:
        print(f"using cached data: {cache_path}")
        data = _load_cache(cache_path)

    _plot(data, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
