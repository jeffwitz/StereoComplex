#!/usr/bin/env python3
"""Sensitivity of the Schur coupling norm c to parameter scaling choices.

Recomputes c = coupling_norm for the real Pycaso 26-parameter configuration
under several scaling strategies, to assess how strongly the reported value
depends on the (arbitrary) choice of parameter scales.

Produces ``docs/assets/pycaso_real_data/schur_ba/coupling_sensitivity.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from stereocomplex.optical_ba import (
    PycasoCMOObservations,
    build_fisher_blocks,
    default_parameter_scales,
    diagnose_schur_modes,
    point_to_ray_residuals_cmo_se3,
)

PIXEL_PITCH_MM = 0.0055

# ── parameter groups (indices into the 26-vector) ──────────────────────
GROUP_FOCAL = [0, 1, 2, 5]            # f_obj, WD, b, f_angular
GROUP_PRINCIPAL = [3, 4]               # cx, cy
GROUP_ANGLE = [6, 7]                   # theta_conv, dy_common
GROUP_SHEAR = list(range(8, 14))       # s_x_L/R, s_y_L/R, rho_x, rho_y
GROUP_ROTATION = [14, 15, 16, 20, 21, 22]  # rv_L/R_{x,y,z}
GROUP_TRANSLATION = [17, 18, 19, 23, 24, 25]  # t_L/R_{x,y,z}

GROUPS = {
    "focal_geometric": GROUP_FOCAL,
    "principal_point": GROUP_PRINCIPAL,
    "angles": GROUP_ANGLE,
    "shear": GROUP_SHEAR,
    "se3_rotations": GROUP_ROTATION,
    "se3_translations": GROUP_TRANSLATION,
}


def _pack_poses(opt_R: np.ndarray, opt_t: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    n = opt_R.shape[0]
    out = np.empty(6 * n, dtype=np.float64)
    for i in range(n):
        rv = Rotation.from_matrix(opt_R[i]).as_rotvec()
        out[6 * i:6 * i + 3] = rv
        out[6 * i + 3:6 * i + 6] = opt_t[i]
    return out


def compute_coupling(
    obs, theta0, pose0, theta_scales, pose_scales,
    fd_method="central", rel_step=1e-6, damping_pose=1e-8, weak_threshold=1e-3,
) -> float:
    def _res(x): return point_to_ray_residuals_cmo_se3(x, obs)
    fisher = build_fisher_blocks(
        residual_fun=_res, theta0=theta0, pose0=pose0,
        theta_scales=theta_scales, pose_scales=pose_scales,
        rel_step=rel_step, method=fd_method,
    )
    diag = diagnose_schur_modes(
        fisher.I_tt, fisher.I_tp, fisher.I_pp,
        weak_threshold=weak_threshold, damping_pose=damping_pose,
    )
    return float(diag.coupling_norm)


def main() -> int:
    input_path = Path("docs/assets/pycaso_real_data/intermediate_state.npz")
    data = np.load(input_path, allow_pickle=True)
    obs = PycasoCMOObservations(
        obj_pts=np.asarray(data["obj_pts"], dtype=np.float64),
        left_pixels=np.asarray(data["left_pixels"], dtype=np.float64),
        right_pixels=np.asarray(data["right_pixels"], dtype=np.float64),
        image_size=tuple(int(x) for x in data["image_size"]),
        pixel_pitch_mm=PIXEL_PITCH_MM,
    )
    theta0 = np.asarray(data["x_26p"], dtype=np.float64)
    opt_R = np.asarray(data["opt_R"], dtype=np.float64)
    opt_t = np.asarray(data["opt_t"], dtype=np.float64)
    pose0 = _pack_poses(opt_R, opt_t)
    n_frames = obs.n_frames

    base_theta, base_pose = default_parameter_scales(n_frames)

    # ── baseline ──────────────────────────────────────────────────────
    print("Baseline...")
    c_base = compute_coupling(obs, theta0, pose0, base_theta, base_pose)
    print(f"  c(baseline) = {c_base:.6f}")

    results = {
        "description": "Schur coupling norm sensitivity to parameter scaling",
        "baseline": {"c": c_base, "theta_scales": base_theta.tolist()},
        "sweeps": {},
    }

    # ── per-group sweeps (±1 order of magnitude) ─────────────────────
    for group_name, indices in GROUPS.items():
        group_results = {}
        for factor_label, factor in [("×0.1", 0.1), ("×1.0", 1.0), ("×10", 10.0)]:
            if factor == 1.0:
                # Already computed as baseline for this group with factor=1
                group_results[factor_label] = {"c": c_base, "factor": 1.0}
                continue

            scales = base_theta.copy()
            scales[indices] *= factor
            label = f"{group_name} {factor_label}"
            print(f"  {label}...")
            c = compute_coupling(obs, theta0, pose0, scales, base_pose)
            group_results[factor_label] = {"c": c, "factor": factor}
            print(f"    c = {c:.6f}  (Δ = {c - c_base:+.6f})")
        results["sweeps"][group_name] = group_results

    # ── uniform scaling sweep ────────────────────────────────────────
    print("Uniform scaling sweep...")
    uniform = {}
    for factor_label, factor in [("×0.1", 0.1), ("×1.0", 1.0), ("×10", 10.0),
                                  ("×100", 100.0)]:
        if factor == 1.0:
            uniform[factor_label] = {"c": c_base, "factor": 1.0}
            continue
        scales = base_theta * factor
        label = f"all {factor_label}"
        print(f"  {label}...")
        c = compute_coupling(obs, theta0, pose0, scales, base_pose)
        uniform[factor_label] = {"c": c, "factor": factor}
        print(f"    c = {c:.6f}  (Δ = {c - c_base:+.6f})")
    results["sweeps"]["uniform_all_params"] = uniform

    out_dir = Path("docs/assets/pycaso_real_data/schur_ba")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "coupling_sensitivity.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    # ── quick summary ────────────────────────────────────────────────
    print("\nSummary:")
    print(f"  baseline c = {c_base:.6f}")
    for group_name in GROUPS:
        c_01 = results["sweeps"][group_name]["×0.1"]["c"]
        c_10 = results["sweeps"][group_name]["×10"]["c"]
        print(f"  {group_name:20s}  ×0.1→{c_01:.4f}   ×1→{c_base:.4f}   ×10→{c_10:.4f}")
    for factor_label in ["×0.1", "×10", "×100"]:
        c_u = uniform[factor_label]["c"]
        print(f"  uniform {factor_label:>6s}           →{c_u:.4f}  (Δ = {c_u - c_base:+.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
