#!/usr/bin/env python3
"""Parameter identifiability via projection onto Schur strong subspace.

Each of the 26 optical parameters is a unit direction in parameter space.
Projecting it onto the strong subspace (rank 5, well-observed after pose
marginalisation) gives the fraction of its variance that is identifiable
vs lost to pose coupling.

Output: ``docs/assets/pycaso_real_data/schur_ba/parameter_identifiability.json``
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

THETA_LABELS = (
    "f_obj_mm", "working_distance_mm", "b_mm",
    "cx_principal_px", "cy_principal_px",
    "f_angular_mm", "theta_convergence_half_rad",
    "d_y_common",
    "s_x_L", "s_y_L", "s_x_R", "s_y_R",
    "rho_x_shared", "rho_y_shared",
    "rv_L_x", "rv_L_y", "rv_L_z", "t_L_x_mm", "t_L_y_mm", "t_L_z_mm",
    "rv_R_x", "rv_R_y", "rv_R_z", "t_R_x_mm", "t_R_y_mm", "t_R_z_mm",
)


def _pack_poses(opt_R, opt_t):
    from scipy.spatial.transform import Rotation
    n = opt_R.shape[0]
    out = np.empty(6 * n, dtype=np.float64)
    for i in range(n):
        rv = Rotation.from_matrix(opt_R[i]).as_rotvec()
        out[6 * i:6 * i + 3] = rv
        out[6 * i + 3:6 * i + 6] = opt_t[i]
    return out


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

    theta_scales, pose_scales = default_parameter_scales(n_frames)

    def _res(x): return point_to_ray_residuals_cmo_se3(x, obs)

    print("Building Fisher blocks...")
    fisher = build_fisher_blocks(
        residual_fun=_res, theta0=theta0, pose0=pose0,
        theta_scales=theta_scales, pose_scales=pose_scales,
        rel_step=1e-6, method="central",
    )

    diag = diagnose_schur_modes(
        fisher.I_tt, fisher.I_tp, fisher.I_pp,
        weak_threshold=1e-3, damping_pose=1e-8,
    )

    # Strong subspace: eigenvectors with index < rank_effective
    V_strong = diag.eigvecs[:, :diag.rank_effective]  # (26, 5)
    V_weak = diag.eigvecs[:, diag.rank_effective:]    # (26, 21)

    # Project each parameter direction (unit vector e_i) onto strong subspace
    # identifiability_i = ||P_strong e_i||^2 = sum_{j in strong} (e_i · v_j)^2
    identifiability = np.sum(V_strong ** 2, axis=1)  # ∈ [0, 1]

    # Per-parameter report
    params = []
    for i in range(len(THETA_LABELS)):
        params.append({
            "index": i,
            "name": THETA_LABELS[i],
            "value": float(theta0[i]),
            "scale": float(theta_scales[i]),
            "identifiability": round(float(identifiability[i]), 4),
        })

    # ── Group summaries ────────────────────────────────────────────
    rot_indices = [14, 15, 16, 20, 21, 22]
    trans_indices = [17, 18, 19, 23, 24, 25]
    optical_indices = list(range(14))

    def group_stats(indices, label):
        vals = identifiability[list(indices)]
        return {
            "group": label,
            "indices": list(indices),
            "mean_identifiability": round(float(np.mean(vals)), 4),
            "min_identifiability": round(float(np.min(vals)), 4),
            "identified": [THETA_LABELS[i] for i in indices if identifiability[i] > 0.5],
            "effective": [THETA_LABELS[i] for i in indices if identifiability[i] < 0.1],
        }

    report = {
        "description": "Parameter identifiability via strong-subspace projection",
        "rank_effective": diag.rank_effective,
        "n_theta": len(THETA_LABELS),
        "parameters": params,
        "groups": [
            group_stats(optical_indices, "CMO telecentric"),
            group_stats(rot_indices, "SE(3) rotations"),
            group_stats(trans_indices, "SE(3) translations"),
        ],
    }

    out_dir = Path("docs/assets/pycaso_real_data/schur_ba")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parameter_identifiability.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")

    # ── Print ──────────────────────────────────────────────────────
    print(f"\nEffective rank: {diag.rank_effective} / 26")
    print(f"\nIdentifiability (fraction of variance in strong subspace):")
    for p in params:
        bar = "█" * int(p["identifiability"] * 40) + "░" * (40 - int(p["identifiability"] * 40))
        status = "identified" if p["identifiability"] > 0.5 else (
            "effective" if p["identifiability"] < 0.1 else "partial"
        )
        print(f"  [{p['index']:2d}] {p['name']:28s} {p['identifiability']:.3f} {bar} {status}")

    print("\nGroup summary:")
    for g in report["groups"]:
        print(f"  {g['group']}:")
        print(f"    mean identifiability: {g['mean_identifiability']:.3f}")
        if g["identified"]:
            print(f"    identified (>0.5): {', '.join(g['identified'])}")
        if g["effective"]:
            print(f"    effective (<0.1):  {', '.join(g['effective'])}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
