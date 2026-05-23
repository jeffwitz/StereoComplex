#!/usr/bin/env python3
"""Schur-based observability diagnostic of the Pycaso CMO bundle adjustment.

Implements Step 1 of ``CdC_BA_optique_Schur_CMO_Pycaso.md``: load the
26-parameter CMO + per-arm SE(3) checkpoint, compute the Fisher matrix of
the point-to-ray residual at that point, and report the Schur complement
on the optical block (eigen-spectrum, weak directions, coupling norm).

Only the ``--mode diagnostic`` path is implemented here; subsequent CDC
steps (unregularised BA, isotropic prior, Schur prior, bootstrap) will be
added incrementally.

Usage
-----

::

    PYTHONPATH=src python examples/pycaso_schur_regularized_ba.py \\
      --input docs/assets/pycaso_real_data/intermediate_state.npz \\
      --out docs/assets/pycaso_real_data/schur_ba \\
      --mode diagnostic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from stereocomplex.optical_ba import (  # noqa: E402  (sys.path tweak above)
    PycasoCMOObservations,
    build_fisher_blocks,
    default_parameter_scales,
    diagnose_schur_modes,
    point_to_ray_residuals_cmo_se3,
)

PIXEL_PITCH_MM = 0.0055  # Pycaso sensor


def _pack_poses(opt_R: np.ndarray, opt_t: np.ndarray) -> np.ndarray:
    """Pack per-frame ``(R, t)`` into a flat ``(rotvec, tvec)`` vector."""
    n = opt_R.shape[0]
    out = np.empty(6 * n, dtype=np.float64)
    for i in range(n):
        rv = Rotation.from_matrix(opt_R[i]).as_rotvec()
        out[6 * i : 6 * i + 3] = rv
        out[6 * i + 3 : 6 * i + 6] = opt_t[i]
    return out


def _residual_rms_px(residuals: np.ndarray, observations: PycasoCMOObservations,
                     opt_t: np.ndarray, fx_ref_px: float) -> tuple[float, float]:
    """Return (rms_mm, rms_px) estimates of the point-to-ray distances.

    The transverse residual is a vector of three components per observation;
    its norm is the millimetre-scale distance. The pixel-equivalent
    approximation of CDC §2.4 uses ``fx / Z`` with ``Z`` taken as the mean
    working distance across frames.
    """
    r3 = residuals.reshape(-1, 3)
    d_mm = np.linalg.norm(r3, axis=1)
    z_ref = float(np.mean(np.abs(opt_t[:, 2])))
    rms_mm = float(np.sqrt(np.mean(d_mm**2)))
    rms_px = rms_mm * (fx_ref_px / z_ref) if z_ref > 0 else float("nan")
    return rms_mm, rms_px


def run_diagnostic(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    fx_ref = float(data["FX"])
    pose0 = _pack_poses(opt_R, opt_t)
    n_frames = obs.n_frames

    print(f"Pycaso BA diagnostic — frames={n_frames}, corners={obs.n_corners}, "
          f"theta(26), eta({pose0.size}), residuals={3 * n_frames * 2 * obs.n_corners}")

    def residual_fun(x: np.ndarray) -> np.ndarray:
        return point_to_ray_residuals_cmo_se3(x, obs)

    # Sanity: residuals at theta0.
    r0 = residual_fun(np.concatenate([theta0, pose0]))
    rms_mm, rms_px = _residual_rms_px(r0, obs, opt_t, fx_ref)
    print(f"  initial residual: rms = {rms_mm:.4f} mm  ~ {rms_px:.3f} px-equivalent")

    theta_scales, pose_scales = default_parameter_scales(n_frames)
    print(f"  building scaled Fisher (FD {args.fd_method}, rel_step={args.rel_step:.0e})...")
    fisher = build_fisher_blocks(
        residual_fun=residual_fun,
        theta0=theta0,
        pose0=pose0,
        theta_scales=theta_scales,
        pose_scales=pose_scales,
        rel_step=args.rel_step,
        method=args.fd_method,
    )

    print(f"  Schur complement (damping_pose={args.damping_pose:.0e})...")
    diag = diagnose_schur_modes(
        fisher.I_tt,
        fisher.I_tp,
        fisher.I_pp,
        weak_threshold=args.weak_threshold,
        damping_pose=args.damping_pose,
    )

    coupling = diag.coupling_norm
    print(f"  coupling norm c = {coupling:.4f}")
    print("  eigenvalues (descending): "
          + ", ".join(f"{v:.3e}" for v in diag.eigvals[:6]) + ", ...")
    print(f"  weak modes ({len(diag.weak_mode_indices)}): "
          + str(diag.weak_mode_indices.tolist()))
    print(f"  effective rank: {diag.rank_effective} / {diag.eigvals.size}")
    print(f"  condition number: {diag.condition_number:.3e}")

    diagnostic_payload = {
        "input": str(input_path),
        "n_frames": int(n_frames),
        "n_corners": int(obs.n_corners),
        "n_optical_params": int(theta0.size),
        "n_pose_params": int(pose0.size),
        "pixel_pitch_mm": float(PIXEL_PITCH_MM),
        "image_size": [int(x) for x in obs.image_size],
        "fx_ref_px": fx_ref,
        "fd_method": args.fd_method,
        "rel_step": args.rel_step,
        "damping_pose": args.damping_pose,
        "weak_threshold": args.weak_threshold,
        "residual_rms_mm_initial": rms_mm,
        "residual_rms_px_equivalent_initial": rms_px,
        "coupling_norm": coupling,
        "schur_eigvals_descending": diag.eigvals.tolist(),
        "weak_mode_indices": diag.weak_mode_indices.tolist(),
        "rank_effective": int(diag.rank_effective),
        "condition_number": (float(diag.condition_number)
                             if np.isfinite(diag.condition_number) else None),
        "condition_number_is_finite": bool(np.isfinite(diag.condition_number)),
        "theta_scales": theta_scales.tolist(),
        "pose_scales_per_frame": pose_scales[:6].tolist(),
    }

    json_path = out_dir / "schur_ba_diagnostic.json"
    json_path.write_text(json.dumps(diagnostic_payload, indent=2), encoding="utf-8")
    print(f"  wrote {json_path}")

    _plot_schur_spectrum(
        diag.eigvals,
        diag.weak_mode_indices,
        args.weak_threshold,
        out_dir / "schur_spectrum.png",
    )
    print(f"  wrote {out_dir / 'schur_spectrum.png'}")
    return 0


def _plot_schur_spectrum(eigvals: np.ndarray, weak_idx: np.ndarray,
                         weak_threshold: float, out_path: Path) -> None:
    """Save a normalised Schur eigen-spectrum plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam_max = float(np.max(eigvals)) if eigvals.size else 1.0
    normed = eigvals / max(lam_max, 1e-30)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    idx = np.arange(eigvals.size)
    strong_mask = np.ones_like(idx, dtype=bool)
    strong_mask[weak_idx] = False

    ax.semilogy(idx[strong_mask], normed[strong_mask], "o", color="#1f77b4",
                label=f"strong ({int(strong_mask.sum())})")
    if weak_idx.size:
        ax.semilogy(weak_idx, normed[weak_idx], "x", color="#d62728",
                    label=f"weak ({int(weak_idx.size)})")
    ax.axhline(weak_threshold, color="gray", linestyle=":",
               label=f"weak threshold = {weak_threshold:g}")
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel(r"$\lambda_i / \lambda_{\max}$")
    ax.set_title("Schur complement spectrum on the optical block")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default="docs/assets/pycaso_real_data/intermediate_state.npz",
        help="path to the Pycaso checkpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        default="docs/assets/pycaso_real_data/schur_ba",
        help="output directory for JSON / figures (default: %(default)s)",
    )
    parser.add_argument(
        "--mode",
        choices=("diagnostic",),
        default="diagnostic",
        help="which CDC step to run (only 'diagnostic' is implemented for now)",
    )
    parser.add_argument("--rel-step", type=float, default=1e-6,
                        help="FD relative step in scaled parameter space")
    parser.add_argument("--fd-method", choices=("central", "forward"), default="central",
                        help="finite-difference scheme for the Jacobian")
    parser.add_argument("--damping-pose", type=float, default=1e-8,
                        help="Tikhonov damping added to the pose block before inversion")
    parser.add_argument("--weak-threshold", type=float, default=1e-3,
                        help="weak-mode threshold (relative to the largest eigenvalue)")

    args = parser.parse_args(argv)
    if args.mode == "diagnostic":
        return run_diagnostic(args)
    parser.error(f"mode {args.mode!r} is not implemented yet")
    return 2


if __name__ == "__main__":
    sys.exit(main())
