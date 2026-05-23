#!/usr/bin/env python3
"""Schur-based observability diagnostic of the Pycaso CMO bundle adjustment.

Implements Step 1 of ``CdC_BA_optique_Schur_CMO_Pycaso.md``: load the
26-parameter CMO + per-arm SE(3) checkpoint, compute the Fisher matrix of
the point-to-ray residual at that point, and report the Schur complement
on the optical block (eigen-spectrum, weak directions, coupling norm).

Implemented CDC steps: ``diagnostic`` (Step 1), ``ba`` (Step 2),
``isotropic-sweep`` (Step 3), ``schur-sweep`` (Step 4).
Bootstrap (Step 5) is not yet implemented.

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
    EUR_2_CENT_DIAMETER_MM,
    PycasoCMOObservations,
    SpecimenReconstruction,
    build_fisher_blocks,
    default_parameter_scales,
    diagnose_schur_modes,
    load_zernike_baseline,
    magnification_ratio,
    point_to_ray_residuals_cmo_se3,
    reconstruct_with_cmo_se3,
    run_optical_ba,
    run_schur_regularized_optical_ba,
)

PIXEL_PITCH_MM = 0.0055  # Pycaso sensor


# Physical names of the 26 optical parameters, in the order produced by
# CMOTelecentricStereoModel.parameter_vector (with shared_slopes=False,
# shared_shear=True, the layout used by refine_26p_on_corners.py) followed
# by the 12 per-arm SE(3) parameters.
THETA_LABELS = (
    # 14 telecentric CMO parameters
    "f_obj_mm", "working_distance_mm", "b_mm",
    "cx_principal_px", "cy_principal_px",
    "f_angular_mm", "theta_convergence_half_rad",
    "d_y_common",
    "s_x_L", "s_y_L", "s_x_R", "s_y_R",
    "rho_x_shared", "rho_y_shared",
    # 12 per-arm SE(3) parameters
    "rv_L_x", "rv_L_y", "rv_L_z", "t_L_x_mm", "t_L_y_mm", "t_L_z_mm",
    "rv_R_x", "rv_R_y", "rv_R_z", "t_R_x_mm", "t_R_y_mm", "t_R_z_mm",
)


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


def _load_checkpoint(input_path: Path) -> tuple[
    PycasoCMOObservations, np.ndarray, np.ndarray, np.ndarray, float
]:
    """Load the Pycaso checkpoint into the shapes the script needs."""
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
    return obs, theta0, opt_R, opt_t, fx_ref


def run_diagnostic(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, theta0, opt_R, opt_t, fx_ref = _load_checkpoint(input_path)
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

    # Inspect the strong-mode subspace (CDC sanity check before Step 2):
    # which physical directions do the rank_effective eigenvectors represent?
    strong_modes = _describe_strong_modes(diag, top_components=4)
    print("  strong-mode physical content (top 4 components, absolute):")
    for entry in strong_modes:
        comp_str = ", ".join(f"{name}={w:+.2f}" for name, w in entry["components"])
        print(f"    mode {entry['index']:2d}  lambda={entry['eigenvalue']:.3e}  | {comp_str}")

    # Sweep the pose damping (CDC §11.2): the diagnostic must be insensitive
    # to lambda for a reliable observability conclusion.
    damping_sweep = _sweep_pose_damping(
        fisher,
        damping_values=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2),
        weak_threshold=args.weak_threshold,
    )
    print("  damping-pose sweep:")
    print(f"    {'lambda':>10s}  {'c':>8s}  {'rank_eff':>9s}  "
          f"{'top5_subspace_overlap':>22s}")
    for row in damping_sweep:
        print(f"    {row['damping_pose']:10.0e}  {row['coupling_norm']:8.4f}  "
              f"{row['rank_effective']:9d}  {row['top5_subspace_overlap']:22.4f}")

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
        "theta_labels": list(THETA_LABELS),
        "strong_modes": strong_modes,
        "damping_pose_sweep": damping_sweep,
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


def _describe_strong_modes(diag, top_components: int = 4) -> list[dict]:
    """Return the top-K components of each strong eigenvector with labels.

    For each mode with ``i < diag.rank_effective``, pick the ``top_components``
    parameter entries with the largest absolute weight in
    ``diag.eigvecs[:, i]``. Helps tell whether the strong modes correspond to
    interpretable physical directions (e.g. ``f_obj + working_distance``).
    """
    out: list[dict] = []
    for i in range(diag.rank_effective):
        v = diag.eigvecs[:, i]
        order = np.argsort(np.abs(v))[::-1][:top_components]
        comps = [(THETA_LABELS[k], float(v[k])) for k in order]
        out.append({
            "index": int(i),
            "eigenvalue": float(diag.eigvals[i]),
            "components": comps,
        })
    return out


def _sweep_pose_damping(
    fisher,
    damping_values: tuple[float, ...],
    weak_threshold: float,
) -> list[dict]:
    """Re-evaluate the Schur diagnostic across pose-damping values (CDC §11.2).

    Returns one record per damping value with the coupling norm, the
    effective rank, and the principal-angle overlap between the top-5
    eigenspace at that damping and the reference one (the smallest
    damping). A stable diagnostic has overlaps close to 1 across the range.
    """
    from stereocomplex.optical_ba.schur import diagnose_schur_modes  # noqa: PLC0415

    diags = [
        diagnose_schur_modes(
            fisher.I_tt, fisher.I_tp, fisher.I_pp,
            damping_pose=lam, weak_threshold=weak_threshold,
        )
        for lam in damping_values
    ]
    # Reference: smallest damping value (least biased Schur).
    ref_idx = int(np.argmin(damping_values))
    V_ref = diags[ref_idx].eigvecs[:, :5]
    rows: list[dict] = []
    for lam, d in zip(damping_values, diags, strict=True):
        V = d.eigvecs[:, :5]
        # Squared principal-angle overlap = ||V_ref^T V||_F^2 / 5 in [0, 1].
        overlap = float(np.linalg.norm(V_ref.T @ V, ord="fro") ** 2 / 5.0)
        rows.append({
            "damping_pose": float(lam),
            "coupling_norm": float(d.coupling_norm),
            "rank_effective": int(d.rank_effective),
            "top5_subspace_overlap": overlap,
            "eigvals_top5": [float(v) for v in d.eigvals[:5]],
        })
    return rows


def _align_cmo_to_zernike_frame(
    cmo_rec: SpecimenReconstruction,
    zer_rec: SpecimenReconstruction,
) -> SpecimenReconstruction:
    """Correct the CMO reconstruction's world Y axis to match the Zernike frame.

    The CMO telecentric model internally uses ``cy_principal_px`` at the
    bottom edge of the image (pixel coordinate v = H), which combined with
    negative ``rho_y`` / ``s_y`` parameters inverts the v → Y mapping
    relative to the Zernike rayfield.  The small per-arm SE(3) rotation
    (~2°) cannot correct a sign flip, so the CMO point cloud leaves the
    model with its Y axis mirrored.

    We apply the simplest correction that preserves the point-cloud shape:
    a Y reflection about the CMO centroid followed by the translation that
    re-centres on the Zernike Y centroid.  The X and Z axes are left
    unchanged because their observational sign conventions already match
    (u → X has the same sign in both models).
    """
    y_offset = float(np.mean(zer_rec.Y) + np.mean(cmo_rec.Y))
    Y_corr = -cmo_rec.Y + y_offset
    return SpecimenReconstruction(
        X=cmo_rec.X,
        Y=Y_corr,
        Z=cmo_rec.Z,
        gap_mm=cmo_rec.gap_mm,
        valid_mask=cmo_rec.valid_mask,
        image_extent_xy_mm=(
            float(cmo_rec.X.max() - cmo_rec.X.min()),
            float(Y_corr.max() - Y_corr.min()),
        ),
        median_z_mm=cmo_rec.median_z_mm,
        z_mad_mm=cmo_rec.z_mad_mm,
        median_gap_mm=cmo_rec.median_gap_mm,
        variant=cmo_rec.variant,
    )


def _reconstruct_and_plot_specimens(
    theta_initial: np.ndarray,
    theta_final: np.ndarray,
    *,
    variant_label: str,
    out_dir: Path,
    correspondences_path: Path,
    zernike_baseline_path: Path,
) -> dict:
    """Reconstruct the Pycaso coin with three model variants and plot them.

    Variants compared:

    1. Zernike rayfield (57p) — loaded from the published checkpoint.
    2. CMO 26p at the rayfield-derived initialisation ``theta_initial``.
    3. CMO 26p after the current BA, ``theta_final``, labelled ``variant_label``.

    All three Z maps are rendered on a shared colour scale; the figure
    title makes the magnification ratio of every variant explicit so the
    rayfield-vs-CMO scale gap is visible at a glance.

    CMO reconstructions are produced in the CMO model's internal frame
    (``cy_principal_px`` at the image bottom edge creates a sign convention
    that differs from the Zernike reference frame).  A per-axis correction
    brings them into the Zernike world frame so the XY-scatter and Z-map
    columns share a consistent orientation.
    """
    rec_zer = load_zernike_baseline(zernike_baseline_path)
    rec_cmo_init = reconstruct_with_cmo_se3(
        theta_initial,
        correspondences_path=correspondences_path,
        variant="cmo_26p_initial",
    )
    rec_cmo_ba = reconstruct_with_cmo_se3(
        theta_final,
        correspondences_path=correspondences_path,
        variant=variant_label,
    )

    rec_cmo_init = _align_cmo_to_zernike_frame(rec_cmo_init, rec_zer)
    rec_cmo_ba = _align_cmo_to_zernike_frame(rec_cmo_ba, rec_zer)

    recs: list[SpecimenReconstruction] = [rec_zer, rec_cmo_init, rec_cmo_ba]

    # Save per-variant point clouds for follow-up analysis.
    for rec in recs:
        np.savez_compressed(
            out_dir / f"specimen_{rec.variant}.npz",
            X=rec.X.astype(np.float32), Y=rec.Y.astype(np.float32),
            Z=rec.Z.astype(np.float32), gap_mm=rec.gap_mm.astype(np.float32),
        )

    metrics = {
        rec.variant: {
            "median_z_mm": rec.median_z_mm,
            "z_mad_mm": rec.z_mad_mm,
            "median_gap_mm": rec.median_gap_mm,
            "image_extent_xy_mm": [rec.image_extent_xy_mm[0], rec.image_extent_xy_mm[1]],
            "magnification_vs_eur_2c": magnification_ratio(rec, EUR_2_CENT_DIAMETER_MM),
            "n_points": int(rec.X.size),
        }
        for rec in recs
    }
    metrics["nominal_diameter_mm"] = float(EUR_2_CENT_DIAMETER_MM)
    (out_dir / f"specimen_comparison_{variant_label}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _plot_specimen_grid(recs, correspondences_path,
                        out_dir / f"specimen_comparison_{variant_label}.png")

    print("  specimen comparison vs 2-cent euro (18.75 mm):")
    for rec in recs:
        ratio = magnification_ratio(rec, EUR_2_CENT_DIAMETER_MM)
        w, h = rec.image_extent_xy_mm
        print(f"    {rec.variant:30s} extent={w:6.2f}x{h:6.2f} mm  "
              f"ratio={ratio:.4f}  Z_med={rec.median_z_mm:+.3f} mm  "
              f"Z_MAD={rec.z_mad_mm:.4f} mm  gap_med={rec.median_gap_mm:.4f} mm")
    return metrics


def _plot_specimen_grid(
    recs: list,
    correspondences_path: Path,
    out_path: Path,
) -> None:
    """3-column comparison: Z map, XY footprint, ray-gap histogram per variant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(correspondences_path)
    roi = [int(x) for x in data["roi"]]
    roi_x0, roi_x1, roi_y0, roi_y1 = roi
    h_roi, w_roi = (roi_y1 - roi_y0), (roi_x1 - roi_x0)
    uL = np.asarray(data["uL"], dtype=np.float64)
    vL = np.asarray(data["vL"], dtype=np.float64)
    n_corr = uL.size
    for rec in recs:
        if rec.valid_mask.size != n_corr:
            raise ValueError(
                f"variant {rec.variant!r} valid_mask has {rec.valid_mask.size} "
                f"entries, expected {n_corr} (same as correspondences)"
            )

    ui_full = (uL - roi_x0).astype(np.int64)
    vi_full = (vL - roi_y0).astype(np.int64)

    # Detrend each variant by its best-fit mean plane so the Z-map and
    # scatter show surface relief (deviation from the local plane) rather
    # than absolute working distance — all rows share the same colour scale.
    z_rel = []
    for rec in recs:
        A = np.column_stack([rec.X, rec.Y, np.ones_like(rec.X)])
        a, b, c = np.linalg.lstsq(A, rec.Z, rcond=None)[0]
        z_rel.append(rec.Z - (a * rec.X + b * rec.Y + c))
    all_z_rel = np.concatenate(z_rel)
    z_limit = float(np.percentile(np.abs(all_z_rel), 98))
    vmin, vmax = -z_limit, z_limit

    # Shared ray-gap range: median gaps are ~1e-3 mm on Pycaso, so a fixed
    # 0.2 mm window crushes everything into the first bin. Use the worst
    # variant's 99-th percentile (with a small buffer) so the histogram tails
    # are readable AND comparable across rows.
    gap_p99_max = max(float(np.percentile(rec.gap_mm, 99)) for rec in recs)
    gap_upper = max(1.1 * gap_p99_max, 1e-4)

    n_var = len(recs)
    fig, axes = plt.subplots(n_var, 3, figsize=(15, 4 * n_var))
    if n_var == 1:
        axes = np.atleast_2d(axes)

    for row, rec in enumerate(recs):
        z_map = np.full((h_roi, w_roi), np.nan, dtype=np.float64)
        z_map[vi_full[rec.valid_mask], ui_full[rec.valid_mask]] = z_rel[row]
        ax_z, ax_xy, ax_gap = axes[row]

        im = ax_z.imshow(z_map, cmap="viridis", origin="upper",
                         vmin=vmin, vmax=vmax)
        ax_z.set_title(
            f"{rec.variant}\nZ MAD = {rec.z_mad_mm:.4f} mm"
        )
        ax_z.set_xlabel("u-ROI (px)")
        ax_z.set_ylabel("v-ROI (px)")
        fig.colorbar(im, ax=ax_z, label="Z − mean plane (mm)")

        ratio = magnification_ratio(rec, EUR_2_CENT_DIAMETER_MM)
        ax_xy.scatter(rec.X, rec.Y, s=1, c=z_rel[row], cmap="viridis",
                      vmin=vmin, vmax=vmax)
        ax_xy.set_aspect("equal")
        # Z-map and scatter share the OpenCV convention where Y_world
        # follows image v (downward).  Matplotlib defaults to Y-up, so
        # invert the scatter's Y axis to match the Z-map origin="upper".
        ax_xy.invert_yaxis()
        w, h = rec.image_extent_xy_mm
        ax_xy.set_title(
            f"XY footprint = {w:.2f} x {h:.2f} mm\n"
            f"magnification ratio vs 18.75 mm coin = {ratio:.4f}"
        )
        ax_xy.set_xlabel("X (mm)")
        ax_xy.set_ylabel("Y (mm)")

        ax_gap.hist(rec.gap_mm, bins=80, range=(0.0, gap_upper),
                    color="steelblue", alpha=0.85)
        ax_gap.axvline(rec.median_gap_mm, color="red", lw=1,
                       label=f"median = {rec.median_gap_mm:.4f} mm")
        ax_gap.set_yscale("log")
        ax_gap.set_title(f"Ray-pair gap (log y, shared range 0–{gap_upper:.4f} mm)")
        ax_gap.set_xlabel("gap (mm)")
        ax_gap.set_ylabel("count (log)")
        ax_gap.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Pycaso 2-cent coin — surface relief (Z − mean plane) across variants  "
        f"(n_corr={n_corr})",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sweep_alphas(arg_value: str) -> list[float]:
    """Parse a comma-separated alpha list, e.g. ``1e-4,1e-3,0.01,0.1,1,10``."""
    return [float(s.strip()) for s in arg_value.split(",")]


def run_isotropic_sweep(args: argparse.Namespace) -> int:
    """CDC Step 3: isotropic (Tikhonov) prior sweep over *alpha*."""
    from stereocomplex.optical_ba.priors import isotropic_prior_residuals

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, theta0, opt_R, opt_t, fx_ref = _load_checkpoint(input_path)
    pose0 = _pack_poses(opt_R, opt_t)
    theta_scales, pose_scales = default_parameter_scales(obs.n_frames)

    alphas = _sweep_alphas(args.alpha_list)
    rows: list[dict] = []
    print(f"Isotropic prior sweep — {len(alphas)} alpha values, "
          f"loss={args.loss}, max_nfev={args.max_nfev}")

    for alpha in alphas:
        prior_fn = lambda th: isotropic_prior_residuals(  # noqa: E731
            th, theta0, theta_scales, alpha
        )
        result = run_schur_regularized_optical_ba(
            theta0=theta0, pose0=pose0, observations=obs,
            fx_ref_px=fx_ref, prior=prior_fn,
            loss=args.loss, f_scale_mm=args.f_scale_mm,
            max_nfev=args.max_nfev,
            weak_threshold=args.weak_threshold,
            damping_pose=args.damping_pose,
            fd_method=args.fd_method, fd_rel_step=args.rel_step,
        )
        row = {
            "alpha": alpha,
            "success": result.success,
            "nfev": result.nfev,
            "rms_mm_final": result.rms_mm,
            "rms_px_equivalent_final": result.rms_px_equivalent,
            "theta_drift_norm": result.theta_drift_norm,
            "weak_mode_drift_norm": result.weak_mode_drift_norm,
            "strong_mode_drift_norm": result.strong_mode_drift_norm,
            "coupling_before": result.schur_coupling_before,
            "coupling_after": result.schur_coupling_after,
            "descriptors": result.descriptors,
        }
        rows.append(row)
        print(f"  alpha={alpha:.1e}  RMS={result.rms_px_equivalent:.3f} px  "
              f"drift_weak={result.weak_mode_drift_norm:.4f}  "
              f"nfev={result.nfev}  ok={result.success}")

    (out_dir / "optical_ba_isotropic_prior_sweep.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"  wrote {out_dir / 'optical_ba_isotropic_prior_sweep.json'}")
    return 0


def run_schur_sweep(args: argparse.Namespace) -> int:
    """CDC Step 4: Schur-prior regularised BA sweep over *alpha*."""
    from stereocomplex.optical_ba.priors import SchurPrior

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, theta0, opt_R, opt_t, fx_ref = _load_checkpoint(input_path)
    pose0 = _pack_poses(opt_R, opt_t)
    theta_scales, pose_scales = default_parameter_scales(obs.n_frames)

    # Build the "before" Schur diagnostic to seed the prior.
    data_fun = lambda x: point_to_ray_residuals_cmo_se3(x, obs)  # noqa: E731
    fisher = build_fisher_blocks(
        residual_fun=data_fun, theta0=theta0, pose0=pose0,
        theta_scales=theta_scales, pose_scales=pose_scales,
        rel_step=args.rel_step, method=args.fd_method,
    )
    diag = diagnose_schur_modes(
        fisher.I_tt, fisher.I_tp, fisher.I_pp,
        weak_threshold=args.weak_threshold,
        damping_pose=args.damping_pose,
    )

    alphas = _sweep_alphas(args.alpha_list)
    rows: list[dict] = []
    print(f"Schur prior sweep — {len(alphas)} alpha values, "
          f"power={args.schur_power}, eps={args.schur_epsilon}, "
          f"loss={args.loss}, max_nfev={args.max_nfev}")

    for alpha in alphas:
        prior = SchurPrior(
            theta0=theta0, eigvals=diag.eigvals, eigvecs=diag.eigvecs,
            theta_scales=theta_scales, alpha=alpha,
            power=args.schur_power, epsilon=args.schur_epsilon,
        )
        result = run_schur_regularized_optical_ba(
            theta0=theta0, pose0=pose0, observations=obs,
            fx_ref_px=fx_ref, prior=prior,
            loss=args.loss, f_scale_mm=args.f_scale_mm,
            max_nfev=args.max_nfev,
            weak_threshold=args.weak_threshold,
            damping_pose=args.damping_pose,
            fd_method=args.fd_method, fd_rel_step=args.rel_step,
            fisher_before=fisher,
            compute_fisher_after=False,
        )
        row = {
            "alpha": alpha,
            "power": args.schur_power,
            "epsilon": args.schur_epsilon,
            "success": result.success,
            "nfev": result.nfev,
            "rms_mm_final": result.rms_mm,
            "rms_px_equivalent_final": result.rms_px_equivalent,
            "theta_drift_norm": result.theta_drift_norm,
            "weak_mode_drift_norm": result.weak_mode_drift_norm,
            "strong_mode_drift_norm": result.strong_mode_drift_norm,
            "coupling_before": result.schur_coupling_before,
            "coupling_after": None,  # skipped (compute_fisher_after=False in sweep)
            "descriptors": result.descriptors,
        }
        rows.append(row)
        print(f"  alpha={alpha:.1e}  RMS={result.rms_px_equivalent:.3f} px  "
              f"drift_weak={result.weak_mode_drift_norm:.4f}  "
              f"nfev={result.nfev}  ok={result.success}")

    (out_dir / "optical_ba_schur_prior_sweep.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"  wrote {out_dir / 'optical_ba_schur_prior_sweep.json'}")
    return 0


def run_ba(args: argparse.Namespace) -> int:
    """CDC Step 2: direct (unregularised) optical BA on the Pycaso checkpoint."""
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs, theta0, opt_R, opt_t, fx_ref = _load_checkpoint(input_path)
    pose0 = _pack_poses(opt_R, opt_t)
    print(f"Pycaso direct BA — frames={obs.n_frames}, corners={obs.n_corners}, "
          f"theta(26), eta({pose0.size}), f_scale={args.f_scale_mm} mm, "
          f"loss={args.loss}, max_nfev={args.max_nfev}")

    result = run_optical_ba(
        theta0=theta0,
        pose0=pose0,
        observations=obs,
        fx_ref_px=fx_ref,
        loss=args.loss,
        f_scale_mm=args.f_scale_mm,
        max_nfev=args.max_nfev,
        weak_threshold=args.weak_threshold,
        damping_pose=args.damping_pose,
        fd_method=args.fd_method,
        fd_rel_step=args.rel_step,
    )

    print(f"  optimiser: success={result.success}, nfev={result.nfev}, "
          f"message={result.message!r}")
    print(f"  RMS         init={result.diagnostics['rms_mm_initial']:.5f} mm  "
          f"final={result.rms_mm:.5f} mm  (~{result.rms_px_equivalent:.3f} px)")
    print(f"  P50/P95 px  {result.p50_px_equivalent:.3f} / {result.p95_px_equivalent:.3f}")
    print(f"  coupling c  before={result.schur_coupling_before:.4f}  "
          f"after={result.schur_coupling_after:.4f}")
    print(f"  drift       total={result.theta_drift_norm:.4f}  "
          f"weak={result.weak_mode_drift_norm:.4f}  "
          f"strong={result.strong_mode_drift_norm:.4f}")
    print("  descriptors:")
    for k, v in result.descriptors.items():
        print(f"    {k:30s} {v:+.6f}")

    payload = {
        "input": str(input_path),
        "method": "unregularized_direct_ba",
        "loss": args.loss,
        "f_scale_mm": args.f_scale_mm,
        "max_nfev": args.max_nfev,
        "weak_threshold": args.weak_threshold,
        "damping_pose": args.damping_pose,
        "success": result.success,
        "nfev": result.nfev,
        "message": result.message,
        "rms_mm_initial": result.diagnostics["rms_mm_initial"],
        "rms_mm_final": result.rms_mm,
        "rms_px_equivalent_final": result.rms_px_equivalent,
        "p50_px_equivalent_final": result.p50_px_equivalent,
        "p95_px_equivalent_final": result.p95_px_equivalent,
        "schur_coupling_before": result.schur_coupling_before,
        "schur_coupling_after": result.schur_coupling_after,
        "theta_drift_norm": result.theta_drift_norm,
        "weak_mode_drift_norm": result.weak_mode_drift_norm,
        "strong_mode_drift_norm": result.strong_mode_drift_norm,
        "descriptors_final": result.descriptors,
        "theta_final": result.theta.tolist(),
        "theta_initial": theta0.tolist(),
        "diagnostics": result.diagnostics,
    }
    json_path = out_dir / "optical_ba_unregularized.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {json_path}")

    if args.with_specimen:
        correspondences_path = Path(args.correspondences)
        zernike_path = Path(args.zernike_baseline)
        if not correspondences_path.is_file() or not zernike_path.is_file():
            print(f"  [specimen] skipping — missing {correspondences_path} "
                  f"or {zernike_path}")
        else:
            _reconstruct_and_plot_specimens(
                theta_initial=theta0,
                theta_final=result.theta,
                variant_label="cmo_26p_ba_unregularized",
                out_dir=out_dir,
                correspondences_path=correspondences_path,
                zernike_baseline_path=zernike_path,
            )
            print(f"  wrote specimen comparison "
                  f"{out_dir / 'specimen_comparison_cmo_26p_ba_unregularized.png'}")
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
        choices=("diagnostic", "ba", "isotropic-sweep", "schur-sweep"),
        default="diagnostic",
        help="which CDC step to run (diagnostic = Step 1, ba = Step 2 direct BA)",
    )
    parser.add_argument("--rel-step", type=float, default=1e-6,
                        help="FD relative step in scaled parameter space")
    parser.add_argument("--fd-method", choices=("central", "forward"), default="central",
                        help="finite-difference scheme for the Jacobian")
    parser.add_argument("--damping-pose", type=float, default=1e-8,
                        help="Tikhonov damping added to the pose block before inversion")
    parser.add_argument("--weak-threshold", type=float, default=1e-3,
                        help="weak-mode threshold (relative to the largest eigenvalue)")
    # --mode ba / isotropic-sweep / schur-sweep
    parser.add_argument("--loss", default="soft_l1",
                        help="robust loss for least_squares (BA modes)")
    parser.add_argument("--f-scale-mm", type=float, default=0.005,
                        help="robust-loss transition scale in mm (BA modes)")
    parser.add_argument("--max-nfev", type=int, default=200,
                        help="maximum residual evaluations for the BA (BA modes)")
    # --mode isotropic-sweep / schur-sweep
    parser.add_argument("--alpha-list", default="1e-4,1e-3,1e-2,1e-1,1,10",
                        help="comma-separated alpha values for the sweep")
    parser.add_argument("--schur-power", type=float, default=1.0,
                        help="eigenvalue-weight exponent for --mode schur-sweep")
    parser.add_argument("--schur-epsilon", type=float, default=1e-6,
                        help="regularisation floor for --mode schur-sweep")
    parser.add_argument("--with-specimen", action="store_true",
                        help="also reconstruct the 2-cent coin and emit a comparison figure")
    parser.add_argument("--correspondences",
                        default="docs/assets/pycaso_real_data/specimen_correspondences.npz",
                        help="path to the DIS-flow correspondences npz")
    parser.add_argument("--zernike-baseline",
                        default="docs/assets/pycaso_real_data/specimen_reconstruction_zernike.npz",
                        help="path to the published Zernike rayfield reconstruction npz")

    args = parser.parse_args(argv)
    if args.mode == "diagnostic":
        return run_diagnostic(args)
    if args.mode == "ba":
        return run_ba(args)
    if args.mode == "isotropic-sweep":
        return run_isotropic_sweep(args)
    if args.mode == "schur-sweep":
        return run_schur_sweep(args)
    parser.error(f"mode {args.mode!r} is not implemented yet")
    return 2


if __name__ == "__main__":
    sys.exit(main())
