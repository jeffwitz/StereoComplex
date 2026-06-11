#!/usr/bin/env python3
"""Generate Figure — CMO rayfield interpretation in the X–Z plane.

Each pixel defines a 3-D ray :math:`L_c(u,v) = O_c(u,v) + t \\, d_c(u,v)`.
For visualisation, rays are intersected with an effective sub-pupil plane
and with the object (working) plane.  The figure is generated directly from
the identified 26-parameter CMO + per-arm SE(3) model stored in
``docs/assets/pycaso_real_data/intermediate_state.npz``.

Usage
-----
    rtk .venv/bin/python examples/notebooks/generate_fig_cmo_rayfield_xz.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["mathtext.fontset"] = "stix"

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parents[2]
_SYS_PATH = [_REPO / "src"]
for _p in _SYS_PATH:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel

PIXEL_PITCH_MM = 0.0055       # Pycaso sensor
N_TEL = 14                    # telecentric CMO parameters
OUT_PDF = _REPO / "paper/cmo/figures/cmo_rayfield_xz.pdf"
OUT_PNG = _REPO / "paper/cmo/figures/cmo_rayfield_xz.png"
OUT_JSON = _REPO / "docs/assets/cmo_paper/figureX_cmo_rayfield_xz/labels.json"


def intersect_z_plane(O: np.ndarray, d: np.ndarray, z_plane: float) -> np.ndarray:
    """Intersection of 3-D rays ``O + t d`` with the plane ``z = z_plane``.

    Parameters
    ----------
    O : ndarray, shape (N, 3)
        Points on each ray.
    d : ndarray, shape (N, 3)
        Unit ray directions.
    z_plane : float
        Target Z coordinate.

    Returns
    -------
    P : ndarray, shape (N, 3)
        Intersection points; rows with ``|dz| < eps`` become NaN.
    """
    eps = 1e-12
    dz = np.where(np.abs(d[..., 2]) < eps, np.nan, d[..., 2])
    t = (z_plane - O[..., 2]) / dz
    return O + t[..., None] * d


def _apply_se3(O, d, rv, t):
    """Rigid transform of ray origins and directions."""
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], (R @ d.T).T


def main() -> int:
    # ── Load identified model ──────────────────────────────────────────
    state = np.load(
        _REPO / "docs/assets/pycaso_real_data/intermediate_state.npz",
        allow_pickle=True,
    )
    x_26p = np.asarray(state["x_26p"], dtype=np.float64)
    image_size = tuple(int(x) for x in state["image_size"])
    W, H = image_size

    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        x_26p[:N_TEL], pixel_pitch_mm=PIXEL_PITCH_MM, image_size=image_size,
    )

    # Per-arm SE(3)
    rv_L, t_L = x_26p[14:17], x_26p[17:20]
    rv_R, t_R = x_26p[20:23], x_26p[23:26]

    # ── Sample pixels across the sensor ────────────────────────────────
    # Mid-height horizontal scan for a clean X–Z view
    v_mid = H / 2
    u_samples = np.linspace(0.05 * W, 0.95 * W, 7)
    u_grid, v_grid = np.meshgrid(u_samples, [v_mid])
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    # Also a few extra rows for visual richness (3 rows × 7 cols = 21 rays/channel)
    v_extra = np.linspace(0.15 * H, 0.85 * H, 3)
    UU, VV = np.meshgrid(u_samples, v_extra)
    u_all = UU.ravel()
    v_all = VV.ravel()

    # ── Compute rays from the telecentric skeleton ─────────────────────
    OL0, dL0 = m_tel.ray(u_all, v_all, "left")
    OR0, dR0 = m_tel.ray(u_all, v_all, "right")

    # Apply per-arm SE(3)
    OL, dL = _apply_se3(OL0, dL0, rv_L, t_L)
    OR, dR = _apply_se3(OR0, dR0, rv_R, t_R)

    # ── Choose visualisation planes ────────────────────────────────────
    # Effective sub-pupil plane: median Z of all ray origins
    z_pupil = float(np.median(np.concatenate([OL[:, 2], OR[:, 2]])))
    # Object plane: working distance (Z-forward in OpenCV frame)
    z_obj = float(m_tel.working_distance_mm)

    # Intersect every ray with both planes
    QL = intersect_z_plane(OL, dL, z_pupil)
    PL = intersect_z_plane(OL, dL, z_obj)
    QR = intersect_z_plane(OR, dR, z_pupil)
    PR = intersect_z_plane(OR, dR, z_obj)

    # ── Descriptors ────────────────────────────────────────────────────
    b_mm = float(m_tel.b_mm)
    WD = float(m_tel.working_distance_mm)
    f_obj = float(m_tel.f_obj_mm)
    theta_deg = float(np.degrees(2.0 * m_tel.theta_convergence_half_rad))
    z_pupil_model = WD - f_obj  # sub-pupil Z before SE(3)

    print(f"Effective pupil plane z = {z_pupil:.3f} mm  (model z_pupil = {z_pupil_model:.3f} mm)")
    print(f"Object plane z = {z_obj:.3f} mm")
    print(f"Baseline b = {b_mm:.2f} mm")
    print(f"Convergence angle Θ = {theta_deg:.1f}°")
    print(f"WD = {WD:.2f} mm,  f_obj = {f_obj:.2f} mm")

    # ── Figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Plot rays as line segments from pupil plane to object plane
    for i in range(len(QL)):
        ax.plot(
            [QL[i, 0], PL[i, 0]], [QL[i, 2], PL[i, 2]],
            color="tab:blue", lw=1.4, alpha=0.55,
            label="Left channel rays" if i == 0 else None,
        )
    for i in range(len(QR)):
        ax.plot(
            [QR[i, 0], PR[i, 0]], [QR[i, 2], PR[i, 2]],
            color="tab:red", lw=1.4, alpha=0.55,
            label="Right channel rays" if i == 0 else None,
        )

    # Effective sub-pupil points (intersections at pupil plane)
    ax.scatter(QL[:, 0], QL[:, 2], c="tab:blue", s=30, zorder=5, edgecolors="white", linewidths=0.5)
    ax.scatter(QR[:, 0], QR[:, 2], c="tab:red", s=30, zorder=5, edgecolors="white", linewidths=0.5)

    # Object-plane intersections
    ax.scatter(PL[:, 0], PL[:, 2], c="tab:blue", s=12, alpha=0.6, zorder=4)
    ax.scatter(PR[:, 0], PR[:, 2], c="tab:red", s=12, alpha=0.6, zorder=4)

    # Horizontal reference planes
    x_min = min(QL[:, 0].min(), QR[:, 0].min(), PL[:, 0].min(), PR[:, 0].min()) - 1.0
    x_max = max(QL[:, 0].max(), QR[:, 0].max(), PL[:, 0].max(), PR[:, 0].max()) + 1.0

    ax.axhline(z_pupil, color="0.35", lw=1.0, ls="--", zorder=1)
    ax.text(x_min, z_pupil + 0.5, "Effective sub-pupil plane",
            va="bottom", fontsize=8.5, color="0.3")

    ax.axhline(z_obj, color="0.15", lw=1.3, zorder=1)
    ax.text(x_min, z_obj + 0.5, "Object plane (WD)",
            va="bottom", fontsize=8.5, color="0.15")

    # Annotations
    # Baseline
    mid_z = (z_pupil + z_obj) / 2
    SL_x = -b_mm / 2
    SR_x = +b_mm / 2
    ax.annotate(
        "", xy=(SR_x, z_pupil), xytext=(SL_x, z_pupil),
        arrowprops=dict(arrowstyle="<->", color="0.3", lw=1.2),
    )
    ax.text(0, z_pupil - 1.2, f"$b$ = {b_mm:.1f} mm", ha="center", va="top",
            fontsize=9, color="0.25")

    # Working distance (right side)
    ax.annotate(
        "", xy=(x_max - 0.5, z_obj), xytext=(x_max - 0.5, z_pupil),
        arrowprops=dict(arrowstyle="<->", color="0.35", lw=1.0),
    )
    ax.text(x_max - 0.2, mid_z, f"WD\n{WD:.1f} mm", ha="left", va="center",
            fontsize=8, color="0.35")

    # Convergence angle
    ax.text(0, z_obj + 2.5, f"$\\Theta$ = {theta_deg:.1f}°",
            ha="center", fontsize=9.5, fontweight="bold")

    # Labels
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]  (optical axis, OpenCV frame)")

    ax.legend(loc="lower left", framealpha=0.85, fontsize=8.5, ncol=2)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_pupil - 3, z_obj + 8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    ax.set_title("CMO rayfield — per-pixel rays (X–Z plane)", fontsize=10)

    fig.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────
    for path in [OUT_PDF, OUT_PNG]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_PDF.relative_to(_REPO)}")
    print(f"Saved {OUT_PNG.relative_to(_REPO)}")

    # ── Labels manifest ─────────────────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "figure": "cmo_rayfield_xz",
        "description": "CMO rayfield X-Z interpretation from 26p model",
        "z_pupil_mm": round(z_pupil, 4),
        "z_obj_mm": round(z_obj, 4),
        "baseline_mm": round(b_mm, 3),
        "WD_mm": round(WD, 3),
        "f_obj_mm": round(f_obj, 3),
        "convergence_angle_deg": round(theta_deg, 2),
        "n_rays_per_channel": len(QL),
        "pixel_samples_u": [round(float(x), 1) for x in u_samples],
        "pixel_sample_rows_v": [round(float(x), 1) for x in v_extra],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
