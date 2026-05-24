#!/usr/bin/env python3
"""3-D sub-pupil reconstruction from CMO rayfield (Figure 4 of the CMO paper).

Reads the 26-parameter CMO + per-arm SE(3) vector from the Pycaso
checkpoint, builds the centre-pixel rays of both channels, applies the
per-arm SE(3), and plots the two sub-pupil positions, their chief rays
toward the working plane, the baseline, and the working-plane intersections.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure4_subpupil_3d/``. Emits both PDF (paper)
and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, "src")
from stereocomplex.physics.cmo_physical import (  # noqa: E402
    CMOTelecentricStereoModel,
    _normalize,
)

MANIFEST = Path("docs/assets/cmo_paper/figure4_subpupil_3d/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "subpupil_3d"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def _apply_se3(O, d, rv, t):  # noqa: E741
    R = Rotation.from_rotvec(rv).as_matrix()
    return (R @ O.T).T + t[None, :], _normalize((R @ d.T).T)


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    state_path = (manifest_root / manifest["intermediate_state"]).resolve()
    state = np.load(state_path)

    x_26p = np.asarray(state["x_26p"], dtype=np.float64)
    image_size = tuple(int(x) for x in state["image_size"])
    W, H = image_size
    pixel_pitch_mm = float(manifest["pixel_pitch_mm"])

    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        x_26p[:14], pixel_pitch_mm=pixel_pitch_mm, image_size=image_size
    )
    OL, dL = m_tel.ray(np.array([W / 2]), np.array([H / 2]), "left")
    OR, dR = m_tel.ray(np.array([W / 2]), np.array([H / 2]), "right")
    OL_a, dL_a = _apply_se3(OL, dL, x_26p[14:17], x_26p[17:20])
    OR_a, dR_a = _apply_se3(OR, dR, x_26p[20:23], x_26p[23:26])

    OcL, OcR = OL_a[0], OR_a[0]
    dcL, dcR = dL_a[0], dR_a[0]

    b = float(np.linalg.norm(OcR - OcL))
    WD = float(np.mean([float(state["opt_t"][i][2]) for i in range(10)]))
    theta_deg = float(np.degrees(np.arctan2(b / 2, WD)))

    def hit_plane(O, d, z):  # noqa: E741
        tw = (z - O[2]) / d[2] if abs(d[2]) > 1e-12 else z
        return O + tw * d

    PL_w = hit_plane(OcL, dcL, WD)
    PR_w = hit_plane(OcR, dcR, WD)

    fig = plt.figure(figsize=tuple(manifest["figure_size"]))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*OcL, c="#2a55c7", s=200, marker="o",
               edgecolors="black", linewidth=0.5,
               label=f"O$_L$ ({OcL[0]:.1f}, {OcL[1]:.1f}, {OcL[2]:.1f})")
    ax.scatter(*OcR, c="#c4332a", s=200, marker="o",
               edgecolors="black", linewidth=0.5,
               label=f"O$_R$ ({OcR[0]:.1f}, {OcR[1]:.1f}, {OcR[2]:.1f})")

    ray_len = float(manifest["chief_ray_length_mm"])
    ray_L_end = OcL + ray_len * dcL
    ray_R_end = OcR + ray_len * dcR
    ax.plot([OcL[0], ray_L_end[0]], [OcL[1], ray_L_end[1]], [OcL[2], ray_L_end[2]],
            color="#2a55c7", lw=1.5, alpha=0.7, label="Chief ray L")
    ax.plot([OcR[0], ray_R_end[0]], [OcR[1], ray_R_end[1]], [OcR[2], ray_R_end[2]],
            color="#c4332a", lw=1.5, alpha=0.7, label="Chief ray R")

    ax.scatter(*PL_w, c="green", s=120, marker="x", linewidth=1.5,
               label=f"Working plane hit L (z={WD:.1f})")
    ax.scatter(*PR_w, c="green", s=120, marker="x", linewidth=1.5)

    ax.plot([OcL[0], OcR[0]], [OcL[1], OcR[1]], [OcL[2], OcR[2]],
            "k--", lw=2, alpha=0.6)
    bc = (OcL + OcR) / 2
    ax.text(bc[0], bc[1], bc[2] - 2, f"b = {b:.1f} mm",
            fontsize=12, ha="center", fontweight="bold")

    wp_x = np.linspace(min(PL_w[0], PR_w[0]) - 5, max(PL_w[0], PR_w[0]) + 5, 2)
    wp_y = np.linspace(min(PL_w[1], PR_w[1]) - 5, max(PL_w[1], PR_w[1]) + 5, 2)
    ax.plot(wp_x, wp_y, np.full(2, WD), "gray", lw=1, alpha=0.3, ls="--")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"3D sub-pupil reconstruction\n"
        f"b={b:.1f} mm, WD={WD:.1f} mm, $\\theta$={theta_deg:.1f}°"
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
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    render(manifest, MANIFEST.parent, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
