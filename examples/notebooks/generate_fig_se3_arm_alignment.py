#!/usr/bin/env python3
r"""Schematic of the per-channel SE(3) arm misalignment in the CMO model.

Conceptual diagram (no measurement data) for \S3.5.6: it shows the ideal
telecentric CMO skeleton---two off-axis sub-pupils separated by the baseline
$b$, emitting chief rays that converge on the optical axis at the working
distance $WD$ with full convergence angle $\Theta$---and, superimposed, the
*actual* orientation of each optical arm after the fitted rigid SE(3) rotation
($\sim2.5^\circ$ left, $\sim3.7^\circ$ right). These small rotations are the
residual $Z_0^0$ piston the rayfield diagnostic identified; they originate in
the assembly tolerances of the internal relay optics, not in a macroscopic
flexure of the head.

The tilt is drawn exaggerated (factor read from the manifest) for visibility;
the caption and the manifest carry the true per-channel magnitudes and the
dominant (x-axis) rotation component.

All numbers come from
``docs/assets/cmo_paper/figure_se3_arm_alignment/se3_schematic.json``; this
script is layout-only, per the no-orphan-figures rule.

Run::

    rtk .venv/bin/python examples/notebooks/generate_fig_se3_arm_alignment.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

REPO = Path(__file__).resolve().parents[2]
ASSET = REPO / "docs/assets/cmo_paper/figure_se3_arm_alignment"


def _rot(v: np.ndarray, deg: float) -> np.ndarray:
    """Rotate a 2-D vector counter-clockwise by ``deg`` degrees."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def main() -> int:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"]})
    m = json.loads((ASSET / "se3_schematic.json").read_text())
    b, wd, theta = m["baseline_mm"], m["working_distance_mm"], m["convergence_angle_deg"]
    a_left, a_right = m["left_arm_rotation_deg"], m["right_arm_rotation_deg"]
    exag = m["exaggeration_factor"]

    s_left = np.array([-b / 2, 0.0])
    s_right = np.array([b / 2, 0.0])
    obj = np.array([0.0, -wd])

    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    ax.set_aspect("equal")

    # pupil and working planes
    ax.plot([-b / 2 - 7, b / 2 + 7], [0, 0], color="0.4", lw=1.2)
    ax.plot([-13, 13], [-wd, -wd], color="0.4", lw=1.2)
    ax.text(b / 2 + 8, 0, "exit-pupil plane", va="center", fontsize=9, color="0.3")
    ax.text(13.5, -wd, "working plane", va="center", fontsize=9, color="0.3")

    # optical axis (the dashed vertical line; anchored by O and Theta below)
    ax.plot([0, 0], [4, -wd - 5], ls=(0, (6, 4)), color="0.5", lw=1.0)

    # sub-pupils + object
    ax.add_patch(Circle(s_left, 1.1, color="k", zorder=5))
    ax.add_patch(Circle(s_right, 1.1, color="k", zorder=5))
    ax.add_patch(Circle(obj, 1.1, color="k", zorder=5))
    ax.text(s_left[0] - 3.0, 2.6, "$S_L$", fontsize=12)
    ax.text(s_right[0] + 1.4, 2.6, "$S_R$", fontsize=12)
    ax.text(1.6, -wd - 3.8, "object point $O$", fontsize=10)

    colors = {"L": "tab:blue", "R": "tab:red"}
    # outward signs: left arm rotates CW (-), right arm CCW (+) -> no crossing
    for s_pup, ang, key, sign in [(s_left, a_left, "L", -1), (s_right, a_right, "R", +1)]:
        d_ideal = (obj - s_pup) / np.linalg.norm(obj - s_pup)
        # ideal chief ray (dashed grey): converges at O
        ax.plot([s_pup[0], obj[0]], [s_pup[1], obj[1]], ls=(0, (5, 3)),
                color="0.55", lw=1.3, zorder=3)
        # actual arm (solid), shorter segment so the tilt reads near the pupil
        d_act = _rot(d_ideal, sign * ang * exag)
        end = s_pup + d_act * 0.62 * wd
        ax.plot([s_pup[0], end[0]], [s_pup[1], end[1]], color=colors[key], lw=2.2, zorder=4)
        # rotation arc + angle label between ideal and actual ray
        a_id = np.degrees(np.arctan2(d_ideal[1], d_ideal[0]))
        a_ac = np.degrees(np.arctan2(d_act[1], d_act[0]))
        ax.add_patch(Arc(s_pup, 26, 26, angle=0, theta1=min(a_id, a_ac),
                         theta2=max(a_id, a_ac), color=colors[key], lw=1.8, zorder=4))
        lab = _rot(d_ideal, sign * ang * exag / 2) * 17
        ax.text(s_pup[0] + lab[0] - 3.4 * (key == "L"), s_pup[1] + lab[1],
                rf"${ang:.1f}^\circ$", color=colors[key], fontsize=12, fontweight="bold")

    # baseline double arrow
    ax.add_patch(FancyArrowPatch(s_left + np.array([1.3, 6.0]), s_right + np.array([-1.3, 6.0]),
                                 arrowstyle="<->", mutation_scale=12, color="k", lw=1.0))
    ax.text(0, 8.6, f"baseline $b={b:.1f}$ mm", ha="center", fontsize=10)

    # working distance arrow
    ax.add_patch(FancyArrowPatch([-b / 2 - 4, 0], [-b / 2 - 4, -wd],
                                 arrowstyle="<->", mutation_scale=12, color="k", lw=1.0))
    ax.text(-b / 2 - 5, -wd / 2, f"$WD={wd:.1f}$ mm", rotation=90, va="center",
            ha="right", fontsize=10)

    # convergence angle at O
    ax.add_patch(Arc(obj, 26, 26, angle=0, theta1=57, theta2=123, color="0.4", lw=1.2))
    ax.text(0, -wd + 19, rf"$\Theta={theta:.1f}^\circ$", ha="center", fontsize=10, color="0.3")

    # legend + magnitude note (placed top-right, clear of the geometry)
    ax.plot([], [], ls=(0, (5, 3)), color="0.55", lw=1.3,
            label="ideal CMO skeleton (converges at $O$)")
    ax.plot([], [], color="tab:blue", lw=2.2, label="actual arm after SE(3) rotation")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2,
              fontsize=8.5, framealpha=0.95)
    note = (rf"tilt drawn $\times{exag}$;" "\n"
            "rigid SE(3) rotation,\n"
            r"dominant axis $x$." "\n"
            rf"real: left ${a_left:.1f}^\circ$, right ${a_right:.1f}^\circ$")
    ax.text(b / 2 + 8, -wd * 0.62, note, fontsize=8.5, color="0.25", va="top",
            bbox={"boxstyle": "round,pad=0.4", "fc": "0.96", "ec": "0.7", "lw": 0.6})

    ax.set_xlim(-b / 2 - 10, b / 2 + 30)
    ax.set_ylim(-wd - 12, 15)
    ax.axis("off")
    fig.tight_layout()

    pdf = ASSET / "se3_arm_alignment.pdf"
    png = ASSET / "se3_arm_alignment.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf.relative_to(REPO)} + .png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
