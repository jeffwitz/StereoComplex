"""Per-model diagram composers.

Each function takes the relevant physical model (or its parameters) and
draws a pedagogical 2D optical-cut diagram.  The diagrams use the **same**
numerical models as the fitting code, so they stay in sync automatically.
"""

from __future__ import annotations

import numpy as np

from stereocomplex.viz.primitives import (
    annotate_math,
    draw_dimension,
    draw_lens,
    draw_optical_axis,
    draw_ray,
    draw_sensor,
    draw_specimen,
)
from stereocomplex.viz.style import COLORS


def diagram_pinhole_stereo(
    ax,
    *,
    O_left=(0, 150),
    O_right=(0, 150),
    specimen=(0, 0),
    sensor_offset=50,
    sensor_width=40,
    sensor_height=8,
    baseline_label="B",
):
    """Classical pinhole stereo pair in the (x, z) optical cut.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    O_left, O_right : (2,) array_like
        Camera centres in (x, z) mm.  Will be plotted as filled circles.
    specimen : (2,) array_like  — object point (x, z) mm.
    sensor_offset : float  — z-offset from camera centre to sensor.
    sensor_width, sensor_height : float  — sensor dimensions in mm.
    baseline_label : str  — LaTeX label for the baseline dimension.
    """
    OL = np.asarray(O_left, dtype=float).reshape(2)
    OR = np.asarray(O_right, dtype=float).reshape(2)
    sp = np.asarray(specimen, dtype=float).reshape(2)

    # Optical centres
    ax.scatter(*OL, s=80, color=COLORS["left"], edgecolors="white",
               linewidth=0.8, zorder=15)
    ax.scatter(*OR, s=80, color=COLORS["right"], edgecolors="white",
               linewidth=0.8, zorder=15)
    annotate_math(ax, OL, "O_L", offset=(-14, 0))
    annotate_math(ax, OR, "O_R", offset=(6, 0))

    # Sensors (placed at camera centre + sensor_offset in z)
    draw_sensor(ax, OL + (0, sensor_offset), sensor_width, sensor_height,
                channel="left", label="L")
    draw_sensor(ax, OR + (0, sensor_offset), sensor_width, sensor_height,
                channel="right", label="R")

    # Specimen
    draw_specimen(ax, sp, label="X")

    # Central rays from camera centres to specimen
    dL = sp - OL
    dR = sp - OR
    draw_ray(ax, OL, dL, length=np.linalg.norm(dL), channel="left",
             width=1.6)
    draw_ray(ax, OR, dR, length=np.linalg.norm(dR), channel="right",
             width=1.6)

    # Optical axis
    mid_z = (OL[1] + sp[1]) / 2
    draw_optical_axis(ax, (0, sp[1] - 10), (0, OL[1] + sensor_offset + 15))

    # Baseline
    draw_dimension(ax, OL, OR, baseline_label, offset=(0, -8))

    # Sensor → camera centre guide lines (dashed)
    for O, ch in [(OL, "left"), (OR, "right")]:
        draw_ray(ax, O, (0, 1), length=sensor_offset, channel=ch, style="dashed",
                 width=0.8)

    ax.set_aspect("equal")
    ax.axis("off")


def diagram_cmo_physical(
    ax,
    model=None,
    *,
    f_obj=80.0,
    working_distance=120.0,
    b=8.0,
    exaggerated=True,
):
    """Physical CMO shared-rig diagram in the (x, z) optical cut.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    model : CMOPhysicalStereoModel | None
        If given, geometry is extracted directly from the model.  Otherwise
        the keyword arguments are used to build a diagram-specific instance.
    f_obj, working_distance, b : float
        Main-objective focal length, working distance, and sub-pupil baseline
        in mm.  Only used when *model* is None.
    exaggerated : bool
        If True, use a larger b/f_obj ratio for visual clarity.  The caption
        should note that the geometry is pedagogically exaggerated.
    """
    if model is not None:
        f_obj = model.f_obj_mm
        working_distance = model.working_distance_mm
        b = model.b_mm

    if exaggerated:
        b = max(float(b), float(f_obj) * 0.3)

    # Derived positions in the (x, z) plane
    z_wp = 0.0                         # working plane at z = 0
    z_obj = float(working_distance)     # main objective at z = working_distance
    z_pupil = z_obj - float(f_obj)      # sub-pupil plane
    b2 = float(b) / 2

    # Main objective
    draw_lens(ax, (0, z_obj), radius=f_obj / 2.5, label="f_\\mathrm{obj}",
              kind="biconvex")

    # Optical axis
    draw_optical_axis(ax, (0, z_wp - 10), (0, z_pupil - 15))

    # Sub-pupils
    SL = np.array([-b2, z_pupil])
    SR = np.array([+b2, z_pupil])
    for S, ch, name in [(SL, "left", "S_L"), (SR, "right", "S_R")]:
        ax.scatter(*S, s=70, color=COLORS[ch], edgecolors="white",
                   linewidth=0.8, zorder=15)
        annotate_math(ax, S, name, offset=(-12 if ch == "left" else 4, -8),
                      color=COLORS[ch])

    # Working plane point
    specimen = np.array([0, z_wp])
    draw_specimen(ax, specimen, label="C")

    # Chief rays from sub-pupils to convergence point
    dL = specimen - SL
    dR = specimen - SR
    draw_ray(ax, SL, dL, length=np.linalg.norm(dL), channel="left", width=1.6)
    draw_ray(ax, SR, dR, length=np.linalg.norm(dR), channel="right", width=1.6)

    # Marginal rays (off-axis) — lightly, for non-central illustration
    for offset_x in [-b2 * 0.6, b2 * 0.6]:
        wp_point = np.array([offset_x, z_wp])
        for S, ch in [(SL, "left"), (SR, "right")]:
            d = wp_point - S
            draw_ray(ax, S, d, length=np.linalg.norm(d), channel=ch,
                     style="dotted", width=0.6)

    # Working distance annotation
    draw_dimension(ax, specimen, (0, z_obj), "Z_w", offset=(12, 0))

    # Baseline annotation
    draw_dimension(ax, SL, SR, "b", offset=(0, 8))

    ax.set_aspect("equal")
    ax.axis("off")


def diagram_greenough(
    ax,
    *,
    O_left=(-15, 0),
    O_right=(15, 0),
    specimen=(0, 40),
    convergence_angle_deg=20.0,
    sensor_offset=30,
):
    """Greenough stereo microscope: two independent convergent objectives.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    O_left, O_right : (2,) array_like  — objective centres (x, z) mm.
    specimen : (2,) array_like  — object point (x, z) mm.
    convergence_angle_deg : float  — inward angle of each optical axis.
    sensor_offset : float  — z-offset from objective centre to sensor.
    """
    import math
    OL = np.asarray(O_left, dtype=float).reshape(2)
    OR = np.asarray(O_right, dtype=float).reshape(2)
    sp = np.asarray(specimen, dtype=float).reshape(2)

    # Camera centres (objectives)
    ax.scatter(*OL, s=70, color=COLORS["left"], edgecolors="white",
               zorder=15)
    ax.scatter(*OR, s=70, color=COLORS["right"], edgecolors="white",
               zorder=15)
    annotate_math(ax, OL, "O_L", offset=(-14, -2))
    annotate_math(ax, OR, "O_R", offset=(6, -2))

    # Optical axes (convergent)
    theta = math.radians(convergence_angle_deg / 2)
    axis_L = np.array([math.sin(theta), math.cos(theta)])
    axis_R = np.array([-math.sin(theta), math.cos(theta)])
    for O, axis, ch in [(OL, axis_L, "left"), (OR, axis_R, "right")]:
        draw_ray(ax, O, axis, length=80, channel=ch, style="dashed", width=0.8)

    # Sensors
    draw_sensor(ax, OL + axis_L * sensor_offset, 30, 6, channel="left")
    draw_sensor(ax, OR + axis_R * sensor_offset, 30, 6, channel="right")

    # Specimen
    draw_specimen(ax, sp, label="X")

    # Chief rays from objectives to specimen
    dL = sp - OL
    dR = sp - OR
    draw_ray(ax, OL, dL, length=np.linalg.norm(dL), channel="left", width=1.6)
    draw_ray(ax, OR, dR, length=np.linalg.norm(dR), channel="right", width=1.6)

    ax.set_aspect("equal")
    ax.axis("off")
