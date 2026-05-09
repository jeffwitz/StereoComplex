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
from stereocomplex.viz.style import COLORS, LINEWIDTHS


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
    f_tube=50.0,
    pixel_pitch=0.05,
    cx_px=320.0,
    exaggerated=True,
):
    """Physical CMO shared-rig — (x, z) optical cut, both channels.

    Shows the full ray path for one off-axis pixel per channel:
    sensor → tube lens → sub-pupil → main objective → working plane.
    All geometric parameters are labelled.
    """
    import math

    if model is not None:
        f_obj = model.f_obj_mm
        working_distance = model.working_distance_mm
        b = model.b_mm
        f_tube = model.f_tube_mm
        pixel_pitch = model.pixel_pitch_mm
        cx_px = model.cx_principal_px

    if exaggerated:
        b = max(float(b), float(f_obj) * 0.35)

    # ---- vertical layout (z = 0 at specimen, increasing upward) ----
    z_specimen = 0.0
    z_obj = float(working_distance)
    z_pupil = z_obj - float(f_obj)
    z_tube = z_pupil + 55
    z_sensor = z_tube + max(float(f_tube) * 0.5, 40)
    b2 = float(b) / 2

    # ---- optical axis ----
    ax.axvline(x=0, ymin=0.02, ymax=0.98, color=COLORS["axis"],
               linewidth=0.7, linestyle="--", dashes=(6, 4), zorder=0)

    # ── Shared object-side optics ───────────────────────────────────

    # Main objective
    obj_h = 30
    ax.annotate("", xy=(0, z_obj + obj_h), xytext=(0, z_obj - obj_h),
                arrowprops={"arrowstyle": "<->", "color": COLORS["lens"],
                            "linewidth": 2.0, "shrinkA": 0, "shrinkB": 0})
    annotate_math(ax, (20, z_obj + obj_h + 2), "f_\\mathrm{obj}", offset=(0, 0))

    # Sub-pupils
    SL = np.array([-b2, z_pupil])
    SR = np.array([+b2, z_pupil])
    channel_info = [("left", SL), ("right", SR)]
    for ch, S in channel_info:
        name = "S_L" if ch == "left" else "S_R"
        ax.scatter(*S, s=65, color=COLORS[ch], edgecolors="white",
                   linewidth=1.0, zorder=15)
        ox = -14 if ch == "left" else 5
        annotate_math(ax, S, name, offset=(ox, -8), color=COLORS[ch])

    # Convergence point C
    C_pt = np.array([0, z_specimen])
    ax.scatter(*C_pt, s=55, color=COLORS["specimen"], edgecolors="black",
               linewidth=0.5, zorder=15)
    annotate_math(ax, C_pt, "C", offset=(3, -6))

    # Chief rays: sub-pupil → C (solid)
    for ch, S in channel_info:
        ax.plot([S[0], C_pt[0]], [S[1], C_pt[1]], color=COLORS[ch],
                linewidth=2.0, solid_capstyle="round")

    # Off-axis rays: sub-pupil → working plane (dashed, object side)
    delta_u = 100.0
    alpha_x = delta_u * float(pixel_pitch) / float(f_tube)
    for ch, S in channel_info:
        sign = -1 if ch == "left" else 1
        Px = sign * float(working_distance) * alpha_x
        P_off = np.array([Px, z_specimen])
        ax.plot([S[0], P_off[0]], [S[1], P_off[1]], color=COLORS[ch],
                linewidth=1.3, linestyle="--", dashes=(5, 3))

    # ── Image-side optics (both channels) ───────────────────────────

    # Tube lens (shared, on-axis)
    tube_h = 22
    ax.annotate("", xy=(0, z_tube + tube_h), xytext=(0, z_tube - tube_h),
                arrowprops={"arrowstyle": "<->", "color": COLORS["lens"],
                            "linewidth": 2.0, "shrinkA": 0, "shrinkB": 0})
    annotate_math(ax, (20, z_tube + tube_h + 2), "f_\\mathrm{tube}", offset=(0, 0))

    # Sensor planes: two horizontal segments, left and right
    sensor_w = 28
    sensor_gap = 8   # gap between the two sensor halves at the axis
    for ch, S in channel_info:
        sign = -1 if ch == "left" else 1
        x0 = sign * sensor_gap / 2
        x1 = sign * (sensor_gap / 2 + sensor_w)
        ax.plot([x0, x1], [z_sensor, z_sensor], color=COLORS[ch],
                linewidth=2.5, solid_capstyle="butt")

    # Principal points on each sensor
    cx_mm = float(cx_px) * float(pixel_pitch)
    pp_mm = float(pixel_pitch)  # pixel pitch in mm
    for ch, S in channel_info:
        sign = -1 if ch == "left" else 1
        cx_x = sign * cx_mm
        ax.plot(cx_x, z_sensor, marker=".", color=COLORS[ch], markersize=10,
                zorder=20)
        annotate_math(ax, (cx_x + sign * 3, z_sensor + 4), "c_x",
                      offset=(0, 2), color=COLORS[ch], fontsize=9)

    # Pixel pitch detail: show on left sensor only (to avoid clutter)
    cx_left = -cx_mm
    u0 = cx_left
    u1 = cx_left + pp_mm
    for uu in [u0, u1]:
        ax.plot([uu, uu], [z_sensor, z_sensor + 4], color=COLORS["left"],
                linewidth=0.8)
    draw_dimension(ax, (u0, z_sensor + 7), (u1, z_sensor + 7), "p",
                   offset=(0, 2))

    # Off-axis rays going upward: sub-pupil → tube lens → sensor
    for ch, S in channel_info:
        sign = -1 if ch == "left" else 1
        # Object-side ray direction
        Px = sign * float(working_distance) * alpha_x
        d_off = np.array([Px - S[0], z_specimen - S[1]])
        ray_angle = math.atan2(d_off[0], d_off[1])
        # Through tube lens plane
        dz = z_tube - z_pupil
        x_at_tube = S[0] + math.tan(ray_angle) * dz
        ax.plot([S[0], x_at_tube], [z_pupil, z_tube], color=COLORS[ch],
                linewidth=1.3, solid_capstyle="round")
        # Tube lens → sensor: converge to off-axis pixel
        sensor_x = sign * cx_mm + sign * pp_mm  # off-axis by one pixel
        ax.plot([x_at_tube, sensor_x], [z_tube, z_sensor], color=COLORS[ch],
                linewidth=1.3, solid_capstyle="round")

    # ── Alpha_x arc (left sub-pupil) ────────────────────────────────
    arc_r = 10
    theta_chief = math.atan2(C_pt[0] - SL[0], C_pt[1] - SL[1])
    ray_angle_L = math.atan2(-alpha_x * float(working_distance) - SL[0],
                              z_specimen - SL[1])
    from matplotlib.patches import Arc as _Arc
    arc = _Arc(SL, arc_r * 2, arc_r * 2, angle=0,
               theta1=math.degrees(theta_chief),
               theta2=math.degrees(ray_angle_L),
               color=COLORS["left"], linewidth=1.2)
    ax.add_patch(arc)
    mid_theta = (theta_chief + ray_angle_L) / 2
    arc_lbl = SL + np.array([math.sin(mid_theta) * (arc_r + 10),
                              math.cos(mid_theta) * (arc_r + 4)])
    annotate_math(ax, arc_lbl, "\\alpha_x", offset=(0, 0), color=COLORS["left"],
                  fontsize=9)

    # ── Dimensions ──────────────────────────────────────────────────
    draw_dimension(ax, C_pt, (0, z_obj), "Z_w", offset=(45, 0))
    draw_dimension(ax, (0, z_obj), (0, z_pupil), "f_\\mathrm{obj}", offset=(-45, 0))
    draw_dimension(ax, SL, SR, "b", offset=(0, 12))
    draw_dimension(ax, (0, z_tube), (0, z_sensor), "f_\\mathrm{tube}",
                   offset=(-45, 0))

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
