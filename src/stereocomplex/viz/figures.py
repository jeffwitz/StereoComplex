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
    """Physical CMO shared-rig — (x, z) optical cut, both channels."""
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

    # ---- geometry (z=0 at specimen, +z upward) ----
    z_specimen = 0.0
    z_obj = float(working_distance)
    z_pupil = z_obj - float(f_obj)
    z_tube = z_pupil + 50
    z_sensor = z_tube + max(float(f_tube) * 0.5, 40)
    b2 = float(b) / 2
    # Horizontal span for plane lines
    x_left = -55
    x_right = 55

    # ---- optical planes (drawn as simple horizontal lines) ----
    plane_style = {"color": "#cccccc", "linewidth": 0.6, "zorder": 0}
    planes = [
        (z_specimen, "working plane ($Z_w$)"),
        (z_obj, "objective ($f_\\mathrm{obj}$)"),
        (z_pupil, "sub-pupils ($S_L, S_R$)"),
        (z_tube, "tube lens ($f_\\mathrm{tube}$)"),
        (z_sensor, "sensor ($c_x, p$)"),
    ]
    for z, label in planes:
        ax.plot([x_left, x_right], [z, z], **plane_style)
        ax.text(x_right + 2, z, label, va="center", ha="left", fontsize=8,
                color=COLORS["annotation"])

    # ---- optical axis ----
    ax.plot([0, 0], [z_specimen - 5, z_sensor + 15], color=COLORS["axis"],
            linewidth=0.7, linestyle="--", dashes=(6, 4), zorder=0)

    # ---- sub-pupils ----
    SL = np.array([-b2, z_pupil])
    SR = np.array([+b2, z_pupil])
    channels = [("left", SL, "S_L"), ("right", SR, "S_R")]
    for ch, S, name in channels:
        ax.scatter(*S, s=80, color=COLORS[ch], edgecolors="white",
                   linewidth=1.5, zorder=15)
        ox = -16 if ch == "left" else 6
        annotate_math(ax, S, name, offset=(ox, -10), color=COLORS[ch])

    # ---- convergence point C ----
    C_pt = np.array([0, z_specimen])
    ax.scatter(*C_pt, s=60, color=COLORS["specimen"], edgecolors="black",
               linewidth=0.5, zorder=15)
    annotate_math(ax, C_pt, "C", offset=(4, -8))

    # ---- chief rays: S → C ----
    for ch, S, _ in channels:
        ax.plot([S[0], C_pt[0]], [S[1], C_pt[1]], color=COLORS[ch],
                linewidth=2.2, solid_capstyle="round", zorder=5)

    # ---- one off-axis ray per channel ----
    delta_u = 100.0
    alpha_x = delta_u * float(pixel_pitch) / float(f_tube)
    for ch, S, _ in channels:
        sign = -1 if ch == "left" else 1
        # Object side: S → working plane
        Px = sign * float(working_distance) * alpha_x
        P_off = np.array([Px, z_specimen])
        ax.plot([S[0], P_off[0]], [S[1], P_off[1]], color=COLORS[ch],
                linewidth=1.3, linestyle="--", dashes=(6, 4), zorder=4)
        # Image side: S → tube lens (collimated space)
        ray_angle = math.atan2(P_off[0] - S[0], P_off[1] - S[1])
        x_at_tube = S[0] + math.tan(ray_angle) * (z_tube - z_pupil)
        ax.plot([S[0], x_at_tube], [z_pupil, z_tube], color=COLORS[ch],
                linewidth=1.3, zorder=4)
        # Deviation point at tube lens
        ax.scatter(x_at_tube, z_tube, s=25, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=1.0)
        # Tube lens → sensor
        sensor_x = sign * (float(cx_px) + delta_u) * float(pixel_pitch)
        ax.plot([x_at_tube, sensor_x], [z_tube, z_sensor], color=COLORS[ch],
                linewidth=1.3, zorder=4)
        ax.scatter(sensor_x, z_sensor, s=25, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=1.0)

    # ---- sensor detail: c_x + pixel pitch (left sensor) ----
    cx_mm = float(cx_px) * float(pixel_pitch)
    p_mm = float(pixel_pitch)
    ax.scatter(-cx_mm, z_sensor, s=35, color="black", zorder=20)
    annotate_math(ax, (-cx_mm, z_sensor), "c_x", offset=(-10, 6))
    u0, u1 = -cx_mm, -cx_mm + p_mm
    for uu in [u0, u1]:
        ax.plot([uu, uu], [z_sensor, z_sensor + 5], color=COLORS["left"],
                linewidth=0.8, zorder=10)
    ax.annotate("", xy=(u1, z_sensor + 9), xytext=(u0, z_sensor + 9),
                arrowprops={"arrowstyle": "<->", "color": "#666666", "lw": 0.8})
    ax.text((u0 + u1) / 2, z_sensor + 13, "$p$", fontsize=9, ha="center",
            color="#666666")

    # ---- alpha_x arc (left sub-pupil) ----
    arc_r = 14
    theta_chief = math.atan2(C_pt[0] - SL[0], C_pt[1] - SL[1])
    theta_off = math.atan2(-alpha_x * float(working_distance) - SL[0],
                            z_specimen - SL[1])
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(_Arc(SL, arc_r * 2, arc_r * 2, angle=0,
                      theta1=math.degrees(theta_chief),
                      theta2=math.degrees(theta_off),
                      color=COLORS["left"], linewidth=1.3, zorder=12))
    mid_t = (theta_chief + theta_off) / 2
    ax.text(SL[0] + math.sin(mid_t) * (arc_r + 14),
            SL[1] + math.cos(mid_t) * (arc_r + 8),
            "$\\alpha_x$", fontsize=9, color=COLORS["left"], ha="center")

    # ---- gamma angle (right side) ----
    gamma = math.atan2(b2, float(f_obj))
    ax.text(SR[0] + 20, SR[1] - 12,
            f"$\\gamma={math.degrees(gamma):.1f}^\\circ$",
            fontsize=9, color=COLORS["right"])

    # ---- dimensions ----
    draw_dimension(ax, C_pt, (0, z_obj), "Z_w", offset=(60, 0))
    draw_dimension(ax, (0, z_obj), (0, z_pupil), "f_\\mathrm{obj}", offset=(-60, 0))
    draw_dimension(ax, SL, SR, "b", offset=(0, 16))
    draw_dimension(ax, (0, z_tube), (0, z_sensor), "f_\\mathrm{tube}", offset=(-60, 0))

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
