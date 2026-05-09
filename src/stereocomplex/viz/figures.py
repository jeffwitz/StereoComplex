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

    Correct optical order (top → bottom):
      sensor → tube lens → afocal space → sub-pupils (aperture stop)
      → main objective → working distance → object plane (C).

    In the afocal region rays are parallel to the optical axis; this is
    the defining property of an infinity-corrected CMO and justifies
    modelling each channel as a virtual pinhole at its sub-pupil.
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

    # ---- geometry (z = 0 at object plane, +z upward) ----
    z_object = 0.0
    z_objective = float(working_distance)               # main objective
    z_pupil = z_objective - float(f_obj)                # sub-pupils = back focal plane
    z_tube = z_objective + 55                           # tube lens (afocal gap above objective)
    z_sensor = z_tube + max(float(f_tube) * 0.5, 40)   # sensor
    b2 = float(b) / 2
    x_span = 65

    # ---- optical planes ----
    plane_style = {"color": "#cccccc", "linewidth": 0.6, "zorder": 0}
    planes = [
        (z_object,   "object plane ($C$)"),
        (z_objective, "main objective ($f_\\mathrm{obj}$)"),
        (z_pupil,    "sub-pupils / aperture stop ($S_L, S_R$)"),
        (z_tube,     "tube lens ($f_\\mathrm{tube}$)"),
        (z_sensor,   "sensor ($p\\;=\\;\\mathrm{pixel\\,pitch}$)"),
    ]
    for z, label in planes:
        ax.plot([-x_span, x_span], [z, z], **plane_style)
        ax.text(x_span + 2, z, label, va="center", ha="left", fontsize=8,
                color=COLORS["annotation"])

    # ---- optical axis ----
    ax.plot([0, 0], [z_object - 10, z_sensor + 15], color=COLORS["axis"],
            linewidth=0.7, linestyle="--", dashes=(6, 4), zorder=0)

    # ---- sub-pupils (at back focal plane of objective) ----
    SL = np.array([-b2, z_pupil])
    SR = np.array([+b2, z_pupil])
    channels = [("left", SL, "S_L"), ("right", SR, "S_R")]
    for ch, S, name in channels:
        ax.scatter(*S, s=80, color=COLORS[ch], edgecolors="white",
                   linewidth=1.5, zorder=15)
        ox = -16 if ch == "left" else 6
        annotate_math(ax, S, name, offset=(ox, -10), color=COLORS[ch])

    # ---- convergence point C (object plane) ----
    C_pt = np.array([0, z_object])
    ax.scatter(*C_pt, s=60, color=COLORS["specimen"], edgecolors="black",
               linewidth=0.5, zorder=15)
    annotate_math(ax, C_pt, "C", offset=(4, -8))

    # ---- chief rays: S → C (converging through objective) ----
    for ch, S, _ in channels:
        ax.plot([S[0], C_pt[0]], [S[1], C_pt[1]], color=COLORS[ch],
                linewidth=2.2, solid_capstyle="round", zorder=5)

    # ---- one chief ray + one off-axis ray per channel ----
    delta_u = 100.0
    alpha_x = delta_u * float(pixel_pitch) / float(f_tube)
    for ch, S, _ in channels:
        sign = -1 if ch == "left" else 1
        # --- chief ray (solid, already drawn above: S → C) ---
        # --- afocal portion of chief ray: VERTICAL from S up to tube lens ---
        #     The chief ray is at constant x = S[0] throughout the afocal space
        #     because in an infinity-corrected system, a chief ray (through the
        #     aperture-stop centre) emerges from the tube lens parallel to the axis.
        ax.plot([S[0], S[0]], [z_pupil, z_tube], color=COLORS[ch],
                linewidth=2.2, solid_capstyle="round", zorder=5)
        # Chief ray at tube lens: dot, then connect to sensor pixel
        ax.scatter(S[0], z_tube, s=25, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=1.0)
        # Tube lens → sensor pixel (chief ray converges to principal point)
        cx_mm = float(cx_px) * float(pixel_pitch)
        sensor_cx = sign * cx_mm
        ax.plot([S[0], sensor_cx], [z_tube, z_sensor], color=COLORS[ch],
                linewidth=2.2, zorder=5)
        ax.scatter(sensor_cx, z_sensor, s=30, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=1.0)

        # --- off-axis ray (dashed on object side, solid on image side) ---
        # Object side: S → working-plane point
        Px = sign * float(working_distance) * alpha_x
        P_off = np.array([Px, z_object])
        ax.plot([S[0], P_off[0]], [S[1], P_off[1]], color=COLORS[ch],
                linewidth=1.2, linestyle="--", dashes=(6, 4), zorder=4)
        # Afocal space: off-axis ray is at a DIFFERENT x than the sub-pupil
        # (it encodes a different pixel).  It is still parallel to the axis.
        x_off_afocal = S[0] + sign * delta_u * float(pixel_pitch)
        ax.plot([x_off_afocal, x_off_afocal], [z_pupil, z_tube], color=COLORS[ch],
                linewidth=1.2, zorder=4)
        ax.scatter(x_off_afocal, z_tube, s=18, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=0.5)
        # Tube lens → sensor (off-axis pixel)
        sensor_off = sign * (cx_mm + delta_u * float(pixel_pitch))
        ax.plot([x_off_afocal, sensor_off], [z_tube, z_sensor], color=COLORS[ch],
                linewidth=1.2, zorder=4)
        ax.scatter(sensor_off, z_sensor, s=18, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=0.5)

    # ---- afocal region annotation ----
    ax.text(0, (z_pupil + z_tube) / 2, "afocal (rays ∥ axis)",
            ha="center", va="center", fontsize=8, color=COLORS["annotation"],
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "#cccccc", "alpha": 0.85})

    # ---- sensor detail: pixel pitch p (left sensor) ----
    cx_mm = float(cx_px) * float(pixel_pitch)
    p_mm = float(pixel_pitch)
    u0, u1 = -cx_mm, -cx_mm + p_mm
    for uu in [u0, u1]:
        ax.plot([uu, uu], [z_sensor, z_sensor + 5], color=COLORS["left"],
                linewidth=0.8, zorder=10)
    ax.annotate("", xy=(u1, z_sensor + 9), xytext=(u0, z_sensor + 9),
                arrowprops={"arrowstyle": "<->", "color": "#666666", "lw": 0.8})
    ax.text((u0 + u1) / 2, z_sensor + 13, "$p$", fontsize=9, ha="center",
            color="#666666")

    # ---- stereo angle gamma (right side) ----
    gamma = math.atan2(b2, float(working_distance))
    gamma_deg = math.degrees(gamma)
    # Small arc at C showing gamma
    arc_r = 18
    from matplotlib.patches import Arc as _Arc
    ax.add_patch(_Arc(C_pt, arc_r * 2, arc_r * 2, angle=0,
                      theta1=-gamma_deg, theta2=gamma_deg,
                      color="#666666", linewidth=1.0, zorder=12))
    ax.text(C_pt[0] + arc_r + 10, C_pt[1] - 2,
            f"$\\gamma={gamma_deg:.1f}^\\circ$", fontsize=9, color="#666666")

    # ---- dimensions ----
    draw_dimension(ax, C_pt, (0, z_objective), "Z_w", offset=(75, 0))
    draw_dimension(ax, (0, z_objective), (0, z_pupil), "f_\\mathrm{obj}", offset=(-75, 0))
    draw_dimension(ax, SL, SR, "b", offset=(0, 16))
    draw_dimension(ax, (0, z_tube), (0, z_sensor), "f_\\mathrm{tube}", offset=(-75, 0))

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
