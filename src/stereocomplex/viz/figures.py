"""Per-model diagram composers.

Each function takes the relevant physical model (or its parameters) and
draws a pedagogical 2D optical-cut diagram.  The diagrams use the **same**
numerical models as the fitting code, so they stay in sync automatically.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Arc as _Arc

from stereocomplex.viz.primitives import (
    annotate_math,
    draw_dimension,
)
from stereocomplex.viz.style import COLORS


def diagram_pinhole_stereo(
    ax,
    *,
    O_left=(-6, 80),
    O_right=(6, 80),
    specimen=(0, 0),
    pixel_pitch=0.05,
    baseline_label="B",
):
    """Classical pinhole stereo pair — (x, z) optical cut.

    Two camera centres O_L, O_R separated by baseline B, viewing a
    specimen point X.  One chief ray per camera.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    O_left, O_right : tuple of float
        Left and right optical centres in (x, z) millimetres.
    specimen : tuple of float
        Specimen position in (x, z) millimetres.
    pixel_pitch : float
        Pixel pitch in millimetres for sensor-scale indicators.
    baseline_label : str
        Text used for the baseline dimension annotation.
    """
    OL = np.array([float(O_left[0]), float(O_left[1])])
    OR = np.array([float(O_right[0]), float(O_right[1])])
    sp = np.array([float(specimen[0]), float(specimen[1])])
    x_span = abs(OL[0]) + 25
    z_sensor = OL[1] + 25
    p_mm = float(pixel_pitch)

    # ---- planes ----
    plane_style = {"color": "#cccccc", "linewidth": 0.6, "zorder": 0}
    for z, label in [
        (sp[1],    "object plane ($X$)"),
        (OL[1],    "camera centres ($O_L, O_R$)"),
        (z_sensor, "sensor ($p\\;=\\;\\mathrm{pixel\\,pitch}$)"),
    ]:
        ax.plot([-x_span, x_span], [z, z], **plane_style)
        ax.text(x_span + 2, z, label, va="center", ha="left",
                fontsize=8, color=COLORS["annotation"])

    # ---- optical axis ----
    ax.plot([0, 0], [sp[1] - 10, z_sensor + 10], color=COLORS["axis"],
            linewidth=0.7, linestyle="--", dashes=(6, 4), zorder=0)

    # ---- camera centres + sensor dots ----
    for O, ch, name in [(OL, "left", "O_L"), (OR, "right", "O_R")]:
        ax.scatter(*O, s=80, color=COLORS[ch], edgecolors="white",
                   linewidth=1.0, zorder=15)
        ox = -14 if ch == "left" else 6
        annotate_math(ax, O, name, offset=(ox, -8), color=COLORS[ch])
        # Sensor pixel dot above camera centre
        ax.scatter(O[0], z_sensor, s=25, color=COLORS[ch], zorder=15,
                   edgecolors="white", linewidth=0.5)
        # Vertical guide line (dashed)
        ax.plot([O[0], O[0]], [O[1], z_sensor], color=COLORS[ch],
                linewidth=0.8, linestyle="--", dashes=(4, 4), zorder=3)

    # ---- specimen ----
    ax.scatter(*sp, s=60, color=COLORS["specimen"], edgecolors="black",
               linewidth=0.5, zorder=15)
    annotate_math(ax, sp, "X", offset=(4, -8))

    # ---- chief rays ----
    for O, ch in [(OL, "left"), (OR, "right")]:
        ax.plot([O[0], sp[0]], [O[1], sp[1]], color=COLORS[ch],
                linewidth=2.2, solid_capstyle="round", zorder=5)

    # ---- pixel pitch detail (left sensor) ----
    u0, u1 = OL[0], OL[0] + p_mm
    for uu in [u0, u1]:
        ax.plot([uu, uu], [z_sensor, z_sensor + 4], color=COLORS["left"],
                linewidth=0.7, zorder=10)
    ax.annotate("", xy=(u1, z_sensor + 8), xytext=(u0, z_sensor + 8),
                arrowprops={"arrowstyle": "<->", "color": "#666666", "lw": 0.8})
    ax.text((u0 + u1) / 2, z_sensor + 12, "$p$", fontsize=9, ha="center",
            color="#666666")

    # ---- dimensions ----
    draw_dimension(ax, OL, OR, baseline_label, offset=(0, -10))
    draw_dimension(ax, sp, (0, OL[1]), "Z", offset=(25, 0))

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

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    model : CMOPhysicalStereoModel, optional
        Pre-fitted model; overrides the individual geometry parameters.
    f_obj : float
        Objective focal length in millimetres.
    working_distance : float
        Distance from objective to specimen in millimetres.
    b : float
        Stereo baseline in millimetres.
    f_tube : float
        Tube lens focal length in millimetres.
    pixel_pitch : float
        Sensor pixel pitch in millimetres.
    cx_px : float
        Principal point x-coordinate in pixels.
    exaggerated : bool
        If True, exaggerate sub-pupil separation for visual clarity.
    """

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

    # ---- one chief ray per channel (object → afocal → sensor) ----
    # Only the chief ray is shown: it goes through the centre of the
    # sub-pupil, is parallel to the optical axis in the afocal space,
    # and converges at the principal pixel of its sensor.
    cx_mm = float(cx_px) * float(pixel_pitch)
    for ch, S, _ in channels:
        sign = -1 if ch == "left" else 1
        sensor_cx = sign * cx_mm
        # 1) Object side (S → C) already drawn above.
        # 2) Afocal portion: VERTICAL from sub-pupil up to tube lens.
        ax.plot([S[0], S[0]], [z_pupil, z_tube], color=COLORS[ch],
                linewidth=2.2, solid_capstyle="round", zorder=5)
        ax.scatter(S[0], z_tube, s=25, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=1.0)
        # 3) Tube lens → sensor principal pixel.
        ax.plot([S[0], sensor_cx], [z_tube, z_sensor], color=COLORS[ch],
                linewidth=2.2, zorder=5)
        ax.scatter(sensor_cx, z_sensor, s=30, color=COLORS[ch], zorder=16,
                   edgecolors="white", linewidth=1.0)

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
    O_left=(-22, 60),
    O_right=(22, 60),
    specimen=(0, 0),
    sensor_offset=30,
    pixel_pitch=0.05,
):
    """Greenough stereo microscope — (x, z) optical cut.

    Two independent objectives with convergent optical axes.  Each
    channel has its own tilted sensor and tilted optical axis; there
    is no shared main objective.  This is the defining difference
    from a standard pinhole stereo pair (where axes are parallel).

    Geometric convention (consistent with other diagrams):
    objectives at z > 0, specimen at z = 0 below them.  Each optical
    axis points DOWN from objective toward specimen.  Each sensor
    sits ABOVE its objective along the tilted axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    O_left, O_right : tuple of float
        Objective centre positions in (x, z) millimetres.
    specimen : tuple of float
        Specimen position in (x, z) millimetres.
    sensor_offset : float
        Distance from objective to sensor along the optical axis, in mm.
    pixel_pitch : float
        Pixel pitch in millimetres for sensor-scale indicators.
    """

    OL = np.array([float(O_left[0]), float(O_left[1])])
    OR = np.array([float(O_right[0]), float(O_right[1])])
    sp = np.array([float(specimen[0]), float(specimen[1])])
    Z = OL[1] - sp[1]
    p_mm = float(pixel_pitch)

    # Tilt of each optical axis w.r.t. the downward vertical.
    #   O_L (negative x): axis tilts toward +x  → theta_L > 0
    #   O_R (positive x): axis tilts toward -x  → theta_R < 0
    theta_L = math.atan2(sp[0] - OL[0], OL[1] - sp[1])
    theta_R = math.atan2(sp[0] - OR[0], OR[1] - sp[1])

    x_span = abs(OL[0]) + 30
    z_top = OL[1] + sensor_offset + 10

    # ---- planes ----
    plane_style = {"color": "#cccccc", "linewidth": 0.6, "zorder": 0}
    ax.plot([-x_span, x_span], [sp[1], sp[1]], **plane_style)
    ax.plot([-x_span, x_span], [OL[1], OL[1]], **plane_style)
    ax.text(x_span + 2, sp[1], "object plane ($X$)", va="center", ha="left",
            fontsize=8, color=COLORS["annotation"])
    ax.text(x_span + 2, OL[1], "objectives ($O_L, O_R$)", va="center",
            ha="left", fontsize=8, color=COLORS["annotation"])

    # ---- vertical reference ----
    ax.plot([0, 0], [sp[1] - 10, z_top + 5], color=COLORS["axis"],
            linewidth=0.7, linestyle="--", dashes=(6, 4), zorder=0)

    for ch, O, theta in [("left", OL, theta_L), ("right", OR, theta_R)]:
        # u_down points toward the specimen, u_up toward the sensor
        u_down = np.array([math.sin(theta), -math.cos(theta)])
        u_up = -u_down
        u_perp = np.array([math.cos(theta), math.sin(theta)])

        # Sensor centre above the objective
        sensor_c = O + u_up * sensor_offset

        # Tilted sensor segment
        hw = 14
        s0 = sensor_c + u_perp * hw
        s1 = sensor_c - u_perp * hw
        ax.plot([s0[0], s1[0]], [s0[1], s1[1]], color=COLORS[ch],
                linewidth=3.0, solid_capstyle="round", zorder=10)

        # Dashed optical axis: from above sensor down past specimen
        top = O + u_up * (sensor_offset + 8)
        bot = O + u_down * (Z + 4)
        ax.plot([top[0], bot[0]], [top[1], bot[1]], color=COLORS[ch],
                linewidth=0.8, linestyle="--", dashes=(5, 5), zorder=1)

        # Objective dot + label
        name = "O_L" if ch == "left" else "O_R"
        ax.scatter(*O, s=80, color=COLORS[ch], edgecolors="white",
                   linewidth=1.0, zorder=15)
        ox = -16 if ch == "left" else 6
        annotate_math(ax, O, name, offset=(ox, -3), color=COLORS[ch])

        # Sensor dot
        ax.scatter(*sensor_c, s=20, color=COLORS[ch], zorder=15,
                   edgecolors="white", linewidth=0.5)

        # Chief ray: sensor → objective → specimen (two straight segments)
        ax.plot([sensor_c[0], O[0]], [sensor_c[1], O[1]],
                color=COLORS[ch], linewidth=2.2, solid_capstyle="round", zorder=5)
        ax.plot([O[0], sp[0]], [O[1], sp[1]],
                color=COLORS[ch], linewidth=2.2, solid_capstyle="round", zorder=5)

        # Theta arc between the downward vertical and the tilted axis
        t_deg = math.degrees(theta)
        arc_d = 16
        if ch == "left":
            t1, t2 = -90.0, -90.0 + t_deg
        else:
            t1, t2 = -90.0 + t_deg, -90.0
        ax.add_patch(_Arc((O[0], O[1]), arc_d, arc_d, theta1=t1, theta2=t2,
                          edgecolor=COLORS[ch], linewidth=1.0, zorder=12))
        # Label outside the arc
        mid = theta / 2.0
        r = arc_d / 2 + 10
        lbl = O + np.array([math.sin(mid), -math.cos(mid)]) * r
        ha = "right" if ch == "left" else "left"
        ax.text(lbl[0] + (-3 if ch == "left" else 3), lbl[1],
                f"$\\theta={abs(t_deg):.0f}^\\circ$",
                fontsize=9, color=COLORS[ch], ha=ha, va="center")

    # ---- specimen ----
    ax.scatter(*sp, s=60, color=COLORS["specimen"], edgecolors="black",
               linewidth=0.5, zorder=15)
    annotate_math(ax, sp, "X", offset=(4, -8))

    # ---- pixel pitch on left sensor ----
    u_down_L = np.array([math.sin(theta_L), -math.cos(theta_L)])
    u_up_L = -u_down_L
    u_perp_L = np.array([math.cos(theta_L), math.sin(theta_L)])
    sc_L = OL + u_up_L * sensor_offset
    u0 = sc_L
    u1 = sc_L + u_perp_L * p_mm
    for uu in [u0, u1]:
        ax.plot([uu[0], uu[0] + u_up_L[0] * 4],
                [uu[1], uu[1] + u_up_L[1] * 4],
                color=COLORS["left"], linewidth=0.7, zorder=10)
    ax.annotate("", xy=u1 + u_up_L * 6, xytext=u0 + u_up_L * 6,
                arrowprops={"arrowstyle": "<->", "color": "#666666", "lw": 0.8})
    mid_pp = (u0 + u1) / 2 + u_up_L * 10
    ax.text(mid_pp[0], mid_pp[1], "$p$", fontsize=9, ha="center", color="#666666")

    # ---- dimensions ----
    draw_dimension(ax, OL, OR, "B", offset=(0, -8))
    draw_dimension(ax, sp, (0, OL[1]), "Z", offset=(35, 0))

    ax.set_aspect("equal")
    ax.axis("off")
