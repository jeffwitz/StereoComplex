"""Reusable 2D drawing primitives for optical diagrams.

All coordinates are in the (x, z) optical-cut plane (x = horizontal /
baseline direction, z = optical axis / depth).  The y-axis is ignored
because all six model families are axially symmetric in the Y-Z plane.
"""

from __future__ import annotations

import numpy as np

from stereocomplex.viz.style import COLORS, FONT, LINEWIDTHS
from matplotlib.patches import Arc
from matplotlib.patches import Rectangle


def draw_ray(ax, origin_2d, direction_2d, length, *, channel="left",
             style="solid", width=None):
    """Draw a ray segment from *origin_2d* along *direction_2d*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    origin_2d : (2,) array_like  — (x, z) in millimetres.
    direction_2d : (2,) array_like  — unit direction or any non-zero vector.
    length : float  — length of the visible segment in millimetres.
    channel : str  — ``"left"`` or ``"right"``; selects colour.
    style : str  — ``"solid"``, ``"dashed"``, ``":"``, or any valid linestyle.
    width : float | None  — override linewidth; defaults to ``LINEWIDTHS["ray"]``.
    """
    origin = np.asarray(origin_2d, dtype=float).reshape(2)
    d = np.asarray(direction_2d, dtype=float).reshape(2)
    d = d / np.linalg.norm(d)
    end = origin + float(length) * d
    ax.plot(
        [origin[0], end[0]], [origin[1], end[1]],
        color=COLORS[channel],
        linewidth=width or LINEWIDTHS["ray"],
        linestyle=style,
        solid_capstyle="round",
    )


def draw_lens(ax, center_2d, radius, *, label=None, kind="biconvex"):
    """Draw a schematic lens symbol.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    center_2d : (2,) array_like  — (x, z) centre.
    radius : float  — half-width of the lens symbol in mm.
    label : str | None  — LaTeX annotation placed above the lens.
    kind : str  — ``"biconvex"`` (future: ``"plano_convex"``, ``"meniscus"``).
    """
    cx, cz = float(center_2d[0]), float(center_2d[1])
    r = float(radius)
    # Draw a biconvex shape: two circular arcs
    # Use two Arc patches meeting at the edges
    arc_left = Arc((cx, cz), 2 * r, 2 * r, angle=0, theta1=290, theta2=70,
                   color=COLORS["lens"], linewidth=LINEWIDTHS["lens"])
    arc_right = Arc((cx, cz), 2 * r, 2 * r, angle=0, theta1=110, theta2=250,
                    color=COLORS["lens"], linewidth=LINEWIDTHS["lens"])
    ax.add_patch(arc_left)
    ax.add_patch(arc_right)
    # Vertical ticks at top and bottom edges for the "lens" look
    tip = r * 0.25
    for sign in (-1, 1):
        ax.plot([cx - tip, cx + tip], [cz + sign * r, cz + sign * r],
                color=COLORS["lens"], linewidth=LINEWIDTHS["lens"])
    if label:
        annotate_math(ax, (cx, cz + r + 3), label, offset=(0, 2))


def draw_sensor(ax, center_2d, width, height, *, channel="left", label=None):
    """Draw a camera sensor as a filled rectangle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    center_2d : (2,) array_like
    width, height : float  — sensor dimensions in mm.
    channel : str  — ``"left"`` or ``"right"``.
    label : str | None
    """
    cx, cz = float(center_2d[0]), float(center_2d[1])
    rect = Rectangle(
        (cx - float(width) / 2, cz - float(height) / 2),
        float(width), float(height),
        facecolor=COLORS["sensor"], edgecolor=COLORS[channel],
        linewidth=LINEWIDTHS["sensor"], zorder=5,
    )
    ax.add_patch(rect)
    if label:
        annotate_math(ax, (cx, cz + height / 2 + 2), label,
                      offset=(0, 2), color=COLORS[channel])


def draw_specimen(ax, position_2d, *, radius=2.0, label="X"):
    """Draw a specimen point as a filled circle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    position_2d : (2,) array_like
    radius : float  — marker radius.
    label : str  — annotation text (LaTeX).
    """
    px, pz = float(position_2d[0]), float(position_2d[1])
    ax.scatter(px, pz, s=float(radius) * 20, c=COLORS["specimen"],
               edgecolors="black", linewidth=0.5, zorder=10)
    annotate_math(ax, (px, pz), label, offset=(4, -4))


def draw_optical_axis(ax, start, end):
    """Draw the optical axis as a dashed grey line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    start, end : (2,) array_like
    """
    ax.plot(
        [float(start[0]), float(end[0])],
        [float(start[1]), float(end[1])],
        color=COLORS["axis"], linewidth=LINEWIDTHS["axis"],
        linestyle="--", dashes=(5, 5),
    )


def annotate_math(ax, position, latex, *, offset=(0, 0), color=None,
                  fontsize=None):
    """Place a LaTeX annotation near *position*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    position : (2,) array_like  — anchor point.
    latex : str  — LaTeX string (without surrounding ``$``).
    offset : (float, float)  — text offset from anchor in data coordinates.
    color : str | None  — annotation colour.
    fontsize : float | None  — override default.
    """
    x, y = float(position[0]), float(position[1])
    fs = fontsize or FONT["math"]["size"]
    ax.annotate(
        f"${latex}$",
        xy=(x, y), xytext=(x + offset[0], y + offset[1]),
        fontfamily=FONT["math"]["family"],
        fontstyle=FONT["math"]["style"],
        fontsize=fs, color=color or COLORS["annotation"],
    )


def draw_dimension(ax, point_a, point_b, label, *, offset=(0, 0)):
    """Draw a double-arrow dimension line between two points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    point_a, point_b : (2,) array_like
    label : str  — LaTeX label placed at the midpoint.
    offset : (float, float)  — vertical offset from the line.
    """
    a = np.asarray(point_a, dtype=float).reshape(2)
    b = np.asarray(point_b, dtype=float).reshape(2)
    ax.annotate(
        "", xy=b, xytext=a,
        arrowprops={"arrowstyle": "<->", "color": COLORS["annotation"],
                    "linewidth": LINEWIDTHS["decoration"]},
    )
    mid = (a + b) / 2
    annotate_math(ax, mid, label, offset=offset)
