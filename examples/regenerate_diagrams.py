"""Regenerate all optical diagrams from the physical models.

Run from the repository root::

    python examples/regenerate_diagrams.py

Outputs land in ``docs/assets/diagrams/`` (SVG for web, PDF for LaTeX).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from stereocomplex.viz.figures import (
    diagram_cmo_physical,
    diagram_greenough,
    diagram_pinhole_stereo,
)

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "serif", "font.size": 11})

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "docs" / "assets" / "diagrams"
ASSETS.mkdir(parents=True, exist_ok=True)


def save(name, builder, **kwargs):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    builder(ax=ax, **kwargs)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(ASSETS / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  {name}")


print("Regenerating optical diagrams …")
save("pinhole_stereo", diagram_pinhole_stereo,
     O_left=(-6, 80), O_right=(6, 80), specimen=(0, 0))
save("cmo_physical", diagram_cmo_physical,
     f_obj=80, working_distance=120, b=8, exaggerated=True)
save("greenough", diagram_greenough,
     O_left=(-5, 60), O_right=(5, 60), specimen=(0, 0))
print(f"Done — assets in {ASSETS}")
