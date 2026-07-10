#!/usr/bin/env python3
"""CMO stereo microscope optical layout (Figure 1 of the CMO paper).

Pedagogical 2-D optical-cut diagram showing the shared common main
objective, two off-axis sub-pupils, tube lenses, sensors, and the
chief-ray paths converging at the object plane.

Geometry parameters live in
``docs/assets/cmo_paper/figure1_cmo_physical/manifest.json``. Emits both
PDF (paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from stereocomplex.viz.figures import diagram_cmo_physical

MANIFEST = Path("docs/assets/cmo_paper/figure1_cmo_physical/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "cmo_physical"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def render(manifest: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=tuple(manifest["figure_size"]))
    diagram_cmo_physical(
        ax=ax,
        f_obj=float(manifest["effective_axial_length_mm"]),
        effective_axial_length=float(manifest["effective_axial_length_mm"]),
        working_distance=float(manifest["working_distance_mm"]),
        b=float(manifest["b_mm"]),
        exaggerated=bool(manifest["exaggerated"]),
        show_gamma=bool(manifest.get("show_gamma", False)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    render(manifest, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
