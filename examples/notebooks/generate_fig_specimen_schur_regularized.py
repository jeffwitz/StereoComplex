#!/usr/bin/env python3
"""Five-variant specimen reconstruction (Figure 11 of the CMO paper).

Loads the five pre-computed point clouds declared in
``docs/assets/cmo_paper/figure11_specimen_schur_regularized/manifest.json``
and renders the 5-row comparison (Z-relief map, XY footprint with
magnification ratio, ray-gap log-scale histogram) via the shared
``stereocomplex.optical_ba.plot_specimen_grid`` helper.

Emits both PDF (paper) and PNG (docs) in one run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from stereocomplex.optical_ba import (  # noqa: E402  (sys.path tweak above)
    load_specimen_npz,
    plot_specimen_grid,
)

MANIFEST = Path("docs/assets/cmo_paper/figure11_specimen_schur_regularized/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "specimen_schur_regularized"


def main() -> int:
    manifest_data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_root = MANIFEST.parent

    correspondences_path = (manifest_root / manifest_data["correspondences"]).resolve()
    recs = [
        load_specimen_npz(
            (manifest_root / entry["npz"]).resolve(),
            variant=entry["label"],
        )
        for entry in manifest_data["variants"]
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # PNG only: the 5-row figure embeds ~10 million scatter points; a
    # vector PDF blows up to ~60 MB while the paper's \includegraphics
    # already references the PNG (manuscript.tex line 687).
    out_path = OUT_DIR / f"{OUT_BASENAME}.png"
    plot_specimen_grid(
        recs,
        correspondences_path,
        out_path,
        title=manifest_data.get("title"),
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
