#!/usr/bin/env python3
"""Schur complement singular value spectra (Figure 8 of the CMO paper).

Compares the singular values of the optics information matrix's
design-matrix proxy at Zernike orders 2 and 4, evaluated on a regular
41x41 pixel grid (Phase 1 of the conditioning diagnostic). The trailing
singular values collapse for the higher-order family — this is the
observability signature reported in §3.6 of the paper.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure8_schur_singular_values/``. Emits both PDF
(paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MANIFEST = Path("docs/assets/cmo_paper/figure8_schur_singular_values/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "schur_singular_values"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    diag_path = (manifest_root / manifest["diagnostic_json"]).resolve()
    diag = json.loads(diag_path.read_text(encoding="utf-8"))

    phase1 = diag["phase1_design_matrix"]
    grid_data = phase1.get("regular_grid_41x41", phase1.get("regular_grid_41×41", {}))
    sparse_data = phase1.get("sparse_central_85pct", {})

    sv2 = np.asarray(grid_data.get("singular_values_order2", []), dtype=np.float64)
    sv4 = np.asarray(grid_data.get("singular_values_order4", []), dtype=np.float64)

    fig, ax = plt.subplots(figsize=tuple(manifest["figure_size"]))

    if sv2.size:
        cond2 = grid_data.get("condition_number_order2", float("nan"))
        ax.semilogy(
            np.arange(1, sv2.size + 1), sv2, "o-", color="#2a55c7",
            lw=2, ms=8,
            label=f"Design matrix $\\Phi_2$ ({sv2.size} modes, cond={cond2:.1f})",
        )

    if sv4.size:
        cond4 = grid_data.get("condition_number_order4", float("nan"))
        ax.semilogy(
            np.arange(1, sv4.size + 1), sv4, "s--", color="#c4332a",
            lw=1.5, ms=6,
            label=f"Design matrix $\\Phi_4$ ({sv4.size} modes, cond={cond4:.1f})",
        )

    if "condition_number_order2" in sparse_data:
        sparse_cond = float(sparse_data["condition_number_order2"])
        ax.axhline(
            sparse_cond, color="gray", ls=":", alpha=0.5,
            label=f"Sparse grid cond={sparse_cond:.1f}",
        )

    if "mode_svd" in grid_data and sv2.size:
        names = [m.get("mode", "") for m in grid_data["mode_svd"]]
        for i, (sv, name) in enumerate(zip(sv2, names)):
            if i < int(manifest["n_annotated_modes"]):
                ax.annotate(
                    name, (i + 1, sv),
                    textcoords="offset points", xytext=(5, -10),
                    fontsize=8, alpha=0.7,
                )

    ax.set_xlabel("Mode index")
    ax.set_ylabel("Singular value (log scale)")
    ax.set_title("Design matrix singular value spectra")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    render(manifest, MANIFEST.parent, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
