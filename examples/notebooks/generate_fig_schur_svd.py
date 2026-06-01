#!/usr/bin/env python3
"""Schur complement eigenvalue spectrum (Figure 8 of the CMO paper).

Plots the normalised eigenvalue spectrum of the Schur complement
:math:`S_\\theta` of the Fisher information matrix on the **26-parameter CMO
optical block**, evaluated at the rayfield-initialised Pycaso solution. After
pose marginalisation only a handful of optical directions remain observable:
the trailing eigenvalues (:math:`\\lambda_i/\\lambda_{\\max} < 10^{-3}`, red
crosses) collapse and are precisely the directions penalised by the Schur prior
(§3.7). The pose/optics coupling norm is annotated (c = 0.98 on the real Pycaso
configuration).

Data are read from the tracked manifest in
``docs/assets/cmo_paper/figure8_schur_singular_values/``; the eigenvalues were
extracted from the Schur bundle-adjustment diagnostic (see ``source`` in
``schur_spectrum.json``). Emits both PDF (paper) and PNG (docs) in one run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MANIFEST = Path("docs/assets/cmo_paper/figure8_schur_singular_values/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "schur_singular_values"
DISPLAY_FLOOR = 1e-9  # clip normalised eigenvalues for the log axis

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    """Render the Schur eigenvalue spectrum to PDF + PNG."""
    spec_path = (manifest_root / manifest["spectrum_json"]).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    eig = np.asarray(spec["schur_eigvals_descending"], dtype=np.float64)
    thr = float(spec["weak_threshold"])
    coupling = float(spec["coupling_norm"])
    rank_eff = int(spec["rank_effective"])
    n = eig.size

    lam_max = eig[0]
    norm = np.clip(eig / lam_max, DISPLAY_FLOOR, None)
    idx = np.arange(1, n + 1)
    strong = norm >= thr  # eigenvalues above the weak threshold

    fig, ax = plt.subplots(figsize=tuple(manifest["figure_size"]))
    ax.axhline(thr, color="gray", ls="--", lw=1.2,
               label=r"weak threshold $\lambda_i/\lambda_{\max}=10^{-3}$")
    ax.semilogy(idx[strong], norm[strong], "o", color="#2a55c7", ms=9,
                label=f"observable optical modes (rank $={rank_eff}$)")
    ax.semilogy(idx[~strong], norm[~strong], "x", color="#c4332a", ms=9, mew=2,
                label=f"weak (pose-absorbed) modes ($n={int(np.sum(~strong))}$)")

    ax.set_xlabel("Eigenvalue index $i$")
    ax.set_ylabel(r"$\lambda_i / \lambda_{\max}$")
    ax.set_title("Schur complement eigenvalue spectrum (26-parameter optical block)")
    ax.set_xlim(0.5, n + 0.5)
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(fontsize=9, loc="upper right")
    ax.text(0.02, 0.04,
            f"pose/optics coupling norm $c = {coupling:.2f}$",
            transform=ax.transAxes, fontsize=10,
            bbox={"boxstyle": "round", "fc": "white", "ec": "gray", "alpha": 0.8})

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = out_dir / f"{OUT_BASENAME}.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> int:
    """Entry point: load the manifest and render Figure 8."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    render(manifest, MANIFEST.parent, OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
