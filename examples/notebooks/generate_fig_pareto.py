#!/usr/bin/env python3
"""Pareto frontier of the gauge-regularized Zernike sweep (Figure 7).

Three-panel figure:

- (a) Pareto frontier of pixel RMS vs Z0 direction drift with
  Pareto-optimal points highlighted.
- (b) Baseline stability across regularisation strengths.
- (c) Convergence-angle stability across regularisation strengths.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure7_pareto_gauge_regularization/``. Emits both
PDF (paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MANIFEST = Path("docs/assets/cmo_paper/figure7_pareto_gauge_regularization/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "pareto_gauge_regularization"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
})


def _pareto_optimal(runs: list[dict]) -> list[dict]:
    """Return the Pareto-optimal subset on (ray_rms_mm, drift_z0_deg)."""
    out = []
    for i, r in enumerate(runs):
        dominated = False
        for j, s in enumerate(runs):
            if i == j:
                continue
            if (s["ray_rms_mm"] <= r["ray_rms_mm"]
                    and s["drift_z0_deg"] <= r["drift_z0_deg"]
                    and (s["ray_rms_mm"] < r["ray_rms_mm"]
                         or s["drift_z0_deg"] < r["drift_z0_deg"])):
                dominated = True
                break
        if not dominated:
            out.append(r)
    return out


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    sweep_path = (manifest_root / manifest["sweep_json"]).resolve()
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))

    runs = sweep.get("sweep", [])
    constrained_rms = float(sweep.get("constrained_rms_mm",
                                      manifest["constrained_rms_mm_fallback"]))
    pareto = _pareto_optimal(runs)

    fig, axes = plt.subplots(1, 3, figsize=tuple(manifest["figure_size"]))

    ax = axes[0]
    ax.scatter([r["ray_rms_mm"] for r in runs],
               [r["drift_z0_deg"] for r in runs],
               c="steelblue", s=60, zorder=3, label="all runs")
    ax.scatter([r["ray_rms_mm"] for r in pareto],
               [r["drift_z0_deg"] for r in pareto],
               c="darkorange", s=120, zorder=4,
               edgecolors="black", linewidth=0.5, label="Pareto-optimal")
    po_sorted = sorted(pareto, key=lambda r: r["ray_rms_mm"])
    if len(po_sorted) >= 2:
        ax.plot([r["ray_rms_mm"] for r in po_sorted],
                [r["drift_z0_deg"] for r in po_sorted],
                "darkorange", lw=1.5, alpha=0.6, zorder=2)
    ax.axvline(constrained_rms, color="gray", ls="--", alpha=0.5,
               label="constrained ref")
    ax.set_xlabel("Ray RMS (mm)")
    ax.set_ylabel("Z$_0$ drift (deg)")
    ax.set_title("Pareto frontier: RMS vs gauge drift")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    axes[1].plot([r["drift_z0_deg"] for r in runs],
                 [r.get("baseline_mm", 0) for r in runs], "o", ms=8, alpha=0.7)
    axes[1].set_xlabel("Z$_0$ drift (deg)")
    axes[1].set_ylabel("Baseline (mm)")
    axes[1].set_title("Baseline stability")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot([r["drift_z0_deg"] for r in runs],
                 [r.get("convergence_angle_deg", 0) for r in runs],
                 "o", ms=8, alpha=0.7)
    axes[2].set_xlabel("Z$_0$ drift (deg)")
    axes[2].set_ylabel("Convergence angle (deg)")
    axes[2].set_title("Convergence angle stability")
    axes[2].grid(True, alpha=0.3)

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
