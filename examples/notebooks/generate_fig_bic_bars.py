#!/usr/bin/env python3
"""BIC bar chart (Figure 12 of the CMO paper).

Two panels:

- (a) Ray-space BIC: candidate physical models ordered by BIC score.
- (b) Operational reprojection guard applied after family selection.

All inputs and display names are read from the manifest in
``docs/assets/cmo_paper/figure12_bic_bars/``. Emits both PDF (paper) and
PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MANIFEST = Path("docs/assets/cmo_paper/figure12_bic_bars/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "bic_bars"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
})


def _display(name: str, mapping: dict[str, str]) -> str:
    return mapping.get(name, name)


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    bic_path = (manifest_root / manifest["bic_json"]).resolve()
    bic = json.loads(bic_path.read_text(encoding="utf-8"))

    candidates = bic["candidates"]
    name_map = manifest["model_display_names"]
    guard_px = float(manifest["reprojection_guard_px"])

    sorted_cands = sorted(candidates, key=lambda c: c["bic_ray"])
    labels_ray = [_display(c["model"], name_map) for c in sorted_cands]
    bic_vals = [c["bic_ray"] for c in sorted_cands]
    n_params = [c["parameters"] for c in sorted_cands]
    colors_ray = ["#2a7a3b" if "CMO" in lbl else "#7a9ab5" for lbl in labels_ray]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=tuple(manifest["figure_size"]))

    bars1 = ax1.barh(labels_ray, bic_vals, color=colors_ray,
                     edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("BIC (ray-space)")
    ax1.set_title("(a) Ray-space BIC")
    for bar, p in zip(bars1, n_params, strict=True):
        ax1.text(bar.get_width() - 500, bar.get_y() + bar.get_height() / 2,
                 f"{p}p", va="center", ha="right",
                 fontsize=9, color="white", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    op_models = manifest["operational_models"]
    op_labels = [entry["label"] for entry in op_models]
    op_scores = [float(entry["px_rms"]) for entry in op_models]
    op_colors = [entry["color"] for entry in op_models]

    bars2 = ax2.barh(op_labels, op_scores, color=op_colors,
                     edgecolor="black", linewidth=0.5)
    for bar, entry in zip(bars2, op_models, strict=True):
        if entry["status"] == "PASS":
            continue
        ax2.text(
            bar.get_width() / 2,
            bar.get_y() + bar.get_height() / 2,
            entry["status"],
            va="center",
            ha="center",
            fontsize=9,
            color="darkred",
            fontweight="bold",
        )
    ax2.axvline(guard_px, color="darkred", linestyle="--", linewidth=1.2,
                label=f"{guard_px:.1f} px guard")
    ax2.legend(loc="lower right", frameon=False)
    ax2.set_xlabel("Pixel reprojection RMS (px)")
    ax2.set_title("(b) Operational guard after family selection")
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Model selection: BIC comparison",
                 fontweight="bold", fontsize=13)
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
