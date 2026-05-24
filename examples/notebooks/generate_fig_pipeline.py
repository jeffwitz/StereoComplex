#!/usr/bin/env python3
"""Pipeline diagram (Figure 2 of the CMO paper).

Two-stage decomposition flowchart. All editable text/numbers live in
``docs/assets/cmo_paper/figure2_pipeline/pipeline.json`` — this script only
defines the layout. Both PDF (paper) and PNG (docs) are produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ASSET = Path("docs/assets/cmo_paper/figure2_pipeline/pipeline.json")
OUT = Path("paper/cmo/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 10,
})


def _step_box(ax, x, y, w, h, step, edge):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor="white", edgecolor=edge, linewidth=1.4,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.22, step["title"],
            ha="center", va="top", fontsize=10, fontweight="bold")
    detail = "\n".join(step["lines"])
    ax.text(x + w / 2, y + h / 2 - 0.18, detail,
            ha="center", va="center", fontsize=8.5, color="#333333")


def _arrow(ax, x_from, x_to, y, label):
    ax.annotate("", xy=(x_to, y), xytext=(x_from, y),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))
    ax.text((x_from + x_to) / 2, y + 0.28, label,
            ha="center", va="bottom", fontsize=8, color="#444444")


def render(data: dict, out_pdf: Path, out_png: Path) -> None:
    fig_w, fig_h = 16.0, 4.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(fig_w / 2, fig_h - 0.25, data["title"],
            ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(fig_w / 2, fig_h - 0.7, data["subtitle"],
            ha="center", va="top", fontsize=9.5, color="#555555", style="italic")

    io_w = 1.6
    step_w, step_h = 2.7, 1.6
    step_gap = 0.55
    stage_pad_x, stage_pad_y = 0.30, 0.45
    y_center = 1.65
    y_step = y_center - step_h / 2

    stage_w = 2 * step_w + step_gap + 2 * stage_pad_x
    gap_stages = 0.85
    total_w = io_w + 0.3 + stage_w + gap_stages + stage_w + 0.3 + io_w
    x0 = (fig_w - total_w) / 2.0

    # INPUT block
    ax.text(x0 + io_w / 2, y_center + 0.55, data["input"]["label"],
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#555555")
    ax.text(x0 + io_w / 2, y_center + 0.35, "\n".join(data["input"]["lines"]),
            ha="center", va="top", fontsize=8.5, color="#555555")

    x_cursor = x0 + io_w + 0.3
    step_x_coords: list[float] = []

    for stage in data["stages"]:
        rect = mpatches.FancyBboxPatch(
            (x_cursor, y_step - stage_pad_y), stage_w, step_h + 2 * stage_pad_y,
            boxstyle="round,pad=0.05", facecolor=stage["fill"],
            edgecolor=stage["edge"], linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x_cursor + stage_w / 2, y_step + step_h + stage_pad_y - 0.12,
                stage["name"], ha="center", va="top",
                fontsize=10.5, fontweight="bold", color=stage["edge"])

        x_s1 = x_cursor + stage_pad_x
        x_s2 = x_s1 + step_w + step_gap
        _step_box(ax, x_s1, y_step, step_w, step_h, stage["steps"][0], stage["edge"])
        _step_box(ax, x_s2, y_step, step_w, step_h, stage["steps"][1], stage["edge"])
        step_x_coords.extend([x_s1, x_s1 + step_w, x_s2, x_s2 + step_w])

        x_cursor += stage_w + gap_stages

    # OUTPUT block
    x_out = x_cursor - gap_stages + 0.3
    ax.text(x_out + io_w / 2, y_center + 0.55, data["output"]["label"],
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#2a7a3b")
    ax.text(x_out + io_w / 2, y_center + 0.35, "\n".join(data["output"]["lines"]),
            ha="center", va="top", fontsize=8.5, color="#2a7a3b")

    arrow_pairs = [
        (x0 + io_w, step_x_coords[0]),        # INPUT  → step 1.1
        (step_x_coords[1], step_x_coords[2]), # step 1.1 → step 1.2
        (step_x_coords[3], step_x_coords[4]), # step 1.2 → step 2.1
        (step_x_coords[5], step_x_coords[6]), # step 2.1 → step 2.2
        (step_x_coords[7], x_out),            # step 2.2 → OUTPUT
    ]
    labels = ["pixels"] + data["arrow_labels"] + ["calibrated\nrays"]
    for (x_from, x_to), label in zip(arrow_pairs, labels):
        _arrow(ax, x_from + 0.05, x_to - 0.05, y_center, label)

    fig.tight_layout(pad=0.3)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    data = json.loads(ASSET.read_text(encoding="utf-8"))
    render(data, OUT / "pipeline.pdf", OUT / "pipeline.png")
    print(f"wrote {OUT / 'pipeline.pdf'} and {OUT / 'pipeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
