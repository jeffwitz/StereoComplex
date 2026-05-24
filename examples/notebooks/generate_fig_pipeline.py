#!/usr/bin/env python3
"""Pipeline diagram (Figure 2 of the CMO paper).

Vertical two-stage decomposition flowchart: INPUT at the top flows down
through Stage 1 (Rayfield Measurement, 2 steps) and Stage 2 (Physical
Model Identification, 2 steps) to OUTPUT at the bottom. Stacking the
steps vertically gives each box room enough that the per-step text
stays legible in print.

All editable text/numbers live in
``docs/assets/cmo_paper/figure2_pipeline/pipeline.json`` — this script
only defines the layout. Both PDF (paper) and PNG (docs) are produced.
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
    "font.size": 12,
})


def _step_box(ax, x, y, w, h, step, edge):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.04",
        facecolor="white", edgecolor=edge, linewidth=1.6,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.12, step["title"],
            ha="center", va="top", fontsize=13, fontweight="bold")
    detail = "\n".join(step["lines"])
    ax.text(x + w / 2, y + h - 0.38, detail,
            ha="center", va="top", fontsize=11, color="#333333")


def _down_arrow(ax, x, y_from, y_to, label):
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.8))
    ax.text(x + 0.18, (y_from + y_to) / 2, label,
            ha="left", va="center", fontsize=10,
            color="#444444", style="italic")


def render(data: dict, out_pdf: Path, out_png: Path) -> None:
    fig_w, fig_h = 8.5, 10.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    ax.text(fig_w / 2, fig_h - 0.18, data["title"],
            ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(fig_w / 2, fig_h - 0.55, data["subtitle"],
            ha="center", va="top", fontsize=10,
            color="#555555", style="italic")

    step_w = 5.2
    step_h = 0.95
    arrow_gap = 0.50
    intra_arrow = 0.45
    stage_pad = 0.28
    title_band = 0.32
    x_center = fig_w / 2.0
    x_step = x_center - step_w / 2.0

    io_w = 4.4
    io_h = 0.78
    x_io = x_center - io_w / 2.0

    y_cursor = fig_h - 1.05

    # INPUT box
    rect_in = mpatches.FancyBboxPatch(
        (x_io, y_cursor - io_h), io_w, io_h,
        boxstyle="round,pad=0.05",
        facecolor="#f4f4f4", edgecolor="#777777", linewidth=1.4,
    )
    ax.add_patch(rect_in)
    ax.text(x_center, y_cursor - 0.12, data["input"]["label"],
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#555555")
    ax.text(x_center, y_cursor - 0.30,
            "\n".join(data["input"]["lines"]),
            ha="center", va="top", fontsize=10, color="#555555")
    y_cursor -= io_h

    labels = ["refined pixel\ncorrespondences"] + list(data["arrow_labels"])
    label_idx = 0

    for stage in data["stages"]:
        _down_arrow(ax, x_center, y_cursor - 0.05,
                    y_cursor - arrow_gap + 0.05, labels[label_idx])
        label_idx += 1
        y_cursor -= arrow_gap

        stage_h = title_band + step_h + intra_arrow + step_h + 2 * stage_pad
        rect = mpatches.FancyBboxPatch(
            (x_step - stage_pad, y_cursor - stage_h),
            step_w + 2 * stage_pad, stage_h,
            boxstyle="round,pad=0.05",
            facecolor=stage["fill"], edgecolor=stage["edge"], linewidth=1.6,
        )
        ax.add_patch(rect)
        ax.text(x_center, y_cursor - 0.16, stage["name"],
                ha="center", va="top",
                fontsize=12, fontweight="bold", color=stage["edge"])

        y_step1_top = y_cursor - stage_pad - title_band
        _step_box(ax, x_step, y_step1_top - step_h, step_w, step_h,
                  stage["steps"][0], stage["edge"])

        y_arrow_top = y_step1_top - step_h - 0.05
        y_arrow_bot = y_arrow_top - intra_arrow + 0.10
        _down_arrow(ax, x_center, y_arrow_top, y_arrow_bot, labels[label_idx])
        label_idx += 1

        y_step2_top = y_arrow_bot - 0.05
        _step_box(ax, x_step, y_step2_top - step_h, step_w, step_h,
                  stage["steps"][1], stage["edge"])

        y_cursor = y_step2_top - step_h - stage_pad

    _down_arrow(ax, x_center, y_cursor - 0.05,
                y_cursor - arrow_gap + 0.05, "calibrated rays")

    # OUTPUT box
    rect_out = mpatches.FancyBboxPatch(
        (x_io, y_cursor - arrow_gap - io_h), io_w, io_h,
        boxstyle="round,pad=0.05",
        facecolor="#eaf6ec", edgecolor="#2a7a3b", linewidth=1.6,
    )
    ax.add_patch(rect_out)
    ax.text(x_center, y_cursor - arrow_gap - 0.12, data["output"]["label"],
            ha="center", va="top", fontsize=11, fontweight="bold",
            color="#2a7a3b")
    ax.text(x_center, y_cursor - arrow_gap - 0.30,
            "\n".join(data["output"]["lines"]),
            ha="center", va="top", fontsize=10, color="#2a7a3b")

    fig.tight_layout(pad=0.3)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, format="png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def main() -> int:
    data = json.loads(ASSET.read_text(encoding="utf-8"))
    render(data, OUT / "pipeline.pdf", OUT / "pipeline.png")
    print(f"wrote {OUT / 'pipeline.pdf'} and {OUT / 'pipeline.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
