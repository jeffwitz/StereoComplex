#!/usr/bin/env python3
"""Generate the Zernike translation-stage prior experiment figure.

The editable numerical source is
``docs/assets/cmo_paper/figure10b_zernike_stage_prior/stage_prior_results.json``.
It is produced by ``evaluate_zernike_stage_prior.py`` from the cached Pycaso
corners, the raw coin images, and the versioned profilometry registration.

This script only performs layout. It writes the vector PDF used by the paper
and a PNG preview in the same run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSET = REPO / "docs/assets/cmo_paper/figure10b_zernike_stage_prior"
DATA = ASSET / "stage_prior_results.json"
OUT = REPO / "paper/cmo/figures"
OUTPUT_NAME = "zernike_stage_prior"

CASE_ORDER = [
    "weak_proposal",
    "tight_scale_moderate_jitter",
    "near_hard_ladder",
]
CASE_LABELS = {
    "weak_proposal": "Weak hierarchy",
    "tight_scale_moderate_jitter": "Strong hierarchy",
    "near_hard_ladder": "Near-fixed ladder",
}
CASE_COLORS = {
    "weak_proposal": "#e69f00",
    "tight_scale_moderate_jitter": "#0072b2",
    "near_hard_ladder": "#009e73",
}


def _centred(values: np.ndarray) -> np.ndarray:
    """Return values after removing their common translation gauge."""
    return values - float(np.mean(values))


def render() -> None:
    """Render the two-panel stage-ladder and specimen-relief comparison."""
    data = json.loads(DATA.read_text())
    nominal = np.asarray(data["nominal_stage_positions_mm"], dtype=np.float64)
    frame_index = np.arange(len(nominal))

    fig, (ax_ladder, ax_relief) = plt.subplots(
        1, 2, figsize=(12.6, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    ax_ladder.plot(
        frame_index,
        _centred(nominal),
        "k--o",
        linewidth=2.0,
        markersize=5,
        label="Nominal stage ladder",
        zorder=5,
    )
    published = np.asarray(
        data["published_zernike_z_positions_mm"], dtype=np.float64
    )
    ax_ladder.plot(
        frame_index,
        _centred(published),
        color="#7f7f7f",
        marker="s",
        linewidth=1.8,
        markersize=4,
        label="Free axial poses",
    )
    for key in CASE_ORDER:
        values = np.asarray(
            data["cases"][key]["fitted_z_positions_mm"], dtype=np.float64
        )
        ax_ladder.plot(
            frame_index,
            _centred(values),
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=CASE_COLORS[key],
            label=CASE_LABELS[key],
        )
    ax_ladder.set_xlabel("Calibration frame")
    ax_ladder.set_ylabel("Centred board position along camera Z (mm)")
    ax_ladder.set_title("(a) Metric information transmitted by the pose model")
    ax_ladder.set_xticks(frame_index)
    ax_ladder.grid(alpha=0.25)
    ax_ladder.legend(fontsize=8.5, loc="upper left")

    labels = ["Free\naxial poses"]
    ratios = [float(data["published_zernike_ratio"])]
    colors = ["#7f7f7f"]
    annotations = ["no stage prior"]
    for key in CASE_ORDER:
        case = data["cases"][key]
        labels.append(CASE_LABELS[key].replace(" ", "\n", 1))
        ratios.append(float(case["relief_ratio_to_profilometry"]))
        colors.append(CASE_COLORS[key])
        annotations.append(
            rf"$\sigma_s={100 * case['scale_sigma']:g}\%$"
            + "\n"
            + rf"$\sigma_\epsilon={1000 * case['jitter_sigma_mm']:.0f}\,\mu$m"
        )

    bars = ax_relief.bar(
        np.arange(len(ratios)), ratios, color=colors, edgecolor="black", linewidth=0.7
    )
    profilo_std = float(data["profilometry_std_um"])
    soloff_ratio = float(data["soloff_std_um"]) / profilo_std
    cmo_ratio = float(data["cmo26_std_um"]) / profilo_std
    ax_relief.axhline(1.0, color="black", linewidth=1.4, label="Profilometry")
    ax_relief.axhline(
        soloff_ratio, color="#56b4e9", linestyle="--", linewidth=1.5, label="Soloff"
    )
    ax_relief.axhline(
        cmo_ratio, color="#009e73", linestyle=":", linewidth=1.8, label="CMO 26p"
    )
    for bar, ratio, annotation in zip(bars, ratios, annotations, strict=True):
        ax_relief.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + 0.045,
            f"{ratio:.2f}x",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        ax_relief.text(
            bar.get_x() + bar.get_width() / 2,
            0.08,
            annotation,
            ha="center",
            va="bottom",
            fontsize=7.7,
            color="white" if ratio > 1.2 else "black",
        )
    ax_relief.set_xticks(np.arange(len(labels)), labels)
    ax_relief.set_ylabel("Relief standard deviation / profilometry")
    ax_relief.set_ylim(0.0, max(ratios) * 1.18)
    ax_relief.set_title("(b) Consequence for independent specimen relief")
    ax_relief.grid(axis="y", alpha=0.25)
    ax_relief.legend(fontsize=8.5, loc="upper right")

    fig.suptitle(
        "Axial stage metrology is an observability prior for the Zernike rayfield",
        fontsize=13,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    pdf = OUT / f"{OUTPUT_NAME}.pdf"
    png = OUT / f"{OUTPUT_NAME}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf.relative_to(REPO)} and {png.relative_to(REPO)}")


if __name__ == "__main__":
    render()
