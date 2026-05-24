#!/usr/bin/env python3
"""BIC vs Zernike order sweep (Figure 3 of the CMO paper).

Dual-axis plot: BIC (left) and pixel RMS (right) versus total Zernike
parameters. Annotates the BIC minimum and the selected ``O(0)+d(2)``
model used downstream for physical model construction.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure3_BIC_vs_order/``. Emits both PDF (paper)
and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MANIFEST = Path("docs/assets/cmo_paper/figure3_BIC_vs_order/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "BIC_vs_order"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    sweep_path = (manifest_root / manifest["sweep_json"]).resolve()
    entries = json.loads(sweep_path.read_text(encoding="utf-8"))

    n_obs = (manifest["n_frames"] * manifest["n_corners"]
             * manifest["n_channels"] * manifest["n_coords_per_point"])

    n_params = np.array([e["p"] for e in entries], dtype=np.int64)
    rms_vals = np.array([e["rms"] for e in entries], dtype=np.float64)
    labels = [f"O({e['O']})+d({e['d']})" for e in entries]
    bic_vals = n_params * np.log(n_obs) + n_obs * np.log(rms_vals ** 2)

    fig, ax1 = plt.subplots(figsize=tuple(manifest["figure_size"]))
    color1 = "#2a55c7"
    color2 = "#c4332a"

    ax1.plot(n_params, bic_vals, "o-", color=color1, lw=2, ms=8, label="BIC")
    ax1.set_xlabel("Total parameters")
    ax1.set_ylabel("BIC", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    bic_min_idx = int(np.argmin(bic_vals))
    ax1.annotate(
        f"BIC min: {labels[bic_min_idx]}\n"
        f"({n_params[bic_min_idx]}p, {rms_vals[bic_min_idx]:.2f} px)",
        (n_params[bic_min_idx], bic_vals[bic_min_idx]),
        xytext=(n_params[bic_min_idx] + 10, bic_vals[bic_min_idx] - 50),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, color="black",
    )

    selected_label = manifest["selected_label"]
    idx_sel = next(i for i, lbl in enumerate(labels) if lbl == selected_label)
    ax1.annotate(
        f"Selected: {labels[idx_sel]}\n"
        f"({n_params[idx_sel]}p, {rms_vals[idx_sel]:.2f} px)",
        (n_params[idx_sel], bic_vals[idx_sel]),
        xytext=(n_params[idx_sel] - 15, bic_vals[idx_sel] + 80),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=10, color="black",
    )

    for i, lbl in enumerate(labels):
        ax1.annotate(lbl, (n_params[i], bic_vals[i]),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=9, ha="center", color=color1)

    ax2 = ax1.twinx()
    ax2.plot(n_params, rms_vals, "s--", color=color2, lw=2, ms=8,
             label="Pixel RMS")
    ax2.set_ylabel("Pixel RMS (px)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(bottom=0.38, top=0.50)

    ax1.set_title("Zernike order selection by BIC")
    ax1.grid(alpha=0.2)

    lines1, lbls1 = ax1.get_legend_handles_labels()
    lines2, lbls2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lbls1 + lbls2, loc="upper right", fontsize=9)

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
