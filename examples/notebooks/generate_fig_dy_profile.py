#!/usr/bin/env python3
"""d_y profile comparison (Figure 5 of the CMO paper).

Compares the vertical ray-direction component ``d_y(u, v)`` along the
sensor centre column for three models: Zernike rayfield (measured),
Telecentric CMO, and Perspective CMO. The Zernike measurement and the
Telecentric CMO overlap; the Perspective CMO predicts a much larger
linear gradient.

All inputs are read from the manifest in
``docs/assets/cmo_paper/figure5_dy_profile_comparison/``. Emits both PDF
(paper) and PNG (docs) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MANIFEST = Path("docs/assets/cmo_paper/figure5_dy_profile_comparison/manifest.json")
OUT_DIR = Path("paper/cmo/figures")
OUT_BASENAME = "dy_profile_comparison"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
})


def render(manifest: dict, manifest_root: Path, out_dir: Path) -> None:
    profile_path = (manifest_root / manifest["profile_json"]).resolve()
    dd = json.loads(profile_path.read_text(encoding="utf-8"))

    v_px = np.asarray(dd["v_px"], dtype=np.float64)
    zernike = np.asarray(dd["zernike"], dtype=np.float64)
    telecentric = np.asarray(dd["telecentric"], dtype=np.float64)
    perspective = np.asarray(dd["perspective"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=tuple(manifest["figure_size"]))
    ax.plot(v_px, zernike, "ko-", label="Zernike (measured)", ms=6)
    ax.plot(v_px, telecentric, "s--", color="darkgreen",
            label="Telecentric CMO", ms=7)
    ax.plot(v_px, perspective, "^:", color="darkred",
            label="Perspective CMO", ms=7)
    ax.axhline(0.0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("v (px)")
    ax.set_ylabel(r"$d_y$")
    ax.set_title(r"$d_y(u,v)$ profiles across sensor centre column")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    range_z = float(np.ptp(zernike))
    range_t = float(np.ptp(telecentric))
    range_p = float(np.ptp(perspective))
    ax.text(0.98, 0.98,
            f"Range: Zernike={range_z:.3f}, "
            f"Telecentric={range_t:.3f}, Perspective={range_p:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

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
