"""Fit a compact scale law to the rigid-motion apparent-strain curves.

The exact gauge analysis often separates naturally into a short-range endpoint
term and a long-range coherent strain floor.  We fit, direction by direction,

    sigma_epsilon(L)^2 = (A / L)^2 + B^2,

where A has dimensions of displacement and B is a scale-independent strain
floor.  This is an empirical diagnostic, not an assumption used by the
calibration.  A nearly constant ``L * sigma_epsilon`` corresponds to B ~= 0;
a non-zero B indicates a spatially coherent distortion that survives gauge
averaging.

Input
-----
paper/strain/results/mechanical_observables.json

Outputs
-------
paper/strain/results/gauge_scale_law.json
paper/strain/figures/gauge_scale_decomposition.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
KINDS = ("ray", "soloff", "direct")
SERIES = ("calibration", "calibration2")
DIRECTIONS = ("horizontal", "vertical", "diag_pos", "diag_neg")
LABELS = {"ray": "Ray field", "soloff": "Soloff-type", "direct": "Direct polynomial"}


def fit_scale_law(rows: list[dict]) -> dict:
    rows = [r for r in rows if np.isfinite(r["exact_rms_microstrain"]) and r["nominal_length_mm"] > 0]
    L = np.asarray([r["nominal_length_mm"] for r in rows], dtype=float)
    eps = np.asarray([r["exact_rms_microstrain"] for r in rows], dtype=float)
    x = 1.0 / L**2
    y = eps**2

    X = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

    # Non-negative boundary solution for this two-parameter diagnostic model.
    if slope < 0 and intercept < 0:
        slope = 0.0
        intercept = 0.0
    elif slope < 0:
        slope = 0.0
        intercept = max(float(np.mean(y)), 0.0)
    elif intercept < 0:
        intercept = 0.0
        slope = max(float(np.dot(x, y) / np.dot(x, x)), 0.0)

    yhat = slope * x + intercept
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - np.sum((y - yhat) ** 2) / denom) if denom > 0 else float("nan")

    # sqrt(slope) is [microstrain * mm].  1 microstrain*mm = 1e-3 micrometre.
    A_um = float(np.sqrt(slope) * 1e-3)
    B_microstrain = float(np.sqrt(intercept))
    return {
        "n_lengths": int(len(L)),
        "length_min_mm": float(L.min()),
        "length_max_mm": float(L.max()),
        "endpoint_increment_scale_A_um": A_um,
        "coherent_strain_floor_B_microstrain": B_microstrain,
        "r2_squared_rms_space": r2,
        "length_mm": L.tolist(),
        "measured_rms_microstrain": eps.tolist(),
        "fitted_rms_microstrain": np.sqrt(np.maximum(yhat, 0)).tolist(),
    }


def main() -> None:
    source = RESULTS / "mechanical_observables.json"
    data = json.loads(source.read_text())
    fits: dict = {}

    for series in SERIES:
        fits[series] = {}
        for kind in KINDS:
            fits[series][kind] = {}
            for direction in DIRECTIONS:
                rows = data[series]["models"][kind]["directions"][direction]
                fits[series][kind][direction] = fit_scale_law(rows)

    (RESULTS / "gauge_scale_law.json").write_text(json.dumps(fits, indent=2, allow_nan=False))

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), sharey=True)
    for ax, series in zip(axes, SERIES):
        for kind in KINDS:
            # Use horizontal direction for the line/marker plot; vertical fit is
            # reported numerically and can be shown in the directional panel.
            f = fits[series][kind]["horizontal"]
            L = np.asarray(f["length_mm"])
            measured = np.asarray(f["measured_rms_microstrain"])
            fitted = np.asarray(f["fitted_rms_microstrain"])
            ax.plot(L, measured, "o", label=LABELS[kind])
            ax.plot(L, fitted, "--", linewidth=1.4)
        ax.set_xlabel("Horizontal virtual gauge length (mm)")
        ax.set_title("Wide sweep" if series == "calibration" else "Fine sweep")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"Apparent strain RMS ($\mu\varepsilon$)")
    axes[1].legend()
    fig.suptitle(r"Gauge-scale decomposition: $\sigma_\varepsilon^2=(A/L)^2+B^2$")
    fig.savefig(FIGURES / "gauge_scale_decomposition.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "gauge_scale_decomposition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    for series in SERIES:
        print(f"[{series}]")
        for kind in KINDS:
            h = fits[series][kind]["horizontal"]
            v = fits[series][kind]["vertical"]
            print(
                f"  {kind:7s}: H A={h['endpoint_increment_scale_A_um']:.3f} um, "
                f"B={h['coherent_strain_floor_B_microstrain']:.1f} ue, R2={h['r2_squared_rms_space']:.5f}; "
                f"V A={v['endpoint_increment_scale_A_um']:.3f} um, "
                f"B={v['coherent_strain_floor_B_microstrain']:.1f} ue, R2={v['r2_squared_rms_space']:.5f}"
            )


if __name__ == "__main__":
    main()
