#!/usr/bin/env python3
"""Robustness of the profilometry relief comparison to the DIS front-end.

The production DIS optical-flow parameters of the external-profilometry figure
were selected by inspecting the profilometric agreement (the tuning trace is
kept in ``dis_params.json``). A reviewer could therefore object that the
external validation was used to tune a hyper-parameter of the dense
reconstruction. This script answers that objection quantitatively: it re-runs
the *whole* relief comparison under several **non-tuned** DIS settings (the bare
OpenCV ULTRAFAST / FAST / MEDIUM preset defaults, no manual variational
refinement) and shows that the **relative** result is invariant to the
front-end:

* the CMO 26p and Soloff reliefs stay indistinguishable (CMO--Soloff RMS well
  below µ1 µm, cc > 0.999);
* the per-pixel Zernike rayfield keeps over-amplifying the relief by ~1.5-1.6x.

Because the same dense correspondence field feeds all three models for a given
setting, the comparison is an external *consistency check of the calibrated
models on a fixed correspondence field*, not a blind validation of the DIS
front-end -- and that conclusion does not depend on the tuning.

Reproducibility (``no orphan figures`` rule)
--------------------------------------------
Default run reads the versioned cache
``docs/assets/cmo_paper/figure_external_profilo_relief/dis_robustness.json`` and
writes the LaTeX table ``paper/cmo/tables/dis_robustness.tex``. ``--recompute``
re-derives the cache from the raw Pycaso coin images (git-ignored), reusing the
reconstruction helpers of ``generate_fig_profilo_relief_comparison.py`` so the
two artefacts can never drift apart.

Run
---
    rtk .venv/bin/python examples/notebooks/generate_tab_dis_robustness.py
    rtk .venv/bin/python examples/notebooks/generate_tab_dis_robustness.py --recompute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_fig_profilo_relief_comparison import (
    ASSET,
    REPO,
    _dis_correspondences,
    _reconstruct_windowed_reliefs,
)

CACHE = ASSET / "dis_robustness.json"
TABLE = REPO / "paper/cmo/tables/dis_robustness.tex"
PYCASO = REPO / "examples/pycaso_data/Exemple/Images_example"

# Row order in the table. The first row is the tuned production setting; the rest
# are bare OpenCV preset defaults (no manual variational-refinement tuning).
SETTINGS_ORDER = ["tuned", "ultrafast_default", "fast_default", "medium_default"]
SETTINGS_LABEL = {
    "tuned": "Tuned (production)",
    "ultrafast_default": "ULTRAFAST (defaults)",
    "fast_default": "FAST (defaults)",
    "medium_default": "MEDIUM (defaults)",
}


def _measure(reliefs: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-setting summary: amplitude ratios vs profilometry + CMO--Soloff agreement.

    Parameters
    ----------
    reliefs : dict[str, ndarray]
        Plane-normalised windowed reliefs (``profilometry``, ``soloff``,
        ``cmo26``, ``zernike``), micrometres.

    Returns
    -------
    dict[str, float]
        ``{<model>_ratio}`` = std(model) / std(profilometry); ``cc_cmo_soloff``
        and ``rms_cmo_soloff`` (µm) for the CMO-vs-Soloff agreement; plus the
        raw ``profilo_std`` for reference.
    """
    ref = reliefs["profilometry"]
    sp = float(np.nanstd(ref[np.isfinite(ref)]))
    out: dict[str, float] = {"profilo_std": sp}
    for name in ("soloff", "cmo26", "zernike"):
        r = reliefs[name]
        m = np.isfinite(r)
        out[f"{name}_std"] = float(np.nanstd(r[m]))
        out[f"{name}_ratio"] = float(np.nanstd(r[m]) / sp)
    mcs = np.isfinite(reliefs["cmo26"]) & np.isfinite(reliefs["soloff"])
    out["cc_cmo_soloff"] = float(np.corrcoef(reliefs["cmo26"][mcs], reliefs["soloff"][mcs])[0, 1])
    out["rms_cmo_soloff"] = float(
        np.sqrt(np.nanmean((reliefs["cmo26"][mcs] - reliefs["soloff"][mcs]) ** 2)))
    return out


def recompute_cache() -> None:
    """Re-run every DIS setting from the raw images and rewrite the JSON cache.

    Reads the tuned parameters and rigid registration from the figure asset
    folder, loads the raw Pycaso coin images, and for each setting runs DIS ->
    reconstruction -> windowing via the figure helpers, then stores the
    per-setting summary metrics. See the module docstring for provenance.
    """
    import cv2

    tuned = json.loads((ASSET / "dis_params.json").read_text())
    reg = json.loads((ASSET / "registration.json").read_text())
    base = {"border_px": tuned["border_px"],
            "disparity_filter_frac": tuned["disparity_filter_frac"]}
    settings = {
        "tuned": tuned,
        "ultrafast_default": {**base, "preset": "ultrafast", "preset_defaults": True},
        "fast_default": {**base, "preset": "fast", "preset_defaults": True},
        "medium_default": {**base, "preset": "medium", "preset_defaults": True},
    }

    imgL = cv2.imread(str(PYCASO / "left_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(str(PYCASO / "right_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise SystemExit(
            f"Raw Pycaso coin images not found under {PYCASO} (git-ignored dataset). "
            "Run without --recompute to use the versioned cache.")
    imgL = imgL.astype(np.float32) / 255.0
    imgR = imgR.astype(np.float32) / 255.0

    results = {}
    for key in SETTINGS_ORDER:
        uL, vL, uR, vR, inb, W = _dis_correspondences(imgL, imgR, settings[key])
        reliefs = _reconstruct_windowed_reliefs(uL, vL, uR, vR, inb, W, reg)
        results[key] = _measure(reliefs)
        m = results[key]
        print(f"  {SETTINGS_LABEL[key]:22s} "
              f"Soloff {m['soloff_ratio']:.2f}x  CMO {m['cmo26_ratio']:.2f}x  "
              f"Zernike {m['zernike_ratio']:.2f}x  "
              f"CMO-Soloff RMS {m['rms_cmo_soloff']:.2f}um cc {m['cc_cmo_soloff']:.4f}")

    CACHE.write_text(json.dumps(results, indent=2))
    print(f"  wrote cache {CACHE.relative_to(REPO)}")


def write_table() -> None:
    """Render ``paper/cmo/tables/dis_robustness.tex`` from the JSON cache."""
    res = json.loads(CACHE.read_text())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robustness of the relief comparison to the dense-matching "
        r"front-end. Amplitude ratios are the plane-normalised relief standard "
        r"deviation relative to the profilometry; the last two columns quantify "
        r"the CMO--Soloff agreement. The production DIS parameters were tuned "
        r"against the profilometry, but the bare OpenCV preset defaults "
        r"(no manual variational refinement) reproduce the same relative result: "
        r"CMO~$\approx$~Soloff and the Zernike ray-field over-amplifies. The "
        r"external comparison is therefore a consistency check of the calibrated "
        r"models on a fixed correspondence field, not a tuning of the DIS "
        r"front-end.}",
        r"\label{tab:dis_robustness}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        r"DIS setting & Soloff & CMO 26p & Zernike & CMO--Soloff & CMO--Soloff \\",
        r" & ($\times$) & ($\times$) & ($\times$) & RMS (\si{\micro\metre}) & cc \\",
        r"\hline",
    ]
    for key in SETTINGS_ORDER:
        m = res[key]
        lines.append(
            f"{SETTINGS_LABEL[key]} & {m['soloff_ratio']:.2f} & {m['cmo26_ratio']:.2f} & "
            f"{m['zernike_ratio']:.2f} & {m['rms_cmo_soloff']:.2f} & {m['cc_cmo_soloff']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines))
    print(f"  wrote {TABLE.relative_to(REPO)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true",
                    help="re-run all DIS settings from the Pycaso images, rewrite the cache")
    args = ap.parse_args()
    if args.recompute or not CACHE.exists():
        if not CACHE.exists() and not args.recompute:
            print(f"cache {CACHE.name} missing -> recomputing")
        recompute_cache()
    write_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
