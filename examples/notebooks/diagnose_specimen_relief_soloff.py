#!/usr/bin/env python3
"""Specimen depth-relief comparison: Soloff (degree sweep) vs CMO 26p vs Zernike.

Diagnostic behind the question "does the ~30 % CMO--Zernike relief discrepancy
mean the CMO *loses* axial scale, or the Zernike rayfield *over-amplifies* it?".

The three reconstructions consume the **same** dense DIS correspondence field on
the coin specimen, so the fine relief is shared; what differs between models is
the disparity-to-depth *gain*. The relief amplitude ratio therefore probes that
gain directly. This script:

* fits a Soloff 4-variable polynomial (degrees 2/3/4, ridge 1e-3) on the ChArUco
  calibration corners and predicts the specimen depth;
* loads the committed CMO 26p and Zernike 57p specimen reconstructions;
* plane-normalises every surface on one common, seeded sub-sample and reports
  the relief std / MAD and the per-point regression against the Soloff degree-3
  reference (slope + R^2 -> coherent gain vs. random noise);
* writes a JSON audit next to the other Pycaso diagnostics.

KNOWN CIRCULARITY CAVEAT (recorded in the JSON ``caveat`` field). The Soloff
fit here is trained on calibration ``XYZ`` obtained from ``obj_pts`` transformed
by the per-frame poses ``opt_R``/``opt_t`` of ``intermediate_state.npz`` -- which
come from the CMO joint fit -- and it shares the ``fx = 25600`` length gauge with
the CMO model. Part of any Soloff-vs-CMO agreement is therefore inherited, not
independent. The metrologically decisive reference is the **native Pycaso Soloff**
(independently calibrated on stage positions and validated by profilometry),
applied to the same specimen; plug it in via ``--native-soloff-npz`` once
available.

Run::

    PYTHONPATH=src rtk .venv/bin/python \\
        examples/notebooks/diagnose_specimen_relief_soloff.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Use the working-tree source so the current SoloffPolynomialModel is picked up
# even if an older stereocomplex is installed in the environment.
_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if _REPO_SRC.is_dir():
    sys.path.insert(0, str(_REPO_SRC))

from stereocomplex.eval.soloff_poly import SoloffPolynomialModel  # noqa: E402

ASSETS = Path("docs/assets/pycaso_real_data")
OUT_JSON = ASSETS / "specimen_relief_soloff_comparison.json"


def _plane_normalised_relief(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Subtract the best-fit affine plane ``z ~ a*x + b*y + c`` from a surface.

    Parameters
    ----------
    x, y, z : ndarray, shape (N,)
        Reconstructed point coordinates in millimetres (one model, one frame).

    Returns
    -------
    ndarray, shape (N,)
        The plane-normalised relief in millimetres (the surface signal with its
        dominant tilt removed). Frame-invariant amplitude statistics can be read
        from this, independently of each model's own gauge orientation.
    """
    design = np.column_stack([x, y, np.ones(x.shape[0])])
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    return z - design @ coeffs


def _build_calibration(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the Soloff calibration set from the saved intermediate state.

    The 3-D target of the Soloff fit is the known board geometry ``obj_pts``
    placed in the reference frame by each frame's pose; the inputs are the
    refined left/right corner pixels.

    Parameters
    ----------
    state : dict
        Contents of ``intermediate_state.npz`` (keys ``left_pixels``,
        ``right_pixels``, ``obj_pts``, ``opt_R``, ``opt_t``).

    Returns
    -------
    uv_left : ndarray, shape (F*C, 2)
        Left corner pixels stacked over frames, px.
    uv_right : ndarray, shape (F*C, 2)
        Right corner pixels stacked over frames, px.
    xyz_mm : ndarray, shape (F*C, 3)
        Calibration points in the reference frame, mm.
    """
    left = np.asarray(state["left_pixels"], dtype=np.float64).reshape(-1, 2)
    right = np.asarray(state["right_pixels"], dtype=np.float64).reshape(-1, 2)
    obj = np.asarray(state["obj_pts"], dtype=np.float64)
    rot = np.asarray(state["opt_R"], dtype=np.float64)
    tvec = np.asarray(state["opt_t"], dtype=np.float64)
    xyz = np.concatenate([obj @ rot[i].T + tvec[i] for i in range(rot.shape[0])], axis=0)
    return left, right, xyz


def _relief_stats(relief: np.ndarray) -> dict:
    """Return std and median-absolute-deviation (both mm) of a relief sample."""
    med = float(np.median(relief))
    return {
        "std_mm": float(np.std(relief)),
        "mad_mm": float(np.median(np.abs(relief - med))),
    }


def _regress(ref: np.ndarray, other: np.ndarray) -> dict:
    """Least-squares fit ``other = slope * ref + intercept`` with its R^2.

    A slope near 1 with high R^2 means the two relief maps are the same surface
    up to a scalar depth gain; a low R^2 means the difference is incoherent
    (noise) rather than a clean gain.
    """
    slope, intercept = np.polyfit(ref, other, 1)
    pred = slope * ref + intercept
    ss_res = float(np.sum((other - pred) ** 2))
    ss_tot = float(np.sum((other - other.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": float(slope), "r2": r2}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sample", type=int, default=200_000,
                        help="common-valid points used for the comparison")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=1e-3,
                        help="Soloff ridge strength (paper convention)")
    parser.add_argument("--degrees", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--ref-degree", type=int, default=3,
                        help="Soloff degree used as the regression reference")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    state = dict(np.load(ASSETS / "intermediate_state.npz"))
    uv_left, uv_right, xyz_cal = _build_calibration(state)

    corr = np.load(ASSETS / "specimen_correspondences.npz")
    uvl = np.column_stack([corr["uL"], corr["vL"]]).astype(np.float64)
    uvr = np.column_stack([corr["uR"], corr["vR"]]).astype(np.float64)

    cmo = np.load(ASSETS / "specimen_reconstruction_cmo26.npz")
    zer = np.load(ASSETS / "specimen_reconstruction_zernike.npz")

    valid = (
        np.asarray(cmo["valid"], dtype=bool)
        & np.asarray(zer["valid"], dtype=bool)
        & np.isfinite(cmo["Z"])
        & np.isfinite(zer["Z"])
    )
    idx_all = np.flatnonzero(valid)
    idx = rng.choice(idx_all, size=min(args.n_sample, idx_all.size), replace=False)

    # Soloff degree sweep: fit on calibration, predict on the specimen sample.
    soloff_models = {
        d: SoloffPolynomialModel.fit(uv_left, uv_right, xyz_cal, degree=d, ridge=args.ridge)
        for d in args.degrees
    }
    calib_rms_um = {}
    reliefs: dict[str, np.ndarray] = {}
    for d, model in soloff_models.items():
        pred_cal = model.predict(uv_left, uv_right)
        cal_err = np.sqrt(np.mean(np.sum((pred_cal - xyz_cal) ** 2, axis=1)))
        calib_rms_um[d] = float(cal_err * 1000.0)
        xyz = model.predict(uvl[idx], uvr[idx])
        reliefs[f"soloff_deg{d}"] = _plane_normalised_relief(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    def _xyz(arr: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
        return arr[key][idx].astype(np.float64)

    reliefs["cmo_26p"] = _plane_normalised_relief(_xyz(cmo, "X"), _xyz(cmo, "Y"), _xyz(cmo, "Z"))
    reliefs["zernike_57p"] = _plane_normalised_relief(
        _xyz(zer, "X"), _xyz(zer, "Y"), _xyz(zer, "Z")
    )

    ref_key = f"soloff_deg{args.ref_degree}"
    ref_std = float(np.std(reliefs[ref_key]))
    models_out = {}
    for key, relief in reliefs.items():
        stats = _relief_stats(relief)
        stats["std_over_ref"] = stats["std_mm"] / ref_std
        if key != ref_key:
            stats.update(_regress(reliefs[ref_key], relief))
        models_out[key] = stats

    # Full-set CMO/Zernike std to confirm the sub-sample is representative.
    full = {
        "cmo_26p": float(np.std(_plane_normalised_relief(
            cmo["X"][idx_all].astype(np.float64), cmo["Y"][idx_all].astype(np.float64),
            cmo["Z"][idx_all].astype(np.float64)))),
        "zernike_57p": float(np.std(_plane_normalised_relief(
            zer["X"][idx_all].astype(np.float64), zer["Y"][idx_all].astype(np.float64),
            zer["Z"][idx_all].astype(np.float64)))),
    }

    out = {
        "description": (
            "Plane-normalised specimen relief amplitude for Soloff (degree sweep), "
            "CMO 26p and Zernike 57p, on one common seeded sub-sample of the dense "
            "DIS correspondence field. All models share the same correspondences, so "
            "the relief amplitude ratio probes the disparity-to-depth gain."
        ),
        "caveat": (
            "CIRCULARITY: this Soloff is re-fitted on calibration XYZ = obj_pts @ "
            "(opt_R, opt_t) taken from the CMO joint fit, and shares the fx=25600 "
            "gauge with the CMO model, so Soloff-vs-CMO agreement is partly inherited. "
            "The metrologically independent reference is the native profilometry-"
            "validated Pycaso Soloff applied to the same specimen."
        ),
        "inputs": {
            "calibration": "intermediate_state.npz (corners + CMO-fit poses)",
            "specimen_correspondences": "specimen_correspondences.npz",
            "cmo_reconstruction": "specimen_reconstruction_cmo26.npz",
            "zernike_reconstruction": "specimen_reconstruction_zernike.npz",
        },
        "method": {
            "soloff_ridge": args.ridge,
            "soloff_degrees": list(args.degrees),
            "regression_reference": ref_key,
            "n_common_valid": int(idx_all.size),
            "n_sample": int(idx.size),
            "seed": args.seed,
        },
        "soloff_calibration_rms_um": {str(k): v for k, v in calib_rms_um.items()},
        "relief_full_set_std_mm": full,
        "models": models_out,
        "command": (
            "PYTHONPATH=src rtk .venv/bin/python "
            "examples/notebooks/diagnose_specimen_relief_soloff.py"
        ),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT_JSON}")
    print(f"\nSoloff calibration 3D RMS (um): {calib_rms_um}")
    cols = f"{'std(mm)':>9s} {'MAD(mm)':>9s} {'std/ref':>9s} {'slope':>8s} {'R2':>7s}"
    print(f"\n{'model':14s} {cols}")
    for key, stats in models_out.items():
        slope = stats.get("slope", float("nan"))
        r2 = stats.get("r2", float("nan"))
        print(f"{key:14s} {stats['std_mm']:9.4f} {stats['mad_mm']:9.4f} "
              f"{stats['std_over_ref']:9.3f} {slope:8.3f} {r2:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
