"""Mechanical held-out evaluation of the compact CMO 26p calibration.

The 26-parameter model was identified from the ten *wide-sweep* calibration
frames in the CMO study.  This script deliberately evaluates it only on that
same acquisition's reserved depth planes.  The fine sweep has a visibly
different image pose/scale and is calibrated independently in the Strain
analysis, so the wide CMO model is not silently transferred to it.

The CMO calibration used double-TPS completed/denoised calibration corners,
whereas the three retrospective estimators in ``evaluate.py`` use directly
detected corners.  The held-out evaluation below nevertheless uses exactly the
same raw correspondences as the Strain predictions; this distinction is kept
explicit in the manuscript rather than treating the four calibration pipelines
as identical fits.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from stereocomplex.physics import CMOTelecentricStereoModel  # noqa: E402
from mechanical_observables import (  # noqa: E402
    _cloud,
    _pair_family,
    _pairs,
    _analyse_group,
    _representative_maps,
    KINDS,
    LABELS,
    GRID_ROWS,
    GRID_COLS,
    PITCH_MM,
)
from fit_scale_law import fit_scale_law  # noqa: E402

RESULTS = ROOT / "paper/strain/results"
FIGURES = ROOT / "paper/strain/figures"
ASSETS = ROOT / "docs/assets/pycaso_real_data"


def _rotvec_matrix(rv: np.ndarray) -> np.ndarray:
    """Rodrigues exponential, matching scipy Rotation.from_rotvec."""
    rv = np.asarray(rv, dtype=float)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-15:
        return np.eye(3)
    axis = rv / theta
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-15)


def _apply_se3(O: np.ndarray, d: np.ndarray, rv: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = _rotvec_matrix(rv)
    return (R @ O.T).T + np.asarray(t)[None, :], _normalize((R @ d.T).T)


def _triangulate_rays(O1: np.ndarray, d1: np.ndarray, O2: np.ndarray, d2: np.ndarray):
    """Vectorised closest-point midpoint, identical to notebook 10."""
    n = np.cross(d1, d2, axis=1)
    n_norm = np.linalg.norm(n, axis=1)
    valid = n_norm > 1e-12
    w = O2 - O1
    denom = np.maximum(n_norm**2, 1e-30)
    t1 = np.sum(w * np.cross(d2, n, axis=1), axis=1) / denom
    t2 = np.sum(w * np.cross(d1, n, axis=1), axis=1) / denom
    P1 = O1 + t1[:, None] * d1
    P2 = O2 + t2[:, None] * d2
    midpoint = 0.5 * (P1 + P2)
    gap = np.linalg.norm(P1 - P2, axis=1)
    midpoint[~valid] = np.nan
    gap[~valid] = np.nan
    return midpoint, gap, valid


def _fit_model() -> tuple[CMOTelecentricStereoModel, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.load(ASSETS / "intermediate_state.npz")
    x = np.asarray(state["x_26p"], dtype=float)
    image_size = tuple(int(v) for v in state["image_size"])
    model = CMOTelecentricStereoModel.from_parameter_vector(
        x[:14], pixel_pitch_mm=0.0055, image_size=image_size
    )
    return model, x[14:17], x[17:20], x[20:23], x[23:26], x


def _predict_cmo(pixels: np.ndarray):
    model, rvL, tL, rvR, tR, x = _fit_model()
    uL, vL, uR, vR = pixels.T
    OL0, dL0 = model.ray(uL, vL, "left")
    OR0, dR0 = model.ray(uR, vR, "right")
    OL, dL = _apply_se3(OL0, dL0, rvL, tL)
    OR, dR = _apply_se3(OR0, dR0, rvR, tR)
    pred, gap, valid = _triangulate_rays(OL, dL, OR, dR)
    return pred, gap, valid, x


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = np.load(RESULTS / "predictions_calibration.npz")
    pixels = np.asarray(data["pixels"], dtype=float)
    frames = np.asarray(data["frames"], dtype=int)
    ids = np.asarray(data["ids"], dtype=int)
    z = np.asarray(data["z"], dtype=float)
    test = np.asarray(data["test_indices"], dtype=int)
    ref = int(np.argmin(np.abs(z - 3.0)))

    pred, gap, valid, x = _predict_cmo(pixels)
    if not np.all(valid):
        print(f"CMO valid triangulations: {valid.sum()}/{len(valid)}")

    np.savez_compressed(
        RESULTS / "predictions_cmo_calibration.npz",
        cmo=pred,
        gap=gap,
        valid=valid,
        pixels=pixels,
        frames=frames,
        ids=ids,
        z=z,
        test_indices=test,
        reference_frame=np.asarray([ref]),
        x_26p=x,
    )

    cloud = _cloud(pred, frames, ids, len(z))
    summary = {
        "series": "calibration",
        "scope": "wide sweep only; CMO 26p was identified on the ten wide calibration frames",
        "reference_frame": ref,
        "reference_z_mm": float(z[ref]),
        "median_ray_gap_um_all_raw_correspondences": float(np.nanmedian(gap) * 1e3),
        "p95_ray_gap_um_all_raw_correspondences": float(np.nanquantile(gap, 0.95) * 1e3),
        "models": {"cmo26": {"directions": {}}},
    }

    for direction, entries in _pair_family().items():
        rows = []
        for entry in entries:
            result = _analyse_group(cloud, ref, test, _pairs(entry["dx"], entry["dy"]))
            rows.append({**entry, **result})
        summary["models"]["cmo26"]["directions"][direction] = rows

    summary["scale_law"] = {
        direction: fit_scale_law(summary["models"]["cmo26"]["directions"][direction])
        for direction in ("horizontal", "vertical", "diag_pos", "diag_neg")
    }

    maps = _representative_maps(cloud, z, test, ref, step=4)
    np.savez_compressed(
        RESULTS / "cmo_mechanical_maps.npz",
        **{k: v for k, v in maps.items() if isinstance(v, np.ndarray)},
        representative_frame=np.asarray([maps["frame"]]),
        representative_z_mm=np.asarray([maps["z_mm"]]),
        representative_nominal_dz_mm=np.asarray([maps["nominal_dz_mm"]]),
    )
    summary["representative_frame"] = {
        "frame": int(maps["frame"]),
        "z_mm": float(maps["z_mm"]),
        "nominal_dz_mm": float(maps["nominal_dz_mm"]),
    }
    (RESULTS / "cmo_mechanical_observables.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False)
    )

    # Four-model wide-sweep gauge-length comparison.
    other = json.loads((RESULTS / "mechanical_observables.json").read_text())["calibration"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharey=True)
    for ax, direction in zip(axes, ("horizontal", "vertical")):
        for kind in KINDS:
            rows = other["models"][kind]["directions"][direction]
            ax.plot(
                [r["nominal_length_mm"] for r in rows],
                [r["exact_rms_microstrain"] for r in rows],
                marker="o", label=LABELS[kind]
            )
        rows = summary["models"]["cmo26"]["directions"][direction]
        ax.plot(
            [r["nominal_length_mm"] for r in rows],
            [r["exact_rms_microstrain"] for r in rows],
            marker="s", linewidth=2.0, label="CMO 26p"
        )
        ax.set_xlabel("Virtual gauge length (mm)")
        ax.set_title(direction.capitalize())
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"Rigid-motion apparent strain RMS ($\mu\varepsilon$)")
    axes[1].legend()
    fig.suptitle("Wide sweep: held-out mechanical consistency of the compact CMO calibration")
    fig.savefig(FIGURES / "cmo_wide_gauge_length.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "cmo_wide_gauge_length.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Representative CMO pseudo-strain map.
    pts_h = maps["horizontal_midpoints_mm"]
    eps_h = maps["horizontal_exact_microstrain"]
    pts_v = maps["vertical_midpoints_mm"]
    eps_v = maps["vertical_exact_microstrain"]
    vmax = float(np.quantile(np.abs(np.r_[eps_h, eps_v]), 0.98))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), sharex=True, sharey=True)
    for ax, pts, eps, title in zip(axes, (pts_h, pts_v), (eps_h, eps_v), ("Horizontal", "Vertical")):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=eps, s=22, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("X in CMO frame (mm)")
    axes[0].set_ylabel("Y in CMO frame (mm)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label=r"Apparent strain ($\mu\varepsilon$)", shrink=0.82)
    fig.suptitle("CMO 26p: 1.2 mm-equivalent rigid-motion pseudo-strain")
    fig.savefig(FIGURES / "cmo_wide_pseudo_strain.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "cmo_wide_pseudo_strain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"CMO median ray gap on all raw wide correspondences: {np.nanmedian(gap)*1e3:.3f} µm")
    for direction in ("horizontal", "vertical"):
        rows = summary["models"]["cmo26"]["directions"][direction]
        r4 = next(r for r in rows if r["step"] == 4)
        fit = summary["scale_law"][direction]
        print(
            f"CMO {direction}: L=1.2 mm nominal-grid step -> {r4['exact_rms_microstrain']:.1f} µε; "
            f"A={fit['endpoint_increment_scale_A_um']:.3f} µm, "
            f"B={fit['coherent_strain_floor_B_microstrain']:.1f} µε, "
            f"R2={fit['r2_squared_rms_space']:.5f}"
        )


if __name__ == "__main__":
    main()
