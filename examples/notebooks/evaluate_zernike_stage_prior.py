#!/usr/bin/env python3
"""Evaluate a hierarchical translation-stage prior on the Pycaso Zernike BA.

This experiment keeps the published corner observations, dense DIS
correspondences, profilometry registration, and plane-normalisation unchanged.
Only the constrained Zernike pose model changes:

``t_i = t0 + axis * (scale * z_nominal_i + jitter_i)``.

The output distinguishes the weak-prior proposal from increasingly hard stage
anchors. It is intentionally separate from the published calibration pipeline.

Run
---
    rtk .venv/bin/python examples/notebooks/evaluate_zernike_stage_prior.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_fig_profilo_relief_comparison import (  # noqa: E402
    _dis_correspondences,
    _plane_fit,
)

from stereocomplex.benchmarks.charuco_observation_simulator import (  # noqa: E402
    CharucoObservationSet,
)
from stereocomplex.benchmarks.rayfield_from_observations import (  # noqa: E402
    StagePosePrior,
    fit_constrained_zernike_rayfield,
)

CALIB = REPO / "docs/assets/pycaso_real_data"
PROFILE_ASSET = REPO / "docs/assets/cmo_paper/figure_external_profilo_relief"
FIGURE_ASSET = REPO / "docs/assets/cmo_paper/figure10b_zernike_stage_prior"
PYCASO = REPO / "examples/pycaso_data/Exemple/Images_example"
OUTPUT = FIGURE_ASSET / "stage_prior_results.json"
NEAR_HARD_RAYFIELD = FIGURE_ASSET / "near_hard_zernike_rayfield.npz"

K = np.array(
    [[25600.0, 0.0, 1024.0], [0.0, 25600.0, 1024.0], [0.0, 0.0, 1.0]]
)

CASES = {
    "weak_proposal": {
        "scale_sigma": 0.1,
        "jitter_sigma_mm": 0.07,
    },
    "tight_scale_moderate_jitter": {
        "scale_sigma": 0.001,
        "jitter_sigma_mm": 0.005,
    },
    "near_hard_ladder": {
        "scale_sigma": 0.001,
        "jitter_sigma_mm": 0.001,
    },
}


def _load_observations() -> CharucoObservationSet:
    """Build the calibration observation container from the versioned cache."""
    state = np.load(CALIB / "intermediate_state.npz")
    object_points = state["obj_pts"].astype(np.float64)
    rotations = state["opt_R"].astype(np.float64)
    translations = state["opt_t"].astype(np.float64)
    left_pixels = [frame.astype(np.float64) for frame in state["left_pixels"]]
    right_pixels = [frame.astype(np.float64) for frame in state["right_pixels"]]
    return CharucoObservationSet(
        object_points_mm=object_points,
        pose_rvecs=np.array(
            [Rotation.from_matrix(matrix).as_rotvec() for matrix in rotations]
        ),
        pose_tvecs=translations,
        left_pixels=left_pixels,
        right_pixels=right_pixels,
        point_indices=[
            np.arange(len(object_points), dtype=int) for _frame in left_pixels
        ],
        noise_std_px=0.0,
        image_size=tuple(int(value) for value in state["image_size"]),
    )


def _nominal_stage_positions() -> np.ndarray:
    """Read the common numeric calibration filenames in acquisition order."""
    left_dir = PYCASO / "left_calibration11"
    right_dir = PYCASO / "right_calibration11"
    left = {path.stem for path in left_dir.glob("*.png")}
    right = {path.stem for path in right_dir.glob("*.png")}
    return np.array(sorted(float(value) for value in left & right))


def _dense_correspondences() -> tuple[np.ndarray, ...]:
    """Recompute the tuned DIS field used by the profilometry figure."""
    params = json.loads((PROFILE_ASSET / "dis_params.json").read_text())
    left = cv2.imread(
        str(PYCASO / "left_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE
    )
    right = cv2.imread(
        str(PYCASO / "right_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE
    )
    if left is None or right is None:
        raise FileNotFoundError("raw Pycaso coin images are required")
    return _dis_correspondences(
        left.astype(np.float32) / 255.0,
        right.astype(np.float32) / 255.0,
        params,
    )


def _profilometry_window(
    canvas_size: int,
) -> tuple[float, tuple[int, int, int, int]]:
    """Return profilometry relief standard deviation and registered crop."""
    registration = json.loads((PROFILE_ASSET / "registration.json").read_text())
    profilometry = np.load(PROFILE_ASSET / "coin_profilo_recale.npy").astype(
        np.float64
    )
    registered = cv2.warpAffine(
        profilometry,
        np.asarray(registration["affine_2x3"], dtype=np.float64),
        (canvas_size, canvas_size),
        flags=cv2.INTER_LINEAR,
        borderValue=np.nan,
    )
    rows, columns = np.where(np.isfinite(registered))
    crop = (
        int(rows.min()),
        int(rows.max() + 1),
        int(columns.min()),
        int(columns.max() + 1),
    )
    y0, y1, x0, x1 = crop
    relief = _plane_fit(registered[y0:y1, x0:x1])
    return float(np.nanstd(relief)), crop


def _relief_std_um(
    left_field,
    right_field,
    correspondences: tuple[np.ndarray, ...],
    crop: tuple[int, int, int, int],
) -> float:
    """Triangulate the tuned dense field and return cropped relief amplitude."""
    u_left, v_left, u_right, v_right, in_bounds, canvas_size = correspondences
    z_mm = np.full(len(u_left), np.nan, dtype=np.float64)
    chunk_size = 250_000
    for start in range(0, len(u_left), chunk_size):
        stop = min(start + chunk_size, len(u_left))
        origin_left, direction_left = left_field.ray(
            u_left[start:stop], v_left[start:stop]
        )
        origin_right, direction_right = right_field.ray(
            u_right[start:stop], v_right[start:stop]
        )
        normal = np.cross(direction_left, direction_right)
        normal_sq = np.sum(normal * normal, axis=1)
        valid = normal_sq > 1e-24
        delta = origin_right - origin_left
        distance_left = (
            np.sum(delta * np.cross(direction_right, normal), axis=1) / normal_sq
        )
        distance_right = (
            np.sum(delta * np.cross(direction_left, normal), axis=1) / normal_sq
        )
        points = (
            origin_left
            + distance_left[:, None] * direction_left
            + origin_right
            + distance_right[:, None] * direction_right
        ) / 2.0
        z_chunk = points[:, 2]
        z_chunk[~valid] = np.nan
        z_mm[start:stop] = z_chunk

    canvas = np.full((canvas_size, canvas_size), np.nan, dtype=np.float64)
    columns = u_left.astype(int)
    rows = v_left.astype(int)
    valid = in_bounds & np.isfinite(z_mm)
    canvas[rows[valid], columns[valid]] = 1000.0 * z_mm[valid]
    y0, y1, x0, x1 = crop
    return float(np.nanstd(_plane_fit(canvas[y0:y1, x0:x1])))


def main() -> int:
    """Run the stage-prior sweep and write its calibration/relief diagnostics."""
    FIGURE_ASSET.mkdir(parents=True, exist_ok=True)
    observations = _load_observations()
    nominal_positions = _nominal_stage_positions()
    if len(nominal_positions) != len(observations.left_pixels):
        raise ValueError("nominal stage filenames do not match cached observations")

    correspondences = _dense_correspondences()
    profilometry_std_um, crop = _profilometry_window(int(correspondences[-1]))
    published = np.load(PROFILE_ASSET / "relief_comparison_data.npz")
    published_zernike_std_um = float(np.nanstd(published["zernike"]))

    results = {}
    for name, parameters in CASES.items():
        prior = StagePosePrior(
            nominal_positions_mm=nominal_positions,
            scale_sigma=parameters["scale_sigma"],
            jitter_sigma_mm=parameters["jitter_sigma_mm"],
            ray_sigma_mm=5.8e-4,
            estimate_axis=True,
        )
        left, right, diagnostics, _rotations, translations = (
            fit_constrained_zernike_rayfield(
                observations,
                observations.image_size,
                K,
                K,
                max_order_o=2,
                max_order_d=2,
                max_nfev=250,
                stage_prior=prior,
            )
        )
        relief_std_um = _relief_std_um(
            left, right, correspondences, crop
        )
        z_values = np.array(translations)[:, 2]
        results[name] = {
            **parameters,
            "ray_sigma_mm": prior.ray_sigma_mm,
            "ray_rms_mm": diagnostics.ray_rms_mm,
            "converged": diagnostics.converged,
            "nfev": diagnostics.nfev,
            "stage_scale": diagnostics.stage_scale,
            "stage_jitter_rms_mm": diagnostics.stage_jitter_rms_mm,
            "stage_axis": diagnostics.stage_axis,
            "fitted_z_positions_mm": z_values.tolist(),
            "fitted_z_span_mm": float(np.ptp(z_values)),
            "relief_std_um": relief_std_um,
            "relief_ratio_to_profilometry": relief_std_um / profilometry_std_um,
        }
        if name == "near_hard_ladder":
            np.savez_compressed(
                NEAR_HARD_RAYFIELD,
                left_origin_coeffs=left.origin_coeffs,
                left_direction_coeffs=left.direction_coeffs,
                right_origin_coeffs=right.origin_coeffs,
                right_direction_coeffs=right.direction_coeffs,
                stage_scale=np.array(diagnostics.stage_scale),
                stage_jitter_rms_mm=np.array(diagnostics.stage_jitter_rms_mm),
                stage_axis=np.asarray(diagnostics.stage_axis),
                fitted_translations_mm=np.asarray(translations),
            )
        print(
            f"{name}: scale={diagnostics.stage_scale:.4f}, "
            f"jitter={diagnostics.stage_jitter_rms_mm:.4f} mm, "
            f"relief={relief_std_um / profilometry_std_um:.3f}x"
        )

    published_pose_data = json.loads(
        (CALIB / "zernike_pose_variants.json").read_text()
    )["zernike_constrained"]
    profile_cache = np.load(PROFILE_ASSET / "relief_comparison_data.npz")
    output = {
        "description": "Hierarchical stage prior experiment on dense O(2)+d(2) Zernike BA",
        "nominal_stage_positions_mm": nominal_positions.tolist(),
        "profilometry_std_um": profilometry_std_um,
        "soloff_std_um": float(np.nanstd(profile_cache["soloff"])),
        "cmo26_std_um": float(np.nanstd(profile_cache["cmo26"])),
        "published_zernike_std_um": published_zernike_std_um,
        "published_zernike_ratio": published_zernike_std_um / profilometry_std_um,
        "published_zernike_z_positions_mm": published_pose_data["z_per_pose_mm"],
        "cases": results,
        "interpretation": (
            "The weak hierarchical prior does not resolve the published depth "
            "over-amplification. A near-hard per-frame ladder anchor is required."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT.relative_to(REPO)}")
    print(f"Wrote {NEAR_HARD_RAYFIELD.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
