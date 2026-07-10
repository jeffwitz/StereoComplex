#!/usr/bin/env python3
"""Ablate direct, marker-TPS, and double-TPS corners on CMO model diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from cmo_corner_preprocessing import (  # noqa: E402
    marker_tps_corners,
    second_tps_pass,
)

from stereocomplex.benchmarks.charuco_observation_simulator import (  # noqa: E402
    CharucoObservationSet,
)
from stereocomplex.benchmarks.rayfield_from_observations import (  # noqa: E402
    fit_constrained_zernike_rayfield,
)
from stereocomplex.physics.cmo_physical import (  # noqa: E402
    CMOTelecentricStereoModel,
    _normalize,
)

PYCASO = ROOT / "examples" / "pycaso_data" / "Exemple" / "Images_example"
LEFT_DIR = PYCASO / "left_calibration11"
RIGHT_DIR = PYCASO / "right_calibration11"
OUT = ROOT / "docs" / "assets" / "pycaso_real_data" / "corner_preprocessing_ablation.json"
NCX, NCY, SQUARE_MM = 16, 12, 0.3
IMAGE_SIZE = (2048, 2048)
W, H = IMAGE_SIZE
FX = 25600.0
K = np.array([[FX, 0, 1024], [0, FX, 1024], [0, 0, 1]], dtype=np.float64)


def _detectors():
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    board = aruco.CharucoBoard(
        (NCX, NCY), SQUARE_MM, SQUARE_MM / 2, dictionary
    )
    board.setLegacyPattern(True)
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 75
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.005
    params.maxMarkerPerimeterRate = 0.20
    params.polygonalApproxAccuracyRate = 0.08
    params.minCornerDistanceRate = 0.02
    params.minDistanceToBorder = 1
    return (
        board,
        aruco.ArucoDetector(dictionary, params),
        aruco.CharucoDetector(board),
    )


def _apply_se3(origins, directions, rotvec, translation):
    rotation = Rotation.from_rotvec(rotvec).as_matrix()
    return (
        (rotation @ origins.T).T + translation[None, :],
        _normalize((rotation @ directions.T).T),
    )


def _fit_compact(left_field, right_field, pose_t):
    u_grid, v_grid = np.meshgrid(
        np.linspace(0, W - 1, 41), np.linspace(0, H - 1, 41)
    )
    u, v = u_grid.ravel(), v_grid.ravel()
    origin_l, direction_l = left_field.ray(u, v)
    origin_r, direction_r = right_field.ray(u, v)
    centre = np.array([1024.0])
    centre_origin_l, centre_direction_l = left_field.ray(centre, centre)
    centre_origin_r, _ = right_field.ray(centre, centre)
    working_distance = float(np.mean(np.asarray(pose_t)[:, 2]))
    baseline = float(np.linalg.norm(centre_origin_r[0] - centre_origin_l[0]))
    effective_length = working_distance - float(
        (abs(centre_origin_l[0, 2]) + abs(centre_origin_r[0, 2])) / 2
    )
    theta = float(np.arctan2(baseline / 2, effective_length))
    x0 = np.array([
        effective_length, working_distance, baseline, W / 2, H / 2,
        effective_length, theta, centre_direction_l[0, 1],
        0, 0, 0, 0, 0, 0,
    ])
    lower = np.array([
        1, 1, 0, 0, 0, 20, 0, -0.3,
        -10, -10, -10, -10, -10, -10,
    ], dtype=float)
    upper = np.array([
        500, 1000, 200, W, H, 200, 0.5, 0.3,
        10, 10, 10, 10, 10, 10,
    ], dtype=float)

    def residuals_base(parameters):
        model = CMOTelecentricStereoModel.from_parameter_vector(
            parameters, pixel_pitch_mm=0.0055, image_size=IMAGE_SIZE
        )
        model_l = model.ray(u, v, "left")
        model_r = model.ray(u, v, "right")
        return _two_plane_residuals(
            (model_l, model_r),
            ((origin_l, direction_l), (origin_r, direction_r)),
        )

    base_solution = least_squares(
        residuals_base,
        x0=x0,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        max_nfev=500,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    base_model = CMOTelecentricStereoModel.from_parameter_vector(
        base_solution.x, pixel_pitch_mm=0.0055, image_size=IMAGE_SIZE
    )
    direction_residuals = []
    for channel, reference in (
        ("left", direction_l), ("right", direction_r)
    ):
        _, model_direction = base_model.ray(u, v, channel)
        direction_residuals.append(reference - model_direction)
    total_energy = sum(float(np.sum(block**2)) for block in direction_residuals)
    piston_energy = sum(
        len(block) * float(np.sum(np.mean(block, axis=0) ** 2))
        for block in direction_residuals
    )

    rotation_bound = np.full(3, 0.08)
    translation_bound = np.full(3, 3.0)
    arm_bound = np.concatenate([
        rotation_bound, translation_bound, rotation_bound, translation_bound
    ])

    def residuals_aligned(parameters):
        model = CMOTelecentricStereoModel.from_parameter_vector(
            parameters[:14], pixel_pitch_mm=0.0055, image_size=IMAGE_SIZE
        )
        model_l = _apply_se3(
            *model.ray(u, v, "left"), parameters[14:17], parameters[17:20]
        )
        model_r = _apply_se3(
            *model.ray(u, v, "right"), parameters[20:23], parameters[23:26]
        )
        return _two_plane_residuals(
            (model_l, model_r),
            ((origin_l, direction_l), (origin_r, direction_r)),
        )

    aligned_solution = least_squares(
        residuals_aligned,
        x0=np.concatenate([base_solution.x, np.zeros(12)]),
        bounds=(
            np.concatenate([lower, -arm_bound]),
            np.concatenate([upper, arm_bound]),
        ),
        loss="huber",
        f_scale=1.0,
        max_nfev=500,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    rms_base = float(np.sqrt(np.mean(residuals_base(base_solution.x) ** 2)))
    rms_aligned = float(np.sqrt(np.mean(residuals_aligned(aligned_solution.x) ** 2)))
    return {
        "base_14p_ray_rms_mm": rms_base,
        "aligned_26p_ray_rms_mm": rms_aligned,
        "se3_improvement_factor": rms_base / rms_aligned,
        "base_direction_piston_energy_fraction": piston_energy / total_energy,
    }


def _two_plane_residuals(models, references):
    blocks = []
    for z_plane in (50.0, 80.0):
        for (model_o, model_d), (ref_o, ref_d) in zip(
            models, references, strict=True
        ):
            ref_t = (z_plane - ref_o[:, 2]) / ref_d[:, 2]
            model_t = (z_plane - model_o[:, 2]) / model_d[:, 2]
            ref_points = ref_o + ref_t[:, None] * ref_d
            model_points = model_o + model_t[:, None] * model_d
            blocks.append((ref_points - model_points).reshape(-1))
    return np.concatenate(blocks)


def _make_observations(obj_pts, pixels_l, pixels_r, indices):
    rvecs, tvecs = [], []
    for left, point_ids in zip(pixels_l, indices, strict=True):
        success, rvec, tvec = cv2.solvePnP(
            obj_pts[point_ids].astype(np.float32),
            left.astype(np.float32),
            K.astype(np.float32),
            np.zeros(5, dtype=np.float32),
        )
        rvecs.append(rvec.ravel() if success else np.zeros(3))
        tvecs.append(tvec.ravel() if success else np.array([0.0, 0.0, 65.0]))
    return CharucoObservationSet(
        object_points_mm=obj_pts,
        pose_rvecs=np.asarray(rvecs, dtype=np.float64),
        pose_tvecs=np.asarray(tvecs, dtype=np.float64),
        left_pixels=pixels_l,
        right_pixels=pixels_r,
        point_indices=indices,
        noise_std_px=0.0,
        image_size=IMAGE_SIZE,
    )


def main() -> int:
    board, marker_detector, charuco_detector = _detectors()
    obj_pts = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    chessboard_xy = obj_pts[:, :2]
    marker_object_xy = {
        int(marker_id): np.asarray(points, dtype=np.float64)[:, :2]
        for marker_id, points in zip(
            np.asarray(board.getIds()).reshape(-1),
            board.getObjPoints(),
            strict=True,
        )
    }
    paired = sorted(
        {p.stem for p in LEFT_DIR.glob("*.png")}
        & {p.stem for p in RIGHT_DIR.glob("*.png")},
        key=float,
    )
    variants = {
        name: {"left": [], "right": [], "indices": []}
        for name in ("direct_common", "marker_tps", "double_tps")
    }
    for z_value in paired:
        detected = {}
        for channel, image_dir in (("left", LEFT_DIR), ("right", RIGHT_DIR)):
            gray = cv2.imread(str(image_dir / f"{z_value}.png"), cv2.IMREAD_GRAYSCALE)
            corners, ids, _, _ = charuco_detector.detectBoard(gray)
            marker_corners, marker_ids, _ = marker_detector.detectMarkers(gray)
            ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
            corners_xy = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            first = marker_tps_corners(
                marker_corners, marker_ids, marker_object_xy, chessboard_xy
            )
            detected[channel] = {
                "ids": ids_flat,
                "corners": corners_xy,
                "first": first,
                "second": second_tps_pass(chessboard_xy, first),
            }
        common = np.intersect1d(detected["left"]["ids"], detected["right"]["ids"])
        for channel in ("left", "right"):
            lookup = {
                int(corner_id): corner
                for corner_id, corner in zip(
                    detected[channel]["ids"], detected[channel]["corners"], strict=True
                )
            }
            variants["direct_common"][channel].append(
                np.asarray([lookup[int(corner_id)] for corner_id in common])
            )
            variants["marker_tps"][channel].append(detected[channel]["first"])
            variants["double_tps"][channel].append(detected[channel]["second"])
        variants["direct_common"]["indices"].append(common)
        full_ids = np.arange(len(obj_pts), dtype=int)
        variants["marker_tps"]["indices"].append(full_ids)
        variants["double_tps"]["indices"].append(full_ids)

    results = {}
    for name, data in variants.items():
        observations = _make_observations(
            obj_pts, data["left"], data["right"], data["indices"]
        )
        left_field, right_field, diagnostics, _, pose_t = (
            fit_constrained_zernike_rayfield(
                observations,
                image_size=IMAGE_SIZE,
                K_left=K,
                K_right=K.copy(),
                max_order_d=2,
                max_nfev=500,
                origin_reg_weight=0.0,
            )
        )
        v = np.linspace(0, H - 1, 401)
        u = np.full_like(v, (W - 1) / 2)
        dy_ranges = []
        for field in (left_field, right_field):
            _, directions = field.ray(u, v)
            dy_ranges.append(float(np.ptp(directions[:, 1])))
        result = {
            "n_stereo_corners": int(sum(map(len, data["indices"]))),
            "zernike_ray_rms_mm": float(diagnostics.ray_rms_mm),
            "dy_range_mean": float(np.mean(dy_ranges)),
            "dy_range_per_channel": dy_ranges,
        }
        result.update(_fit_compact(left_field, right_field, pose_t))
        results[name] = result
        print(
            f"{name}: ray={result['zernike_ray_rms_mm']:.6f} mm, "
            f"dy={result['dy_range_mean']:.3f}, "
            f"14p/26p={result['se3_improvement_factor']:.1f}x"
        )

    output = {
        "perspective_cmo_dy_range_reference": 0.232,
        "variants": results,
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
