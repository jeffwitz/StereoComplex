from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


Side = Literal["left", "right"]


@dataclass(frozen=True)
class ViewDetections:
    marker_ids: np.ndarray  # (M,)
    marker_corners: list[np.ndarray]  # list of (4,2)
    charuco_ids: np.ndarray  # (K,)
    charuco_xy: np.ndarray  # (K,2) in dataset pixel-center convention


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"rms": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "rms": _rms(x),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(scene_dir: Path) -> list[dict[str, Any]]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict[str, Any]] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def build_charuco_from_meta(meta: dict[str, Any]):
    import cv2  # type: ignore
    import cv2.aruco as aruco  # type: ignore

    board_meta = meta["board"]
    dict_name = str(board_meta["aruco_dictionary"])
    dictionary = getattr(aruco, dict_name)
    dictionary = aruco.getPredefinedDictionary(dictionary)

    squares_x = int(board_meta["squares_x"])
    squares_y = int(board_meta["squares_y"])
    square_size = float(board_meta["square_size_mm"])
    marker_size = float(board_meta["marker_size_mm"])

    board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    detector_params = aruco.DetectorParameters()

    aruco_detector = None
    charuco_detector = None
    if hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(board, aruco.CharucoParameters(), detector_params)
    return cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector


def detect_view(
    cv2,
    aruco,
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img: np.ndarray,
) -> ViewDetections | None:
    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)
        if marker_ids is None or marker_corners is None or len(marker_ids) == 0:
            return None
        if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
            return None
        # Charuco corners are returned in an (i+0.5, j+0.5) convention; convert to dataset convention
        # (pixel centers at integer coordinates).
        charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
        return ViewDetections(
            marker_ids=np.asarray(marker_ids, dtype=np.int32).reshape(-1),
            # Marker corners appear already aligned with the dataset convention (integer-centered).
            marker_corners=[np.asarray(c, dtype=np.float64).reshape(4, 2) for c in marker_corners],
            charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
            charuco_xy=charuco_xy,
        )

    if aruco_detector is not None:
        corners, ids, _rejected = aruco_detector.detectMarkers(img)
    else:  # pragma: no cover
        corners, ids, _rejected = aruco.detectMarkers(img, dictionary, parameters=detector_params)
    if ids is None or len(ids) == 0:
        return None
    ret = aruco.interpolateCornersCharuco(corners, ids, img, board)
    if ret is None:
        return None
    charuco_corners, charuco_ids, _ = ret
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) == 0:
        return None
    charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
    return ViewDetections(
        marker_ids=np.asarray(ids, dtype=np.int32).reshape(-1),
        marker_corners=[np.asarray(c, dtype=np.float64).reshape(4, 2) for c in corners],
        charuco_ids=np.asarray(charuco_ids, dtype=np.int32).reshape(-1),
        charuco_xy=charuco_xy,
    )


def _dict_from_ids_xy(ids: np.ndarray, xy: np.ndarray) -> dict[int, np.ndarray]:
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    return {int(i): xy[k].astype(np.float64) for k, i in enumerate(ids.tolist())}


def _stack_for_ids(ids: list[int], m: dict[int, np.ndarray]) -> np.ndarray:
    return np.stack([m[int(i)] for i in ids], axis=0).astype(np.float64)


def _project_brown(
    *,
    view_meta: dict[str, Any],
    f_um: float,
    brown: dict[str, Any],
    XYZ_cam_mm: np.ndarray,
) -> np.ndarray:
    """
    Forward project 3D points expressed in camera coordinates to distorted pixels.
    """
    from stereocomplex.core.distortion import brown_from_dict  # noqa: PLC0415
    from stereocomplex.core.geometry import sensor_um_to_pixel  # noqa: PLC0415
    from stereocomplex.meta import parse_view_meta  # noqa: PLC0415

    view = parse_view_meta(view_meta)
    XYZ_cam_mm = np.asarray(XYZ_cam_mm, dtype=np.float64)
    X = XYZ_cam_mm[:, 0]
    Y = XYZ_cam_mm[:, 1]
    Z = XYZ_cam_mm[:, 2]

    uv = np.full((XYZ_cam_mm.shape[0], 2), np.nan, dtype=np.float64)
    good = np.isfinite(Z) & (np.abs(Z) > 1e-12)
    if not np.any(good):
        return uv

    x = X[good] / Z[good]
    y = Y[good] / Z[good]
    dist = brown_from_dict(brown)
    xd, yd = dist.distort(x, y)
    x_um = xd * float(f_um)
    y_um = yd * float(f_um)
    u_px, v_px = sensor_um_to_pixel(view, x_um, y_um)
    uv[good, 0] = u_px
    uv[good, 1] = v_px
    return uv


def _pinhole_rays_from_pixels(
    *,
    view_meta: dict[str, Any],
    f_um: float,
    brown: dict[str, Any],
    uv_px: np.ndarray,
) -> np.ndarray:
    """
    Convert distorted pixels (uv) -> undistorted ray directions using the known pinhole+Brown model.
    """
    from stereocomplex.core.distortion import brown_from_dict  # noqa: PLC0415
    from stereocomplex.core.geometry import PinholeCamera, pixel_to_sensor_um  # noqa: PLC0415
    from stereocomplex.meta import parse_view_meta  # noqa: PLC0415

    view = parse_view_meta(view_meta)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    x_um, y_um = pixel_to_sensor_um(view, uv_px[:, 0], uv_px[:, 1])

    xd = x_um / float(f_um)
    yd = y_um / float(f_um)

    dist = brown_from_dict(brown)
    x, y = dist.undistort(xd, yd, iterations=12)
    return PinholeCamera(f_um=float(f_um)).ray_directions_cam_from_norm(x, y)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Detect points from images, apply 2D ray-field correction, then compare 3D reconstructions."
    )
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit frames (0=all).")

    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=3.0)
    ap.add_argument("--tps-iters", type=int, default=3)

    ap.add_argument("--nmax", type=int, default=12, help="Zernike max radial order for central ray-field 3D.")
    ap.add_argument("--lam3d", type=float, default=1e-3, help="Ridge lambda for central ray-field 3D fit.")
    ap.add_argument("--out", type=Path, default=Path("paper/tables/3d_from_images_rayfield2d.json"))
    args = ap.parse_args()

    scene_dir = Path(args.dataset_root) / str(args.split) / str(args.scene)
    meta = load_json(scene_dir / "meta.json")
    frames = load_frames(scene_dir)
    if args.max_frames and args.max_frames > 0:
        frames = frames[: int(args.max_frames)]

    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(meta)

    # Ray-field 2D (TPS robust) predictor.
    from stereocomplex.eval.charuco_detection import _predict_points_rayfield_tps_robust  # noqa: PLC0415

    # For ray-field prediction: marker-id -> board-plane marker corners.
    board_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = board.getObjPoints()
    id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

    # GT correspondences.
    gt = np.load(str(scene_dir / "gt_charuco_corners.npz"))
    gt_frame_id = gt["frame_id"].astype(np.int32).reshape(-1)
    gt_corner_id = gt["corner_id"].astype(np.int32).reshape(-1)
    gt_xyz_L = gt["XYZ_world_mm"].astype(np.float64).reshape(-1, 3)
    gt_uv_L = gt["uv_left_px"].astype(np.float64).reshape(-1, 2)
    gt_uv_R = gt["uv_right_px"].astype(np.float64).reshape(-1, 2)
    gt_by_frame: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for fid in np.unique(gt_frame_id).tolist():
        mask = gt_frame_id == int(fid)
        ids = gt_corner_id[mask].tolist()
        xyz = gt_xyz_L[mask]
        uvL = gt_uv_L[mask]
        uvR = gt_uv_R[mask]
        gt_by_frame[int(fid)] = {
            int(i): {"XYZ_L": xyz[k], "uvL": uvL[k], "uvR": uvR[k]} for k, i in enumerate(ids)
        }

    sim = meta.get("sim_params", {})
    baseline_mm = float(sim["baseline_mm"])
    T_L_to_R = np.array([-baseline_mm, 0.0, 0.0], dtype=np.float64)

    # Oracle pinhole model from meta (for rays and reprojection).
    f_um = float(sim["f_um"])
    dist_model = str(sim.get("distortion_model", "none"))
    dist_L = sim.get("distortion_left", {}) if dist_model == "brown" else {}
    dist_R = sim.get("distortion_right", {}) if dist_model == "brown" else {}

    def rayfield2d_predict(
        marker_ids: np.ndarray,
        marker_corners: list[np.ndarray],
        target_ids: np.ndarray,
    ) -> dict[int, np.ndarray] | None:
        obj_pts: list[np.ndarray] = []
        img_pts: list[np.ndarray] = []
        for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
            o = id_to_obj2.get(int(mid))
            if o is None or mc.shape != (4, 2) or o.shape != (4, 2):
                continue
            obj_pts.append(o)
            img_pts.append(mc)
        if not obj_pts:
            return None
        obj_xy = np.concatenate(obj_pts, axis=0)
        img_uv = np.concatenate(img_pts, axis=0)

        target_ids = np.asarray(target_ids, dtype=np.int32).reshape(-1)
        if target_ids.size == 0:
            return {}
        chess2 = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
        target_xy = chess2[target_ids]
        pred = _predict_points_rayfield_tps_robust(
            obj_xy,
            img_uv,
            target_xy,
            lam=float(args.tps_lam),
            huber_c=float(args.tps_huber),
            iters=int(args.tps_iters),
        )
        return _dict_from_ids_xy(target_ids, pred)

    # Collect correspondences from images.
    methods_2d = ["raw", "rayfield_tps_robust"]
    per_method: dict[str, dict[str, Any]] = {}

    for m2d in methods_2d:
        obs_uv_L: list[np.ndarray] = []
        obs_uv_R: list[np.ndarray] = []
        gt_uvL: list[np.ndarray] = []
        gt_uvR: list[np.ndarray] = []
        gt_XYZ_L: list[np.ndarray] = []

        used_frames: set[int] = set()

        for fr in frames:
            fid = int(fr["frame_id"])
            gt_frame = gt_by_frame.get(fid)
            if gt_frame is None:
                continue

            pts_by_side: dict[Side, dict[int, np.ndarray]] = {}
            for side in ("left", "right"):
                img_path = scene_dir / side / str(fr[side])
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                det = detect_view(
                    cv2, aruco, dictionary, board, detector_params, aruco_detector, charuco_detector, img
                )
                if det is None:
                    continue

                if m2d == "raw":
                    pts_by_side[side] = _dict_from_ids_xy(det.charuco_ids, det.charuco_xy)
                else:
                    pred = rayfield2d_predict(det.marker_ids, det.marker_corners, det.charuco_ids)
                    if pred is None:
                        continue
                    pts_by_side[side] = pred

            if "left" not in pts_by_side or "right" not in pts_by_side:
                continue

            # Keep only ids present in GT for this frame on both sides.
            ids = sorted(set(pts_by_side["left"]).intersection(pts_by_side["right"]).intersection(gt_frame))
            if len(ids) < 8:
                continue

            used_frames.add(fid)
            uvL = _stack_for_ids(ids, pts_by_side["left"])
            uvR = _stack_for_ids(ids, pts_by_side["right"])
            obs_uv_L.append(uvL)
            obs_uv_R.append(uvR)
            gt_uvL.append(np.stack([gt_frame[i]["uvL"] for i in ids], axis=0))
            gt_uvR.append(np.stack([gt_frame[i]["uvR"] for i in ids], axis=0))
            gt_XYZ_L.append(np.stack([gt_frame[i]["XYZ_L"] for i in ids], axis=0))

        if not obs_uv_L:
            per_method[m2d] = {"n_frames": 0, "n_points": 0, "error": "no detections"}
            continue

        uvL_obs = np.concatenate(obs_uv_L, axis=0)
        uvR_obs = np.concatenate(obs_uv_R, axis=0)
        uvL_gt = np.concatenate(gt_uvL, axis=0)
        uvR_gt = np.concatenate(gt_uvR, axis=0)
        XYZ_L = np.concatenate(gt_XYZ_L, axis=0)
        XYZ_R = XYZ_L + T_L_to_R[None, :]

        # 2D error vs GT (px)
        err2d_L = np.linalg.norm(uvL_obs - uvL_gt, axis=-1)
        err2d_R = np.linalg.norm(uvR_obs - uvR_gt, axis=-1)

        # Pinhole oracle reconstruction from observed pixels (uses true f/dist).
        dL_pin = _pinhole_rays_from_pixels(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, uv_px=uvL_obs)
        dR_pin = _pinhole_rays_from_pixels(view_meta=meta["stereo"]["right"], f_um=f_um, brown=dist_R, uv_px=uvR_obs)
        C_L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        C_R_in_L = np.array([baseline_mm, 0.0, 0.0], dtype=np.float64)
        from stereocomplex.core.geometry import triangulate_midpoint  # noqa: PLC0415

        XYZ_hat_pin, skew_pin = triangulate_midpoint(C_L, dL_pin, C_R_in_L, dR_pin)
        err3d_pin = np.linalg.norm(XYZ_hat_pin - XYZ_L, axis=-1)

        # Reprojection error (px) vs GT pixels, using the oracle forward model.
        uv_hat_L_pin = _project_brown(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, XYZ_cam_mm=XYZ_hat_pin)
        uv_hat_R_pin = _project_brown(
            view_meta=meta["stereo"]["right"], f_um=f_um, brown=dist_R, XYZ_cam_mm=XYZ_hat_pin + T_L_to_R[None, :]
        )
        err_repr_L_pin = np.linalg.norm(uv_hat_L_pin - uvL_gt, axis=-1)
        err_repr_R_pin = np.linalg.norm(uv_hat_R_pin - uvR_gt, axis=-1)

        # Central ray-field 3D: fit on (observed pixels, GT 3D) then triangulate.
        from stereocomplex.core.model_compact.central_rayfield import CentralRayFieldZernike  # noqa: PLC0415

        w = int(meta["stereo"]["left"]["image"]["width_px"])
        h = int(meta["stereo"]["left"]["image"]["height_px"])
        u0, v0, radius = CentralRayFieldZernike.default_disk(w, h)

        rf_L, rf_fit_L = CentralRayFieldZernike.fit_from_gt(
            u_px=uvL_obs[:, 0],
            v_px=uvL_obs[:, 1],
            XYZ_cam_mm=XYZ_L,
            nmax=int(args.nmax),
            u0_px=u0,
            v0_px=v0,
            radius_px=radius,
            lam=float(args.lam3d),
            C_mm=C_L,
        )
        rf_R, rf_fit_R = CentralRayFieldZernike.fit_from_gt(
            u_px=uvR_obs[:, 0],
            v_px=uvR_obs[:, 1],
            XYZ_cam_mm=XYZ_R,
            nmax=int(args.nmax),
            u0_px=u0,
            v0_px=v0,
            radius_px=radius,
            lam=float(args.lam3d),
            C_mm=C_L,
        )
        dL_rf = rf_L.ray_directions_cam(uvL_obs[:, 0], uvL_obs[:, 1])
        dR_rf = rf_R.ray_directions_cam(uvR_obs[:, 0], uvR_obs[:, 1])
        XYZ_hat_rf, skew_rf = triangulate_midpoint(C_L, dL_rf, C_R_in_L, dR_rf)
        err3d_rf = np.linalg.norm(XYZ_hat_rf - XYZ_L, axis=-1)

        uv_hat_L_rf = _project_brown(view_meta=meta["stereo"]["left"], f_um=f_um, brown=dist_L, XYZ_cam_mm=XYZ_hat_rf)
        uv_hat_R_rf = _project_brown(
            view_meta=meta["stereo"]["right"], f_um=f_um, brown=dist_R, XYZ_cam_mm=XYZ_hat_rf + T_L_to_R[None, :]
        )
        err_repr_L_rf = np.linalg.norm(uv_hat_L_rf - uvL_gt, axis=-1)
        err_repr_R_rf = np.linalg.norm(uv_hat_R_rf - uvR_gt, axis=-1)

        Z = XYZ_L[:, 2]
        Z_mean = float(np.mean(Z))
        rel3d_pin = 100.0 * err3d_pin / (Z_mean + 1e-12)
        rel3d_rf = 100.0 * err3d_rf / (Z_mean + 1e-12)

        per_method[m2d] = {
            "n_frames": int(len(used_frames)),
            "n_points": int(XYZ_L.shape[0]),
            "error2d_left_px": _stats(err2d_L),
            "error2d_right_px": _stats(err2d_R),
            "depth_mm": {"mean": Z_mean, "p50": float(np.quantile(Z, 0.50))},
            "pinhole_oracle": {
                "triangulation_error_mm": _stats(err3d_pin),
                "triangulation_error_rel_depth_percent": _stats(rel3d_pin),
                "ray_skew_mm": _stats(skew_pin),
                "reprojection_error_left_px": _stats(err_repr_L_pin),
                "reprojection_error_right_px": _stats(err_repr_R_pin),
            },
            "central_zernike_rayfield": {
                "nmax": int(args.nmax),
                "lam3d": float(args.lam3d),
                "fit_left": rf_fit_L,
                "fit_right": rf_fit_R,
                "triangulation_error_mm": _stats(err3d_rf),
                "triangulation_error_rel_depth_percent": _stats(rel3d_rf),
                "ray_skew_mm": _stats(skew_rf),
                "reprojection_error_left_px": _stats(err_repr_L_rf),
                "reprojection_error_right_px": _stats(err_repr_R_rf),
            },
        }

    out = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": str(args.split),
        "scene": str(args.scene),
        "tps": {"lam": float(args.tps_lam), "huber_c": float(args.tps_huber), "iters": int(args.tps_iters)},
        "rayfield3d": {"nmax": int(args.nmax), "lam3d": float(args.lam3d)},
        "baseline_mm": baseline_mm,
        "methods_2d": per_method,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
