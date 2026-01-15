from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))

from stereocomplex.core.distortion import brown_from_dict
from stereocomplex.core.geometry import (
    PinholeCamera,
    pixel_to_sensor_um,
    sensor_um_to_pixel,
    triangulate_midpoint,
)
from stereocomplex.core.model_compact.central_rayfield import CentralRayFieldZernike
from stereocomplex.meta import parse_view_meta


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _err_stats(err: np.ndarray) -> dict[str, float]:
    err = np.asarray(err, dtype=np.float64)
    return {
        "rms": _rms(err),
        "p50": float(np.quantile(err, 0.50)),
        "p95": float(np.quantile(err, 0.95)),
        "max": float(np.max(err)),
    }


def _project_brown(
    view_meta: dict,
    *,
    f_um: float,
    dist: dict,
    XYZ_cam_mm: np.ndarray,
) -> np.ndarray:
    """
    Forward project 3D points expressed in camera coordinates to distorted pixels.
    """
    view = parse_view_meta(view_meta)
    XYZ_cam_mm = np.asarray(XYZ_cam_mm, dtype=np.float64)
    X = XYZ_cam_mm[:, 0]
    Y = XYZ_cam_mm[:, 1]
    Z = XYZ_cam_mm[:, 2]
    good = np.isfinite(Z) & (np.abs(Z) > 1e-12)

    uv = np.full((XYZ_cam_mm.shape[0], 2), np.nan, dtype=np.float64)
    x = X[good] / Z[good]
    y = Y[good] / Z[good]

    brown = brown_from_dict(dist)
    xd, yd = brown.distort(x, y)
    x_um = xd * float(f_um)
    y_um = yd * float(f_um)
    u_px, v_px = sensor_um_to_pixel(view, x_um, y_um)
    uv[good, 0] = u_px
    uv[good, 1] = v_px
    return uv


def _pixel_error(uv_hat: np.ndarray, uv_gt: np.ndarray) -> np.ndarray:
    uv_hat = np.asarray(uv_hat, dtype=np.float64)
    uv_gt = np.asarray(uv_gt, dtype=np.float64)
    d = uv_hat - uv_gt
    ok = np.isfinite(d).all(axis=-1)
    d = d[ok]
    return np.linalg.norm(d, axis=-1)


def pinhole_rays_from_gt_uv(
    view_meta: dict,
    *,
    f_um: float,
    dist: dict,
    uv_px: np.ndarray,
) -> np.ndarray:
    """
    Convert distorted pixels (uv) -> undistorted ray directions using the known pinhole+Brown model.
    """
    view = parse_view_meta(view_meta)
    x_um, y_um = pixel_to_sensor_um(view, uv_px[:, 0], uv_px[:, 1])

    # normalized distorted coords (unitless): xd = x/f, yd = y/f
    xd = x_um / float(f_um)
    yd = y_um / float(f_um)

    brown = brown_from_dict(dist)
    x, y = brown.undistort(xd, yd, iterations=12)
    return PinholeCamera(f_um=float(f_um)).ray_directions_cam_from_norm(x, y)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare pinhole (oracle) vs central Zernike ray-field on GT correspondences."
    )
    ap.add_argument("--scene", type=str, required=True, help="Path to scene directory (meta.json + gt_*.npz).")
    ap.add_argument("--gt", type=str, default="gt_points.npz", help="GT NPZ filename in the scene dir.")
    ap.add_argument("--nmax", type=int, default=12, help="Max Zernike radial order.")
    ap.add_argument("--lam", type=float, default=1e-6, help="Ridge regularization lambda (ray-field).")
    args = ap.parse_args()

    scene_dir = Path(args.scene)
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    sim_params = meta["sim_params"]
    baseline_mm = float(sim_params["baseline_mm"])

    gt = np.load(scene_dir / args.gt)
    XYZ_L = gt["XYZ_world_mm"].astype(np.float64)
    uv_L = gt["uv_left_px"].astype(np.float64)
    uv_R = gt["uv_right_px"].astype(np.float64)

    # Dataset convention: right camera is translated by +baseline along x (in left camera frame).
    C_L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    C_R_in_L = np.array([baseline_mm, 0.0, 0.0], dtype=np.float64)
    T_L_to_R = np.array([-baseline_mm, 0.0, 0.0], dtype=np.float64)
    XYZ_R = XYZ_L + T_L_to_R[None, :]

    # --- Oracle pinhole baseline (known simulation parameters)
    dL_pinhole = pinhole_rays_from_gt_uv(
        meta["stereo"]["left"],
        f_um=float(sim_params["f_um"]),
        dist=sim_params["distortion_left"],
        uv_px=uv_L,
    )
    dR_pinhole = pinhole_rays_from_gt_uv(
        meta["stereo"]["right"],
        f_um=float(sim_params["f_um"]),
        dist=sim_params["distortion_right"],
        uv_px=uv_R,
    )
    XYZ_hat_pinhole, skew_pinhole = triangulate_midpoint(C_L, dL_pinhole, C_R_in_L, dR_pinhole)
    err_pinhole = np.linalg.norm(XYZ_hat_pinhole - XYZ_L, axis=-1)

    keep = np.isfinite(err_pinhole) & np.isfinite(skew_pinhole)
    err_pinhole = err_pinhole[keep]
    skew_pinhole = skew_pinhole[keep]

    uv_hat_L_pinhole = _project_brown(
        meta["stereo"]["left"], f_um=float(sim_params["f_um"]), dist=sim_params["distortion_left"], XYZ_cam_mm=XYZ_hat_pinhole
    )
    uv_hat_R_pinhole = _project_brown(
        meta["stereo"]["right"],
        f_um=float(sim_params["f_um"]),
        dist=sim_params["distortion_right"],
        XYZ_cam_mm=XYZ_hat_pinhole + T_L_to_R[None, :],
    )
    pxerr_L_pinhole = _pixel_error(uv_hat_L_pinhole, uv_L)
    pxerr_R_pinhole = _pixel_error(uv_hat_R_pinhole, uv_R)

    # --- Central Zernike ray-field fit on GT
    w = int(meta["stereo"]["left"]["image"]["width_px"])
    h = int(meta["stereo"]["left"]["image"]["height_px"])
    u0, v0, radius = CentralRayFieldZernike.default_disk(w, h)

    rf_L, rf_stats_L = CentralRayFieldZernike.fit_from_gt(
        u_px=uv_L[:, 0],
        v_px=uv_L[:, 1],
        XYZ_cam_mm=XYZ_L,
        nmax=args.nmax,
        u0_px=u0,
        v0_px=v0,
        radius_px=radius,
        lam=args.lam,
        C_mm=C_L,
    )
    rf_R, rf_stats_R = CentralRayFieldZernike.fit_from_gt(
        u_px=uv_R[:, 0],
        v_px=uv_R[:, 1],
        XYZ_cam_mm=XYZ_R,
        nmax=args.nmax,
        u0_px=u0,
        v0_px=v0,
        radius_px=radius,
        lam=args.lam,
        C_mm=C_L,
    )
    dL_rf = rf_L.ray_directions_cam(uv_L[:, 0], uv_L[:, 1])
    dR_rf = rf_R.ray_directions_cam(uv_R[:, 0], uv_R[:, 1])
    XYZ_hat_rf, skew_rf = triangulate_midpoint(C_L, dL_rf, C_R_in_L, dR_rf)
    err_rf = np.linalg.norm(XYZ_hat_rf - XYZ_L, axis=-1)

    keep = np.isfinite(err_rf) & np.isfinite(skew_rf)
    err_rf = err_rf[keep]
    skew_rf = skew_rf[keep]

    uv_hat_L_rf = _project_brown(
        meta["stereo"]["left"], f_um=float(sim_params["f_um"]), dist=sim_params["distortion_left"], XYZ_cam_mm=XYZ_hat_rf
    )
    uv_hat_R_rf = _project_brown(
        meta["stereo"]["right"],
        f_um=float(sim_params["f_um"]),
        dist=sim_params["distortion_right"],
        XYZ_cam_mm=XYZ_hat_rf + T_L_to_R[None, :],
    )
    pxerr_L_rf = _pixel_error(uv_hat_L_rf, uv_L)
    pxerr_R_rf = _pixel_error(uv_hat_R_rf, uv_R)

    Z = XYZ_L[:, 2]
    depth_stats = {
        "p50_mm": float(np.quantile(Z, 0.50)),
        "mean_mm": float(np.mean(Z)),
    }
    relz_pinhole = 100.0 * err_pinhole / float(depth_stats["mean_mm"])
    relz_rf = 100.0 * err_rf / float(depth_stats["mean_mm"])

    out = {
        "scene": str(scene_dir),
        "gt": args.gt,
        "baseline_mm": baseline_mm,
        "n_used": int(min(err_pinhole.size, err_rf.size)),
        "depth_mm": depth_stats,
        "pinhole_oracle": {
            "triangulation_error_mm": _err_stats(err_pinhole),
            "triangulation_error_rel_depth_percent": _err_stats(relz_pinhole),
            "ray_skew_mm": _err_stats(skew_pinhole),
            "reprojection_error_left_px": _err_stats(pxerr_L_pinhole),
            "reprojection_error_right_px": _err_stats(pxerr_R_pinhole),
        },
        "central_zernike_rayfield": {
            "nmax": int(args.nmax),
            "lam": float(args.lam),
            "fit_left": rf_stats_L,
            "fit_right": rf_stats_R,
            "triangulation_error_mm": _err_stats(err_rf),
            "triangulation_error_rel_depth_percent": _err_stats(relz_rf),
            "ray_skew_mm": _err_stats(skew_rf),
            "reprojection_error_left_px": _err_stats(pxerr_L_rf),
            "reprojection_error_right_px": _err_stats(pxerr_R_rf),
        },
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
