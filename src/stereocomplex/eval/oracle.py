from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from stereocomplex.core.distortion import BrownDistortion, brown_from_dict
from stereocomplex.core.geometry import PinholeCamera, pixel_to_sensor_um, sensor_um_to_pixel, triangulate_midpoint
from stereocomplex.meta import parse_view_meta


def eval_oracle_dataset(dataset_root: Path) -> None:
    dataset_root = dataset_root.resolve()
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            stats = eval_oracle_scene(scene_dir)
            print(f"{split}/{scene_dir.name}: {json.dumps(stats, indent=None, sort_keys=True)}")


def eval_oracle_scene(scene_dir: Path) -> dict[str, float]:
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    left_view = parse_view_meta(meta["stereo"]["left"])
    right_view = parse_view_meta(meta["stereo"]["right"])

    sim = meta.get("sim_params", {})
    if sim.get("camera_model") != "pinhole":
        raise ValueError(f"Oracle only supports camera_model=pinhole (got {sim.get('camera_model')})")
    f_um = float(sim["f_um"])
    baseline_mm = float(sim["baseline_mm"])
    distortion_model = str(sim.get("distortion_model", "none"))
    dist_left = _load_distortion(distortion_model, sim.get("distortion_left"))
    dist_right = _load_distortion(distortion_model, sim.get("distortion_right"))

    cam = PinholeCamera(f_um=f_um)
    oL = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    oR = np.array([baseline_mm, 0.0, 0.0], dtype=np.float64)

    gt = np.load(scene_dir / "gt_points.npz")
    xyz = gt["XYZ_world_mm"].astype(np.float64)
    uvL_gt = gt["uv_left_px"].astype(np.float64)
    uvR_gt = gt["uv_right_px"].astype(np.float64)

    # Reproject GT 3D -> pixels.
    uvL_pred = _project_points_pinhole(left_view, cam, dist_left, oL, xyz)
    uvR_pred = _project_points_pinhole(right_view, cam, dist_right, oR, xyz)

    eL = np.linalg.norm(uvL_pred - uvL_gt, axis=-1)
    eR = np.linalg.norm(uvR_pred - uvR_gt, axis=-1)

    # Triangulate from GT pixels -> 3D and compare to GT 3D.
    dL = _rays_from_pixels(left_view, cam, dist_left, uvL_gt)
    dR = _rays_from_pixels(right_view, cam, dist_right, uvR_gt)
    xyz_tri, ray_dist = triangulate_midpoint(oL, dL, oR, dR)
    e3d = np.linalg.norm(xyz_tri - xyz, axis=-1)

    return {
        "reproj_rms_left_px": float(np.sqrt(np.mean(eL**2))) if eL.size else float("nan"),
        "reproj_rms_right_px": float(np.sqrt(np.mean(eR**2))) if eR.size else float("nan"),
        "triang_rms_mm": float(np.sqrt(np.mean(e3d**2))) if e3d.size else float("nan"),
        "ray_dist_p95_mm": float(np.quantile(ray_dist, 0.95)) if ray_dist.size else float("nan"),
        "n_points": float(xyz.shape[0]),
    }


def _load_distortion(model: str, params: object) -> BrownDistortion:
    model = str(model)
    if model == "none":
        return BrownDistortion()
    if model != "brown":
        raise ValueError(f"Unsupported distortion_model: {model}")
    if isinstance(params, dict):
        return brown_from_dict(params)
    return BrownDistortion()


def _rays_from_pixels(view, cam: PinholeCamera, dist: BrownDistortion, uv_px: np.ndarray) -> np.ndarray:
    x_um, y_um = pixel_to_sensor_um(view, uv_px[:, 0], uv_px[:, 1])
    xd = x_um / float(cam.f_um)
    yd = y_um / float(cam.f_um)
    x, y = dist.undistort(xd, yd)
    return cam.ray_directions_cam_from_norm(x, y)


def _project_points_pinhole(
    view, cam: PinholeCamera, dist: BrownDistortion, origin_mm: np.ndarray, xyz_world_mm: np.ndarray
) -> np.ndarray:
    p_cam = xyz_world_mm - origin_mm[None, :]
    x_mm = p_cam[:, 0]
    y_mm = p_cam[:, 1]
    z_mm = p_cam[:, 2]
    x = x_mm / z_mm
    y = y_mm / z_mm
    xd, yd = dist.distort(x, y)
    x_um = xd * float(cam.f_um)
    y_um = yd * float(cam.f_um)
    u_px, v_px = sensor_um_to_pixel(view, x_um, y_um)
    return np.stack([u_px, v_px], axis=-1)
