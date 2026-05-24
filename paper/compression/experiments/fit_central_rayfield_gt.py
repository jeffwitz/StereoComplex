from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "src")))

from stereocomplex.core.geometry import triangulate_midpoint
from stereocomplex.core.model_compact.central_rayfield import CentralRayFieldZernike


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit a central Zernike ray-field from GT and evaluate triangulation.")
    ap.add_argument("--scene", type=str, required=True, help="Path to scene directory (contains meta.json, gt_*.npz).")
    ap.add_argument("--gt", type=str, default="gt_charuco_corners.npz", help="GT NPZ filename in the scene dir.")
    ap.add_argument("--nmax", type=int, default=12, help="Max Zernike radial order.")
    ap.add_argument("--lam", type=float, default=1e-6, help="Ridge regularization lambda.")
    ap.add_argument("--disk", type=str, default="center", choices=["center"], help="Unit disk mapping (currently only center).")
    args = ap.parse_args()

    scene_dir = Path(args.scene)
    meta = json.loads((scene_dir / "meta.json").read_text())
    sim_params = meta.get("sim_params", {})
    baseline_mm = float(sim_params["baseline_mm"])

    w = int(meta["stereo"]["left"]["image"]["width_px"])
    h = int(meta["stereo"]["left"]["image"]["height_px"])
    u0, v0, radius = CentralRayFieldZernike.default_disk(w, h)

    gt = np.load(scene_dir / args.gt)
    XYZ_L = gt["XYZ_world_mm"].astype(np.float64)
    uv_L = gt["uv_left_px"].astype(np.float64)
    uv_R = gt["uv_right_px"].astype(np.float64)

    # Dataset convention: right camera is translated by +baseline along x in left coords,
    # i.e. X_R = X_L - baseline.
    T_L_to_R = np.array([-baseline_mm, 0.0, 0.0], dtype=np.float64)
    XYZ_R = XYZ_L + T_L_to_R[None, :]

    model_L, stats_L = CentralRayFieldZernike.fit_from_gt(
        u_px=uv_L[:, 0],
        v_px=uv_L[:, 1],
        XYZ_cam_mm=XYZ_L,
        nmax=args.nmax,
        u0_px=u0,
        v0_px=v0,
        radius_px=radius,
        lam=args.lam,
        C_mm=np.zeros((3,), dtype=np.float64),
    )
    model_R, stats_R = CentralRayFieldZernike.fit_from_gt(
        u_px=uv_R[:, 0],
        v_px=uv_R[:, 1],
        XYZ_cam_mm=XYZ_R,
        nmax=args.nmax,
        u0_px=u0,
        v0_px=v0,
        radius_px=radius,
        lam=args.lam,
        C_mm=np.zeros((3,), dtype=np.float64),
    )

    dL = model_L.ray_directions_cam(uv_L[:, 0], uv_L[:, 1])
    dR = model_R.ray_directions_cam(uv_R[:, 0], uv_R[:, 1])

    C_L = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    C_R_in_L = np.array([baseline_mm, 0.0, 0.0], dtype=np.float64)

    XYZ_hat, skew = triangulate_midpoint(C_L, dL, C_R_in_L, dR)
    err_mm = np.linalg.norm(XYZ_hat - XYZ_L, axis=-1)
    keep = np.isfinite(err_mm) & np.isfinite(skew)
    err_mm = err_mm[keep]
    skew = skew[keep]

    out = {
        "scene": str(scene_dir),
        "nmax": int(args.nmax),
        "lam": float(args.lam),
        "baseline_mm": baseline_mm,
        "fit_left": stats_L,
        "fit_right": stats_R,
        "triangulation": {
            "rms_mm": _rms(err_mm),
            "p50_mm": float(np.quantile(err_mm, 0.50)),
            "p95_mm": float(np.quantile(err_mm, 0.95)),
        },
        "ray_skew": {"rms_mm": _rms(skew), "p95_mm": float(np.quantile(skew, 0.95))},
        "n_used": int(err_mm.size),
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
