"""
Sanity check for the virtual rectification maps (ray-field backend).

This uses a synthetic central pinhole as the ray model, builds virtual
rectification maps, and verifies that rectified epipolar lines are
effectively horizontal (small vertical disparity) when sampling the
maps on a grid.
"""
from __future__ import annotations

import numpy as np

from stereocomplex.ray3d.rayfield_rectify import (
    RectifyParams,
    _build_rect_axes,  # type: ignore
    build_virtual_rectify_maps,
)


class PinholeRayModel:
    """Minimal pinhole ray model exposing dir(u, v) -> unit direction."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        # Expose width/height attributes expected by the rectifier for coarse grids.
        self.width = int(cx * 2)
        self.height = int(cy * 2)

    def dir(self, u: float, v: float) -> np.ndarray:
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d = np.array([x, y, 1.0], dtype=np.float64)
        return d / np.linalg.norm(d)


def main():
    # Synthetic pinhole rig
    width, height = 160, 120
    fx = fy = 300.0
    cx = width / 2.0
    cy = height / 2.0
    ray_L = PinholeRayModel(fx, fy, cx, cy)
    ray_R = PinholeRayModel(fx, fy, cx, cy)

    # Baseline along +x in left frame
    R_lr = np.eye(3, dtype=np.float64)
    t_lr = np.array([0.1, 0.0, 0.0], dtype=np.float64)  # 10 cm

    params = RectifyParams(width=width, height=height, coarse_step=12)
    mapx_L, mapy_L, mapx_R, mapy_R, R_rect = build_virtual_rectify_maps(
        ray_L, ray_R, R_lr, t_lr, params
    )

    # Sample a grid in the rectified image and check vertical disparity
    ys = np.linspace(10, height - 10, 40)
    xs = np.linspace(10, width - 10, 60)
    y_diffs = []
    invalid = 0

    for y in ys:
        for x in xs:
            uL = mapx_L[int(y), int(x)]
            vL = mapy_L[int(y), int(x)]
            uR = mapx_R[int(y), int(x)]
            vR = mapy_R[int(y), int(x)]
            if uL < 0 or vL < 0 or uR < 0 or vR < 0:
                invalid += 1
                continue
            dL = ray_L.dir(float(uL), float(vL))
            dR = ray_R.dir(float(uR), float(vR))
            dL_rect = R_rect @ dL
            dR_rect = R_rect @ dR
            yL_rect = dL_rect[1] / dL_rect[2]
            yR_rect = dR_rect[1] / dR_rect[2]
            y_diffs.append(abs(yL_rect - yR_rect))

    y_diffs = np.array(y_diffs)
    print(f"Sample count (valid): {len(y_diffs)}, invalid: {invalid}")
    if len(y_diffs) == 0:
        print("No valid samples; check configuration.")
        return
    print(
        "Vertical disparity stats in rectified space (dimensionless slopes): "
        f"median={np.median(y_diffs):.3e}, "
        f"P95={np.percentile(y_diffs, 95):.3e}, "
        f"max={np.max(y_diffs):.3e}"
    )

    # A slope of 1e-3 corresponds roughly to ~0.6 px at fx=600
    px_equiv = np.percentile(y_diffs, 95) * fx
    print(f"P95 vertical disparity equivalent ≈ {px_equiv:.3f} px at fx={fx}")


if __name__ == "__main__":
    main()
