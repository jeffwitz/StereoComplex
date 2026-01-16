from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.core.geometry import triangulate_midpoint
from stereocomplex.core.model_compact.central_rayfield import CentralRayFieldZernike


@dataclass(frozen=True)
class StereoCentralRayFieldModel:
    """
    Stereo model for reconstruction with a *central* ray-field per camera.

    Convention:
    - left camera frame is the reference frame
    - right camera pose is expressed as X_R = R_RL X_L + t_RL
      therefore the right camera center in left frame is C_R = -R_RL^T t_RL
    """

    image_width_px: int
    image_height_px: int
    left: CentralRayFieldZernike
    right: CentralRayFieldZernike
    R_RL: np.ndarray  # (3,3)
    t_RL: np.ndarray  # (3,)

    @property
    def C_L_mm(self) -> np.ndarray:
        return np.zeros((3,), dtype=np.float64)

    @property
    def C_R_in_L_mm(self) -> np.ndarray:
        return (-self.R_RL.T @ self.t_RL.reshape(3)).astype(np.float64)

    @classmethod
    def from_coeffs(
        cls,
        *,
        image_width_px: int,
        image_height_px: int,
        nmax: int,
        u0_px: float,
        v0_px: float,
        radius_px: float,
        coeffs_left_x: np.ndarray,
        coeffs_left_y: np.ndarray,
        coeffs_right_x: np.ndarray,
        coeffs_right_y: np.ndarray,
        R_RL: np.ndarray,
        t_RL: np.ndarray,
        C_L_mm: np.ndarray | None = None,
    ) -> "StereoCentralRayFieldModel":
        if C_L_mm is None:
            C_L_mm = np.zeros((3,), dtype=np.float64)
        C_L_mm = np.asarray(C_L_mm, dtype=np.float64).reshape(3)
        R_RL = np.asarray(R_RL, dtype=np.float64).reshape(3, 3)
        t_RL = np.asarray(t_RL, dtype=np.float64).reshape(3)

        left = CentralRayFieldZernike(
            nmax=int(nmax),
            u0_px=float(u0_px),
            v0_px=float(v0_px),
            radius_px=float(radius_px),
            coeffs_x=np.asarray(coeffs_left_x, dtype=np.float64).reshape(-1),
            coeffs_y=np.asarray(coeffs_left_y, dtype=np.float64).reshape(-1),
            modes=tuple(),
            C_mm=C_L_mm,
        )
        right = CentralRayFieldZernike(
            nmax=int(nmax),
            u0_px=float(u0_px),
            v0_px=float(v0_px),
            radius_px=float(radius_px),
            coeffs_x=np.asarray(coeffs_right_x, dtype=np.float64).reshape(-1),
            coeffs_y=np.asarray(coeffs_right_y, dtype=np.float64).reshape(-1),
            modes=tuple(),
            C_mm=C_L_mm,
        )
        return cls(
            image_width_px=int(image_width_px),
            image_height_px=int(image_height_px),
            left=left,
            right=right,
            R_RL=R_RL,
            t_RL=t_RL,
        )

    def triangulate(self, uv_left_px: np.ndarray, uv_right_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Triangulate corresponding pixels into 3D points in the left camera frame.

        Returns (XYZ_L_mm, skew_mm).
        """
        uv_left_px = np.asarray(uv_left_px, dtype=np.float64).reshape(-1, 2)
        uv_right_px = np.asarray(uv_right_px, dtype=np.float64).reshape(-1, 2)
        if uv_left_px.shape[0] != uv_right_px.shape[0]:
            raise ValueError("uv_left_px and uv_right_px must have the same length")

        dL = self.left.ray_directions_cam(uv_left_px[:, 0], uv_left_px[:, 1])
        dR = self.right.ray_directions_cam(uv_right_px[:, 0], uv_right_px[:, 1])
        dR_in_L = (self.R_RL.T @ dR.T).T
        XYZ, skew = triangulate_midpoint(self.C_L_mm, dL, self.C_R_in_L_mm, dR_in_L)
        return XYZ, skew

    def ray_direction_maps(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute per-pixel ray direction maps for left and right cameras.

        Returns (dL_map, dR_map) with shape (H,W,3).
        """
        h = int(self.image_height_px)
        w = int(self.image_width_px)
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
        u = xx.reshape(-1)
        v = yy.reshape(-1)
        dL = self.left.ray_directions_cam(u, v).reshape(h, w, 3)
        dR = self.right.ray_directions_cam(u, v).reshape(h, w, 3)
        return dL.astype(np.float32), dR.astype(np.float32)

