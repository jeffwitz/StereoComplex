from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.core.model_compact.zernike import ZernikeMode, zernike_design_matrix


def _ridge_solve(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve (A^T A + lam I)c = A^T y with lam>=0.
    """
    A = np.asarray(A, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if A.ndim != 2:
        raise ValueError("A must be 2D")
    if A.shape[0] != y.shape[0]:
        raise ValueError("A and y row counts must match")
    if lam < 0:
        raise ValueError("lam must be >= 0")
    ATA = A.T @ A
    if lam > 0:
        ATA = ATA + lam * np.eye(ATA.shape[0], dtype=np.float64)
    ATy = A.T @ y
    return np.linalg.solve(ATA, ATy)


@dataclass(frozen=True)
class CentralRayFieldZernike:
    """
    Central ray-field: a pixel defines a ray (origin C, direction d(u,v)).

    We model normalized camera coordinates (x(u,v), y(u,v)) with a Zernike basis
    over a unit disk in pixel coordinates:

      x(u,v) = sum_k a_k Z_k(u,v)
      y(u,v) = sum_k b_k Z_k(u,v)
      d = normalize([x, y, 1])

    The origin is constant: o(u,v) = C.
    """

    nmax: int
    u0_px: float
    v0_px: float
    radius_px: float
    coeffs_x: np.ndarray  # (K,)
    coeffs_y: np.ndarray  # (K,)
    modes: tuple[ZernikeMode, ...]
    C_mm: np.ndarray  # (3,)

    @staticmethod
    def default_disk(width_px: int, height_px: int) -> tuple[float, float, float]:
        u0 = (width_px - 1) / 2.0
        v0 = (height_px - 1) / 2.0
        # Circumscribed circle (covers the full image rectangle).
        r = 0.5 * float(np.hypot(width_px - 1, height_px - 1))
        return u0, v0, r

    def _eval_xy(self, u_px: np.ndarray, v_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A, mask, _modes = zernike_design_matrix(
            u_px, v_px, nmax=self.nmax, u0_px=self.u0_px, v0_px=self.v0_px, radius_px=self.radius_px
        )
        x = np.full((np.asarray(u_px).shape[0],), np.nan, dtype=np.float64)
        y = np.full((np.asarray(v_px).shape[0],), np.nan, dtype=np.float64)
        x[mask] = A @ self.coeffs_x
        y[mask] = A @ self.coeffs_y
        return x, y

    def ray_directions_cam(self, u_px: np.ndarray, v_px: np.ndarray) -> np.ndarray:
        u_px = np.asarray(u_px, dtype=np.float64).reshape(-1)
        v_px = np.asarray(v_px, dtype=np.float64).reshape(-1)
        x, y = self._eval_xy(u_px, v_px)
        d = np.stack([x, y, np.ones_like(x)], axis=-1)
        norms = np.linalg.norm(d, axis=-1, keepdims=True)
        return d / norms

    def ray_origins_cam_mm(self, n: int) -> np.ndarray:
        return np.repeat(self.C_mm.reshape(1, 3), n, axis=0)

    @classmethod
    def fit_from_gt(
        cls,
        *,
        u_px: np.ndarray,
        v_px: np.ndarray,
        XYZ_cam_mm: np.ndarray,
        nmax: int,
        u0_px: float,
        v0_px: float,
        radius_px: float,
        lam: float = 1e-6,
        C_mm: np.ndarray | None = None,
    ) -> tuple["CentralRayFieldZernike", dict[str, float]]:
        """
        Fit (x(u,v), y(u,v)) from ground-truth 3D points expressed in camera frame.

        Each observation satisfies: XYZ_cam_mm lies on the ray from the origin.
        Targets are x = X/Z and y = Y/Z.
        """
        XYZ_cam_mm = np.asarray(XYZ_cam_mm, dtype=np.float64)
        if XYZ_cam_mm.ndim != 2 or XYZ_cam_mm.shape[1] != 3:
            raise ValueError("XYZ_cam_mm must have shape (N,3)")
        u_px = np.asarray(u_px, dtype=np.float64).reshape(-1)
        v_px = np.asarray(v_px, dtype=np.float64).reshape(-1)
        if u_px.shape[0] != XYZ_cam_mm.shape[0] or v_px.shape[0] != XYZ_cam_mm.shape[0]:
            raise ValueError("u_px/v_px and XYZ_cam_mm must have the same length")

        Z = XYZ_cam_mm[:, 2]
        good = np.isfinite(Z) & (np.abs(Z) > 1e-12)
        if np.count_nonzero(good) < 10:
            raise ValueError("not enough valid GT points to fit")

        x_gt = XYZ_cam_mm[good, 0] / Z[good]
        y_gt = XYZ_cam_mm[good, 1] / Z[good]
        A, mask, modes = zernike_design_matrix(
            u_px[good], v_px[good], nmax=nmax, u0_px=u0_px, v0_px=v0_px, radius_px=radius_px
        )
        x_gt = x_gt[mask]
        y_gt = y_gt[mask]
        if x_gt.size < 10:
            raise ValueError("not enough GT points inside the unit disk to fit")

        coeffs_x = _ridge_solve(A, x_gt, lam)
        coeffs_y = _ridge_solve(A, y_gt, lam)

        if C_mm is None:
            C_mm = np.zeros((3,), dtype=np.float64)
        else:
            C_mm = np.asarray(C_mm, dtype=np.float64).reshape(3)

        model = cls(
            nmax=nmax,
            u0_px=float(u0_px),
            v0_px=float(v0_px),
            radius_px=float(radius_px),
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
            modes=tuple(modes),
            C_mm=C_mm,
        )

        # Simple fit diagnostics: point-to-ray distance on training data.
        d = model.ray_directions_cam(u_px[good][mask], v_px[good][mask])
        X = XYZ_cam_mm[good][mask]
        proj = np.sum(X * d, axis=-1, keepdims=True) * d
        dist = np.linalg.norm(X - proj, axis=-1)
        stats = {
            "train_point_to_ray_rms_mm": float(np.sqrt(np.mean(dist**2))),
            "train_point_to_ray_p95_mm": float(np.quantile(dist, 0.95)),
            "n_obs": float(X.shape[0]),
            "n_modes": float(len(modes)),
        }
        return model, stats
