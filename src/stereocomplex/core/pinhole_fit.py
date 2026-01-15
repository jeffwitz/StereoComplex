from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from stereocomplex.core.distortion import BrownDistortion


@dataclass(frozen=True)
class BrownPinholeParams:
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def K(self) -> np.ndarray:
        return np.array(
            [[float(self.fx), 0.0, float(self.cx)], [0.0, float(self.fy), float(self.cy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def dist(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)

    def distortion(self) -> BrownDistortion:
        return BrownDistortion(k1=self.k1, k2=self.k2, p1=self.p1, p2=self.p2, k3=self.k3)


def project_brown_pinhole(
    params: BrownPinholeParams,
    XYZ_cam: np.ndarray,
) -> np.ndarray:
    XYZ_cam = np.asarray(XYZ_cam, dtype=np.float64).reshape(-1, 3)
    X = XYZ_cam[:, 0]
    Y = XYZ_cam[:, 1]
    Z = XYZ_cam[:, 2]
    uv = np.full((XYZ_cam.shape[0], 2), np.nan, dtype=np.float64)
    good = np.isfinite(Z) & (np.abs(Z) > 1e-12)
    if not np.any(good):
        return uv

    x = X[good] / Z[good]
    y = Y[good] / Z[good]
    dist = params.distortion()
    xd, yd = dist.distort(x, y)

    uv[good, 0] = params.fx * xd + params.cx
    uv[good, 1] = params.fy * yd + params.cy
    return uv


def project_brown_pinhole_with_rvec(
    params: BrownPinholeParams,
    XYZ_cam: np.ndarray,
    rvec: np.ndarray,
) -> np.ndarray:
    """
    Convenience wrapper for post-hoc fits that include a global rotation.
    """
    from scipy.spatial.transform import Rotation as Rot  # type: ignore

    XYZ_cam = np.asarray(XYZ_cam, dtype=np.float64).reshape(-1, 3)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    Rm = Rot.from_rotvec(rvec).as_matrix()
    return project_brown_pinhole(params, (Rm @ XYZ_cam.T).T)


def _linear_init_fx_cx(x: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    A = np.stack([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, u, rcond=None)
    fx, cx = float(sol[0]), float(sol[1])
    if not np.isfinite(fx) or abs(fx) < 1e-9:
        fx = float("nan")
    return fx, cx


def fit_brown_pinhole_from_camera_points(
    *,
    XYZ_cam: np.ndarray,
    uv_px: np.ndarray,
    image_size: tuple[int, int],
    init: BrownPinholeParams | None = None,
    fit_rotation: bool = False,
    loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"] = "huber",
    f_scale_px: float = 2.0,
    max_nfev: int = 2000,
    fit_distortion: bool = True,
) -> tuple[BrownPinholeParams, dict[str, float]]:
    """
    Fit a Brown pinhole model (fx,fy,cx,cy,k1,k2,p1,p2,k3) from known 3D points in the
    *camera frame* and their observed pixels.

    This is meant for post-hoc identification (e.g., after a ray-based 3D reconstruction).
    """
    from scipy.optimize import least_squares  # type: ignore

    w, h = int(image_size[0]), int(image_size[1])
    XYZ_cam = np.asarray(XYZ_cam, dtype=np.float64).reshape(-1, 3)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    if XYZ_cam.shape[0] != uv_px.shape[0] or XYZ_cam.shape[0] < 12:
        raise ValueError("need >= 12 correspondences and matching sizes")

    X = XYZ_cam[:, 0]
    Y = XYZ_cam[:, 1]
    Z = XYZ_cam[:, 2]
    good = np.isfinite(Z) & (Z > 1e-6) & np.all(np.isfinite(uv_px), axis=1)
    if int(np.sum(good)) < 12:
        raise ValueError("not enough valid points (Z>0 and finite)")
    XYZ_cam = XYZ_cam[good]
    uv_px = uv_px[good]
    X = XYZ_cam[:, 0]
    Y = XYZ_cam[:, 1]
    Z = XYZ_cam[:, 2]

    x = X / Z
    y = Y / Z
    u = uv_px[:, 0]
    v = uv_px[:, 1]

    if init is None:
        fx0, cx0 = _linear_init_fx_cx(x, u)
        fy0, cy0 = _linear_init_fx_cx(y, v)
        if not np.isfinite(fx0):
            fx0 = 1.5 * float(max(w, h))
        if not np.isfinite(fy0):
            fy0 = 1.5 * float(max(w, h))
        if not np.isfinite(cx0):
            cx0 = (w - 1) / 2.0
        if not np.isfinite(cy0):
            cy0 = (h - 1) / 2.0
        init = BrownPinholeParams(fx=fx0, fy=fy0, cx=cx0, cy=cy0)

    if fit_rotation:
        p0 = np.array(
            [0.0, 0.0, 0.0, init.fx, init.fy, init.cx, init.cy, init.k1, init.k2, init.p1, init.p2, init.k3],
            dtype=np.float64,
        )
    else:
        p0 = np.array(
            [init.fx, init.fy, init.cx, init.cy, init.k1, init.k2, init.p1, init.p2, init.k3], dtype=np.float64
        )

    # Conservative bounds: keep distortion moderate to avoid pathological fits.
    lb_base = np.array([10.0, 10.0, -0.5, -0.5, -1.0, -1.0, -0.1, -0.1, -1.0], dtype=np.float64)
    ub_base = np.array([10_000.0, 10_000.0, w - 0.5, h - 0.5, 1.0, 1.0, 0.1, 0.1, 1.0], dtype=np.float64)
    if fit_rotation:
        lb = np.concatenate([np.full((3,), -np.pi, dtype=np.float64), lb_base], axis=0)
        ub = np.concatenate([np.full((3,), np.pi, dtype=np.float64), ub_base], axis=0)
    else:
        lb = lb_base
        ub = ub_base
    if not fit_distortion:
        if fit_rotation:
            lb[3 + 4 :] = 0.0
            ub[3 + 4 :] = 0.0
        else:
            lb[4:] = 0.0
            ub[4:] = 0.0
    # Ensure the initial guess is inside bounds (required by SciPy).
    p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)

    def fun(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        if fit_rotation:
            from scipy.spatial.transform import Rotation as Rot  # type: ignore

            rvec = p[:3]
            q = p[3:]
            Rm = Rot.from_rotvec(rvec).as_matrix()
            Xp = (Rm @ XYZ_cam.T).T
            xx = Xp[:, 0] / Xp[:, 2]
            yy = Xp[:, 1] / Xp[:, 2]
        else:
            q = p
            xx = x
            yy = y

        fx, fy, cx, cy, k1, k2, p1, p2, k3 = (float(v) for v in q.tolist())
        dist = BrownDistortion(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
        xd, yd = dist.distort(xx, yy)
        u_hat = fx * xd + cx
        v_hat = fy * yd + cy
        r = np.stack([u_hat - u, v_hat - v], axis=1).reshape(-1)
        return r

    sol = least_squares(
        fun,
        p0,
        method="trf",
        loss=str(loss),
        f_scale=float(f_scale_px),
        max_nfev=int(max_nfev),
        bounds=(lb, ub),
    )

    if fit_rotation:
        rvec_opt = sol.x[:3].copy()
        q_opt = sol.x[3:]
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = (float(v) for v in q_opt.tolist())
    else:
        rvec_opt = None
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = (float(v) for v in sol.x.tolist())
    params = BrownPinholeParams(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2, k3=k3)
    diag = {"opt_cost": float(sol.cost), "opt_nfev": float(sol.nfev), "opt_success": float(bool(sol.success))}
    if rvec_opt is not None:
        diag["rvec"] = [float(x) for x in rvec_opt.tolist()]
    return params, diag


def distortion_displacement_metrics(
    *,
    K_gt: np.ndarray,
    dist_gt: np.ndarray,
    dist_est: np.ndarray,
    image_size: tuple[int, int],
    n_angles: int = 180,
    radii_fracs: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9),
) -> dict[str, float]:
    """
    Compare two Brown distortion models in pixel space (physically meaningful on a sensor).

    We sample ideal (undistorted) pixels on concentric circles, apply both distortion
    models, and compare distortion displacement vectors (in pixels).
    """
    w, h = image_size
    K_gt = np.asarray(K_gt, dtype=np.float64).reshape(3, 3)
    dist_gt = np.asarray(dist_gt, dtype=np.float64).reshape(-1)
    dist_est = np.asarray(dist_est, dtype=np.float64).reshape(-1)
    if dist_gt.size < 5 or dist_est.size < 5:
        raise ValueError("dist vectors must have at least 5 coefficients (k1,k2,p1,p2,k3)")

    cx = float(K_gt[0, 2])
    cy = float(K_gt[1, 2])
    fx = float(K_gt[0, 0])
    fy = float(K_gt[1, 1])

    rmax = max(1.0, min(cx, (w - 1) - cx, cy, (h - 1) - cy))
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False, dtype=np.float64)

    uv_list: list[np.ndarray] = []
    for rf in radii_fracs:
        r = float(rf) * rmax
        u = cx + r * np.cos(angles)
        v = cy + r * np.sin(angles)
        uv_list.append(np.stack([u, v], axis=1))
    uv = np.concatenate(uv_list, axis=0).astype(np.float64)  # (N,2) ideal pixels

    x = (uv[:, 0] - cx) / fx
    y = (uv[:, 1] - cy) / fy

    gt = BrownDistortion(k1=float(dist_gt[0]), k2=float(dist_gt[1]), p1=float(dist_gt[2]), p2=float(dist_gt[3]), k3=float(dist_gt[4]))
    est = BrownDistortion(k1=float(dist_est[0]), k2=float(dist_est[1]), p1=float(dist_est[2]), p2=float(dist_est[3]), k3=float(dist_est[4]))

    xd_gt, yd_gt = gt.distort(x, y)
    xd_est, yd_est = est.distort(x, y)

    uv_gt = np.stack([fx * xd_gt + cx, fy * yd_gt + cy], axis=1)
    uv_est = np.stack([fx * xd_est + cx, fy * yd_est + cy], axis=1)

    disp_gt = uv_gt - uv
    disp_est = uv_est - uv
    disp_err = disp_est - disp_gt

    gt_mag = np.linalg.norm(disp_gt, axis=1)
    err_mag = np.linalg.norm(disp_err, axis=1)

    gt_rms = float(np.sqrt(np.mean(gt_mag * gt_mag)))
    err_rms = float(np.sqrt(np.mean(err_mag * err_mag)))
    rel_rms_pct = float(100.0 * err_rms / (gt_rms + 1e-12))
    return {
        "gt_rms_px": gt_rms,
        "err_rms_px": err_rms,
        "err_rms_percent_of_gt": rel_rms_pct,
    }
