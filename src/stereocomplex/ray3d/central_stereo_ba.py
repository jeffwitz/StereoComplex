from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from stereocomplex.core.model_compact.zernike import zernike_design_matrix
from stereocomplex.ray3d.central_ba import default_disk


@dataclass(frozen=True)
class StereoFrameObservations:
    """
    Observations for one stereo frame.

    - `uv_left_px`, `uv_right_px`: observed pixels (dataset pixel-center convention)
    - `P_board_mm`: corresponding 3D points in the board frame (mm)
    """

    uv_left_px: np.ndarray  # (N,2)
    uv_right_px: np.ndarray  # (N,2)
    P_board_mm: np.ndarray  # (N,3)


@dataclass(frozen=True)
class CentralStereoRayFieldBAResult:
    nmax: int
    u0_px: float
    v0_px: float
    radius_px: float
    coeffs_left_x: np.ndarray  # (K,)
    coeffs_left_y: np.ndarray  # (K,)
    coeffs_right_x: np.ndarray  # (K,)
    coeffs_right_y: np.ndarray  # (K,)
    rig_rvec: np.ndarray  # (3,)
    rig_tvec: np.ndarray  # (3,)
    rvecs: dict[int, np.ndarray]  # frame_id -> (3,) (board pose in left camera coords)
    tvecs: dict[int, np.ndarray]  # frame_id -> (3,)
    diagnostics: dict[str, float]


def _dir_from_coeffs(A: np.ndarray, coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> np.ndarray:
    x = A @ coeffs_x
    y = A @ coeffs_y
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def _point_to_ray_residual(P_cam: np.ndarray, d: np.ndarray) -> np.ndarray:
    proj = np.sum(P_cam * d, axis=-1, keepdims=True) * d
    return (P_cam - proj).reshape(-1)


def _pack_coeffs(
    coeffs_left_x: np.ndarray,
    coeffs_left_y: np.ndarray,
    coeffs_right_x: np.ndarray,
    coeffs_right_y: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [
            coeffs_left_x.reshape(-1),
            coeffs_left_y.reshape(-1),
            coeffs_right_x.reshape(-1),
            coeffs_right_y.reshape(-1),
        ],
        axis=0,
    )


def _unpack_coeffs(p: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if p.size != 4 * K:
        raise ValueError("invalid coeff vector size")
    a = p[0 * K : 1 * K].copy()
    b = p[1 * K : 2 * K].copy()
    c = p[2 * K : 3 * K].copy()
    d = p[3 * K : 4 * K].copy()
    return a, b, c, d


def fit_central_stereo_rayfield_ba(
    *,
    frames: dict[int, StereoFrameObservations],
    image_width_px: int,
    image_height_px: int,
    nmax: int,
    rvecs0: dict[int, np.ndarray],
    tvecs0: dict[int, np.ndarray],
    rig_rvec0: np.ndarray,
    rig_tvec0: np.ndarray,
    coeffs0_left_x: np.ndarray,
    coeffs0_left_y: np.ndarray,
    coeffs0_right_x: np.ndarray,
    coeffs0_right_y: np.ndarray,
    lam_coeff: float = 1e-3,
    lam_center: float = 1e-1,
    lam_jacobian: float = 1.0,
    loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"] = "huber",
    f_scale_mm: float = 1.0,
    max_nfev: int = 200,
) -> CentralStereoRayFieldBAResult:
    """
    Joint stereo bundle adjustment for a *central* ray-field in both cameras.

    Variables:
      - Zernike coefficients for left/right x,y
      - A single stereo rig transform (R_RL, t_RL) with P_R = R_RL P_L + t_RL
      - Per-frame board poses in left camera coordinates (R_i, t_i)

    Residuals (for each observed correspondence in each frame):
      r_L = (I - d_L d_L^T) P_L
      r_R = (I - d_R d_R^T) P_R

    where P_L = R_i P_board + t_i and P_R = R_RL P_L + t_RL.
    """
    if not frames:
        raise ValueError("frames is empty")
    if nmax < 0:
        raise ValueError("nmax must be >= 0")

    u0, v0, radius = default_disk(int(image_width_px), int(image_height_px))

    # Stable parameter ordering.
    fids = sorted(int(fid) for fid in frames.keys())
    for fid in fids:
        if fid not in rvecs0 or fid not in tvecs0:
            raise ValueError("missing initial pose for some frame")

    # Precompute design matrices A_L/A_R per frame.
    A_left_by_fid: dict[int, np.ndarray] = {}
    A_right_by_fid: dict[int, np.ndarray] = {}
    frames_clean: dict[int, StereoFrameObservations] = {}
    K = None
    for fid in fids:
        fr = frames[fid]
        uvL = np.asarray(fr.uv_left_px, dtype=np.float64).reshape(-1, 2)
        uvR = np.asarray(fr.uv_right_px, dtype=np.float64).reshape(-1, 2)
        P = np.asarray(fr.P_board_mm, dtype=np.float64).reshape(-1, 3)
        if uvL.shape[0] != uvR.shape[0] or uvL.shape[0] != P.shape[0]:
            raise ValueError("inconsistent per-frame observation sizes")

        A_L, maskL, _modes = zernike_design_matrix(
            uvL[:, 0], uvL[:, 1], nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius
        )
        A_R, maskR, _modes2 = zernike_design_matrix(
            uvR[:, 0], uvR[:, 1], nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius
        )
        mask = maskL & maskR
        if not np.all(mask):
            uvL = uvL[mask]
            uvR = uvR[mask]
            P = P[mask]
            A_L = A_L[mask]
            A_R = A_R[mask]

        frames_clean[fid] = StereoFrameObservations(uv_left_px=uvL, uv_right_px=uvR, P_board_mm=P)
        A_left_by_fid[fid] = A_L
        A_right_by_fid[fid] = A_R
        K = A_L.shape[1] if K is None else K
        if A_L.shape[1] != K or A_R.shape[1] != K:
            raise RuntimeError("inconsistent Zernike design matrix width")
    assert K is not None

    from scipy.optimize import least_squares  # type: ignore
    from scipy.spatial.transform import Rotation as R  # type: ignore

    coeff0 = _pack_coeffs(coeffs0_left_x, coeffs0_left_y, coeffs0_right_x, coeffs0_right_y)
    rig0 = np.concatenate([np.asarray(rig_rvec0, dtype=np.float64).reshape(3), np.asarray(rig_tvec0, dtype=np.float64).reshape(3)])
    poses0 = np.concatenate(
        [np.concatenate([np.asarray(rvecs0[fid], dtype=np.float64).reshape(3), np.asarray(tvecs0[fid], dtype=np.float64).reshape(3)]) for fid in fids],
        axis=0,
    )
    p0 = np.concatenate([coeff0, rig0, poses0], axis=0)

    # Gauge constraints to reduce drift:
    # - center: x(u0,v0)=0, y(u0,v0)=0
    # - local Jacobian near center: d(x,y)/d(u,v) ~ diag(1/f0, 1/f0), cross terms ~ 0
    A0 = zernike_design_matrix(np.array([u0]), np.array([v0]), nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius)[0]
    eps = 1.0
    Au_p = zernike_design_matrix(np.array([u0 + eps]), np.array([v0]), nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius)[0]
    Au_m = zernike_design_matrix(np.array([u0 - eps]), np.array([v0]), nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius)[0]
    Av_p = zernike_design_matrix(np.array([u0]), np.array([v0 + eps]), nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius)[0]
    Av_m = zernike_design_matrix(np.array([u0]), np.array([v0 - eps]), nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius)[0]
    f0_px = 1.5 * float(max(image_width_px, image_height_px))
    target = 1.0 / f0_px

    def fun(p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64).reshape(-1)
        coeffs = p[: 4 * K]
        rig = p[4 * K : 4 * K + 6]
        poses = p[4 * K + 6 :]
        if poses.size != 6 * len(fids):
            raise RuntimeError("invalid pose vector size")

        cLx, cLy, cRx, cRy = _unpack_coeffs(coeffs, K)
        rig_rvec = rig[:3]
        rig_tvec = rig[3:]
        R_RL = R.from_rotvec(rig_rvec).as_matrix()

        res_parts: list[np.ndarray] = []
        for k, fid in enumerate(fids):
            fr = frames_clean[fid]
            A_L = A_left_by_fid[fid]
            A_R = A_right_by_fid[fid]
            dL = _dir_from_coeffs(A_L, cLx, cLy)
            dR = _dir_from_coeffs(A_R, cRx, cRy)

            rvec = poses[6 * k : 6 * k + 3]
            tvec = poses[6 * k + 3 : 6 * k + 6]
            R_i = R.from_rotvec(rvec).as_matrix()
            P_L = (R_i @ fr.P_board_mm.T).T + tvec.reshape(1, 3)
            P_R = (R_RL @ P_L.T).T + rig_tvec.reshape(1, 3)

            res_parts.append(_point_to_ray_residual(P_L, dL))
            res_parts.append(_point_to_ray_residual(P_R, dR))

        if lam_center > 0:
            x0_L = float(A0 @ cLx)
            y0_L = float(A0 @ cLy)
            x0_R = float(A0 @ cRx)
            y0_R = float(A0 @ cRy)
            res_parts.append(np.sqrt(lam_center) * np.array([x0_L, y0_L, x0_R, y0_R], dtype=np.float64))

        if lam_jacobian > 0:
            # Finite-difference Jacobian at (u0,v0) in normalized coordinates.
            dxdu_L = float((Au_p @ cLx - Au_m @ cLx) / (2.0 * eps))
            dxdv_L = float((Av_p @ cLx - Av_m @ cLx) / (2.0 * eps))
            dydu_L = float((Au_p @ cLy - Au_m @ cLy) / (2.0 * eps))
            dydv_L = float((Av_p @ cLy - Av_m @ cLy) / (2.0 * eps))
            dxdu_R = float((Au_p @ cRx - Au_m @ cRx) / (2.0 * eps))
            dxdv_R = float((Av_p @ cRx - Av_m @ cRx) / (2.0 * eps))
            dydu_R = float((Au_p @ cRy - Au_m @ cRy) / (2.0 * eps))
            dydv_R = float((Av_p @ cRy - Av_m @ cRy) / (2.0 * eps))

            res_parts.append(
                np.sqrt(lam_jacobian)
                * np.array(
                    [
                        dxdu_L - target,
                        dydv_L - target,
                        dxdv_L,
                        dydu_L,
                        dxdu_R - target,
                        dydv_R - target,
                        dxdv_R,
                        dydu_R,
                    ],
                    dtype=np.float64,
                )
            )

        if lam_coeff > 0:
            res_parts.append(np.sqrt(lam_coeff) * coeffs)

        return np.concatenate(res_parts, axis=0)

    sol = least_squares(fun, p0, method="trf", loss=loss, f_scale=float(f_scale_mm), max_nfev=int(max_nfev))

    coeffs = sol.x[: 4 * K]
    rig = sol.x[4 * K : 4 * K + 6]
    poses = sol.x[4 * K + 6 :]
    cLx, cLy, cRx, cRy = _unpack_coeffs(coeffs, K)
    rig_rvec = rig[:3].copy()
    rig_tvec = rig[3:].copy()

    rvecs: dict[int, np.ndarray] = {}
    tvecs: dict[int, np.ndarray] = {}
    for k, fid in enumerate(fids):
        rvecs[fid] = poses[6 * k : 6 * k + 3].copy()
        tvecs[fid] = poses[6 * k + 3 : 6 * k + 6].copy()

    diag = {
        "opt_cost": float(sol.cost),
        "opt_nfev": float(sol.nfev),
        "opt_success": float(bool(sol.success)),
        "n_frames": float(len(fids)),
        "n_points_total": float(sum(int(frames_clean[fid].P_board_mm.shape[0]) for fid in fids)),
        "n_modes": float(int(K)),
    }

    return CentralStereoRayFieldBAResult(
        nmax=int(nmax),
        u0_px=float(u0),
        v0_px=float(v0),
        radius_px=float(radius),
        coeffs_left_x=cLx,
        coeffs_left_y=cLy,
        coeffs_right_x=cRx,
        coeffs_right_y=cRy,
        rig_rvec=rig_rvec,
        rig_tvec=rig_tvec,
        rvecs=rvecs,
        tvecs=tvecs,
        diagnostics=diag,
    )
