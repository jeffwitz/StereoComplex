from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from stereocomplex.core.model_compact.zernike import zernike_design_matrix


@dataclass(frozen=True)
class FrameObservations:
    """
    Observations for one camera and one frame.

    - `uv_px`: observed pixels (dataset pixel-center convention)
    - `P_board_mm`: corresponding 3D points in the board frame (mm)
    """

    uv_px: np.ndarray  # (N,2)
    P_board_mm: np.ndarray  # (N,3)


@dataclass(frozen=True)
class CentralRayFieldBAResult:
    nmax: int
    u0_px: float
    v0_px: float
    radius_px: float
    coeffs_x: np.ndarray  # (K,)
    coeffs_y: np.ndarray  # (K,)
    rvecs: dict[int, np.ndarray]  # frame_id -> (3,)
    tvecs: dict[int, np.ndarray]  # frame_id -> (3,)
    diagnostics: dict[str, float]


def default_disk(width_px: int, height_px: int) -> tuple[float, float, float]:
    u0 = (width_px - 1) / 2.0
    v0 = (height_px - 1) / 2.0
    radius = 0.5 * float(np.hypot(width_px - 1, height_px - 1))
    return u0, v0, radius


def _dir_from_coeffs(A: np.ndarray, coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> np.ndarray:
    x = A @ coeffs_x
    y = A @ coeffs_y
    d = np.stack([x, y, np.ones_like(x)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def _point_to_ray_residual(P_cam: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Residual vector (3,) for each correspondence: (I - dd^T) P_cam.
    """
    proj = np.sum(P_cam * d, axis=-1, keepdims=True) * d
    return (P_cam - proj).reshape(-1)


def _pack_coeffs(coeffs_x: np.ndarray, coeffs_y: np.ndarray) -> np.ndarray:
    return np.concatenate([coeffs_x.reshape(-1), coeffs_y.reshape(-1)], axis=0)


def _unpack_coeffs(p: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if p.size != 2 * K:
        raise ValueError("invalid coeff vector size")
    return p[:K].copy(), p[K:].copy()


def _fit_coeffs_least_squares(
    *,
    frames: dict[int, FrameObservations],
    A_by_frame: dict[int, np.ndarray],
    rvecs: dict[int, np.ndarray],
    tvecs: dict[int, np.ndarray],
    coeffs0_x: np.ndarray,
    coeffs0_y: np.ndarray,
    lam_coeff: float,
    loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"],
    f_scale_mm: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    from scipy.optimize import least_squares  # type: ignore
    from scipy.spatial.transform import Rotation as R  # type: ignore

    K = int(coeffs0_x.size)
    p0 = _pack_coeffs(coeffs0_x, coeffs0_y)

    def fun(p: np.ndarray) -> np.ndarray:
        coeffs_x, coeffs_y = _unpack_coeffs(p, K)
        res_parts: list[np.ndarray] = []
        for fid, fr in frames.items():
            A = A_by_frame[fid]
            d = _dir_from_coeffs(A, coeffs_x, coeffs_y)
            rot = R.from_rotvec(rvecs[fid].reshape(3)).as_matrix()
            P_cam = (rot @ fr.P_board_mm.T).T + tvecs[fid].reshape(1, 3)
            res_parts.append(_point_to_ray_residual(P_cam, d))
        if lam_coeff > 0:
            res_parts.append(np.sqrt(lam_coeff) * p)
        return np.concatenate(res_parts, axis=0)

    sol = least_squares(fun, p0, method="trf", loss=loss, f_scale=float(f_scale_mm), max_nfev=int(max_nfev))
    cx, cy = _unpack_coeffs(sol.x, K)
    diag = {
        "coeff_opt_cost": float(sol.cost),
        "coeff_opt_nfev": float(sol.nfev),
        "coeff_opt_success": float(bool(sol.success)),
    }
    return cx, cy, diag


def _fit_pose_least_squares(
    *,
    fr: FrameObservations,
    A: np.ndarray,
    coeffs_x: np.ndarray,
    coeffs_y: np.ndarray,
    rvec0: np.ndarray,
    tvec0: np.ndarray,
    loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"],
    f_scale_mm: float,
    max_nfev: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    from scipy.optimize import least_squares  # type: ignore
    from scipy.spatial.transform import Rotation as R  # type: ignore

    p0 = np.concatenate([rvec0.reshape(3), tvec0.reshape(3)], axis=0)
    d = _dir_from_coeffs(A, coeffs_x, coeffs_y)

    def fun(p: np.ndarray) -> np.ndarray:
        rvec = p[:3]
        tvec = p[3:]
        rot = R.from_rotvec(rvec).as_matrix()
        P_cam = (rot @ fr.P_board_mm.T).T + tvec.reshape(1, 3)
        return _point_to_ray_residual(P_cam, d)

    sol = least_squares(fun, p0, method="trf", loss=loss, f_scale=float(f_scale_mm), max_nfev=int(max_nfev))
    rvec = sol.x[:3].copy()
    tvec = sol.x[3:].copy()
    diag = {
        "pose_opt_cost": float(sol.cost),
        "pose_opt_nfev": float(sol.nfev),
        "pose_opt_success": float(bool(sol.success)),
    }
    return rvec, tvec, diag


def fit_central_rayfield_ba(
    *,
    frames: dict[int, FrameObservations],
    image_width_px: int,
    image_height_px: int,
    nmax: int,
    rvecs0: dict[int, np.ndarray],
    tvecs0: dict[int, np.ndarray],
    coeffs0_x: np.ndarray | None = None,
    coeffs0_y: np.ndarray | None = None,
    lam_coeff: float = 1e-3,
    loss: Literal["linear", "huber", "soft_l1", "cauchy", "arctan"] = "huber",
    f_scale_mm: float = 1.0,
    outer_iters: int = 6,
    max_nfev_coeff: int = 80,
    max_nfev_pose: int = 50,
) -> CentralRayFieldBAResult:
    """
    Alternating minimization for a central ray-field and per-frame poses.

    We minimize point-to-ray distances:
      r_ij = (I - d d^T) (R_i P_j + t_i)

    where d = d(u_ij, v_ij) comes from a compact Zernike model, and (R_i,t_i)
    is the board pose for each frame in camera coordinates.
    """
    if not frames:
        raise ValueError("frames is empty")
    if nmax < 0:
        raise ValueError("nmax must be >= 0")

    u0, v0, radius = default_disk(int(image_width_px), int(image_height_px))

    # Precompute design matrices A_i from observed pixels (one per frame).
    # (Keep a local cleaned copy: we don't want to mutate the caller's dict.)
    frames_clean: dict[int, FrameObservations] = {}
    A_by_frame: dict[int, np.ndarray] = {}
    K = None
    for fid, fr in frames.items():
        uv = np.asarray(fr.uv_px, dtype=np.float64).reshape(-1, 2)
        P_board = np.asarray(fr.P_board_mm, dtype=np.float64).reshape(-1, 3)
        A, mask, _modes = zernike_design_matrix(
            uv[:, 0], uv[:, 1], nmax=int(nmax), u0_px=u0, v0_px=v0, radius_px=radius
        )
        if not np.all(mask):
            # With the circumscribed radius, mask should be all-true; keep a safe fallback anyway.
            P_board = P_board[mask]
            uv = uv[mask]
            A = A[mask]
        frames_clean[int(fid)] = FrameObservations(uv_px=uv, P_board_mm=P_board)
        A_by_frame[int(fid)] = A
        K = A.shape[1] if K is None else K
        if A.shape[1] != K:
            raise RuntimeError("inconsistent Zernike design matrix width")
    assert K is not None

    rvecs = {int(fid): np.asarray(rvecs0[int(fid)], dtype=np.float64).reshape(3) for fid in frames_clean}
    tvecs = {int(fid): np.asarray(tvecs0[int(fid)], dtype=np.float64).reshape(3) for fid in frames_clean}

    if coeffs0_x is None:
        coeffs_x = np.zeros((K,), dtype=np.float64)
    else:
        coeffs_x = np.asarray(coeffs0_x, dtype=np.float64).reshape(K).copy()
    if coeffs0_y is None:
        coeffs_y = np.zeros((K,), dtype=np.float64)
    else:
        coeffs_y = np.asarray(coeffs0_y, dtype=np.float64).reshape(K).copy()

    diag: dict[str, float] = {
        "outer_iters": float(int(outer_iters)),
        "n_frames": float(len(frames_clean)),
        "n_points_total": float(sum(int(fr.uv_px.shape[0]) for fr in frames_clean.values())),
        "n_modes": float(int(K)),
    }

    for it in range(int(outer_iters)):
        coeffs_x, coeffs_y, d1 = _fit_coeffs_least_squares(
            frames=frames_clean,
            A_by_frame=A_by_frame,
            rvecs=rvecs,
            tvecs=tvecs,
            coeffs0_x=coeffs_x,
            coeffs0_y=coeffs_y,
            lam_coeff=float(lam_coeff),
            loss=loss,
            f_scale_mm=float(f_scale_mm),
            max_nfev=int(max_nfev_coeff),
        )
        diag[f"iter{it}_coeff_cost"] = d1["coeff_opt_cost"]

        pose_cost_sum = 0.0
        for fid, fr in frames_clean.items():
            rvec, tvec, d2 = _fit_pose_least_squares(
                fr=fr,
                A=A_by_frame[fid],
                coeffs_x=coeffs_x,
                coeffs_y=coeffs_y,
                rvec0=rvecs[fid],
                tvec0=tvecs[fid],
                loss=loss,
                f_scale_mm=float(f_scale_mm),
                max_nfev=int(max_nfev_pose),
            )
            rvecs[fid] = rvec
            tvecs[fid] = tvec
            pose_cost_sum += float(d2["pose_opt_cost"])
        diag[f"iter{it}_pose_cost_sum"] = pose_cost_sum

    return CentralRayFieldBAResult(
        nmax=int(nmax),
        u0_px=float(u0),
        v0_px=float(v0),
        radius_px=float(radius),
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        rvecs=rvecs,
        tvecs=tvecs,
        diagnostics=diag,
    )
