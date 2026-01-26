"""
Virtual rectification for ray-field stereo (central model, Zernike backend).

This module builds dense remap LUTs (mapx/mapy) that warp a target rectified
image grid into the original images, allowing standard 1D dense stereo
(SGM/BM/Census) on horizontally aligned epipolar lines.

Notes
-----
- Backend-agnostic: relies only on `ray_model.dir(u, v)` giving a unit 3D
  direction in the camera frame.
- Rectified “virtual” camera is pinhole; two dense warps (left/right) map
  rectified pixels to source pixels via direction→pixel inversion.
- For speed, remaps are expected to be precomputed and cached once per model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


@dataclass
class RectifyParams:
    width: int
    height: int
    fx: Optional[float] = None
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    max_iters: int = 15
    eps_angle: float = 1e-6
    eps_step: float = 1e-3
    coarse_step: int = 8  # pixels; 0 disables coarse init
    coarse_topk: int = 4
    border_mode: int = cv2.BORDER_CONSTANT
    border_value: float = 0.0
    # LUT inverse (direction->pixel) settings
    lut_quant: int = 32  # number of bins per axis for direction quantization (theta/phi)
    lut_use: bool = True


def _build_rect_axes(t_lr: np.ndarray, up_hint: np.ndarray = np.array([0, 1, 0], dtype=np.float64)) -> np.ndarray:
    """Return R_rect (rect frame rows expressed in left cam frame)."""
    b = np.asarray(t_lr, dtype=np.float64).reshape(3)
    e1 = _normalize(b)
    u = up_hint.astype(np.float64).reshape(3)
    if abs(float(np.dot(u, e1))) > 0.95:
        u = np.array([0, 0, 1], dtype=np.float64)
    e2 = _normalize(u - np.dot(u, e1) * e1)
    e3 = np.cross(e1, e2)
    R_rect = np.stack([e1, e2, e3], axis=0)  # rows are rect axes in left frame
    return R_rect


def _default_intrinsics(width: int, height: int) -> Tuple[float, float, float, float]:
    cx = width * 0.5
    cy = height * 0.5
    fx = fy = 0.9 * width  # heuristic; wide-enough virtual FOV
    return fx, fy, cx, cy


def _direction_field_from_model(ray_model, width: int, height: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Coarse grid of directions for init. Returns dirs (Hc,Wc,3), coords (Hc,Wc,2)."""
    ys = np.arange(0, height, step, dtype=np.float64)
    xs = np.arange(0, width, step, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    coords = np.stack([grid_x, grid_y], axis=-1)
    dirs = np.zeros((coords.shape[0], coords.shape[1], 3), dtype=np.float64)
    for j in range(coords.shape[0]):
        for i in range(coords.shape[1]):
            u, v = coords[j, i]
            dirs[j, i] = ray_model.dir(float(u), float(v))
    dirs = _normalize(dirs)
    return dirs, coords


def _build_direction_lut(ray_model, width: int, height: int, quant: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a coarse inverse LUT: quantize directions on the unit sphere (theta,phi grid)
    and store one representative pixel (u,v) for each bin.

    Returns
    -------
    lut_dirs : (Q,Q,3) representative direction per bin (unit)
    lut_uv   : (Q,Q,2) pixel coordinate for that bin (-1,-1 if empty)
    """
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    uu = uu.reshape(-1)
    vv = vv.reshape(-1)
    dirs = np.zeros((uu.size, 3), dtype=np.float64)
    for i in range(uu.size):
        dirs[i] = ray_model.dir(float(uu[i]), float(vv[i]))
    dirs = _normalize(dirs)
    # Spherical bins
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    theta = np.arctan2(y, x)  # [-pi,pi]
    phi = np.arctan2(np.sqrt(x * x + y * y), z)  # [0,pi]
    theta_idx = ((theta + np.pi) / (2 * np.pi) * quant).astype(int)
    phi_idx = (phi / np.pi * quant).astype(int)
    theta_idx = np.clip(theta_idx, 0, quant - 1)
    phi_idx = np.clip(phi_idx, 0, quant - 1)
    lut_uv = -np.ones((quant, quant, 2), dtype=np.float32)
    lut_dirs = np.zeros((quant, quant, 3), dtype=np.float32)
    for i in range(uu.size):
        ti = theta_idx[i]
        pi = phi_idx[i]
        if lut_uv[pi, ti, 0] < 0:  # empty bin
            lut_uv[pi, ti, 0] = uu[i]
            lut_uv[pi, ti, 1] = vv[i]
            lut_dirs[pi, ti] = dirs[i]
    lut_dirs = _normalize(lut_dirs)
    return lut_dirs, lut_uv


def _invert_direction_newton(ray_model, d_target: np.ndarray, init_uv: np.ndarray, max_iters: int, eps_angle: float, eps_step: float) -> Optional[np.ndarray]:
    """Invert direction→pixel via Gauss-Newton with finite differences."""
    uv = init_uv.astype(np.float64).reshape(2)
    for _ in range(max_iters):
        d = ray_model.dir(float(uv[0]), float(uv[1]))
        d = _normalize(d)
        r = d - d_target
        angle2 = np.dot(r, r)
        if angle2 < eps_angle**2:
            return uv
        # finite-diff Jacobian
        h = 1e-3
        du = _normalize(ray_model.dir(float(uv[0] + h), float(uv[1]))) - d
        dv = _normalize(ray_model.dir(float(uv[0]), float(uv[1] + h))) - d
        J = np.stack([du / h, dv / h], axis=1)  # (3,2)
        try:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]  # (2,)
        except np.linalg.LinAlgError:
            return None
        uv += step
        if np.linalg.norm(step) < eps_step:
            return uv
    return None


def _coarse_init(d_target: np.ndarray, dirs: np.ndarray, coords: np.ndarray, topk: int) -> np.ndarray:
    """Return best coarse coordinate by cosine similarity."""
    d_flat = dirs.reshape(-1, 3)
    c_flat = coords.reshape(-1, 2)
    dots = d_flat @ d_target
    idx = int(np.argmax(dots))
    return c_flat[idx]


def build_virtual_rectify_maps(
    ray_model_L,
    ray_model_R,
    R_lr: np.ndarray,
    t_lr: np.ndarray,
    params: RectifyParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build rectification remap LUTs (mapx/mapy) for left and right images.

    Returns
    -------
    mapx_L, mapy_L, mapx_R, mapy_R : float32 arrays (H', W')
    R_rect : 3x3 rotation (rect frame rows in left frame)
    """
    H, W = params.height, params.width
    fx = params.fx
    fy = params.fy
    cx = params.cx
    cy = params.cy
    if fx is None or fy is None or cx is None or cy is None:
        fx, fy, cx, cy = _default_intrinsics(W, H)

    R_lr = np.asarray(R_lr, dtype=np.float64).reshape(3, 3)
    t_lr = np.asarray(t_lr, dtype=np.float64).reshape(3)
    R_rect = _build_rect_axes(t_lr)
    # Right rect rotation so that rectified frame is common
    R_rect_R = R_rect @ R_lr.T

    # Coarse init grids (optional)
    if params.coarse_step > 0:
        dirs_L, coords_L = _direction_field_from_model(ray_model_L, ray_model_L.width, ray_model_L.height, params.coarse_step)
        dirs_R, coords_R = _direction_field_from_model(ray_model_R, ray_model_R.width, ray_model_R.height, params.coarse_step)
    else:
        dirs_L = coords_L = dirs_R = coords_R = None
    # Direction->pixel inverse LUT (optional)
    if params.lut_use:
        lut_dirs_L, lut_uv_L = _build_direction_lut(ray_model_L, ray_model_L.width, ray_model_L.height, params.lut_quant)
        lut_dirs_R, lut_uv_R = _build_direction_lut(ray_model_R, ray_model_R.width, ray_model_R.height, params.lut_quant)
    else:
        lut_dirs_L = lut_uv_L = lut_dirs_R = lut_uv_R = None

    mapx_L = np.full((H, W), -1.0, dtype=np.float32)
    mapy_L = np.full((H, W), -1.0, dtype=np.float32)
    mapx_R = np.full((H, W), -1.0, dtype=np.float32)
    mapy_R = np.full((H, W), -1.0, dtype=np.float32)

    for v in range(H):
        y = (v - cy) / fy
        for u in range(W):
            x = (u - cx) / fx
            d_rect = _normalize(np.array([x, y, 1.0], dtype=np.float64))
            # left camera direction
            d_L = _normalize(R_rect.T @ d_rect)
            # right camera direction
            d_R = _normalize(R_rect_R.T @ d_rect)

            # invert left
            init_L = None
            if params.lut_use and lut_dirs_L is not None and lut_uv_L is not None:
                dots = (lut_dirs_L.reshape(-1, 3) @ d_L)
                idx = int(np.argmax(dots))
                pi = idx // params.lut_quant
                ti = idx - pi * params.lut_quant
                init_L = lut_uv_L[pi, ti]
                if init_L[0] < 0:
                    init_L = None
            if init_L is None and dirs_L is not None:
                init_L = _coarse_init(d_L, dirs_L, coords_L, params.coarse_topk)
            if init_L is None:
                init_L = np.array([ray_model_L.width * 0.5, ray_model_L.height * 0.5], dtype=np.float64)
            uv_L = _invert_direction_newton(ray_model_L, d_L, init_L, params.max_iters, params.eps_angle, params.eps_step)

            # invert right
            init_R = None
            if params.lut_use and lut_dirs_R is not None and lut_uv_R is not None:
                dots = (lut_dirs_R.reshape(-1, 3) @ d_R)
                idx = int(np.argmax(dots))
                pi = idx // params.lut_quant
                ti = idx - pi * params.lut_quant
                init_R = lut_uv_R[pi, ti]
                if init_R[0] < 0:
                    init_R = None
            if init_R is None and dirs_R is not None:
                init_R = _coarse_init(d_R, dirs_R, coords_R, params.coarse_topk)
            if init_R is None:
                init_R = np.array([ray_model_R.width * 0.5, ray_model_R.height * 0.5], dtype=np.float64)
            uv_R = _invert_direction_newton(ray_model_R, d_R, init_R, params.max_iters, params.eps_angle, params.eps_step)

            if uv_L is not None:
                mapx_L[v, u] = uv_L[0]
                mapy_L[v, u] = uv_L[1]
            if uv_R is not None:
                mapx_R[v, u] = uv_R[0]
                mapy_R[v, u] = uv_R[1]

    return mapx_L, mapy_L, mapx_R, mapy_R, R_rect


def rectify_pair(images: Tuple[np.ndarray, np.ndarray], maps: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], params: RectifyParams) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply precomputed remap to rectify a pair of images.
    """
    I_L, I_R = images
    mapx_L, mapy_L, mapx_R, mapy_R = maps
    I_L_rect = cv2.remap(I_L, mapx_L, mapy_L, interpolation=cv2.INTER_LINEAR, borderMode=params.border_mode, borderValue=params.border_value)
    I_R_rect = cv2.remap(I_R, mapx_R, mapy_R, interpolation=cv2.INTER_LINEAR, borderMode=params.border_mode, borderValue=params.border_value)
    return I_L_rect, I_R_rect
