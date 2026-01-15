from __future__ import annotations

import math

import numpy as np


def fwhm_to_sigma(fwhm: float) -> float:
    return float(fwhm) / 2.3548200450309493


def gaussian_blur_u8(img_u8: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    """
    Applies Gaussian blur to a uint8 grayscale image.
    Uses OpenCV if available, otherwise falls back to a separable numpy implementation.
    """
    if sigma_x <= 0 and sigma_y <= 0:
        return img_u8

    sigma_x = float(max(0.0, sigma_x))
    sigma_y = float(max(0.0, sigma_y))

    try:
        import cv2  # type: ignore

        out = cv2.GaussianBlur(img_u8, ksize=(0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)
        return out
    except Exception:
        return _gaussian_blur_numpy(img_u8, sigma_x, sigma_y)


def radial_weight(h: int, w: int, start: float = 0.6, power: float = 2.0) -> np.ndarray:
    """
    Returns a (H,W) float32 weight map in [0,1] that increases from center to edges.
    start: radius fraction where edge blur begins (0..1).
    power: controls how quickly it ramps.
    """
    start = float(np.clip(start, 0.0, 1.0))
    power = float(max(1e-6, power))

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    rx = (w - 1) * 0.5
    ry = (h - 1) * 0.5
    r = np.sqrt(((xx - cx) / max(rx, 1.0)) ** 2 + ((yy - cy) / max(ry, 1.0)) ** 2)  # ~[0, sqrt(2)]
    r = np.clip(r / np.sqrt(2.0), 0.0, 1.0)
    t = np.clip((r - start) / max(1e-6, (1.0 - start)), 0.0, 1.0)
    return (t**power).astype(np.float32)


def gaussian_blur_edge_varying_u8(
    img_u8: np.ndarray,
    sigma_x_center: float,
    sigma_y_center: float,
    edge_factor: float = 1.0,
    edge_start: float = 0.6,
    edge_power: float = 2.0,
) -> np.ndarray:
    """
    Approximate spatially varying blur (stronger at edges) by blending two blurred images.
    """
    if edge_factor <= 1.0 or (sigma_x_center <= 0 and sigma_y_center <= 0):
        return gaussian_blur_u8(img_u8, sigma_x=sigma_x_center, sigma_y=sigma_y_center)

    sigma_x_edge = float(sigma_x_center) * float(edge_factor)
    sigma_y_edge = float(sigma_y_center) * float(edge_factor)

    base = gaussian_blur_u8(img_u8, sigma_x=sigma_x_center, sigma_y=sigma_y_center)
    edge = gaussian_blur_u8(img_u8, sigma_x=sigma_x_edge, sigma_y=sigma_y_edge)

    wmap = radial_weight(img_u8.shape[0], img_u8.shape[1], start=edge_start, power=edge_power)
    base_f = base.astype(np.float32)
    edge_f = edge.astype(np.float32)
    out = (1.0 - wmap) * base_f + wmap * edge_f
    return np.clip(out + 0.5, 0.0, 255.0).astype(np.uint8)


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = int(math.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _convolve1d_reflect(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = (kernel.size - 1) // 2
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(img, pad_width=pad_width, mode="reflect")

    out = np.empty_like(img, dtype=np.float64)
    if axis == 0:
        for i in range(img.shape[0]):
            out[i, :] = np.tensordot(kernel, padded[i : i + kernel.size, :], axes=(0, 0))
    else:
        for j in range(img.shape[1]):
            out[:, j] = np.tensordot(kernel, padded[:, j : j + kernel.size], axes=(0, 1))
    return out


def _gaussian_blur_numpy(img_u8: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    img = img_u8.astype(np.float64) / 255.0
    kx = _gaussian_kernel1d(sigma_x)
    ky = _gaussian_kernel1d(sigma_y)
    tmp = _convolve1d_reflect(img, kx, axis=1)
    out = _convolve1d_reflect(tmp, ky, axis=0)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)
