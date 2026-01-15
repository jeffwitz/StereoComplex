from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ZernikeMode:
    """
    Real-valued Zernike mode on the unit disk.

    - (n, m) follow the classical Zernike definition (n >= 0, 0 <= m <= n, n-m even).
    - `kind` is:
        - "m0" for m=0 (purely radial)
        - "cos" for cos(m*theta)
        - "sin" for sin(m*theta)
    """

    n: int
    m: int
    kind: str  # "m0" | "cos" | "sin"

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be >= 0")
        if self.m < 0 or self.m > self.n:
            raise ValueError("m must satisfy 0 <= m <= n")
        if (self.n - self.m) % 2 != 0:
            raise ValueError("n-m must be even")
        if self.m == 0 and self.kind != "m0":
            raise ValueError("m=0 requires kind='m0'")
        if self.m > 0 and self.kind not in {"cos", "sin"}:
            raise ValueError("m>0 requires kind in {'cos','sin'}")


def zernike_modes(nmax: int) -> list[ZernikeMode]:
    """
    Generate real Zernike modes up to radial order `nmax` (inclusive).

    Ordering: increasing n, then increasing m; for m>0: (cos, sin).
    """
    if nmax < 0:
        raise ValueError("nmax must be >= 0")
    modes: list[ZernikeMode] = []
    for n in range(nmax + 1):
        for m in range(0, n + 1):
            if (n - m) % 2 != 0:
                continue
            if m == 0:
                modes.append(ZernikeMode(n=n, m=m, kind="m0"))
            else:
                modes.append(ZernikeMode(n=n, m=m, kind="cos"))
                modes.append(ZernikeMode(n=n, m=m, kind="sin"))
    return modes


def _radial_coeffs(n: int, m: int) -> np.ndarray:
    """
    Coefficients of R_n^m(r) as a polynomial in r:
      R_n^m(r) = sum_k c[k] r^{n-2k},  k = 0..(n-m)/2
    Returned as (powers, coeffs) packed into a structured array for fast eval.
    """
    m = abs(m)
    if (n - m) % 2 != 0:
        raise ValueError("n-m must be even")
    kmax = (n - m) // 2
    powers = np.array([n - 2 * k for k in range(kmax + 1)], dtype=np.int32)
    coeffs = np.empty((kmax + 1,), dtype=np.float64)
    for k in range(kmax + 1):
        num = math.factorial(n - k)
        den = (
            math.factorial(k)
            * math.factorial((n + m) // 2 - k)
            * math.factorial((n - m) // 2 - k)
        )
        coeffs[k] = ((-1.0) ** k) * (num / den)
    return np.stack([powers.astype(np.float64), coeffs], axis=0)  # (2, K)


_RADIAL_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _radial_poly(n: int, m: int, r: np.ndarray) -> np.ndarray:
    key = (n, abs(m))
    packed = _RADIAL_CACHE.get(key)
    if packed is None:
        packed = _radial_coeffs(n, abs(m))
        _RADIAL_CACHE[key] = packed
    powers = packed[0]
    coeffs = packed[1]
    out = np.zeros_like(r, dtype=np.float64)
    for p, c in zip(powers, coeffs, strict=True):
        out += c * (r**p)
    return out


def eval_real_zernike(mode: ZernikeMode, r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    R = _radial_poly(mode.n, mode.m, r)
    if mode.m == 0:
        return R
    if mode.kind == "cos":
        return R * np.cos(mode.m * theta)
    return R * np.sin(mode.m * theta)


def pixel_to_unit_disk(
    u_px: np.ndarray,
    v_px: np.ndarray,
    u0_px: float,
    v0_px: float,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map pixels to unit-disk polar coordinates (r, theta) with a mask for r<=1.
    """
    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")
    u_px = np.asarray(u_px, dtype=np.float64)
    v_px = np.asarray(v_px, dtype=np.float64)
    x = (u_px - float(u0_px)) / float(radius_px)
    y = (v_px - float(v0_px)) / float(radius_px)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    mask = r <= 1.0
    return r, theta, mask


def zernike_design_matrix(
    u_px: np.ndarray,
    v_px: np.ndarray,
    *,
    nmax: int,
    u0_px: float,
    v0_px: float,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray, list[ZernikeMode]]:
    """
    Build design matrix A (N,K) for real Zernike modes up to `nmax`.
    Returns (A, mask, modes) where mask selects points inside the unit disk.
    """
    modes = zernike_modes(nmax)
    r, theta, mask = pixel_to_unit_disk(u_px, v_px, u0_px=u0_px, v0_px=v0_px, radius_px=radius_px)
    r_in = r[mask]
    th_in = theta[mask]
    A = np.empty((r_in.size, len(modes)), dtype=np.float64)
    for k, mode in enumerate(modes):
        A[:, k] = eval_real_zernike(mode, r_in, th_in)
    return A, mask, modes

