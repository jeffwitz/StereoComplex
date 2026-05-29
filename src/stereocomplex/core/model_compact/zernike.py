from __future__ import annotations

import math
from dataclasses import dataclass

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
    """Evaluate real-valued Zernike polynomials at given polar coordinates.

    Uses the Noll indexing scheme.  Returns the value of Z_n^m(rho, theta)
    for each mode, where rho is the normalised radial coordinate (0 at centre,
    1 at the edge of the unit disk) and theta is the azimuthal angle in radians.

    Parameters
    ----------
    mode : ZernikeMode
        Zernike mode descriptor carrying radial order, azimuthal order and
        sine/cosine branch.
    r : ndarray
        Normalised radial coordinates, in [0, 1].
    theta : ndarray
        Azimuthal angles in radians, same shape as ``r``.

    Returns
    -------
    ndarray
        Zernike polynomial values, same shape as ``r``.
    """
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
    """Map pixel coordinates to unit-disk polar coordinates for Zernike evaluation.

    Pixels are centred at (u0_px, v0_px) and scaled by ``radius_px`` so that
    the disk of that radius maps to rho = 1.  Pixels outside the disk produce
    rho > 1.

    Parameters
    ----------
    u_px : ndarray
        Pixel x-coordinates.
    v_px : ndarray
        Pixel y-coordinates.
    u0_px : float
        x-coordinate of the disk centre in pixels.
    v0_px : float
        y-coordinate of the disk centre in pixels.
    radius_px : float
        Disk radius in pixels (must be > 0).

    Returns
    -------
    r : ndarray
        Normalised radial coordinates (0 at centre, 1 at the disk edge).
    theta : ndarray
        Azimuthal angles in radians.
    mask : ndarray, bool
        True where r <= 1 (inside the unit disk).
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
    """Build the Zernike design matrix for a pixel grid.

    Evaluates all real Zernike polynomials up to radial order ``nmax`` at the
    unit-disk coordinates of each pixel.  Columns correspond to Zernike modes
    in Noll order; rows correspond to pixels inside the unit disk.

    Parameters
    ----------
    u_px : ndarray
        Pixel x-coordinates.
    v_px : ndarray
        Pixel y-coordinates, same shape as u_px.
    nmax : int
        Maximum radial order (inclusive).  Produces K = (nmax+1)(nmax+2)/2 modes.
    u0_px : float
        x-coordinate of the unit-disk centre in pixels.
    v0_px : float
        y-coordinate of the unit-disk centre in pixels.
    radius_px : float
        Unit-disk radius in pixels.

    Returns
    -------
    A : ndarray, shape (N_in, K)
        Design matrix.  Row i, column j = mode j evaluated at pixel i.
    mask : ndarray, bool
        True where the pixel lies inside the unit disk.
    modes : list of ZernikeMode
        Zernike mode descriptors in Noll order (length K).
    """
    modes = zernike_modes(nmax)
    r, theta, mask = pixel_to_unit_disk(u_px, v_px, u0_px=u0_px, v0_px=v0_px, radius_px=radius_px)
    r_in = r[mask]
    th_in = theta[mask]
    A = np.empty((r_in.size, len(modes)), dtype=np.float64)
    for k, mode in enumerate(modes):
        A[:, k] = eval_real_zernike(mode, r_in, th_in)
    return A, mask, modes
