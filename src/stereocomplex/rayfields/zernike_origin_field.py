from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.core.model_compact.zernike import ZernikeMode, eval_real_zernike, zernike_modes
from stereocomplex.synthetic.parallel_plate import pinhole_ray_from_pixel


def _project_transverse(v: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Remove the component of v along d: v - (v·d) d.

    Used for the transverse gauge O(u,v)⊥d(u,v) and for projecting direction
    perturbations. Shared with the BA closure in fit_zernike_origin_field.py
    so a change here propagates to both code paths.
    """
    return v - np.sum(v * d, axis=-1, keepdims=True) * d


@dataclass(frozen=True)
class ZernikeOriginFieldConfig:
    image_size: tuple[int, int]
    max_order: int = 4
    normalization: str = "diagonal_disk"
    enforce_transverse_gauge: bool = True


@dataclass(frozen=True)
class ZernikeOriginFieldCoefficients:
    """Zernike coefficients shaped `(n_terms, 3)` in millimetres."""

    coeffs: np.ndarray


@dataclass(frozen=True)
class ZernikeRayFieldCoefficients:
    """
    Zernike coefficients for a generic central-reference rayfield.

    `origin_coeffs` are in millimetres. `direction_coeffs` are dimensionless
    perturbations added to the pinhole direction before renormalization.
    """

    origin_coeffs: np.ndarray
    direction_coeffs: np.ndarray


class ZernikeOriginField:
    """
    Generic non-central ray model with pinhole directions and a Zernike origin field.

    The physical oracle may choose any point on a ray. This model uses the canonical
    transverse gauge `O(u,v) dot d(u,v) = 0` by default.
    """

    def __init__(
        self,
        K: np.ndarray,
        config: ZernikeOriginFieldConfig,
        coefficients: ZernikeOriginFieldCoefficients | None = None,
    ):
        if config.normalization != "diagonal_disk":
            raise ValueError("only normalization='diagonal_disk' is currently supported")
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.config = config
        self.modes: tuple[ZernikeMode, ...] = tuple(zernike_modes(int(config.max_order)))
        if coefficients is None:
            coeffs = np.zeros((len(self.modes), 3), dtype=np.float64)
        else:
            coeffs = np.asarray(coefficients.coeffs, dtype=np.float64).reshape(-1, 3)
        if coeffs.shape != (len(self.modes), 3):
            raise ValueError(f"coeffs must have shape {(len(self.modes), 3)}")
        self.coefficients = ZernikeOriginFieldCoefficients(coeffs=coeffs)

    @property
    def coeffs(self) -> np.ndarray:
        return self.coefficients.coeffs

    def basis(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the Zernike design matrix for pixel arrays flattened to `(N,)`."""
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)
        u_flat, v_flat = np.broadcast_arrays(u_arr, v_arr)
        width, height = self.config.image_size
        if width <= 1 or height <= 1:
            raise ValueError("image_size must be larger than one pixel in each dimension")
        xi = 2.0 * u_flat.reshape(-1) / float(width - 1) - 1.0
        zeta = 2.0 * v_flat.reshape(-1) / float(height - 1) - 1.0
        rho = np.sqrt(xi * xi + zeta * zeta) / np.sqrt(2.0)
        theta = np.arctan2(zeta, xi)
        A = np.empty((rho.size, len(self.modes)), dtype=np.float64)
        for j, mode in enumerate(self.modes):
            A[:, j] = eval_real_zernike(mode, rho, theta)
        return A

    def direction(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return normalized pinhole directions `d_pinhole(u,v)`."""
        return pinhole_ray_from_pixel(u, v, self.K).reshape(-1, 3)

    def raw_origin(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the unconstrained Zernike origin field."""
        return self.basis(u, v) @ self.coeffs

    def origin(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the canonical origin field, optionally transverse to the ray direction."""
        O_raw = self.raw_origin(u, v)
        if not self.config.enforce_transverse_gauge:
            return O_raw
        return _project_transverse(O_raw, self.direction(u, v))

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return `(O(u,v), d(u,v))`, both shaped `(N,3)`."""
        return self.origin(u, v), self.direction(u, v)


class ZernikeRayField(ZernikeOriginField):
    """
    Generic ray model with both origin and direction Zernike fields.

    Directions are initialized from the pinhole model and corrected by a smooth
    Zernike perturbation:

    `d(u,v) = normalize(d0(u,v) + delta_d_perp(u,v))`.

    The perturbation is projected transverse to the pinhole direction. This keeps
    the parameterization well conditioned around the central initialization and
    avoids wasting coefficients on direction-scale changes.
    """

    def __init__(
        self,
        K: np.ndarray,
        config: ZernikeOriginFieldConfig,
        coefficients: ZernikeRayFieldCoefficients | None = None,
    ):
        if coefficients is None:
            super().__init__(K, config)
            direction_coeffs = np.zeros_like(self.coeffs)
        else:
            super().__init__(
                K,
                config,
                ZernikeOriginFieldCoefficients(np.asarray(coefficients.origin_coeffs, dtype=np.float64)),
            )
            direction_coeffs = np.asarray(coefficients.direction_coeffs, dtype=np.float64).reshape(-1, 3)
        if direction_coeffs.shape != self.coeffs.shape:
            raise ValueError(f"direction_coeffs must have shape {self.coeffs.shape}")
        self.rayfield_coefficients = ZernikeRayFieldCoefficients(
            origin_coeffs=self.coeffs,
            direction_coeffs=direction_coeffs,
        )

    @property
    def origin_coeffs(self) -> np.ndarray:
        return self.rayfield_coefficients.origin_coeffs

    @property
    def direction_coeffs(self) -> np.ndarray:
        return self.rayfield_coefficients.direction_coeffs

    def direction_delta(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the transverse direction perturbation before normalization."""
        d0 = super().direction(u, v)
        return _project_transverse(self.basis(u, v) @ self.direction_coeffs, d0)

    def direction(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return normalized fitted directions."""
        d0 = super().direction(u, v)
        d = d0 + self.direction_delta(u, v)
        return d / np.linalg.norm(d, axis=1, keepdims=True)
