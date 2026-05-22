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

    def modes(self) -> tuple[ZernikeMode, ...]:
        return tuple(zernike_modes(int(self.max_order)))


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


@dataclass(frozen=True)
class ZernikeRayFieldChannel:
    """Named camera channel carrying one Zernike rayfield."""

    name: str
    field: "ZernikeRayField"


@dataclass(frozen=True)
class MultiCameraZernikeRayField:
    """Ordered collection of named Zernike rayfields for N-camera calibration."""

    channels: tuple[ZernikeRayFieldChannel, ...]

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("at least one channel is required")
        names = [channel.name for channel in self.channels]
        if any(not name for name in names):
            raise ValueError("channel names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")

    @classmethod
    def from_fields(cls, fields: dict[str, "ZernikeRayField"]) -> "MultiCameraZernikeRayField":
        return cls(
            tuple(ZernikeRayFieldChannel(name=name, field=field) for name, field in fields.items())
        )

    @classmethod
    def from_camera_configs(
        cls,
        intrinsics_by_channel: dict[str, np.ndarray],
        configs_by_channel: dict[str, ZernikeOriginFieldConfig],
    ) -> "MultiCameraZernikeRayField":
        missing = [name for name in intrinsics_by_channel if name not in configs_by_channel]
        if missing:
            raise ValueError(f"missing Zernike configs for channels: {missing}")
        fields = {
            name: ZernikeRayField(K=intrinsics, config=configs_by_channel[name])
            for name, intrinsics in intrinsics_by_channel.items()
        }
        return cls.from_fields(fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(channel.name for channel in self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def channel(self, name: str) -> "ZernikeRayField":
        for channel in self.channels:
            if channel.name == name:
                return channel.field
        raise KeyError(name)

    def ray(self, name: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.channel(name).ray(u, v)


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
        self.modes: tuple[ZernikeMode, ...] = config.modes()
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
                ZernikeOriginFieldCoefficients(
                    np.asarray(coefficients.origin_coeffs, dtype=np.float64)
                ),
            )
            direction_coeffs = np.asarray(coefficients.direction_coeffs, dtype=np.float64).reshape(
                -1, 3
            )
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


@dataclass(frozen=True)
class ZernikeCandidate:
    """Compact Zernike origin-field candidate for ray-space model selection.

    This wraps a :class:`ZernikeOriginField` into the
    :class:`~stereocomplex.physics.base.PhysicalRayFieldModel` protocol so it can
    compete alongside the physical candidates (pinhole, Brown-Conrady, CMO,
    polynomial surrogate) in :func:`~stereocomplex.physics.model_selection.select_physical_model_from_rayfield`.

    The candidate uses a **lower** max-order than the measured Zernike rayfield
    (typically ``max_order=1``, 18 parameters per channel with directions, versus
    ``max_order=4``, 60 parameters for the measurement).  If no physical model
    explains the measured rayfield well enough, the compact Zernike candidate
    wins BIC — signalling that the optics fall outside the physical catalogue.

    By default both origin and direction Zernike fields are fitted
    (``fit_directions=True``, using :class:`ZernikeRayField`).  Set
    ``fit_directions=False`` for origin-only fitting with
    :class:`ZernikeOriginField` (3 scalar coefficients per mode instead of 6).

    .. note::

       The transverse gauge :math:`O(u,v)\\cdot d(u,v)=0` is enforced at
       ray-evaluation time.  This removes one redundant degree of freedom per
       mode (the longitudinal displacement along the ray), so the *effective*
       parameter count is ``n_modes * 5`` with directions or ``n_modes * 2``
       without.  The ``n_parameters`` property reports the raw coefficient
       count (6 or 3 per mode), which slightly over-penalises the Zernike
       candidate in BIC comparisons.  The margin is 1 parameter per mode —
       negligible against the BIC differences seen in practice.
    """

    K: np.ndarray
    config: ZernikeOriginFieldConfig
    coefficients: ZernikeRayFieldCoefficients
    name: str = "zernike_compact"
    is_stereo_shared: bool = False
    fit_directions: bool = True

    @property
    def n_parameters(self) -> int:
        n_modes = len(self.config.modes())
        return n_modes * 6 if self.fit_directions else n_modes * 3

    def parameter_vector(self) -> np.ndarray:
        if self.fit_directions:
            return np.concatenate(
                [
                    np.asarray(self.coefficients.origin_coeffs, dtype=np.float64).reshape(-1),
                    np.asarray(self.coefficients.direction_coeffs, dtype=np.float64).reshape(-1),
                ]
            )
        return np.asarray(self.coefficients.origin_coeffs, dtype=np.float64).reshape(-1)

    @classmethod
    def from_parameter_vector(cls, x: np.ndarray, **kwargs) -> "ZernikeCandidate":
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        config = kwargs["config"]
        fit_directions = bool(kwargs.get("fit_directions", True))
        n_modes = len(config.modes())
        n_per_mode = 6 if fit_directions else 3
        expected = n_modes * n_per_mode
        if arr.size != expected:
            raise ValueError(
                f"ZernikeCandidate expects {expected} parameters "
                f"(max_order={config.max_order}, fit_directions={fit_directions})"
            )
        if fit_directions:
            return cls(
                K=np.asarray(kwargs["K"], dtype=np.float64).reshape(3, 3),
                config=config,
                coefficients=ZernikeRayFieldCoefficients(
                    origin_coeffs=arr[: n_modes * 3].reshape(n_modes, 3),
                    direction_coeffs=arr[n_modes * 3 :].reshape(n_modes, 3),
                ),
                fit_directions=True,
            )
        return cls(
            K=np.asarray(kwargs["K"], dtype=np.float64).reshape(3, 3),
            config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=arr.reshape(n_modes, 3),
                direction_coeffs=np.zeros((n_modes, 3), dtype=np.float64),
            ),
            fit_directions=False,
        )

    def parameter_dict(self) -> dict[str, float]:
        d: dict[str, float] = {}
        for j, mode in enumerate(self.config.modes()):
            key = f"z{j:02d}_n{mode.n}_m{mode.m}_{mode.kind}"
            d[f"{key}_Ox"] = float(self.coefficients.origin_coeffs[j, 0])
            d[f"{key}_Oy"] = float(self.coefficients.origin_coeffs[j, 1])
            d[f"{key}_Oz"] = float(self.coefficients.origin_coeffs[j, 2])
            if self.fit_directions:
                d[f"{key}_dx"] = float(self.coefficients.direction_coeffs[j, 0])
                d[f"{key}_dy"] = float(self.coefficients.direction_coeffs[j, 1])
                d[f"{key}_dz"] = float(self.coefficients.direction_coeffs[j, 2])
        return d

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.fit_directions:
            field = ZernikeRayField(K=self.K, config=self.config, coefficients=self.coefficients)
        else:
            field = ZernikeOriginField(
                K=self.K,
                config=self.config,
                coefficients=ZernikeOriginFieldCoefficients(coeffs=self.coefficients.origin_coeffs),
            )
        return field.ray(u, v)
