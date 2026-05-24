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
    """Configuration of a Zernike origin / ray field: domain, order and gauge.

    Attributes
    ----------
    image_size : tuple of 2 int
        Image ``(width, height)`` in pixels; defines the disk over which the
        Zernike polynomials are evaluated.
    max_order : int
        Maximum Zernike radial order; sets the number of modes.
    normalization : str
        Zernike normalisation scheme. Only ``"diagonal_disk"`` is supported:
        pixel coordinates are mapped to the unit disk via the image diagonal.
    enforce_transverse_gauge : bool
        If True, the origin field is projected transverse to the ray direction
        (``O . d = 0``) at evaluation time, removing the redundant longitudinal
        degree of freedom along each ray.
    """

    image_size: tuple[int, int]
    max_order: int = 4
    normalization: str = "diagonal_disk"
    enforce_transverse_gauge: bool = True

    def modes(self) -> tuple[ZernikeMode, ...]:
        """Return the Zernike mode descriptors selected by ``max_order``.

        Returns
        -------
        tuple of ZernikeMode
            One descriptor (radial order n, azimuthal order m, parity) per
            polynomial, in canonical order — this is the column order of the
            design matrix returned by :meth:`ZernikeOriginField.basis`.
        """
        return tuple(zernike_modes(int(self.max_order)))


@dataclass(frozen=True)
class ZernikeOriginFieldCoefficients:
    """Zernike coefficients of an origin field ``O(u, v)``.

    Attributes
    ----------
    coeffs : ndarray, shape (n_modes, 3)
        One 3-vector per Zernike mode, in millimetres; the origin field is
        ``basis(u, v) @ coeffs``.
    """

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
    """One named camera channel of a multi-camera Zernike rayfield.

    Attributes
    ----------
    name : str
        Channel identifier (e.g. ``"left"``, ``"cam0"``).
    field : ZernikeRayField
        The channel's fitted rayfield.
    """

    name: str
    field: ZernikeRayField


@dataclass(frozen=True)
class MultiCameraZernikeRayField:
    """Ordered collection of named Zernike rayfields, one per camera.

    Container for an N-camera calibration result: each camera contributes one
    :class:`ZernikeRayField`, addressed by name. Channel names must be unique
    and non-empty.

    Attributes
    ----------
    channels : tuple of ZernikeRayFieldChannel
        The per-camera rayfields, in a fixed order.
    """

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
    def from_fields(cls, fields: dict[str, ZernikeRayField]) -> MultiCameraZernikeRayField:
        """Build a multi-camera rayfield from a mapping of named rayfields.

        Parameters
        ----------
        fields : dict[str, ZernikeRayField]
            Rayfield per channel name; iteration order fixes the channel order.

        Returns
        -------
        MultiCameraZernikeRayField
        """
        return cls(
            tuple(ZernikeRayFieldChannel(name=name, field=field) for name, field in fields.items())
        )

    @classmethod
    def from_camera_configs(
        cls,
        intrinsics_by_channel: dict[str, np.ndarray],
        configs_by_channel: dict[str, ZernikeOriginFieldConfig],
    ) -> MultiCameraZernikeRayField:
        """Build zero-initialised rayfields for a set of cameras.

        Creates one :class:`ZernikeRayField` per channel from its intrinsics
        and Zernike configuration, with all coefficients at zero (i.e. the
        plain pinhole model). The two mappings must cover exactly the same
        set of channel names.

        Parameters
        ----------
        intrinsics_by_channel : dict[str, ndarray]
            3x3 pinhole intrinsic matrix per channel.
        configs_by_channel : dict[str, ZernikeOriginFieldConfig]
            Zernike field configuration per channel.

        Returns
        -------
        MultiCameraZernikeRayField

        Raises
        ------
        ValueError
            If the two mappings do not cover the same set of channel names.
        """
        intrinsics_names = set(intrinsics_by_channel)
        config_names = set(configs_by_channel)
        missing_configs = sorted(intrinsics_names - config_names)
        missing_intrinsics = sorted(config_names - intrinsics_names)
        if missing_configs or missing_intrinsics:
            details = []
            if missing_configs:
                details.append(f"missing configs for channels: {missing_configs}")
            if missing_intrinsics:
                details.append(f"missing intrinsics for channels: {missing_intrinsics}")
            raise ValueError("; ".join(details))
        fields = {
            name: ZernikeRayField(K=intrinsics, config=configs_by_channel[name])
            for name, intrinsics in intrinsics_by_channel.items()
        }
        return cls.from_fields(fields)

    @property
    def names(self) -> tuple[str, ...]:
        """Channel names in insertion order."""
        return tuple(channel.name for channel in self.channels)

    @property
    def n_channels(self) -> int:
        """Total number of channels."""
        return len(self.channels)

    def channel(self, name: str) -> ZernikeRayField:
        """Return the rayfield of the named channel.

        Parameters
        ----------
        name : str
            Channel name.

        Returns
        -------
        ZernikeRayField

        Raises
        ------
        KeyError
            If no channel has that name.
        """
        for channel in self.channels:
            if channel.name == name:
                return channel.field
        raise KeyError(name)

    def ray(self, name: str, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the 3D ray for a pixel in a named channel.

        Parameters
        ----------
        name : str
            Channel name.
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        origin, direction : ndarray, shape (N, 3)
            Ray origin (mm) and unit direction per pixel, in the channel frame.
        """
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
        """Build an origin-field rayfield from intrinsics and a Zernike config.

        Parameters
        ----------
        K : ndarray, shape (3, 3)
            Pinhole intrinsic matrix; defines the reference ray directions.
        config : ZernikeOriginFieldConfig
            Zernike domain, order and gauge settings.
        coefficients : ZernikeOriginFieldCoefficients, optional
            Origin-field coefficients; default to zeros (a pure pinhole model).

        Raises
        ------
        ValueError
            If the configuration normalisation is unsupported, or the
            coefficient array does not have shape ``(n_modes, 3)``.
        """
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
        # Rays are returned in the internal OpenCV frame
        # (u → +X, v → +Y, Z forward).  See core/conventions.py.
        self.frame_convention: str = "opencv_y_down"

    @property
    def coeffs(self) -> np.ndarray:
        """Origin-field Zernike coefficients, shape ``(n_modes, 3)``, in millimetres."""
        return self.coefficients.coeffs

    def basis(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Evaluate the Zernike design matrix at the given pixels.

        Pixel coordinates are mapped to the unit disk (via the image diagonal),
        then every Zernike mode is evaluated to form one column of the matrix.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates (broadcast against each other).

        Returns
        -------
        ndarray, shape (N, n_modes)
            Design matrix ``A``; ``A @ coeffs`` gives the field sampled at the
            ``N`` flattened pixels.

        Raises
        ------
        ValueError
            If the configured image is not larger than one pixel per axis.
        """
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
        """Return the unit pinhole ray directions at the given pixels.

        In this model the origin field carries the non-central behaviour;
        directions are the plain pinhole rays from the intrinsic matrix ``K``.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        ndarray, shape (N, 3)
            Unit ray directions.
        """
        return pinhole_ray_from_pixel(u, v, self.K).reshape(-1, 3)

    def raw_origin(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the Zernike origin field before the transverse gauge is applied.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        ndarray, shape (N, 3)
            Origin field ``basis(u, v) @ coeffs``, in millimetres.
        """
        return self.basis(u, v) @ self.coeffs

    def origin(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the origin field ``O(u, v)``, with the transverse gauge applied.

        When ``config.enforce_transverse_gauge`` is set, the raw Zernike origin
        is projected perpendicular to the ray direction (``O . d = 0``),
        removing the redundant longitudinal displacement along each ray.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        ndarray, shape (N, 3)
            Ray origin per pixel, in millimetres.
        """
        O_raw = self.raw_origin(u, v)
        if not self.config.enforce_transverse_gauge:
            return O_raw
        return _project_transverse(O_raw, self.direction(u, v))

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the rays ``(O(u, v), d(u, v))`` for the given pixels.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        origin : ndarray, shape (N, 3)
            Ray origin per pixel (mm), gauge-projected when configured.
        direction : ndarray, shape (N, 3)
            Unit pinhole ray direction per pixel.
        """
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
        """Build a rayfield with both a Zernike origin field and a direction field.

        Parameters
        ----------
        K : ndarray, shape (3, 3)
            Pinhole intrinsic matrix; defines the reference directions ``d0``.
        config : ZernikeOriginFieldConfig
            Zernike domain, order and gauge settings.
        coefficients : ZernikeRayFieldCoefficients, optional
            Origin and direction coefficients; default to zeros (pinhole model).

        Raises
        ------
        ValueError
            If the direction coefficients do not match the origin coefficient
            shape ``(n_modes, 3)``.
        """
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
        """Origin-field Zernike coefficients ``O(u, v)``, shape ``(n_modes, 3)``, in mm."""
        return self.rayfield_coefficients.origin_coeffs

    @property
    def direction_coeffs(self) -> np.ndarray:
        """Direction-perturbation Zernike coefficients, shape ``(n_modes, 3)``, dimensionless."""
        return self.rayfield_coefficients.direction_coeffs

    def direction_delta(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the transverse direction perturbation, before renormalisation.

        The raw Zernike direction field is projected perpendicular to the
        pinhole direction ``d0``, so the perturbation cannot change the ray
        scale — only its tilt.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        ndarray, shape (N, 3)
            Additive direction perturbation, transverse to ``d0``.
        """
        d0 = super().direction(u, v)
        return _project_transverse(self.basis(u, v) @ self.direction_coeffs, d0)

    def direction(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return the unit fitted ray directions.

        The pinhole direction ``d0`` plus the transverse Zernike perturbation,
        then renormalised: ``d = normalize(d0 + direction_delta(u, v))``.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        ndarray, shape (N, 3)
            Unit ray directions.
        """
        d0 = super().direction(u, v)
        d = d0 + self.direction_delta(u, v)
        return d / np.linalg.norm(d, axis=1, keepdims=True)


@dataclass(frozen=True)
class ZernikeCandidate:
    """Compact Zernike origin-field candidate for ray-space model selection.

    This wraps a :class:`ZernikeOriginField` into the
    :class:`~stereocomplex.physics.base.PhysicalRayFieldModel` protocol so it can
    compete alongside the physical candidates (pinhole, Brown-Conrady, CMO,
    polynomial surrogate) in
    :func:`~stereocomplex.physics.model_selection.select_physical_model_from_rayfield`.

    The candidate usually uses a **lower** max-order than the measured Zernike
    rayfield.  The raw coefficient count is ``n_modes * 6`` when directions are
    fitted and ``n_modes * 3`` for origin-only fitting.  The number therefore
    depends on the exact Zernike mode list and on whether direction coefficients
    are part of the candidate.

    By default both origin and direction Zernike fields are fitted
    (``fit_directions=True``, using :class:`ZernikeRayField`).  Set
    ``fit_directions=False`` for origin-only fitting with
    :class:`ZernikeOriginField` (3 scalar coefficients per mode instead of 6).

    .. note::

       The transverse gauge :math:`O(u,v)\\cdot d(u,v)=0` is enforced at
       ray-evaluation time.  Depending on the gauge and on whether pose
       parameters are included in a surrounding fit, the effective comparison
       count may differ from the raw coefficient count.  The ``n_parameters``
       property intentionally reports the raw coefficient count (6 or 3 per
       mode) for BIC consistency across candidates.
    """

    K: np.ndarray
    config: ZernikeOriginFieldConfig
    coefficients: ZernikeRayFieldCoefficients
    name: str = "zernike_compact"
    is_stereo_shared: bool = False
    fit_directions: bool = True

    @property
    def n_parameters(self) -> int:
        """Raw free-parameter count: ``n_modes * 6`` with directions, ``* 3`` without.

        This is the raw coefficient count. The transverse gauge removes one
        redundant degree of freedom per mode, so the *effective* count is
        slightly lower — see the class docstring.
        """
        n_modes = len(self.config.modes())
        return n_modes * 6 if self.fit_directions else n_modes * 3

    def parameter_vector(self) -> np.ndarray:
        """Pack the Zernike coefficients into a flat optimisation vector.

        Layout: the ``(n_modes, 3)`` origin coefficients flattened, followed —
        when ``fit_directions`` is set — by the ``(n_modes, 3)`` direction
        coefficients.

        Returns
        -------
        ndarray, shape (n_modes * 6,) or (n_modes * 3,)
        """
        if self.fit_directions:
            return np.concatenate(
                [
                    np.asarray(self.coefficients.origin_coeffs, dtype=np.float64).reshape(-1),
                    np.asarray(self.coefficients.direction_coeffs, dtype=np.float64).reshape(-1),
                ]
            )
        return np.asarray(self.coefficients.origin_coeffs, dtype=np.float64).reshape(-1)

    @classmethod
    def from_parameter_vector(cls, x: np.ndarray, **kwargs) -> ZernikeCandidate:
        """Rebuild a candidate from a flat parameter vector.

        Inverse of :meth:`parameter_vector`; see it for the layout.

        Parameters
        ----------
        x : ndarray
            Flat coefficient vector.
        **kwargs
            Must include ``config`` (ZernikeOriginFieldConfig) and ``K``
            (3x3 intrinsics); may include ``fit_directions`` (default True).

        Returns
        -------
        ZernikeCandidate

        Raises
        ------
        ValueError
            If the length of ``x`` does not match the mode count and the
            ``fit_directions`` setting.
        """
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
        """Return all coefficients as a flat ``{name: value}`` dictionary.

        Keys encode the mode index and its ``(n, m, kind)`` descriptor,
        suffixed by the component: ``_Ox/_Oy/_Oz`` for the origin field (mm)
        and, when directions are fitted, ``_dx/_dy/_dz`` for the direction
        perturbation (dimensionless).

        Returns
        -------
        dict[str, float]
        """
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
        """Compute the 3D ray ``(origin, direction)`` for the given pixels.

        Builds the underlying :class:`ZernikeRayField` (or
        :class:`ZernikeOriginField` when ``fit_directions`` is False) from the
        candidate coefficients and evaluates it.

        Parameters
        ----------
        u, v : ndarray
            Pixel coordinates.

        Returns
        -------
        origin, direction : ndarray, shape (N, 3)
            Ray origin (mm) and unit direction per pixel.
        """
        if self.fit_directions:
            field = ZernikeRayField(K=self.K, config=self.config, coefficients=self.coefficients)
        else:
            field = ZernikeOriginField(
                K=self.K,
                config=self.config,
                coefficients=ZernikeOriginFieldCoefficients(coeffs=self.coefficients.origin_coeffs),
            )
        return field.ray(u, v)
