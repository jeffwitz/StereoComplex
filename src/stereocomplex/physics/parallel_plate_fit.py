from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereocomplex.synthetic.parallel_plate import (
    ParallelPlateSyntheticParams,
    parallel_plate_ray_from_pixel,
)


@dataclass(frozen=True)
class PinholeParallelPlateFitParams:
    """Physical pinhole + inclined parallel-plate parameters.

    Units are millimetres and degrees.  `d1_mm` is kept for ray generation, but
    is not fitted by default because changing it only moves the plate exit point
    along the emergent ray and therefore does not change the 3D line.
    """

    alpha_deg: float
    beta_deg: float
    thickness_mm: float
    eta: float = 1.5
    d1_mm: float = 80.0


@dataclass(frozen=True)
class ParallelPlateFromRayfieldFitResult:
    params: PinholeParallelPlateFitParams
    success: bool
    message: str
    rayfield_rms_support_mm: float
    rayfield_median_support_mm: float
    rayfield_p95_support_mm: float
    rayfield_rms_full_mm: float
    rayfield_median_full_mm: float
    rayfield_p95_full_mm: float
    n_support_samples: int
    n_full_samples: int
    parameter_error: dict[str, float] | None = None


class PinholeParallelPlateRayField:
    """Small adapter exposing a `.ray(u, v)` method for a fitted plate model."""

    def __init__(self, K: np.ndarray, params: PinholeParallelPlateFitParams):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.params = params

    def ray(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute ray (origin, direction) for a pixel through the inclined parallel plate model."""
        return pinhole_parallel_plate_ray_from_pixel(u, v, self.K, self.params)


class PinholeParallelPlateModel(PinholeParallelPlateRayField):
    """Physical pinhole + inclined parallel-plate model-selection candidate."""

    name = "pinhole_parallel_plate"

    @property
    def n_parameters(self) -> int:
        """Number of free parameters (2 for plate tilt, 0 for pinhole baseline)."""
        return 3

    def parameter_vector(self) -> np.ndarray:
        """Pack model parameters into a flat vector for optimisation."""
        return np.array(
            [self.params.alpha_deg, self.params.beta_deg, self.params.thickness_mm],
            dtype=np.float64,
        )

    @classmethod
    def from_parameter_vector(cls, x: np.ndarray, **kwargs) -> PinholeParallelPlateModel:
        """Reconstruct model from a parameter vector. K must be passed via kwargs."""
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size != 3:
            raise ValueError("PinholeParallelPlateModel expects three parameters")
        params = PinholeParallelPlateFitParams(
            alpha_deg=float(arr[0]),
            beta_deg=float(arr[1]),
            thickness_mm=float(arr[2]),
            eta=float(kwargs.get("eta", 1.5)),
            d1_mm=float(kwargs.get("d1_mm", 80.0)),
        )
        return cls(np.asarray(kwargs["K"], dtype=np.float64).reshape(3, 3), params)

    def parameter_dict(self) -> dict[str, float]:
        """Model parameters as a dict keyed by coefficient name."""
        return {
            "alpha_deg": float(self.params.alpha_deg),
            "beta_deg": float(self.params.beta_deg),
            "thickness_mm": float(self.params.thickness_mm),
            "eta": float(self.params.eta),
            "d1_mm": float(self.params.d1_mm),
        }


def _as_synthetic_params(params: PinholeParallelPlateFitParams) -> ParallelPlateSyntheticParams:
    return ParallelPlateSyntheticParams(
        eta=float(params.eta),
        thickness=float(params.thickness_mm),
        alpha_deg=float(params.alpha_deg),
        beta_deg=float(params.beta_deg),
        d1=float(params.d1_mm),
    )


def pinhole_parallel_plate_ray_from_pixel(
    u: np.ndarray,
    v: np.ndarray,
    K: np.ndarray,
    params: PinholeParallelPlateFitParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 3-D ray (origin, direction) for a pixel through a pinhole
    camera with an inclined parallel plate in front of the sensor.

    The plate shifts the apparent ray origin while preserving the direction.

    Parameters
    ----------
    u : ndarray
        Pixel x-coordinates.
    v : ndarray
        Pixel y-coordinates.
    K : ndarray, shape (3, 3)
        Camera matrix.
    params : PinholeParallelPlateFitParams
        Plate geometry (normal, thickness, refractive index, distance).

    Returns
    -------
    (origin, direction) : tuple of ndarray
        Ray origins in mm and unit directions, each shape (N, 3).
    """
    return parallel_plate_ray_from_pixel(u, v, K, _as_synthetic_params(params))


def intersect_ray_with_z_plane(origin_points: np.ndarray, d: np.ndarray, z: float) -> np.ndarray:
    """Intersect one or more rays with a horizontal z-plane at a given depth.

    Parameters
    ----------
    origin_points : ndarray, shape (N, 3)
        Ray origins in world coordinates, in millimetres.
    d : ndarray, shape (N, 3)
        Ray directions (unit vectors).
    z : float
        Z-coordinate of the target plane in millimetres.

    Returns
    -------
    ndarray, shape (N, 3)
        Intersection points in millimetres.

    Raises
    ------
    ValueError
        If any ray direction has a z-component smaller than 1e-12 in
        absolute value (ray is parallel to the plane).
    """
    origin = np.asarray(origin_points, dtype=np.float64).reshape(-1, 3)
    direction = np.asarray(d, dtype=np.float64).reshape(-1, 3)
    denom = direction[:, 2]
    if np.any(np.abs(denom) < 1e-12):
        raise ValueError("ray is parallel to z plane")
    lam = (float(z) - origin[:, 2]) / denom
    return origin + lam[:, None] * direction


def _eval_rayfield(
    field,
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(field, "ray"):
        origins, d = field.ray(u, v)
    elif callable(field):
        origins, d = field(u, v)
    else:
        raise TypeError("rayfield must expose .ray(u, v) or be callable")
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    d = np.asarray(d, dtype=np.float64).reshape(-1, 3)
    return origins, d / np.linalg.norm(d, axis=1, keepdims=True)


def _grid_pixels(image_size: tuple[int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    nx, ny = grid_shape
    u = np.linspace(0.0, float(width - 1), int(nx))
    v = np.linspace(0.0, float(height - 1), int(ny))
    uu, vv = np.meshgrid(u, v)
    return np.column_stack([uu.reshape(-1), vv.reshape(-1)])


def rayfield_two_plane_residuals(
    field_a,
    field_b,
    pixels: np.ndarray,
    z_planes: tuple[float, float] = (100.0, 1000.0),
) -> np.ndarray:
    """Compare two rayfields by intersections with two reference z-planes.

    Both *field_a* and *field_b* must declare ``frame_convention =
    "opencv_y_down"`` (the internal StereoComplex convention).  A
    ``ValueError`` is raised if either declares a different convention.
    """
    from stereocomplex.core.conventions import check_frame_convention

    check_frame_convention(field_a, field_b, label="rayfield_two_plane_residuals")
    px = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    Oa, da = _eval_rayfield(field_a, px[:, 0], px[:, 1])
    Ob, db = _eval_rayfield(field_b, px[:, 0], px[:, 1])
    blocks: list[np.ndarray] = []
    for z in z_planes:
        Aa = intersect_ray_with_z_plane(Oa, da, z)
        Ab = intersect_ray_with_z_plane(Ob, db, z)
        blocks.append(Aa - Ab)
    return np.concatenate(blocks, axis=1).reshape(-1)


def _residual_norm_stats(residuals: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 6)
    # The 6-vector stores 3D errors at two planes.  The scalar rayfield error is
    # the Euclidean norm of that two-plane discrepancy.
    norms = np.linalg.norm(r, axis=1)
    return (
        float(np.sqrt(np.mean(norms**2))),
        float(np.median(norms)),
        float(np.percentile(norms, 95)),
    )


def _param_vector(params: PinholeParallelPlateFitParams, *, fit_eta: bool) -> np.ndarray:
    base = [float(params.alpha_deg), float(params.beta_deg), float(params.thickness_mm)]
    if fit_eta:
        base.append(float(params.eta))
    return np.asarray(base, dtype=np.float64)


def _params_from_vector(
    x: np.ndarray, *, eta: float, d1_mm: float, fit_eta: bool,
) -> PinholeParallelPlateFitParams:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    eta_val = float(arr[3]) if fit_eta else float(eta)
    return PinholeParallelPlateFitParams(
        alpha_deg=float(arr[0]),
        beta_deg=float(arr[1]),
        thickness_mm=float(arr[2]),
        eta=eta_val,
        d1_mm=float(d1_mm),
    )


def _parameter_error(
    fitted: PinholeParallelPlateFitParams,
    oracle: PinholeParallelPlateFitParams | ParallelPlateSyntheticParams | None,
) -> dict[str, float] | None:
    if oracle is None:
        return None
    if isinstance(oracle, ParallelPlateSyntheticParams):
        truth = PinholeParallelPlateFitParams(
            alpha_deg=oracle.alpha_deg,
            beta_deg=oracle.beta_deg,
            thickness_mm=oracle.thickness,
            eta=oracle.eta,
            d1_mm=oracle.d1,
        )
    else:
        truth = oracle
    return {
        "alpha_deg": float(fitted.alpha_deg - truth.alpha_deg),
        "beta_deg": float(fitted.beta_deg - truth.beta_deg),
        "thickness_mm": float(fitted.thickness_mm - truth.thickness_mm),
        "eta": float(fitted.eta - truth.eta),
    }


def fit_parallel_plate_to_zernike_rayfield(
    zernike_field,
    K: np.ndarray,
    image_size: tuple[int, int],
    initial_params: PinholeParallelPlateFitParams | None = None,
    eta: float = 1.5,
    z_planes: tuple[float, float] = (100.0, 1000.0),
    grid_shape: tuple[int, int] = (25, 19),
    support_pixels: np.ndarray | None = None,
    support_weight: float = 1.0,
    full_grid_weight: float = 0.25,
    fit_eta: bool = False,
    robust_loss: str = "huber",
    oracle_params: PinholeParallelPlateFitParams | ParallelPlateSyntheticParams | None = None,
) -> ParallelPlateFromRayfieldFitResult:
    """Fit a compact pinhole + plate model to an already measured rayfield.

    The target rayfield is treated as a geometric observable.  The residual is
    evaluated in ray space by intersecting both rayfields with two z-planes; raw
    ray origins are never compared directly.
    """
    from scipy.optimize import least_squares  # type: ignore

    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    full_pixels = _grid_pixels(image_size, grid_shape)
    if support_pixels is None:
        support = full_pixels
    else:
        support = np.asarray(support_pixels, dtype=np.float64).reshape(-1, 2)
    if support.size == 0:
        raise ValueError("support_pixels must not be empty")

    if initial_params is None:
        initial_params = PinholeParallelPlateFitParams(
            alpha_deg=0.0,
            beta_deg=0.0,
            thickness_mm=8.0,
            eta=float(eta),
            d1_mm=80.0,
        )

    def plate_field(x: np.ndarray) -> PinholeParallelPlateRayField:
        """Factory: extract the fitted ParallelPlateRayField from a PinholeParallelPlateModel."""
        params = _params_from_vector(x, eta=eta, d1_mm=initial_params.d1_mm, fit_eta=fit_eta)
        return PinholeParallelPlateRayField(K_arr, params)

    def fun(x: np.ndarray) -> np.ndarray:
        """Objective function for plate parameter optimisation (ray-space residual)."""
        residual_blocks = [
            float(support_weight)
            * rayfield_two_plane_residuals(
                zernike_field, plate_field(x), support, z_planes=z_planes
            )
        ]
        if full_grid_weight > 0:
            residual_blocks.append(
                float(full_grid_weight)
                * rayfield_two_plane_residuals(
                    zernike_field, plate_field(x), full_pixels, z_planes=z_planes
                )
            )
        return np.concatenate(residual_blocks)

    x0 = _param_vector(initial_params, fit_eta=fit_eta)
    lower = [-30.0, -30.0, 0.0]
    upper = [30.0, 30.0, 50.0]
    if fit_eta:
        lower.append(1.0001)
        upper.append(2.0)

    sol = least_squares(
        fun,
        x0=x0,
        bounds=(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)),
        loss=robust_loss,
        f_scale=1.0,
        max_nfev=300,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    fitted_params = _params_from_vector(sol.x, eta=eta, d1_mm=initial_params.d1_mm, fit_eta=fit_eta)
    fitted_field = PinholeParallelPlateRayField(K_arr, fitted_params)
    support_res = rayfield_two_plane_residuals(
        zernike_field, fitted_field, support, z_planes=z_planes
    )
    full_res = rayfield_two_plane_residuals(
        zernike_field, fitted_field, full_pixels, z_planes=z_planes
    )
    support_stats = _residual_norm_stats(support_res)
    full_stats = _residual_norm_stats(full_res)
    return ParallelPlateFromRayfieldFitResult(
        params=fitted_params,
        success=bool(sol.success),
        message=str(sol.message),
        rayfield_rms_support_mm=support_stats[0],
        rayfield_median_support_mm=support_stats[1],
        rayfield_p95_support_mm=support_stats[2],
        rayfield_rms_full_mm=full_stats[0],
        rayfield_median_full_mm=full_stats[1],
        rayfield_p95_full_mm=full_stats[2],
        n_support_samples=int(support.shape[0]),
        n_full_samples=int(full_pixels.shape[0]),
        parameter_error=_parameter_error(fitted_params, oracle_params),
    )


__all__ = [
    "ParallelPlateFromRayfieldFitResult",
    "PinholeParallelPlateFitParams",
    "PinholeParallelPlateModel",
    "PinholeParallelPlateRayField",
    "fit_parallel_plate_to_zernike_rayfield",
    "intersect_ray_with_z_plane",
    "pinhole_parallel_plate_ray_from_pixel",
    "rayfield_two_plane_residuals",
]
