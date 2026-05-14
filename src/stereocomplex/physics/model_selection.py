from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stereocomplex.physics.base import PhysicalRayFieldModel
from stereocomplex.physics.central_models import CentralBrownConradyModel, CentralPinholeModel
from stereocomplex.physics.parallel_plate_fit import (
    PinholeParallelPlateModel,
    rayfield_two_plane_residuals,
)


@dataclass(frozen=True)
class PhysicalModelSpec:
    """Specification for one physical candidate to fit or score."""

    name: str
    model_class: type
    initial_parameters: np.ndarray
    bounds: tuple[np.ndarray, np.ndarray] | None = None
    model_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class PhysicalModelFitResult:
    model_name: str
    model: PhysicalRayFieldModel
    success: bool
    message: str
    n_parameters: int
    n_samples: int
    n_residual_scalars: int
    rss: float
    rms_mm: float
    median_mm: float
    p95_mm: float
    support_rms_mm: float
    full_grid_rms_mm: float
    aic: float
    bic: float
    parameter_vector: np.ndarray
    parameter_dict: dict[str, Any]


@dataclass(frozen=True)
class OpticalModelSelectionReport:
    target_name: str
    candidates: list[PhysicalModelFitResult]
    best_by_aic: str
    best_by_bic: str
    best_by_rms: str
    warnings: list[str]

    def rows(self) -> list[dict[str, float | int | str | bool]]:
        return [
            {
                "model": c.model_name,
                "parameters": c.n_parameters,
                "rms_mm": c.rms_mm,
                "support_rms_mm": c.support_rms_mm,
                "full_grid_rms_mm": c.full_grid_rms_mm,
                "bic": c.bic,
                "aic": c.aic,
                "selected_bic": c.model_name == self.best_by_bic,
            }
            for c in self.candidates
        ]


def reprojection_guard_penalty(
    px_rms: float,
    *,
    threshold_px: float = 1.5,
    n_pixel_observations: int,
    hard_penalty: float = 1.0e6,
    alpha: float = 1.0,
) -> float:
    """Penalty term for models exceeding a pixel reprojection threshold.

    Returns 0 if ``px_rms <= threshold_px``.  Otherwise returns a hard
    barrier plus a log-ratio term that ranks non-usable models among
    themselves::

        penalty = hard_penalty + alpha * n_obs * log((px_rms / threshold)^2)

    This is designed to be added to a ray-space BIC to produce an
    operational ``bic_usable`` that enforces a usability constraint.
    """
    if px_rms <= threshold_px:
        return 0.0
    ratio2 = (float(px_rms) / float(threshold_px)) ** 2
    return float(hard_penalty) + float(alpha) * float(n_pixel_observations) * np.log(max(ratio2, 1.0 + 1e-10))


def _grid_pixels(image_size: tuple[int, int], grid_shape: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    nx, ny = grid_shape
    u = np.linspace(0.0, float(width - 1), int(nx))
    v = np.linspace(0.0, float(height - 1), int(ny))
    uu, vv = np.meshgrid(u, v)
    return np.column_stack([uu.reshape(-1), vv.reshape(-1)])


def _residual_norm_stats(residuals: np.ndarray) -> tuple[float, float, float]:
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 6)
    norms = np.linalg.norm(r, axis=1)
    return (
        float(np.sqrt(np.mean(norms**2))),
        float(np.median(norms)),
        float(np.percentile(norms, 95)),
    )


def _aic_bic(rss: float, n_residual_scalars: int, n_observations: int, p: int) -> tuple[float, float]:
    """Return Gaussian AIC/BIC for the weighted ray-space objective.

    The likelihood term uses the scalar residual count because the two-plane
    ray distance contributes six metric residual components per sampled pixel.
    The BIC parameter penalty uses the number of sampled pixel observations,
    which is the defensible independent-sample count for model selection.
    """
    rss_per_scalar = max(float(rss) / max(int(n_residual_scalars), 1), 1e-30)
    n_scalar = float(max(int(n_residual_scalars), 1))
    n_obs = float(max(int(n_observations), 1))
    aic = 2.0 * float(p) + n_scalar * np.log(rss_per_scalar)
    bic = float(p) * np.log(n_obs) + n_scalar * np.log(rss_per_scalar)
    return float(aic), float(bic)


def default_physical_model_specs(*, include_telecentric: bool = False) -> list[PhysicalModelSpec]:
    """Return the standard candidate set used in the parallel-plate notebook.

    Set *include_telecentric* to True to add the quasi-telecentric CMO candidate.
    """
    specs = [
        PhysicalModelSpec(
            name="central_pinhole",
            model_class=CentralPinholeModel,
            initial_parameters=np.zeros(0, dtype=np.float64),
        ),
        PhysicalModelSpec(
            name="central_brown_conrady",
            model_class=CentralBrownConradyModel,
            initial_parameters=np.zeros(5, dtype=np.float64),
            bounds=(
                np.array([-1.0, -1.0, -0.1, -0.1, -1.0], dtype=np.float64),
                np.array([1.0, 1.0, 0.1, 0.1, 1.0], dtype=np.float64),
            ),
        ),
        PhysicalModelSpec(
            name="pinhole_parallel_plate",
            model_class=PinholeParallelPlateModel,
            initial_parameters=np.array([0.0, 0.0, 8.0], dtype=np.float64),
            bounds=(
                np.array([-30.0, -30.0, 0.0], dtype=np.float64),
                np.array([30.0, 30.0, 50.0], dtype=np.float64),
            ),
            model_kwargs={"eta": 1.5, "d1_mm": 80.0},
        ),
    ]
    if include_telecentric:
        from stereocomplex.physics.cmo_physical import CMOTelecentricChannelModel  # noqa: PLC0415
        specs.append(
            PhysicalModelSpec(
                name="cmo_telecentric",
                model_class=CMOTelecentricChannelModel,
                initial_parameters=np.array(
                    [62.0, 65.0, 25.0, 1024., 1024., 62.0, 0.2, 0.06, 0.0, 0.0],
                    dtype=np.float64,
                ),
                model_kwargs={"pixel_pitch_mm": 0.0055},
            ),
        )
    return specs


def _make_model(model_class: type, x: np.ndarray, K: np.ndarray, model_kwargs: dict[str, Any]) -> PhysicalRayFieldModel:
    if hasattr(model_class, "from_parameter_vector"):
        return model_class.from_parameter_vector(x, K=K, **model_kwargs)
    return model_class(K=K, **model_kwargs)


def _as_stereo_selection_result(
    *,
    name: str,
    left: PhysicalModelFitResult,
    right: PhysicalModelFitResult,
) -> PhysicalModelFitResult:
    rss = float(left.rss + right.rss)
    n_res = int(left.n_residual_scalars + right.n_residual_scalars)
    n_samples = int(left.n_samples + right.n_samples)
    p = int(left.n_parameters + right.n_parameters)
    aic, bic = _aic_bic(rss, n_res, n_samples, p)
    rms = float(np.sqrt(0.5 * (left.rms_mm**2 + right.rms_mm**2)))
    support_rms = float(np.sqrt(0.5 * (left.support_rms_mm**2 + right.support_rms_mm**2)))
    full_rms = float(np.sqrt(0.5 * (left.full_grid_rms_mm**2 + right.full_grid_rms_mm**2)))
    median = float(0.5 * (left.median_mm + right.median_mm))
    p95 = float(max(left.p95_mm, right.p95_mm))
    return PhysicalModelFitResult(
        model_name=name,
        model=left.model,
        success=bool(left.success and right.success),
        message=f"left: {left.message}; right: {right.message}",
        n_parameters=p,
        n_samples=n_samples,
        n_residual_scalars=n_res,
        rss=rss,
        rms_mm=rms,
        median_mm=median,
        p95_mm=p95,
        support_rms_mm=support_rms,
        full_grid_rms_mm=full_rms,
        aic=aic,
        bic=bic,
        parameter_vector=np.r_[left.parameter_vector, right.parameter_vector],
        parameter_dict={"left": left.parameter_dict, "right": right.parameter_dict},
    )


def _as_shared_cmo_selection_result(name: str, result: Any) -> PhysicalModelFitResult:
    return PhysicalModelFitResult(
        model_name=name,
        model=result.model,
        success=bool(result.success),
        message=str(result.message),
        n_parameters=int(result.n_parameters),
        n_samples=int(result.n_samples),
        n_residual_scalars=int(result.n_residual_scalars),
        rss=float(result.rss),
        rms_mm=float(result.rms_mm),
        median_mm=float("nan"),
        p95_mm=float("nan"),
        support_rms_mm=float(result.rms_mm),
        full_grid_rms_mm=float(result.rms_mm),
        aic=float(result.aic),
        bic=float(result.bic),
        parameter_vector=np.asarray(result.parameter_vector, dtype=np.float64),
        parameter_dict=result.parameter_dict,
    )


def fit_physical_model_to_rayfield(
    model_class: type,
    target_field,
    K: np.ndarray,
    image_size: tuple[int, int],
    initial_parameters: np.ndarray | None = None,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    z_planes: tuple[float, float] = (100.0, 1000.0),
    grid_shape: tuple[int, int] = (25, 19),
    support_pixels: np.ndarray | None = None,
    support_weight: float = 1.0,
    full_grid_weight: float = 0.25,
    robust_loss: str = "huber",
    max_nfev: int = 2000,
    name: str | None = None,
    **model_kwargs,
) -> PhysicalModelFitResult:
    """Fit or score a physical model against a target rayfield in ray space."""
    from scipy.optimize import least_squares  # type: ignore

    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    full_pixels = _grid_pixels(image_size, grid_shape)
    if support_pixels is None:
        support = full_pixels
        include_full_grid = False
    else:
        support = np.asarray(support_pixels, dtype=np.float64).reshape(-1, 2)
        include_full_grid = full_grid_weight > 0
    if support.size == 0:
        raise ValueError("support_pixels must not be empty")
    if support_weight <= 0:
        raise ValueError("support_weight must be strictly positive")
    if full_grid_weight < 0:
        raise ValueError("full_grid_weight must be non-negative")

    if initial_parameters is None:
        initial_parameters = np.zeros(0, dtype=np.float64)
    x0 = np.asarray(initial_parameters, dtype=np.float64).reshape(-1)
    p = int(x0.size)

    def model_at(x: np.ndarray) -> PhysicalRayFieldModel:
        return _make_model(model_class, x, K_arr, model_kwargs)

    def combined_residuals(x: np.ndarray) -> np.ndarray:
        model = model_at(x)
        blocks = [
            float(support_weight) * rayfield_two_plane_residuals(target_field, model, support, z_planes=z_planes),
        ]
        if include_full_grid:
            blocks.append(
                float(full_grid_weight)
                * rayfield_two_plane_residuals(target_field, model, full_pixels, z_planes=z_planes)
            )
        return np.concatenate(blocks)

    if p == 0:
        sol_x = x0
        success = True
        message = "zero-parameter model scored without optimization"
    else:
        if bounds is None:
            bounds = (-np.inf * np.ones_like(x0), np.inf * np.ones_like(x0))
        sol = least_squares(
            combined_residuals,
            x0=x0,
            bounds=bounds,
            loss=robust_loss,
            f_scale=1.0,
            max_nfev=int(max_nfev),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        sol_x = np.asarray(sol.x, dtype=np.float64)
        success = bool(sol.success)
        message = str(sol.message)

    fitted_model = model_at(sol_x)
    combined = combined_residuals(sol_x)
    rss = float(np.sum(combined * combined))
    n_res = int(combined.size)
    n_samples = int(support.shape[0] + (full_pixels.shape[0] if include_full_grid else 0))
    aic, bic = _aic_bic(rss, n_res, n_samples, p)
    # Slice combined to recover per-region unweighted residuals without extra
    # rayfield evaluations. Layout: [sw*support | fgw*full] (when fgw > 0).
    n_support_flat = support.shape[0] * 6
    support_stats = _residual_norm_stats(combined[:n_support_flat] / float(support_weight))
    rms, median, p95 = support_stats
    support_rms = support_stats[0]
    if include_full_grid:
        full_rms = _residual_norm_stats(combined[n_support_flat:] / float(full_grid_weight))[0]
    elif support_pixels is None:
        full_rms = support_rms
    else:
        full_rms = _residual_norm_stats(
            rayfield_two_plane_residuals(target_field, fitted_model, full_pixels, z_planes=z_planes)
        )[0]
    parameter_dict = fitted_model.parameter_dict() if hasattr(fitted_model, "parameter_dict") else {}
    model_name = name if name is not None else str(getattr(fitted_model, "name", model_class.__name__))
    return PhysicalModelFitResult(
        model_name=model_name,
        model=fitted_model,
        success=success,
        message=message,
        n_parameters=p,
        n_samples=n_samples,
        n_residual_scalars=n_res,
        rss=rss,
        rms_mm=rms,
        median_mm=median,
        p95_mm=p95,
        support_rms_mm=support_rms,
        full_grid_rms_mm=full_rms,
        aic=aic,
        bic=bic,
        parameter_vector=sol_x,
        parameter_dict=parameter_dict,
    )


def select_physical_model_from_rayfield(
    target_field,
    candidate_specs: list[PhysicalModelSpec] | None,
    K: np.ndarray,
    image_size: tuple[int, int],
    z_planes: tuple[float, float] = (100.0, 1000.0),
    grid_shape: tuple[int, int] = (25, 19),
    support_pixels: np.ndarray | None = None,
    support_weight: float = 1.0,
    full_grid_weight: float = 0.25,
    max_nfev: int = 2000,
    target_right=None,
    K_right: np.ndarray | None = None,
    support_pixels_right: np.ndarray | None = None,
) -> OpticalModelSelectionReport:
    """Fit candidate physical models and select the best in ray space.

    With only ``target_field`` this keeps the historical single-channel
    behavior. When ``target_right`` is provided, non-shared candidates are fitted
    independently to both channels and aggregated, while stereo-shared
    candidates such as ``CMOPhysicalStereoModel`` are fitted once with their
    shared parameter vector.
    """
    from stereocomplex.physics.cmo_physical import (
        CMOPhysicalStereoModel,
        fit_cmo_physical_stereo_model_to_rayfields,
    )

    specs = default_physical_model_specs() if candidate_specs is None else list(candidate_specs)
    results: list[PhysicalModelFitResult] = []
    warnings: list[str] = []
    K_r = np.asarray(K if K_right is None else K_right, dtype=np.float64).reshape(3, 3)
    stereo_mode = target_right is not None
    for spec in specs:
        kwargs = dict(spec.model_kwargs or {})
        try:
            if stereo_mode and bool(getattr(spec.model_class, "is_stereo_shared", False)):
                pixel_pitch = kwargs.pop("pixel_pitch_mm", None)
                if pixel_pitch is None:
                    raise ValueError("pixel_pitch_mm is required for stereo-shared CMO selection")

                if spec.model_class is CMOPhysicalStereoModel or spec.model_class.__name__ == "CMOPhysicalStereoModel":
                    result = _as_shared_cmo_selection_result(
                        spec.name,
                        fit_cmo_physical_stereo_model_to_rayfields(
                            left_field=target_field,
                            right_field=target_right,
                            image_size=image_size,
                            initial_parameters=spec.initial_parameters,
                            bounds=spec.bounds,
                            pixel_pitch_mm=float(pixel_pitch),
                            z_planes=z_planes,
                        grid_shape=grid_shape,
                        support_pixels_left=support_pixels,
                        support_pixels_right=support_pixels_right,
                        support_weight=support_weight,
                        full_grid_weight=full_grid_weight,
                        max_nfev=max_nfev,
                    ),
                )
                elif spec.model_class.__name__ in ("CMOTelecentricStereoModel", "CMOTelecentricChannelModel"):
                    from stereocomplex.physics.cmo_physical import fit_cmo_telecentric_model_to_rayfields  # noqa: PLC0415
                    result = _as_shared_cmo_selection_result(
                        spec.name,
                        fit_cmo_telecentric_model_to_rayfields(
                            left_field=target_field,
                            right_field=target_right,
                            image_size=image_size,
                            initial_parameters=spec.initial_parameters,
                            pixel_pitch_mm=float(pixel_pitch),
                            z_planes=z_planes,
                            grid_shape=grid_shape,
                            full_grid_weight=full_grid_weight,
                            max_nfev=max_nfev,
                        ),
                    )
                else:
                    raise NotImplementedError(f"stereo-shared dispatch is not implemented for {spec.model_class}")
            elif stereo_mode:
                left = fit_physical_model_to_rayfield(
                    model_class=spec.model_class,
                    target_field=target_field,
                    K=K,
                    image_size=image_size,
                    initial_parameters=spec.initial_parameters,
                    bounds=spec.bounds,
                    z_planes=z_planes,
                    grid_shape=grid_shape,
                    support_pixels=support_pixels,
                    support_weight=support_weight,
                    full_grid_weight=full_grid_weight,
                    max_nfev=max_nfev,
                    name=f"{spec.name}_left",
                    **kwargs,
                )
                right = fit_physical_model_to_rayfield(
                    model_class=spec.model_class,
                    target_field=target_right,
                    K=K_r,
                    image_size=image_size,
                    initial_parameters=spec.initial_parameters,
                    bounds=spec.bounds,
                    z_planes=z_planes,
                    grid_shape=grid_shape,
                    support_pixels=support_pixels_right,
                    support_weight=support_weight,
                    full_grid_weight=full_grid_weight,
                    max_nfev=max_nfev,
                    name=f"{spec.name}_right",
                    **kwargs,
                )
                result = _as_stereo_selection_result(name=spec.name, left=left, right=right)
            else:
                result = fit_physical_model_to_rayfield(
                    model_class=spec.model_class,
                    target_field=target_field,
                    K=K,
                    image_size=image_size,
                    initial_parameters=spec.initial_parameters,
                    bounds=spec.bounds,
                    z_planes=z_planes,
                    grid_shape=grid_shape,
                    support_pixels=support_pixels,
                    support_weight=support_weight,
                    full_grid_weight=full_grid_weight,
                    max_nfev=max_nfev,
                    name=spec.name,
                    **kwargs,
                )
        except Exception as exc:  # pragma: no cover - surfaced in report for users
            warnings.append(f"{spec.name} failed: {exc}")
            continue
        results.append(result)
    if not results:
        raise RuntimeError("all physical candidate fits failed")
    return OpticalModelSelectionReport(
        target_name=str(getattr(target_field, "name", target_field.__class__.__name__)),
        candidates=results,
        best_by_aic=min(results, key=lambda r: r.aic).model_name,
        best_by_bic=min(results, key=lambda r: r.bic).model_name,
        best_by_rms=min(results, key=lambda r: r.rms_mm).model_name,
        warnings=warnings,
    )


__all__ = [
    "OpticalModelSelectionReport",
    "PhysicalModelFitResult",
    "PhysicalModelSpec",
    "default_physical_model_specs",
    "fit_physical_model_to_rayfield",
    "select_physical_model_from_rayfield",
]
