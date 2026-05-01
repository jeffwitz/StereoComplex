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
    parameter_dict: dict[str, float]


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


def _aic_bic(rss: float, n: int, p: int) -> tuple[float, float]:
    # n = number of residual scalars (6 × n_pixels for two-plane residuals),
    # not independent pixel observations. Consistent across candidates, so the
    # relative BIC ordering is preserved despite differing from the textbook formula.
    rss_per_scalar = max(float(rss) / max(int(n), 1), 1e-30)
    aic = 2.0 * float(p) + float(n) * np.log(rss_per_scalar)
    bic = float(p) * np.log(max(float(n), 1.0)) + float(n) * np.log(rss_per_scalar)
    return float(aic), float(bic)


def default_physical_model_specs() -> list[PhysicalModelSpec]:
    """Return the standard candidate set used in the parallel-plate notebook."""
    return [
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


def _make_model(model_class: type, x: np.ndarray, K: np.ndarray, model_kwargs: dict[str, Any]) -> PhysicalRayFieldModel:
    if hasattr(model_class, "from_parameter_vector"):
        return model_class.from_parameter_vector(x, K=K, **model_kwargs)
    return model_class(K=K, **model_kwargs)


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
    else:
        support = np.asarray(support_pixels, dtype=np.float64).reshape(-1, 2)
    if support.size == 0:
        raise ValueError("support_pixels must not be empty")

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
        if full_grid_weight > 0:
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
    rms, median, p95 = _residual_norm_stats(combined)
    rss = float(np.sum(combined * combined))
    n_res = int(combined.size)
    aic, bic = _aic_bic(rss, n_res, p)
    # Slice combined to recover per-region unweighted residuals without extra
    # rayfield evaluations. Layout: [sw*support | fgw*full] (when fgw > 0).
    n_support_flat = support.shape[0] * 6
    support_rms = _residual_norm_stats(combined[:n_support_flat] / float(support_weight))[0]
    if full_grid_weight > 0:
        full_rms = _residual_norm_stats(combined[n_support_flat:] / float(full_grid_weight))[0]
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
        n_samples=int(support.shape[0] + (full_pixels.shape[0] if full_grid_weight > 0 else 0)),
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
) -> OpticalModelSelectionReport:
    """Fit candidate physical models and select the best in ray space."""
    specs = default_physical_model_specs() if candidate_specs is None else list(candidate_specs)
    results: list[PhysicalModelFitResult] = []
    warnings: list[str] = []
    for spec in specs:
        kwargs = dict(spec.model_kwargs or {})
        try:
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
