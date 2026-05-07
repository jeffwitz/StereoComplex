"""Geometrically constrained Common Main Objective stereo model.

This module implements a compact paraxial CMO model for ray-space model
selection and bundle-adjustment experiments. It is intentionally distinct from
the polynomial non-central surrogate in :mod:`stereocomplex.physics.cmo`: the
two channels share a main-objective geometry, and their chief rays converge at
the working plane instead of being represented by independent origins.

Scientific background:

* Olympus US 7,564,619 describes a CMO architecture with decentered aperture
  stops near the image-side focal plane of the common main objective.
* Wang et al., Optics and Lasers in Engineering 134 (2020), describe
  common-main-objective stereo microscopes as having parallel optical paths and
  image planes parallel to the focal plane.
* Schreier, Garcia and Sutton, Experimental Mechanics 44(3), 278-288 (2004),
  and Pan, Wang and Cheng, Optics Express 22(15), 18373-18387 (2014), discuss
  calibration and 3D measurement with stereo light microscopes.

The implementation is a first-order ray model, not a full lens-design model.
It is meant to test whether a measured rayfield is compatible with a compact
CMO-like shared-rig parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares  # type: ignore

from stereocomplex.physics.central_models import undistort_brown_normalized
from stereocomplex.physics.parallel_plate_fit import rayfield_two_plane_residuals

Array = np.ndarray


def _normalize(v: Array, eps: float = 1e-15) -> Array:
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(n, float(eps))


def _roty(angle_rad: float) -> Array:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _grid_pixels(image_size: tuple[int, int], grid_shape: tuple[int, int]) -> Array:
    width, height = image_size
    nx, ny = grid_shape
    u = np.linspace(0.0, float(width - 1), int(nx))
    v = np.linspace(0.0, float(height - 1), int(ny))
    uu, vv = np.meshgrid(u, v, indexing="xy")
    return np.column_stack([uu.reshape(-1), vv.reshape(-1)])


def _ray_rms(residuals: Array) -> float:
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 6)
    return float(np.sqrt(np.mean(np.linalg.norm(r, axis=1) ** 2)))


def _aic_bic(rss: float, n_residual_scalars: int, n_observations: int, p: int) -> tuple[float, float]:
    rss_per_scalar = max(float(rss) / max(int(n_residual_scalars), 1), 1e-30)
    n_scalar = float(max(int(n_residual_scalars), 1))
    n_obs = float(max(int(n_observations), 1))
    return (
        float(2.0 * float(p) + n_scalar * np.log(rss_per_scalar)),
        float(float(p) * np.log(n_obs) + n_scalar * np.log(rss_per_scalar)),
    )


@dataclass(frozen=True)
class CMOPhysicalStereoModel:
    """Shared-rig paraxial CMO stereo model.

    The model uses a common main objective with two decentered effective
    sub-pupils separated by ``b_mm``. For the central pixel, the left and right
    chief rays converge to the optical axis at ``working_distance_mm``. The
    chief-ray angle is controlled by ``b_mm / (2 f_obj_mm)`` because the
    entrance-pupil plane is placed one objective focal length before the
    working plane.

    Sensor coordinates are converted to object-plane coordinates through the
    tube-lens focal length and pixel pitch. The per-channel distortion is an
    effective direction distortion ``D_c``, parameterized by five
    Brown-Conrady-like coefficients applied to normalized angular coordinates.
    It absorbs residual aberrations of the tube lens and main objective; it is
    not a derivation from one specific physical aberration model.
    """

    f_obj_mm: float
    working_distance_mm: float
    b_mm: float
    f_tube_mm: float
    cx_principal_px: float
    cy_principal_px: float
    pixel_pitch_mm: float
    theta_axis_tilt_rad: float = 0.0
    distortion_left: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    distortion_right: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    image_size: tuple[int, int] | None = None
    share_principal_point: bool = True
    delta_cx_diff_px: float = 0.0
    delta_cy_diff_px: float = 0.0
    name: str = "cmo_physical_stereo"
    is_stereo_shared: bool = True

    @property
    def n_parameters(self) -> int:
        return 17 if self.share_principal_point else 19

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs) -> "CMOPhysicalStereoModel":
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if "pixel_pitch_mm" not in kwargs:
            raise ValueError("pixel_pitch_mm must be provided as a fixed CMOPhysicalStereoModel parameter")
        share_principal_point = bool(kwargs.get("share_principal_point", True))
        expected = 17 if share_principal_point else 19
        if arr.size != expected:
            raise ValueError(f"CMOPhysicalStereoModel expects {expected} parameters")
        if share_principal_point:
            delta_cx_diff = 0.0
            delta_cy_diff = 0.0
            theta_idx = 6
            left_start = 7
        else:
            delta_cx_diff = float(arr[6])
            delta_cy_diff = float(arr[7])
            theta_idx = 8
            left_start = 9
        return cls(
            f_obj_mm=float(arr[0]),
            working_distance_mm=float(arr[1]),
            b_mm=float(arr[2]),
            f_tube_mm=float(arr[3]),
            cx_principal_px=float(arr[4]),
            cy_principal_px=float(arr[5]),
            pixel_pitch_mm=float(kwargs["pixel_pitch_mm"]),
            theta_axis_tilt_rad=float(arr[theta_idx]),
            distortion_left=tuple(float(v) for v in arr[left_start : left_start + 5]),  # type: ignore[arg-type]
            distortion_right=tuple(float(v) for v in arr[left_start + 5 : left_start + 10]),  # type: ignore[arg-type]
            image_size=kwargs.get("image_size"),
            share_principal_point=share_principal_point,
            delta_cx_diff_px=delta_cx_diff,
            delta_cy_diff_px=delta_cy_diff,
        )

    def parameter_vector(self) -> Array:
        common = [
            self.f_obj_mm,
            self.working_distance_mm,
            self.b_mm,
            self.f_tube_mm,
            self.cx_principal_px,
            self.cy_principal_px,
        ]
        if not self.share_principal_point:
            common.extend([self.delta_cx_diff_px, self.delta_cy_diff_px])
        common.append(self.theta_axis_tilt_rad)
        return np.r_[
            np.asarray(common, dtype=np.float64),
            np.asarray(self.distortion_left, dtype=np.float64),
            np.asarray(self.distortion_right, dtype=np.float64),
        ].astype(np.float64)

    def flat_parameter_dict(self) -> dict[str, float]:
        keys = ("k1", "k2", "p1", "p2", "k3")
        params = {
            "f_obj_mm": float(self.f_obj_mm),
            "working_distance_mm": float(self.working_distance_mm),
            "b_mm": float(self.b_mm),
            "f_tube_mm": float(self.f_tube_mm),
            "cx_principal_px": float(self.cx_principal_px),
            "cy_principal_px": float(self.cy_principal_px),
            "theta_axis_tilt_rad": float(self.theta_axis_tilt_rad),
        }
        if not self.share_principal_point:
            params.update(
                {
                    "delta_cx_diff_px": float(self.delta_cx_diff_px),
                    "delta_cy_diff_px": float(self.delta_cy_diff_px),
                }
            )
        params.update({f"left_{k}": float(v) for k, v in zip(keys, self.distortion_left, strict=True)})
        params.update({f"right_{k}": float(v) for k, v in zip(keys, self.distortion_right, strict=True)})
        return params

    def parameter_dict(self) -> dict[str, object]:
        fixed = {
            "pixel_pitch_mm": float(self.pixel_pitch_mm),
            "image_width": float(self.image_size[0]) if self.image_size is not None else float("nan"),
            "image_height": float(self.image_size[1]) if self.image_size is not None else float("nan"),
            "share_principal_point": bool(self.share_principal_point),
        }
        return {"free": self.flat_parameter_dict(), "fixed": fixed}

    def channel(self, channel: Literal["left", "right"]) -> "CMOPhysicalChannelModel":
        return CMOPhysicalChannelModel(rig=self, channel=channel)

    def principal_point_for_channel(self, channel: Literal["left", "right"]) -> tuple[float, float]:
        if self.share_principal_point:
            return float(self.cx_principal_px), float(self.cy_principal_px)
        sign = -1.0 if channel == "left" else 1.0
        return (
            float(self.cx_principal_px) + sign * 0.5 * float(self.delta_cx_diff_px),
            float(self.cy_principal_px) + sign * 0.5 * float(self.delta_cy_diff_px),
        )

    def ray(self, u: Array, v: Array, channel: Literal["left", "right"]) -> tuple[Array, Array]:
        uu, vv = np.broadcast_arrays(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        shape = uu.shape
        uf = uu.reshape(-1)
        vf = vv.reshape(-1)

        cx, cy = self.principal_point_for_channel(channel)
        alpha_x_d = (uf - cx) * float(self.pixel_pitch_mm) / float(self.f_tube_mm)
        alpha_y_d = (vf - cy) * float(self.pixel_pitch_mm) / float(self.f_tube_mm)
        coeffs = self.distortion_left if channel == "left" else self.distortion_right
        alpha_x, alpha_y = undistort_brown_normalized(alpha_x_d, alpha_y_d, *coeffs, n_iter=10)

        sign = -1.0 if channel == "left" else 1.0
        z_pupil = float(self.working_distance_mm) - float(self.f_obj_mm)
        pupil = np.column_stack(
            [
                np.full_like(alpha_x, sign * 0.5 * float(self.b_mm)),
                np.zeros_like(alpha_y),
                np.full_like(alpha_x, z_pupil),
            ]
        )
        object_plane_point = np.column_stack(
            [
                float(self.working_distance_mm) * alpha_x,
                float(self.working_distance_mm) * alpha_y,
                np.full_like(alpha_x, float(self.working_distance_mm)),
            ]
        )
        directions = _normalize(object_plane_point - pupil)

        if self.theta_axis_tilt_rad != 0.0:
            R = _roty(float(self.theta_axis_tilt_rad))
            pupil = (R @ pupil.T).T
            directions = (R @ directions.T).T

        return pupil.reshape(shape + (3,)), directions.reshape(shape + (3,))


@dataclass(frozen=True)
class CMOPhysicalChannelModel:
    """Single-channel facade for a shared CMO physical rig."""

    rig: CMOPhysicalStereoModel
    channel: Literal["left", "right"]
    name: str = "cmo_physical_channel"
    is_stereo_shared: bool = True

    @property
    def n_parameters(self) -> int:
        return self.rig.n_parameters

    def parameter_vector(self) -> Array:
        return self.rig.parameter_vector()

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs) -> "CMOPhysicalChannelModel":
        rig = CMOPhysicalStereoModel.from_parameter_vector(
            x,
            image_size=kwargs.get("image_size"),
            pixel_pitch_mm=kwargs.get("pixel_pitch_mm"),
            share_principal_point=kwargs.get("share_principal_point", True),
        )
        channel = kwargs.get("channel", "left")
        if channel not in {"left", "right"}:
            raise ValueError("channel must be 'left' or 'right'")
        return cls(rig=rig, channel=channel)

    def parameter_dict(self) -> dict[str, object]:
        return self.rig.parameter_dict()

    def ray(self, u: Array, v: Array) -> tuple[Array, Array]:
        return self.rig.ray(u, v, self.channel)


@dataclass(frozen=True)
class CMOPhysicalStereoFitResult:
    """Result of fitting a shared physical CMO model to two rayfields."""

    model: CMOPhysicalStereoModel
    success: bool
    message: str
    n_parameters: int
    n_samples: int
    n_residual_scalars: int
    rss: float
    left_rms_mm: float
    right_rms_mm: float
    rms_mm: float
    aic: float
    bic: float
    parameter_vector: Array
    parameter_dict: dict[str, object]


def fit_cmo_physical_stereo_model_to_rayfields(
    left_field,
    right_field,
    image_size: tuple[int, int],
    initial_parameters: Array,
    bounds: tuple[Array, Array] | None = None,
    *,
    pixel_pitch_mm: float,
    z_planes: tuple[float, float] = (50.0, 250.0),
    grid_shape: tuple[int, int] = (17, 13),
    support_pixels_left: Array | None = None,
    support_pixels_right: Array | None = None,
    support_weight: float = 1.0,
    full_grid_weight: float = 0.25,
    robust_loss: str = "huber",
    max_nfev: int = 2000,
) -> CMOPhysicalStereoFitResult:
    """Fit the shared physical CMO rig to left/right measured rayfields."""

    x0 = np.asarray(initial_parameters, dtype=np.float64).reshape(-1)
    if x0.size not in {17, 19}:
        raise ValueError("initial_parameters must contain 17 shared-PP or 19 aligned-PP values")
    share_principal_point = x0.size == 17
    full = _grid_pixels(image_size, grid_shape)
    support_l = full if support_pixels_left is None else np.asarray(support_pixels_left, dtype=np.float64).reshape(-1, 2)
    support_r = full if support_pixels_right is None else np.asarray(support_pixels_right, dtype=np.float64).reshape(-1, 2)
    include_full = full_grid_weight > 0 and (support_pixels_left is not None or support_pixels_right is not None)

    def model_at(x: Array) -> CMOPhysicalStereoModel:
        return CMOPhysicalStereoModel.from_parameter_vector(
            x,
            image_size=image_size,
            pixel_pitch_mm=pixel_pitch_mm,
            share_principal_point=share_principal_point,
        )

    def residuals(x: Array) -> Array:
        model = model_at(x)
        left = model.channel("left")
        right = model.channel("right")
        blocks = [
            float(support_weight)
            * rayfield_two_plane_residuals(left_field, left, support_l, z_planes=z_planes),
            float(support_weight)
            * rayfield_two_plane_residuals(right_field, right, support_r, z_planes=z_planes),
        ]
        if include_full:
            blocks.extend(
                [
                    float(full_grid_weight)
                    * rayfield_two_plane_residuals(left_field, left, full, z_planes=z_planes),
                    float(full_grid_weight)
                    * rayfield_two_plane_residuals(right_field, right, full, z_planes=z_planes),
                ]
            )
        return np.concatenate(blocks)

    if bounds is None:
        if share_principal_point:
            lower_common = [1.0, 1.0, 0.0, 1.0, -np.inf, -np.inf, -0.25]
            upper_common = [500.0, 1000.0, 200.0, 1000.0, np.inf, np.inf, 0.25]
        else:
            lower_common = [1.0, 1.0, 0.0, 1.0, -np.inf, -np.inf, -50.0, -50.0, -0.25]
            upper_common = [500.0, 1000.0, 200.0, 1000.0, np.inf, np.inf, 50.0, 50.0, 0.25]
        lower = np.array(
            lower_common + [-1.0, -1.0, -0.1, -0.1, -1.0] * 2,
            dtype=np.float64,
        )
        upper = np.array(
            upper_common + [1.0, 1.0, 0.1, 0.1, 1.0] * 2,
            dtype=np.float64,
        )
        bounds = (lower, upper)
    sol = least_squares(
        residuals,
        x0=x0,
        bounds=bounds,
        loss=robust_loss,
        f_scale=1.0,
        max_nfev=int(max_nfev),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    fitted = model_at(sol.x)
    left_res = rayfield_two_plane_residuals(left_field, fitted.channel("left"), support_l, z_planes=z_planes)
    right_res = rayfield_two_plane_residuals(right_field, fitted.channel("right"), support_r, z_planes=z_planes)
    combined = residuals(sol.x)
    rss = float(np.sum(combined * combined))
    n_res = int(combined.size)
    n_samples = int(support_l.shape[0] + support_r.shape[0] + (2 * full.shape[0] if include_full else 0))
    aic, bic = _aic_bic(rss, n_res, n_samples, fitted.n_parameters)
    left_rms = _ray_rms(left_res)
    right_rms = _ray_rms(right_res)
    return CMOPhysicalStereoFitResult(
        model=fitted,
        success=bool(sol.success),
        message=str(sol.message),
        n_parameters=fitted.n_parameters,
        n_samples=n_samples,
        n_residual_scalars=n_res,
        rss=rss,
        left_rms_mm=left_rms,
        right_rms_mm=right_rms,
        rms_mm=float(np.sqrt(0.5 * (left_rms**2 + right_rms**2))),
        aic=aic,
        bic=bic,
        parameter_vector=np.asarray(sol.x, dtype=np.float64).copy(),
        parameter_dict=fitted.parameter_dict(),
    )


__all__ = [
    "CMOPhysicalChannelModel",
    "CMOPhysicalStereoFitResult",
    "CMOPhysicalStereoModel",
    "fit_cmo_physical_stereo_model_to_rayfields",
]
