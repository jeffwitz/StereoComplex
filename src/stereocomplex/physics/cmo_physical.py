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
from typing import Any, Literal

import numpy as np
from scipy.optimize import least_squares  # type: ignore

from stereocomplex.physics.central_models import undistort_brown_normalized
from stereocomplex.physics.parallel_plate_fit import rayfield_two_plane_residuals

Array = np.ndarray


def transverse_gauge_origin(origin: Array, d: Array) -> Array:
    """Project origin to be transverse to direction (line-representation gauge).

    The physical sub-pupil does NOT satisfy O·d=0.  This function returns
    O_perp = O - (O·d)d, which represents the SAME 3D line as (O,d) but
    with an origin that satisfies the transverse gauge convention used by
    the Zernike rayfield.  Use ONLY for comparison with Zernike fields,
    NOT inside the physical model's ray() method.
    """
    O_arr = np.asarray(origin, dtype=np.float64)
    d_arr = np.asarray(d, dtype=np.float64)
    return O_arr - np.sum(O_arr * d_arr, axis=-1, keepdims=True) * d_arr


def _normalize(v: Array, eps: float = 1e-15) -> Array:
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(n, float(eps))


def _roty(angle_rad: float) -> Array:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rotx(angle_rad: float) -> Array:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


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


def _aic_bic(
    rss: float, n_residual_scalars: int, n_observations: int, p: int
) -> tuple[float, float]:
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
    theta_pitch_rad: float = 0.0
    telecentric_offset_mm: float = 0.0
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
        return 19 if self.share_principal_point else 21

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs) -> CMOPhysicalStereoModel:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if "pixel_pitch_mm" not in kwargs:
            raise ValueError(
                "pixel_pitch_mm must be provided as a fixed CMOPhysicalStereoModel parameter"
            )
        share_principal_point = bool(kwargs.get("share_principal_point", True))
        expected = 19 if share_principal_point else 21
        if arr.size != expected:
            raise ValueError(f"CMOPhysicalStereoModel expects {expected} parameters")
        if share_principal_point:
            delta_cx_diff = 0.0
            delta_cy_diff = 0.0
            theta_idx = 6
            pitch_idx = 7
            tele_idx = 8
            left_start = 9
        else:
            delta_cx_diff = float(arr[6])
            delta_cy_diff = float(arr[7])
            theta_idx = 8
            pitch_idx = 9
            tele_idx = 10
            left_start = 11
        return cls(
            f_obj_mm=float(arr[0]),
            working_distance_mm=float(arr[1]),
            b_mm=float(arr[2]),
            f_tube_mm=float(arr[3]),
            cx_principal_px=float(arr[4]),
            cy_principal_px=float(arr[5]),
            pixel_pitch_mm=float(kwargs["pixel_pitch_mm"]),
            theta_axis_tilt_rad=float(arr[theta_idx]),
            theta_pitch_rad=float(arr[pitch_idx]),
            telecentric_offset_mm=float(arr[tele_idx]),
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
        common.append(self.theta_pitch_rad)
        common.append(self.telecentric_offset_mm)
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
            "theta_pitch_rad": float(self.theta_pitch_rad),
            "telecentric_offset_mm": float(self.telecentric_offset_mm),
        }
        if not self.share_principal_point:
            params.update(
                {
                    "delta_cx_diff_px": float(self.delta_cx_diff_px),
                    "delta_cy_diff_px": float(self.delta_cy_diff_px),
                }
            )
        params.update(
            {f"left_{k}": float(v) for k, v in zip(keys, self.distortion_left, strict=True)}
        )
        params.update(
            {f"right_{k}": float(v) for k, v in zip(keys, self.distortion_right, strict=True)}
        )
        return params

    def parameter_dict(self) -> dict[str, object]:
        fixed = {
            "pixel_pitch_mm": float(self.pixel_pitch_mm),
            "image_width": float(self.image_size[0])
            if self.image_size is not None
            else float("nan"),
            "image_height": float(self.image_size[1])
            if self.image_size is not None
            else float("nan"),
            "share_principal_point": bool(self.share_principal_point),
        }
        return {"free": self.flat_parameter_dict(), "fixed": fixed}

    def channel(self, channel: Literal["left", "right"]) -> CMOPhysicalChannelModel:
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
        uu, vv = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        )
        shape = uu.shape
        uf = uu.reshape(-1)
        vf = vv.reshape(-1)

        cx, cy = self.principal_point_for_channel(channel)
        alpha_x_d = (uf - cx) * float(self.pixel_pitch_mm) / float(self.f_tube_mm)
        alpha_y_d = (vf - cy) * float(self.pixel_pitch_mm) / float(self.f_tube_mm)
        coeffs = self.distortion_left if channel == "left" else self.distortion_right
        alpha_x, alpha_y = undistort_brown_normalized(alpha_x_d, alpha_y_d, *coeffs, n_iter=10)

        sign = -1.0 if channel == "left" else 1.0
        z_pupil = (
            float(self.working_distance_mm)
            - float(self.f_obj_mm)
            + float(self.telecentric_offset_mm)
        )
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

        if self.theta_pitch_rad != 0.0:
            R = _rotx(float(self.theta_pitch_rad))
            pupil = (R @ pupil.T).T
            directions = (R @ directions.T).T

        return pupil.reshape((*shape, 3)), directions.reshape((*shape, 3))


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
    def from_parameter_vector(cls, x: Array, **kwargs) -> CMOPhysicalChannelModel:
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

    def project_point(self, X: Array, max_iter: int = 20) -> tuple[Array, bool]:
        """Analytic projection of a 3-D point to pixel coordinates.

        Inverts the CMO ray model: given a world point *X*, find the
        pixel whose ray passes through *X*.  The distortion inversion
        uses fixed-point iteration (converges in ~5 steps for typical
        coefficients).

        Returns ``(uv, ok)`` where *uv* is (2,) and *ok* is True if
        the point projects inside the sensor.
        """
        import numpy as np
        from stereocomplex.physics.central_models import (
            brown_conrady_distort_normalized,
        )

        X_arr = np.asarray(X, dtype=np.float64).reshape(3)
        rig = self.rig
        cx, cy = rig.principal_point_for_channel(self.channel)
        sign = -1.0 if self.channel == "left" else 1.0
        S = np.array([sign * 0.5 * rig.b_mm, 0.0, rig.working_distance_mm - rig.f_obj_mm])

        d = X_arr - S
        if abs(d[2]) < 1e-12:
            return np.full(2, np.nan), False
        lam = (rig.working_distance_mm - S[2]) / d[2]
        P = S + lam * d
        alpha_x_u = P[0] / rig.working_distance_mm
        alpha_y_u = P[1] / rig.working_distance_mm

        coeffs = rig.distortion_left if self.channel == "left" else rig.distortion_right
        # Apply distortion (forward) to go from undistorted to distorted angular coords
        alpha_x_d, alpha_y_d = brown_conrady_distort_normalized(
            np.array([alpha_x_u]),
            np.array([alpha_y_u]),
            *coeffs,
        )
        u = cx + alpha_x_d[0] * rig.f_tube_mm / rig.pixel_pitch_mm
        v = cy + alpha_y_d[0] * rig.f_tube_mm / rig.pixel_pitch_mm

        W, H = rig.image_size or (int(2 * cx), int(2 * cy))
        ok = (0 <= float(u) <= W - 1) and (0 <= float(v) <= H - 1)
        return np.array([float(u), float(v)]), bool(ok)


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
    if x0.size not in {19, 21}:
        raise ValueError("initial_parameters must contain 19 shared-PP or 21 aligned-PP values")
    share_principal_point = x0.size == 19
    full = _grid_pixels(image_size, grid_shape)
    support_l = (
        full
        if support_pixels_left is None
        else np.asarray(support_pixels_left, dtype=np.float64).reshape(-1, 2)
    )
    support_r = (
        full
        if support_pixels_right is None
        else np.asarray(support_pixels_right, dtype=np.float64).reshape(-1, 2)
    )
    include_full = full_grid_weight > 0 and (
        support_pixels_left is not None or support_pixels_right is not None
    )

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
            lower_common = [1.0, 1.0, 0.0, 1.0, -np.inf, -np.inf, -0.25, -0.25, -500.0]
            upper_common = [500.0, 1000.0, 200.0, 1000.0, np.inf, np.inf, 0.25, 0.25, 500.0]
        else:
            lower_common = [
                1.0,
                1.0,
                0.0,
                1.0,
                -np.inf,
                -np.inf,
                -50.0,
                -50.0,
                -0.25,
                -0.25,
                -500.0,
            ]
            upper_common = [
                500.0,
                1000.0,
                200.0,
                1000.0,
                np.inf,
                np.inf,
                50.0,
                50.0,
                0.25,
                0.25,
                500.0,
            ]
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
    left_res = rayfield_two_plane_residuals(
        left_field, fitted.channel("left"), support_l, z_planes=z_planes
    )
    right_res = rayfield_two_plane_residuals(
        right_field, fitted.channel("right"), support_r, z_planes=z_planes
    )
    combined = residuals(sol.x)
    rss = float(np.sum(combined * combined))
    n_res = int(combined.size)
    n_samples = int(
        support_l.shape[0] + support_r.shape[0] + (2 * full.shape[0] if include_full else 0)
    )
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


@dataclass(frozen=True)
class CMOTelecentricStereoModel:
    """Quasi-telecentric CMO model with rigid sub-pupil + affine direction field.

    Each channel has a **constant origin** :math:`S_c` (the effective sub-pupil)
    and a direction field that varies **affinely** across pixels.  This captures
    the quasi-telecentric character of infinity-corrected CMO microscopes where
    the direction is nearly constant across the field of view.

    The direction field is:

    .. math::
        d_c(u,v) = \\operatorname{normalize}\\left(
            d_{c,0} + s_x \\tilde{u}\\, e_x + s_y \\tilde{v}\\, e_y
        \\right)

    where :math:`\\tilde{u} = (u - c_x) \\cdot p_{\\text{pix}} / f_{\\text{obj}}`
    and :math:`\\tilde{v} = (v - c_y) \\cdot p_{\\text{pix}} / f_{\\text{obj}}`.
    """

    f_obj_mm: float  # for sub-pupil depth: z_pupil = WD - f_obj
    working_distance_mm: float
    b_mm: float
    cx_principal_px: float
    cy_principal_px: float
    pixel_pitch_mm: float

    f_angular_mm: float = 0.0  # pixel→angle normalisation (≈ f_obj, but decoupled)
    theta_convergence_half_rad: float = 0.0
    d_y_common: float = 0.0
    s_x_L: float = 0.0
    s_y_L: float = 0.0
    s_x_R: float = 0.0
    s_y_R: float = 0.0
    s_xy_L: float = 0.0  # cross-term: d_x += s_xy * ṽ
    s_yx_L: float = 0.0  # cross-term: d_y += s_yx * ũ
    s_xy_R: float = 0.0
    s_yx_R: float = 0.0
    shared_slopes: bool = True

    # Pupil shear: small affine origin variation (transverse to direction)
    rho_x_L: float = 0.0
    rho_y_L: float = 0.0
    rho_x_R: float = 0.0
    rho_y_R: float = 0.0
    shared_shear: bool = True

    # Quadratic direction correction (centered, before normalization)
    q_xx_L: float = 0.0
    q_yy_L: float = 0.0
    q_xx_R: float = 0.0
    q_yy_R: float = 0.0
    shared_quadratic: bool = True

    image_size: tuple[int, int] | None = None
    name: str = "cmo_telecentric"
    is_stereo_shared: bool = True

    @property
    def n_parameters(self) -> int:
        n = 12 if self.shared_slopes else 14
        if not self.shared_shear:
            n += 2  # per-channel rho
        if not self.shared_quadratic:
            n += 2  # per-channel q
        return n

    @classmethod
    def from_parameter_vector(
        cls,
        x: Array,
        **kwargs: Any,
    ) -> CMOTelecentricStereoModel:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size not in {12, 14, 16}:
            raise ValueError(
                f"CMOTelecentricStereoModel expects 12/14/16 parameters, got {arr.size}"
            )
        sh_slope = arr.size == 12  # shared slopes
        sh_shear = arr.size in {12, 14}  # shared shear (12=shared both, 14=per-ch slopes+sh shear)
        sh_quad = True  # quadratic always shared for now
        # Slope extraction
        sxL = float(arr[8])
        syL = float(arr[9])
        if sh_slope:
            sxR = sxL
            syR = syL
            rho_idx = 10
        else:
            sxR = float(arr[10])
            syR = float(arr[11])
            rho_idx = 12
        # Shear extraction
        if sh_shear:
            rxL = float(arr[rho_idx])
            ryL = float(arr[rho_idx + 1])
            rxR = rxL
            ryR = ryL
            q_idx = rho_idx + 2
        else:
            rxL = float(arr[rho_idx])
            ryL = float(arr[rho_idx + 1])
            rxR = float(arr[rho_idx + 2])
            ryR = float(arr[rho_idx + 3])
            q_idx = rho_idx + 4
        # Quadratic extraction (only if present in vector)
        if arr.size > q_idx + 1:
            qxL = float(arr[q_idx])
            qyL = float(arr[q_idx + 1])
            qxR = qxL
            qyR = qyL  # shared quadratic
        else:
            qxL = 0.0
            qyL = 0.0
            qxR = 0.0
            qyR = 0.0
        return cls(
            f_obj_mm=float(arr[0]),
            working_distance_mm=float(arr[1]),
            b_mm=float(arr[2]),
            cx_principal_px=float(arr[3]),
            cy_principal_px=float(arr[4]),
            pixel_pitch_mm=float(kwargs["pixel_pitch_mm"]),
            f_angular_mm=float(arr[5]),
            theta_convergence_half_rad=float(arr[6]),
            d_y_common=float(arr[7]),
            s_x_L=sxL,
            s_y_L=syL,
            s_x_R=sxR,
            s_y_R=syR,
            shared_slopes=sh_slope,
            rho_x_L=rxL,
            rho_y_L=ryL,
            rho_x_R=rxR,
            rho_y_R=ryR,
            shared_shear=sh_shear,
            q_xx_L=qxL,
            q_yy_L=qyL,
            q_xx_R=qxR,
            q_yy_R=qyR,
            shared_quadratic=sh_quad,
            image_size=kwargs.get("image_size"),
        )

    def parameter_vector(self) -> Array:
        vec = [
            self.f_obj_mm,
            self.working_distance_mm,
            self.b_mm,
            self.cx_principal_px,
            self.cy_principal_px,
            self.f_angular_mm,
            self.theta_convergence_half_rad,
            self.d_y_common,
            self.s_x_L,
            self.s_y_L,
        ]
        if not self.shared_slopes:
            vec.extend([self.s_x_R, self.s_y_R])
        if self.shared_shear:
            vec.extend([self.rho_x_L, self.rho_y_L])
        else:
            vec.extend([self.rho_x_L, self.rho_y_L, self.rho_x_R, self.rho_y_R])
        return np.array(vec, dtype=np.float64)

    def parameter_dict(self) -> dict[str, object]:
        free = {
            "f_obj_mm": float(self.f_obj_mm),
            "working_distance_mm": float(self.working_distance_mm),
            "b_mm": float(self.b_mm),
            "cx_principal_px": float(self.cx_principal_px),
            "cy_principal_px": float(self.cy_principal_px),
            "f_angular_mm": float(self.f_angular_mm),
            "theta_convergence_half_deg": float(np.degrees(self.theta_convergence_half_rad)),
            "d_y_common": float(self.d_y_common),
            "s_x_L": float(self.s_x_L),
            "s_y_L": float(self.s_y_L),
            "shared_slopes": bool(self.shared_slopes),
            "shared_shear": bool(self.shared_shear),
        }
        if not self.shared_slopes:
            free["s_x_R"] = float(self.s_x_R)
            free["s_y_R"] = float(self.s_y_R)
        else:
            free["s_x"] = float(self.s_x_L)
            free["s_y"] = float(self.s_y_L)
        free["rho_x_L"] = float(self.rho_x_L)
        free["rho_y_L"] = float(self.rho_y_L)
        if not self.shared_shear:
            free["rho_x_R"] = float(self.rho_x_R)
            free["rho_y_R"] = float(self.rho_y_R)
        else:
            free["rho_x"] = float(self.rho_x_L)
            free["rho_y"] = float(self.rho_y_L)
        return {
            "free": free,
            "fixed": {
                "pixel_pitch_mm": float(self.pixel_pitch_mm),
                "image_width": float(self.image_size[0]) if self.image_size else float("nan"),
                "image_height": float(self.image_size[1]) if self.image_size else float("nan"),
            },
        }

    def channel(self, channel: Literal["left", "right"]) -> CMOTelecentricChannelModel:
        return CMOTelecentricChannelModel(rig=self, channel=channel)

    def ray(self, u: Array, v: Array, channel: Literal["left", "right"]) -> tuple[Array, Array]:
        uu, vv = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        )
        shape = uu.shape
        uf, vf = uu.reshape(-1), vv.reshape(-1)

        # Normalised angular coordinates (using f_angular, decoupled from sub-pupil depth)
        f_ang = float(self.f_angular_mm) if float(self.f_angular_mm) > 0.1 else float(self.f_obj_mm)
        tilde_u = (uf - float(self.cx_principal_px)) * float(self.pixel_pitch_mm) / f_ang
        tilde_v = (vf - float(self.cy_principal_px)) * float(self.pixel_pitch_mm) / f_ang

        # Chief-ray direction (symmetric X, shared Y)
        # Left channel: ray from (-b/2, 0, z_pupil) to (0, 0, WD) → d_x > 0
        # Right channel: ray from (+b/2, 0, z_pupil) to (0, 0, WD) → d_x < 0
        dir_sign = 1.0 if channel == "left" else -1.0
        sin_th = float(np.sin(float(self.theta_convergence_half_rad)))
        cos_th = float(np.cos(float(self.theta_convergence_half_rad)))
        dy = float(self.d_y_common)

        d0 = np.column_stack(
            [
                np.full_like(uf, dir_sign * sin_th),
                np.full_like(uf, dy),
                np.full_like(uf, cos_th),
            ]
        )

        # Affine + cross + quadratic correction (per-channel)
        sx_ch = float(self.s_x_L) if channel == "left" else float(self.s_x_R)
        sy_ch = float(self.s_y_L) if channel == "left" else float(self.s_y_R)
        sxy_ch = float(self.s_xy_L) if channel == "left" else float(self.s_xy_R)
        syx_ch = float(self.s_yx_L) if channel == "left" else float(self.s_yx_R)
        qx_ch = float(self.q_xx_L) if channel == "left" else float(self.q_xx_R)
        qy_ch = float(self.q_yy_L) if channel == "left" else float(self.q_yy_R)

        # Centered quadratics
        umean2 = float(np.mean(tilde_u**2))
        vmean2 = float(np.mean(tilde_v**2))

        d_raw = np.column_stack(
            [
                d0[:, 0] + sx_ch * tilde_u + sxy_ch * tilde_v + qx_ch * (tilde_u**2 - umean2),
                d0[:, 1] + sy_ch * tilde_v + syx_ch * tilde_u + qy_ch * (tilde_v**2 - vmean2),
                d0[:, 2],
            ]
        )

        directions = _normalize(d_raw)

        # Origin: sub-pupil (left at -b/2, right at +b/2) + pupil shear
        pupil_sign = -1.0 if channel == "left" else 1.0
        z_pupil = float(self.working_distance_mm) - float(self.f_obj_mm)
        rho_x = float(self.rho_x_L) if channel == "left" else float(self.rho_x_R)
        rho_y = float(self.rho_y_L) if channel == "left" else float(self.rho_y_R)

        O_rigid = np.column_stack(
            [
                np.full_like(uf, pupil_sign * 0.5 * float(self.b_mm)),
                np.zeros_like(uf),
                np.full_like(uf, z_pupil),
            ]
        )
        # Pupil shear: affine origin variation
        delta_O = np.column_stack([rho_x * tilde_u, rho_y * tilde_v, np.zeros_like(uf)])
        pupil = O_rigid + delta_O  # physical origin, NOT gauge-projected

        return pupil.reshape((*shape, 3)), directions.reshape((*shape, 3))


@dataclass(frozen=True)
class CMOTelecentricNModel:
    """Named collection of telecentric CMO channels for N-camera workflows."""

    channels: tuple[CMOTelecentricChannelModel, ...]
    name: str = "cmo_telecentric_n"
    is_stereo_shared: bool = False

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("at least one channel is required")
        names = [channel.channel for channel in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("channel names must be unique")

    @classmethod
    def from_stereo(cls, stereo: CMOTelecentricStereoModel) -> CMOTelecentricNModel:
        return cls((stereo.channel("left"), stereo.channel("right")))

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(channel.channel for channel in self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def n_parameters(self) -> int:
        return sum(channel.n_parameters for channel in self.channels)

    def channel(self, name: str) -> CMOTelecentricChannelModel:
        for channel in self.channels:
            if channel.channel == name:
                return channel
        raise KeyError(name)

    def ray(self, u: Array, v: Array, channel: str) -> tuple[Array, Array]:
        return self.channel(channel).ray(u, v)


@dataclass(frozen=True)
class CMOTelecentricChannelModel:
    """Single-channel facade for a shared telecentric CMO rig."""

    rig: CMOTelecentricStereoModel
    channel: Literal["left", "right"]
    name: str = "cmo_telecentric_channel"
    is_stereo_shared: bool = True

    @property
    def n_parameters(self) -> int:
        return self.rig.n_parameters

    def parameter_vector(self) -> Array:
        return self.rig.parameter_vector()

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs: Any) -> CMOTelecentricChannelModel:
        rig = CMOTelecentricStereoModel.from_parameter_vector(
            x,
            pixel_pitch_mm=kwargs["pixel_pitch_mm"],
            image_size=kwargs.get("image_size"),
        )
        channel = kwargs.get("channel", "left")
        if channel not in {"left", "right"}:
            raise ValueError("channel must be 'left' or 'right'")
        return cls(rig=rig, channel=channel)

    def parameter_dict(self) -> dict[str, object]:
        return self.rig.parameter_dict()

    def ray(self, u: Array, v: Array) -> tuple[Array, Array]:
        return self.rig.ray(u, v, self.channel)


def fit_cmo_telecentric_model_to_rayfields(
    left_field,
    right_field,
    image_size: tuple[int, int],
    initial_parameters: Array,
    *,
    pixel_pitch_mm: float,
    z_planes: tuple[float, float] = (50.0, 250.0),
    grid_shape: tuple[int, int] = (17, 13),
    full_grid_weight: float = 0.0,
    max_nfev: int = 500,
) -> CMOPhysicalStereoFitResult:
    """Fit the telecentric CMO model to left/right measured rayfields."""
    x0 = np.asarray(initial_parameters, dtype=np.float64).reshape(-1)
    if x0.size not in {10, 12, 14, 16}:
        raise ValueError(f"initial_parameters must contain 10, 12, 14, or 16 values, got {x0.size}")

    full = _grid_pixels(image_size, grid_shape)
    support_l = full
    support_r = full
    include_full = full_grid_weight > 0

    def model_at(x: Array) -> CMOTelecentricStereoModel:
        return CMOTelecentricStereoModel.from_parameter_vector(
            x, pixel_pitch_mm=pixel_pitch_mm, image_size=image_size
        )

    def residuals(x: Array) -> Array:
        model = model_at(x)
        left = model.channel("left")
        right = model.channel("right")
        blocks = [
            rayfield_two_plane_residuals(left_field, left, support_l, z_planes=z_planes),
            rayfield_two_plane_residuals(right_field, right, support_r, z_planes=z_planes),
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

    # Bounds: physically reasonable ranges, extended for larger parameter vectors
    lower = np.array(
        [1.0, 1.0, 0.0, -np.inf, -np.inf, 1.0, 0.0, -0.3, -10.0, -10.0], dtype=np.float64
    )
    upper = np.array(
        [500.0, 1000.0, 200.0, np.inf, np.inf, 500.0, 0.5, 0.3, 10.0, 10.0], dtype=np.float64
    )
    if x0.size >= 12:
        lower = np.concatenate([lower, [-10.0, -10.0]])
        upper = np.concatenate([upper, [10.0, 10.0]])
    if x0.size >= 14:
        lower = np.concatenate([lower, [-10.0, -10.0]])
        upper = np.concatenate([upper, [10.0, 10.0]])
    if x0.size >= 16:
        lower = np.concatenate([lower, [-10.0, -10.0]])
        upper = np.concatenate([upper, [10.0, 10.0]])

    sol = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    fitted = model_at(sol.x)
    left_res = rayfield_two_plane_residuals(
        left_field, fitted.channel("left"), support_l, z_planes=z_planes
    )
    right_res = rayfield_two_plane_residuals(
        right_field, fitted.channel("right"), support_r, z_planes=z_planes
    )
    combined = residuals(sol.x)
    rss = float(np.sum(combined * combined))
    n_res = int(combined.size)
    n_samples = int(support_l.shape[0] + support_r.shape[0])
    left_rms = _ray_rms(left_res)
    right_rms = _ray_rms(right_res)
    aic, bic = _aic_bic(rss, n_res, n_samples, fitted.n_parameters)

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


# ═══════════════════════════════════════════════════════════════════════
# Polynomial pre-warp helpers
# ═══════════════════════════════════════════════════════════════════════


def _poly_terms_2d(level: int) -> list[tuple[int, int]]:
    """Monomial exponent pairs (pu, pv) for a 2D polynomial of given level.

    Level 0: constant (1 term).
    Level 1: + u, v (3 terms).
    Level 2: + u^2, uv, v^2 (6 terms).
    Level 3: + u^3, u^2 v, u v^2, v^3 (10 terms).
    """
    terms: list[tuple[int, int]] = [(0, 0)]
    if level >= 1:
        terms.extend([(1, 0), (0, 1)])
    if level >= 2:
        terms.extend([(2, 0), (1, 1), (0, 2)])
    if level >= 3:
        terms.extend([(3, 0), (2, 1), (1, 2), (0, 3)])
    return terms


def _n_warp_coeff_per_axis(level: int) -> int:
    return len(_poly_terms_2d(level))


def _n_warp_coeff_total(level: int, shared: bool) -> int:
    """Total warp coefficients across both axes and channels."""
    if level == 0:
        return 0  # identity warp, no coefficients
    per_axis = _n_warp_coeff_per_axis(level)
    per_channel = per_axis * 2  # xi + eta
    return per_channel if shared else per_channel * 2


def _polyval_2d(u: Array, v: Array, coeffs: tuple[float, ...], level: int) -> Array:
    """Evaluate a 2D polynomial at (u, v)."""
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    result = np.zeros_like(u_arr)
    terms = _poly_terms_2d(level)
    for (pu, pv), c in zip(terms, coeffs, strict=True):
        result = result + float(c) * (u_arr ** float(pu)) * (v_arr ** float(pv))
    return result


# ═══════════════════════════════════════════════════════════════════════
# CMOWarpedStereoModel
# ═══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CMOWarpedStereoModel:
    """CMO model with configurable polynomial pixel pre-warp.

    Pipeline per channel::

        (xi, eta) = W_c(u, v)              -- polynomial pre-warp (level 0..3)
        d = normalize(d_chief + affine correction on warped coords)
        O = S_c + pupil shear on warped coords

    The warp maps pixel coordinates to optical field coordinates before
    the telecentric direction model acts.  This captures non-radial,
    asymmetric field-angle distortions that a pure affine direction field
    cannot.
    """

    # --- Telecentric stereo base ---
    f_obj_mm: float
    working_distance_mm: float
    b_mm: float
    cx_principal_px: float
    cy_principal_px: float
    pixel_pitch_mm: float
    f_angular_mm: float = 0.0
    theta_convergence_half_rad: float = 0.0
    d_y_common: float = 0.0
    s_x_L: float = 0.0
    s_y_L: float = 0.0
    s_x_R: float = 0.0
    s_y_R: float = 0.0
    s_xy_L: float = 0.0
    s_yx_L: float = 0.0
    s_xy_R: float = 0.0
    s_yx_R: float = 0.0
    shared_slopes: bool = True
    rho_x_L: float = 0.0
    rho_y_L: float = 0.0
    rho_x_R: float = 0.0
    rho_y_R: float = 0.0
    shared_shear: bool = True
    q_xx_L: float = 0.0
    q_yy_L: float = 0.0
    q_xx_R: float = 0.0
    q_yy_R: float = 0.0
    shared_quadratic: bool = True
    image_size: tuple[int, int] | None = None
    name: str = "cmo_warped_stereo"
    is_stereo_shared: bool = True

    # --- Warp fields ---
    warp_xi_L: tuple[float, ...] = ()
    warp_eta_L: tuple[float, ...] = ()
    warp_xi_R: tuple[float, ...] = ()
    warp_eta_R: tuple[float, ...] = ()
    warp_level: int = 0
    shared_warp: bool = True

    def __post_init__(self) -> None:
        if int(self.warp_level) == 0:
            return  # level 0 = identity, no coeff validation needed
        per_axis = _n_warp_coeff_per_axis(int(self.warp_level))
        for name, tup in [
            ("warp_xi_L", self.warp_xi_L),
            ("warp_eta_L", self.warp_eta_L),
            ("warp_xi_R", self.warp_xi_R),
            ("warp_eta_R", self.warp_eta_R),
        ]:
            if len(tup) != per_axis:
                raise ValueError(
                    f"{name} has {len(tup)} coeffs, expected {per_axis} for level {self.warp_level}"
                )

    @property
    def n_parameters(self) -> int:
        n = 12 if self.shared_slopes else 14
        if not self.shared_shear:
            n += 2
        if not self.shared_quadratic:
            n += 2
        n += _n_warp_coeff_total(int(self.warp_level), bool(self.shared_warp))
        return n

    def _get_warp_coeffs(
        self, channel: Literal["left", "right"]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if channel == "left":
            return self.warp_xi_L, self.warp_eta_L
        return self.warp_xi_R, self.warp_eta_R

    def parameter_vector(self) -> Array:
        vec: list[float] = [
            float(self.f_obj_mm),
            float(self.working_distance_mm),
            float(self.b_mm),
            float(self.cx_principal_px),
            float(self.cy_principal_px),
            float(self.f_angular_mm),
            float(self.theta_convergence_half_rad),
            float(self.d_y_common),
            float(self.s_x_L),
            float(self.s_y_L),
        ]
        if not self.shared_slopes:
            vec.extend([float(self.s_x_R), float(self.s_y_R)])
        if self.shared_shear:
            vec.extend([float(self.rho_x_L), float(self.rho_y_L)])
        else:
            vec.extend(
                [float(self.rho_x_L), float(self.rho_y_L), float(self.rho_x_R), float(self.rho_y_R)]
            )
        if self.warp_level > 0:
            vec.extend(self.warp_xi_L)
            vec.extend(self.warp_eta_L)
            if not self.shared_warp:
                vec.extend(self.warp_xi_R)
                vec.extend(self.warp_eta_R)
        return np.array(vec, dtype=np.float64)

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs: Any) -> CMOWarpedStereoModel:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        pixel_pitch_mm = float(kwargs["pixel_pitch_mm"])
        image_size = kwargs.get("image_size")
        warp_level = int(kwargs.get("warp_level", 0))
        shared_warp = bool(kwargs.get("shared_warp", True))

        n_warp = _n_warp_coeff_total(warp_level, shared_warp)
        n_base = int(arr.size) - n_warp

        if n_base not in {12, 14, 16}:
            raise ValueError(
                f"CMOWarpedStereoModel: invalid base param count {n_base} "
                f"(warp_level={warp_level}, shared_warp={shared_warp})"
            )
        shared_slopes = n_base == 12
        shared_shear = n_base in {12, 14}

        base = arr[:n_base]
        sx_L = float(base[8])
        sy_L = float(base[9])
        if shared_slopes:
            sx_R, sy_R = sx_L, sy_L
            rho_idx = 10
        else:
            sx_R = float(base[10])
            sy_R = float(base[11])
            rho_idx = 12
        if shared_shear:
            rx_L = float(base[rho_idx])
            ry_L = float(base[rho_idx + 1])
            rx_R, ry_R = rx_L, ry_L
        else:
            rx_L = float(base[rho_idx])
            ry_L = float(base[rho_idx + 1])
            rx_R = float(base[rho_idx + 2])
            ry_R = float(base[rho_idx + 3])

        warp_flat = arr[n_base:]
        if warp_level > 0 and n_warp > 0:
            per_axis = _n_warp_coeff_per_axis(warp_level)
            xi_L = tuple(float(v) for v in warp_flat[:per_axis])
            eta_L = tuple(float(v) for v in warp_flat[per_axis : 2 * per_axis])
            if shared_warp:
                xi_R, eta_R = xi_L, eta_L
            else:
                xi_R = tuple(float(v) for v in warp_flat[2 * per_axis : 3 * per_axis])
                eta_R = tuple(float(v) for v in warp_flat[3 * per_axis : 4 * per_axis])
        else:
            xi_L = eta_L = xi_R = eta_R = ()

        return cls(
            f_obj_mm=float(base[0]),
            working_distance_mm=float(base[1]),
            b_mm=float(base[2]),
            cx_principal_px=float(base[3]),
            cy_principal_px=float(base[4]),
            pixel_pitch_mm=pixel_pitch_mm,
            f_angular_mm=float(base[5]),
            theta_convergence_half_rad=float(base[6]),
            d_y_common=float(base[7]),
            s_x_L=sx_L,
            s_y_L=sy_L,
            s_x_R=sx_R,
            s_y_R=sy_R,
            shared_slopes=shared_slopes,
            rho_x_L=rx_L,
            rho_y_L=ry_L,
            rho_x_R=rx_R,
            rho_y_R=ry_R,
            shared_shear=shared_shear,
            image_size=image_size,
            warp_xi_L=xi_L,
            warp_eta_L=eta_L,
            warp_xi_R=xi_R,
            warp_eta_R=eta_R,
            warp_level=warp_level,
            shared_warp=shared_warp,
        )

    def parameter_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "f_obj_mm": float(self.f_obj_mm),
            "working_distance_mm": float(self.working_distance_mm),
            "b_mm": float(self.b_mm),
            "cx_principal_px": float(self.cx_principal_px),
            "cy_principal_px": float(self.cy_principal_px),
            "f_angular_mm": float(self.f_angular_mm),
            "theta_convergence_half_rad": float(self.theta_convergence_half_rad),
            "theta_convergence_half_deg": float(np.degrees(float(self.theta_convergence_half_rad))),
            "d_y_common": float(self.d_y_common),
            "warp_level": int(self.warp_level),
            "shared_warp": bool(self.shared_warp),
        }
        free: dict[str, float] = {}
        free["s_x_L"] = float(self.s_x_L)
        free["s_y_L"] = float(self.s_y_L)
        if not self.shared_slopes:
            free["s_x_R"] = float(self.s_x_R)
            free["s_y_R"] = float(self.s_y_R)
        if self.shared_shear:
            free["rho_x"] = float(self.rho_x_L)
            free["rho_y"] = float(self.rho_y_L)
        else:
            free["rho_x_L"] = float(self.rho_x_L)
            free["rho_y_L"] = float(self.rho_y_L)
            free["rho_x_R"] = float(self.rho_x_R)
            free["rho_y_R"] = float(self.rho_y_R)
        d["free"] = free

        terms = _poly_terms_2d(int(self.warp_level))
        for ch, xi_tup, eta_tup in [
            ("L", self.warp_xi_L, self.warp_eta_L),
            ("R", self.warp_xi_R, self.warp_eta_R),
        ]:
            for i, (pu, pv) in enumerate(terms):
                if i < len(xi_tup):
                    d[f"warp_xi_{ch}_u{pu}v{pv}"] = float(xi_tup[i])
                    d[f"warp_eta_{ch}_u{pu}v{pv}"] = float(eta_tup[i])
        return d

    def channel(self, channel: Literal["left", "right"]) -> CMOWarpedChannelModel:
        return CMOWarpedChannelModel(rig=self, channel=channel)

    def ray(self, u: Array, v: Array, channel: Literal["left", "right"]) -> tuple[Array, Array]:
        uu, vv = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        )
        shape = uu.shape
        uf, vf = uu.reshape(-1), vv.reshape(-1)

        if self.warp_level > 0:
            xi_c, eta_c = self._get_warp_coeffs(channel)
            xi = _polyval_2d(uf, vf, xi_c, int(self.warp_level))
            eta = _polyval_2d(uf, vf, eta_c, int(self.warp_level))
        else:
            xi, eta = uf.astype(np.float64), vf.astype(np.float64)

        f_ang = float(self.f_angular_mm) if float(self.f_angular_mm) > 0.1 else float(self.f_obj_mm)
        pp = float(self.pixel_pitch_mm) / f_ang
        tilde_u = (xi - float(self.cx_principal_px)) * pp
        tilde_v = (eta - float(self.cy_principal_px)) * pp

        dir_sign = 1.0 if channel == "left" else -1.0
        sin_th = float(np.sin(float(self.theta_convergence_half_rad)))
        cos_th = float(np.cos(float(self.theta_convergence_half_rad)))
        dy = float(self.d_y_common)
        d0 = np.column_stack(
            [
                np.full_like(uf, dir_sign * sin_th),
                np.full_like(uf, dy),
                np.full_like(uf, cos_th),
            ]
        )

        sx_ch = float(self.s_x_L) if channel == "left" else float(self.s_x_R)
        sy_ch = float(self.s_y_L) if channel == "left" else float(self.s_y_R)
        sxy_ch = float(self.s_xy_L) if channel == "left" else float(self.s_xy_R)
        syx_ch = float(self.s_yx_L) if channel == "left" else float(self.s_yx_R)
        qx_ch = float(self.q_xx_L) if channel == "left" else float(self.q_xx_R)
        qy_ch = float(self.q_yy_L) if channel == "left" else float(self.q_yy_R)

        umean2 = float(np.mean(tilde_u**2))
        vmean2 = float(np.mean(tilde_v**2))

        d_raw = np.column_stack(
            [
                d0[:, 0] + sx_ch * tilde_u + sxy_ch * tilde_v + qx_ch * (tilde_u**2 - umean2),
                d0[:, 1] + sy_ch * tilde_v + syx_ch * tilde_u + qy_ch * (tilde_v**2 - vmean2),
                d0[:, 2],
            ]
        )
        directions = _normalize(d_raw)

        pupil_sign = -1.0 if channel == "left" else 1.0
        z_pupil = float(self.working_distance_mm) - float(self.f_obj_mm)
        rho_x = float(self.rho_x_L) if channel == "left" else float(self.rho_x_R)
        rho_y = float(self.rho_y_L) if channel == "left" else float(self.rho_y_R)

        O_rigid = np.column_stack(
            [
                np.full_like(uf, pupil_sign * 0.5 * float(self.b_mm)),
                np.zeros_like(uf),
                np.full_like(uf, z_pupil),
            ]
        )
        delta_O = np.column_stack([rho_x * tilde_u, rho_y * tilde_v, np.zeros_like(uf)])
        pupil = O_rigid + delta_O

        return pupil.reshape((*shape, 3)), directions.reshape((*shape, 3))


@dataclass(frozen=True)
class CMOWarpedChannelModel:
    """Single-channel facade for a shared warped CMO rig."""

    rig: CMOWarpedStereoModel
    channel: Literal["left", "right"]
    name: str = "cmo_warped_channel"
    is_stereo_shared: bool = True

    @property
    def n_parameters(self) -> int:
        return self.rig.n_parameters

    def parameter_vector(self) -> Array:
        return self.rig.parameter_vector()

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs: Any) -> CMOWarpedChannelModel:
        rig = CMOWarpedStereoModel.from_parameter_vector(x, **kwargs)
        channel = kwargs.get("channel", "left")
        if channel not in {"left", "right"}:
            raise ValueError("channel must be 'left' or 'right'")
        return cls(rig=rig, channel=channel)

    def parameter_dict(self) -> dict[str, object]:
        return self.rig.parameter_dict()

    def ray(self, u: Array, v: Array) -> tuple[Array, Array]:
        return self.rig.ray(u, v, self.channel)


# ═══════════════════════════════════════════════════════════════════════
# Fitting and residual analysis
# ═══════════════════════════════════════════════════════════════════════


def fit_cmo_warped_model_to_rayfields(
    left_field,
    right_field,
    image_size: tuple[int, int],
    initial_parameters: Array,
    *,
    pixel_pitch_mm: float,
    z_planes: tuple[float, float] = (50.0, 250.0),
    grid_shape: tuple[int, int] = (17, 13),
    max_nfev: int = 500,
    warp_level: int = 0,
    shared_warp: bool = True,
) -> CMOPhysicalStereoFitResult:
    """Fit the warped CMO model to left/right measured rayfields."""
    x0 = np.asarray(initial_parameters, dtype=np.float64).reshape(-1)
    full = _grid_pixels(image_size, grid_shape)

    def model_at(x: Array) -> CMOWarpedStereoModel:
        return CMOWarpedStereoModel.from_parameter_vector(
            x,
            pixel_pitch_mm=pixel_pitch_mm,
            image_size=image_size,
            warp_level=warp_level,
            shared_warp=shared_warp,
        )

    def residuals(x: Array) -> Array:
        model = model_at(x)
        left = model.channel("left")
        right = model.channel("right")
        return np.concatenate(
            [
                rayfield_two_plane_residuals(left_field, left, full, z_planes=z_planes),
                rayfield_two_plane_residuals(right_field, right, full, z_planes=z_planes),
            ]
        )

    # Bounds
    tele_lo = [1.0, 1.0, 0.0, -np.inf, -np.inf, 1.0, 0.0, -0.3, -10.0, -10.0]
    tele_hi = [500.0, 1000.0, 200.0, np.inf, np.inf, 500.0, 0.5, 0.3, 10.0, 10.0]
    if x0.size > 10:
        n_extra = x0.size - 10
        tele_lo.extend([-np.inf] * n_extra)
        tele_hi.extend([np.inf] * n_extra)

    n_warp = _n_warp_coeff_total(warp_level, shared_warp)
    warp_lo: list[float] = []
    warp_hi: list[float] = []
    if n_warp > 0:
        for pu, pv in _poly_terms_2d(warp_level):
            deg = pu + pv
            lim = {0: 500.0, 1: 10.0, 2: 0.01, 3: 1e-4}.get(deg, 1e-4)
            warp_lo.append(-lim)
            warp_hi.append(lim)
        n_repeats = 2 if shared_warp else 4
        warp_lo = warp_lo * n_repeats
        warp_hi = warp_hi * n_repeats

    n_tele = x0.size - n_warp
    lower = np.array(tele_lo[:n_tele] + warp_lo, dtype=np.float64)
    upper = np.array(tele_hi[:n_tele] + warp_hi, dtype=np.float64)

    sol = least_squares(
        residuals,
        x0=x0,
        bounds=(lower, upper),
        loss="huber",
        f_scale=1.0,
        max_nfev=int(max_nfev),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )

    fitted = model_at(sol.x)
    left_res = rayfield_two_plane_residuals(
        left_field, fitted.channel("left"), full, z_planes=z_planes
    )
    right_res = rayfield_two_plane_residuals(
        right_field, fitted.channel("right"), full, z_planes=z_planes
    )
    combined = residuals(sol.x)
    rss = float(np.sum(combined * combined))
    n_res = int(combined.size)
    n_samples = int(full.shape[0] * 2)
    left_rms = _ray_rms(left_res)
    right_rms = _ray_rms(right_res)
    aic, bic = _aic_bic(rss, n_res, n_samples, fitted.n_parameters)

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


def compute_cmo_zernike_residuals(
    cmo_model,
    zernike_left,
    zernike_right,
    grid_shape: tuple[int, int] = (17, 13),
    image_size: tuple[int, int] | None = None,
    zernike_order: int = 4,
) -> dict[str, Any]:
    """Compute direction/moment residuals between a CMO model and Zernike rayfield.

    Returns delta_d (deg), delta_m (mm), and Zernike modal projection
    of the direction residual field.
    """
    if image_size is None:
        raise ValueError("image_size is required")
    W, H = int(image_size[0]), int(image_size[1])
    pixels = _grid_pixels(image_size, grid_shape)
    uf, vf = pixels[:, 0], pixels[:, 1]

    OzL, dzL = zernike_left.ray(uf, vf)
    OzR, dzR = zernike_right.ray(uf, vf)

    if hasattr(cmo_model, "channel"):
        OmL, dmL = cmo_model.ray(uf, vf, "left")
        OmR, dmR = cmo_model.ray(uf, vf, "right")
    else:
        OmL, dmL = cmo_model.ray(uf, vf)
        OmR, dmR = OmL, dmL

    dot_L = np.clip(np.sum(dzL * dmL, axis=1), -1.0, 1.0)
    dot_R = np.clip(np.sum(dzR * dmR, axis=1), -1.0, 1.0)
    ang_L = np.degrees(np.arccos(dot_L))
    ang_R = np.degrees(np.arccos(dot_R))

    mzL = np.cross(OzL, dzL)
    mmL = np.cross(OmL, dmL)
    mzR = np.cross(OzR, dzR)
    mmR = np.cross(OmR, dmR)
    mom_L = np.linalg.norm(mzL - mmL, axis=1)
    mom_R = np.linalg.norm(mzR - mmR, axis=1)

    delta_d_L = dzL - dmL
    delta_d_R = dzR - dmR

    from stereocomplex.core.model_compact.zernike import zernike_modes, eval_real_zernike

    modes = zernike_modes(int(zernike_order))
    xi = 2.0 * uf / float(W - 1) - 1.0
    zeta = 2.0 * vf / float(H - 1) - 1.0
    rho = np.sqrt(xi * xi + zeta * zeta) / np.sqrt(2.0)
    theta = np.arctan2(zeta, xi)
    B = np.empty((uf.size, len(modes)), dtype=np.float64)
    for j, mode in enumerate(modes):
        B[:, j] = eval_real_zernike(mode, rho, theta)
    BtB_inv = np.linalg.inv(B.T @ B + 1e-10 * np.eye(B.shape[1]))
    B_pinv = BtB_inv @ B.T

    projection: dict[str, float] = {}
    for ch_label, dd in [("L", delta_d_L), ("R", delta_d_R)]:
        for comp, cl in enumerate(["dx", "dy", "dz"]):
            c = B_pinv @ dd[:, comp]
            for j, mode in enumerate(modes):
                projection[f"{ch_label}_n{mode.n}_m{mode.m}_{mode.kind}_{cl}"] = float(c[j])

    Bsq = np.sum(B**2, axis=0)
    var_total_L = float(np.sum(delta_d_L**2))
    var_total_R = float(np.sum(delta_d_R**2))
    mode_contribs = []
    for j, mode in enumerate(modes):
        cLx = projection.get(f"L_n{mode.n}_m{mode.m}_{mode.kind}_dx", 0.0)
        cLy = projection.get(f"L_n{mode.n}_m{mode.m}_{mode.kind}_dy", 0.0)
        cLz = projection.get(f"L_n{mode.n}_m{mode.m}_{mode.kind}_dz", 0.0)
        cRx = projection.get(f"R_n{mode.n}_m{mode.m}_{mode.kind}_dx", 0.0)
        cRy = projection.get(f"R_n{mode.n}_m{mode.m}_{mode.kind}_dy", 0.0)
        cRz = projection.get(f"R_n{mode.n}_m{mode.m}_{mode.kind}_dz", 0.0)
        frac_L = float((cLx**2 + cLy**2 + cLz**2) * Bsq[j]) / max(var_total_L, 1e-16)
        frac_R = float((cRx**2 + cRy**2 + cRz**2) * Bsq[j]) / max(var_total_R, 1e-16)
        mode_contribs.append(
            {
                "mode": f"Z_{mode.n}^{mode.m}({mode.kind})",
                "n": mode.n,
                "m": mode.m,
                "kind": mode.kind,
                "frac_var_L": frac_L,
                "frac_var_R": frac_R,
            }
        )
    mode_contribs.sort(key=lambda x: abs(x["frac_var_L"]) + abs(x["frac_var_R"]), reverse=True)

    return {
        "dir_rms_deg_L": float(np.sqrt(np.mean(ang_L**2))),
        "dir_rms_deg_R": float(np.sqrt(np.mean(ang_R**2))),
        "dir_rms_deg_total": float(np.sqrt(np.mean(np.concatenate([ang_L, ang_R]) ** 2))),
        "mom_rms_mm_L": float(np.sqrt(np.mean(mom_L**2))),
        "mom_rms_mm_R": float(np.sqrt(np.mean(mom_R**2))),
        "mom_rms_mm_total": float(np.sqrt(np.mean(np.concatenate([mom_L, mom_R]) ** 2))),
        "zernike_projection": projection,
        "top_direction_modes": mode_contribs[:8],
        "all_mode_contributions": mode_contribs,
        "n_points": int(pixels.shape[0]),
        "grid_shape": grid_shape,
    }


__all__ = [
    "CMOPhysicalChannelModel",
    "CMOPhysicalStereoFitResult",
    "CMOPhysicalStereoModel",
    "CMOTelecentricChannelModel",
    "CMOTelecentricNModel",
    "CMOTelecentricStereoModel",
    "fit_cmo_physical_stereo_model_to_rayfields",
    "fit_cmo_telecentric_model_to_rayfields",
]
