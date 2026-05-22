from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from stereocomplex.physics.central_models import (
    brown_conrady_distort_normalized,
    undistort_brown_normalized,
)

try:  # pragma: no cover - optional rendering backend
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional image writer
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


Array = np.ndarray


def normalize_vectors(v: Array, eps: float = 1e-15) -> Array:
    """Normalize vectors along the last axis."""
    arr = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(n, float(eps))


def rotx(a: float) -> Array:
    ca, sa = math.cos(float(a)), math.sin(float(a))
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]],
        dtype=np.float64,
    )


def roty(a: float) -> Array:
    ca, sa = math.cos(float(a)), math.sin(float(a))
    return np.array(
        [[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]],
        dtype=np.float64,
    )


def rotz(a: float) -> Array:
    ca, sa = math.cos(float(a)), math.sin(float(a))
    return np.array(
        [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def pose_from_euler_xyz(
    rx: float,
    ry: float,
    rz: float,
    t_xyz: tuple[float, float, float],
) -> CMOPlanePose:
    """Create a calibration-plane pose in the CMO/world frame."""
    return CMOPlanePose(
        R=rotz(rz) @ roty(ry) @ rotx(rx),
        t=np.asarray(t_xyz, dtype=np.float64),
    )


@dataclass(frozen=True)
class CMOPlanePose:
    """Rigid pose of a planar target in the CMO/world frame."""

    R: Array
    t: Array

    def local_to_world(self, xy_plane_mm: Array) -> Array:
        xy = np.asarray(xy_plane_mm, dtype=np.float64).reshape(-1, 2)
        xyz_local = np.column_stack(
            [xy[:, 0], xy[:, 1], np.zeros(xy.shape[0], dtype=np.float64)]
        )
        return (np.asarray(self.R, dtype=np.float64) @ xyz_local.T).T + np.asarray(
            self.t, dtype=np.float64
        )[None, :]

    @property
    def normal_world(self) -> Array:
        return np.asarray(self.R, dtype=np.float64) @ np.array([0.0, 0.0, 1.0])

    def world_to_local(self, xyz_world: Array) -> Array:
        x = np.asarray(xyz_world, dtype=np.float64) - np.asarray(self.t, dtype=np.float64)
        shp = x.shape
        flat = x.reshape(-1, 3)
        return (np.asarray(self.R, dtype=np.float64).T @ flat.T).T.reshape(shp)


def _pose_to_vector(pose: CMOPlanePose) -> Array:
    return np.r_[
        Rotation.from_matrix(np.asarray(pose.R, dtype=np.float64).reshape(3, 3)).as_rotvec(),
        np.asarray(pose.t, dtype=np.float64).reshape(3),
    ]


def _vector_to_pose(params: Array) -> CMOPlanePose:
    arr = np.asarray(params, dtype=np.float64).reshape(6)
    return CMOPlanePose(R=Rotation.from_rotvec(arr[:3]).as_matrix(), t=arr[3:6].copy())


@dataclass(frozen=True)
class CMOIntrinsics:
    """Pixel intrinsics for one CMO channel."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_focal_and_pitch(
        cls,
        width: int,
        height: int,
        focal_mm: float,
        pitch_um: float,
    ) -> CMOIntrinsics:
        f_px = float(focal_mm) * 1000.0 / float(pitch_um)
        return cls(
            width=int(width),
            height=int(height),
            fx=f_px,
            fy=f_px,
            cx=(float(width) - 1.0) / 2.0,
            cy=(float(height) - 1.0) / 2.0,
        )

    def as_K(self) -> Array:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def pixel_grid(self) -> tuple[Array, Array]:
        u, v = np.meshgrid(
            np.arange(int(self.width), dtype=np.float64),
            np.arange(int(self.height), dtype=np.float64),
            indexing="xy",
        )
        return u, v

    def pixel_to_norm(self, u: Array, v: Array) -> tuple[Array, Array]:
        return (np.asarray(u, dtype=np.float64) - self.cx) / self.fx, (
            np.asarray(v, dtype=np.float64) - self.cy
        ) / self.fy

    def norm_to_pixel(self, x: Array, y: Array) -> Array:
        u = self.fx * np.asarray(x, dtype=np.float64) + self.cx
        v = self.fy * np.asarray(y, dtype=np.float64) + self.cy
        return np.stack([u, v], axis=-1)


@dataclass(frozen=True)
class BrownConrady:
    """OpenCV-like radial/tangential distortion in normalized coordinates."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def distort(self, x: Array, y: Array) -> tuple[Array, Array]:
        return brown_conrady_distort_normalized(x, y, self.k1, self.k2, self.p1, self.p2, self.k3)

    def undistort(self, xd: Array, yd: Array, iterations: int = 10) -> tuple[Array, Array]:
        return undistort_brown_normalized(
            xd,
            yd,
            self.k1,
            self.k2,
            self.p1,
            self.p2,
            self.k3,
            n_iter=iterations,
        )


@dataclass(frozen=True)
class PolynomialRayAberration:
    """Small polynomial angular perturbation in normalized ray coordinates."""

    coeff_x: dict[str, float] = field(default_factory=dict)
    coeff_y: dict[str, float] = field(default_factory=dict)

    def delta(self, x: Array, y: Array) -> tuple[Array, Array]:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        terms = {
            "1": np.ones_like(x_arr),
            "x": x_arr,
            "y": y_arr,
            "x2": x_arr * x_arr,
            "xy": x_arr * y_arr,
            "y2": y_arr * y_arr,
            "x3": x_arr * x_arr * x_arr,
            "x2y": x_arr * x_arr * y_arr,
            "xy2": x_arr * y_arr * y_arr,
            "y3": y_arr * y_arr * y_arr,
        }
        dx = np.zeros_like(x_arr, dtype=np.float64)
        dy = np.zeros_like(y_arr, dtype=np.float64)
        for name, value in self.coeff_x.items():
            if name not in terms:
                raise ValueError(f"Unknown polynomial term for coeff_x: {name}")
            dx = dx + float(value) * terms[name]
        for name, value in self.coeff_y.items():
            if name not in terms:
                raise ValueError(f"Unknown polynomial term for coeff_y: {name}")
            dy = dy + float(value) * terms[name]
        return dx, dy

    def add(self, other: PolynomialRayAberration) -> PolynomialRayAberration:
        coeff_x = dict(self.coeff_x)
        coeff_y = dict(self.coeff_y)
        for key, value in other.coeff_x.items():
            coeff_x[key] = coeff_x.get(key, 0.0) + float(value)
        for key, value in other.coeff_y.items():
            coeff_y[key] = coeff_y.get(key, 0.0) + float(value)
        return PolynomialRayAberration(coeff_x=coeff_x, coeff_y=coeff_y)


@dataclass(frozen=True)
class SensorWarp:
    """Low-order image-plane deformation in pixels."""

    du_coeff_px: dict[str, float] = field(default_factory=dict)
    dv_coeff_px: dict[str, float] = field(default_factory=dict)

    def delta_px(self, uv: Array, intr: CMOIntrinsics) -> Array:
        arr = np.asarray(uv, dtype=np.float64)
        u = arr[..., 0]
        v = arr[..., 1]
        x = (u - intr.cx) / max(abs(intr.cx), 1.0)
        y = (v - intr.cy) / max(abs(intr.cy), 1.0)
        terms = {
            "1": np.ones_like(x),
            "x": x,
            "y": y,
            "x2": x * x,
            "xy": x * y,
            "y2": y * y,
            "x3": x * x * x,
            "x2y": x * x * y,
            "xy2": x * y * y,
            "y3": y * y * y,
        }
        du = np.zeros_like(x, dtype=np.float64)
        dv = np.zeros_like(y, dtype=np.float64)
        for name, value in self.du_coeff_px.items():
            if name not in terms:
                raise ValueError(f"Unknown polynomial term for du_coeff_px: {name}")
            du = du + float(value) * terms[name]
        for name, value in self.dv_coeff_px.items():
            if name not in terms:
                raise ValueError(f"Unknown polynomial term for dv_coeff_px: {name}")
            dv = dv + float(value) * terms[name]
        return np.stack([du, dv], axis=-1)


@dataclass(frozen=True)
class Vignetting:
    """Simple multiplicative pupil/sensor shading model."""

    strength: float = 0.0
    floor: float = 0.25
    x_shift: float = 0.0
    y_shift: float = 0.0

    def gain(self, intr: CMOIntrinsics) -> Array:
        u, v = intr.pixel_grid()
        x = (u - intr.cx) / max(abs(intr.cx), 1.0) - self.x_shift
        y = (v - intr.cy) / max(abs(intr.cy), 1.0) - self.y_shift
        g = 1.0 - float(self.strength) * (x * x + y * y)
        return np.clip(g, float(self.floor), 1.0)


@dataclass(frozen=True)
class CMOChannelSpec:
    """One effective CMO stereo channel."""

    name: Literal["left", "right"]
    intrinsics: CMOIntrinsics
    origin_world_mm: tuple[float, float, float]
    R_cam_to_world: Array = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    distortion: BrownConrady = field(default_factory=BrownConrady)
    differential_aberration: PolynomialRayAberration = field(
        default_factory=PolynomialRayAberration
    )
    sensor_warp: SensorWarp = field(default_factory=SensorWarp)
    vignetting: Vignetting = field(default_factory=Vignetting)

    @property
    def origin(self) -> Array:
        return np.asarray(self.origin_world_mm, dtype=np.float64)


@dataclass(frozen=True)
class CMOStereoSpec:
    """Full CMO stereo model with common and differential ray aberrations."""

    left: CMOChannelSpec
    right: CMOChannelSpec
    common_aberration: PolynomialRayAberration = field(default_factory=PolynomialRayAberration)

    @classmethod
    def symmetric_default(
        cls,
        width: int = 1280,
        height: int = 960,
        focal_mm: float = 25.0,
        pitch_um: float = 3.45,
        baseline_mm: float = 5.0,
        common_aberration: PolynomialRayAberration | None = None,
        left_distortion: BrownConrady | None = None,
        right_distortion: BrownConrady | None = None,
    ) -> CMOStereoSpec:
        intr = CMOIntrinsics.from_focal_and_pitch(width, height, focal_mm, pitch_um)
        b2 = 0.5 * float(baseline_mm)
        return cls(
            left=CMOChannelSpec(
                name="left",
                intrinsics=intr,
                origin_world_mm=(-b2, 0.0, 0.0),
                distortion=left_distortion or BrownConrady(),
                vignetting=Vignetting(strength=0.15, floor=0.55, x_shift=-0.05),
            ),
            right=CMOChannelSpec(
                name="right",
                intrinsics=intr,
                origin_world_mm=(+b2, 0.0, 0.0),
                distortion=right_distortion or BrownConrady(),
                vignetting=Vignetting(strength=0.15, floor=0.55, x_shift=+0.05),
            ),
            common_aberration=common_aberration or PolynomialRayAberration(),
        )

    def channels(self) -> tuple[CMOChannelSpec, CMOChannelSpec]:
        return self.left, self.right


@dataclass(frozen=True)
class CMOChannelRayField:
    """Physical CMO channel rayfield compatible with ray-space fitting tools."""

    channel: CMOChannelSpec
    common_aberration: PolynomialRayAberration = field(default_factory=PolynomialRayAberration)
    name: str = "cmo_channel"

    @property
    def n_parameters(self) -> int:
        return 0

    def parameter_vector(self) -> Array:
        return np.zeros(0, dtype=np.float64)

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs) -> CMOChannelRayField:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        if arr.size:
            raise ValueError("CMOChannelRayField expects zero parameters")
        return cls(
            channel=kwargs["channel"],
            common_aberration=kwargs.get("common_aberration", PolynomialRayAberration()),
        )

    def parameter_dict(self) -> dict[str, float]:
        return {}

    def ray(self, u: Array, v: Array) -> tuple[Array, Array]:
        uu, vv = np.broadcast_arrays(
            np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64)
        )
        shape = uu.shape
        uf = uu.reshape(-1)
        vf = vv.reshape(-1)
        uv = np.column_stack([uf, vf])
        uv_ideal = _inverse_sensor_warp_pixels(
            uv, self.channel.sensor_warp, self.channel.intrinsics
        )
        x_dist, y_dist = self.channel.intrinsics.pixel_to_norm(uv_ideal[:, 0], uv_ideal[:, 1])
        x, y = self.channel.distortion.undistort(x_dist, y_dist)
        aberration = self.common_aberration.add(self.channel.differential_aberration)
        dx, dy = aberration.delta(x, y)
        d_cam = normalize_vectors(np.column_stack([x + dx, y + dy, np.ones_like(x)]))
        R = np.asarray(self.channel.R_cam_to_world, dtype=np.float64)
        d_world = (R @ d_cam.T).T
        origins = np.broadcast_to(self.channel.origin[None, :], d_world.shape).copy()
        return origins.reshape(shape + (3,)), d_world.reshape(shape + (3,))


@dataclass(frozen=True)
class NonCentralPolynomialChannelModel:
    """Fittable effective channel model for ray-space model selection.

    This is a generic non-central surrogate: an effective sub-pupil origin
    (x, y, z free), a central Brown-Conrady distortion, and a low-order
    polynomial angular aberration field.  It can represent a wide range of
    smooth rayfields — including CMO-like ones — without encoding any
    specific shared-objective constraint.

    """

    K: Array
    image_size: tuple[int, int]
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    origin_z_mm: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    aberration_coeff_x: tuple[float, ...] = ()
    aberration_coeff_y: tuple[float, ...] = ()
    aberration_terms: tuple[str, ...] = ("x", "y", "x2", "xy", "y2")
    name: str = "polynomial_surrogate_channel"

    @property
    def n_parameters(self) -> int:
        return 8 + 2 * len(self.aberration_terms)

    @classmethod
    def default_terms(cls) -> tuple[str, ...]:
        return ("x", "y", "x2", "xy", "y2")

    def parameter_vector(self) -> Array:
        cx = self._coeff_array(self.aberration_coeff_x)
        cy = self._coeff_array(self.aberration_coeff_y)
        return np.concatenate(
            [
                np.array(
                    [
                        self.origin_x_mm,
                        self.origin_y_mm,
                        self.origin_z_mm,
                        self.k1,
                        self.k2,
                        self.p1,
                        self.p2,
                        self.k3,
                    ],
                    dtype=np.float64,
                ),
                cx,
                cy,
            ]
        )

    @classmethod
    def from_parameter_vector(cls, x: Array, **kwargs) -> NonCentralPolynomialChannelModel:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        terms = tuple(kwargs.get("aberration_terms", cls.default_terms()))
        image_size = kwargs.get("cmo_image_size", kwargs.get("image_size"))
        if image_size is None:
            raise ValueError("NonCentralPolynomialChannelModel requires cmo_image_size")
        expected = 8 + 2 * len(terms)
        if arr.size != expected:
            raise ValueError(f"NonCentralPolynomialChannelModel expects {expected} parameters")
        return cls(
            K=np.asarray(kwargs["K"], dtype=np.float64).reshape(3, 3),
            image_size=tuple(image_size),
            origin_x_mm=float(arr[0]),
            origin_y_mm=float(arr[1]),
            origin_z_mm=float(arr[2]),
            k1=float(arr[3]),
            k2=float(arr[4]),
            p1=float(arr[5]),
            p2=float(arr[6]),
            k3=float(arr[7]),
            aberration_coeff_x=tuple(float(v) for v in arr[8 : 8 + len(terms)]),
            aberration_coeff_y=tuple(float(v) for v in arr[8 + len(terms) :]),
            aberration_terms=terms,
        )

    def parameter_dict(self) -> dict[str, float]:
        params = {
            "origin_x_mm": float(self.origin_x_mm),
            "origin_y_mm": float(self.origin_y_mm),
            "origin_z_mm": float(self.origin_z_mm),
            "k1": float(self.k1),
            "k2": float(self.k2),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "k3": float(self.k3),
        }
        for name, value in zip(
            self.aberration_terms, self._coeff_array(self.aberration_coeff_x), strict=True
        ):
            params[f"aberr_x_{name}"] = float(value)
        for name, value in zip(
            self.aberration_terms, self._coeff_array(self.aberration_coeff_y), strict=True
        ):
            params[f"aberr_y_{name}"] = float(value)
        return params

    def ray(self, u: Array, v: Array) -> tuple[Array, Array]:
        intr = self._intrinsics()
        channel = CMOChannelSpec(
            name="left",
            intrinsics=intr,
            origin_world_mm=(
                float(self.origin_x_mm), float(self.origin_y_mm), float(self.origin_z_mm),
            ),
            distortion=BrownConrady(self.k1, self.k2, self.p1, self.p2, self.k3),
            differential_aberration=self._aberration(),
            vignetting=Vignetting(strength=0.0),
        )
        return CMOChannelRayField(channel=channel).ray(u, v)

    def _intrinsics(self) -> CMOIntrinsics:
        K_arr = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        width, height = self.image_size
        return CMOIntrinsics(
            width=int(width),
            height=int(height),
            fx=float(K_arr[0, 0]),
            fy=float(K_arr[1, 1]),
            cx=float(K_arr[0, 2]),
            cy=float(K_arr[1, 2]),
        )

    def _coeff_array(self, coeffs: tuple[float, ...]) -> Array:
        if not coeffs:
            return np.zeros(len(self.aberration_terms), dtype=np.float64)
        arr = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        if arr.size != len(self.aberration_terms):
            raise ValueError("aberration coefficient length must match aberration_terms")
        return arr

    def _aberration(self) -> PolynomialRayAberration:
        cx = self._coeff_array(self.aberration_coeff_x)
        cy = self._coeff_array(self.aberration_coeff_y)
        return PolynomialRayAberration(
            coeff_x={
                name: float(value)
                for name, value in zip(self.aberration_terms, cx, strict=True)
            },
            coeff_y={
                name: float(value)
                for name, value in zip(self.aberration_terms, cy, strict=True)
            },
        )


def polynomial_channel_parameters_from_spec(
    channel: CMOChannelSpec,
    common_aberration: PolynomialRayAberration | None = None,
    aberration_terms: tuple[str, ...] | None = None,
) -> Array:
    """Return the effective `NonCentralPolynomialChannelModel` vector for a channel.

    The fittable CMO model is per-channel. Its polynomial aberration is the sum
    of common CMO aberration and channel differential aberration.
    """
    terms = tuple(aberration_terms or NonCentralPolynomialChannelModel.default_terms())
    effective = (
        common_aberration or PolynomialRayAberration()
    ).add(channel.differential_aberration)
    dx, dy = [], []
    for term in terms:
        dx.append(float(effective.coeff_x.get(term, 0.0)))
        dy.append(float(effective.coeff_y.get(term, 0.0)))
    origin = channel.origin
    return np.r_[
        origin[0],
        origin[1],
        origin[2],
        channel.distortion.k1,
        channel.distortion.k2,
        channel.distortion.p1,
        channel.distortion.p2,
        channel.distortion.k3,
        np.asarray(dx, dtype=np.float64),
        np.asarray(dy, dtype=np.float64),
    ].astype(np.float64)


@dataclass(frozen=True)
class CMORayfieldBundleAdjustmentResult:
    """Result of fitting effective CMO channels and board poses from rayfields."""

    left_model: NonCentralPolynomialChannelModel
    right_model: NonCentralPolynomialChannelModel
    poses: list[CMOPlanePose]
    success: bool
    message: str
    n_observations: int
    incidence_rms_mm: float
    incidence_p95_mm: float
    left_rayfield_rms_mm: float
    right_rayfield_rms_mm: float
    parameter_vector: Array
    pose_vectors: Array

    def parameter_summary(self) -> dict[str, dict[str, float]]:
        return {
            "left": self.left_model.parameter_dict(),
            "right": self.right_model.parameter_dict(),
        }


def fit_cmo_stereo_model_and_poses_from_zernike_rayfields(
    left_field,
    right_field,
    K: Array,
    image_size: tuple[int, int],
    object_points: Array | list[Array],
    left_pixels: list[Array],
    right_pixels: list[Array],
    pose_initials: list[CMOPlanePose],
    *,
    initial_left_parameters: Array | None = None,
    initial_right_parameters: Array | None = None,
    parameter_bounds: tuple[Array, Array] | None = None,
    aberration_terms: tuple[str, ...] | None = None,
    z_planes: tuple[float, float] = (80.0, 260.0),
    incidence_weight: float = 1.0,
    rayfield_weight: float = 0.25,
    pose_regularization: float = 0.0,
    max_nfev: int = 800,
    robust_loss: str = "huber",
) -> CMORayfieldBundleAdjustmentResult:
    """Fit effective CMO channel parameters and board poses from measured rayfields.

    The measured Zernike rayfields provide the pixel-to-line observations. The
    CMO model is constrained by two terms: ChArUco point-to-line incidence and
    a ray-space anchor to the measured left/right Zernike fields.
    """
    from scipy.optimize import least_squares  # type: ignore
    from stereocomplex.physics.parallel_plate_fit import rayfield_two_plane_residuals

    terms = tuple(aberration_terms or NonCentralPolynomialChannelModel.default_terms())
    K_arr = np.asarray(K, dtype=np.float64).reshape(3, 3)
    n_params = 8 + 2 * len(terms)
    if initial_left_parameters is None:
        initial_left_parameters = np.zeros(n_params, dtype=np.float64)
    if initial_right_parameters is None:
        initial_right_parameters = np.zeros(n_params, dtype=np.float64)
    left0 = np.asarray(initial_left_parameters, dtype=np.float64).reshape(-1)
    right0 = np.asarray(initial_right_parameters, dtype=np.float64).reshape(-1)
    if left0.size != n_params or right0.size != n_params:
        raise ValueError(f"CMO parameter vectors must have length {n_params}")
    if len(left_pixels) != len(right_pixels) or len(left_pixels) != len(pose_initials):
        raise ValueError("left_pixels, right_pixels and pose_initials must have the same length")

    if isinstance(object_points, list):
        object_per_frame = [
            np.asarray(obj, dtype=np.float64).reshape(-1, 2) for obj in object_points
        ]
    else:
        common_obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 2)
        object_per_frame = [common_obj for _ in left_pixels]
    if len(object_per_frame) != len(left_pixels):
        raise ValueError("object_points list must match the number of frames")

    left_obs = [np.asarray(pix, dtype=np.float64).reshape(-1, 2) for pix in left_pixels]
    right_obs = [np.asarray(pix, dtype=np.float64).reshape(-1, 2) for pix in right_pixels]
    for idx, (obj, left, right) in enumerate(
        zip(object_per_frame, left_obs, right_obs, strict=True)
    ):
        if obj.shape[0] != left.shape[0] or obj.shape[0] != right.shape[0]:
            raise ValueError(f"frame {idx} object/pixel counts do not match")

    pose0 = np.concatenate([_pose_to_vector(pose) for pose in pose_initials])
    x0 = np.r_[left0, right0, pose0]
    if parameter_bounds is None:
        lo_model = np.r_[
            [-12.0, -12.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0],
            -0.05 * np.ones(2 * len(terms)),
        ]
        hi_model = np.r_[
            [+12.0, +12.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0],
            +0.05 * np.ones(2 * len(terms)),
        ]
    else:
        lo_model, hi_model = (np.asarray(v, dtype=np.float64).reshape(-1) for v in parameter_bounds)
    if lo_model.size != n_params or hi_model.size != n_params:
        raise ValueError(f"parameter_bounds must contain vectors of length {n_params}")
    pose_lo = pose0 - np.inf
    pose_hi = pose0 + np.inf
    bounds = (np.r_[lo_model, lo_model, pose_lo], np.r_[hi_model, hi_model, pose_hi])

    def unpack(x: Array) -> tuple[
        NonCentralPolynomialChannelModel, NonCentralPolynomialChannelModel,
        list[CMOPlanePose], Array,
    ]:
        arr = np.asarray(x, dtype=np.float64).reshape(-1)
        left_params = arr[:n_params]
        right_params = arr[n_params : 2 * n_params]
        pose_params = arr[2 * n_params :]
        left_model = NonCentralPolynomialChannelModel.from_parameter_vector(
            left_params,
            K=K_arr,
            cmo_image_size=image_size,
            aberration_terms=terms,
        )
        right_model = NonCentralPolynomialChannelModel.from_parameter_vector(
            right_params,
            K=K_arr,
            cmo_image_size=image_size,
            aberration_terms=terms,
        )
        poses = [_vector_to_pose(pose_params[6 * i : 6 * i + 6]) for i in range(len(pose_initials))]
        return left_model, right_model, poses, pose_params

    support_left = np.vstack(left_obs) if left_obs else np.zeros((0, 2), dtype=np.float64)
    support_right = np.vstack(right_obs) if right_obs else np.zeros((0, 2), dtype=np.float64)
    sqrt_pose_reg = math.sqrt(max(float(pose_regularization), 0.0))

    def residuals(x: Array) -> Array:
        left_model, right_model, poses, pose_params = unpack(x)
        blocks: list[Array] = []
        for pose, obj, uv_l, uv_r in zip(poses, object_per_frame, left_obs, right_obs, strict=True):
            points = pose.local_to_world(obj)
            O_l, d_l = left_model.ray(uv_l[:, 0], uv_l[:, 1])
            O_r, d_r = right_model.ray(uv_r[:, 0], uv_r[:, 1])
            blocks.append(
                float(incidence_weight)
                * np.cross(points - O_l.reshape(-1, 3), d_l.reshape(-1, 3)).reshape(-1)
            )
            blocks.append(
                float(incidence_weight)
                * np.cross(points - O_r.reshape(-1, 3), d_r.reshape(-1, 3)).reshape(-1)
            )
        if rayfield_weight > 0:
            blocks.append(
                float(rayfield_weight)
                * rayfield_two_plane_residuals(
                    left_field, left_model, support_left, z_planes=z_planes
                )
            )
            blocks.append(
                float(rayfield_weight)
                * rayfield_two_plane_residuals(
                    right_field, right_model, support_right, z_planes=z_planes
                )
            )
        if sqrt_pose_reg > 0:
            blocks.append(sqrt_pose_reg * (pose_params - pose0))
        return np.concatenate(blocks)

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
    left_model, right_model, poses, pose_params = unpack(sol.x)

    incidence_blocks: list[Array] = []
    for pose, obj, uv_l, uv_r in zip(poses, object_per_frame, left_obs, right_obs, strict=True):
        points = pose.local_to_world(obj)
        O_l, d_l = left_model.ray(uv_l[:, 0], uv_l[:, 1])
        O_r, d_r = right_model.ray(uv_r[:, 0], uv_r[:, 1])
        incidence_blocks.append(np.cross(points - O_l.reshape(-1, 3), d_l.reshape(-1, 3)))
        incidence_blocks.append(np.cross(points - O_r.reshape(-1, 3), d_r.reshape(-1, 3)))
    incidence = np.vstack(incidence_blocks)
    incidence_norms = np.linalg.norm(incidence, axis=1)
    left_ray = rayfield_two_plane_residuals(
        left_field, left_model, support_left, z_planes=z_planes
    ).reshape(-1, 6)
    right_ray = rayfield_two_plane_residuals(
        right_field, right_model, support_right, z_planes=z_planes
    ).reshape(-1, 6)

    return CMORayfieldBundleAdjustmentResult(
        left_model=left_model,
        right_model=right_model,
        poses=poses,
        success=bool(sol.success),
        message=str(sol.message),
        n_observations=int(sum(obj.shape[0] for obj in object_per_frame)),
        incidence_rms_mm=float(np.sqrt(np.mean(incidence_norms**2))),
        incidence_p95_mm=float(np.percentile(incidence_norms, 95)),
        left_rayfield_rms_mm=float(np.sqrt(np.mean(np.linalg.norm(left_ray, axis=1) ** 2))),
        right_rayfield_rms_mm=float(np.sqrt(np.mean(np.linalg.norm(right_ray, axis=1) ** 2))),
        parameter_vector=np.asarray(sol.x[: 2 * n_params], dtype=np.float64).copy(),
        pose_vectors=np.asarray(pose_params, dtype=np.float64).reshape(-1, 6).copy(),
    )


@dataclass(frozen=True)
class CMOPlaneTargetSpec:
    """Textured planar target used by the CMO image generator."""

    squares_x: int = 11
    squares_y: int = 7
    square_size_mm: float = 1.0
    pixels_per_square: int = 80
    pattern: Literal["checker", "charuco"] = "charuco"
    marker_size_ratio: float = 0.70

    @property
    def width_mm(self) -> float:
        return float(self.squares_x) * float(self.square_size_mm)

    @property
    def height_mm(self) -> float:
        return float(self.squares_y) * float(self.square_size_mm)

    def inner_corners_local_mm(self) -> tuple[Array, Array]:
        xs = -0.5 * self.width_mm + self.square_size_mm * np.arange(1, self.squares_x)
        ys = -0.5 * self.height_mm + self.square_size_mm * np.arange(1, self.squares_y)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1).astype(np.float64)
        ids = np.arange(xy.shape[0], dtype=np.int32)
        return ids, xy

    def make_texture_u8(self) -> Array:
        if self.pattern == "charuco":
            if cv2 is None or not hasattr(cv2, "aruco"):
                raise RuntimeError("CMO ChArUco texture generation requires OpenCV aruco support")
            try:
                aruco = cv2.aruco
                dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
                if hasattr(aruco, "CharucoBoard"):
                    board = aruco.CharucoBoard(
                        (int(self.squares_x), int(self.squares_y)),
                        float(self.square_size_mm),
                        float(self.marker_size_ratio * self.square_size_mm),
                        dictionary,
                    )
                else:  # pragma: no cover - old OpenCV compatibility path
                    board = aruco.CharucoBoard_create(
                        int(self.squares_x),
                        int(self.squares_y),
                        float(self.square_size_mm),
                        float(self.marker_size_ratio * self.square_size_mm),
                        dictionary,
                    )
                w = int(self.squares_x * self.pixels_per_square)
                h = int(self.squares_y * self.pixels_per_square)
                if hasattr(board, "generateImage"):
                    return board.generateImage((w, h)).astype(np.uint8)
                # pragma: no cover - old OpenCV compatibility path
                img = np.zeros((h, w), dtype=np.uint8)
                board.draw((w, h), img)
                return img.astype(np.uint8)
            except Exception as exc:
                raise RuntimeError("Could not generate ChArUco CMO texture") from exc
        return self._make_checker_texture_u8()

    def _make_checker_texture_u8(self) -> Array:
        w = int(self.squares_x * self.pixels_per_square)
        h = int(self.squares_y * self.pixels_per_square)
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        gx = xx // int(self.pixels_per_square)
        gy = yy // int(self.pixels_per_square)
        return np.where(((gx + gy) & 1) == 0, 230, 35).astype(np.uint8)


def rays_from_cmo_pixels(
    channel: CMOChannelSpec,
    common_aberration: PolynomialRayAberration,
) -> tuple[Array, Array]:
    """Return dense per-pixel CMO rays in world coordinates."""
    u, v = channel.intrinsics.pixel_grid()
    return CMOChannelRayField(channel, common_aberration).ray(u, v)


def intersect_rays_with_plane(
    origins: Array,
    directions: Array,
    pose: CMOPlanePose,
) -> tuple[Array, Array]:
    """Intersect a dense rayfield with a plane."""
    n = pose.normal_world
    denom = directions @ n
    numer = (np.asarray(pose.t, dtype=np.float64) - origins) @ n
    tau = numer / np.where(np.abs(denom) < 1e-12, np.nan, denom)
    X = origins + tau[..., None] * directions
    valid = np.isfinite(tau) & (tau > 0.0)
    return X, valid


def sample_cmo_target_texture(
    target: CMOPlaneTargetSpec,
    texture_u8: Array,
    local_xy_mm: Array,
    inside: Array,
    interpolation: Literal["nearest", "linear", "cubic", "lanczos4"] = "linear",
) -> Array:
    """Sample a target texture at target-local metric coordinates."""
    Ht, Wt = texture_u8.shape
    x = local_xy_mm[..., 0]
    y = local_xy_mm[..., 1]
    u_tex = (x + 0.5 * target.width_mm) / target.width_mm * Wt - 0.5
    v_tex = (y + 0.5 * target.height_mm) / target.height_mm * Ht - 0.5
    if cv2 is not None:
        flags = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }[interpolation]
        sampled = cv2.remap(
            texture_u8,
            u_tex.astype(np.float32),
            v_tex.astype(np.float32),
            flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        ui = np.clip(np.rint(u_tex).astype(np.int32), 0, Wt - 1)
        vi = np.clip(np.rint(v_tex).astype(np.int32), 0, Ht - 1)
        sampled = texture_u8[vi, ui]
    out = np.zeros_like(sampled, dtype=np.uint8)
    out[inside] = sampled[inside]
    return out


def apply_sensor_warp(img_u8: Array, warp: SensorWarp, intr: CMOIntrinsics) -> Array:
    """Apply a small image-plane deformation by inverse mapping."""
    if not warp.du_coeff_px and not warp.dv_coeff_px:
        return img_u8
    if cv2 is None:
        return img_u8
    u, v = intr.pixel_grid()
    uv = np.stack([u, v], axis=-1)
    delta = warp.delta_px(uv, intr)
    return cv2.remap(
        img_u8,
        (u - delta[..., 0]).astype(np.float32),
        (v - delta[..., 1]).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _inverse_sensor_warp_pixels(uv_observed: Array, warp: SensorWarp, intr: CMOIntrinsics) -> Array:
    """Approximate ideal pixel coordinates from observed warped coordinates."""
    if not warp.du_coeff_px and not warp.dv_coeff_px:
        return np.asarray(uv_observed, dtype=np.float64)
    observed = np.asarray(uv_observed, dtype=np.float64)
    ideal = observed.copy()
    for _ in range(5):
        ideal = observed - warp.delta_px(ideal, intr)
    return ideal


def apply_blur_noise(
    img_u8: Array,
    blur_sigma_px: float,
    noise_std_gray: float,
    rng: np.random.Generator,
) -> Array:
    out = img_u8
    if blur_sigma_px > 0.0 and cv2 is not None:
        sigma = float(blur_sigma_px)
        k = max(3, int(2 * math.ceil(3 * sigma) + 1))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma)
    if noise_std_gray > 0.0:
        f = out.astype(np.float64)
        f = f + rng.normal(0.0, float(noise_std_gray), out.shape)
        out = np.clip(f, 0.0, 255.0).astype(np.uint8)
    return out


def render_cmo_channel_image(
    cmo: CMOStereoSpec,
    channel: CMOChannelSpec,
    target: CMOPlaneTargetSpec,
    pose: CMOPlanePose,
    texture_u8: Array,
    interpolation: Literal["nearest", "linear", "cubic", "lanczos4"] = "linear",
    background_gray: int = 20,
    blur_sigma_px: float = 0.0,
    noise_std_gray: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Array:
    """Render one CMO channel by pixel -> physical ray -> target plane sampling."""
    rng = rng or np.random.default_rng(0)
    origins, directions = rays_from_cmo_pixels(channel, cmo.common_aberration)
    X_world, valid = intersect_rays_with_plane(origins, directions, pose)
    xy = pose.world_to_local(X_world)[..., :2]
    inside = (
        valid
        & (xy[..., 0] >= -0.5 * target.width_mm)
        & (xy[..., 0] <= +0.5 * target.width_mm)
        & (xy[..., 1] >= -0.5 * target.height_mm)
        & (xy[..., 1] <= +0.5 * target.height_mm)
    )
    sampled = sample_cmo_target_texture(target, texture_u8, xy, inside, interpolation)
    img = np.full_like(sampled, int(background_gray), dtype=np.uint8)
    img[inside] = sampled[inside]
    img = np.clip(
        img.astype(np.float64) * channel.vignetting.gain(channel.intrinsics), 0, 255
    ).astype(np.uint8)
    return apply_blur_noise(img, blur_sigma_px, noise_std_gray, rng)


def project_cmo_points_approx(
    channel: CMOChannelSpec,
    common_aberration: PolynomialRayAberration,
    xyz_world_mm: Array,
) -> Array:
    """Approximate sparse forward projection for CMO target ground truth."""
    R = np.asarray(channel.R_cam_to_world, dtype=np.float64)
    p_world = np.asarray(xyz_world_mm, dtype=np.float64) - channel.origin[None, :]
    p_cam = (R.T @ p_world.T).T
    z = p_cam[:, 2]
    x = p_cam[:, 0] / z
    y = p_cam[:, 1] / z
    aberration = common_aberration.add(channel.differential_aberration)
    dx, dy = aberration.delta(x, y)
    xd, yd = channel.distortion.distort(x - dx, y - dy)
    uv = channel.intrinsics.norm_to_pixel(xd, yd)
    uv = uv + channel.sensor_warp.delta_px(uv, channel.intrinsics)
    uv[z <= 0.0, :] = np.nan
    return uv


def project_cmo_points(
    channel: CMOChannelSpec,
    common_aberration: PolynomialRayAberration,
    xyz_world_mm: Array,
    *,
    max_nfev: int = 40,
) -> Array:
    """Project sparse world points by fitting the shared CMO pixel-to-line model.

    This is the sparse counterpart of the renderer's pixel -> ray -> plane
    model. The first-order projection is only used as the optimizer
    initialization; the returned pixel minimizes point-to-ray distance.
    """
    from scipy.optimize import least_squares  # type: ignore

    points = np.asarray(xyz_world_mm, dtype=np.float64).reshape(-1, 3)
    initial = project_cmo_points_approx(channel, common_aberration, points)
    rayfield = CMOChannelRayField(channel=channel, common_aberration=common_aberration)
    out = np.full((points.shape[0], 2), np.nan, dtype=np.float64)

    lower = np.array([-float(channel.intrinsics.width), -float(channel.intrinsics.height)])
    upper = np.array([
        2.0 * float(channel.intrinsics.width), 2.0 * float(channel.intrinsics.height)
    ])
    center = np.array([channel.intrinsics.cx, channel.intrinsics.cy], dtype=np.float64)

    for idx, point in enumerate(points):
        x0 = initial[idx]
        if not np.all(np.isfinite(x0)):
            x0 = center
        x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)

        def residual(uv: Array, point: Array = point) -> Array:
            origin, direction = rayfield.ray(np.array([uv[0]]), np.array([uv[1]]))
            return np.cross(point - origin.reshape(3), direction.reshape(3))

        result = least_squares(
            residual,
            x0=x0,
            bounds=(lower, upper),
            loss="linear",
            max_nfev=int(max_nfev),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
        )
        out[idx] = result.x
    return out


def project_cmo_target_corners(
    cmo: CMOStereoSpec,
    target: CMOPlaneTargetSpec,
    pose: CMOPlanePose,
) -> dict[str, Array]:
    """Project target inner corners into both CMO channels."""
    ids, xy = target.inner_corners_local_mm()
    xyz = pose.local_to_world(xy)
    uv_left = project_cmo_points(cmo.left, cmo.common_aberration, xyz)
    uv_right = project_cmo_points(cmo.right, cmo.common_aberration, xyz)

    def in_image(uv: Array, intr: CMOIntrinsics) -> Array:
        return (
            np.isfinite(uv[:, 0])
            & np.isfinite(uv[:, 1])
            & (uv[:, 0] >= 0.0)
            & (uv[:, 0] <= intr.width - 1.0)
            & (uv[:, 1] >= 0.0)
            & (uv[:, 1] <= intr.height - 1.0)
        )

    valid = in_image(uv_left, cmo.left.intrinsics) & in_image(uv_right, cmo.right.intrinsics)
    return {
        "corner_id": ids[valid],
        "XYZ_world_mm": xyz[valid].astype(np.float32),
        "uv_left_px": uv_left[valid].astype(np.float32),
        "uv_right_px": uv_right[valid].astype(np.float32),
    }


def _jsonable_dataclass(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _jsonable_dataclass(value) for key, value in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(key): _jsonable_dataclass(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_dataclass(value) for value in obj]
    return obj


def save_gray(path: Path, img_u8: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is not None:
        Image.fromarray(img_u8).save(path)
        return
    if cv2 is not None:
        ok = cv2.imwrite(str(path), img_u8)
        if not ok:
            raise RuntimeError(f"Could not save image: {path}")
        return
    raise RuntimeError("Need Pillow or OpenCV to save generated CMO images")


def generate_cmo_plane_dataset(
    out_dir: Path,
    cmo: CMOStereoSpec,
    target: CMOPlaneTargetSpec,
    poses: list[CMOPlanePose],
    image_format: Literal["png"] = "png",
    blur_sigma_px: float = 0.0,
    noise_std_gray: float = 2.0,
    seed: int = 0,
) -> None:
    """Generate a CMO stereo planar-target dataset from the physics ray model."""
    if image_format != "png":
        raise ValueError("Only png is implemented")
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    left_dir = out_dir / "left"
    right_dir = out_dir / "right"
    left_dir.mkdir(exist_ok=True)
    right_dir.mkdir(exist_ok=True)
    texture = target.make_texture_u8()
    meta = {
        "schema_version": "stereocomplex.cmo_dataset.v0",
        "generator": "stereocomplex.physics.cmo",
        "target": _jsonable_dataclass(target),
        "cmo": _jsonable_dataclass(cmo),
        "sim_params": {
            "image_format": image_format,
            "blur_sigma_px": float(blur_sigma_px),
            "noise_std_gray": float(noise_std_gray),
            "seed": int(seed),
        },
        "model_notes": [
            "The renderer and sparse projection both use stereocomplex.physics.cmo.",
            "CMOChannelRayField is the shared pixel-to-line physics model.",
            "Sparse GT projection minimizes point-to-ray distance "
            "with the same CMOChannelRayField.",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    all_frame: list[Array] = []
    all_id: list[Array] = []
    all_xyz: list[Array] = []
    all_uv_l: list[Array] = []
    all_uv_r: list[Array] = []
    with (out_dir / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for frame_id, pose in enumerate(poses):
            left_name = f"{frame_id:06d}.png"
            right_name = f"{frame_id:06d}.png"
            img_l = render_cmo_channel_image(
                cmo,
                cmo.left,
                target,
                pose,
                texture,
                blur_sigma_px=blur_sigma_px,
                noise_std_gray=noise_std_gray,
                rng=rng,
            )
            img_r = render_cmo_channel_image(
                cmo,
                cmo.right,
                target,
                pose,
                texture,
                blur_sigma_px=blur_sigma_px,
                noise_std_gray=noise_std_gray,
                rng=rng,
            )
            save_gray(left_dir / left_name, img_l)
            save_gray(right_dir / right_name, img_r)
            frame_record = {
                "frame_id": frame_id,
                "left": left_name,
                "right": right_name,
            }
            stream.write(json.dumps(frame_record) + "\n")
            gt = project_cmo_target_corners(cmo, target, pose)
            n = gt["corner_id"].shape[0]
            all_frame.append(np.full((n,), frame_id, dtype=np.int32))
            all_id.append(gt["corner_id"].astype(np.int32))
            all_xyz.append(gt["XYZ_world_mm"].astype(np.float32))
            all_uv_l.append(gt["uv_left_px"].astype(np.float32))
            all_uv_r.append(gt["uv_right_px"].astype(np.float32))

    np.savez_compressed(
        out_dir / "gt_charuco_corners.npz",
        frame_id=np.concatenate(all_frame) if all_frame else np.zeros((0,), np.int32),
        corner_id=np.concatenate(all_id) if all_id else np.zeros((0,), np.int32),
        XYZ_world_mm=np.concatenate(all_xyz) if all_xyz else np.zeros((0, 3), np.float32),
        uv_left_px=np.concatenate(all_uv_l) if all_uv_l else np.zeros((0, 2), np.float32),
        uv_right_px=np.concatenate(all_uv_r) if all_uv_r else np.zeros((0, 2), np.float32),
    )


def make_reference_cmo_scenario() -> tuple[CMOStereoSpec, CMOPlaneTargetSpec, list[CMOPlanePose]]:
    """Reference CMO scenario with common aberration and channel asymmetries."""
    common = PolynomialRayAberration(
        coeff_x={"x2": +2.0e-4, "y2": -1.5e-4},
        coeff_y={"xy": +1.0e-4},
    )
    cmo = CMOStereoSpec.symmetric_default(
        width=1280,
        height=960,
        focal_mm=35.0,
        pitch_um=3.45,
        baseline_mm=6.0,
        common_aberration=common,
        left_distortion=BrownConrady(k1=-0.06, k2=0.01, p1=2.0e-4, p2=-1.0e-4),
        right_distortion=BrownConrady(k1=-0.055, k2=0.008, p1=-2.0e-4, p2=1.0e-4),
    )
    cmo = CMOStereoSpec(
        left=replace(
            cmo.left,
            differential_aberration=PolynomialRayAberration(
                coeff_x={"x": +1.0e-4},
                coeff_y={"y": -8.0e-5},
            ),
            sensor_warp=SensorWarp(du_coeff_px={"xy": 0.25}, dv_coeff_px={"x2": -0.18}),
        ),
        right=replace(
            cmo.right,
            differential_aberration=PolynomialRayAberration(
                coeff_x={"x": -1.0e-4},
                coeff_y={"y": +8.0e-5},
            ),
            sensor_warp=SensorWarp(du_coeff_px={"xy": -0.20}, dv_coeff_px={"y2": 0.15}),
        ),
        common_aberration=common,
    )
    target = CMOPlaneTargetSpec(
        squares_x=11,
        squares_y=7,
        square_size_mm=2.0,
        pixels_per_square=80,
        pattern="charuco",
    )
    poses = [
        pose_from_euler_xyz(+0.00, +0.00, +0.00, (0.0, 0.0, 180.0)),
        pose_from_euler_xyz(+0.05, -0.04, +0.03, (-3.0, +2.0, 185.0)),
        pose_from_euler_xyz(-0.04, +0.06, -0.02, (+2.0, -2.0, 175.0)),
        pose_from_euler_xyz(+0.08, +0.03, +0.04, (+4.0, +1.5, 195.0)),
    ]
    return cmo, target, poses


__all__ = [
    "BrownConrady",
    "CMOChannelRayField",
    "CMOChannelSpec",
    "CMOIntrinsics",
    "CMOPlanePose",
    "CMOPlaneTargetSpec",
    "CMORayfieldBundleAdjustmentResult",
    "CMOStereoSpec",
    "NonCentralPolynomialChannelModel",
    "PolynomialRayAberration",
    "SensorWarp",
    "Vignetting",
    "apply_blur_noise",
    "apply_sensor_warp",
    "fit_cmo_stereo_model_and_poses_from_zernike_rayfields",
    "generate_cmo_plane_dataset",
    "intersect_rays_with_plane",
    "make_reference_cmo_scenario",
    "normalize_vectors",
    "polynomial_channel_parameters_from_spec",
    "pose_from_euler_xyz",
    "project_cmo_points",
    "project_cmo_points_approx",
    "project_cmo_target_corners",
    "rays_from_cmo_pixels",
    "render_cmo_channel_image",
    "rotx",
    "roty",
    "rotz",
    "sample_cmo_target_texture",
    "save_gray",
]
