"""Shared optical oracle builders for model-selection benchmarks.

Each oracle is a synthetic stereo rayfield pair that represents one
optical architecture (pinhole, Brown-Conrady, inclined plate, CMO
shared-rig, Greenough, or an uncatalogued exotic Zernike field).

All builders return a :class:`StereoOracle` dataclass with left/right
rayfields, K matrices, and metadata needed by both the classification
matrix (notebook 07) and the direct-vs-rayfield study (notebook 08).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stereocomplex.physics import (
    CentralBrownConradyModel,
    CentralPinholeModel,
)
from stereocomplex.physics.cmo_physical import CMOPhysicalStereoModel
from stereocomplex.physics.parallel_plate_fit import (
    PinholeParallelPlateFitParams,
    PinholeParallelPlateModel,
)
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)

Array = np.ndarray


@dataclass(frozen=True)
class StereoOracle:
    """A synthetic stereo rayfield pair representing one optical architecture.

    Attributes
    ----------
    name : str
        Human-readable label (``"central pinhole"``, …).
    expected_winner : str
        Model name that should win BIC on this oracle.
    left_field :
        Object with a ``.ray(u, v)`` method returning ``(O, d)`` arrays.
    right_field :
        Same for the right channel.
    K_left, K_right : np.ndarray
        3×3 intrinsic matrices for the left and right channels.
    image_size : tuple[int, int]
        ``(width, height)`` in pixels.
    pixel_pitch_mm : float | None
        Sensor pixel pitch.  Only set for the CMO oracle where the
        physical model requires it as a fixed parameter.
    ground_truth_parameters : dict
        Known oracle parameters (empty for exotic / non-parametric oracles).
    """

    name: str
    expected_winner: str
    left_field: Any
    right_field: Any
    K_left: np.ndarray
    K_right: np.ndarray
    image_size: tuple[int, int]
    pixel_pitch_mm: float | None = None
    ground_truth_parameters: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.ground_truth_parameters is None:
            object.__setattr__(self, "ground_truth_parameters", {})


@dataclass(frozen=True)
class MultiCameraOracle:
    """Synthetic N-camera rayfield rig used by N-camera validation tests."""

    name: str
    expected_winner: str
    fields_by_channel: dict[str, Any]
    intrinsics_by_channel: dict[str, np.ndarray]
    image_size: tuple[int, int]
    ground_truth_parameters: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if not self.fields_by_channel:
            raise ValueError("at least one channel is required")
        if set(self.fields_by_channel) != set(self.intrinsics_by_channel):
            raise ValueError("fields and intrinsics must use the same channel names")
        if self.ground_truth_parameters is None:
            object.__setattr__(self, "ground_truth_parameters", {})

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.fields_by_channel)

    @property
    def n_channels(self) -> int:
        return len(self.fields_by_channel)

    def field(self, channel: str):
        return self.fields_by_channel[channel]

    def K(self, channel: str) -> np.ndarray:
        return self.intrinsics_by_channel[channel]


# ── default image size used by most oracles ──────────────────────────
_IMAGE_SIZE = (160, 120)
_SEED = 42


def build_pinhole_oracle(image_size=_IMAGE_SIZE) -> StereoOracle:
    """Symmetric central pinhole stereo pair."""
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    K = np.array([[200.0, 0.0, cx], [0.0, 200.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return StereoOracle(
        name="central pinhole",
        expected_winner="central_pinhole",
        left_field=CentralPinholeModel(K=K),
        right_field=CentralPinholeModel(K=K),
        K_left=K,
        K_right=K,
        image_size=image_size,
    )


def build_pinhole_n_camera_oracle(
    image_size=_IMAGE_SIZE,
    channel_names: tuple[str, ...] = ("cam0", "cam1", "cam2", "cam3"),
) -> MultiCameraOracle:
    """Central pinhole N-camera rig with known per-channel intrinsics."""
    if not channel_names:
        raise ValueError("at least one channel is required")
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    fields: dict[str, Any] = {}
    intrinsics: dict[str, np.ndarray] = {}
    for idx, name in enumerate(channel_names):
        focal = 190.0 + 10.0 * float(idx)
        K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        fields[name] = CentralPinholeModel(K=K)
        intrinsics[name] = K
    return MultiCameraOracle(
        name=f"central pinhole x{len(channel_names)}",
        expected_winner="central_pinhole",
        fields_by_channel=fields,
        intrinsics_by_channel=intrinsics,
        image_size=image_size,
        ground_truth_parameters={"channel_names": channel_names},
    )


def build_brown_oracle(image_size=_IMAGE_SIZE) -> StereoOracle:
    """Central Brown-Conrady stereo pair with moderate distortion."""
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    K = np.array([[200.0, 0.0, cx], [0.0, 200.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return StereoOracle(
        name="central Brown-Conrady",
        expected_winner="central_brown_conrady",
        left_field=CentralBrownConradyModel(K=K, k1=-0.08, k2=0.03, p1=1e-3, p2=-1e-3, k3=0.0),
        right_field=CentralBrownConradyModel(K=K, k1=-0.06, k2=0.02, p1=-5e-4, p2=8e-4, k3=0.0),
        K_left=K,
        K_right=K,
        image_size=image_size,
        ground_truth_parameters={"k1_L": -0.08, "k2_L": 0.03, "k1_R": -0.06, "k2_R": 0.02},
    )


def build_plate_oracle(image_size=_IMAGE_SIZE) -> StereoOracle:
    """Pinhole + inclined parallel plate with 2 mm thickness."""
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    K = np.array([[200.0, 0.0, cx], [0.0, 200.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    left_params = PinholeParallelPlateFitParams(
        alpha_deg=5.0,
        beta_deg=-3.0,
        thickness_mm=2.0,
        eta=1.5,
        d1_mm=80.0,
    )
    right_params = PinholeParallelPlateFitParams(
        alpha_deg=-5.0,
        beta_deg=2.0,
        thickness_mm=2.0,
        eta=1.5,
        d1_mm=80.0,
    )
    return StereoOracle(
        name="inclined parallel plate",
        expected_winner="pinhole_parallel_plate",
        left_field=PinholeParallelPlateModel(K=K, params=left_params),
        right_field=PinholeParallelPlateModel(K=K, params=right_params),
        K_left=K,
        K_right=K,
        image_size=image_size,
        ground_truth_parameters={
            "alpha_deg_L": 5.0,
            "beta_deg_L": -3.0,
            "thickness_mm": 2.0,
        },
    )


def build_cmo_oracle(image_size=_IMAGE_SIZE) -> StereoOracle:
    """Physical CMO shared-rig stereo microscope."""
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    truth = CMOPhysicalStereoModel(
        f_obj_mm=80.0,
        working_distance_mm=120.0,
        b_mm=8.0,
        f_tube_mm=50.0,
        cx_principal_px=cx,
        cy_principal_px=cy,
        pixel_pitch_mm=0.05,
        image_size=image_size,
        distortion_left=(-0.04, 0.01, 2.0e-4, -1.0e-4, 0.0),
        distortion_right=(-0.035, 0.008, -2.0e-4, 1.0e-4, 0.0),
    )
    fx = truth.f_tube_mm / truth.pixel_pitch_mm
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return StereoOracle(
        name="CMO shared-rig",
        expected_winner="cmo_physical_shared",
        left_field=truth.channel("left"),
        right_field=truth.channel("right"),
        K_left=K,
        K_right=K,
        image_size=image_size,
        pixel_pitch_mm=truth.pixel_pitch_mm,
        ground_truth_parameters={
            "f_obj_mm": 80.0,
            "working_distance_mm": 120.0,
            "b_mm": 8.0,
            "f_tube_mm": 50.0,
            "pixel_pitch_mm": 0.05,
        },
    )


def build_greenough_oracle(image_size=_IMAGE_SIZE) -> StereoOracle:
    """Greenough stereo: two independent central Brown-Conrady channels."""
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    K_L = np.array([[210.0, 0.0, cx], [0.0, 210.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_R = np.array([[195.0, 0.0, cx], [0.0, 195.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return StereoOracle(
        name="Greenough (Brown-Conrady ×2)",
        expected_winner="central_brown_conrady",
        left_field=CentralBrownConradyModel(K=K_L, k1=-0.08, k2=0.03, p1=1e-3, p2=-1e-3, k3=0.0),
        right_field=CentralBrownConradyModel(K=K_R, k1=-0.06, k2=0.02, p1=-5e-4, p2=8e-4, k3=0.0),
        K_left=K_L,
        K_right=K_R,
        image_size=image_size,
    )


def build_exotic_zernike_oracle(image_size=_IMAGE_SIZE, seed=_SEED) -> StereoOracle:
    """Low-amplitude high-order Zernike — smooth but outside physical families."""
    w, h = image_size
    cx, cy = (w - 1) / 2, (h - 1) / 2
    rng = np.random.default_rng(seed)
    K = np.array([[200.0, 0.0, cx], [0.0, 200.0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    config = ZernikeOriginFieldConfig(image_size=image_size, max_order=3)
    n_modes = len(config.modes())
    return StereoOracle(
        name="uncatalogued Zernike",
        expected_winner="zernike_compact",
        left_field=ZernikeRayField(
            K=K,
            config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=rng.normal(scale=0.08, size=(n_modes, 3)),
                direction_coeffs=rng.normal(scale=0.003, size=(n_modes, 3)),
            ),
        ),
        right_field=ZernikeRayField(
            K=K,
            config=config,
            coefficients=ZernikeRayFieldCoefficients(
                origin_coeffs=rng.normal(scale=0.08, size=(n_modes, 3)),
                direction_coeffs=rng.normal(scale=0.003, size=(n_modes, 3)),
            ),
        ),
        K_left=K,
        K_right=K,
        image_size=image_size,
    )


def build_all_oracles(image_size=_IMAGE_SIZE, seed=_SEED) -> list[StereoOracle]:
    """Return the six standard oracles in a fixed order."""
    return [
        build_pinhole_oracle(image_size),
        build_brown_oracle(image_size),
        build_plate_oracle(image_size),
        build_cmo_oracle(image_size),
        build_greenough_oracle(image_size),
        build_exotic_zernike_oracle(image_size, seed),
    ]
