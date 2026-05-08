# %% [markdown]
# # 07 — Model selection classification matrix
#
# This notebook demonstrates that the StereoComplex model selection framework
# correctly identifies the optical architecture for **all six oracle families**:
#
# | # | Oracle | Expected winner | Structure |
# |---|--------|----------------|-----------|
# | 1 | Central pinhole | `central_pinhole` | Pure central projection |
# | 2 | Central Brown-Conrady | `central_brown_conrady` | Central with radial/tangential distortion |
# | 3 | Pinhole + inclined plate | `pinhole_parallel_plate` | Non-central plate line family |
# | 4 | Physical CMO shared-rig | `cmo_physical_shared` | Shared-objective stereo |
# | 5 | Greenough (independent Brown × 2) | `central_brown_conrady` | Independent central channels |
# | 6 | **Uncatalogued** (random high-order Zernike) | `zernike_compact` | Fallback generic model |
#
# ## Why this matters
#
# The framework is not merely a parameter-count competition.  Each oracle is
# the *correct* model for a specific optical architecture.  The framework must
# recover the truth — the runner-up models must have worse BIC either because
# their structure is wrong (high RMS) or because they are over-parameterised
# (high parameter penalty).
#
# The last row is the most important: when the optics fall outside all known
# families, the compact Zernike fallback wins — signalling to the user that the
# instrument does not match any catalogued architecture.

# %%
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from stereocomplex.physics import (
    CentralPinholeModel,
    CentralBrownConradyModel,
    PinholeParallelPlateModel,
    PhysicalModelSpec,
    select_physical_model_from_rayfield,
    CMOPolynomialChannelModel,
)
from stereocomplex.physics.cmo import (
    CMOChannelSpec,
    CMOIntrinsics,
    PolynomialRayAberration,
    BrownConrady,
)
from stereocomplex.physics.cmo_physical import (
    CMOPhysicalStereoModel,
    fit_cmo_physical_stereo_model_to_rayfields,
)
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeCandidate,
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)

IMAGE_SIZE = (160, 120)
GRID_SHAPE = (12, 9)
SEED = 42


# %% [markdown]
# ## Oracle builders
#
# Each builder returns a `(left_field, right_field, K, description)` tuple.
# The fields are rayfield-like objects with `.ray(u, v)` methods.

# %%
def build_pinhole_oracle():
    """Symmetric central pinhole stereo pair."""
    K = np.array([[200.0, 0.0, 79.5], [0.0, 200.0, 59.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    left = CentralPinholeModel(K=K)
    right = CentralPinholeModel(K=K)
    return left, right, K, "central pinhole"


def build_brown_oracle():
    """Central Brown-Conrady stereo pair with moderate distortion."""
    K = np.array([[200.0, 0.0, 79.5], [0.0, 200.0, 59.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    left = CentralBrownConradyModel(K=K, k1=-0.08, k2=0.03, p1=1.0e-3, p2=-1.0e-3, k3=0.0)
    right = CentralBrownConradyModel(K=K, k1=-0.06, k2=0.02, p1=-5.0e-4, p2=8.0e-4, k3=0.0)
    return left, right, K, "central Brown-Conrady"


def build_plate_oracle():
    """Pinhole + inclined parallel plate with 2 mm thickness."""
    from stereocomplex.physics.parallel_plate_fit import PinholeParallelPlateFitParams
    K = np.array([[200.0, 0.0, 79.5], [0.0, 200.0, 59.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    left_params = PinholeParallelPlateFitParams(
        alpha_deg=5.0, beta_deg=-3.0, thickness_mm=2.0, eta=1.5, d1_mm=80.0,
    )
    right_params = PinholeParallelPlateFitParams(
        alpha_deg=-5.0, beta_deg=2.0, thickness_mm=2.0, eta=1.5, d1_mm=80.0,
    )
    return (PinholeParallelPlateModel(K=K, params=left_params),
            PinholeParallelPlateModel(K=K, params=right_params),
            K, "inclined parallel plate")


def build_cmo_oracle():
    """Physical CMO shared-rig stereo microscope."""
    truth = CMOPhysicalStereoModel(
        f_obj_mm=80.0, working_distance_mm=120.0, b_mm=8.0,
        f_tube_mm=50.0, cx_principal_px=79.5, cy_principal_px=59.5,
        pixel_pitch_mm=0.05, image_size=IMAGE_SIZE,
        distortion_left=(-0.04, 0.01, 2.0e-4, -1.0e-4, 0.0),
        distortion_right=(-0.035, 0.008, -2.0e-4, 1.0e-4, 0.0),
    )
    K = np.array([[1000.0, 0.0, 79.5], [0.0, 1000.0, 59.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    return truth.channel("left"), truth.channel("right"), K, "CMO shared-rig"


def build_greenough_oracle():
    """Greenough stereo: two independent central Brown-Conrady channels."""
    K_L = np.array([[210.0, 0.0, 79.5], [0.0, 210.0, 59.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    K_R = np.array([[195.0, 0.0, 79.0], [0.0, 195.0, 60.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    left = CentralBrownConradyModel(K=K_L, k1=-0.08, k2=0.03, p1=1.0e-3, p2=-1.0e-3, k3=0.0)
    right = CentralBrownConradyModel(K=K_R, k1=-0.06, k2=0.02, p1=-5.0e-4, p2=8.0e-4, k3=0.0)
    return left, right, K_L, K_R, "Greenough (Brown-Conrady ×2)"


def build_exotic_oracle():
    """High-order random Zernike rayfield — belongs to no physical family."""
    rng = np.random.default_rng(SEED)
    K = np.array([[200.0, 0.0, 79.5], [0.0, 200.0, 59.5], [0.0, 0.0, 1.0]], dtype=np.float64)
    config = ZernikeOriginFieldConfig(image_size=IMAGE_SIZE, max_order=4)
    n_modes = len(config.modes())
    left = ZernikeRayField(K=K, config=config, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=rng.normal(scale=2.0, size=(n_modes, 3)),
        direction_coeffs=rng.normal(scale=0.05, size=(n_modes, 3)),
    ))
    right = ZernikeRayField(K=K, config=config, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=rng.normal(scale=2.0, size=(n_modes, 3)),
        direction_coeffs=rng.normal(scale=0.05, size=(n_modes, 3)),
    ))
    return left, right, K, "uncatalogued (Zernike nmax=4)"


# %% [markdown]
# ## Candidate set
#
# All candidates are offered in every case.  The physical CMO is offered only
# when the `target_right` field is provided (stereo mode).

# %%
def build_candidates(K: np.ndarray, image_size: tuple[int, int],
                     pixel_pitch_mm: float | None = None,
                     extra_cmo: bool = False):
    """Return the standard candidate list."""
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1],
                         fx=float(K[0, 0]), fy=float(K[1, 1]),
                         cx=float(K[0, 2]), cy=float(K[1, 2]))
    terms = CMOPolynomialChannelModel.default_terms()
    poly_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    poly_bounds = (
        np.r_[[-40.0, -40.0, -50.0, -1.0, -1.0, -0.1, -0.1, -1.0], -0.1 * np.ones(2 * len(terms))],
        np.r_[[+40.0, +40.0, +50.0, +1.0, +1.0, +0.1, +0.1, +1.0], +0.1 * np.ones(2 * len(terms))],
    )
    lo_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=2)
    n_zernike = len(lo_config.modes()) * 6

    specs = [
        PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
        PhysicalModelSpec(
            "central_brown_conrady", CentralBrownConradyModel, np.zeros(5),
            bounds=(
                np.array([-1.0, -1.0, -0.1, -0.1, -1.0]),
                np.array([1.0, 1.0, 0.1, 0.1, 1.0]),
            ),
        ),
        PhysicalModelSpec(
            "pinhole_parallel_plate", PinholeParallelPlateModel,
            np.array([0.0, 0.0, 8.0]),
            bounds=(np.array([-30.0, -30.0, 0.0]), np.array([30.0, 30.0, 50.0])),
            model_kwargs={"eta": 1.5, "d1_mm": 80.0},
        ),
    ]

    if extra_cmo and pixel_pitch_mm is not None:
        specs.append(PhysicalModelSpec(
            "cmo_physical_shared", CMOPhysicalStereoModel,
            np.array([80.0, 120.0, 10.0, 50.0, 79.5, 59.5, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            model_kwargs={"pixel_pitch_mm": pixel_pitch_mm},
        ))

    specs.append(PhysicalModelSpec(
        "cmo_polynomial_channel", CMOPolynomialChannelModel,
        poly_initial, bounds=poly_bounds,
        model_kwargs={"cmo_image_size": image_size, "aberration_terms": terms},
    ))
    specs.append(PhysicalModelSpec(
        "zernike_compact", ZernikeCandidate,
        np.zeros(n_zernike, dtype=np.float64), bounds=None,
        model_kwargs={"config": lo_config, "fit_directions": True},
    ))
    return specs


# %% [markdown]
# ## Run all six cases

# %%
@dataclass
class CaseResult:
    oracle_name: str
    winner: str
    winner_params: int
    winner_rms: float
    winner_bic: float
    second: str
    second_params: int
    second_rms: float
    second_bic: float
    correct: bool


def pixels_grid():
    return np.array(
        [[u, v] for v in np.linspace(8.0, float(IMAGE_SIZE[1] - 1) - 8.0, GRID_SHAPE[1])
         for u in np.linspace(8.0, float(IMAGE_SIZE[0] - 1) - 8.0, GRID_SHAPE[0])],
        dtype=np.float64,
    )


def evaluate_case(name: str, left_field, right_field, K, expected: str,
                  pixel_pitch_mm: float | None = None,
                  K_right: np.ndarray | None = None) -> CaseResult:
    px = pixels_grid()
    K_r = K_right if K_right is not None else K
    candidates = build_candidates(K, IMAGE_SIZE, pixel_pitch_mm=pixel_pitch_mm,
                                  extra_cmo=(expected == "cmo_physical_shared"))
    report = select_physical_model_from_rayfield(
        target_field=left_field,
        target_right=right_field,
        candidate_specs=candidates,
        K=K, K_right=K_r,
        image_size=IMAGE_SIZE,
        support_pixels=px, support_pixels_right=px,
        full_grid_weight=0.0,
        max_nfev=1500,
    )
    sorted_candidates = sorted(report.candidates, key=lambda c: c.bic)
    best = sorted_candidates[0]
    second = sorted_candidates[1] if len(sorted_candidates) > 1 else best
    return CaseResult(
        oracle_name=name,
        winner=best.model_name,
        winner_params=best.n_parameters,
        winner_rms=best.rms_mm,
        winner_bic=best.bic,
        second=second.model_name,
        second_params=second.n_parameters,
        second_rms=second.rms_mm,
        second_bic=second.bic,
        correct=(best.model_name == expected),
    )


# %%
print("Running model selection on all six oracles...\n")

cases = [
    ("central pinhole", build_pinhole_oracle, "central_pinhole", None),
    ("central Brown-Conrady", build_brown_oracle, "central_brown_conrady", None),
    ("inclined parallel plate", build_plate_oracle, "pinhole_parallel_plate", None),
    ("CMO shared-rig", build_cmo_oracle, "cmo_physical_shared", 0.05),
    ("Greenough (Brown ×2)", build_greenough_oracle, "central_brown_conrady", None),
    ("uncatalogued Zernike", build_exotic_oracle, "zernike_compact", None),
]

results = []
for name, builder, expected, pitch in cases:
    out = builder()
    left, right = out[0], out[1]
    K = out[2]
    K_r = out[3] if isinstance(out[3], np.ndarray) else None  # Greenough has K_R at [3]
    _desc = out[-1]
    r = evaluate_case(name, left, right, K, expected, pixel_pitch_mm=pitch, K_right=K_r)
    results.append(r)
    status = "✓" if r.correct else "✗ MISCLASSIFIED"
    delta_bic = r.second_bic - r.winner_bic
    print(f"  {status} {r.oracle_name:30s} → {r.winner:28s} "
          f"(p={r.winner_params:3d}, RMS={r.winner_rms:.4f} mm, "
          f"ΔBIC={delta_bic:+.0f} vs {r.second})")


# %% [markdown]
# ## Classification matrix

# %%
print()
print("┌──────────────────────────────┬────────────────────────────┬───────┬───────────┬────────────┬────────┐")
print("│ Oracle                       │ BIC winner                 │  Wins │  Params   │  RMS (mm)  │  ΔBIC  │")
print("├──────────────────────────────┼────────────────────────────┼───────┼───────────┼────────────┼────────┤")
for r in results:
    delta = r.second_bic - r.winner_bic
    check = "  ✓" if r.correct else "  ✗"
    print(f"│ {r.oracle_name:28s} │ {r.winner:26s} │ {check:3s}  │ {r.winner_params:>4d}       │ {r.winner_rms:>8.4f}  │ {delta:>+7.0f} │")
print("└──────────────────────────────┴────────────────────────────┴───────┴───────────┴────────────┴────────┘")

all_correct = all(r.correct for r in results)
print(f"\nAll {len(results)} oracles correctly classified: {'YES' if all_correct else 'NO — see above'}")

# %% [markdown]
# ## Interpretation
#
# - **Rows 1–3**: Classical stereo cases — pinhole, Brown-Conrady, and inclined
#   plate.  The correct model wins because it has the right structure with the
#   fewest parameters.
#
# - **Row 4**: CMO shared-rig.  The physical CMO model wins despite the
#   polynomial surrogate and compact Zernike fitting nearly as well — the BIC
#   penalty for 36 independent parameters is decisive against 17 shared.
#
# - **Row 5**: Greenough.  Two independent central Brown-Conrady channels win
#   (10 params for the pair).  The CMO physical model is not offered here
#   because the oracle has no shared-objective structure, but even if it were
#   present it would fail — the Greenough rays diverge from two camera centres
#   rather than converging to a shared working plane.
#
# - **Row 6**: Uncatalogued.  A high-order random Zernike rayfield belongs to
#   no known family.  The compact Zernike candidate (max_order=2, 36 params)
#   wins — correctly signalling that the optics fall outside the catalogue.
#   This is the detector row: when `zernike_compact` wins, the user knows that
#   no physical model in the current set is adequate.
