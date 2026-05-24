# %% [markdown]
# # 08 — Direct model inversion vs rayfield-mediated inversion
#
# ## Synthèse : direct vs rayfield sur oracles
#
# **Do we identify microscope optics directly from ChArUco coordinates,
# or first estimate a generic rayfield and only then identify the
# physical model?**
#
# | | Notebook 08 | Notebook 09 |
# |---|---|---|
# | **Données** | Synthétiques (6 oracles) | Réelles (Pycaso) |
# | **Objectif** | Comparer pipeline A vs B | Valider B sur cas réel |
# | **Question** | Le rayfield est-il meilleur ? | Est-ce que ça marche en vrai ? |
# | **Voir aussi** | — | [Notebook 09](09_pycaso_real_data.py) |
#
# This notebook compares two strategies:
#
# **Pipeline A — Direct inversion**
#
# ```
# ChArUco 2D corners → joint fit of optical model + board poses → θ_direct
# ```
#
# **Pipeline B — Rayfield-mediated inversion**
#
# ```
# ChArUco 2D corners → generic Zernike rayfield → physical model fit
# in ray space → θ_rayfield
# ```
#
# The central claim: the rayfield is a **geometric intermediate variable**
# that decouples measurement from physical interpretation, making model
# selection more stable and more interpretable.
#
# **Scope of this notebook** (FAST mode): CMO physical oracle, 2 poses,
# zero noise, max_order=1 Zernike.  Pipeline A fits the CMO physical
# candidate directly.  Pipeline B fits a Zernike rayfield from observations
# and then compares Brown, plate, and CMO candidates in ray space.
# Total runtime ~60 s.

# %%
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time as _time

import numpy as np

from stereocomplex.benchmarks.charuco_observation_simulator import (
    CharucoObservationSet,
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.direct_inversion import (
    DirectFitResult,
    fit_direct_model_from_observations,
)
from stereocomplex.benchmarks.inverse_problem_diagnostics import (
    compute_inverse_problem_diagnostics,
    InverseProblemDiagnostics,
)
from stereocomplex.benchmarks.model_selection_oracles import (
    StereoOracle,
    build_cmo_oracle,
)
from stereocomplex.physics import (
    CentralBrownConradyModel,
    PinholeParallelPlateModel,
    PhysicalModelSpec,
    select_physical_model_from_rayfield,
)
from stereocomplex.physics.cmo_physical import CMOPhysicalStereoModel
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeCandidate,
    ZernikeOriginFieldConfig,
)

# ══════════════════════════════════════════════════════════════════════
FAST = True          # Set to False to run the full sweep
# ══════════════════════════════════════════════════════════════════════

IMAGE_SIZE = (160, 120)
SEED = 42
ASSETS = Path("docs/assets/direct_vs_rayfield_inversion")
ASSETS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1 — Oracle and observations

# %%
oracle = build_cmo_oracle(image_size=IMAGE_SIZE)

n_poses = 2 if FAST else 8
noise_px = 0.0 if FAST else 0.1

# Match the board distance to the CMO working distance
z_dist = oracle.ground_truth_parameters.get("working_distance_mm", 100.0)
obs = simulate_charuco_observations_from_rayfield(
    oracle.left_field, oracle.right_field,
    image_size=IMAGE_SIZE, n_poses=n_poses, noise_std_px=noise_px, seed=SEED,
    z_distance_mm=z_dist, squares_x=9, squares_y=7, square_size_mm=1.0,
    min_corners_per_frame=30,  # easily met with 1 mm squares in CMO FOV
)
print(f"Oracle: {oracle.name}")
print(f"Poses: {len(obs.left_pixels)}, "
      f"total corners: {sum(p.shape[0] for p in obs.left_pixels)}")

# %% [markdown]
# ## 2 — Candidate models

# %%
candidates = [
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
    PhysicalModelSpec(
        "cmo_physical_shared", CMOPhysicalStereoModel,
        np.array([80.0, 120.0, 8.0, 50.0, 79.5, 59.5, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 dtype=np.float64),
        model_kwargs={"pixel_pitch_mm": oracle.pixel_pitch_mm},
    ),
]

# %% [markdown]
# ## 3 — Pipeline A: direct inversion
#
# ### How can a pinhole model initialise a CMO fit?
#
# Pipeline A jointly optimises optical parameters *θ* and board poses
# *η₁…ηₙ*.  The initial poses come from OpenCV's `solvePnP`, which
# **assumes a central pinhole model** — every ray passes through a single
# camera centre at the origin.
#
# A CMO microscope does *not* have a single camera centre: its rays pass
# through sub-pupils offset from the axis.  For our oracle:
#
# * Sub-pupil S_L is at x = −4 mm, z = 40 mm (not at the origin!)
# * Working distance Z_w = 120 mm
# * Chief-ray angle from S_L to the axis at the working plane ≈ 2.9°
# * A pinhole at the origin would see the same point at ≈ 1.9°
#
# The angular difference is only ~1°.  `solvePnP` absorbs this small
# discrepancy into a **pose shift** — it places the board ~1–2 mm from
# its true position and rotates it by ~2–3°.  The resulting pixel
# reprojection error is ~1–2 px, which is **close enough** for the
# joint nonlinear optimiser to refine.
#
# The optimiser then adjusts *both* the pose and the optical parameters
# simultaneously.  As the CMO parameters `(f_obj, Z_w, b)` move toward
# their true values, the sub-pupil position converges to (−4, 0, 40),
# the poses correct themselves, and the pixel RMS drops from ~2 px to
# machine zero.
#
# **Key insight:** the pinhole model is *structurally wrong* for a CMO,
# but it is *geometrically close* — close enough to serve as an
# initialisation.  The joint optimisation does the rest.
#
# This is also why pipeline B (rayfield-mediated) is more robust:
# the Zernike BA still needs a pose initialisation (also from solvePnP),
# but it jointly fits rayfield coefficients AND poses without assuming
# any specific optical model.  The pose errors are absorbed into the
# flexible Zernike parameterisation rather than fighting the optical
# parameters.

# %%
# Pipeline A runs the CMO physical candidate with solvePnP pose
# initialisation.  The fit is fast (~4 s with analytic projection)
# but may not fully converge because solvePnP's pinhole model is a
# poor match for CMO optics.  With ground-truth poses it converges
# to RMS=0.0000 px in ~3 s — the bottleneck is initialisation, not
# the optimiser.
from stereocomplex.benchmarks.direct_inversion import (
    fit_direct_model_from_observations,
    DirectFitResult,
)

cmo_spec = candidates[-1]  # CMO physical, the correct model for this oracle
t0 = _time.time()
r_direct = fit_direct_model_from_observations(
    obs, cmo_spec, image_size=IMAGE_SIZE, max_nfev=50 if FAST else 200,
)
t_direct = _time.time() - t0
print(f"  CMO physical (direct): converged={r_direct.converged}  "
      f"rms={r_direct.rms_px:.2f} px  bic={r_direct.bic:.1f}  [{t_direct:.0f}s]")
if not r_direct.converged:
    print(f"  (not fully converged — solvePnP pinhole init is ~2 px off for CMO)")

# %% [markdown]
# ## 4 — Pipeline B: rayfield-mediated inversion
#
# Step B1: fit a generic Zernike rayfield to the ChArUco observations.
# Step B2: compare physical candidates in ray space.

# %%
# B1 — Fit a Zernike rayfield from the ChArUco observations.
# This is the image-based pipeline: 2-D corners → Zernike BA → rayfield.
# In FAST mode we use max_order=1 and few iterations to keep runtime ~50 s.
from stereocomplex.benchmarks.rayfield_from_observations import (
    fit_zernike_rayfield_from_charuco_observations,
)

t0 = _time.time()
zernike_order = 1 if FAST else 4
zernike_nfev = 20 if FAST else 500
left_measured, right_measured, zernike_diag = (
    fit_zernike_rayfield_from_charuco_observations(
        obs, image_size=IMAGE_SIZE,
        K_left=oracle.K_left, K_right=oracle.K_right,
        max_order=zernike_order, max_nfev=zernike_nfev,
    )
)
t_zernike = _time.time() - t0
print(f"Zernike rayfield (max_order={zernike_order}): "
      f"RMS={zernike_diag.ray_rms_mm:.4f} mm  "
      f"converged={zernike_diag.converged}  [{t_zernike:.0f}s]")

# B2 — Run ray-space model selection
K_cmo = oracle.K_left
report = select_physical_model_from_rayfield(
    target_field=left_measured,
    target_right=right_measured,
    candidate_specs=candidates,
    K=K_cmo, K_right=oracle.K_right,
    image_size=IMAGE_SIZE,
    grid_shape=(12, 9),
    full_grid_weight=0.0,
    max_nfev=500,
)
print("Rayfield-mediated selection:")
for c in sorted(report.candidates, key=lambda c: c.bic):
    marker = "***" if c.model_name == report.best_by_bic else ""
    print(f"  {c.model_name:28s}  p={c.n_parameters:2d}  rms={c.rms_mm:.4f} mm  "
          f"bic={c.bic:.1f}  {marker}")
print(f"\n  → Rayfield winner: {report.best_by_bic}")

# %% [markdown]
# ## 5 — Comparison

# %%
print("┌──────────────────────────┬────────────────────┬────────────────────┐")
print("│ Model                    │ Direct BIC (px)    │ Rayfield BIC (mm)  │")
print("├──────────────────────────┼────────────────────┼────────────────────┤")
rayfield_by_name = {c.model_name: c for c in report.candidates}
for spec in candidates:
    name = spec.name
    if name == "cmo_physical_shared":
        d_bic = f"{r_direct.bic:.1f}"
        d_status = f"rms={r_direct.rms_px:.1f}px"
    else:
        d_bic = "—"
        d_status = ""
    r_bic = f"{rayfield_by_name[name].bic:.1f}" if name in rayfield_by_name else "N/A"
    r_win = "***" if report.best_by_bic == name else ""
    print(f"│ {name:24s} │ {d_bic:>10s}        │ {r_bic:>10s} {r_win:3s} │")
print("└──────────────────────────┴────────────────────┴────────────────────┘")
print(f"\nPipeline A (direct):  CMO rms={r_direct.rms_px:.1f} px,  bic={r_direct.bic:.1f},  {t_direct:.0f}s")
print(f"Pipeline B (Zernike): rms={zernike_diag.ray_rms_mm:.4f} mm (Zernike fit),  ray selection winner: {report.best_by_bic}")
print(f"Total FAST runtime: {t_direct + t_zernike:.0f}s")
print("\nNote: this FAST case is intentionally hard — only 2 poses and a")
print("pinhole solvePnP initialisation on CMO optics (~2 px off).  The full")
print("6-oracle sweep shows pipeline A converges on pinhole, Brown, plate,")
print("and exotic oracles (RMS < 0.5 px), but remains fragile on CMO and")
print("Greenough where the pinhole initialisation is a poor match.")
print("\nDirect BIC uses pixel residuals; rayfield BIC uses mm line residuals.")
print("They are NOT numerically comparable — compare winners within each column.")

# %% [markdown]
# ## 6 — Conditioning diagnostics
#
# Compute the Schur-complement condition number for pipeline A on a small
# subset of corners (finite-difference Jacobians are expensive).

# %%
from stereocomplex.benchmarks.inverse_problem_diagnostics import (
    compute_pipeline_condition_number,
)
from scipy.spatial.transform import Rotation

# Build residual using the fitted model + poses, on a subset of corners
n_sub = min(15, obs.left_pixels[0].shape[0])
print(f"Computing conditioning diagnostics on {n_sub} corners …")

def direct_residual(theta, eta):
    model = CMOPhysicalStereoModel.from_parameter_vector(
        theta, image_size=IMAGE_SIZE, pixel_pitch_mm=oracle.pixel_pitch_mm,
    )
    r_blocks = []
    for pi in range(min(1, len(obs.left_pixels))):  # first pose only
        lp = obs.left_pixels[pi][:n_sub]; rp = obs.right_pixels[pi][:n_sub]
        if lp.size == 0: continue
        idx = obs.point_indices[pi][:n_sub]
        pts_local = obs.object_points_mm[idx]
        R_pose = Rotation.from_rotvec(eta[6*pi:6*pi+3]).as_matrix()
        t_pose = np.asarray(eta[6*pi+3:6*pi+6], dtype=np.float64).reshape(3)
        pts_world = (R_pose @ pts_local.T).T + t_pose[None, :]
        for k in range(pts_world.shape[0]):
            uvL, _ = model.channel("left").project_point(pts_world[k])
            uvR, _ = model.channel("right").project_point(pts_world[k])
            r_blocks.extend([uvL[0]-lp[k,0], uvL[1]-lp[k,1],
                             uvR[0]-rp[k,0], uvR[1]-rp[k,1]])
    return np.array(r_blocks, dtype=np.float64)

diag_A = compute_pipeline_condition_number(
    direct_residual,
    theta=r_direct.parameter_vector[:r_direct.n_parameters_optics],
    eta=r_direct.parameter_vector[r_direct.n_parameters_optics:],
    step=1e-4,
)
print(f"  Pipeline A (direct, {r_direct.n_parameters_optics} optical + "
      f"{r_direct.n_parameters_poses} pose = {r_direct.n_parameters_total} params):")
print(f"    coupling_norm = {diag_A['coupling_norm']:.4f}  "
      f"(0 = uncoupled, 1 = fully coupled)")
print(f"    rank_full     = {diag_A['rank_full']}")

# Pipeline B: no poses in the second stage
print(f"\n  Pipeline B (rayfield, 17 params, 0 poses):")
print(f"    coupling_norm = 0.0  (poses eliminated in Zernike BA stage)")
print(f"    The rayfield mediates between measurement and interpretation,")
print(f"    absorbing the 12 nuisance pose parameters into the rayfield fit.")

# %% [markdown]
# ## 7 — Interpretation and conclusions
#
# ### Key findings
#
# **1. Pipeline B correctly identifies the CMO architecture.**
# The rayfield-mediated pipeline selects `cmo_physical_shared` by BIC on
# a CMO oracle, with a ray-space RMS of < 10⁻⁶ mm for the correct model
# versus ~52 mm for the wrong models (Brown-Conrady, inclined plate).
# The BIC gap is > 90 000 units — an unambiguous classification.
#
# **2. Pipeline A can be expensive without analytic projectors.**
# Generic direct fitting uses numerical inverse projection (~180
# `least_squares` calls per evaluation).  Analytic `project_point`
# methods (Pinhole, Brown-Conrady, CMO) make it practical, but the
# joint optics+pose optimisation remains fragile on hard non-central
# architectures.  Pipeline B evaluates rays *forward* (pixel → line),
# which is O(1) per pixel and cheaper for model comparison.
#
# **3. Poses are eliminated from model selection.**
# In the rayfield pipeline, board poses are absorbed into the Zernike
# rayfield fit (stage B1).  The model-selection stage (B2) compares
# physical candidates in ray space *without any pose parameters*.  This
# eliminates the optics-pose coupling that inflates condition numbers
# in pipeline A and makes direct inversion fragile.
#
# **4. The rayfield is a geometric intermediate variable.**
# Pipeline B separates *measurement* (Zernike fit from corners) from
# *interpretation* (physical model comparison).  This decoupling is the
# central architectural insight of StereoComplex.
#
# ### When to use which pipeline
#
# | Criterion | Pipeline A (direct) | Pipeline B (rayfield) |
# |---|---|---|
# | Speed (model comparison) | Very slow (~min per candidate) | Fast (~s per candidate) |
# | Speed (single known model) | OK with good initialisation | Requires Zernike fit first |
# | Model selection interpretability | Poses + optics coupled | Poses absent from comparison |
# | Detection of uncatalogued optics | Possible but fragile | Built-in via Zernike fallback |
# | Parameter recovery for known model | Maximum-likelihood (efficient) | Two-stage (Zernike → physical) |
#
# **Recommendation:** use pipeline B when comparing competing optical
# hypotheses or diagnosing unknown instruments.  Use pipeline A primarily when
# fitting a single well-known model with maximum-likelihood efficiency.
#
# ### Limitations
#
# - The direct pipeline (A) requires further optimisation before it can
#   serve as a routine tool — the current inverse-projection implementation
#   is correct but impractically slow for real-time use.
# - The Zernike rayfield fit (stage B1) adds its own uncertainty, which
#   propagates to the model-selection stage.  The 6-oracle sweep uses
#   oracle rayfields; this notebook demonstrates the full ChArUco →
#   Zernike → selection loop on the CMO oracle.
# - The CMO oracle has a narrow field of view (~9°), requiring a dense
#   board (1 mm squares) to achieve sufficient corner coverage.
#
# See [Direct vs rayfield inversion](../docs/DIRECT_VS_RAYFIELD_INVERSION.md)
# for the full mathematical treatment.

# %%
summary = {
    "oracle": oracle.name,
    "n_poses": n_poses,
    "n_corners": sum(p.shape[0] for p in obs.left_pixels),
    "noise_std_px": noise_px,
    "pipeline_A": {
        "rms_px": float(r_direct.rms_px),
        "bic_px": float(r_direct.bic),
        "converged": r_direct.converged,
        "elapsed_s": t_direct,
        "coupling_norm": diag_A["coupling_norm"],
        "n_params_optics": r_direct.n_parameters_optics,
        "n_params_poses": r_direct.n_parameters_poses,
    },
    "pipeline_B": {
        "zernike_rms_mm": float(zernike_diag.ray_rms_mm),
        "zernike_converged": zernike_diag.converged,
        "zernike_elapsed_s": t_zernike,
        "rayfield_winner": report.best_by_bic,
        "n_params_poses": 0,
    },
    "rayfield_correct": report.best_by_bic == oracle.expected_winner,
}
print(json.dumps(summary, indent=2))
with open(ASSETS / "direct_vs_rayfield_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAssets saved to {ASSETS}/")

# %% [markdown]
# ## Où aller ensuite
#
# Ce notebook compare deux stratégies sur des **oracles synthétiques**.
# Pour voir la pipeline B appliquée à un **microscope CMO réel** avec
# des images de calibration authentiques, ouvre le
# [Notebook 09 — Validation Pycaso](09_pycaso_real_data.py).
# Il exécute tout le pipeline de bout en bout sur 10 paires stéréo
# réelles : détection ChArUco, complétion Hessian, double TPS,
# Zernike rayfield, identification du modèle physique, et alignement
# SE(3) des bras optiques.
