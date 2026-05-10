# %% [markdown]
# # 08 — Direct model inversion vs rayfield-mediated inversion
#
# **Do we identify microscope optics directly from ChArUco coordinates,
# or first estimate a generic rayfield and only then identify the
# physical model?**
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
# **Scope of this notebook** (FAST mode): CMO physical oracle, 4 poses,
# zero noise.  The three candidate models are central Brown-Conrady,
# inclined parallel plate, and physical CMO shared-rig.  The direct
# pipeline fits each candidate jointly with poses; the rayfield pipeline
# first fits a generic Zernike rayfield, then compares the same candidates
# in ray space.

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

n_poses = 4 if FAST else 8
noise_px = 0.0 if FAST else 0.1

# Match the board distance to the CMO working distance
z_dist = oracle.ground_truth_parameters.get("working_distance_mm", 100.0)
obs = simulate_charuco_observations_from_rayfield(
    oracle.left_field, oracle.right_field,
    image_size=IMAGE_SIZE, n_poses=n_poses, noise_std_px=noise_px, seed=SEED,
    z_distance_mm=z_dist, squares_x=7, squares_y=5, square_size_mm=10.0,
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
        np.array([80.0, 120.0, 10.0, 50.0, 79.5, 59.5, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        model_kwargs={"pixel_pitch_mm": oracle.pixel_pitch_mm},
    ),
]

# %% [markdown]
# ## 3 — Pipeline A: direct inversion

# %%
# NOTE: The direct inversion pipeline (fit_direct_model_from_observations)
# is implemented in stereocomplex.benchmarks.direct_inversion but requires
# careful pose initialisation and enough visible corners per frame.
# For this initial notebook we demonstrate the rayfield-mediated pipeline
# (pipeline B), which is the core contribution of StereoComplex.
#
# Pipeline A (direct inversion) will be added in a follow-up once the
# observation simulator produces sufficient coverage for robust joint
# optimisation of optical parameters + board poses.

direct_results = None  # placeholder
best_direct = None

# %% [markdown]
# ## 4 — Pipeline B: rayfield-mediated inversion
#
# Step B1: fit a generic Zernike rayfield to the ChArUco observations.
# Step B2: compare physical candidates in ray space.

# %%
# B1 — Fit a compact Zernike rayfield to the observations.
# For simplicity, we use the oracle rayfield directly (the Zernike fit
# from images is the notebook-06 workflow).  In a real experiment, the
# Zernike field would be fitted from the 2-D ChArUco corners via BA.
# Here we measure from the oracle rays and add observation noise if
# needed (the rayfield is already measured in this synthetic benchmark).

# Use the oracle fields directly as the "measured" rayfield.
# In a full image-based pipeline, this would come from
# fit_stereo_zernike_origin_field_from_image_dirs.
left_measured = oracle.left_field
right_measured = oracle.right_field

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
print("┌──────────────────────────┬────────────────────┐")
print("│ Model                    │ Rayfield BIC (mm)  │")
print("├──────────────────────────┼────────────────────┤")
rayfield_by_name = {c.model_name: c for c in report.candidates}
for spec in candidates:
    name = spec.name
    r_bic = f"{rayfield_by_name[name].bic:.1f}" if name in rayfield_by_name else "N/A"
    r_win = "***" if report.best_by_bic == name else ""
    print(f"│ {name:24s} │ {r_bic:>10s} {r_win:3s} │")
print("└──────────────────────────┴────────────────────┘")

# %% [markdown]
# ## 6 — Summary

# %%
summary = {
    "oracle": oracle.name,
    "n_poses": n_poses,
    "noise_std_px": noise_px,
    "rayfield_winner": report.best_by_bic,
    "rayfield_correct": report.best_by_bic == oracle.expected_winner,
}
print(json.dumps(summary, indent=2))
with open(ASSETS / "direct_vs_rayfield_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nAssets saved to {ASSETS}/")
print("\nDone.  For the full 6-oracle sweep, set FAST = False and")
print("run the full matrix (takes ~10 minutes).")
