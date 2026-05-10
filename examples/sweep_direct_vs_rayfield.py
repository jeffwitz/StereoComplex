"""Full 6-oracle sweep: direct (pipeline A) vs rayfield (pipeline B).

Saves results to docs/assets/direct_vs_rayfield_inversion/sweep_results.json
"""

from __future__ import annotations
import json, time, numpy as np
from pathlib import Path

from stereocomplex.benchmarks.model_selection_oracles import (
    build_all_oracles, build_pinhole_oracle, build_brown_oracle,
    build_plate_oracle, build_cmo_oracle, build_greenough_oracle,
    build_exotic_zernike_oracle,
)
from stereocomplex.benchmarks.charuco_observation_simulator import (
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.direct_inversion import (
    fit_direct_model_from_observations,
)
from stereocomplex.benchmarks.rayfield_from_observations import (
    fit_zernike_rayfield_from_charuco_observations,
)
from stereocomplex.physics import (
    PhysicalModelSpec, CentralPinholeModel, CentralBrownConradyModel,
    PinholeParallelPlateModel, select_physical_model_from_rayfield,
    NonCentralPolynomialChannelModel,
)
from stereocomplex.physics.cmo_physical import CMOPhysicalStereoModel
from stereocomplex.physics.cmo import CMOIntrinsics
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeCandidate, ZernikeOriginFieldConfig,
)

IMAGE_SIZE = (160, 120)
SEED = 42
ASSETS = Path("docs/assets/direct_vs_rayfield_inversion")
ASSETS.mkdir(parents=True, exist_ok=True)


def build_candidates(K, image_size, pixel_pitch_mm=None):
    _pitch = pixel_pitch_mm or 0.005
    intr = CMOIntrinsics(width=image_size[0], height=image_size[1],
                         fx=float(K[0, 0]), fy=float(K[1, 1]),
                         cx=float(K[0, 2]), cy=float(K[1, 2]))
    terms = NonCentralPolynomialChannelModel.default_terms()
    poly_initial = np.zeros(8 + 2 * len(terms), dtype=np.float64)
    poly_bounds = (
        np.r_[[-40, -40, -50, -1, -1, -0.1, -0.1, -1], -0.1 * np.ones(2 * len(terms))],
        np.r_[[+40, +40, +50, +1, +1, +0.1, +0.1, +1], +0.1 * np.ones(2 * len(terms))],
    )
    lo_config = ZernikeOriginFieldConfig(image_size=image_size, max_order=2)
    n_zernike = len(lo_config.modes()) * 6
    return [
        PhysicalModelSpec("central_pinhole", CentralPinholeModel, np.zeros(0)),
        PhysicalModelSpec("central_brown_conrady", CentralBrownConradyModel, np.zeros(5),
                          bounds=(np.array([-1, -1, -0.1, -0.1, -1]),
                                  np.array([1, 1, 0.1, 0.1, 1]))),
        PhysicalModelSpec("pinhole_parallel_plate", PinholeParallelPlateModel,
                          np.array([0, 0, 8]),
                          bounds=(np.array([-30, -30, 0]), np.array([30, 30, 50])),
                          model_kwargs={"eta": 1.5, "d1_mm": 80}),
        PhysicalModelSpec("cmo_physical_shared", CMOPhysicalStereoModel,
                          np.array([80, 120, 10, 50, 79.5, 59.5, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64),
                          model_kwargs={"pixel_pitch_mm": _pitch}),
        PhysicalModelSpec("polynomial_surrogate_channel",
                          NonCentralPolynomialChannelModel, poly_initial,
                          bounds=poly_bounds,
                          model_kwargs={"cmo_image_size": image_size,
                                        "aberration_terms": terms}),
        PhysicalModelSpec("zernike_compact", ZernikeCandidate,
                          np.zeros(n_zernike, dtype=np.float64), bounds=None,
                          model_kwargs={"config": lo_config, "fit_directions": True}),
    ]


oracle_builders = [
    ("pinhole", build_pinhole_oracle, 100.0, 5, 4, 20.0),
    ("brown", build_brown_oracle, 100.0, 5, 4, 20.0),
    ("plate", build_plate_oracle, 100.0, 5, 4, 20.0),
    ("cmo", build_cmo_oracle, 120.0, 9, 7, 1.0),
    ("greenough", build_greenough_oracle, 100.0, 5, 4, 20.0),
    ("exotic", build_exotic_zernike_oracle, 100.0, 5, 4, 20.0),
]

results = []
for name, builder, z_dist, sx, sy, sq in oracle_builders:
    print(f"\n{'='*60}")
    print(f"Oracle: {name}")
    print(f"{'='*60}")

    oracle = builder(image_size=IMAGE_SIZE)
    expected = oracle.expected_winner

    # Simulate observations
    t0 = time.time()
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=IMAGE_SIZE, n_poses=4, z_distance_mm=z_dist,
        squares_x=sx, squares_y=sy, square_size_mm=sq,
        min_corners_per_frame=5, max_pose_attempts=100, seed=SEED,
    )
    n_corners = sum(p.shape[0] for p in obs.left_pixels)
    t_sim = time.time() - t0
    print(f"  Sim: {len(obs.left_pixels)} poses, {n_corners} corners [{t_sim:.0f}s]")

    if n_corners < 10:
        print(f"  SKIP: not enough corners")
        results.append({"oracle": name, "status": "skip", "reason": "too few corners"})
        continue

    # Pipeline A: direct fit on expected winner only (fast path)
    winner_spec = None
    candidates = build_candidates(oracle.K_left, IMAGE_SIZE, oracle.pixel_pitch_mm)
    for spec in candidates:
        if spec.name == expected:
            winner_spec = spec
            break
    if winner_spec is None:
        winner_spec = candidates[0]

    t0 = time.time()
    try:
        r_direct = fit_direct_model_from_observations(
            obs, winner_spec, image_size=IMAGE_SIZE, max_nfev=100,
        )
    except Exception as e:
        print(f"  Pipeline A FAILED: {e}")
        r_direct = None
    tA = time.time() - t0

    # Pipeline B: Zernike from observations → ray-space selection
    t0 = time.time()
    try:
        left_z, right_z, z_diag = fit_zernike_rayfield_from_charuco_observations(
            obs, IMAGE_SIZE, oracle.K_left, oracle.K_right,
            max_order=2, max_nfev=50,
        )
        report = select_physical_model_from_rayfield(
            target_field=left_z, target_right=right_z,
            candidate_specs=candidates,
            K=oracle.K_left, K_right=oracle.K_right,
            image_size=IMAGE_SIZE, grid_shape=(12, 9),
            full_grid_weight=0.0, max_nfev=500,
        )
    except Exception as e:
        print(f"  Pipeline B FAILED: {e}")
        z_diag = None
        report = None
    tB = time.time() - t0

    row = {
        "oracle": name,
        "expected_winner": expected,
        "n_poses": len(obs.left_pixels),
        "n_corners": n_corners,
        "t_sim_s": t_sim,
        "pipeline_A": {
            "rms_px": float(r_direct.rms_px) if r_direct else None,
            "bic": float(r_direct.bic) if r_direct else None,
            "converged": r_direct.converged if r_direct else False,
            "elapsed_s": tA,
        } if r_direct else {"status": "failed"},
        "pipeline_B": {
            "zernike_rms_mm": float(z_diag.ray_rms_mm) if z_diag else None,
            "zernike_converged": z_diag.converged if z_diag else False,
            "winner": report.best_by_bic if report else None,
            "correct": (report.best_by_bic == expected) if report else False,
            "elapsed_s": tB,
            "candidates": [{"name": c.model_name, "rms_mm": float(c.rms_mm), "bic": float(c.bic)}
                           for c in report.candidates] if report else [],
        } if report else {"status": "failed"},
    }

    if r_direct and report:
        print(f"  A: rms={r_direct.rms_px:.1f}px conv={r_direct.converged} [{tA:.0f}s]")
        print(f"  B: winner={report.best_by_bic} correct={report.best_by_bic==expected} [{tB:.0f}s]")
    results.append(row)

    # Save incrementally
    with open(ASSETS / "sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

print(f"\n\n=== SWEEP COMPLETE ===")
print(f"Results saved to {ASSETS}/sweep_results.json")

# Summary table
print(f"\n{'Oracle':<20s} {'A converged':>12s} {'A rms':>10s} {'B winner':<28s} {'B correct':>10s}")
print("-" * 85)
for r in results:
    a = r.get("pipeline_A", {})
    b = r.get("pipeline_B", {})
    print(f"{r['oracle']:<20s} {str(a.get('converged','?')):>12s} "
          f"{str(a.get('rms_px','?')):>10s} "
          f"{str(b.get('winner','?')):<28s} {str(b.get('correct','?')):>10s}")
