"""Non-regression gate: the multi-camera Zernike facade must reproduce the
stereo solver bit-exactly on the left/right case.

``fit_zernike_rayfields_from_multi_camera_observations`` is the Phase-1
channel-indexed entry point. For the ``("left", "right")`` topology it
reconstructs a stereo observation set and delegates to
``fit_zernike_rayfield_from_charuco_observations`` with identical arguments, so
the recovered coefficients, poses and ray RMS must be *identical* -- not merely
close. This is the equivalence gate specified in CLAUDE.md (Phase 2.1, step 5):
identical residual order, x0 and bounds force the optimiser down the same path,
so the max absolute difference must be exactly 0. Any drift means the facade has
diverged from the validated stereo path.
"""

from __future__ import annotations

import numpy as np

from stereocomplex.benchmarks.charuco_observation_simulator import (
    simulate_charuco_observations_from_rayfield,
)
from stereocomplex.benchmarks.model_selection_oracles import build_pinhole_oracle
from stereocomplex.benchmarks.rayfield_from_observations import (
    fit_zernike_rayfield_from_charuco_observations,
    fit_zernike_rayfields_from_multi_camera_observations,
)

IMG = (160, 120)


def _seeded_observations():
    oracle = build_pinhole_oracle(image_size=IMG)
    obs = simulate_charuco_observations_from_rayfield(
        oracle.left_field, oracle.right_field,
        image_size=IMG, n_poses=4, noise_std_px=0.0, seed=42,
        min_corners_per_frame=0,
    )
    return oracle, obs


def test_multicamera_facade_matches_stereo_bit_exact():
    oracle, obs = _seeded_observations()
    common = {"image_size": IMG, "max_order": 2, "max_nfev": 50}

    left, right, diag_s = fit_zernike_rayfield_from_charuco_observations(
        obs, IMG, oracle.K_left, oracle.K_right, max_order=2, max_nfev=50,
    )
    fields, diag_m = fit_zernike_rayfields_from_multi_camera_observations(
        obs.to_multi_camera(),
        intrinsics_by_channel={"left": oracle.K_left, "right": oracle.K_right},
        **common,
    )

    # Diagnostics: identical residual floor and optimiser effort.
    assert diag_m.ray_rms_mm == diag_s.ray_rms_mm
    assert diag_m.nfev == diag_s.nfev
    assert diag_m.channel_names == ("left", "right")

    # Coefficients: max absolute difference must be exactly zero.
    for name, stereo_field in (("left", left), ("right", right)):
        facade_field = fields.channel(name)
        assert np.max(np.abs(facade_field.origin_coeffs - stereo_field.origin_coeffs)) == 0.0
        assert (
            np.max(np.abs(facade_field.direction_coeffs - stereo_field.direction_coeffs)) == 0.0
        )
