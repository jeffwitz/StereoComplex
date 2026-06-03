"""Smoke tests for the shared oracle builders."""

from __future__ import annotations

import numpy as np
import pytest

from stereocomplex.benchmarks.model_selection_oracles import (
    MultiCameraOracle,
    StereoOracle,
    build_all_oracles,
    build_brown_oracle,
    build_cmo_oracle,
    build_exotic_zernike_oracle,
    build_greenough_oracle,
    build_pinhole_n_camera_oracle,
    build_pinhole_oracle,
    build_plate_oracle,
)


def test_build_all_oracles_returns_six_cases():
    oracles = build_all_oracles()
    assert len(oracles) == 6
    names = [o.name for o in oracles]
    assert "central pinhole" in names
    assert "CMO shared-rig" in names


def test_each_oracle_has_left_right_ray_methods():
    for o in build_all_oracles():
        u = np.array([o.image_size[0] / 2])
        v = np.array([o.image_size[1] / 2])
        OL, dL = o.left_field.ray(u, v)
        _OR, _dR = o.right_field.ray(u, v)
        assert OL.shape == (1, 3)
        assert dL.shape == (1, 3)
        assert np.all(np.isfinite(OL))
        assert np.allclose(np.linalg.norm(dL, axis=1), 1.0)


def test_each_oracle_has_expected_winner():
    # Pin the expected winning model family per oracle. A bare isinstance/len
    # check would not catch a mislabelled oracle or a silently changed winner;
    # this golden map ties each synthetic ground-truth scene to the family the
    # model-selection pipeline is meant to recover (exercised end-to-end in
    # test_physical_model_selection.py).
    expected = {
        "central pinhole": "central_pinhole",
        "central Brown-Conrady": "central_brown_conrady",
        "inclined parallel plate": "pinhole_parallel_plate",
        "CMO shared-rig": "cmo_physical_shared",
        "Greenough (Brown-Conrady x2)": "central_brown_conrady",
        "uncatalogued Zernike": "zernike_compact",
    }
    winners = {o.name: o.expected_winner for o in build_all_oracles()}
    assert winners == expected


def test_cmo_oracle_has_identifiable_quantities():
    o = build_cmo_oracle()
    assert o.pixel_pitch_mm == 0.05
    gt = o.ground_truth_parameters
    assert abs(gt["f_obj_mm"] - 80.0) < 1e-9
    assert abs(gt["working_distance_mm"] - 120.0) < 1e-9
    assert abs(gt["b_mm"] - 8.0) < 1e-9


def test_cmo_oracle_K_matches_angular_scale():
    o = build_cmo_oracle()
    expected_fx = (
        o.ground_truth_parameters["f_tube_mm"] / o.ground_truth_parameters["pixel_pitch_mm"]
    )
    assert abs(o.K_left[0, 0] - expected_fx) < 1e-9


def test_pinhole_n_camera_oracle_builds_four_named_channels():
    oracle = build_pinhole_n_camera_oracle()

    assert isinstance(oracle, MultiCameraOracle)
    assert oracle.n_channels == 4
    assert oracle.channel_names == ("cam0", "cam1", "cam2", "cam3")
    for name in oracle.channel_names:
        u = np.array([oracle.image_size[0] / 2])
        v = np.array([oracle.image_size[1] / 2])
        origin, direction = oracle.field(name).ray(u, v)
        assert oracle.K(name).shape == (3, 3)
        assert origin.shape == direction.shape == (1, 3)
        assert np.allclose(np.linalg.norm(direction, axis=1), 1.0)


@pytest.mark.parametrize(
    "builder",
    [
        build_pinhole_oracle,
        build_brown_oracle,
        build_plate_oracle,
        build_cmo_oracle,
        build_greenough_oracle,
        build_exotic_zernike_oracle,
    ],
)
def test_each_oracle_builder_returns_stereo_oracle(builder):
    o = builder()
    assert isinstance(o, StereoOracle)
    assert o.K_left.shape == (3, 3)
    assert o.K_right.shape == (3, 3)
    assert len(o.image_size) == 2
