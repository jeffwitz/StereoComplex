"""Tests for CMOWarpedStereoModel and image-space pre-warp."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stereocomplex.physics.cmo_physical import (
    CMOTelecentricStereoModel,
    CMOWarpedChannelModel,
    CMOWarpedStereoModel,
    _n_warp_coeff_per_axis,
    _n_warp_coeff_total,
    _poly_terms_2d,
    _polyval_2d,
    _ray_rms,
    compute_cmo_zernike_residuals,
)
from stereocomplex.physics.model_selection import _grid_pixels, rayfield_two_plane_residuals


def test_poly_terms_2d():
    assert _poly_terms_2d(0) == [(0, 0)]
    assert _poly_terms_2d(1) == [(0, 0), (1, 0), (0, 1)]
    assert len(_poly_terms_2d(2)) == 6
    assert len(_poly_terms_2d(3)) == 10


def test_n_warp_coeff_counts():
    assert _n_warp_coeff_per_axis(0) == 1
    assert _n_warp_coeff_per_axis(1) == 3
    assert _n_warp_coeff_per_axis(2) == 6
    assert _n_warp_coeff_total(1, shared=True) == 6   # 3 xi + 3 eta
    assert _n_warp_coeff_total(1, shared=False) == 12  # 2 channels × (3+3)
    assert _n_warp_coeff_total(0, shared=False) == 0   # identity, no coeffs


def test_polyval_2d_identity():
    """Level 1 with [0, 1, 0] should return u, [0, 0, 1] should return v."""
    u = np.array([0., 100., 1024., 2047.])
    v = np.array([2047., 1024., 512., 0.])
    xi = _polyval_2d(u, v, (0.0, 1.0, 0.0), level=1)
    eta = _polyval_2d(u, v, (0.0, 0.0, 1.0), level=1)
    np.testing.assert_allclose(xi, u)
    np.testing.assert_allclose(eta, v)


def test_warp_level0_equals_telecentric():
    """Level 0 with same parameters should produce identical rays."""
    base_kwargs = {
        "f_obj_mm": 62., "working_distance_mm": 65., "b_mm": 25.,
        "cx_principal_px": 1024., "cy_principal_px": 1024.,
        "pixel_pitch_mm": 0.0055, "f_angular_mm": 62.,
        "theta_convergence_half_rad": 0.2, "d_y_common": 0.06,
        "s_x_L": 0.3, "s_y_L": -0.3, "s_x_R": 0.3, "s_y_R": -0.3,
        "rho_x_L": 1., "rho_y_L": -1., "image_size": (2048, 2048),
    }
    tele = CMOTelecentricStereoModel(**base_kwargs)
    warped = CMOWarpedStereoModel(**base_kwargs, warp_level=0)
    u = np.array([0., 1024., 2047.])
    v = np.array([0., 1024., 2047.])
    for ch in ["left", "right"]:
        Ot, dt = tele.ray(u, v, ch)
        Ow, dw = warped.ray(u, v, ch)
        np.testing.assert_allclose(Ot, Ow, atol=1e-12)
        np.testing.assert_allclose(dt, dw, atol=1e-12)


def test_warp_level1_round_trip():
    """Parameter vector round-trip for level 1 with shared warp."""
    m = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.3, s_y_L=-0.3,
        rho_x_L=1.0, rho_y_L=-1.0,
        warp_level=1, shared_warp=True,
        warp_xi_L=(0.0, 1.0, 0.0),
        warp_eta_L=(0.0, 0.0, 1.0),
        warp_xi_R=(0.0, 1.0, 0.0),
        warp_eta_R=(0.0, 0.0, 1.0),
        image_size=(2048, 2048),
    )
    x = m.parameter_vector()
    m2 = CMOWarpedStereoModel.from_parameter_vector(
        x, pixel_pitch_mm=0.0055, image_size=(2048, 2048),
        warp_level=1, shared_warp=True)
    np.testing.assert_allclose(m2.parameter_vector(), x, atol=1e-14)
    assert m2.warp_level == 1
    assert m2.shared_warp
    np.testing.assert_allclose(m2.warp_xi_L, (0.0, 1.0, 0.0), atol=1e-14)


def test_parameter_vector_sizes():
    """Verify sizes for all (level, shared) combos."""
    sizes = []
    for level in [0, 1, 2]:
        for shared in [True, False]:
            m = CMOWarpedStereoModel(
                f_obj_mm=62., working_distance_mm=65., b_mm=25.,
                cx_principal_px=1024., cy_principal_px=1024.,
                pixel_pitch_mm=0.0055, f_angular_mm=62.,
                theta_convergence_half_rad=0.2, d_y_common=0.06,
                s_x_L=0.3, s_y_L=-0.3,
                rho_x_L=1.0, rho_y_L=-1.0,
                warp_level=level, shared_warp=shared,
                warp_xi_L=(0.,) * _n_warp_coeff_per_axis(level),
                warp_eta_L=(0.,) * _n_warp_coeff_per_axis(level),
                warp_xi_R=(0.,) * _n_warp_coeff_per_axis(level),
                warp_eta_R=(0.,) * _n_warp_coeff_per_axis(level),
                image_size=(2048, 2048),
            )
            n_base = 12
            n_warp = _n_warp_coeff_total(level, shared)
            assert m.n_parameters == n_base + n_warp
            assert m.parameter_vector().size == n_base + n_warp
            sizes.append((level, shared, m.n_parameters))
    assert sizes[0] == (0, True, 12)    # 12 base + 0 warp (identity)
    assert sizes[1] == (0, False, 12)   # level 0 always same
    assert sizes[2] == (1, True, 18)    # 12 + 6
    assert sizes[3] == (1, False, 24)   # 12 + 12
    assert sizes[4] == (2, True, 24)    # 12 + 12
    assert sizes[5] == (2, False, 36)   # 12 + 24


def test_shared_warp_copies_to_right():
    """With shared_warp=True, both channels have identical warp coeffs."""
    m = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.3, s_y_L=-0.3,
        rho_x_L=1.0, rho_y_L=-1.0,
        warp_level=1, shared_warp=True,
        warp_xi_L=(0.1, 1.0, 0.05),
        warp_eta_L=(0.2, -0.03, 1.0),
        warp_xi_R=(0.1, 1.0, 0.05),
        warp_eta_R=(0.2, -0.03, 1.0),
        image_size=(2048, 2048),
    )
    x = m.parameter_vector()
    m2 = CMOWarpedStereoModel.from_parameter_vector(
        x, pixel_pitch_mm=0.0055, image_size=(2048, 2048),
        warp_level=1, shared_warp=True)
    assert m2.warp_xi_L == m2.warp_xi_R
    assert m2.warp_eta_L == m2.warp_eta_R
    u = np.array([0., 1024., 2047.])
    v = np.array([512., 1024., 1536.])
    OL, _dL = m2.ray(u, v, "left")
    OR, _dR = m2.ray(u, v, "right")
    assert np.linalg.norm(OL - OR) > 0.01  # origins differ (stereo)
    # With shared warp, directions at centre pixel should be symmetric
    _, dLc = m2.ray(np.array([1024.]), np.array([1024.]), "left")
    _, dRc = m2.ray(np.array([1024.]), np.array([1024.]), "right")
    # d_y and d_z should be close (renormalization may cause tiny differences)
    np.testing.assert_allclose(dLc[0, 1], dRc[0, 1], atol=1e-3)
    np.testing.assert_allclose(dLc[0, 2], dRc[0, 2], atol=1e-3)


def test_residual_analysis_self_consistency():
    """Computing residuals between a model and itself gives zero."""
    m = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.3, s_y_L=-0.3,
        rho_x_L=1.0, rho_y_L=-1.0,
        warp_level=1, shared_warp=True,
        warp_xi_L=(0., 1., 0.), warp_eta_L=(0., 0., 1.),
        warp_xi_R=(0., 1., 0.), warp_eta_R=(0., 0., 1.),
        image_size=(2048, 2048),
    )
    left_ch = m.channel("left")
    right_ch = m.channel("right")
    result = compute_cmo_zernike_residuals(
        m, left_ch, right_ch,
        grid_shape=(9, 7), image_size=(2048, 2048), zernike_order=2,
    )
    assert result["dir_rms_deg_total"] < 1e-6
    assert result["mom_rms_mm_total"] < 1e-6


def test_oracle_self_consistent():
    """A warped model evaluated against itself should give zero residuals."""
    oracle = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.4, s_y_L=-0.4,
        rho_x_L=1.0, rho_y_L=-1.0,
        warp_level=1, shared_warp=True,
        warp_xi_L=(5.0, 1.02, -0.01), warp_eta_L=(-3.0, 0.005, 0.98),
        warp_xi_R=(5.0, 1.02, -0.01), warp_eta_R=(-3.0, 0.005, 0.98),
        image_size=(2048, 2048),
    )
    support = _grid_pixels((2048, 2048), (11, 9))
    lc = oracle.channel("left")
    rc = oracle.channel("right")
    lr = rayfield_two_plane_residuals(lc, lc, support, z_planes=(50., 80.))
    rr = rayfield_two_plane_residuals(rc, rc, support, z_planes=(50., 80.))
    rms = float(np.sqrt(0.5 * (_ray_rms(lr)**2 + _ray_rms(rr)**2)))
    assert rms < 1e-12, f"Self-consistent RMS should be zero, got {rms}"


def test_channel_model_delegates_to_rig():
    """CMOWarpedChannelModel should delegate to CMOWarpedStereoModel."""
    rig = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.3, s_y_L=-0.3,
        rho_x_L=1.0, rho_y_L=-1.0,
        warp_level=1, shared_warp=True,
        warp_xi_L=(0., 1., 0.), warp_eta_L=(0., 0., 1.),
        warp_xi_R=(0., 1., 0.), warp_eta_R=(0., 0., 1.),
        image_size=(2048, 2048),
    )
    ch = CMOWarpedChannelModel(rig=rig, channel="left")
    assert ch.n_parameters == rig.n_parameters
    np.testing.assert_allclose(ch.parameter_vector(), rig.parameter_vector())
    u = np.array([0., 1024.])
    v = np.array([0., 1024.])
    O_rig, d_rig = rig.ray(u, v, "left")
    O_ch, d_ch = ch.ray(u, v)
    np.testing.assert_allclose(O_rig, O_ch)
    np.testing.assert_allclose(d_rig, d_ch)

    # from_parameter_vector for channel model
    ch2 = CMOWarpedChannelModel.from_parameter_vector(
        rig.parameter_vector(), pixel_pitch_mm=0.0055,
        image_size=(2048, 2048), warp_level=1, shared_warp=True,
        channel="left")
    np.testing.assert_allclose(ch2.parameter_vector(), rig.parameter_vector())


def test_from_parameter_vector_rejects_invalid_sizes():
    """Invalid parameter vector sizes raise ValueError."""
    for bad_size in [5, 10, 11, 13, 15, 17]:
        x = np.zeros(bad_size, dtype=np.float64)
        try:
            CMOWarpedStereoModel.from_parameter_vector(
                x, pixel_pitch_mm=0.0055, image_size=(2048, 2048),
                warp_level=0, shared_warp=True)
            raise AssertionError(f"Expected ValueError for size {bad_size}")
        except ValueError:
            pass


def test_warp_level1_changes_rays():
    """Non-identity warp should change ray directions relative to level 0."""
    # Level 0: identity
    m0 = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.3, s_y_L=-0.3, rho_x_L=1., rho_y_L=-1.,
        warp_level=0, image_size=(2048, 2048),
    )
    # Level 1 with scale 1.5 in u (strong effect for clear detection)
    m1 = CMOWarpedStereoModel(
        f_obj_mm=62., working_distance_mm=65., b_mm=25.,
        cx_principal_px=1024., cy_principal_px=1024.,
        pixel_pitch_mm=0.0055, f_angular_mm=62.,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x_L=0.3, s_y_L=-0.3, rho_x_L=1., rho_y_L=-1.,
        warp_level=1, shared_warp=True,
        warp_xi_L=(0., 1.5, 0.), warp_eta_L=(0., 0., 1.0),
        warp_xi_R=(0., 1.5, 0.), warp_eta_R=(0., 0., 1.0),
        image_size=(2048, 2048),
    )
    # At u=2047 (far from centre), warp should change direction
    _, d0 = m0.ray(np.array([2047.]), np.array([1024.]), "left")
    _, d1 = m1.ray(np.array([2047.]), np.array([1024.]), "left")
    assert abs(float(d1[0, 0]) - float(d0[0, 0])) > 1e-8
    # At centre, warp with identity coeffs (0,1,0)/(0,0,1) does NOT change dir
    _, d0c = m0.ray(np.array([1024.]), np.array([1024.]), "left")
    _, d1c = m1.ray(np.array([1024.]), np.array([1024.]), "left")
    assert abs(float(d1c[0, 0]) - float(d0c[0, 0])) > 1e-8  # scale 1.5 affects centre too
