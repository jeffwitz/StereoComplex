"""Tests for CMOTelecentricStereoModel."""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from stereocomplex.physics import (
    CMOTelecentricStereoModel,
    CMOTelecentricChannelModel,
    fit_cmo_telecentric_model_to_rayfields,
)
from stereocomplex.physics.cmo_physical import _ray_rms
from stereocomplex.physics.model_selection import _grid_pixels, rayfield_two_plane_residuals


def test_cmo_telecentric_exports():
    """All telecentric symbols are importable from stereocomplex.physics."""
    assert CMOTelecentricStereoModel is not None
    assert CMOTelecentricChannelModel is not None
    assert fit_cmo_telecentric_model_to_rayfields is not None


def test_telecentric_zero_slope_constant_direction():
    """With s_x=s_y=0, direction should be constant across pixels."""
    m = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=np.deg2rad(11.25),
        d_y_common=0.06, s_x=0.0, s_y=0.0,
        image_size=(2048, 2048),
    )
    u_test = np.array([0., 1024., 2047.])
    v_test = np.array([0., 1024., 2047.])
    _, dL = m.ray(u_test, v_test, "left")
    _, dR = m.ray(u_test, v_test, "right")
    # All directions should be identical (s_x=s_y=0 means no pixel dependence)
    for d in [dL, dR]:
        for i in range(1, len(d)):
            np.testing.assert_allclose(d[i], d[0], atol=1e-12)


def test_telecentric_origin_is_constant_per_channel():
    """Origins should be constant per channel."""
    m = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=0.2, d_y_common=0.0,
        s_x=0.1, s_y=-0.1, image_size=(2048, 2048),
    )
    u_test = np.array([0., 1024., 2047.])
    v_test = np.array([0., 1024., 2047.])
    OL, _ = m.ray(u_test, v_test, "left")
    OR, _ = m.ray(u_test, v_test, "right")
    z_pupil = 65.0 - 62.0
    expected_L = np.array([-12.5, 0.0, z_pupil])
    expected_R = np.array([12.5, 0.0, z_pupil])
    for i in range(len(OL)):
        np.testing.assert_allclose(OL[i], expected_L, atol=1e-12)
        np.testing.assert_allclose(OR[i], expected_R, atol=1e-12)


def test_telecentric_left_right_symmetry():
    """With d_y=0, s_x=s_y=0, left/right d_x should be antisymmetric."""
    m = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=0.2, d_y_common=0.0,
        s_x=0.0, s_y=0.0, image_size=(2048, 2048),
    )
    _, dL = m.ray(np.array([1024.]), np.array([1024.]), "left")
    _, dR = m.ray(np.array([1024.]), np.array([1024.]), "right")
    np.testing.assert_allclose(dL[0, 0], -dR[0, 0], atol=1e-12)
    np.testing.assert_allclose(dL[0, 2], dR[0, 2], atol=1e-12)


def test_telecentric_slope_controls_dy_range():
    """s_y should control the d_y variation across the field."""
    m_zero = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=0.0, d_y_common=0.0,
        s_x=0.0, s_y=0.0, image_size=(2048, 2048),
    )
    m_pos = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=0.0, d_y_common=0.0,
        s_x=0.0, s_y=0.5, image_size=(2048, 2048),
    )
    u_test = np.array([1024., 1024.])
    v_test = np.array([0., 2047.])
    _, d0 = m_zero.ray(u_test, v_test, "left")
    _, dp = m_pos.ray(u_test, v_test, "left")
    assert abs(d0[1, 1] - d0[0, 1]) < 1e-12  # s_y=0: no variation
    assert dp[0, 1] != dp[1, 1]  # s_y!=0: some variation


def test_parameter_vector_round_trip():
    """parameter_vector → from_parameter_vector should round-trip."""
    m = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x=0.4, s_y=-0.4, image_size=(2048, 2048),
    )
    x = m.parameter_vector()
    m2 = CMOTelecentricStereoModel.from_parameter_vector(
        x, pixel_pitch_mm=0.0055, image_size=(2048, 2048))
    np.testing.assert_allclose(m2.parameter_vector(), x, atol=1e-14)


def test_fit_cmo_telecentric_recovers_oracle():
    """Fit should recover a zero-residual oracle."""
    oracle = CMOTelecentricStereoModel(
        f_obj_mm=62.0, working_distance_mm=65.0, b_mm=25.0,
        cx_principal_px=1024, cy_principal_px=1024,
        pixel_pitch_mm=0.0055, f_angular_mm=62.0,
        theta_convergence_half_rad=0.2, d_y_common=0.06,
        s_x=0.4, s_y=-0.4, image_size=(2048, 2048),
    )
    left_oracle = oracle.channel("left")
    right_oracle = oracle.channel("right")

    # Fit with origin fixed, direction free
    x0 = np.array([1024., 1024., 62.0, 0.2, 0.06, 0.4, -0.4], dtype=np.float64)
    lower = np.array([0., 0., 20., 0., -0.3, -10., -10.], dtype=np.float64)
    upper = np.array([2048., 2048., 200., 0.5, 0.3, 10., 10.], dtype=np.float64)
    base = np.array([62.0, 65.0, 25.0, 0,0,0,0,0,0,0], dtype=np.float64)
    from scipy.optimize import least_squares
    support = _grid_pixels((2048,2048), (12,9))
    def model_at(x):
        p=base.copy(); p[3]=x[0]; p[4]=x[1]; p[5]=x[2]; p[6]=x[3]; p[7]=x[4]; p[8]=x[5]; p[9]=x[6]
        return CMOTelecentricStereoModel.from_parameter_vector(p,pixel_pitch_mm=0.0055,image_size=(2048,2048))
    def residuals(x):
        m=model_at(x); l=m.channel("left"); r=m.channel("right")
        return np.concatenate([
            rayfield_two_plane_residuals(left_oracle,l,support,z_planes=(50.,80.)),
            rayfield_two_plane_residuals(right_oracle,r,support,z_planes=(50.,80.)),
        ])
    sol = least_squares(residuals,x0=x0,bounds=(lower,upper),loss="linear",max_nfev=200,xtol=1e-12,ftol=1e-12,gtol=1e-12)
    fitted = model_at(sol.x)
    lr = rayfield_two_plane_residuals(left_oracle,fitted.channel("left"),support,z_planes=(50.,80.))
    rr = rayfield_two_plane_residuals(right_oracle,fitted.channel("right"),support,z_planes=(50.,80.))
    rms = float(np.sqrt(0.5*(_ray_rms(lr)**2+_ray_rms(rr)**2)))
    assert rms < 1e-6, f"RMS should be near zero on oracle, got {rms}"
