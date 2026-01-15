import numpy as np

from stereocomplex.core.model_compact.central_rayfield import CentralRayFieldZernike
from stereocomplex.core.model_compact.zernike import zernike_design_matrix, zernike_modes


def test_zernike_modes_count_small_orders():
    assert len(zernike_modes(0)) == 1
    assert len(zernike_modes(1)) == 3
    assert len(zernike_modes(2)) == 6


def test_fit_from_gt_recovers_coeffs_on_synthetic_data():
    rng = np.random.default_rng(0)
    nmax = 4

    w, h = 200, 120
    u0, v0, radius = CentralRayFieldZernike.default_disk(w, h)

    # Sample points uniformly in the unit disk.
    N = 2000
    r = np.sqrt(rng.uniform(0.0, 1.0, size=N))
    theta = rng.uniform(-np.pi, np.pi, size=N)
    x_disk = r * np.cos(theta)
    y_disk = r * np.sin(theta)
    u = u0 + radius * x_disk
    v = v0 + radius * y_disk

    A, mask, modes = zernike_design_matrix(u, v, nmax=nmax, u0_px=u0, v0_px=v0, radius_px=radius)
    assert mask.all()

    K = len(modes)
    coeffs_x_true = rng.normal(scale=0.05, size=K)
    coeffs_y_true = rng.normal(scale=0.05, size=K)

    x = A @ coeffs_x_true
    y = A @ coeffs_y_true

    Z = rng.uniform(500.0, 1500.0, size=N)
    XYZ = np.stack([x * Z, y * Z, Z], axis=-1)

    model, stats = CentralRayFieldZernike.fit_from_gt(
        u_px=u,
        v_px=v,
        XYZ_cam_mm=XYZ,
        nmax=nmax,
        u0_px=u0,
        v0_px=v0,
        radius_px=radius,
        lam=0.0,
    )
    assert stats["train_point_to_ray_rms_mm"] < 1e-9

    assert np.max(np.abs(model.coeffs_x - coeffs_x_true)) < 1e-8
    assert np.max(np.abs(model.coeffs_y - coeffs_y_true)) < 1e-8

