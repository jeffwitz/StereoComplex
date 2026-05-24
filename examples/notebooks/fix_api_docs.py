#!/usr/bin/env python3
"""Add docstrings to remaining undocumented public functions in api/"""
import ast, os, re

base = "/home/jeff/StereoComplex/src/stereocomplex/api"

FIXES = {
    "calibration.py": {
        "cost_for_f": (
            'def cost_for_f(\n',
            'def cost_for_f(\n    """Compute the cost contributed by a focal-length regularisation term.\n\n    Parameters\n    ----------\n    f_mm : float\n        Focal length estimate in millimetres.\n    model_cls : type\n        Model class providing ``n_parameters``.\n    f_trust_mm : float\n        Trusted focal length in millimetres.\n    f_cost_weight : float\n        Relative weight of the focal-length regularisation.\n    **kwargs\n        Additional keyword arguments forwarded to the model constructor.\n\n    Returns\n    -------\n    float\n        Regularisation cost.\n    """\n'
        ),
        "v_ij": (
            'def v_ij(\n',
            'def v_ij(\n    """Compute the i,j element of the image-of-the-absolute-conic vector.\n\n    Parameters\n    ----------\n    H : ndarray, shape (3, 3)\n        Homography matrix.\n    i : int\n        Column index (0-based).\n    j : int\n        Column index (0-based).\n\n    Returns\n    -------\n    ndarray, shape (6,)\n        The v_ij vector used for Zhang\'s calibration.\n    """\n'
        ),
    },
    "model_io.py": {
        "load_stereo_central_rayfield": (
            'def load_stereo_central_rayfield(\n',
            'def load_stereo_central_rayfield(\n    """Load a fitted StereoCentralRayFieldModel from a directory.\n\n    Parameters\n    ----------\n    model_dir : str or Path\n        Directory containing the saved model.\n\n    Returns\n    -------\n    StereoCentralRayFieldModel\n        The loaded central rayfield model.\n    """\n'
        ),
    },
    "stereo_reconstruction.py": {
        "C_L_mm": (
            'def C_L_mm(\n',
            'def C_L_mm(\n    """Left camera centre in world coordinates.\n\n    Parameters\n    ----------\n    model : StereoCentralRayFieldModel\n        The fitted central rayfield model.\n\n    Returns\n    -------\n    ndarray, shape (3,)\n        Camera centre in millimetres.\n    """\n'
        ),
        "C_R_in_L_mm": (
            'def C_R_in_L_mm(\n',
            'def C_R_in_L_mm(\n    """Right camera centre expressed in the left camera frame.\n\n    Parameters\n    ----------\n    model : StereoCentralRayFieldModel\n        The fitted central rayfield model.\n\n    Returns\n    -------\n    ndarray, shape (3,)\n        Right camera centre in the left frame, in millimetres.\n    """\n'
        ),
        "from_coeffs": (
            'def from_coeffs(\n',
            'def from_coeffs(\n    """Build a StereoCentralRayFieldModel from raw coefficient arrays.\n\n    Parameters\n    ----------\n    coeffs_left_x, coeffs_left_y : ndarray\n        Zernike coefficients for the left camera.\n    coeffs_right_x, coeffs_right_y : ndarray\n        Zernike coefficients for the right camera.\n    u0_px, v0_px : float\n        Unit-disk centre in pixels.\n    radius_px : float\n        Unit-disk radius in pixels.\n    nmax : int\n        Maximum Zernike radial order.\n    R_RL : ndarray, shape (3, 3)\n        Rotation from left to right camera frame.\n    t_RL : ndarray, shape (3,)\n        Translation from left to right camera frame.\n    rvecs : dict\n        Per-frame rotation vectors.\n    tvecs : dict\n        Per-frame translation vectors.\n    image_width_px, image_height_px : int\n        Sensor dimensions in pixels.\n\n    Returns\n    -------\n    StereoCentralRayFieldModel\n        The reconstructed central rayfield model.\n    """\n'
        ),
    },
    "_calibration_types.py": {
        "from_dict": (
            'def from_dict(\n',
            'def from_dict(\n    """Build a calibration type instance from a dictionary.\n\n    Parameters\n    ----------\n    cls : type\n        The target class.\n    d : dict\n        Dictionary with field names matching the class attributes.\n\n    Returns\n    -------\n    object\n        An instance of the target class.\n    """\n'
        ),
        "from_meta": (
            'def from_meta(\n',
            'def from_meta(\n    """Build a calibration type instance from dataset metadata.\n\n    Parameters\n    ----------\n    cls : type\n        The target class.\n    meta : dict\n        Dataset metadata dictionary.\n\n    Returns\n    -------\n    object\n        An instance of the target class.\n    """\n'
        ),
        "n_channels": (
            'def n_channels(\n',
            'def n_channels(\n    """Return the number of channels in a multi-camera dataset.\n\n    Parameters\n    ----------\n    self\n        The dataset instance.\n\n    Returns\n    -------\n    int\n        Number of channels.\n    """\n'
        ),
    },
}

for rel, funcs in FIXES.items():
    fp = os.path.join(base, rel)
    with open(fp) as f: content = f.read()
    for name, (old_pat, new_pat) in funcs.items():
        if new_pat in content:
            continue  # already fixed
        content = content.replace(old_pat, new_pat)
    with open(fp, 'w') as f: f.write(content)
    # verify syntax
    with open(fp) as f: ast.parse(f.read())
    print(f"{rel}: OK")
