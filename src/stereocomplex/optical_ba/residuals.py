"""Point-to-ray residuals for the Pycaso CMO + per-arm SE(3) bundle adjustment.

Implements the geometric residual specified by
``CdC_BA_optique_Schur_CMO_Pycaso.md`` §2.3: for each observed corner the
residual is the orthogonal component of ``X_world - O`` with respect to the
unit ray direction ``d``,

.. math::

    r_{pcj} = \\left(I - d\\,d^{T}\\right)\\,(X_{pj} - O),

where ``(O, d) = model.ray(pixel)`` after the per-channel SE(3) arm
correction, and ``X_{pj} = R_p X_j^{B} + t_p`` is the corner in the
reference frame for the per-frame board pose. The vector residual has three
scalar components, so the full residual stack of the Pycaso problem has
length ``3 * n_frames * n_channels * n_corners``.

The 26-parameter optical vector ``theta`` follows the layout used in the
existing ``examples/notebooks/refine_26p_on_corners.py`` script:

==========  ===============================================================
indices     content
==========  ===============================================================
``0..13``   :class:`~stereocomplex.physics.cmo_physical.CMOTelecentricStereoModel`
            parameter vector (14 telecentric CMO parameters).
``14..16``  Rotation vector ``rv_L`` of the left-arm SE(3), in radians.
``17..19``  Translation ``t_L`` of the left-arm SE(3), in millimetres.
``20..22``  Rotation vector ``rv_R`` of the right-arm SE(3), in radians.
``23..25``  Translation ``t_R`` of the right-arm SE(3), in millimetres.
==========  ===============================================================

The full BA parameter vector concatenates ``theta`` (26) with one
``(rotvec, tvec)`` block per frame (6 each).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel

N_THETA = 26
N_TEL = 14
N_POSE_PER_FRAME = 6


@dataclass(frozen=True)
class PycasoCMOObservations:
    """ChArUco observations packed for the Pycaso BA diagnostic.

    Attributes
    ----------
    obj_pts : ndarray, shape (M, 3)
        Inner ChArUco corner coordinates in the board frame, in millimetres.
    left_pixels, right_pixels : ndarray, shape (N, M, 2)
        Observed corner pixel coordinates per frame and channel; pixels are
        in image coordinates with the dataset's pixel-centre convention.
    image_size : tuple of 2 int
        Image ``(width, height)`` in pixels, used by the CMO model.
    pixel_pitch_mm : float
        Sensor pixel pitch in millimetres (Pycaso convention: 0.0055 mm).
    """

    obj_pts: np.ndarray
    left_pixels: np.ndarray
    right_pixels: np.ndarray
    image_size: tuple[int, int]
    pixel_pitch_mm: float

    @property
    def n_frames(self) -> int:
        return int(self.left_pixels.shape[0])

    @property
    def n_corners(self) -> int:
        return int(self.left_pixels.shape[1])


def _apply_se3(O: np.ndarray, d: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: E741 — `O` is the canonical optical-origin symbol
    """Apply a rigid SE(3) ``(R, t)`` to a stack of rays.

    Rotation preserves the unit norm of ``d`` analytically; the explicit
    renormalisation that ``refine_26p_on_corners.py`` performs is omitted on
    purpose so the residual is a smooth function of the parameters (the
    norm-preservation under ``R`` is exact).
    """
    return (R @ O.T).T + t[None, :], (R @ d.T).T


def point_to_ray_residuals_cmo_se3(
    x_full: np.ndarray, observations: PycasoCMOObservations
) -> np.ndarray:
    """Stack of 3-D transverse point-to-ray residuals (CDC §2.3).

    Parameters
    ----------
    x_full : ndarray, shape (26 + 6 * n_frames,)
        Concatenated parameter vector ``[theta, eta]``: 26 optical
        parameters (see module docstring) followed by one ``(rotvec, tvec)``
        block per frame.
    observations : PycasoCMOObservations
        ChArUco observations and CMO model metadata.

    Returns
    -------
    ndarray, shape (3 * n_frames * 2 * n_corners,)
        Flat residual vector. Each group of three consecutive entries is the
        transverse-distance vector of one ``(frame, channel, corner)``
        observation. Channels are stacked left-then-right; within a channel,
        frames vary slowest and corners fastest.

    Raises
    ------
    ValueError
        If ``x_full`` does not have the expected length.
    """
    x_full = np.asarray(x_full, dtype=np.float64).reshape(-1)
    n_frames = observations.n_frames
    expected = N_THETA + N_POSE_PER_FRAME * n_frames
    if x_full.size != expected:
        raise ValueError(
            f"x_full has length {x_full.size}, expected {expected} "
            f"({N_THETA} optical + 6 * {n_frames} pose)"
        )

    theta = x_full[:N_THETA]
    pose_vecs = x_full[N_THETA:].reshape(n_frames, N_POSE_PER_FRAME)

    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        theta[:N_TEL],
        pixel_pitch_mm=observations.pixel_pitch_mm,
        image_size=observations.image_size,
    )
    rv_L = theta[14:17]
    t_L = theta[17:20]
    rv_R = theta[20:23]
    t_R = theta[23:26]
    R_L = Rotation.from_rotvec(rv_L).as_matrix()
    R_R = Rotation.from_rotvec(rv_R).as_matrix()

    # Per-frame board-to-reference poses, built once.
    R_p_all = np.empty((n_frames, 3, 3), dtype=np.float64)
    for p in range(n_frames):
        R_p_all[p] = Rotation.from_rotvec(pose_vecs[p, :3]).as_matrix()
    t_p_all = pose_vecs[:, 3:6]

    obj_pts = observations.obj_pts
    # X_world[p, j] = R_p[p] @ obj_pts[j] + t_p[p]
    X_world = np.einsum("nij,mj->nmi", R_p_all, obj_pts) + t_p_all[:, None, :]
    X_world_flat = X_world.reshape(-1, 3)  # (N*M, 3)

    blocks: list[np.ndarray] = []
    for ch_name, R_ch, t_ch, pixels in (
        ("left", R_L, t_L, observations.left_pixels),
        ("right", R_R, t_R, observations.right_pixels),
    ):
        u_all = pixels[:, :, 0].reshape(-1)
        v_all = pixels[:, :, 1].reshape(-1)
        O, d = m_tel.ray(u_all, v_all, ch_name)  # noqa: E741
        O, d = _apply_se3(O, d, R_ch, t_ch)  # noqa: E741

        delta = X_world_flat - O
        # Transverse residual: r = (I - d d^T) (X - O)
        parallel = np.sum(delta * d, axis=1, keepdims=True) * d
        r = delta - parallel
        blocks.append(r.reshape(-1))

    return np.concatenate(blocks)


def default_parameter_scales(n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Default ``D_theta`` / ``D_eta`` scales for the Pycaso BA diagnostic.

    Scales follow the orders of magnitude listed in
    ``CdC_BA_optique_Schur_CMO_Pycaso.md`` §4.3, adapted to the Pycaso
    geometry (~25 mm baseline, ~65 mm working distance, ~2048 px image):

    - **Optical** (length-26): geometric distances ~10 mm, principal-point
      coordinates ~100 px, small dimensionless slopes ~0.1, SE(3) arm
      rotations ~1° (in radians) and translations ~1 mm.
    - **Pose** (length 6 per frame): per-frame board rotation ~1° (rad),
      translation ~1 mm.

    These scales are documented and exported by the diagnostic so that the
    coupling norm and Schur eigenvalues can be compared across runs.

    Parameters
    ----------
    n_frames : int
        Number of board frames in the BA.

    Returns
    -------
    theta_scales : ndarray, shape (26,)
        Per-parameter optical scale.
    pose_scales : ndarray, shape (6 * n_frames,)
        Per-parameter pose scale.
    """
    # 14 telecentric CMO params: layout follows
    # CMOTelecentricStereoModel.parameter_vector — mostly mm-valued plus a
    # couple of small dimensionless slopes (last two entries).
    theta_tel_scales = np.array(
        [
            10.0,  # f_obj_mm
            10.0,  # working_distance_mm
            5.0,   # b_mm (baseline)
            100.0, # cx_principal_px
            100.0, # cy_principal_px
            10.0,  # f_tube_mm
            0.1,   # theta_chief_rad (small)
            0.1,   # slope
            1.0,   # mm (z translation / offset)
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=np.float64,
    )
    arm_rad = np.deg2rad(1.0)
    theta_arm_scales = np.array(
        [
            arm_rad, arm_rad, arm_rad,  # rv_L
            1.0, 1.0, 1.0,              # t_L (mm)
            arm_rad, arm_rad, arm_rad,  # rv_R
            1.0, 1.0, 1.0,              # t_R (mm)
        ],
        dtype=np.float64,
    )
    theta_scales = np.concatenate([theta_tel_scales, theta_arm_scales])

    per_frame = np.array(
        [arm_rad, arm_rad, arm_rad, 1.0, 1.0, 1.0],
        dtype=np.float64,
    )
    pose_scales = np.tile(per_frame, n_frames)

    return theta_scales, pose_scales
