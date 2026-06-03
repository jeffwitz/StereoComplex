"""Slow end-to-end reproduction of the CMO paper's flagship number.

Unlike ``test_paper_numbers_regression`` (which pins the *stored* JSON
artefacts), this test **recomputes** the 26-parameter CMO+SE(3) reprojection
error from the raw tracked inputs --- the ChArUco corner pixels, the board
poses and the 26-parameter optical vector in ``intermediate_state.npz`` --- using
the production :class:`CMOTelecentricStereoModel`. It closes the
compute -> paper loop: if the model evaluation or its parameterisation regresses,
the manuscript's headline ``1.06 px`` (P50 0.87, P95 1.84) is no longer
reproducible and this fails.

The projection mirrors ``examples/notebooks/refine_26p_corners_fast.py``
(``before_rayfield`` stage): each corner is ray-cast through the telecentric CMO
model, the per-arm SE(3) is applied, the ray is intersected with the board plane,
and the residual is converted to a pixel equivalent via the reference focal
length ``FX``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize

NPZ = Path("docs/assets/pycaso_real_data/intermediate_state.npz")
PIXEL_PITCH_MM = 0.0055


def _reprojection_px(data) -> tuple[float, float, float]:
    """Recompute (RMS, P50, P95) pixel reprojection of the 26p model."""
    left_px = data["left_pixels"]
    right_px = data["right_pixels"]
    obj = data["obj_pts"]
    opt_R = data["opt_R"]
    opt_t = data["opt_t"]
    x = data["x_26p"]
    n_frames = int(data["n_frames"])
    img = tuple(int(s) for s in data["image_size"])
    fx = float(data["FX"])

    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        x[:14], pixel_pitch_mm=PIXEL_PITCH_MM, image_size=img
    )
    arms = {"left": (x[14:17], x[17:20]), "right": (x[20:23], x[23:26])}

    errs: list[float] = []
    for pi in range(n_frames):
        Rm = Rotation.from_rotvec(Rotation.from_matrix(opt_R[pi]).as_rotvec()).as_matrix()
        t = opt_t[pi]
        Xw = (Rm @ obj.T).T + t[None, :]
        n_plane = Rm[:, 2]
        for k in range(obj.shape[0]):
            for uv, ch in ((left_px[pi, k], "left"), (right_px[pi, k], "right")):
                O_tel, d_tel = m_tel.ray(np.array([uv[0]]), np.array([uv[1]]), ch)
                rv, t_arm = arms[ch]
                R_arm = Rotation.from_rotvec(rv).as_matrix()
                O_u = (R_arm @ O_tel.reshape(3)) + t_arm
                d_u = _normalize((R_arm @ d_tel.reshape(1, 3).T).T)[0]
                dn = float(np.dot(d_u, n_plane))
                if abs(dn) <= 1e-10:
                    continue
                tl = float(np.dot(t - O_u, n_plane)) / dn
                e_mm = float(np.linalg.norm((O_u + tl * d_u) - Xw[k]))
                errs.append(e_mm / max(abs(tl), 1.0) * fx)
    a = np.asarray(errs)
    return float(np.sqrt(np.mean(a**2))), float(np.percentile(a, 50)), float(np.percentile(a, 95))


@pytest.mark.slow
def test_reproduce_cmo_26p_headline_reprojection():
    if not NPZ.exists():
        pytest.skip(f"missing tracked input {NPZ}")
    data = np.load(NPZ)
    rms, p50, p95 = _reprojection_px(data)
    # Manuscript abstract / Table: 1.06 px (P50 0.87, P95 1.84).
    assert rms == pytest.approx(1.06, abs=0.03), f"recomputed CMO 26p RMS = {rms:.4f} px"
    assert p50 == pytest.approx(0.87, abs=0.03), f"recomputed P50 = {p50:.4f} px"
    assert p95 == pytest.approx(1.84, abs=0.05), f"recomputed P95 = {p95:.4f} px"
