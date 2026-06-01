"""Reproduce Table `tab:extrapolation_3d` of the CMO paper.

Common-metric (3-D reconstruction, micrometres) extrapolation comparison of the
Zernike rayfield against the Soloff polynomial, evaluated on held-out extreme
poses under TWO symmetric board-pose references:

  (i)  CMO-pose reference   : Zernike refit with poses frozen to the CMO model
                              (only the 180 rayfield coefficients optimised), so
                              both models share the pose set Soloff implicitly uses.
  (ii) Zernike-pose reference: per-frame poses self-consistent with the free-fit
                              Zernike field are recovered by pose-only least
                              squares, and Soloff is refit against that geometry.

A shared pose reference is mandatory because the absolute board pose is a gauge
freedom neither uncalibrated model fixes on its own; comparing the Zernike
reconstruction against the CMO poses without freezing conflates the rayfield
error with the (sub-mm) disagreement between the two bundle adjustments' pose
estimates. A single global rigid (Kabsch) transform, estimated on the training
frames, removes the residual camera-to-world gauge of the triangulated points.

Input  : docs/assets/cmo_paper/table_extrapolation_3d/intermediate_state.npz
         (symlink/copy of docs/assets/pycaso_real_data/intermediate_state.npz)
Output : docs/assets/cmo_paper/table_extrapolation_3d/extrapolation_3d.json

Run:
    rtk .venv/bin/python examples/notebooks/generate_table_extrapolation_3d.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from stereocomplex.eval.pycaso_soloff import PycasoSoloffStereoModel
from stereocomplex.rayfields.zernike_origin_field import (
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)
from stereocomplex.synthetic.parallel_plate import pinhole_ray_from_pixel

ASSET = Path("docs/assets/cmo_paper/table_extrapolation_3d")
DATA = ASSET / "intermediate_state.npz"
if not DATA.exists():  # fall back to the shared real-data asset
    DATA = Path("docs/assets/pycaso_real_data/intermediate_state.npz")
TRAIN = [1, 2, 3, 4, 5, 6, 7]
TEST = [0, 8, 9]
MAX_ORDER = 4


def _load():
    d = np.load(DATA, allow_pickle=True)
    obj = np.asarray(d["obj_pts"], float)
    Lpx = np.asarray(d["left_pixels"], float)
    Rpx = np.asarray(d["right_pixels"], float)
    oR = np.asarray(d["opt_R"], float)
    oT = np.asarray(d["opt_t"], float)
    fx = float(d["FX"])
    W, H = 2048, 2048
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]], float)
    return obj, Lpx, Rpx, oR, oT, K, (W, H)


def _perp(O, dv, X):
    delta = X - O
    return delta - np.sum(delta * dv, axis=1, keepdims=True) * dv


def _triangulate(OL, dvL, OR, dvR):
    """Two-ray least-squares midpoint per corner."""
    I = np.eye(3)
    P = np.empty((len(OL), 3))
    for k in range(len(OL)):
        A = (I - np.outer(dvL[k], dvL[k])) + (I - np.outer(dvR[k], dvR[k]))
        b = (I - np.outer(dvL[k], dvL[k])) @ OL[k] + (I - np.outer(dvR[k], dvR[k])) @ OR[k]
        P[k] = np.linalg.solve(A, b)
    return P


def _kabsch(P, Q):
    """Rigid R, t mapping P onto Q (no scale)."""
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, cQ - R @ cP


def _fit_zernike_frozen(obj, Lpx, Rpx, oR, oT, K, cfg, nm):
    """Coefficients-only fit with board poses frozen to (oR, oT)."""
    tmp = ZernikeRayField(K, cfg)

    def _basis_d0(px):
        out = []
        for i in range(len(px)):
            A = tmp.basis(px[i][:, 0], px[i][:, 1])
            d0 = pinhole_ray_from_pixel(px[i][:, 0], px[i][:, 1], K).reshape(-1, 3)
            out.append((A, d0))
        return out

    AL = _basis_d0(Lpx)
    AR = _basis_d0(Rpx)
    Xf = [(oR[i] @ obj.T).T + oT[i] for i in range(len(oR))]

    def chan(A, d0, oc, dc):
        dd = A @ dc
        dd = dd - np.sum(dd * d0, 1, keepdims=True) * d0
        dr = d0 + dd
        dv = dr / np.linalg.norm(dr, axis=1, keepdims=True)
        O = A @ oc
        O = O - np.sum(O * dv, 1, keepdims=True) * dv
        return O, dv

    def unpack(x):
        return (x[:nm * 3].reshape(nm, 3), x[nm * 3:nm * 6].reshape(nm, 3),
                x[nm * 6:nm * 9].reshape(nm, 3), x[nm * 9:].reshape(nm, 3))

    def resid(x):
        oL, dL, oR_, dR_ = unpack(x)
        out = []
        for i in TRAIN:
            OL, dvL = chan(*AL[i], oL, dL)
            ORr, dvR = chan(*AR[i], oR_, dR_)
            out += [_perp(OL, dvL, Xf[i]).ravel(), _perp(ORr, dvR, Xf[i]).ravel()]
        return np.concatenate(out)

    sol = least_squares(resid, np.zeros(nm * 12), method="trf", max_nfev=100)
    oL, dL, oR_, dR_ = unpack(sol.x)
    Lf = ZernikeRayField(K, cfg, ZernikeRayFieldCoefficients(oL, dL))
    Rf = ZernikeRayField(K, cfg, ZernikeRayFieldCoefficients(oR_, dR_))
    return Lf, Rf


def _rays(field_L, field_R, Lpx, Rpx, i):
    def n(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    OL, dL = field_L.ray(Lpx[i][:, 0], Lpx[i][:, 1])
    ORr, dR = field_R.ray(Rpx[i][:, 0], Rpx[i][:, 1])
    return OL, n(dL), ORr, n(dR)


def _score(Pd, Xref):
    R, t = _kabsch(np.vstack([Pd[i] for i in TRAIN]), np.vstack([Xref[i] for i in TRAIN]))

    def rms(idx):
        sq = [np.sum(((R @ Pd[i].T).T + t - Xref[i]) ** 2, axis=1) for i in idx]
        return float(np.sqrt(np.mean(np.concatenate(sq))) * 1000.0)

    tr, te = rms(TRAIN), rms(TEST)
    return {"train_um": tr, "test_um": te, "degradation": te / tr}


def main():
    obj, Lpx, Rpx, oR, oT, K, (W, H) = _load()
    cfg = ZernikeOriginFieldConfig(image_size=(W, H), max_order=MAX_ORDER)
    nm = len(cfg.modes())
    nF = len(Lpx)
    results = {}

    # ── (i) CMO-pose reference ───────────────────────────────────────────────
    Xcmo = {i: (oR[i] @ obj.T).T + oT[i] for i in range(nF)}
    Lf_fr, Rf_fr = _fit_zernike_frozen(obj, Lpx, Rpx, oR, oT, K, cfg, nm)
    Pz_fr = {i: _triangulate(*_rays(Lf_fr, Rf_fr, Lpx, Rpx, i)) for i in range(nF)}
    XYZ = np.vstack([Xcmo[i] for i in TRAIN])
    uL = np.vstack([Lpx[i] for i in TRAIN])
    uR = np.vstack([Rpx[i] for i in TRAIN])
    s2 = PycasoSoloffStereoModel.fit(XYZ_mm=XYZ, uv_left_px=uL, uv_right_px=uR, degree=2)
    s3 = PycasoSoloffStereoModel.fit(XYZ_mm=XYZ, uv_left_px=uL, uv_right_px=uR, degree=3)
    results["cmo_pose_ref"] = {
        "soloff_deg2": _score({i: s2.solve(Lpx[i], Rpx[i]) for i in range(nF)}, Xcmo),
        "soloff_deg3": _score({i: s3.solve(Lpx[i], Rpx[i]) for i in range(nF)}, Xcmo),
        "zernike_180p": _score(Pz_fr, Xcmo),
    }

    # ── (ii) Zernike-pose reference ──────────────────────────────────────────
    # Free-fit Zernike: full BA (poses + coefficients). Reuse the frozen field as
    # a warm start is not valid here; recover poses self-consistent with a free
    # field obtained from the standard solver.
    from stereocomplex.benchmarks.charuco_observation_simulator import CharucoObservationSet
    from stereocomplex.benchmarks.rayfield_from_observations import (
        fit_zernike_rayfield_from_charuco_observations,
    )
    obs = CharucoObservationSet(
        object_points_mm=obj,
        pose_rvecs=np.array([Rotation.from_matrix(oR[i]).as_rotvec() for i in TRAIN]),
        pose_tvecs=np.array([oT[i] for i in TRAIN]),
        left_pixels=[Lpx[i] for i in TRAIN], right_pixels=[Rpx[i] for i in TRAIN],
        point_indices=[np.arange(obj.shape[0], dtype=np.int32) for _ in TRAIN],
        noise_std_px=0.0, image_size=(W, H),
    )
    Lf, Rf, _ = fit_zernike_rayfield_from_charuco_observations(
        obs=obs, image_size=(W, H), K_left=K, K_right=K, max_order=MAX_ORDER,
        initial_poses_R=[oR[i] for i in TRAIN], initial_poses_t=[oT[i] for i in TRAIN],
        origin_reg_weight=1e-3, max_nfev=300,
    )
    # Recover Zernike-consistent poses (pose-only LS) for ALL frames.
    Xzern = {}
    for i in range(nF):
        OL, dL, ORr, dR = _rays(Lf, Rf, Lpx, Rpx, i)

        def res(p, OL=OL, dL=dL, ORr=ORr, dR=dR):
            R = Rotation.from_rotvec(p[:3]).as_matrix()
            X = (R @ obj.T).T + p[3:]
            return np.concatenate([_perp(OL, dL, X).ravel(), _perp(ORr, dR, X).ravel()])

        p0 = np.concatenate([Rotation.from_matrix(oR[i]).as_rotvec(), oT[i]])
        p = least_squares(res, p0, method="lm", xtol=1e-13).x
        Xzern[i] = (Rotation.from_rotvec(p[:3]).as_matrix() @ obj.T).T + p[3:]
    Pz = {i: _triangulate(*_rays(Lf, Rf, Lpx, Rpx, i)) for i in range(nF)}
    XYZz = np.vstack([Xzern[i] for i in TRAIN])
    s2z = PycasoSoloffStereoModel.fit(XYZ_mm=XYZz, uv_left_px=uL, uv_right_px=uR, degree=2)
    s3z = PycasoSoloffStereoModel.fit(XYZ_mm=XYZz, uv_left_px=uL, uv_right_px=uR, degree=3)
    results["zernike_pose_ref"] = {
        "soloff_deg2": _score({i: s2z.solve(Lpx[i], Rpx[i]) for i in range(nF)}, Xzern),
        "soloff_deg3": _score({i: s3z.solve(Lpx[i], Rpx[i]) for i in range(nF)}, Xzern),
        "zernike_180p": _score(Pz, Xzern),
    }

    ASSET.mkdir(parents=True, exist_ok=True)
    out = ASSET / "extrapolation_3d.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}\n")
    for ref, tab in results.items():
        print(f"[{ref}]")
        for m, v in tab.items():
            print(f"  {m:14s}: {v['train_um']:.2f} -> {v['test_um']:.2f} um "
                  f"({v['degradation']:.2f}x)")


if __name__ == "__main__":
    main()
