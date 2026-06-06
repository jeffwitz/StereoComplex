#!/usr/bin/env python3
"""Regenerate the specimen (coin) reconstructions reproducibly.

This is the missing producer for the specimen artefacts behind Figures 9-11 of
the CMO paper. The five-variant Schur-BA reconstructions were previously only
archived on Zenodo (no local script); this driver rebuilds them from the
*calibrations* that the repo already carries, so the figures become fully
reproducible.

Pipeline
--------
1. Dense coin correspondences by DIS optical flow. ``--dis improved`` (default)
   runs the variational-refinement configuration on the FULL image and extracts
   the ROI of the resulting field; ``--dis ultrafast`` reproduces the original
   ULTRAFAST-on-ROI configuration for validation.
2. Reconstruct the coin through five calibrations, reusing the validated
   ``stereocomplex.optical_ba.specimen`` machinery:
     - Zernike rayfield 57p (triangulated here),
     - CMO 26p rayfield initialisation (``x_26p`` of intermediate_state.npz),
     - CMO 26p free-pose BA: unregularised / isotropic 1e-2 / Schur 1e-3
       (``theta`` of schur_ba/ba_full_*.json — corner-based, specimen-DIS
       independent, hence reusable).
3. Metrics (Z MAD, magnification, median ray gap) are computed on the common
   valid ROI so they stay comparable to the published numbers.

Run::

    # validation (reproduces the committed ROI/ULTRAFAST numbers):
    rtk .venv/bin/python examples/notebooks/regenerate_specimen_reconstructions.py \
        --dis ultrafast --dry-run
    # full regeneration (overwrites the shared specimen npz):
    rtk .venv/bin/python examples/notebooks/regenerate_specimen_reconstructions.py \
        --dis improved
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402

from stereocomplex.physics.cmo_physical import _normalize  # noqa: E402
from stereocomplex.rayfields.zernike_origin_field import (  # noqa: E402
    ZernikeOriginFieldConfig,
    ZernikeRayField,
    ZernikeRayFieldCoefficients,
)

ASSETS = REPO / "docs/assets/pycaso_real_data"
SCHUR = ASSETS / "schur_ba"
PYCASO = REPO / "examples/pycaso_data/Exemple/Images_example"
ROI = (300, 1748, 300, 1748)  # x0, x1, y0, y1 — original specimen ROI (matches Table 9 extent)

DIS_CFG = {
    "ultrafast": {"border": 300, "finest": 0, "gd": 20, "vr": 0, "alpha": 20.0, "eps": 1e-2},
    "improved": {"border": 0, "finest": 0, "gd": 40, "vr": 50, "alpha": 120.0, "eps": 1e-4},
}


def dis_correspondences(cfg: dict) -> dict:
    """Compute dense coin correspondences with the requested DIS configuration."""
    imgL = cv2.imread(str(PYCASO / "left_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(str(PYCASO / "right_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise SystemExit(f"Pycaso coin images not found under {PYCASO} (git-ignored dataset).")
    imgL = imgL.astype(np.float32) / 255.0
    imgR = imgR.astype(np.float32) / 255.0
    H, W = imgL.shape
    b = cfg["border"]
    roiL, roiR = imgL[b:H - b, b:W - b], imgR[b:H - b, b:W - b]
    hr, wr = roiL.shape
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    dis.setFinestScale(cfg["finest"])
    dis.setPatchSize(8)
    dis.setPatchStride(4)
    dis.setGradientDescentIterations(cfg["gd"])
    dis.setUseMeanNormalization(True)
    dis.setUseSpatialPropagation(True)
    dis.setVariationalRefinementIterations(cfg["vr"])
    dis.setVariationalRefinementAlpha(cfg["alpha"])
    dis.setVariationalRefinementDelta(5.0)
    dis.setVariationalRefinementGamma(10.0)
    dis.setVariationalRefinementEpsilon(cfg["eps"])
    flow = dis.calc((roiL * 255).astype(np.uint8), (roiR * 255).astype(np.uint8), None)
    yy, xx = np.mgrid[0:hr, 0:wr]
    dx, dy = flow[..., 0], flow[..., 1]
    uL = (xx + b).ravel().astype(np.float64)
    vL = (yy + b).ravel().astype(np.float64)
    uR = (xx + b + dx).ravel().astype(np.float64)
    vR = (yy + b + dy).ravel().astype(np.float64)
    disp = np.sqrt(dx ** 2 + dy ** 2).ravel()
    inb = (uR >= 0) & (uR < W) & (vR >= 0) & (vR < H) & (disp < wr * 0.3)
    return {"uL": uL, "vL": vL, "uR": uR, "vR": vR,
            "image_size": np.array([W, H]), "inb": inb}


def reconstruct_zernike(corr: dict, image_size: tuple[int, int]) -> dict:
    """Triangulate the coin through the published constrained Zernike rayfield."""
    zv = json.loads((ASSETS / "zernike_pose_variants.json").read_text())["zernike_constrained"]
    arr = lambda a: np.asarray(a, np.float64).reshape(-1, 3)  # noqa: E731
    cfg = ZernikeOriginFieldConfig(image_size=image_size, max_order=2)
    k = np.array([[25600, 0, 1024], [0, 25600, 1024], [0, 0, 1]], np.float64)
    lf = ZernikeRayField(K=k, config=cfg, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=arr(zv["left_origin_coeffs"]),
        direction_coeffs=arr(zv["left_direction_coeffs"])))
    rf = ZernikeRayField(K=k, config=cfg, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=arr(zv["right_origin_coeffs"]),
        direction_coeffs=arr(zv["right_direction_coeffs"])))
    OL, dL = lf.ray(corr["uL"], corr["vL"])
    OR, dR = rf.ray(corr["uR"], corr["vR"])
    n = np.cross(dL, dR)
    nn = np.linalg.norm(n, axis=1)
    w = OR - OL
    t1 = np.sum(w * np.cross(dR, n), 1) / nn ** 2
    t2 = np.sum(w * np.cross(dL, n), 1) / nn ** 2
    P1 = OL + t1[:, None] * _normalize(dL)
    P2 = OR + t2[:, None] * _normalize(dR)
    P = (P1 + P2) / 2
    gap = np.linalg.norm(P1 - P2, axis=1)
    valid = (nn > 1e-12) & np.isfinite(P[:, 2])
    return {"X": P[:, 0], "Y": P[:, 1], "Z": P[:, 2], "gap": gap, "valid": valid}


def _recon_cmo(theta: np.ndarray, corr: dict, img: tuple[int, int]) -> dict:
    """Triangulate the coin through a CMO + per-arm SE(3) calibration (aligned)."""
    from scipy.spatial.transform import Rotation

    from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel

    def se3(O, d, rv, t):
        r = Rotation.from_rotvec(rv).as_matrix()
        return (r @ O.T).T + t, _normalize((r @ d.T).T)

    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        theta[:14], pixel_pitch_mm=0.0055, image_size=img)
    OL, dL = se3(*m_tel.ray(corr["uL"], corr["vL"], "left"), theta[14:17], theta[17:20])
    OR, dR = se3(*m_tel.ray(corr["uR"], corr["vR"], "right"), theta[20:23], theta[23:26])
    n = np.cross(dL, dR)
    nn = np.linalg.norm(n, axis=1)
    w = OR - OL
    t1 = np.sum(w * np.cross(dR, n), 1) / nn ** 2
    t2 = np.sum(w * np.cross(dL, n), 1) / nn ** 2
    P1, P2 = OL + t1[:, None] * dL, OR + t2[:, None] * dR
    P = (P1 + P2) / 2
    gap = np.linalg.norm(P1 - P2, axis=1)
    valid = (nn > 1e-12) & np.isfinite(P[:, 2])
    return {"X": P[:, 0], "Y": P[:, 1], "Z": P[:, 2], "gap": gap, "valid": valid}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dis", choices=["ultrafast", "improved"], default="improved")
    ap.add_argument("--dry-run", action="store_true", help="print metrics, write nothing")
    a = ap.parse_args()
    cfg = DIS_CFG[a.dis]
    print(f"DIS config [{a.dis}]: {cfg}")

    corr = dis_correspondences(cfg)
    img = tuple(int(x) for x in corr["image_size"])
    print(f"  correspondences: {corr['inb'].sum()}/{corr['inb'].size} in-bounds")

    def _theta(fname):
        return np.array(json.loads((SCHUR / fname).read_text())["theta"])

    thetas = {
        "cmo_26p_initial": np.load(ASSETS / "intermediate_state.npz")["x_26p"],
        "cmo_26p_ba_unregularized": _theta("ba_full_unregularized.json"),
        "cmo_26p_ba_iso_a1e-2": _theta("ba_full_isotropic_1e-2.json"),
        "cmo_26p_ba_schur_a1e-3": _theta("ba_full_schur_1e-3.json"),
    }
    zer = reconstruct_zernike(corr, img)
    cmo = {name: _recon_cmo(th, corr, img) for name, th in thetas.items()}

    # Common valid set: in-bounds, all variants triangulated, inside the ROI window.
    x0, x1, y0, y1 = ROI
    common = corr["inb"] & zer["valid"] & (corr["uL"] >= x0) & (corr["uL"] < x1) \
        & (corr["vL"] >= y0) & (corr["vL"] < y1)
    for r in cmo.values():
        common &= r["valid"]
    sel = np.flatnonzero(common)
    print(f"  common valid ROI points: {sel.size}")

    def raw_mad(Z):
        return float(np.median(np.abs(Z - np.median(Z))))

    metrics_out = {}
    variants = {"zernike_rayfield_57p": zer, **cmo}
    print("\n  variant                     Z_MAD(mm)  magnif.   med_gap(mm)")
    for name, r in variants.items():
        Xs, Ys, Zs, gs = r["X"][sel], r["Y"][sel], r["Z"][sel], r["gap"][sel]
        mad = raw_mad(Zs)
        mag = max(Xs.max() - Xs.min(), Ys.max() - Ys.min()) / 18.75
        metrics_out[name] = {"z_mad_mm": mad, "magnification": float(mag),
                             "median_gap_mm": float(np.median(gs))}
        print(f"  {name:27s} {mad:.4f}    {mag:.4f}  {np.median(gs):.4f}")
    ratio = (raw_mad_relief(zer, sel) / raw_mad_relief(cmo["cmo_26p_initial"], sel))
    print(f"\n  Zernike/CMO relief-std ratio = {ratio:.3f}")

    if a.dry_run:
        print("\n  --dry-run: nothing written.")
        return 0

    # ---- write artefacts (index-aligned on the common valid ROI set) ----
    uL, vL = corr["uL"][sel], corr["vL"][sel]
    uR, vR = corr["uR"][sel], corr["vR"][sel]
    np.savez_compressed(ASSETS / "specimen_correspondences.npz",
                        uL=uL, vL=vL, uR=uR, vR=vR,
                        image_size=np.array(img), roi=np.array(ROI))
    ones = np.ones(sel.size, dtype=bool)
    for fname, r in [("specimen_reconstruction_cmo26.npz", cmo["cmo_26p_initial"]),
                     ("specimen_reconstruction_zernike.npz", zer)]:
        np.savez_compressed(ASSETS / fname, X=r["X"][sel].astype(np.float32),
                            Y=r["Y"][sel].astype(np.float32), Z=r["Z"][sel].astype(np.float32),
                            gap=r["gap"][sel].astype(np.float32), valid=ones)
    schur_npz = {"zernike_rayfield_57p": zer, "cmo_26p_initial": cmo["cmo_26p_initial"],
                 "cmo_26p_ba_unregularized": cmo["cmo_26p_ba_unregularized"],
                 "cmo_26p_ba_iso_a1e-2": cmo["cmo_26p_ba_iso_a1e-2"],
                 "cmo_26p_ba_schur_a1e-3": cmo["cmo_26p_ba_schur_a1e-3"]}
    for name, r in schur_npz.items():
        np.savez_compressed(SCHUR / f"specimen_{name}.npz",
                            X=r["X"][sel].astype(np.float32), Y=r["Y"][sel].astype(np.float32),
                            Z=r["Z"][sel].astype(np.float32),
                            gap_mm=r["gap"][sel].astype(np.float32))
    (ASSETS / "specimen_reconstruction_metrics.json").write_text(
        json.dumps({"dis": a.dis, "dis_config": cfg, "roi": list(ROI),
                    "n_common_valid": int(sel.size), "variants": metrics_out,
                    "zernike_cmo_relief_ratio": ratio}, indent=2) + "\n")
    print(f"\n  wrote correspondences + 2 main + 5 schur_ba npz + metrics ({sel.size} points)")
    return 0


def raw_mad_relief(r: dict, sel: np.ndarray) -> float:
    """Plane-normalised relief std (DIS-sensitive) for the Figure-10 ratio."""
    X, Y, Z = r["X"][sel], r["Y"][sel], r["Z"][sel]
    a = np.column_stack([X, Y, np.ones(len(X))])
    c, *_ = np.linalg.lstsq(a, Z, rcond=None)
    return float(np.std(Z - a @ c))


if __name__ == "__main__":
    raise SystemExit(main())
