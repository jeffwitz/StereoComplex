#!/usr/bin/env python3
"""External profilometry validation of the specimen relief — Pycaso-Fig.9-style.

This figure answers the question the CMO paper leaves open in its specimen
section: *is the compact 26-parameter CMO physical model metrologically
faithful, or does it lose / invent axial relief?* A profilometric scan of the
same coin provides an external 3-D ground truth on the embossed letter "E".
The same dense DIS-stereo correspondence field is reconstructed by four models
and each plane-normalised relief is compared, amplitude and shape, against the
profilometry.

Narrative it establishes
------------------------
* The **Soloff polynomial** and the **CMO 26p** reconstructions are
  *indistinguishable* on the relief (``cc = 0.9997``, RMS difference
  ``0.57 µm`` — far below their ~2 µm distance to the profilometry). The
  compact physical model is therefore as faithful as the unconstrained
  polynomial baseline.
* Both Soloff and CMO track the profilometry to ~``1.10×`` amplitude with
  ``cc ≈ 0.96``.
* The **stage-anchored Zernike ray-field** recovers the same relief amplitude
  as the profilometry, showing that the ``~1.6×`` free-pose result was caused
  by the discarded translation-stage metric rather than by the rayfield basis.

Reproducibility (``no orphan figures`` rule)
--------------------------------------------
Default run reads the versioned cache
``docs/assets/cmo_paper/figure_external_profilo_relief/relief_comparison_data.npz``
(the four windowed reliefs) plus the profilometry, registration and DIS
parameter JSONs in the same folder, and writes the PDF + PNG. No heavy data
needed.

``--recompute`` re-derives the cache from scratch: it re-runs the DIS optical
flow (parameters read from ``dis_params.json``) on the Pycaso coin images,
re-reconstructs with the CMO 26p / stage-anchored Zernike / Soloff models,
applies the fixed rigid registration (``registration.json``) and rewrites the
cache. The Zernike coefficients are produced by
``evaluate_zernike_stage_prior.py``. The raw Pycaso images live outside the
repo (``examples/pycaso_data/``, git-ignored), exactly like the cached
reconstructions behind the other specimen figures.

Run
---
    rtk .venv/bin/python examples/notebooks/generate_fig_profilo_relief_comparison.py
    rtk .venv/bin/python examples/notebooks/generate_fig_profilo_relief_comparison.py --recompute
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ASSET = REPO / "docs/assets/cmo_paper/figure_external_profilo_relief"
CALIB = REPO / "docs/assets/pycaso_real_data"
PYCASO = REPO / "examples/pycaso_data/Exemple/Images_example"
CACHE = ASSET / "relief_comparison_data.npz"
PROFILO = ASSET / "coin_profilo_recale.npy"
STAGE_PRIOR_ASSET = REPO / "docs/assets/cmo_paper/figure10b_zernike_stage_prior"
STAGE_ANCHORED_RAYFIELD = STAGE_PRIOR_ASSET / "near_hard_zernike_rayfield.npz"
PAPER_FIGURE = REPO / "paper/cmo/figures/profilo_relief_comparison.png"

METHOD_ORDER = ["profilometry", "soloff", "cmo26", "zernike"]
METHOD_LABEL = {
    "profilometry": "Profilometry (ground truth)",
    "soloff": "Soloff (deg 3)",
    "cmo26": "CMO 26p (this work)",
    "zernike": "Zernike rayfield (stage-anchored)",
}
METHOD_COLOR = {"profilometry": "k", "soloff": "tab:blue",
                "cmo26": "tab:green", "zernike": "tab:red"}


def _plane_fit(z: np.ndarray) -> np.ndarray:
    """Subtract the best-fit affine tilt plane ``z ~ a*x + b*y + c`` (NaN-aware).

    Parameters
    ----------
    z : ndarray, shape (H, W)
        Height map in micrometres; NaN marks missing pixels.

    Returns
    -------
    ndarray, shape (H, W)
        Plane-normalised relief, µm. Removing each surface's own tilt on the
        *same* window makes the residual letter relief directly comparable
        between models regardless of their individual gauge orientation.
    """
    m = np.isfinite(z)
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.column_stack([xx[m], yy[m], np.ones(int(m.sum()))])
    c, *_ = np.linalg.lstsq(a, z[m], rcond=None)
    return z - (c[0] * xx + c[1] * yy + c[2])


def _cut_column(relief: np.ndarray) -> int:
    """Pick the vertical-cut column that crosses the three arms of the "E".

    Scans the central band of the profilometry relief and returns the column
    whose profile has exactly three elevated runs (the top / middle / bottom
    arms of the letter), preferring the one with the largest amplitude — the
    Pycaso-Fig.9 cut geometry.

    Parameters
    ----------
    relief : ndarray, shape (H, W)
        Plane-normalised profilometry relief, µm.

    Returns
    -------
    int
        Column index for the vertical profile.
    """
    h, w = relief.shape
    thr = 8.0

    def runs(col: int) -> int:
        pr = relief[:, col]
        m = np.isfinite(pr)
        if m.sum() < h * 0.7:
            return 0
        b = (pr > thr).astype(int)
        b[~m] = 0
        return int(np.sum(np.diff(np.r_[0, b, 0]) == 1))

    cands = list(range(int(0.40 * w), int(0.92 * w)))
    three = [c for c in cands if runs(c) == 3]
    pool = three or cands
    return max(pool, key=lambda c: np.nanstd(relief[:, c]))


def _dis_correspondences(imgL, imgR, p):
    """Dense left->right correspondences from DIS optical flow.

    Parameters
    ----------
    imgL, imgR : ndarray, shape (H, W)
        Rectified coin images, intensities in [0, 1].
    p : dict
        DIS configuration. ``border_px`` and ``disparity_filter_frac`` are always
        read. When ``preset_defaults`` is false (the tuned production path) every
        variational-refinement setter is applied from ``p``; when true only the
        OpenCV preset defaults are used (the non-tuned robustness baselines).
        ``preset`` selects the OpenCV preset ("ultrafast" | "fast" | "medium").

    Returns
    -------
    uL, vL, uR, vR : ndarray, shape (N,)
        Left/right pixel coordinates of the dense correspondence field.
    inb : ndarray of bool, shape (N,)
        In-bounds / disparity-sane mask.
    W : int
        Image width (square reconstruction-canvas side), pixels.
    """
    import cv2
    presets = {
        "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
        "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
        "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
    }
    H, W = imgL.shape
    b = int(p["border_px"])
    roiL, roiR = imgL[b:H - b, b:W - b], imgR[b:H - b, b:W - b]
    hr, wr = roiL.shape
    dis = cv2.DISOpticalFlow_create(presets[p.get("preset", "ultrafast")])
    if not p.get("preset_defaults", False):
        dis.setFinestScale(p["finestScale"])
        dis.setPatchSize(p["patchSize"])
        dis.setPatchStride(p["patchStride"])
        dis.setGradientDescentIterations(p["gradientDescentIterations"])
        dis.setUseMeanNormalization(p["useMeanNormalization"])
        dis.setUseSpatialPropagation(p["useSpatialPropagation"])
        dis.setVariationalRefinementIterations(p["variationalRefinementIterations"])
        dis.setVariationalRefinementAlpha(p["variationalRefinementAlpha"])
        dis.setVariationalRefinementDelta(p["variationalRefinementDelta"])
        dis.setVariationalRefinementGamma(p["variationalRefinementGamma"])
        dis.setVariationalRefinementEpsilon(p["variationalRefinementEpsilon"])
    flow = dis.calc((roiL * 255).astype(np.uint8), (roiR * 255).astype(np.uint8), None)

    yy, xx = np.mgrid[0:hr, 0:wr]
    dx, dy = flow[..., 0], flow[..., 1]
    uL = (xx + b).ravel().astype(np.float64)
    vL = (yy + b).ravel().astype(np.float64)
    uR = (xx + b + dx).ravel().astype(np.float64)
    vR = (yy + b + dy).ravel().astype(np.float64)
    disp = np.sqrt(dx ** 2 + dy ** 2).ravel()
    inb = (uR >= 0) & (uR < W) & (vR >= 0) & (vR < H) & (disp < wr * p["disparity_filter_frac"])
    return uL, vL, uR, vR, inb, W


def _reconstruct_windowed_reliefs(
    uL,
    vL,
    uR,
    vR,
    inb,
    W,
    reg,
    *,
    stage_anchored_zernike: bool = True,
):
    """Reconstruct the dense relief with Soloff/CMO/Zernike and window to the profilo.

    The three models all consume the *same* correspondence field
    ``(uL, vL) -> (uR, vR)``; only the per-model 3-D depth assignment differs.
    Each reconstructed Z is rasterised onto a square canvas, the fixed rigid
    registration ``reg`` warps the profilometry into the same frame, and all four
    maps (profilometry + three models) are cropped to the profilometry window and
    plane-normalised so the residual letter relief is directly comparable.

    Parameters
    ----------
    uL, vL, uR, vR : ndarray, shape (N,)
        Dense correspondence field from :func:`_dis_correspondences`.
    inb : ndarray of bool, shape (N,)
        In-bounds mask for the correspondence field.
    W : int
        Square canvas side (image width), pixels.
    reg : dict
        Rigid registration; ``affine_2x3`` maps the profilometry onto the images.
    stage_anchored_zernike : bool, default=True
        Use the near-fixed-ladder Zernike calibration selected in Figure 10.
        Set false only for the DIS-robustness diagnostic of the original
        free-pose depth-gain discrepancy.

    Returns
    -------
    dict[str, ndarray]
        ``{"profilometry", "soloff", "cmo26", "zernike"}`` plane-normalised
        reliefs over the common profilometry window, micrometres.
    """
    import cv2
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, str(REPO / "src"))
    from stereocomplex.eval.soloff_poly import SoloffPolynomialModel
    from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel, _normalize
    from stereocomplex.rayfields.zernike_origin_field import (
        ZernikeOriginFieldConfig,
        ZernikeRayField,
        ZernikeRayFieldCoefficients,
    )

    state = np.load(CALIB / "intermediate_state.npz")
    img_size = tuple(state["image_size"])
    x26 = state["x_26p"]
    m_tel = CMOTelecentricStereoModel.from_parameter_vector(
        x26[:14], pixel_pitch_mm=0.0055, image_size=img_size)

    def se3(o, d, rv, t):
        r = Rotation.from_rotvec(rv).as_matrix()
        return (r @ o.T).T + t[None, :], _normalize((r @ d.T).T)

    def triangulate(o1, d1, o2, d2):
        n = np.cross(d1, d2, axis=1)
        nn = np.linalg.norm(n, axis=1)
        w = o2 - o1
        t1 = np.sum(w * np.cross(d2, n, axis=1), axis=1) / nn ** 2
        t2 = np.sum(w * np.cross(d1, n, axis=1), axis=1) / nn ** 2
        return (o1 + t1[:, None] * d1 + o2 + t2[:, None] * d2) / 2, nn > 1e-12

    ol, dl = se3(*m_tel.ray(uL, vL, "left"), x26[14:17], x26[17:20])
    orr, dr = se3(*m_tel.ray(uR, vR, "right"), x26[20:23], x26[23:26])
    p_cmo, v_cmo = triangulate(ol, dl, orr, dr)

    if stage_anchored_zernike:
        if not STAGE_ANCHORED_RAYFIELD.exists():
            raise FileNotFoundError(
                f"{STAGE_ANCHORED_RAYFIELD} is missing; run "
                "evaluate_zernike_stage_prior.py first"
            )
        zv = np.load(STAGE_ANCHORED_RAYFIELD)
    else:
        zv = json.loads(
            (CALIB / "zernike_pose_variants.json").read_text()
        )["zernike_constrained"]
    cfg = ZernikeOriginFieldConfig(image_size=img_size, max_order=2)
    k = np.array([[25600, 0, 1024], [0, 25600, 1024], [0, 0, 1]], np.float64)
    lf = ZernikeRayField(K=k, config=cfg, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=np.asarray(zv["left_origin_coeffs"]).reshape(-1, 3),
        direction_coeffs=np.asarray(zv["left_direction_coeffs"]).reshape(-1, 3)))
    rf = ZernikeRayField(K=k, config=cfg, coefficients=ZernikeRayFieldCoefficients(
        origin_coeffs=np.asarray(zv["right_origin_coeffs"]).reshape(-1, 3),
        direction_coeffs=np.asarray(zv["right_direction_coeffs"]).reshape(-1, 3)))
    p_zer, v_zer = triangulate(*lf.ray(uL, vL), *rf.ray(uR, vR))

    left = state["left_pixels"].reshape(-1, 2).astype(float)
    right = state["right_pixels"].reshape(-1, 2).astype(float)
    obj = state["obj_pts"].astype(float)
    rot, tvec = state["opt_R"].astype(float), state["opt_t"].astype(float)
    xyz = np.concatenate([obj @ rot[i].T + tvec[i] for i in range(rot.shape[0])], 0)
    soloff = SoloffPolynomialModel.fit(left, right, xyz, degree=3, ridge=1e-3)
    p_sol = soloff.predict(np.column_stack([uL, vL]), np.column_stack([uR, vR]))

    ui, vi = uL.astype(int), vL.astype(int)

    def canvas(z, vmask):
        c = np.full((W, W), np.nan)
        good = inb & vmask & np.isfinite(z)
        c[vi[good], ui[good]] = z[good] * 1000.0  # mm -> micrometres
        return c

    cmo_c = canvas(p_cmo[:, 2], v_cmo)
    zer_c = canvas(p_zer[:, 2], v_zer)
    sol_c = canvas(p_sol[:, 2], np.isfinite(p_sol[:, 2]))

    m_aff = np.asarray(reg["affine_2x3"], np.float64)
    rec = np.load(PROFILO).astype(np.float64)
    rec_s = cv2.warpAffine(rec, m_aff, (W, W), flags=cv2.INTER_LINEAR, borderValue=np.nan)
    ys, xs = np.where(np.isfinite(rec_s))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    win = lambda im: _plane_fit(im[y0:y1, x0:x1])  # noqa: E731

    return {
        "profilometry": win(rec_s),
        "soloff": win(sol_c),
        "cmo26": win(cmo_c),
        "zernike": win(zer_c),
    }


def recompute_cache() -> None:
    """Re-run the full DIS -> reconstruction -> registration chain, rewrite cache.

    Reads the DIS parameters and rigid registration from the asset JSONs, the
    calibration state and per-model coefficients from ``docs/assets/
    pycaso_real_data/``, and the raw Pycaso coin images from the (git-ignored)
    ``examples/pycaso_data/`` tree. Writes the four windowed reliefs to
    ``relief_comparison_data.npz``. See the module docstring for provenance.
    """
    import cv2

    p = json.loads((ASSET / "dis_params.json").read_text())
    reg = json.loads((ASSET / "registration.json").read_text())

    imgL = cv2.imread(str(PYCASO / "left_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(str(PYCASO / "right_identification" / "coin.tif"), cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise SystemExit(
            f"Raw Pycaso coin images not found under {PYCASO} (git-ignored dataset). "
            "Run without --recompute to use the versioned cache.")
    imgL = imgL.astype(np.float32) / 255.0
    imgR = imgR.astype(np.float32) / 255.0

    uL, vL, uR, vR, inb, W = _dis_correspondences(imgL, imgR, p)
    reliefs = _reconstruct_windowed_reliefs(uL, vL, uR, vR, inb, W, reg)

    np.savez_compressed(
        CACHE,
        profilometry=reliefs["profilometry"].astype(np.float32),
        soloff=reliefs["soloff"].astype(np.float32),
        cmo26=reliefs["cmo26"].astype(np.float32),
        zernike=reliefs["zernike"].astype(np.float32))
    print(f"  wrote cache {CACHE.relative_to(REPO)}")


def _stats(relief: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Return (std, correlation-with-reference) over the common valid pixels, µm."""
    m = np.isfinite(relief) & np.isfinite(ref)
    return float(np.nanstd(relief[m])), float(np.corrcoef(relief[m], ref[m])[0, 1])


def draw() -> None:
    """Render the comparison figure (4 relief maps + overlaid profile) to PDF+PNG.

    Reads the cached windowed reliefs, computes per-method std / amplitude ratio
    / correlation against the profilometry, and writes
    ``profilo_relief_comparison.{pdf,png}`` into the asset folder. The PDF is the
    manuscript-referenced vector output; the PNG is the docs/preview sibling.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(CACHE)
    reliefs = {k: d[k].astype(np.float64) for k in METHOD_ORDER}
    ref = reliefs["profilometry"]
    h, _ = ref.shape
    ccol = _cut_column(ref)
    sp = float(np.nanstd(ref[np.isfinite(ref)]))

    # CMO vs Soloff agreement — the "marginal difference" headline.
    mcs = np.isfinite(reliefs["cmo26"]) & np.isfinite(reliefs["soloff"])
    cc_cs = float(np.corrcoef(reliefs["cmo26"][mcs], reliefs["soloff"][mcs])[0, 1])
    rms_cs = float(np.sqrt(np.nanmean((reliefs["cmo26"][mcs] - reliefs["soloff"][mcs]) ** 2)))

    fig = plt.figure(figsize=(11, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85], hspace=0.22, wspace=0.16)
    for k, name in enumerate(METHOD_ORDER):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        im = ax.imshow(reliefs[name], cmap="viridis", vmin=-20, vmax=50, aspect="equal")
        ax.axvline(ccol, color=METHOD_COLOR[name], lw=1.4)
        if name == "profilometry":
            sub = "ground truth"
        else:
            s, cc = _stats(reliefs[name], ref)
            sub = f"std={s:.1f} µm ({s / sp:.2f}×)  cc={cc:.3f}"
        ax.set_title(f"({chr(97 + k)}) {METHOD_LABEL[name]}\n{sub}", fontsize=10.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="z projected (µm)")

    axp = fig.add_subplot(gs[2, :])
    for name in METHOD_ORDER:
        axp.plot(np.arange(h), reliefs[name][:, ccol], color=METHOD_COLOR[name],
                 lw=1.8 if name == "profilometry" else 1.1,
                 label=METHOD_LABEL[name].split(" (")[0])
    axp.set_xlabel("pixel along vertical line")
    axp.set_ylabel("z (µm)")
    axp.grid(alpha=0.3)
    axp.legend(fontsize=9, loc="upper right")
    axp.set_title(
        f"(e) z along the vertical line — Soloff and CMO coincide "
        f"(cc={cc_cs:.3f}, RMS={rms_cs:.2f} µm); "
        "stage-anchored Zernike recovers the metric relief",
        fontsize=10.5)
    fig.suptitle("External profilometry validation of the specimen relief", fontsize=13, y=0.997)

    pdf = ASSET / "profilo_relief_comparison.pdf"
    png = ASSET / "profilo_relief_comparison.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(PAPER_FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  CMO vs Soloff: cc={cc_cs:.4f}  RMS={rms_cs:.3f} µm  (marginal)")
    for name in METHOD_ORDER[1:]:
        s, cc = _stats(reliefs[name], ref)
        print(f"  {METHOD_LABEL[name]:30s} std={s:5.1f} µm  ({s / sp:.2f}×)  cc={cc:.3f}")
    print(
        f"  wrote {pdf.relative_to(REPO)} + .png and "
        f"{PAPER_FIGURE.relative_to(REPO)}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true",
                    help="re-run DIS + reconstruction from the Pycaso images, rewrite the cache")
    args = ap.parse_args()
    if args.recompute or not CACHE.exists():
        if not CACHE.exists() and not args.recompute:
            print(f"cache {CACHE.name} missing → recomputing")
        recompute_cache()
    draw()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
