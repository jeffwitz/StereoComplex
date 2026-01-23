from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stereocomplex.api.stereo_reconstruction import StereoCentralRayFieldModel
from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust
from stereocomplex.eval.soloff_poly import SoloffPolynomialModel
from stereocomplex.ray3d.central_stereo_ba import StereoFrameObservations, fit_central_stereo_rayfield_ba


@dataclass(frozen=True)
class ViewMarkerCenters:
    obj_xy_mm: np.ndarray  # (M,2)
    uv_px: np.ndarray  # (M,2)


@dataclass(frozen=True)
class StereoPairData:
    z_mm: float
    uvL_px: np.ndarray  # (N,2) chess corners (predicted)
    uvR_px: np.ndarray  # (N,2)
    P_board_mm: np.ndarray  # (N,3)
    centers_L: ViewMarkerCenters
    centers_R: ViewMarkerCenters


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x)))


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _apply_lossy_codec(*, cv2, img_gray: np.ndarray, codec: str, quality: int) -> np.ndarray:
    """
    Apply an in-memory lossy re-encode/decode to simulate compressed acquisition.
    """
    codec = str(codec).lower().strip()
    if codec in {"none", "raw", ""}:
        return img_gray

    img_gray = np.asarray(img_gray)
    if img_gray.ndim != 2:
        raise ValueError("expected grayscale image (H,W)")

    quality = int(quality)
    if codec == "jpeg":
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 0, 100))]
    elif codec == "webp":
        ext = ".webp"
        params = [int(cv2.IMWRITE_WEBP_QUALITY), int(np.clip(quality, 0, 100))]
    else:
        raise ValueError(f"unsupported codec: {codec!r}")

    ok, buf = cv2.imencode(ext, img_gray, params)
    if not ok:
        raise RuntimeError(f"imencode failed for codec={codec}")
    dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if dec is None:
        raise RuntimeError(f"imdecode failed for codec={codec}")
    return np.asarray(dec)


def _board_id_to_square_rc(*, squares_x: int, squares_y: int, parity: int) -> dict[int, tuple[int, int]]:
    parity = int(parity) & 1
    out: dict[int, tuple[int, int]] = {}
    idx = 0
    for r in range(int(squares_y)):
        for c in range(int(squares_x)):
            if ((r + c) & 1) == parity:
                out[idx] = (r, c)
                idx += 1
    return out


def _chess_corners_xy_mm(*, squares_x: int, squares_y: int, square_size_mm: float) -> np.ndarray:
    xy: list[tuple[float, float]] = []
    sq = float(square_size_mm)
    for r in range(1, int(squares_y)):
        for c in range(1, int(squares_x)):
            xy.append((float(c) * sq, float(r) * sq))
    return np.asarray(xy, dtype=np.float64).reshape(-1, 2)


def _estimate_K0_from_homographies(*, homographies: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    """
    Constrained Zhang-style self-calibration from plane homographies.

    Returns a 3x3 K0 in pixels. If the estimate is unstable, falls back to a generic K0.
    """
    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    K_fallback = np.array([[1.5 * float(max(w, h)), 0.0, cx], [0.0, 1.5 * float(max(w, h)), cy], [0.0, 0.0, 1.0]])

    Hs: list[np.ndarray] = []
    for H in homographies:
        H = np.asarray(H, dtype=np.float64).reshape(3, 3)
        if not np.all(np.isfinite(H)):
            continue
        if abs(float(H[2, 2])) < 1e-12:
            continue
        Hs.append(H / float(H[2, 2]))
    if len(Hs) < 3:
        return K_fallback.astype(np.float64)

    f_min = 0.5 * float(max(w, h))
    f_max = 3.0 * float(max(w, h))
    fs = np.logspace(np.log10(f_min), np.log10(f_max), num=80, dtype=np.float64)

    def cost_for_f(f: float) -> float:
        Kinv = np.array([[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]], dtype=np.float64)
        c = 0.0
        for H in Hs:
            Hn = Kinv @ H
            h1 = Hn[:, 0]
            h2 = Hn[:, 1]
            n1 = float(np.linalg.norm(h1))
            n2 = float(np.linalg.norm(h2))
            if not np.isfinite(n1) or not np.isfinite(n2) or n1 < 1e-12 or n2 < 1e-12:
                continue
            dot = float(np.dot(h1, h2)) / (n1 * n2)
            ratio = (n1 / n2) - 1.0
            c += dot * dot + ratio * ratio
        return c

    costs = np.array([cost_for_f(float(f)) for f in fs], dtype=np.float64)
    if np.all(np.isfinite(costs)):
        f0 = float(fs[int(np.argmin(costs))])
        if np.isfinite(f0) and f0 > 1.0:
            return np.array([[f0, 0.0, cx], [0.0, f0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    return K_fallback.astype(np.float64)


def _init_pose_from_homography(
    cv2,
    *,
    obj_xy_mm: np.ndarray,
    uv_px: np.ndarray,
    K0: np.ndarray,
    ransac_thresh_px: float = 3.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Initialize (rvec,tvec) from a planar homography using an estimated gauge K0.

    This is *not* solvePnP: we only use the plane-induced homography H and a K0
    obtained from homography self-calibration (Zhang-style). K0 is used as a gauge
    to map pixels to normalized coordinates for the decomposition.
    """
    obj_xy_mm = np.asarray(obj_xy_mm, dtype=np.float64).reshape(-1, 2)
    uv_px = np.asarray(uv_px, dtype=np.float64).reshape(-1, 2)
    if obj_xy_mm.shape[0] < 6:
        return None

    H, _mask = cv2.findHomography(obj_xy_mm, uv_px, method=cv2.RANSAC, ransacReprojThreshold=float(ransac_thresh_px))
    if H is None:
        return None

    K0 = np.asarray(K0, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(K0)) or abs(float(K0[2, 2])) < 1e-12:
        return None
    if abs(float(K0[2, 2]) - 1.0) > 1e-6:
        K0 = K0 / float(K0[2, 2])

    Hn = np.linalg.inv(K0) @ H
    h1 = Hn[:, 0]
    h2 = Hn[:, 1]
    h3 = Hn[:, 2]
    s1 = np.linalg.norm(h1)
    s2 = np.linalg.norm(h2)
    if not np.isfinite(s1) or not np.isfinite(s2) or s1 < 1e-12 or s2 < 1e-12:
        return None
    s = 1.0 / (0.5 * (s1 + s2))
    r1 = s * h1
    r2 = s * h2
    r3 = np.cross(r1, r2)
    R0 = np.stack([r1, r2, r3], axis=1)
    U, _S, Vt = np.linalg.svd(R0)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1.0
        Rm = U @ Vt

    t = (s * h3).reshape(3).astype(np.float64)
    if not np.all(np.isfinite(t)) or t[2] <= 0:
        t = (-t).reshape(3)
        if t[2] <= 0:
            return None

    rvec, _ = cv2.Rodrigues(Rm)
    return rvec.reshape(3).astype(np.float64), t.reshape(3).astype(np.float64)


def _rig_from_poses(
    *,
    fids: list[int],
    rvecs_L: dict[int, np.ndarray],
    tvecs_L: dict[int, np.ndarray],
    rvecs_R: dict[int, np.ndarray],
    tvecs_R: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation as R  # type: ignore

    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for fid in fids:
        R_L = R.from_rotvec(rvecs_L[fid]).as_matrix()
        R_R = R.from_rotvec(rvecs_R[fid]).as_matrix()
        t_L = tvecs_L[fid].reshape(3)
        t_R = tvecs_R[fid].reshape(3)

        R_RL = R_R @ R_L.T
        t_RL = t_R - R_RL @ t_L
        Rs.append(R_RL)
        ts.append(t_RL)

    rot_mean = R.from_matrix(np.stack(Rs, axis=0)).mean().as_matrix()
    t_mean = np.mean(np.stack(ts, axis=0), axis=0)
    return rot_mean.astype(np.float64), t_mean.astype(np.float64)


def _init_coeffs_pinhole_prior(
    *,
    uv_all: np.ndarray,
    nmax: int,
    image_size: tuple[int, int],
    f0_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize Zernike coefficients to approximate a simple pinhole prior:
      x ~= (u-cx)/f0,  y ~= (v-cy)/f0
    """
    from stereocomplex.ray3d.central_ba import default_disk  # noqa: PLC0415
    from stereocomplex.core.model_compact.zernike import zernike_design_matrix  # noqa: PLC0415

    w, h = int(image_size[0]), int(image_size[1])
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    u0, v0, radius = default_disk(w, h)

    uv_all = np.asarray(uv_all, dtype=np.float64).reshape(-1, 2)
    if uv_all.size == 0:
        uv_all = np.zeros((1, 2), dtype=np.float64)

    A, mask, _modes = zernike_design_matrix(
        uv_all[:, 0], uv_all[:, 1], nmax=int(nmax), u0_px=float(u0), v0_px=float(v0), radius_px=float(radius)
    )
    if not np.all(mask):
        A = A[mask]
        uv_all = uv_all[mask]

    x_t = (uv_all[:, 0] - cx) / float(f0_px)
    y_t = (uv_all[:, 1] - cy) / float(f0_px)

    lam = 1e-9
    ATA = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float64)
    ax = np.linalg.solve(ATA, A.T @ x_t)
    ay = np.linalg.solve(ATA, A.T @ y_t)
    return ax.astype(np.float64), ay.astype(np.float64)


def _load_pairs(
    *,
    left_dir: Path,
    right_dir: Path,
    exts: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp"),
) -> list[tuple[float, Path, Path]]:
    left = {float(p.stem): p for p in left_dir.iterdir() if p.suffix.lower() in exts}
    right = {float(p.stem): p for p in right_dir.iterdir() if p.suffix.lower() in exts}
    zs = sorted(set(left.keys()) & set(right.keys()))
    return [(float(z), left[float(z)], right[float(z)]) for z in zs]


def _detect_marker_centers(
    *,
    cv2,
    aruco,
    detector,
    img_gray: np.ndarray,
    id_to_center_mm: dict[int, np.ndarray],
) -> ViewMarkerCenters | None:
    corners, ids, _rej = detector.detectMarkers(img_gray)
    if ids is None or len(ids) == 0:
        return None
    ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    obj_xy: list[np.ndarray] = []
    uv: list[np.ndarray] = []
    for k, mid in enumerate(ids.tolist()):
        if int(mid) not in id_to_center_mm:
            continue
        c = np.asarray(corners[k], dtype=np.float64).reshape(4, 2)
        uv_c = np.mean(c, axis=0)
        obj_xy.append(id_to_center_mm[int(mid)].reshape(2))
        uv.append(uv_c.reshape(2))
    if len(obj_xy) < 6:
        return None
    return ViewMarkerCenters(obj_xy_mm=np.stack(obj_xy, axis=0), uv_px=np.stack(uv, axis=0))


def _predict_chess_corners(
    *,
    centers: ViewMarkerCenters,
    chess_xy_mm: np.ndarray,
    tps_lam: float,
    tps_huber: float,
    tps_iters: int,
) -> np.ndarray:
    return predict_points_rayfield_tps_robust(
        centers.obj_xy_mm,
        centers.uv_px,
        chess_xy_mm,
        lam=float(tps_lam),
        huber_c=float(tps_huber),
        iters=int(tps_iters),
        ransac_reproj_px=4.0,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="3D ray-field stereo calibration/reconstruction on Pycaso Z-sweep real images.")
    ap.add_argument(
        "--pycaso-images-root",
        type=Path,
        default=Path("/home/jeff/Code/Pycaso/Exemple/Images_example"),
        help="Root folder containing left_calibration*/right_calibration*.",
    )
    ap.add_argument(
        "--pair",
        default="calibration11",
        choices=["calibration", "calibration2", "calibration11"],
        help="Which Pycaso calibration pair to use (folder suffix).",
    )
    ap.add_argument("--max-frames", type=int, default=10, help="Limit number of frames (0=all).")
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--max-nfev", type=int, default=150)
    ap.add_argument("--max-points-per-frame", type=int, default=0, help="Optional cap on the number of corners per frame (0=all).")
    ap.add_argument("--subsample-seed", type=int, default=0)
    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=3.0)
    ap.add_argument("--tps-iters", type=int, default=3)
    ap.add_argument("--codec", default="none", choices=["none", "jpeg", "webp"], help="Optional lossy re-encoding before detection.")
    ap.add_argument("--quality", type=int, default=75, help="Codec quality (0..100) for --codec.")
    ap.add_argument("--square-mm", type=float, default=0.3)
    ap.add_argument("--squares-x", type=int, default=16)
    ap.add_argument("--squares-y", type=int, default=12)
    ap.add_argument("--marker-parity", type=int, default=0, help="Marker parity pattern (0 matches Pycaso example board).")
    ap.add_argument("--out-json", type=Path, default=Path("validation/pycaso_real_z_sweep_rayfield3d.json"))
    ap.add_argument("--pycaso-degree", type=int, default=3, help="Degree for Pycaso/Soloff direct polynomial mapping baseline.")
    ap.add_argument("--pycaso-ridge", type=float, default=0.0, help="Ridge regularization for Pycaso/Soloff direct polynomial fit.")
    args = ap.parse_args()

    left_dir = args.pycaso_images_root / f"left_{args.pair}"
    right_dir = args.pycaso_images_root / f"right_{args.pair}"
    pairs = _load_pairs(left_dir=left_dir, right_dir=right_dir)
    if not pairs:
        raise SystemExit(f"No paired images found in {left_dir} and {right_dir}")
    if args.max_frames and args.max_frames > 0:
        pairs = pairs[: int(args.max_frames)]

    import cv2  # type: ignore
    import cv2.aruco as aruco  # type: ignore

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    id_to_rc = _board_id_to_square_rc(squares_x=int(args.squares_x), squares_y=int(args.squares_y), parity=int(args.marker_parity))
    sq = float(args.square_mm)
    id_to_center_mm = {
        mid: np.array([(c + 0.5) * sq, (r + 0.5) * sq], dtype=np.float64) for mid, (r, c) in id_to_rc.items()
    }
    chess_xy_mm = _chess_corners_xy_mm(squares_x=int(args.squares_x), squares_y=int(args.squares_y), square_size_mm=float(args.square_mm))
    P_board_mm = np.concatenate([chess_xy_mm, np.zeros((chess_xy_mm.shape[0], 1), dtype=np.float64)], axis=1)

    rng = np.random.default_rng(int(args.subsample_seed))

    stereo_data: list[StereoPairData] = []
    for z_mm, pL, pR in pairs:
        imgL = cv2.imread(str(pL), cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(str(pR), cv2.IMREAD_GRAYSCALE)
        if imgL is None or imgR is None:
            continue
        imgL = _apply_lossy_codec(cv2=cv2, img_gray=imgL, codec=args.codec, quality=args.quality)
        imgR = _apply_lossy_codec(cv2=cv2, img_gray=imgR, codec=args.codec, quality=args.quality)

        cL = _detect_marker_centers(cv2=cv2, aruco=aruco, detector=detector, img_gray=imgL, id_to_center_mm=id_to_center_mm)
        cR = _detect_marker_centers(cv2=cv2, aruco=aruco, detector=detector, img_gray=imgR, id_to_center_mm=id_to_center_mm)
        if cL is None or cR is None:
            continue

        uvL = _predict_chess_corners(centers=cL, chess_xy_mm=chess_xy_mm, tps_lam=args.tps_lam, tps_huber=args.tps_huber, tps_iters=args.tps_iters)
        uvR = _predict_chess_corners(centers=cR, chess_xy_mm=chess_xy_mm, tps_lam=args.tps_lam, tps_huber=args.tps_huber, tps_iters=args.tps_iters)

        if int(args.max_points_per_frame) and int(args.max_points_per_frame) > 0 and uvL.shape[0] > int(args.max_points_per_frame):
            idx = rng.choice(uvL.shape[0], size=int(args.max_points_per_frame), replace=False)
            idx = np.sort(idx)
            uvL = uvL[idx]
            uvR = uvR[idx]
            P = P_board_mm[idx]
        else:
            P = P_board_mm

        stereo_data.append(
            StereoPairData(
                z_mm=float(z_mm),
                uvL_px=np.asarray(uvL, dtype=np.float64).reshape(-1, 2),
                uvR_px=np.asarray(uvR, dtype=np.float64).reshape(-1, 2),
                P_board_mm=np.asarray(P, dtype=np.float64).reshape(-1, 3),
                centers_L=cL,
                centers_R=cR,
            )
        )

    if not stereo_data:
        raise SystemExit("No valid stereo frames (marker detection failed on all pairs).")

    img0 = cv2.imread(str(pairs[0][1]), cv2.IMREAD_GRAYSCALE)
    assert img0 is not None
    h, w = int(img0.shape[0]), int(img0.shape[1])
    image_size = (w, h)

    # K0 estimate per camera from marker-center homographies.
    Hs_L: list[np.ndarray] = []
    Hs_R: list[np.ndarray] = []
    for fr in stereo_data:
        HL, _ = cv2.findHomography(fr.centers_L.obj_xy_mm, fr.centers_L.uv_px, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        HR, _ = cv2.findHomography(fr.centers_R.obj_xy_mm, fr.centers_R.uv_px, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if HL is not None:
            Hs_L.append(np.asarray(HL, dtype=np.float64))
        if HR is not None:
            Hs_R.append(np.asarray(HR, dtype=np.float64))
    K0_L = _estimate_K0_from_homographies(homographies=Hs_L, image_size=image_size)
    K0_R = _estimate_K0_from_homographies(homographies=Hs_R, image_size=image_size)

    # Pose init per frame (marker centers).
    rvecs_L0: dict[int, np.ndarray] = {}
    tvecs_L0: dict[int, np.ndarray] = {}
    rvecs_R0: dict[int, np.ndarray] = {}
    tvecs_R0: dict[int, np.ndarray] = {}
    frames: dict[int, StereoFrameObservations] = {}
    z_by_fid: dict[int, float] = {}
    for fid, fr in enumerate(stereo_data):
        poseL = _init_pose_from_homography(cv2, obj_xy_mm=fr.centers_L.obj_xy_mm, uv_px=fr.centers_L.uv_px, K0=K0_L)
        poseR = _init_pose_from_homography(cv2, obj_xy_mm=fr.centers_R.obj_xy_mm, uv_px=fr.centers_R.uv_px, K0=K0_R)
        if poseL is None or poseR is None:
            continue
        rvecs_L0[fid], tvecs_L0[fid] = poseL
        rvecs_R0[fid], tvecs_R0[fid] = poseR
        frames[fid] = StereoFrameObservations(uv_left_px=fr.uvL_px, uv_right_px=fr.uvR_px, P_board_mm=fr.P_board_mm)
        z_by_fid[fid] = float(fr.z_mm)

    common_fids = sorted(frames.keys())
    if len(common_fids) < 3:
        raise SystemExit("Not enough frames with pose initialization (need >= 3).")

    R_RL0, t_RL0 = _rig_from_poses(fids=common_fids, rvecs_L=rvecs_L0, tvecs_L=tvecs_L0, rvecs_R=rvecs_R0, tvecs_R=tvecs_R0)
    rig_rvec0, _ = cv2.Rodrigues(R_RL0)

    # Coeff init from a pinhole prior.
    f0_px = 1.5 * float(max(w, h))
    uv_all_L = np.concatenate([frames[fid].uv_left_px for fid in common_fids], axis=0)
    uv_all_R = np.concatenate([frames[fid].uv_right_px for fid in common_fids], axis=0)
    cLx0, cLy0 = _init_coeffs_pinhole_prior(uv_all=uv_all_L, nmax=int(args.nmax), image_size=image_size, f0_px=f0_px)
    cRx0, cRy0 = _init_coeffs_pinhole_prior(uv_all=uv_all_R, nmax=int(args.nmax), image_size=image_size, f0_px=f0_px)

    # Stereo BA.
    res = fit_central_stereo_rayfield_ba(
        frames=frames,
        image_width_px=w,
        image_height_px=h,
        nmax=int(args.nmax),
        rvecs0=rvecs_L0,
        tvecs0=tvecs_L0,
        rig_rvec0=rig_rvec0.reshape(3),
        rig_tvec0=t_RL0.reshape(3),
        coeffs0_left_x=cLx0,
        coeffs0_left_y=cLy0,
        coeffs0_right_x=cRx0,
        coeffs0_right_y=cRy0,
        lam_coeff=1e-3,
        lam_center=1e-1,
        lam_jacobian=1.0,
        loss="huber",
        f_scale_mm=1.0,
        max_nfev=int(args.max_nfev),
    )

    model = StereoCentralRayFieldModel.from_coeffs(
        image_width_px=w,
        image_height_px=h,
        nmax=res.nmax,
        u0_px=res.u0_px,
        v0_px=res.v0_px,
        radius_px=res.radius_px,
        coeffs_left_x=res.coeffs_left_x,
        coeffs_left_y=res.coeffs_left_y,
        coeffs_right_x=res.coeffs_right_x,
        coeffs_right_y=res.coeffs_right_y,
        R_RL=cv2.Rodrigues(res.rig_rvec)[0],
        t_RL=res.rig_tvec,
    )

    # Evaluate Z sweep: compare triangulated mean Z (per frame) to filename Z.
    z_gt: list[float] = []
    z_est: list[float] = []
    skew_all: list[np.ndarray] = []
    for fid in common_fids:
        fr = frames[fid]
        XYZ, skew = model.triangulate(fr.uv_left_px, fr.uv_right_px)
        z_gt.append(float(z_by_fid[fid]))
        z_est.append(float(np.nanmean(XYZ[:, 2])))
        skew_all.append(skew.astype(np.float64).reshape(-1))
    z_gt_v = np.asarray(z_gt, dtype=np.float64)
    z_est_v = np.asarray(z_est, dtype=np.float64)

    # Affine fit: z_est ~= a*z_gt + b.
    A = np.stack([z_gt_v, np.ones_like(z_gt_v)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, z_est_v, rcond=None)
    z_fit = a * z_gt_v + b
    err = z_est_v - z_fit

    skew = np.concatenate(skew_all, axis=0)

    # Pycaso baseline: direct polynomial mapping (uL,vL,uR,vR) -> (X,Y,Z).
    uvL_all = np.concatenate([frames[fid].uv_left_px for fid in common_fids], axis=0)
    uvR_all = np.concatenate([frames[fid].uv_right_px for fid in common_fids], axis=0)
    XYZ_all_parts: list[np.ndarray] = []
    for fid in common_fids:
        P = np.asarray(frames[fid].P_board_mm, dtype=np.float64).reshape(-1, 3).copy()
        P[:, 2] = float(z_by_fid[fid])
        XYZ_all_parts.append(P)
    XYZ_all = np.concatenate(XYZ_all_parts, axis=0)

    py = SoloffPolynomialModel.fit(
        uv_left_px=uvL_all,
        uv_right_px=uvR_all,
        XYZ_mm=XYZ_all,
        degree=int(args.pycaso_degree),
        ridge=float(args.pycaso_ridge),
    )

    z_py_gt: list[float] = []
    z_py_est: list[float] = []
    for fid in common_fids:
        fr = frames[fid]
        XYZ_pred = py.predict(fr.uv_left_px, fr.uv_right_px)
        z_py_gt.append(float(z_by_fid[fid]))
        z_py_est.append(float(np.nanmean(XYZ_pred[:, 2])))
    z_py_gt_v = np.asarray(z_py_gt, dtype=np.float64)
    z_py_est_v = np.asarray(z_py_est, dtype=np.float64)
    py_err = z_py_est_v - z_py_gt_v

    out: dict[str, Any] = {
        "dataset": {
            "left_dir": str(left_dir),
            "right_dir": str(right_dir),
            "n_pairs_used": int(len(common_fids)),
            "z_mm_min": float(np.min(z_gt_v)),
            "z_mm_max": float(np.max(z_gt_v)),
            "z_mm_mean": float(np.mean(z_gt_v)),
            "codec": str(args.codec),
            "quality": int(args.quality),
        },
        "board": {
            "squares_x": int(args.squares_x),
            "squares_y": int(args.squares_y),
            "square_size_mm": float(args.square_mm),
            "marker_parity": int(args.marker_parity),
            "n_corners": int(P_board_mm.shape[0]),
        },
        "pycaso_direct_poly": {
            "degree": int(py.degree),
            "n_monomials": int(py.powers.shape[0]),
            "ridge": float(args.pycaso_ridge),
            "z_rms_mm": _rms(py_err),
            "z_rms_percent": float(100.0 * _rms(py_err) / (float(np.mean(z_py_gt_v)) + 1e-12)),
        },
        "rayfield3d": {
            "nmax": int(res.nmax),
            "opt_nfev": float(res.diagnostics.get("opt_nfev", float("nan"))),
            "opt_cost": float(res.diagnostics.get("opt_cost", float("nan"))),
            "baseline_mm": float(np.linalg.norm(model.C_R_in_L_mm)),
            "z_affine_fit": {"a": float(a), "b": float(b)},
            "z_rms_mm_after_affine": _rms(err),
            "z_rms_percent_after_affine": float(100.0 * _rms(err) / (float(np.mean(z_gt_v)) + 1e-12)),
            "skew_p95_mm": _quantile(skew, 0.95),
        },
        "per_frame": [{"z_mm": float(z_gt_v[i]), "z_est_mm": float(z_est_v[i])} for i in range(z_gt_v.size)],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[pycaso-real] wrote {args.out_json}")
    print(
        f"[pycaso-real] codec={args.codec} q={int(args.quality)} "
        f"rayfield3d z_rms={out['rayfield3d']['z_rms_mm_after_affine']:.4f} mm "
        f"({out['rayfield3d']['z_rms_percent_after_affine']:.2f}% of mean Z), "
        f"skew_p95={out['rayfield3d']['skew_p95_mm']:.3f} mm, "
        f"baseline={out['rayfield3d']['baseline_mm']:.1f} mm"
    )
    print(
        f"[pycaso-real] codec={args.codec} q={int(args.quality)} "
        f"pycaso_direct_poly z_rms={out['pycaso_direct_poly']['z_rms_mm']:.4f} mm "
        f"({out['pycaso_direct_poly']['z_rms_percent']:.2f}% of mean Z), "
        f"degree={out['pycaso_direct_poly']['degree']}, n_monomials={out['pycaso_direct_poly']['n_monomials']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
