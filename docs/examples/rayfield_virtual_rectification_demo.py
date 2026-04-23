"""
Virtual rectification demo for ray-field stereo.

The goal is to show, end to end, how to:

1. obtain or load a central 3D ray-field stereo model,
2. build virtual rectification maps,
3. rectify a stereo pair with cv2.remap,
4. run a standard 1D dense matcher on the rectified pair,
5. check whether ChArUco corner pairs become closer to horizontal epipolar lines.

Typical usage
-------------
1) Export a small model once (if you do not already have one):

   PYTHONPATH=src .venv/bin/python paper/experiments/calibrate_central_rayfield3d_from_images.py \
     dataset/v0_png --split train --scene scene_0000 --max-frames 5 \
     --method2d rayfield_tps_robust --nmax 10 --lam-coeff 1e-3 --outer-iters 3 \
     --out paper/tables/rayfield3d_scene0000.json \
     --export-model models/scene0000_rayfield3d

2) Rectify a pair and save the outputs:

   PYTHONPATH=src .venv/bin/python docs/examples/rayfield_virtual_rectification_demo.py \
     dataset/v0_png --split train --scene scene_0000 --frame-id 0 \
     --model models/scene0000_rayfield3d \
     --out docs/assets/rayfield_virtual_rectify_demo

If --model is omitted, this script can auto-export a small model first using the
same calibration script (stored under --out/model by default).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereocomplex.api import load_stereo_central_rayfield
from stereocomplex.ray3d.rayfield_rectify import RectifyParams, build_virtual_rectify_maps, rectify_pair

@dataclass(frozen=True)
class RayModelAdapter:
    """
    Small adapter so `StereoCentralRayFieldModel` can be used by the rectifier.

    The rectifier expects an object exposing:
    - `dir(u, v) -> unit direction`
    - `width`, `height`
    """

    model: Any
    width: int
    height: int

    def dir(self, u: float, v: float) -> np.ndarray:
        d = self.model.ray_directions_cam(np.asarray([u]), np.asarray([v]))[0]
        return np.asarray(d, dtype=np.float64)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames(scene_dir: Path) -> list[dict[str, Any]]:
    frames_path = scene_dir / "frames.jsonl"
    frames: list[dict[str, Any]] = []
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))
    return frames


def summarize(vals: list[float] | np.ndarray) -> dict[str, float]:
    if len(vals) == 0:
        return {"n": 0, "rms": float("nan"), "p50": float("nan"), "p95": float("nan"), "max": float("nan")}
    v = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(v.size),
        "rms": float(np.sqrt(np.mean(v * v))),
        "p50": float(np.quantile(v, 0.50)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }


def build_charuco_from_meta(meta: dict[str, Any]):
    import cv2.aruco as aruco  # type: ignore

    board_meta = meta["board"]
    dict_name = str(board_meta.get("aruco_dictionary", "DICT_4X4_1000"))
    dict_id = getattr(aruco, dict_name, None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco_dictionary: {dict_name}")
    dictionary = aruco.getPredefinedDictionary(dict_id)

    squares_x = int(board_meta["squares_x"])
    squares_y = int(board_meta["squares_y"])
    square_size = float(board_meta["square_size_mm"])
    marker_size = float(board_meta["marker_size_mm"])

    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((squares_x, squares_y), square_size, marker_size, dictionary)
    else:  # pragma: no cover
        board = aruco.CharucoBoard_create(squares_x, squares_y, square_size, marker_size, dictionary)

    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5
    detector_params.cornerRefinementMaxIterations = 50
    detector_params.cornerRefinementMinAccuracy = 1e-3

    charuco_detector = None
    if hasattr(aruco, "CharucoDetector"):
        charuco_detector = aruco.CharucoDetector(board)
        if hasattr(charuco_detector, "setDetectorParameters"):
            charuco_detector.setDetectorParameters(detector_params)

    aruco_detector = None
    if charuco_detector is None and hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)

    return dictionary, board, detector_params, aruco_detector, charuco_detector


def detect_charuco(
    dictionary,
    board,
    detector_params,
    aruco_detector,
    charuco_detector,
    img_gray: np.ndarray,
) -> dict[int, np.ndarray] | None:
    import cv2.aruco as aruco  # type: ignore

    if charuco_detector is not None:
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img_gray)
    else:
        if aruco_detector is not None:
            marker_corners, marker_ids, _rej = aruco_detector.detectMarkers(img_gray)
        else:  # pragma: no cover
            marker_corners, marker_ids, _rej = aruco.detectMarkers(img_gray, dictionary, parameters=detector_params)

        charuco_corners, charuco_ids = None, None
        if hasattr(aruco, "interpolateCornersCharuco") and marker_ids is not None and len(marker_ids) > 0:
            ret = aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, board)
            if ret is not None:
                if len(ret) == 3:
                    charuco_corners, charuco_ids, _ = ret
                elif len(ret) == 4:  # pragma: no cover
                    _, charuco_corners, charuco_ids, _ = ret

    if charuco_ids is None or charuco_corners is None or len(charuco_ids) == 0:
        return None

    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    charuco_xy = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) - 0.5
    return {int(i): charuco_xy[k] for k, i in enumerate(charuco_ids.tolist())}


def load_scene_images(scene_dir: Path, frame_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    frames = load_frames(scene_dir)
    frame = None
    for fr in frames:
        if int(fr["frame_id"]) == int(frame_id):
            frame = fr
            break
    if frame is None:
        raise ValueError(f"frame_id={frame_id} not found in {scene_dir / 'frames.jsonl'}")

    imgL = cv2.imread(str(scene_dir / "left" / str(frame["left"])), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(str(scene_dir / "right" / str(frame["right"])), cv2.IMREAD_GRAYSCALE)
    if imgL is None or imgR is None:
        raise FileNotFoundError("could not load left/right images")
    return imgL, imgR, frame


def vertical_disparity_stats(detL: dict[int, np.ndarray] | None, detR: dict[int, np.ndarray] | None) -> dict[str, float] | None:
    if not detL or not detR:
        return None
    common = sorted(set(detL).intersection(detR))
    if len(common) < 6:
        return None
    dv = np.asarray([abs(float(detL[c][1] - detR[c][1])) for c in common], dtype=np.float64)
    return summarize(dv)


def save_disp_vis(path: Path, disp: np.ndarray) -> None:
    valid = disp > 0.0
    if not np.any(valid):
        vis = np.zeros((*disp.shape, 3), dtype=np.uint8)
        cv2.imwrite(str(path), vis)
        return
    vals = disp[valid]
    lo = float(np.quantile(vals, 0.02))
    hi = float(np.quantile(vals, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals) + 1e-6)
    scaled = np.clip((disp - lo) / (hi - lo + 1e-12), 0.0, 1.0)
    scaled[~valid] = 0.0
    vis = cv2.applyColorMap((scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.imwrite(str(path), vis)


def ensure_model(args, out_dir: Path) -> Path:
    if args.model is not None:
        model_dir = Path(args.model)
        if not (model_dir / "model.json").exists():
            raise FileNotFoundError(f"{model_dir} does not contain model.json")
        return model_dir

    model_dir = Path(args.export_model) if args.export_model is not None else (out_dir / "model")
    if (model_dir / "model.json").exists():
        return model_dir

    cal_script = ROOT / "paper" / "experiments" / "calibrate_central_rayfield3d_from_images.py"
    report_json = out_dir / "rayfield3d_calibration.json"
    cmd = [
        sys.executable,
        str(cal_script),
        str(args.dataset_root),
        "--split",
        str(args.split),
        "--scene",
        str(args.scene),
        "--max-frames",
        str(args.max_frames),
        "--method2d",
        str(args.method2d),
        "--nmax",
        str(args.nmax),
        "--lam-coeff",
        str(args.lam_coeff),
        "--outer-iters",
        str(args.outer_iters),
        "--out",
        str(report_json),
        "--export-model",
        str(model_dir),
    ]
    if args.method2d == "rayfield_tps_robust":
        cmd.extend(
            [
                "--tps-lam",
                str(args.tps_lam),
                "--tps-huber",
                str(args.tps_huber),
                "--tps-iters",
                str(args.tps_iters),
            ]
        )
    if args.max_points_per_frame > 0:
        cmd.extend(["--max-points-per-frame", str(args.max_points_per_frame)])

    env = os.environ.copy()
    src_path = str(SRC)
    env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    print("Auto-calibrating a small exported model...")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT), env=env)
    return model_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Virtual rectification demo for a ray-field stereo rig.")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--frame-id", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("docs/assets/rayfield_virtual_rectify_demo"))
    ap.add_argument("--model", type=Path, default=None, help="Existing exported model directory (model.json + weights.npz).")
    ap.add_argument(
        "--export-model",
        type=Path,
        default=None,
        help="If --model is omitted, auto-export a small model here (defaults to <out>/model).",
    )
    ap.add_argument("--max-frames", type=int, default=5, help="Frames used when auto-calibrating a model.")
    ap.add_argument("--method2d", default="rayfield_tps_robust", choices=["raw", "homography_only", "rayfield_tps_robust"])
    ap.add_argument("--max-points-per-frame", type=int, default=0)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--lam-coeff", type=float, default=1e-3)
    ap.add_argument("--outer-iters", type=int, default=3)
    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=1.0)
    ap.add_argument("--tps-iters", type=int, default=3)
    ap.add_argument("--rect-fx", type=float, default=None)
    ap.add_argument("--rect-fy", type=float, default=None)
    ap.add_argument("--rect-cx", type=float, default=None)
    ap.add_argument("--rect-cy", type=float, default=None)
    ap.add_argument("--sgbm-num-disparities", type=int, default=128)
    ap.add_argument("--sgbm-block-size", type=int, default=5)
    args = ap.parse_args()

    scene_dir = Path(args.dataset_root) / str(args.split) / str(args.scene)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = ensure_model(args, out_dir)
    model = load_stereo_central_rayfield(model_dir)

    imgL, imgR, frame = load_scene_images(scene_dir, args.frame_id)

    rect_params = RectifyParams(
        width=int(model.image_width_px),
        height=int(model.image_height_px),
        fx=args.rect_fx,
        fy=args.rect_fy,
        cx=args.rect_cx,
        cy=args.rect_cy,
    )
    rayL = RayModelAdapter(model.left, model.image_width_px, model.image_height_px)
    rayR = RayModelAdapter(model.right, model.image_width_px, model.image_height_px)
    mapx_L, mapy_L, mapx_R, mapy_R, R_rect = build_virtual_rectify_maps(rayL, rayR, model.R_RL, model.t_RL, rect_params)
    I_L_rect, I_R_rect = rectify_pair((imgL, imgR), (mapx_L, mapy_L, mapx_R, mapy_R), rect_params)

    cv2.imwrite(str(out_dir / "left_raw.png"), imgL)
    cv2.imwrite(str(out_dir / "right_raw.png"), imgR)
    cv2.imwrite(str(out_dir / "left_rectified.png"), I_L_rect)
    cv2.imwrite(str(out_dir / "right_rectified.png"), I_R_rect)
    np.savez_compressed(
        out_dir / "rectify_maps.npz",
        mapx_L=mapx_L,
        mapy_L=mapy_L,
        mapx_R=mapx_R,
        mapy_R=mapy_R,
        R_rect=R_rect,
    )

    # Dense matcher demo: standard scanline SGBM on rectified images.
    block_size = int(args.sgbm_block_size)
    num_disp = int(args.sgbm_num_disparities)
    if num_disp % 16 != 0:
        num_disp = 16 * max(1, round(num_disp / 16))
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=1,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp = sgbm.compute(I_L_rect, I_R_rect).astype(np.float32) / 16.0
    save_disp_vis(out_dir / "disparity.png", disp)
    np.save(out_dir / "disparity.npy", disp)

    # Optional ChArUco-based sanity check: compare raw vs rectified vertical disparity.
    meta = load_json(scene_dir / "meta.json")
    charuco_available = True
    try:
        dictionary, board, detector_params, aruco_detector, charuco_detector = build_charuco_from_meta(meta)
    except Exception as exc:  # pragma: no cover
        charuco_available = False
        dictionary = board = detector_params = aruco_detector = charuco_detector = None
        print(f"[warn] ChArUco not available: {exc}")

    raw_stats = rect_stats = None
    if charuco_available:
        raw_L = detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, imgL)
        raw_R = detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, imgR)
        rect_L = detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, I_L_rect)
        rect_R = detect_charuco(dictionary, board, detector_params, aruco_detector, charuco_detector, I_R_rect)
        raw_stats = vertical_disparity_stats(raw_L, raw_R)
        rect_stats = vertical_disparity_stats(rect_L, rect_R)

    out = {
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "split": str(args.split),
        "scene": str(args.scene),
        "frame_id": int(args.frame_id),
        "frame_files": {"left": frame["left"], "right": frame["right"]},
        "model_dir": str(Path(model_dir).resolve()),
        "rectified_image_size": [int(model.image_width_px), int(model.image_height_px)],
        "rectification": {
            "valid_left_fraction": float(
                np.mean(
                    (mapx_L >= 0.0)
                    & (mapx_L < model.image_width_px)
                    & (mapy_L >= 0.0)
                    & (mapy_L < model.image_height_px)
                )
            ),
            "valid_right_fraction": float(
                np.mean(
                    (mapx_R >= 0.0)
                    & (mapx_R < model.image_width_px)
                    & (mapy_R >= 0.0)
                    & (mapy_R < model.image_height_px)
                )
            ),
        },
        "dense_matcher": {
            "algorithm": "StereoSGBM",
            "num_disparities": int(num_disp),
            "block_size": int(block_size),
            "valid_fraction": float(np.mean(disp > 0.0)),
        },
        "charuco_vertical_disparity_px": {
            "raw": raw_stats,
            "rectified": rect_stats,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nSaved outputs to: {out_dir.resolve()}")
    print("Use the rectified images directly with cv2.StereoSGBM / BM / Census.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
