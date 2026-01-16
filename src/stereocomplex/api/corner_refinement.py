from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


RefineMethod = Literal["raw", "rayfield_tps_robust"]


@dataclass(frozen=True)
class CharucoDetections:
    """
    Minimal detections for one view.

    Coordinates are in the dataset pixel-center convention.
    """

    marker_ids: np.ndarray  # (M,)
    marker_corners: list[np.ndarray]  # list of (4,2)
    charuco_ids: np.ndarray  # (K,)
    charuco_xy: np.ndarray  # (K,2)


def refine_charuco_corners(
    *,
    method: RefineMethod,
    board: Any,
    marker_ids: np.ndarray,
    marker_corners: list[np.ndarray],
    charuco_ids: np.ndarray,
    charuco_xy: np.ndarray,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """
    Refine ChArUco corners using only geometric priors on the board plane.

    Inputs:
    - `board`: OpenCV CharucoBoard instance (used to get object coordinates).
    - marker detections: `marker_ids`, `marker_corners` (AruCo corners in pixels)
    - ChArUco corners to refine: `charuco_ids`, `charuco_xy` (pixels)

    Output:
    - refined corner positions (K,2) in pixels, same order as `charuco_ids`.
    """
    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    charuco_xy = np.asarray(charuco_xy, dtype=np.float64).reshape(-1, 2)
    if charuco_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if method == "raw":
        return charuco_xy.copy()

    if method != "rayfield_tps_robust":
        raise ValueError(f"unknown method: {method}")

    from stereocomplex.eval.charuco_detection import _predict_points_rayfield_tps_robust  # noqa: PLC0415

    marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
    board_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = board.getObjPoints()
    id_to_obj2 = {int(i): np.asarray(p, dtype=np.float64)[:, :2] for i, p in zip(board_ids.tolist(), board_obj, strict=True)}

    obj_pts: list[np.ndarray] = []
    img_pts: list[np.ndarray] = []
    for mid, mc in zip(marker_ids.tolist(), marker_corners, strict=True):
        o = id_to_obj2.get(int(mid))
        if o is None:
            continue
        mc = np.asarray(mc, dtype=np.float64).reshape(-1, 2)
        if mc.shape != (4, 2) or o.shape != (4, 2):
            continue
        obj_pts.append(o)
        img_pts.append(mc)
    if not obj_pts:
        return charuco_xy.copy()
    obj_xy = np.concatenate(obj_pts, axis=0)
    img_uv = np.concatenate(img_pts, axis=0)

    chess2 = np.asarray(board.getChessboardCorners(), dtype=np.float64)[:, :2]
    target_xy = chess2[charuco_ids]
    pred = _predict_points_rayfield_tps_robust(
        obj_xy,
        img_uv,
        target_xy,
        lam=float(tps_lam),
        huber_c=float(huber_c),
        iters=int(iters),
    )
    return np.asarray(pred, dtype=np.float64).reshape(-1, 2)

