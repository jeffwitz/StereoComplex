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
    detections: CharucoDetections | None = None,
    marker_ids: np.ndarray | None = None,
    marker_corners: list[np.ndarray] | None = None,
    charuco_ids: np.ndarray | None = None,
    charuco_xy: np.ndarray | None = None,
    tps_lam: float = 10.0,
    huber_c: float = 3.0,
    iters: int = 3,
) -> np.ndarray:
    """Refine ChArUco corners using geometric priors on the board plane.

    The function accepts either a bundled :class:`CharucoDetections` object or
    the four raw detection arrays.  ``method="raw"`` returns the ChArUco
    detector coordinates unchanged; ``method="rayfield_tps_robust"`` fits a
    robust 2-D board-plane warp from marker corners to image pixels and
    evaluates it at the requested ChArUco corner IDs.

    Parameters
    ----------
    method : {"raw", "rayfield_tps_robust"}
        Refinement strategy.
    board : CharucoBoardSpec or OpenCV CharucoBoard
        Board geometry. ``CharucoBoardSpec`` is converted with
        ``build_charuco_board``; an OpenCV board must provide ``getIds``,
        ``getObjPoints`` and ``getChessboardCorners``.
    detections : CharucoDetections, optional
        Bundled marker and ChArUco detections for one image.  If provided, the
        four raw detection arguments below are ignored.
    marker_ids : ndarray, shape (M,), optional
        ArUco marker IDs detected in the image.
    marker_corners : list of ndarray, optional
        Marker corner coordinates in pixels, one ``(4, 2)`` array per marker.
    charuco_ids : ndarray, shape (K,), optional
        ChArUco corner IDs to refine.
    charuco_xy : ndarray, shape (K, 2), optional
        Initial ChArUco corner coordinates in pixels.
    tps_lam : float
        Thin-plate-spline smoothing parameter for ``rayfield_tps_robust``.
    huber_c : float
        Huber threshold in pixels for robust TPS fitting.
    iters : int
        Number of robust reweighting iterations.

    Returns
    -------
    ndarray, shape (K, 2)
        Refined corner positions in pixels, in the same order as
        ``charuco_ids``.
    """
    if detections is not None:
        marker_ids = detections.marker_ids
        marker_corners = detections.marker_corners
        charuco_ids = detections.charuco_ids
        charuco_xy = detections.charuco_xy
    if marker_ids is None or marker_corners is None or charuco_ids is None or charuco_xy is None:
        raise TypeError(
    "provide either detections=... or "
    "marker_ids/marker_corners/charuco_ids/charuco_xy"
)

    if not (
        hasattr(board, "getIds")
        and hasattr(board, "getObjPoints")
        and hasattr(board, "getChessboardCorners")
    ):
        try:
            from stereocomplex.api.calibration import (
                CharucoBoardSpec,
                build_charuco_board,
            )

            if isinstance(board, CharucoBoardSpec):
                board = build_charuco_board(board)
        except Exception:
            pass

    charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    charuco_xy = np.asarray(charuco_xy, dtype=np.float64).reshape(-1, 2)
    if charuco_ids.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if method == "raw":
        return charuco_xy.copy()

    if method != "rayfield_tps_robust":
        raise ValueError(f"unknown method: {method}")

    from stereocomplex.core.rayfield2d import predict_points_rayfield_tps_robust

    marker_ids = np.asarray(marker_ids, dtype=np.int32).reshape(-1)
    board_ids = np.asarray(board.getIds(), dtype=np.int32).reshape(-1)
    board_obj = board.getObjPoints()
    id_to_obj2 = {
    int(i): np.asarray(p, dtype=np.float64)[:, :2]
    for i, p in zip(board_ids.tolist(), board_obj, strict=True)
}

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
    pred = predict_points_rayfield_tps_robust(
        obj_xy,
        img_uv,
        target_xy,
        lam=float(tps_lam),
        huber_c=float(huber_c),
        iters=int(iters),
    )
    return np.asarray(pred, dtype=np.float64).reshape(-1, 2)
