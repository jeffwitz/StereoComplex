"""Dispatch ChArUco corner refinement methods."""

from __future__ import annotations

import numpy as np

from stereocomplex.eval.refiners.tensor import (
    refine_points_tensor_lines,
    refine_points_tensor_lsq,
    refine_points_tensor_noble,
    refine_points_tensor_symmetry,
)


def refine_detected_points(
    refine: str,
    cv2,
    img: np.ndarray,
    charuco_xy: np.ndarray,
    *,
    search_radius: int,
    tensor_sigma: float,
) -> np.ndarray:
    if refine == "none":
        return charuco_xy

    refiners = {
        "tensor": refine_points_tensor_symmetry,
        "lines": refine_points_tensor_lines,
        "lsq": refine_points_tensor_lsq,
        "noble": refine_points_tensor_noble,
    }
    refiner = refiners.get(refine)
    if refiner is None:
        raise ValueError("refine must be none|tensor|lines|lsq|noble")
    return refiner(
        cv2,
        img,
        charuco_xy,
        search_radius=float(search_radius),
        tensor_sigma=float(tensor_sigma),
    )
