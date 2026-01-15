from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CharucoSpec:
    squares_x: int
    squares_y: int
    square_size_mm: float
    marker_size_mm: float
    aruco_dictionary: str = "DICT_4X4_1000"
    pixels_per_square: int = 80

    @property
    def size_px(self) -> tuple[int, int]:
        return (self.squares_x * self.pixels_per_square, self.squares_y * self.pixels_per_square)


def generate_charuco_texture(spec: CharucoSpec) -> np.ndarray:
    """
    Returns a uint8 grayscale image of the ChArUco board.

    OpenCV dependency is optional at import time; this function will raise if cv2.aruco is missing.
    """
    try:
        import cv2  # type: ignore
        import cv2.aruco as aruco  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ChArUco generation requires opencv-contrib-python (cv2.aruco).") from e

    dict_id = getattr(aruco, spec.aruco_dictionary, None)
    if dict_id is None:
        raise ValueError(f"Unknown aruco dictionary: {spec.aruco_dictionary}")

    dictionary = aruco.getPredefinedDictionary(dict_id)

    # OpenCV API differs across versions: CharucoBoard vs CharucoBoard_create.
    if hasattr(aruco, "CharucoBoard"):
        board = aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_size_mm, spec.marker_size_mm, dictionary)
    elif hasattr(aruco, "CharucoBoard_create"):  # pragma: no cover
        board = aruco.CharucoBoard_create(
            spec.squares_x, spec.squares_y, spec.square_size_mm, spec.marker_size_mm, dictionary
        )
    else:  # pragma: no cover
        raise RuntimeError("cv2.aruco does not expose CharucoBoard APIs in this build.")

    w_px, h_px = spec.size_px
    if hasattr(board, "generateImage"):
        img = board.generateImage((w_px, h_px))
    else:  # pragma: no cover
        img = board.draw((w_px, h_px))

    if img.ndim != 2:
        raise RuntimeError("Expected grayscale charuco image.")
    return img.astype(np.uint8, copy=False)

