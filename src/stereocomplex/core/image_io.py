from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_gray_u8(path: str | Path) -> np.ndarray:
    """
    Load an image as grayscale uint8.

    Primary backend is OpenCV (if installed). Pillow is used as a fallback.
    This makes dataset processing robust across image formats (png/webp/jpg)
    and OpenCV builds that may lack some codec support.
    """
    p = Path(path)
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
    except Exception:
        # Fall back to Pillow below.
        pass

    with Image.open(p) as im:
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.uint8)
    return arr

