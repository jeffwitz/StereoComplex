from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from stereocomplex.core.image_io import load_gray_u8


def _write_gray(path: Path, arr: np.ndarray) -> None:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")
    if path.suffix.lower() == ".webp":
        img.save(path, lossless=True)
    else:
        img.save(path)


def test_load_gray_u8_png_and_webp(tmp_path: Path) -> None:
    arr = (np.arange(64, dtype=np.uint8).reshape(8, 8) * 4) % 255

    p_png = tmp_path / "a.png"
    p_webp = tmp_path / "a.webp"
    _write_gray(p_png, arr)
    _write_gray(p_webp, arr)

    a = load_gray_u8(p_png)
    b = load_gray_u8(p_webp)

    assert a.shape == (8, 8)
    assert b.shape == (8, 8)
    assert a.dtype == np.uint8
    assert b.dtype == np.uint8

