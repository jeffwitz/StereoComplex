from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from stereocomplex.api.stereo_reconstruction import StereoCentralRayFieldModel


def _to_float_matrix(x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(shape)
    if not np.all(np.isfinite(x)):
        raise ValueError("non-finite values")
    return x


def save_stereo_central_rayfield(model_dir: Path, model: StereoCentralRayFieldModel) -> Path:
    """
    Save a stereo central ray-field model into a directory:

      model.json + weights.npz

    The JSON is small and ML-friendly (pure metadata + paths). The NPZ stores arrays.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "weights.npz"
    np.savez_compressed(
        weights_path,
        coeffs_left_x=np.asarray(model.left.coeffs_x, dtype=np.float32),
        coeffs_left_y=np.asarray(model.left.coeffs_y, dtype=np.float32),
        coeffs_right_x=np.asarray(model.right.coeffs_x, dtype=np.float32),
        coeffs_right_y=np.asarray(model.right.coeffs_y, dtype=np.float32),
        R_RL=np.asarray(model.R_RL, dtype=np.float64),
        t_RL=np.asarray(model.t_RL, dtype=np.float64),
    )

    meta: dict[str, Any] = {
        "schema_version": "stereocomplex.model.stereo_central_rayfield.v0",
        "image": {"width_px": int(model.image_width_px), "height_px": int(model.image_height_px)},
        "disk": {"u0_px": float(model.left.u0_px), "v0_px": float(model.left.v0_px), "radius_px": float(model.left.radius_px)},
        "zernike": {"nmax": int(model.left.nmax)},
        "rig": {
            "R_RL": np.asarray(model.R_RL, dtype=np.float64).tolist(),
            "t_RL": np.asarray(model.t_RL, dtype=np.float64).reshape(3).tolist(),
            "C_R_in_L_mm": np.asarray(model.C_R_in_L_mm, dtype=np.float64).reshape(3).tolist(),
        },
        "weights": {
            "format": "npz",
            "path": weights_path.name,
            "keys": {
                "coeffs_left_x": "coeffs_left_x",
                "coeffs_left_y": "coeffs_left_y",
                "coeffs_right_x": "coeffs_right_x",
                "coeffs_right_y": "coeffs_right_y",
                "R_RL": "R_RL",
                "t_RL": "t_RL",
            },
        },
    }

    json_path = model_dir / "model.json"
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return json_path


def load_stereo_central_rayfield(model_dir: Path) -> StereoCentralRayFieldModel:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if str(meta.get("schema_version")) != "stereocomplex.model.stereo_central_rayfield.v0":
        raise ValueError("unsupported model schema")

    image = meta["image"]
    disk = meta["disk"]
    zern = meta["zernike"]
    rig = meta["rig"]
    weights = meta["weights"]

    weights_path = model_dir / str(weights["path"])
    w = np.load(str(weights_path))
    k = weights["keys"]
    coeffs_left_x = np.asarray(w[str(k["coeffs_left_x"])], dtype=np.float64).reshape(-1)
    coeffs_left_y = np.asarray(w[str(k["coeffs_left_y"])], dtype=np.float64).reshape(-1)
    coeffs_right_x = np.asarray(w[str(k["coeffs_right_x"])], dtype=np.float64).reshape(-1)
    coeffs_right_y = np.asarray(w[str(k["coeffs_right_y"])], dtype=np.float64).reshape(-1)
    R_RL = _to_float_matrix(w[str(k["R_RL"])], (3, 3))
    t_RL = _to_float_matrix(w[str(k["t_RL"])], (3,))

    return StereoCentralRayFieldModel.from_coeffs(
        image_width_px=int(image["width_px"]),
        image_height_px=int(image["height_px"]),
        nmax=int(zern["nmax"]),
        u0_px=float(disk["u0_px"]),
        v0_px=float(disk["v0_px"]),
        radius_px=float(disk["radius_px"]),
        coeffs_left_x=coeffs_left_x,
        coeffs_left_y=coeffs_left_y,
        coeffs_right_x=coeffs_right_x,
        coeffs_right_y=coeffs_right_y,
        R_RL=R_RL,
        t_RL=t_RL,
    )

