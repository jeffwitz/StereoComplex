#!/usr/bin/env python3
"""Measure apparent ChArUco magnification versus axial position.

This is a compact diagnostic for the CMO paper.  For each frame and channel,
we fit an affine image map

    [u, v]^T = A [X, Y]^T + b

to the completed ChArUco corners stored in ``intermediate_state.npz``.  The
column norms of ``A`` are reported as the apparent lateral scales Mx and My
in px/mm.  A linear fit versus the nominal stage coordinate gives a concise
image-space check of how much lateral magnification changes across the
calibration stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path("docs/assets/pycaso_real_data")
STATE = ROOT / "intermediate_state.npz"
SUMMARY = ROOT / "summary.json"
OUT = ROOT / "magnification_vs_z.json"


def _load_nominal_z(n_frames: int) -> np.ndarray:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    z0, z1 = summary["dataset"]["z_range_mm"]
    return np.linspace(float(z0), float(z1), int(n_frames), dtype=np.float64)


def _fit_scales(obj_xy: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    design = np.column_stack([obj_xy, np.ones(obj_xy.shape[0], dtype=np.float64)])
    scales = []
    for frame_pixels in pixels:
        coeff, *_ = np.linalg.lstsq(design, frame_pixels, rcond=None)
        affine = coeff[:2, :].T
        scales.append(
            [
                float(np.linalg.norm(affine[:, 0])),
                float(np.linalg.norm(affine[:, 1])),
            ]
        )
    return np.asarray(scales, dtype=np.float64)


def _summarise_trace(z_mm: np.ndarray, values: np.ndarray) -> dict[str, float]:
    z0 = z_mm - float(np.mean(z_mm))
    slope, intercept = np.polyfit(z0, values, 1)
    fit = slope * z0 + intercept
    return {
        "mean_px_per_mm": float(np.mean(values)),
        "std_px_per_mm": float(np.std(values, ddof=1)),
        "range_relative_percent": float(100.0 * (np.max(values) - np.min(values)) / np.mean(values)),
        "slope_px_per_mm_per_mm": float(slope),
        "slope_relative_percent_per_mm": float(100.0 * slope / intercept),
        "total_linear_change_percent": float(100.0 * slope / intercept * np.ptp(z_mm)),
        "rms_linear_residual_percent": float(
            100.0 * np.sqrt(np.mean((values - fit) ** 2)) / intercept
        ),
    }


def main() -> int:
    data = np.load(STATE)
    obj_xy = np.asarray(data["obj_pts"], dtype=np.float64)[:, :2]
    z_nominal = _load_nominal_z(int(data["n_frames"]))
    z_fit = np.asarray(data["opt_t"], dtype=np.float64)[:, 2]

    output: dict[str, object] = {
        "definition": "Affine fit per frame/channel: [u,v]^T = A [X,Y]^T + b; Mx=||A[:,0]||, My=||A[:,1]||.",
        "nominal_z_mm": z_nominal.tolist(),
        "fitted_pose_z_mm": z_fit.tolist(),
        "nominal_z_span_mm": float(np.ptp(z_nominal)),
        "fitted_pose_z_span_mm": float(np.ptp(z_fit)),
        "channels": {},
    }

    for channel, key in (("left", "left_pixels"), ("right", "right_pixels")):
        scales = _fit_scales(obj_xy, np.asarray(data[key], dtype=np.float64))
        output["channels"][channel] = {
            "Mx_px_per_mm": scales[:, 0].tolist(),
            "My_px_per_mm": scales[:, 1].tolist(),
            "nominal_z_fit": {
                "Mx": _summarise_trace(z_nominal, scales[:, 0]),
                "My": _summarise_trace(z_nominal, scales[:, 1]),
            },
            "fitted_pose_z_fit": {
                "Mx": _summarise_trace(z_fit, scales[:, 0]),
                "My": _summarise_trace(z_fit, scales[:, 1]),
            },
        }

    nominal_summaries = []
    for channel in ("left", "right"):
        c = output["channels"][channel]  # type: ignore[index]
        nominal_summaries.append(c["nominal_z_fit"]["Mx"])  # type: ignore[index]
        nominal_summaries.append(c["nominal_z_fit"]["My"])  # type: ignore[index]
    output["aggregate_nominal_z"] = {
        "mean_scale_px_per_mm": float(
            np.mean([s["mean_px_per_mm"] for s in nominal_summaries])
        ),
        "min_total_change_percent": float(
            np.min([abs(s["total_linear_change_percent"]) for s in nominal_summaries])
        ),
        "max_total_change_percent": float(
            np.max([abs(s["total_linear_change_percent"]) for s in nominal_summaries])
        ),
        "min_relative_slope_percent_per_mm": float(
            np.min([abs(s["slope_relative_percent_per_mm"]) for s in nominal_summaries])
        ),
        "max_relative_slope_percent_per_mm": float(
            np.max([abs(s["slope_relative_percent_per_mm"]) for s in nominal_summaries])
        ),
    }

    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
