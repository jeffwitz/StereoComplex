from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int
    squares_y: int
    square_size_mm: float

    @property
    def width_mm(self) -> float:
        return float(self.squares_x) * float(self.square_size_mm)

    @property
    def height_mm(self) -> float:
        return float(self.squares_y) * float(self.square_size_mm)


@dataclass(frozen=True)
class AberrationSpec:
    name: str
    distort: str
    distort_strength: float
    blur_fwhm_px: float
    blur_edge_factor: float
    noise_std: float


@dataclass(frozen=True)
class Scenario:
    name: str
    board: BoardSpec
    tz_mm: float
    pitch_um: float
    baseline_ratio: float
    framing_fill: float
    ab: AberrationSpec


def sensor_half_width_mm(*, width_px: int, pitch_um: float) -> float:
    return ((width_px - 1) / 2.0) * (pitch_um / 1000.0)


def compute_f_um_for_framing(
    *,
    width_px: int,
    pitch_um: float,
    tz_mm: float,
    board_w_mm: float,
    fill: float,
) -> float:
    """
    Choose a focal length so that the board width occupies ~2*fill*sensor_half_width at Z=tz.

    With a pinhole model:
      x_sensor = f * X/Z
    so for X = board_w/2 mapped to x_sensor = fill * sensor_half_width:
      f = fill * sensor_half_width * Z / (board_w/2) = 2*fill*sensor_half_width*Z/board_w
    """
    shw = sensor_half_width_mm(width_px=width_px, pitch_um=pitch_um)
    f_mm = 2.0 * float(fill) * float(shw) * float(tz_mm) / float(board_w_mm)
    return float(f_mm * 1000.0)


def run(cmd: list[str]) -> None:
    """
    Run a subprocess but keep the console quiet.

    On failure, re-raise with stdout/stderr for debugging.
    """
    res = subprocess.run(cmd, text=True, capture_output=True)
    if res.returncode != 0:
        msg = [
            "Command failed:",
            "  " + " ".join(cmd),
        ]
        if res.stdout:
            msg.append("--- stdout ---")
            msg.append(res.stdout)
        if res.stderr:
            msg.append("--- stderr ---")
            msg.append(res.stderr)
        raise RuntimeError("\n".join(msg))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_metric(report: dict[str, Any], method: str, path: tuple[str, ...]) -> float:
    cur: Any = report["methods"][method]
    for k in path:
        cur = cur[k]
    return float(cur)


def summarize_case(report: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for method in ("raw", "rayfield_tps_robust"):
        out[f"{method}.mono_L_rms_px"] = get_metric(report, method, ("mono", "left", "rms"))
        out[f"{method}.mono_R_rms_px"] = get_metric(report, method, ("mono", "right", "rms"))
        out[f"{method}.stereo_rms_px"] = get_metric(report, method, ("stereo", "rms"))
        out[f"{method}.baseline_delta_px_rms"] = get_metric(report, method, ("stereo", "baseline_delta_px", "rms"))
        out[f"{method}.tri_rms_mm"] = get_metric(report, method, ("stereo", "triangulation_error_mm", "rms"))
    return out


def summarize_rayfield3d(report: dict[str, Any]) -> dict[str, float]:
    """
    Metrics for the question: does post-hoc pinhole identification from ray-field 3D
    reconstruction improve the *pinhole parameters* vs a direct OpenCV pinhole calibration?
    """
    out: dict[str, float] = {}

    def k_percent(prefix: str, side: str, field: str) -> float:
        return float(report["pinhole_vs_gt"][prefix][side]["K_percent"][field])

    def disp_err(prefix: str, side: str) -> float:
        return float(report["pinhole_vs_gt"][prefix]["distortion_displacement_vs_gt"][side]["err_rms_px"])

    for prefix in ("opencv_pinhole_calib", "pinhole_from_rayfield3d"):
        out[f"{prefix}.K_fx_percent_left"] = k_percent(prefix, "left", "fx")
        out[f"{prefix}.K_fy_percent_left"] = k_percent(prefix, "left", "fy")
        out[f"{prefix}.K_fx_percent_right"] = k_percent(prefix, "right", "fx")
        out[f"{prefix}.K_fy_percent_right"] = k_percent(prefix, "right", "fy")
        out[f"{prefix}.dist_disp_err_rms_px_left"] = disp_err(prefix, "left")
        out[f"{prefix}.dist_disp_err_rms_px_right"] = disp_err(prefix, "right")
        out[f"{prefix}.mono_repr_rms_left_px"] = float(report[prefix]["reprojection_error_left_px"]["rms"])
        out[f"{prefix}.mono_repr_rms_right_px"] = float(report[prefix]["reprojection_error_right_px"]["rms"])

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Robustness sweep: vary board physical size, focal length (framing), and aberrations.",
    )
    ap.add_argument("--out-root", type=Path, default=Path("dataset/robustness_sweep"))
    ap.add_argument("--results-root", type=Path, default=Path("paper/tables/robustness_sweep"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--seeds", type=str, default="0,1", help="Comma-separated seeds.")
    ap.add_argument("--max-frames", type=int, default=0, help="Forwarded to compare script (0=all).")
    ap.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute all cases (default behavior is resume/skip existing outputs).",
    )
    ap.add_argument(
        "--run-rayfield3d",
        action="store_true",
        help="Also run ray-field 3D bundle adjustment + post-hoc pinhole identification per case.",
    )
    ap.add_argument("--rayfield3d-outer-iters", type=int, default=6)
    ap.add_argument("--rayfield3d-nmax", type=int, default=12)
    ap.add_argument(
        "--rayfield3d-max-points",
        type=int,
        default=200,
        help="Max ChArUco corners per frame for ray-field 3D bundle adjustment.",
    )
    args = ap.parse_args(argv)

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    out_root = args.out_root
    results_root = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)

    # Board sizes (physical) + working distances (chosen so f stays in a plausible range).
    boards = {
        "small": BoardSpec(squares_x=10, squares_y=8, square_size_mm=4.0),  # ~40 mm width
        "medium": BoardSpec(squares_x=24, squares_y=18, square_size_mm=10.0),  # ~240 mm width
        # Keep enough pixels per square for reliable ArUco detection at the chosen framing.
        "large": BoardSpec(squares_x=20, squares_y=15, square_size_mm=50.0),  # ~1000 mm width
    }
    tz_by_board = {"small": 300.0, "medium": 1500.0, "large": 5000.0}

    # Aberration levels: blur + geometric distortion + noise.
    aberrations = [
        AberrationSpec(
            name="low",
            distort="none",
            distort_strength=0.0,
            blur_fwhm_px=0.0,
            blur_edge_factor=1.0,
            noise_std=0.01,
        ),
        AberrationSpec(
            name="medium",
            distort="brown",
            distort_strength=0.3,
            blur_fwhm_px=0.6,
            blur_edge_factor=1.6,
            noise_std=0.02,
        ),
        AberrationSpec(
            name="high",
            distort="brown",
            distort_strength=0.6,
            blur_fwhm_px=1.2,
            blur_edge_factor=2.2,
            noise_std=0.03,
        ),
    ]

    pitch_um = 3.45
    baseline_ratio = 0.03
    fill = 0.75

    scenarios: list[Scenario] = []
    for bname, board in boards.items():
        for ab in aberrations:
            scenarios.append(
                Scenario(
                    name=f"{bname}_{ab.name}",
                    board=board,
                    tz_mm=float(tz_by_board[bname]),
                    pitch_um=pitch_um,
                    baseline_ratio=baseline_ratio,
                    framing_fill=fill,
                    ab=ab,
                )
            )

    # Run.
    all_rows: list[dict[str, Any]] = []
    py = str(Path(".venv/bin/python") if Path(".venv/bin/python").exists() else Path("python3"))
    for sc in scenarios:
        f_um = compute_f_um_for_framing(
            width_px=args.width,
            pitch_um=sc.pitch_um,
            tz_mm=sc.tz_mm,
            board_w_mm=sc.board.width_mm,
            fill=sc.framing_fill,
        )
        baseline_mm = float(sc.baseline_ratio * sc.tz_mm)

        for seed in seeds:
            case_name = f"{sc.name}_seed{seed}"
            ds_root = out_root / case_name
            report_path = results_root / f"{case_name}.json"
            ds_root.mkdir(parents=True, exist_ok=True)

            status = "ok"
            error: str | None = None
            if not ((not args.rerun) and report_path.exists()):
                # 1) Generate dataset (lossless WEBP, Lanczos texture sampling).
                gen_cmd = [
                    py,
                    "-m",
                    "stereocomplex.cli",
                    "generate-cpu-dataset",
                    "--out",
                    str(ds_root),
                    "--scenes",
                    "1",
                    "--frames-per-scene",
                    str(args.frames),
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--pattern",
                    "charuco",
                    "--tex-interp",
                    "lanczos4",
                    "--distort",
                    str(sc.ab.distort),
                    "--distort-strength",
                    str(sc.ab.distort_strength),
                    "--image-format",
                    "webp",
                    "--outside-mask",
                    "hard",
                    "--blur-fwhm-px",
                    str(sc.ab.blur_fwhm_px),
                    "--blur-edge-factor",
                    str(sc.ab.blur_edge_factor),
                    "--noise-std",
                    str(sc.ab.noise_std),
                    "--seed",
                    str(seed),
                    "--pitch-um",
                    str(sc.pitch_um),
                    "--f-um",
                    str(f_um),
                    "--tz-mm",
                    str(sc.tz_mm),
                    "--baseline-mm",
                    str(baseline_mm),
                    "--squares-x",
                    str(sc.board.squares_x),
                    "--squares-y",
                    str(sc.board.squares_y),
                    "--square-size-mm",
                    str(sc.board.square_size_mm),
                ]
                try:
                    run(gen_cmd)
                except Exception as e:  # pragma: no cover
                    status = "gen_failed"
                    error = str(e)

                # 2) Run the same OpenCV calibration comparison as in the paper.
                cmp_cmd = [
                    py,
                    str(Path("paper/experiments/compare_opencv_calibration_rayfield.py")),
                    str(ds_root),
                    "--split",
                    "train",
                    "--scene",
                    "scene_0000",
                    "--max-frames",
                    str(args.max_frames),
                    "--out",
                    str(report_path),
                ]
                if status == "ok":
                    try:
                        run(cmp_cmd)
                    except Exception as e:  # pragma: no cover
                        status = "compare_failed"
                        error = str(e)

            report: dict[str, Any] | None = None
            metrics: dict[str, float] = {}
            if status == "ok" and report_path.exists():
                report = load_json(report_path)
                metrics = summarize_case(report)

            rf3d_status = "skipped"
            rf3d_error: str | None = None
            rf3d_metrics: dict[str, float] = {}
            rf3d_report_path = results_root / f"{case_name}.rayfield3d.json"
            if args.run_rayfield3d:
                rf3d_status = "ok"
                if not ((not args.rerun) and rf3d_report_path.exists()):
                    rf3d_cmd = [
                        py,
                        str(Path("paper/experiments/calibrate_central_rayfield3d_from_images.py")),
                        str(ds_root),
                        "--split",
                        "train",
                        "--scene",
                        "scene_0000",
                        "--max-frames",
                        str(args.max_frames),
                        "--method2d",
                        "rayfield_tps_robust",
                        "--max-points-per-frame",
                        str(int(args.rayfield3d_max_points)),
                        "--outer-iters",
                        str(args.rayfield3d_outer_iters),
                        "--nmax",
                        str(args.rayfield3d_nmax),
                        "--out",
                        str(rf3d_report_path),
                    ]
                    try:
                        run(rf3d_cmd)
                    except Exception as e:  # pragma: no cover
                        rf3d_status = "rayfield3d_failed"
                        rf3d_error = str(e)
                if rf3d_status == "ok" and rf3d_report_path.exists():
                    try:
                        rf3d_report = load_json(rf3d_report_path)
                        if "pinhole_vs_gt" in rf3d_report:
                            rf3d_metrics = summarize_rayfield3d(rf3d_report)
                        else:
                            rf3d_status = "rayfield3d_missing_pinhole_vs_gt"
                    except Exception as e:  # pragma: no cover
                        rf3d_status = "rayfield3d_parse_failed"
                        rf3d_error = str(e)
            row: dict[str, Any] = {
                "case": case_name,
                "scenario": sc.name,
                "status": status,
                "error": error,
                "rayfield3d_status": rf3d_status,
                "rayfield3d_error": rf3d_error,
                "board_w_mm": sc.board.width_mm,
                "board_h_mm": sc.board.height_mm,
                "square_size_mm": sc.board.square_size_mm,
                "squares_x": sc.board.squares_x,
                "squares_y": sc.board.squares_y,
                "tz_mm": sc.tz_mm,
                "pitch_um": sc.pitch_um,
                "f_um": f_um,
                "f_px": float(f_um / sc.pitch_um),
                "baseline_mm": baseline_mm,
                "baseline_ratio": sc.baseline_ratio,
                "aberration": sc.ab.name,
                "distort": sc.ab.distort,
                "distort_strength": sc.ab.distort_strength,
                "blur_fwhm_px": sc.ab.blur_fwhm_px,
                "blur_edge_factor": sc.ab.blur_edge_factor,
                "noise_std": sc.ab.noise_std,
                **metrics,
                **rf3d_metrics,
            }
            all_rows.append(row)

    # Aggregate per scenario.
    def agg(vals: list[float]) -> dict[str, float]:
        v = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=0)),
            "p50": float(np.quantile(v, 0.50)),
            "p95": float(np.quantile(v, 0.95)),
        }

    by_scn: dict[str, list[dict[str, Any]]] = {}
    for r in all_rows:
        if str(r.get("status")) != "ok":
            continue
        by_scn.setdefault(str(r["scenario"]), []).append(r)

    summary: dict[str, Any] = {"cases": all_rows, "scenarios": {}}
    for scn, rows in sorted(by_scn.items()):
        out: dict[str, Any] = {"n": len(rows)}
        if not rows:
            summary["scenarios"][scn] = out
            continue
        for k in (
            "raw.mono_L_rms_px",
            "rayfield_tps_robust.mono_L_rms_px",
            "raw.mono_R_rms_px",
            "rayfield_tps_robust.mono_R_rms_px",
            "raw.stereo_rms_px",
            "rayfield_tps_robust.stereo_rms_px",
            "raw.baseline_delta_px_rms",
            "rayfield_tps_robust.baseline_delta_px_rms",
            "raw.tri_rms_mm",
            "rayfield_tps_robust.tri_rms_mm",
        ):
            out[k] = agg([float(r[k]) for r in rows])

        # Convenient improvement ratios.
        def ratio(a: str, b: str) -> float:
            num = out[a]["mean"]
            den = out[b]["mean"]
            if not np.isfinite(num) or not np.isfinite(den) or den <= 0:
                return float("nan")
            return float(num / den)

        out["improvement"] = {
            "mono_L_rms_ratio": ratio("raw.mono_L_rms_px", "rayfield_tps_robust.mono_L_rms_px"),
            "mono_R_rms_ratio": ratio("raw.mono_R_rms_px", "rayfield_tps_robust.mono_R_rms_px"),
            "stereo_rms_ratio": ratio("raw.stereo_rms_px", "rayfield_tps_robust.stereo_rms_px"),
            "baseline_delta_px_rms_ratio": ratio(
                "raw.baseline_delta_px_rms",
                "rayfield_tps_robust.baseline_delta_px_rms",
            ),
            "tri_rms_mm_ratio": ratio("raw.tri_rms_mm", "rayfield_tps_robust.tri_rms_mm"),
        }
        summary["scenarios"][scn] = out

    out_path = results_root / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
