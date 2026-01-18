from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MethodMetrics:
    mono_rms_left_px: float | None
    mono_rms_right_px: float | None
    stereo_rms_px: float | None
    baseline_abs_error_mm: float
    baseline_abs_error_px_at_mean_depth: float
    tri_rms_mm: float
    tri_rms_rel_depth_percent: float


def run(cmd: list[str]) -> None:
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


def _metrics_opencv(report: dict[str, Any]) -> MethodMetrics:
    o = report["opencv_pinhole_calib"]
    rig = o["rig"]
    tri = o["triangulation_error_mm"]
    tri_rel = o["triangulation_error_rel_depth_percent"]
    return MethodMetrics(
        mono_rms_left_px=float(o["mono_rms_left_px"]),
        mono_rms_right_px=float(o["mono_rms_right_px"]),
        stereo_rms_px=float(o["stereo_rms_px"]),
        baseline_abs_error_mm=float(rig["baseline_abs_error_mm"]),
        baseline_abs_error_px_at_mean_depth=float(rig["baseline_abs_error_px_at_mean_depth"]),
        tri_rms_mm=float(tri["rms"]),
        tri_rms_rel_depth_percent=float(tri_rel["rms"]),
    )


def _metrics_rayfield3d(report: dict[str, Any]) -> MethodMetrics:
    rig = report["rig"]
    tri = report["rayfield3d_ba"]["triangulation_error_mm"]
    tri_rel = report["rayfield3d_ba"]["triangulation_error_rel_depth_percent"]
    return MethodMetrics(
        mono_rms_left_px=None,
        mono_rms_right_px=None,
        stereo_rms_px=None,
        baseline_abs_error_mm=float(rig["baseline_abs_error_mm"]),
        baseline_abs_error_px_at_mean_depth=float(rig["baseline_abs_error_px_at_mean_depth"]),
        tri_rms_mm=float(tri["rms"]),
        tri_rms_rel_depth_percent=float(tri_rel["rms"]),
    )


def _fmt(x: float | None, nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def _markdown_table(title: str, rows: list[tuple[str, MethodMetrics]]) -> str:
    lines: list[str] = []
    lines.append(f"### {title}")
    lines.append("")
    lines.append(
        "| Method | Mono RMS L (px) | Mono RMS R (px) | Stereo RMS (px) | Baseline abs. err (px@Z̄) | Tri RMS (mm) | Tri RMS (%Z̄) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in rows:
        lines.append(
            "| "
            + name
            + " | "
            + _fmt(m.mono_rms_left_px)
            + " | "
            + _fmt(m.mono_rms_right_px)
            + " | "
            + _fmt(m.stereo_rms_px)
            + " | "
            + _fmt(m.baseline_abs_error_px_at_mean_depth, nd=3)
            + " | "
            + _fmt(m.tri_rms_mm, nd=3)
            + " | "
            + _fmt(m.tri_rms_rel_depth_percent, nd=3)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Compare 3D reconstruction robustness to compression (PNG lossless vs WebP lossy)."
    )
    ap.add_argument("--png", type=Path, default=Path("dataset/compression_sweep/png_lossless"))
    ap.add_argument("--webp", type=Path, default=Path("dataset/compression_sweep/webp_q70"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--max-frames", type=int, default=5)
    ap.add_argument("--outer-iters", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--max-points-per-frame", type=int, default=200)
    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=3.0)
    ap.add_argument("--tps-iters", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("paper/tables/compression_compare/compression_compare_3d_methods.json"))
    args = ap.parse_args(argv)

    py = str(Path(".venv/bin/python") if Path(".venv/bin/python").exists() else Path("python3"))
    cal = str(Path("paper/experiments/calibrate_central_rayfield3d_from_images.py"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    work_dir = args.out.parent

    def run_case(dataset_root: Path, label: str) -> dict[str, Any]:
        raw_out = work_dir / f"{label}.raw.json"
        rf_out = work_dir / f"{label}.rayfield2d.json"

        base_cmd = [
            py,
            cal,
            str(dataset_root),
            "--split",
            str(args.split),
            "--scene",
            str(args.scene),
            "--max-frames",
            str(int(args.max_frames)),
            "--max-points-per-frame",
            str(int(args.max_points_per_frame)),
            "--outer-iters",
            str(int(args.outer_iters)),
            "--nmax",
            str(int(args.nmax)),
            "--tps-lam",
            str(float(args.tps_lam)),
            "--tps-huber",
            str(float(args.tps_huber)),
            "--tps-iters",
            str(int(args.tps_iters)),
        ]

        if not raw_out.exists():
            run([*base_cmd, "--method2d", "raw", "--out", str(raw_out)])
        if not rf_out.exists():
            run([*base_cmd, "--method2d", "rayfield_tps_robust", "--out", str(rf_out)])

        raw = load_json(raw_out)
        rf = load_json(rf_out)

        out = {
            "dataset_root": str(dataset_root),
            "raw_report": str(raw_out),
            "rayfield2d_report": str(rf_out),
            "methods": {
                "opencv_pinhole_raw": _metrics_opencv(raw).__dict__,
                "opencv_pinhole_rayfield2d": _metrics_opencv(rf).__dict__,
                "rayfield3d_ba_rayfield2d": _metrics_rayfield3d(rf).__dict__,
            },
        }
        return out

    png = run_case(args.png, "png_lossless")
    webp = run_case(args.webp, "webp_lossy")

    report = {
        "png": png,
        "webp": webp,
        "settings": {
            "split": str(args.split),
            "scene": str(args.scene),
            "max_frames": int(args.max_frames),
            "rayfield3d": {"outer_iters": int(args.outer_iters), "nmax": int(args.nmax), "max_points_per_frame": int(args.max_points_per_frame)},
            "tps": {"lam": float(args.tps_lam), "huber": float(args.tps_huber), "iters": int(args.tps_iters)},
        },
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print a compact markdown summary.
    def _as_metrics(d: dict[str, Any]) -> dict[str, MethodMetrics]:
        mm: dict[str, MethodMetrics] = {}
        for k, v in d["methods"].items():
            mm[k] = MethodMetrics(**v)
        return mm

    md_png = _as_metrics(png)
    md_webp = _as_metrics(webp)
    print(_markdown_table("PNG (lossless)", list(md_png.items())))
    print(_markdown_table("WebP (lossy)", list(md_webp.items())))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
