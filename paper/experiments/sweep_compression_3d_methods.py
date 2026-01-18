from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
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
    n_frames: int | None = None
    n_points_total: int | None = None


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
    diag = o.get("diagnostics", {})
    return MethodMetrics(
        mono_rms_left_px=float(o["mono_rms_left_px"]),
        mono_rms_right_px=float(o["mono_rms_right_px"]),
        stereo_rms_px=float(o["stereo_rms_px"]),
        baseline_abs_error_mm=float(rig["baseline_abs_error_mm"]),
        baseline_abs_error_px_at_mean_depth=float(rig["baseline_abs_error_px_at_mean_depth"]),
        tri_rms_mm=float(tri["rms"]),
        tri_rms_rel_depth_percent=float(tri_rel["rms"]),
        n_frames=int(diag.get("n_frames", 0)) if "n_frames" in diag else None,
        n_points_total=int(diag.get("n_points_total", 0)) if "n_points_total" in diag else None,
    )


def _metrics_rayfield3d(report: dict[str, Any], *, depth_mean_mm: float) -> MethodMetrics:
    rig = report["rig"]
    # Ray-field 3D has a weak gauge (scale/pose drift). Report similarity-aligned metrics.
    tri = report["rayfield3d_ba"]["triangulation_error_mm_aligned_similarity"]
    diag = report["rayfield3d_ba"].get("diagnostics", {})
    return MethodMetrics(
        mono_rms_left_px=None,
        mono_rms_right_px=None,
        stereo_rms_px=None,
        baseline_abs_error_mm=float(rig["baseline_abs_error_mm"]),
        baseline_abs_error_px_at_mean_depth=float(rig["baseline_abs_error_px_at_mean_depth"]),
        tri_rms_mm=float(tri["rms"]),
        tri_rms_rel_depth_percent=float(100.0 * float(tri["rms"]) / (float(depth_mean_mm) + 1e-12)),
        n_frames=int(diag.get("n_frames", 0)) if "n_frames" in diag else None,
        n_points_total=int(diag.get("n_points_total", 0)) if "n_points_total" in diag else None,
    )


def run_case(
    *,
    py: str,
    cal_script: Path,
    dataset_root: Path,
    split: str,
    scene: str,
    max_frames: int,
    max_points_per_frame: int,
    outer_iters: int,
    nmax: int,
    tps_lam: float,
    tps_huber: float,
    tps_iters: int,
    out_dir: Path,
    label: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / f"{label}.raw.json"
    rf_out = out_dir / f"{label}.rayfield2d.json"

    base_cmd = [
        py,
        str(cal_script),
        str(dataset_root),
        "--split",
        str(split),
        "--scene",
        str(scene),
        "--max-frames",
        str(int(max_frames)),
        "--max-points-per-frame",
        str(int(max_points_per_frame)),
        "--outer-iters",
        str(int(outer_iters)),
        "--nmax",
        str(int(nmax)),
        "--tps-lam",
        str(float(tps_lam)),
        "--tps-huber",
        str(float(tps_huber)),
        "--tps-iters",
        str(int(tps_iters)),
    ]

    if not raw_out.exists():
        run([*base_cmd, "--method2d", "raw", "--out", str(raw_out)])
    if not rf_out.exists():
        run([*base_cmd, "--method2d", "rayfield_tps_robust", "--out", str(rf_out)])

    raw = load_json(raw_out)
    rf = load_json(rf_out)
    depth_mean = float(rf["depth_mm"]["mean"]) if "depth_mm" in rf else float("nan")

    return {
        "dataset_root": str(dataset_root),
        "raw_report": str(raw_out),
        "rayfield2d_report": str(rf_out),
        "methods": {
            "opencv_pinhole_raw": asdict(_metrics_opencv(raw)),
            "opencv_pinhole_rayfield2d": asdict(_metrics_opencv(rf)),
            "rayfield3d_ba_rayfield2d": asdict(_metrics_rayfield3d(rf, depth_mean_mm=depth_mean)),
        },
        "depth_mm_mean": depth_mean,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sweep compression quality and compare 3D reconstruction pipelines (PNG vs WebP qualities).",
    )
    ap.add_argument("--root", type=Path, default=Path("dataset/compression_sweep"))
    ap.add_argument(
        "--only",
        type=str,
        default="all",
        choices=["all", "jpeg", "webp"],
        help="Compute only a subset (always reuses cached per-case JSON files when available).",
    )
    ap.add_argument(
        "--skip-compute",
        action="store_true",
        help="Do not run any calibration; only (re)generate plots/JSON snapshots from existing results.",
    )
    ap.add_argument("--png", type=str, default="png_lossless", help="Subdir name for the lossless PNG dataset.")
    ap.add_argument(
        "--webp",
        type=str,
        default="webp_q70,webp_q80,webp_q90,webp_q95",
        help="Comma-separated WebP dataset subdirs to compare.",
    )
    ap.add_argument(
        "--jpeg",
        type=str,
        default="jpeg_q80,jpeg_q90,jpeg_q95,jpeg_q98",
        help="Comma-separated JPEG dataset subdirs to compare.",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--scene", default="scene_0000")
    ap.add_argument("--max-frames", type=int, default=5)
    ap.add_argument("--outer-iters", type=int, default=3)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--max-points-per-frame", type=int, default=200)
    ap.add_argument("--tps-lam", type=float, default=10.0)
    ap.add_argument("--tps-huber", type=float, default=3.0)
    ap.add_argument("--tps-iters", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("paper/tables/compression_compare/sweep_webp_quality.json"))
    ap.add_argument("--plots-out", type=Path, default=Path("docs/assets/compression_sweep"))
    args = ap.parse_args(argv)

    py = str(Path(".venv/bin/python") if Path(".venv/bin/python").exists() else Path("python3"))
    cal_script = Path("paper/experiments/calibrate_central_rayfield3d_from_images.py")

    root = args.root
    png_root = root / str(args.png)
    webp_names = [s.strip() for s in str(args.webp).split(",") if s.strip()]
    jpeg_names = [s.strip() for s in str(args.jpeg).split(",") if s.strip()]
    webp_roots = [(name, root / name) for name in webp_names]
    jpeg_roots = [(name, root / name) for name in jpeg_names]

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resume behavior: if the sweep JSON already exists, extend it in-place.
    if args.out.exists():
        try:
            results = load_json(args.out)
        except Exception:
            results = {}
    else:
        results = {}

    if results.get("schema_version") != "stereocomplex.compression_sweep_3d.v0":
        results = {"schema_version": "stereocomplex.compression_sweep_3d.v0"}

    results["settings"] = {
        "root": str(root),
        "png": str(args.png),
        "webp": webp_names,
        "jpeg": jpeg_names,
        "split": str(args.split),
        "scene": str(args.scene),
        "max_frames": int(args.max_frames),
        "rayfield3d": {
            "outer_iters": int(args.outer_iters),
            "nmax": int(args.nmax),
            "max_points_per_frame": int(args.max_points_per_frame),
        },
        "tps": {"lam": float(args.tps_lam), "huber": float(args.tps_huber), "iters": int(args.tps_iters)},
    }
    if "cases" not in results or not isinstance(results["cases"], dict):
        results["cases"] = {}

    # Optional convenience: if jpeg sub-datasets are missing, generate them from PNG.
    def ensure_reencode_if_missing(name: str, ds: Path) -> None:
        if ds.exists() and (ds / "manifest.json").exists():
            return
        if not (png_root / "manifest.json").exists():
            return
        try:
            from stereocomplex.sim.reencode_dataset import ReencodeOptions, reencode_dataset  # noqa: PLC0415
        except Exception:
            return
        m = re.search(r"_q(\d+)$", name)
        if not m:
            return
        q = int(m.group(1))
        reencode_dataset(png_root, ds, ReencodeOptions(image_format="jpeg", quality=q, webp_lossless=False))

    cases: dict[str, Any] = results["cases"]

    def maybe_compute(name: str, ds_root: Path) -> None:
        if bool(args.skip_compute):
            return
        if name in cases:
            # If previous reports are still present, keep them.
            prev = cases[name]
            rp = Path(str(prev.get("raw_report", "")))
            pp = Path(str(prev.get("rayfield2d_report", "")))
            if rp.exists() and pp.exists():
                return
        cases[name] = run_case(
            py=py,
            cal_script=cal_script,
            dataset_root=ds_root,
            split=str(args.split),
            scene=str(args.scene),
            max_frames=int(args.max_frames),
            max_points_per_frame=int(args.max_points_per_frame),
            outer_iters=int(args.outer_iters),
            nmax=int(args.nmax),
            tps_lam=float(args.tps_lam),
            tps_huber=float(args.tps_huber),
            tps_iters=int(args.tps_iters),
            out_dir=out_dir,
            label=str(name),
        )

    # Ensure the PNG baseline is present (or stays as-is in plot-only mode).
    if "png_lossless" not in cases:
        maybe_compute("png_lossless", png_root)

    # Keep a stable order by increasing quality if possible.
    def quality_key(name: str) -> int:
        m = re.search(r"_q(\d+)$", name)
        if not m:
            return 10_000
        return int(m.group(1))

    if args.only in ("all", "webp"):
        for name, ds in sorted(webp_roots, key=lambda kv: quality_key(kv[0])):
            maybe_compute(name, ds)

    if args.only in ("all", "jpeg"):
        for name, ds in sorted(jpeg_roots, key=lambda kv: quality_key(kv[0])):
            ensure_reencode_if_missing(name, ds)
            maybe_compute(name, ds)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Plots (matplotlib).
    plots_out = args.plots_out
    plots_out.mkdir(parents=True, exist_ok=True)
    # Also write a small, doc-friendly JSON snapshot next to the plots (tracked by git).
    compact = {
        "schema_version": results.get("schema_version"),
        "settings": results.get("settings", {}),
        "cases": {k: v.get("methods", {}) for k, v in (results.get("cases", {}) or {}).items()},
    }
    (plots_out / "sweep_metrics.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return 0

    cases = results["cases"]
    # For plotting, prefer "all available" qualities already present in results,
    # regardless of what was passed in --webp/--jpeg for this run.
    labels_webp = sorted([k for k in cases.keys() if re.search(r"^webp_q\d+$", k)], key=quality_key)
    labels_jpeg = sorted([k for k in cases.keys() if re.search(r"^jpeg_q\d+$", k)], key=quality_key)
    labels = [n for n in ["png_lossless", *labels_webp, *labels_jpeg] if n in cases]

    def xval(name: str) -> float:
        if name == "png_lossless":
            return 101.0
        m = re.search(r"_q(\d+)$", name)
        return float(m.group(1)) if m else float("nan")

    def x_with_offset(name: str) -> float:
        xv = xval(name)
        if name.startswith("webp_"):
            return xv - 0.15
        if name.startswith("jpeg_"):
            return xv + 0.15
        return xv

    x = [x_with_offset(n) for n in labels]

    methods = ["opencv_pinhole_raw", "opencv_pinhole_rayfield2d", "rayfield3d_ba_rayfield2d"]
    colors = {"opencv_pinhole_raw": "tab:red", "opencv_pinhole_rayfield2d": "tab:blue", "rayfield3d_ba_rayfield2d": "tab:green"}

    def series(metric: str, method: str) -> list[float]:
        out: list[float] = []
        for n in labels:
            v = cases[n]["methods"][method].get(metric)
            out.append(float("nan") if v is None else float(v))
        return out

    def plot_metric(metric: str, ylabel: str, title: str, filename: str) -> None:
        plt.figure(figsize=(8.2, 4.6), dpi=150)
        # Plot each method as 2 codec lines (webp/jpeg) + a PNG reference point.
        for m in methods:
            # PNG reference.
            y_png = series(metric, m)[0]
            plt.scatter([x[0]], [y_png], marker="s", s=42, color=colors[m], label=f"{m} (PNG)")

            # WebP line (if present).
            if labels_webp:
                xw = [x_with_offset(n) for n in labels_webp]
                yw = [cases[n]["methods"][m].get(metric) for n in labels_webp]
                yw = [float("nan") if v is None else float(v) for v in yw]
                plt.plot(xw, yw, marker="o", linewidth=2.0, color=colors[m], alpha=0.85, label=f"{m} (WebP)")

            # JPEG line (if present).
            if labels_jpeg:
                xj = [x_with_offset(n) for n in labels_jpeg]
                yj = [cases[n]["methods"][m].get(metric) for n in labels_jpeg]
                yj = [float("nan") if v is None else float(v) for v in yj]
                plt.plot(xj, yj, marker="^", linewidth=2.0, color=colors[m], alpha=0.55, label=f"{m} (JPEG)")

        plt.grid(True, alpha=0.25)
        plt.xlabel("quality (q)   [PNG lossless shown at 101]")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(plots_out / filename)
        plt.close()

    plot_metric(
        "baseline_abs_error_px_at_mean_depth",
        "baseline abs. error (px @ mean depth)",
        "Compression sweep: baseline error vs codec quality",
        "baseline_abs_error_px_at_mean_depth.png",
    )
    plot_metric(
        "tri_rms_rel_depth_percent",
        "triangulation RMS (% mean depth)",
        "Compression sweep: triangulation error vs codec quality",
        "tri_rms_rel_depth_percent.png",
    )
    plot_metric(
        "stereo_rms_px",
        "stereo RMS reprojection (px)",
        "Compression sweep: stereo RMS vs WebP quality (OpenCV only)",
        "stereo_rms_px.png",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
