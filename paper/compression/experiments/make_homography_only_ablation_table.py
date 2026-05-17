from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tri_rms_percent_rayfield(report: dict) -> float:
    depth_mean = float(report.get("depth_mm", {}).get("mean", float("nan")))
    tri_mm = float(report["rayfield3d_ba"]["triangulation_error_mm_aligned_similarity"]["rms"])
    return 100.0 * tri_mm / (depth_mean + 1e-12)


def main() -> int:
    root = Path("paper/tables")
    report_h_png = load_json(root / "homography_only_png_lossless.json")
    report_h_webp = load_json(root / "homography_only_webp_q70.json")

    report_tps_png = load_json(Path("paper/tables/compression_compare/png_lossless.rayfield2d.json"))
    report_tps_webp = load_json(Path("paper/tables/compression_compare/webp_q70.rayfield2d.json"))

    rows = [
        ("PNG (lossless)", "Homography only", report_h_png),
        ("PNG (lossless)", "Homography + TPS", report_tps_png),
        ("WebP (q70)", "Homography only", report_h_webp),
        ("WebP (q70)", "Homography + TPS", report_tps_webp),
    ]

    out_tex = root / "homography_only_ablation.tex"
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append("\\begin{tabular}{l l c c c}")
    lines.append("\\toprule")
    lines.append(
        "Codec & 2D mapping & "
        "Stereo RMS (px) & "
        "Baseline err (px @ $\\bar Z$) & "
        "Tri RMS (\\%$\\bar Z$)\\\\"
    )
    lines.append("\\midrule")

    for codec, name, report in rows:
        ocv = report["opencv_pinhole_calib"]
        stereo_rms = float(ocv["stereo_rms_px"])
        base_px = float(ocv["rig"]["baseline_abs_error_px_at_mean_depth"])
        tri_pct = float(ocv["triangulation_error_rel_depth_percent"]["rms"])

        lines.append(
            f"{codec} & {name} & "
            f"{stereo_rms:.3f} & {base_px:.3f} & {tri_pct:.3f}\\\\"
        )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Ablation (OpenCV pinhole backend): homography-only versus homography+TPS residual warp as the planar mapping used to generate 2D correspondences. "
        "Without the residual warp, a global homography cannot absorb lens distortion and yields unstable stereo calibration under the same settings. "
        "All results use the same compression benchmark configuration (5 frames).}"
    )
    lines.append("\\label{tab:homography_only_ablation}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
