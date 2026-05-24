#!/usr/bin/env python3
"""
Generate a paper-ready plot + LaTeX table for the Pycaso baseline on the
*pose-sweep* compression benchmark (the main stereo calibration dataset).

Inputs: JSON files produced by `paper/experiments/sweep_z_compare_pycaso.py`
        when run on `dataset/compression_sweep/*`.

Outputs:
  - Figure: `paper/figures/pycaso_pose_sweep_rms_z_vs_quality.png`
  - Table:  `paper/tables/pycaso_pose_sweep_summary.tex`
  - (Optional) compact JSON summary for downstream tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _parse_case_name(name: str) -> tuple[str, int]:
    if name == "png_lossless":
        return ("png", 101)
    m = re.match(r"^(webp|jpeg)_q(\d+)$", name)
    if not m:
        return ("other", 0)
    return (m.group(1), int(m.group(2)))


def _x_with_offset(codec: str, quality: int) -> float:
    if codec == "webp":
        return float(quality) - 0.15
    if codec == "jpeg":
        return float(quality) + 0.15
    return float(quality)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("paper/tables/pycaso_pose_sweep"))
    ap.add_argument("--out-fig", type=Path, default=Path("paper/figures/pycaso_pose_sweep_rms_z_vs_quality.png"))
    ap.add_argument("--out-tex", type=Path, default=Path("paper/tables/pycaso_pose_sweep_summary.tex"))
    ap.add_argument("--out-json", type=Path, default=Path("paper/tables/pycaso_pose_sweep_sweep_summary.json"))
    ap.add_argument(
        "--table-cases",
        type=str,
        default="png_lossless,webp_q70,jpeg_q80",
        help="Comma-separated case names to include in the LaTeX summary table.",
    )
    args = ap.parse_args(argv)

    in_dir = args.in_dir
    cases: dict[str, dict] = {}
    for p in sorted(in_dir.glob("*.json")):
        cases[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    if not cases:
        raise SystemExit(f"No JSON cases found in {in_dir}")

    labels = sorted(cases.keys(), key=lambda n: (_parse_case_name(n)[0], _parse_case_name(n)[1]))

    # Plot (focus: Pycaso direct vs 3D ray-field).
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(8.2, 4.6), dpi=150)
    for key, label, color, marker in [
        ("pycaso_direct_poly", "Pycaso direct", "tab:orange", "o"),
        ("rayfield3d_fixed", "3D ray-field", "tab:blue", "D"),
    ]:
        xs = []
        ys = []
        for n in labels:
            codec, q = _parse_case_name(n)
            xs.append(_x_with_offset(codec, q))
            ys.append(float(cases[n][key]["rms_z_mm"]))
        plt.plot(xs, ys, marker=marker, linewidth=2.0, color=color, label=label)

    plt.grid(True, alpha=0.25)
    plt.xlabel("quality (q)   [PNG lossless shown at 101]")
    plt.ylabel("RMS Z error (mm)")
    plt.title("Pose sweep (planar stereo calibration): RMS Z vs lossy compression")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig)
    plt.close()
    print(f"Wrote {args.out_fig}")

    # Compact JSON summary.
    mean_depth_mm = None
    if "png_lossless" in cases and "depth_mm" in cases["png_lossless"]:
        dm = cases["png_lossless"]["depth_mm"]
        mean_depth_mm = 0.5 * (float(dm["min"]) + float(dm["max"]))
    summary = {
        "schema_version": "stereocomplex.pycaso_pose_sweep.v0",
        "mean_depth_mm_approx": float(mean_depth_mm) if mean_depth_mm is not None else None,
        "cases": {},
    }
    for n in labels:
        codec, q = _parse_case_name(n)
        entry = {"codec": codec, "quality": q, "metrics": {}}
        for key in ["pycaso_direct_poly", "rayfield3d_fixed"]:
            rms_z_mm = float(cases[n][key]["rms_z_mm"])
            entry["metrics"][key] = {
                "rms_z_mm": rms_z_mm,
                "rms_z_rel_percent": (
                    float(100.0 * rms_z_mm / (float(mean_depth_mm) + 1e-12)) if mean_depth_mm is not None else None
                ),
            }
        summary["cases"][n] = entry
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")

    # LaTeX summary table (selected cases). Report both absolute RMS Z (mm) and depth-normalized
    # RMS Z (% of mean depth) to make scale differences explicit when compared to the depth-sweep protocol.
    wanted = [s.strip() for s in str(args.table_cases).split(",") if s.strip()]
    rows = []
    for n in wanted:
        if n not in cases:
            continue
        codec, q = _parse_case_name(n)
        if n == "png_lossless":
            label = "PNG (lossless)"
        else:
            label = f"{codec.upper()} (q{q})" if codec != "webp" else f"WebP (q{q})"
        pd = float(cases[n]["pycaso_direct_poly"]["rms_z_mm"])
        rf = float(cases[n]["rayfield3d_fixed"]["rms_z_mm"])
        md = float(mean_depth_mm) if mean_depth_mm is not None else float("nan")
        pdp = 100.0 * pd / (md + 1e-12)
        rfp = 100.0 * rf / (md + 1e-12)
        rows.append((label, (pd, pdp), (rf, rfp)))

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{tabular}{lrr}")
    tex.append(r"\toprule")
    tex.append(
        r"Codec & \shortstack{Pycaso direct\\RMS $Z$ (mm)\\(\%$\bar Z$)} & "
        r"\shortstack{3D ray-field\\RMS $Z$ (mm)\\(\%$\bar Z$)} \\"
    )
    tex.append(r"\midrule")
    for label, (pd, pdp), (rf, rfp) in rows:
        tex.append(f"{label} & {pd:.6f} ({pdp:.3f}\\%) & {rf:.6f} ({rfp:.3f}\\%) \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    if mean_depth_mm is None:
        tex.append(
            r"\caption{Pose-sweep compression benchmark (synthetic): RMS $Z$ error on selected codecs, using planar refinement for 2D correspondences. Values are reported both in mm and as a percentage of mean depth. Soloff+LM is omitted here as it is substantially worse in this setting; see Section~\ref{sec:pycaso}. Note: absolute RMS values are not directly comparable to the depth-sweep protocol (Table~\ref{tab:pycaso_compression}) due to different depth range and conditioning.}"
        )
    else:
        tex.append(
            rf"\caption{{Pose-sweep compression benchmark (synthetic, $\bar Z\approx {float(mean_depth_mm):.0f}$\,mm): RMS $Z$ error on selected codecs, using planar refinement for 2D correspondences. Values are reported both in mm and as a percentage of mean depth. Soloff+LM is omitted here as it is substantially worse in this setting; see Section~\ref{{sec:pycaso}}. Note: absolute RMS values are not directly comparable to the depth-sweep protocol (Table~\ref{{tab:pycaso_compression}}) due to different depth range and conditioning.}}"
        )
    tex.append(r"\label{tab:pycaso_pose_sweep}")
    tex.append(r"\end{table}")

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_tex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
