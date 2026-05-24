#!/usr/bin/env python3
"""
Generate a paper-ready plot + LaTeX table for the Pycaso-style depth-sweep
compression benchmark.

Inputs: JSON files produced by `paper/experiments/sweep_z_compare_pycaso.py`
        and stored under `paper/tables/pycaso_compression/*.json`.

Outputs:
  - Figure: `paper/figures/pycaso_compression_rms_z_vs_quality.png`
  - Table:  `paper/tables/pycaso_compression_summary.tex`
  - (Optional) compact JSON summary for downstream tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


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
    ap.add_argument("--in-dir", type=Path, default=Path("paper/tables/pycaso_compression"))
    ap.add_argument("--out-fig", type=Path, default=Path("paper/figures/pycaso_compression_rms_z_vs_quality.png"))
    ap.add_argument("--out-tex", type=Path, default=Path("paper/tables/pycaso_compression_summary.tex"))
    ap.add_argument("--out-json", type=Path, default=Path("paper/tables/pycaso_compression_sweep_summary.json"))
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

    # Methods to show (focus: Pycaso vs 3D ray-field).
    methods = [
        ("pycaso_direct_poly", "Pycaso direct", "tab:orange", "o"),
        ("pycaso_soloff_lm", "Pycaso Soloff+LM", "tab:purple", "s"),
        ("rayfield3d_fixed", "3D ray-field", "tab:blue", "D"),
    ]

    # Approximate mean depth (mm) from PNG metadata (used only for the summary JSON).
    mean_depth_mm = 3.0
    if "png_lossless" in cases and "depth_mm" in cases["png_lossless"]:
        dm = cases["png_lossless"]["depth_mm"]
        mean_depth_mm = 0.5 * (float(dm["min"]) + float(dm["max"]))

    # Plot.
    import matplotlib.pyplot as plt  # type: ignore

    plt.figure(figsize=(8.2, 4.6), dpi=150)
    for key, label, color, marker in methods:
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
    plt.title("Pycaso-style depth sweep: RMS Z vs lossy compression")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out_fig)
    plt.close()
    print(f"Wrote {args.out_fig}")

    # Compact JSON summary.
    summary = {
        "schema_version": "stereocomplex.pycaso_compression_sweep.v0",
        "mean_depth_mm_approx": float(mean_depth_mm),
        "cases": {},
    }
    for n in labels:
        codec, q = _parse_case_name(n)
        entry = {"codec": codec, "quality": q, "metrics": {}}
        for key, _, _, _ in methods:
            rms_z_mm = float(cases[n][key]["rms_z_mm"])
            entry["metrics"][key] = {
                "rms_z_mm": rms_z_mm,
                "rms_z_rel_percent": float(100.0 * rms_z_mm / (float(mean_depth_mm) + 1e-12)),
            }
        summary["cases"][n] = entry
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")

    # LaTeX summary table (selected cases). To avoid confusion with the pose-sweep benchmark, we report
    # both absolute RMS Z (mm) and depth-normalized RMS Z (% of mean depth) in the same cells.
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
        ps = float(cases[n]["pycaso_soloff_lm"]["rms_z_mm"])
        rf = float(cases[n]["rayfield3d_fixed"]["rms_z_mm"])
        pdp = 100.0 * pd / (float(mean_depth_mm) + 1e-12)
        psp = 100.0 * ps / (float(mean_depth_mm) + 1e-12)
        rfp = 100.0 * rf / (float(mean_depth_mm) + 1e-12)
        rows.append((label, (pd, pdp), (ps, psp), (rf, rfp)))

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{tabular}{lrrr}")
    tex.append(r"\toprule")
    tex.append(
        r"Codec & \shortstack{Pycaso direct\\RMS $Z$ (mm)\\(\%$\bar Z$)} & "
        r"\shortstack{Pycaso Soloff+LM\\RMS $Z$ (mm)\\(\%$\bar Z$)} & "
        r"\shortstack{3D ray-field\\RMS $Z$ (mm)\\(\%$\bar Z$)} \\"
    )
    tex.append(r"\midrule")
    for label, (pd, pdp), (ps, psp), (rf, rfp) in rows:
        tex.append(f"{label} & {pd:.6f} ({pdp:.3f}\\%) & {ps:.6f} ({psp:.3f}\\%) & {rf:.6f} ({rfp:.3f}\\%) \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(
        rf"\caption{{Pycaso-style depth sweep (synthetic, $\bar Z\approx {mean_depth_mm:.1f}$\,mm): RMS $Z$ error on clean PNG and under representative lossy compression settings, using planar refinement for 2D correspondences. Values are reported both in mm and as a percentage of mean depth. Note: absolute RMS values are not directly comparable to the pose-sweep protocol (Table~\ref{{tab:pycaso_pose_sweep}}) due to different depth range and conditioning.}}"
    )
    tex.append(r"\label{tab:pycaso_compression}")
    tex.append(r"\end{table}")

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_tex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
