from __future__ import annotations

import argparse
from pathlib import Path

from stereocomplex.eval.method_comparison import MethodCase, compare_charuco_methods, write_latex_table, write_report_json


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare ChArUco identification methods and generate a LaTeX table.")
    ap.add_argument("dataset_root", type=Path)
    ap.add_argument("--splits", type=str, default="train", help="Comma-separated splits (default: train).")
    ap.add_argument("--out-tex", type=Path, default=Path("paper/tables/charuco_methods.tex"))
    ap.add_argument("--out-json", type=Path, default=Path("paper/tables/charuco_methods.json"))
    args = ap.parse_args()

    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())

    cases = [
        MethodCase(name="OpenCV ChArUco", method="charuco"),
        MethodCase(name="Homography (2nd pass)", method="homography"),
        MethodCase(name="TPS (smooth warp)", method="tps"),
        MethodCase(name="PnP (K+distortion)", method="pnp"),
        MethodCase(name="Plane ray-field (H+smooth residual)", method="rayfield"),
        MethodCase(name="Ray-field TPS residual (H+TPS+IRLS)", method="rayfield_tps_robust"),
    ]

    report = compare_charuco_methods(args.dataset_root, cases=cases, splits=splits)

    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    write_report_json(report, args.out_json)
    write_latex_table(
        report,
        args.out_tex,
        caption=f"ChArUco corner identification error on {args.dataset_root} (splits: {', '.join(splits)}). Errors are in pixels.",
        label="tab:charuco_methods",
    )

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
