from __future__ import annotations

import argparse
from pathlib import Path

from stereocomplex.sim.cpu.generate_dataset import generate_cpu_dataset
from stereocomplex.sim.dataset_validate import validate_dataset
from stereocomplex.eval.oracle import eval_oracle_dataset
from stereocomplex.eval.charuco_detection import eval_charuco_detection
from stereocomplex.eval.compression_sweep import SweepCase, run_compression_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stereocomplex")
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate-cpu-dataset", help="Generate a synthetic stereo dataset (CPU fallback).")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--scenes", type=int, default=1)
    gen.add_argument("--frames-per-scene", type=int, default=16)
    gen.add_argument("--width", type=int, default=640)
    gen.add_argument("--height", type=int, default=480)
    gen.add_argument("--pattern", type=str, default="auto", choices=["auto", "charuco", "grid"])
    gen.add_argument(
        "--tex-interp",
        type=str,
        default="linear",
        choices=["nearest", "linear", "cubic", "lanczos4"],
        help="Texture sampling interpolation (OpenCV if available).",
    )
    gen.add_argument("--distort", type=str, default="none", choices=["none", "brown"], help="Geometric distortion model.")
    gen.add_argument(
        "--distort-strength",
        type=float,
        default=0.0,
        help="Strength for random distortion coefficients (0 disables).",
    )
    gen.add_argument(
        "--image-format",
        type=str,
        default="png",
        choices=["png", "webp"],
        help="Output image format. For webp, lossless encoding is used.",
    )
    gen.add_argument(
        "--outside-mask",
        type=str,
        default="none",
        choices=["none", "hard"],
        help="If hard, forces pixels outside the board to 0 (black).",
    )
    gen.add_argument("--blur-fwhm-um", type=float, default=0.0, help="Gaussian blur FWHM in µm (0 disables).")
    gen.add_argument("--blur-fwhm-px", type=float, default=0.0, help="Gaussian blur FWHM in output pixels (0 disables).")
    gen.add_argument("--blur-edge-factor", type=float, default=1.0, help="Edge blur multiplier (>=1).")
    gen.add_argument("--blur-edge-start", type=float, default=0.6, help="Radius fraction where edge blur starts (0..1).")
    gen.add_argument("--blur-edge-power", type=float, default=2.0, help="Edge blur ramp exponent.")
    gen.add_argument("--noise-std", type=float, default=0.02, help="Additive Gaussian noise (image in [0,1]).")
    gen.add_argument("--seed", type=int, default=0)

    val = sub.add_parser("validate-dataset", help="Validate dataset structure + basic consistency checks.")
    val.add_argument("dataset_root", type=Path)

    oracle = sub.add_parser("eval-oracle", help="Oracle eval on synthetic scenes (pinhole CPU generator).")
    oracle.add_argument("dataset_root", type=Path)

    ch = sub.add_parser(
        "eval-charuco-detection",
        help="Detect ChArUco corners (OpenCV) and compare to GT (pixel error).",
    )
    ch.add_argument("dataset_root", type=Path)
    ch.add_argument("--no-json", action="store_true", help="Do not write charuco_detection_report.json")
    ch.add_argument(
        "--method",
        type=str,
        default="charuco",
        choices=["charuco", "homography", "pnp", "mls", "mls_h", "pw_affine", "tps", "hybrid", "kfield", "rayfield"],
    )
    ch.add_argument("--refine", type=str, default="none", choices=["none", "tensor", "lines", "lsq", "noble"])
    ch.add_argument("--tensor-sigma", type=float, default=1.5)
    ch.add_argument("--search-radius", type=int, default=3)

    sweep = sub.add_parser(
        "sweep-compression",
        help="Re-encode an existing dataset at different quality settings and evaluate ChArUco detection error.",
    )
    sweep.add_argument("--base", type=Path, required=True, help="Base dataset (prefer lossless PNG).")
    sweep.add_argument("--out", type=Path, required=True, help="Output directory for re-encoded datasets and report.")
    sweep.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated splits to evaluate (default: train).",
    )
    sweep.add_argument("--refine", type=str, default="none", choices=["none", "tensor", "lines", "lsq", "noble"])
    sweep.add_argument("--tensor-sigma", type=float, default=1.5)
    sweep.add_argument("--search-radius", type=int, default=3)
    sweep.add_argument(
        "--method",
        type=str,
        default="charuco",
        choices=["charuco", "homography", "pnp", "mls", "mls_h", "pw_affine", "tps", "hybrid", "kfield", "rayfield"],
    )

    args = parser.parse_args(argv)

    if args.cmd == "generate-cpu-dataset":
        generate_cpu_dataset(
            out_root=args.out,
            scenes=args.scenes,
            frames_per_scene=args.frames_per_scene,
            width=args.width,
            height=args.height,
            pattern=args.pattern,
            tex_interp=args.tex_interp,
            distort=args.distort,
            distort_strength=args.distort_strength,
            image_format=args.image_format,
            outside_mask=args.outside_mask,
            blur_fwhm_um=args.blur_fwhm_um,
            blur_fwhm_px=args.blur_fwhm_px,
            blur_edge_factor=args.blur_edge_factor,
            blur_edge_start=args.blur_edge_start,
            blur_edge_power=args.blur_edge_power,
            noise_std=args.noise_std,
            seed=args.seed,
        )
        return 0

    if args.cmd == "validate-dataset":
        validate_dataset(args.dataset_root)
        return 0

    if args.cmd == "eval-oracle":
        eval_oracle_dataset(args.dataset_root)
        return 0

    if args.cmd == "eval-charuco-detection":
        eval_charuco_detection(
            args.dataset_root,
            write_json=not args.no_json,
            method=args.method,
            refine=args.refine,
            tensor_sigma=args.tensor_sigma,
            search_radius=args.search_radius,
        )
        return 0

    if args.cmd == "sweep-compression":
        splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
        cases = [
            SweepCase(name="png_lossless", image_format="png", quality=100, webp_lossless=False),
            SweepCase(name="webp_q95", image_format="webp", quality=95, webp_lossless=False),
            SweepCase(name="webp_q90", image_format="webp", quality=90, webp_lossless=False),
            SweepCase(name="webp_q80", image_format="webp", quality=80, webp_lossless=False),
            SweepCase(name="webp_q70", image_format="webp", quality=70, webp_lossless=False),
            SweepCase(name="jpeg_q98", image_format="jpeg", quality=98, webp_lossless=False),
            SweepCase(name="jpeg_q95", image_format="jpeg", quality=95, webp_lossless=False),
            SweepCase(name="jpeg_q90", image_format="jpeg", quality=90, webp_lossless=False),
            SweepCase(name="jpeg_q80", image_format="jpeg", quality=80, webp_lossless=False),
        ]
        report_path = run_compression_sweep(
            args.base,
            args.out,
            cases=cases,
            splits=splits,
            method=args.method,
            refine=args.refine,
            tensor_sigma=args.tensor_sigma,
            search_radius=args.search_radius,
        )
        print(f"Wrote {report_path}")
        return 0

    raise AssertionError(f"Unhandled cmd: {args.cmd}")
