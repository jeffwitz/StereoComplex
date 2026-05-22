from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from stereocomplex.eval.charuco_detection import eval_charuco_scene
from stereocomplex.sim.reencode_dataset import ReencodeOptions, reencode_dataset


@dataclass(frozen=True)
class SweepCase:
    name: str
    image_format: str  # png|webp|jpeg
    quality: int
    webp_lossless: bool = False


def run_compression_sweep(
    base_dataset: Path,
    out_dir: Path,
    cases: list[SweepCase],
    splits: tuple[str, ...] = ("train",),
    method: str = "charuco",
    refine: str = "none",
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> Path:
    """Re-encode images under compression settings and evaluate ChArUco detection.

    Each case is re-encoded into a subdirectory of ``out_dir``, then
    ``eval_charuco_scene`` is run on every scene.

    Parameters
    ----------
    base_dataset : Path
        Source dataset root.
    out_dir : Path
        Directory where re-encoded datasets and the report are written.
    cases : list[SweepCase]
        Compression variants to test (format, quality, lossless flag).
    splits : tuple[str, ...]
        Dataset splits to process.
    method : str
        Corner identification method.
    refine : str
        Subpixel refinement strategy.
    tensor_sigma : float
        Gaussian sigma for tensor-based refinement.
    search_radius : int
        Search window half-width for refinement.

    Returns
    -------
    Path
        Path to the written ``compression_sweep_report.json``.
    """
    base_dataset = base_dataset.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {
        "base_dataset": str(base_dataset),
        "out_dir": str(out_dir),
        "splits": list(splits),
        "cases": [],
    }

    for case in cases:
        ds_out = out_dir / case.name
        reencode_dataset(
            base_dataset,
            ds_out,
            ReencodeOptions(image_format=case.image_format, quality=case.quality, webp_lossless=case.webp_lossless),
        )

        scene_results: list[dict[str, object]] = []
        for split in splits:
            split_dir = ds_out / split
            if not split_dir.exists():
                continue
            for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                s = eval_charuco_scene(
                    scene_dir,
                    method=method,
                    refine=refine,
                    tensor_sigma=tensor_sigma,
                    search_radius=search_radius,
                )
                s["split"] = split
                s["scene"] = scene_dir.name
                scene_results.append(s)

        entry = {
            "name": case.name,
            "image_format": case.image_format,
            "quality": case.quality,
            "webp_lossless": case.webp_lossless,
            "dataset_path": str(ds_out),
            "scenes": scene_results,
        }
        results["cases"].append(entry)
        print(json.dumps(entry, sort_keys=True))

    report_path = out_dir / "compression_sweep_report.json"
    report_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    return report_path
