"""Lightweight reproduction check for documented results.

Usage:
    python examples/reproduce_docs_results.py --fast    # ~60 s
    python examples/reproduce_docs_results.py --full    # ~15 min

Checks that expected output assets exist (FAST) or regenerates them
(FULL) by running the relevant notebooks and sweep scripts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_CMO = REPO_ROOT / "docs" / "assets" / "cmo_model_selection"
ASSETS_DVR = REPO_ROOT / "docs" / "assets" / "direct_vs_rayfield_inversion"

FAST_PATHS = [
    ASSETS_DVR / "direct_vs_rayfield_summary.json",
]

FULL_PATHS = FAST_PATHS + [
    ASSETS_CMO / "classification_heatmap.png",
    ASSETS_CMO / "classification_heatmap_noisy.png",
]


def load_json_summary(path: Path) -> dict:
    """Load a JSON summary file, returning {} on failure."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def require_paths(paths: list[Path]) -> None:
    """Raise FileNotFoundError if any path is missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected assets:\n  " + "\n  ".join(str(p) for p in missing)
        )


def format_direct_vs_rayfield_summary(summary: dict) -> str:
    """Format the notebook-08 summary as a compact report."""
    if not summary:
        return "No summary found."
    lines = [
        f"  oracle:              {summary.get('oracle', '?')}",
        f"  n_poses:             {summary.get('n_poses', '?')}",
        f"  noise_std_px:        {summary.get('noise_std_px', '?')}",
    ]
    pa = summary.get("pipeline_A", {})
    pb = summary.get("pipeline_B", {})
    if pa:
        lines.append(f"  pipeline A:  rms={pa.get('rms_px','?'):.2f} px  "
                     f"converged={pa.get('converged','?')}  "
                     f"elapsed={pa.get('elapsed_s','?'):.0f}s")
    if pb:
        lines.append(f"  pipeline B:  winner={pb.get('rayfield_winner','?')}  "
                     f"correct={summary.get('rayfield_correct','?')}  "
                     f"Zernike rms={pb.get('zernike_rms_mm','?'):.4f} mm")
    lines.append(f"  total runtime:       ~{pa.get('elapsed_s',0) + pb.get('zernike_elapsed_s',0):.0f}s")
    return "\n".join(lines)


def run_notebook(script: Path) -> None:
    """Run a notebook .py script from the repo root."""
    print(f"  Running {script.name} …")
    subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT, check=True, capture_output=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", default=True,
                        help="Fast check: verify existing assets (default)")
    parser.add_argument("--full", action="store_true",
                        help="Full check: regenerate all assets")
    args = parser.parse_args()

    if args.full:
        print("=== Full reproduction ===")
        run_notebook(REPO_ROOT / "examples" / "notebooks" / "07_model_selection_matrix.py")
        run_notebook(REPO_ROOT / "examples" / "notebooks" / "08_direct_vs_rayfield_inversion.py")
        require_paths(FULL_PATHS)
    else:
        print("=== Fast reproduction check ===")
        # If the summary JSON doesn't exist, run notebook 08 to generate it
        summary_path = ASSETS_DVR / "direct_vs_rayfield_summary.json"
        if not summary_path.exists():
            print("  Summary missing, running notebook 08 (FAST) …")
            run_notebook(REPO_ROOT / "examples" / "notebooks" / "08_direct_vs_rayfield_inversion.py")
        require_paths(FAST_PATHS)

    # Print the summary
    summary = load_json_summary(ASSETS_DVR / "direct_vs_rayfield_summary.json")
    print("\nDirect-vs-rayfield FAST summary:")
    print(format_direct_vs_rayfield_summary(summary))

    print("\nAll checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
