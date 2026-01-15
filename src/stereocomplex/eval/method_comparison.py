from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stereocomplex.eval.charuco_detection import ErrorStats, collect_charuco_scene_errors, _stats_to_dict, _summarize


@dataclass(frozen=True)
class MethodCase:
    name: str
    method: str
    refine: str = "none"


def compare_charuco_methods(
    dataset_root: Path,
    cases: list[MethodCase],
    splits: tuple[str, ...] = ("train",),
    tensor_sigma: float = 1.5,
    search_radius: int = 3,
) -> dict[str, object]:
    dataset_root = dataset_root.resolve()

    report: dict[str, object] = {
        "dataset_root": str(dataset_root),
        "splits": list(splits),
        "cases": [],
    }

    for case in cases:
        all_err_L: list[float] = []
        all_err_R: list[float] = []
        all_dx_L: list[float] = []
        all_dy_L: list[float] = []
        all_dx_R: list[float] = []
        all_dy_R: list[float] = []
        n_det_L = 0
        n_det_R = 0
        n_match_L = 0
        n_match_R = 0

        scene_summaries: list[dict[str, object]] = []
        for split in splits:
            split_dir = dataset_root / split
            if not split_dir.exists():
                continue
            for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                raw = collect_charuco_scene_errors(
                    scene_dir,
                    method=case.method,
                    refine=case.refine,
                    tensor_sigma=tensor_sigma,
                    search_radius=search_radius,
                )

                L = raw["left"]
                R = raw["right"]
                all_err_L.extend(L["errors"])
                all_dx_L.extend(L["dx"])
                all_dy_L.extend(L["dy"])
                all_err_R.extend(R["errors"])
                all_dx_R.extend(R["dx"])
                all_dy_R.extend(R["dy"])
                n_det_L += int(L["n_detected"])
                n_det_R += int(R["n_detected"])
                n_match_L += int(L["n_matched"])
                n_match_R += int(R["n_matched"])

                stats_L = _summarize(L["errors"], L["dx"], L["dy"])
                stats_R = _summarize(R["errors"], R["dx"], R["dy"])
                scene_summaries.append(
                    {
                        "split": split,
                        "scene": scene_dir.name,
                        "left": {"n_detected": int(L["n_detected"]), "n_matched": int(L["n_matched"]), **_stats_to_dict(stats_L)},
                        "right": {"n_detected": int(R["n_detected"]), "n_matched": int(R["n_matched"]), **_stats_to_dict(stats_R)},
                    }
                )

        stats_L_all: ErrorStats | None = _summarize(all_err_L, all_dx_L, all_dy_L)
        stats_R_all: ErrorStats | None = _summarize(all_err_R, all_dx_R, all_dy_R)

        entry = {
            "name": case.name,
            "method": case.method,
            "refine": case.refine,
            "left": {"n_detected": int(n_det_L), "n_matched": int(n_match_L), **_stats_to_dict(stats_L_all)},
            "right": {"n_detected": int(n_det_R), "n_matched": int(n_match_R), **_stats_to_dict(stats_R_all)},
            "scenes": scene_summaries,
        }
        report["cases"].append(entry)

    return report


def write_latex_table(report: dict[str, object], out_path: Path, caption: str, label: str) -> None:
    def esc(s: str) -> str:
        # Minimal LaTeX escaping for captions/labels (not for full LaTeX content).
        return (
            s.replace("\\", "\\textbackslash{}")
            .replace("&", "\\&")
            .replace("%", "\\%")
            .replace("$", "\\$")
            .replace("#", "\\#")
            .replace("_", "\\_")
            .replace("{", "\\{")
            .replace("}", "\\}")
            .replace("~", "\\textasciitilde{}")
            .replace("^", "\\textasciicircum{}")
        )

    cases = report["cases"]
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\toprule")
    lines.append("Method & $n_L$ & RMS$_L$ & P95$_L$ & $n_R$ & RMS$_R$ & P95$_R$\\\\")
    lines.append("\\midrule")
    for c in cases:
        L = c["left"]
        R = c["right"]
        nL = int(L["n_matched"])
        nR = int(R["n_matched"])
        rmsL = float(L["rms_px"])
        rmsR = float(R["rms_px"])
        p95L = float(L["p95_px"])
        p95R = float(R["p95_px"])
        lines.append(f"{c['name']} & {nL:d} & {rmsL:.3f} & {p95L:.3f} & {nR:d} & {rmsR:.3f} & {p95R:.3f}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{esc(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report_json(report: dict[str, object], out_path: Path) -> None:
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
