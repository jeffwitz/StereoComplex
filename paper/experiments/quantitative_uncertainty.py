import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent

METHODS_JSON = ROOT / "tables" / "compression_compare" / "compression_compare_3d_methods.json"
OUTPUT_TABLE = ROOT / "tables" / "uncertainty_comparison.tex"

CONDITIONS = [
    {
        "label": "PNG lossless",
        "key": "png",
        "dataset": Path("dataset/compression_sweep/png_lossless"),
        "rayfile": ROOT / "tables" / "compression_compare" / "png_lossless.rayfield2d.json",
        "pinhole": {"source": "agg", "key": "opencv_pinhole_rayfield2d"},
    },
    {
        "label": "WebP q70",
        "key": "webp",
        "dataset": Path("dataset/compression_sweep/webp_q70"),
        "rayfile": ROOT / "tables" / "compression_compare" / "webp_q70.rayfield2d.json",
        "pinhole": {"source": "agg", "key": "opencv_pinhole_rayfield2d"},
    },
    {
        "label": "JPEG q80",
        "key": "jpeg",
        "dataset": Path("dataset/compression_sweep/jpeg_q80"),
        "rayfile": ROOT / "tables" / "compression_compare" / "jpeg_q80.rayfield2d.json",
        "pinhole": {
            "source": "raw",
            "file": ROOT / "tables" / "compression_compare" / "jpeg_q80.raw.json",
        },
    },
]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def compute_uncertainties():
    data3d = load_json(METHODS_JSON)
    rows = []
    for cond in CONDITIONS:
        label = cond["label"]
        key = cond["key"]
        dataset_root = cond["dataset"]
        rayjson = cond["rayfile"]
        entry = data3d.get(key, None)
        tri_pinhole = None
        if entry:
            tri_pinhole = entry["methods"]["opencv_pinhole_rayfield2d"]["tri_rms_mm"]
        if cond["pinhole"]["source"] == "raw":
            raw_data = load_json(cond["pinhole"]["file"])
            # Use RMS from the pinhole calibration block
            tri_pinhole = raw_data["opencv_pinhole_calib"]["triangulation_error_mm"]["rms"]
        if tri_pinhole is None:
            raise RuntimeError(f"Missing pinhole tri RMS for {cond['label']}")
        tri_ray = entry["methods"]["rayfield3d_ba_rayfield2d"]["tri_rms_mm"] if entry else None
        if tri_ray is None:
            ray_entry = load_json(cond["rayfile"])
            tri_ray = ray_entry["rayfield3d_ba"]["triangulation_error_mm"]["rms"]
        meta_path = REPO_ROOT / cond["dataset"] / "train" / "scene_0000" / "meta.json"
        baseline = load_json(meta_path)["sim_params"]["baseline_mm"]
        ray_data = load_json(rayjson)
        depth_mean = ray_data["depth_mm"]["mean"]
        ray_skew = ray_data["rayfield3d_ba"]["ray_skew_mm"]["rms"]

        u_alpha_pin = (baseline / depth_mean ** 2) * (tri_pinhole / math.sqrt(2))
        u_alpha_ray = ray_skew / depth_mean
        u_Z_pin = (depth_mean ** 2 / baseline) * math.sqrt(2) * u_alpha_pin
        u_Z_ray = (depth_mean ** 2 / baseline) * math.sqrt(2) * u_alpha_ray

        pct_pin = 100 * (2 * u_Z_pin / depth_mean)
        pct_ray = 100 * (2 * u_Z_ray / depth_mean)
        gain = pct_pin / pct_ray if pct_ray != 0 else float("inf")

        rows.append(
            {
                "condition": label,
                "u_alpha_pin": u_alpha_pin * 1000.0,
                "u_alpha_ray": u_alpha_ray * 1000.0,
                "U_rel_pin": pct_pin,
                "U_rel_ray": pct_ray,
                "gain": gain,
            }
        )
    return rows


def format_table(rows):
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Condition & $u_{\\alpha,\\mathrm{pinhole}}$ (mrad) & $u_{\\alpha,\\mathrm{ray}}$ (mrad) & $U_Z/Z$ pinhole (\\%) & $U_Z/Z$ ray (\\%) & Gain (×) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['condition']} & {row['u_alpha_pin']:.2f} & {row['u_alpha_ray']:.2f} & {row['U_rel_pin']:.2f} & {row['U_rel_ray']:.2f} & {row['gain']:.2f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Angular and depth uncertainty comparison between the planar-refined pinhole pipeline and the compact ray-field backend.}\\label{tab:uncertainty_comparison}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    rows = compute_uncertainties()
    OUTPUT_TABLE.write_text(format_table(rows))
    print("Generated", OUTPUT_TABLE)
