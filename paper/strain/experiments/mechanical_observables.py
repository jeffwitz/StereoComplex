"""Mechanical-observable analysis for the Strain/JTCAM manuscript.

This module works only from the saved reconstruction arrays produced by
``evaluate.py``. It does not refit any calibration model. Its purpose is to
ask the mechanical question that pointwise depth metrics cannot answer:
how much non-rigid length change is invented by each calibrated reconstruction
during a rigid translation, as a function of gauge length and direction?

Outputs
-------
results/mechanical_observables.json
    Frame-equal RMS apparent strain, linearised strain predicted from the
    rigid-alignment residual field, and longitudinal structure amplitudes.
results/mechanical_observables_maps.npz
    Gauge and rigid-residual fields for representative held-out frames.
figures/gauge_length_dependence.{pdf,png}
figures/structure_prediction.{pdf,png}
figures/pseudo_strain_maps.{pdf,png}
figures/rigid_residual_maps.{pdf,png}

The analysis deliberately removes only a best-fit rigid SE(3) motion. It never
uses a scale fit: any remaining distance change is therefore a strain-like
measurement error, not something absorbed by the alignment.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

GRID_COLS = 15
GRID_ROWS = 11
PITCH_MM = 0.3
KINDS = ("ray", "soloff", "direct")
SERIES = ("calibration", "calibration2")
LABELS = {"ray": "Ray field", "soloff": "Soloff-type", "direct": "Direct polynomial"}


def _cloud(pred: np.ndarray, frames: np.ndarray, ids: np.ndarray, nframes: int) -> np.ndarray:
    out = np.full((nframes, GRID_ROWS * GRID_COLS, 3), np.nan, dtype=float)
    out[frames, ids] = pred
    return out


def _pairs(dx: int, dy: int) -> np.ndarray:
    pairs = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            rr, cc = r + dy, c + dx
            if 0 <= rr < GRID_ROWS and 0 <= cc < GRID_COLS:
                pairs.append((r * GRID_COLS + c, rr * GRID_COLS + cc))
    return np.asarray(pairs, dtype=int)


def _pair_family() -> dict[str, list[dict]]:
    families: dict[str, list[dict]] = {"horizontal": [], "vertical": [], "diag_pos": [], "diag_neg": []}
    for step in range(1, GRID_COLS):
        families["horizontal"].append(
            {"step": step, "dx": step, "dy": 0, "nominal_length_mm": PITCH_MM * step}
        )
    for step in range(1, GRID_ROWS):
        families["vertical"].append(
            {"step": step, "dx": 0, "dy": step, "nominal_length_mm": PITCH_MM * step}
        )
        families["diag_pos"].append(
            {"step": step, "dx": step, "dy": step, "nominal_length_mm": PITCH_MM * step * np.sqrt(2.0)}
        )
        families["diag_neg"].append(
            {"step": step, "dx": step, "dy": -step, "nominal_length_mm": PITCH_MM * step * np.sqrt(2.0)}
        )
    return families


def _rigid_residual(reference: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return residual field after the proper rigid fit reference -> current.

    Points are row vectors. No reflection and no scale are permitted.
    """
    valid = np.isfinite(reference).all(axis=1) & np.isfinite(current).all(axis=1)
    if valid.sum() < 3:
        return np.full_like(current, np.nan), np.eye(3), np.zeros(3)

    a = reference[valid]
    b = current[valid]
    ca = a.mean(axis=0)
    cb = b.mean(axis=0)
    ac = a - ca
    bc = b - cb
    u, _, vt = np.linalg.svd(ac.T @ bc)
    d = np.ones(3)
    d[-1] = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag(d) @ vt
    trans = cb - ca @ rot

    predicted = reference @ rot + trans
    residual = current - predicted
    residual[~valid] = np.nan
    return residual, rot, trans


def _frame_pair_values(
    reference: np.ndarray,
    current: np.ndarray,
    residual: np.ndarray,
    rot: np.ndarray,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact and linearised strain for one frame and one pair family."""
    valid_points = (
        np.isfinite(reference).all(axis=1)
        & np.isfinite(current).all(axis=1)
        & np.isfinite(residual).all(axis=1)
    )
    ok = valid_points[pairs].all(axis=1)
    pp = pairs[ok]
    if len(pp) == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty

    d0 = reference[pp[:, 1]] - reference[pp[:, 0]]
    l0 = np.linalg.norm(d0, axis=1)
    dt = current[pp[:, 1]] - current[pp[:, 0]]
    lt = np.linalg.norm(dt, axis=1)
    good = np.isfinite(l0) & (l0 > 0) & np.isfinite(lt)
    if not np.any(good):
        empty = np.empty(0, dtype=float)
        return empty, empty, empty, empty

    pp = pp[good]
    d0 = d0[good]
    l0 = l0[good]
    lt = lt[good]

    exact = lt / l0 - 1.0
    tangent = (d0 @ rot) / l0[:, None]
    dr = residual[pp[:, 1]] - residual[pp[:, 0]]
    longitudinal = np.einsum("ij,ij->i", tangent, dr)
    linearised = longitudinal / l0
    return exact, linearised, longitudinal, l0


def _frame_equal_rms(per_frame: list[np.ndarray]) -> float:
    values = [np.sqrt(np.mean(v * v)) for v in per_frame if len(v)]
    if not values:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def _frame_equal_bias(per_frame: list[np.ndarray]) -> float:
    values = [np.mean(v) for v in per_frame if len(v)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _analyse_group(
    cloud: np.ndarray,
    reference_frame: int,
    test_frames: np.ndarray,
    pairs: np.ndarray,
) -> dict:
    ref = cloud[reference_frame]
    exact_frames: list[np.ndarray] = []
    linear_frames: list[np.ndarray] = []
    long_frames: list[np.ndarray] = []
    length_frames: list[np.ndarray] = []

    for frame in test_frames:
        if int(frame) == reference_frame:
            continue
        cur = cloud[int(frame)]
        residual, rot, _ = _rigid_residual(ref, cur)
        exact, linear, longitudinal, lengths = _frame_pair_values(ref, cur, residual, rot, pairs)
        if len(exact):
            exact_frames.append(exact)
            linear_frames.append(linear)
            long_frames.append(longitudinal)
            length_frames.append(lengths)

    if not exact_frames:
        return {
            "n_frames": 0,
            "n_pair_frame_samples": 0,
            "exact_rms_microstrain": float("nan"),
            "linearised_rms_microstrain": float("nan"),
            "linearisation_rmse_microstrain": float("nan"),
            "exact_bias_microstrain": float("nan"),
            "longitudinal_structure_rms_um": float("nan"),
            "reference_length_mean_mm": float("nan"),
            "exact_linear_correlation": float("nan"),
        }

    exact_all = np.concatenate(exact_frames)
    linear_all = np.concatenate(linear_frames)
    long_all = np.concatenate(long_frames)
    l0_all = np.concatenate(length_frames)

    corr = np.corrcoef(exact_all, linear_all)[0, 1] if len(exact_all) > 1 else np.nan
    return {
        "n_frames": len(exact_frames),
        "n_pair_frame_samples": int(len(exact_all)),
        "exact_rms_microstrain": 1e6 * _frame_equal_rms(exact_frames),
        "linearised_rms_microstrain": 1e6 * _frame_equal_rms(linear_frames),
        "linearisation_rmse_microstrain": 1e6 * float(np.sqrt(np.mean((exact_all - linear_all) ** 2))),
        "exact_bias_microstrain": 1e6 * _frame_equal_bias(exact_frames),
        "longitudinal_structure_rms_um": 1e3 * float(np.sqrt(np.mean(long_all * long_all))),
        "reference_length_mean_mm": float(np.mean(l0_all)),
        "exact_linear_correlation": float(corr),
    }


def _representative_maps(
    cloud: np.ndarray,
    z: np.ndarray,
    test_frames: np.ndarray,
    reference_frame: int,
    step: int = 4,
) -> dict[str, np.ndarray | int | float]:
    ref = cloud[reference_frame]
    eligible = np.asarray([f for f in test_frames if int(f) != reference_frame], dtype=int)
    frame = int(eligible[np.argmax(np.abs(z[eligible] - z[reference_frame]))])
    cur = cloud[frame]
    residual, rot, trans = _rigid_residual(ref, cur)

    maps: dict[str, np.ndarray | int | float] = {
        "frame": frame,
        "z_mm": float(z[frame]),
        "nominal_dz_mm": float(z[frame] - z[reference_frame]),
        "residual_xyz_mm": residual,
        "rotation": rot,
        "translation_mm": trans,
    }

    for name, dx, dy in (("horizontal", step, 0), ("vertical", 0, step)):
        pairs = _pairs(dx, dy)
        exact, linear, _, _ = _frame_pair_values(ref, cur, residual, rot, pairs)
        valid_points = (
            np.isfinite(ref).all(axis=1)
            & np.isfinite(cur).all(axis=1)
            & np.isfinite(residual).all(axis=1)
        )
        ok = valid_points[pairs].all(axis=1)
        pp = pairs[ok]
        if len(pp) != len(exact):
            raise RuntimeError("Unexpected gauge filtering mismatch.")
        mid = 0.5 * (ref[pp[:, 0]] + ref[pp[:, 1]])
        maps[f"{name}_midpoints_mm"] = mid
        maps[f"{name}_exact_microstrain"] = 1e6 * exact
        maps[f"{name}_linearised_microstrain"] = 1e6 * linear
    return maps


def analyse_file(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    data = np.load(path)
    z = np.asarray(data["z"], dtype=float)
    frames = np.asarray(data["frames"], dtype=int)
    ids = np.asarray(data["ids"], dtype=int)
    test = np.asarray(data["test_indices"], dtype=int)
    ref = int(np.argmin(np.abs(z - 3.0)))
    families = _pair_family()

    summary: dict = {
        "source": path.name,
        "reference_frame": ref,
        "reference_z_mm": float(z[ref]),
        "test_frames": test.tolist(),
        "pitch_mm": PITCH_MM,
        "grid_shape": [GRID_ROWS, GRID_COLS],
        "models": {},
    }
    map_arrays: dict[str, np.ndarray] = {}

    for kind in KINDS:
        cloud = _cloud(np.asarray(data[kind], dtype=float), frames, ids, len(z))
        model_result: dict = {"directions": {}}
        for direction, entries in families.items():
            rows = []
            for entry in entries:
                pairs = _pairs(entry["dx"], entry["dy"])
                result = _analyse_group(cloud, ref, test, pairs)
                rows.append({**entry, **result})
            model_result["directions"][direction] = rows

        maps = _representative_maps(cloud, z, test, ref, step=4)
        model_result["representative_frame"] = {
            "frame": int(maps["frame"]),
            "z_mm": float(maps["z_mm"]),
            "nominal_dz_mm": float(maps["nominal_dz_mm"]),
        }
        summary["models"][kind] = model_result
        for key, value in maps.items():
            if isinstance(value, np.ndarray):
                map_arrays[f"{kind}_{key}"] = value
        map_arrays[f"{kind}_representative_frame"] = np.asarray([maps["frame"]], dtype=int)
        map_arrays[f"{kind}_representative_z_mm"] = np.asarray([maps["z_mm"]], dtype=float)
        map_arrays[f"{kind}_representative_nominal_dz_mm"] = np.asarray(
            [maps["nominal_dz_mm"]], dtype=float
        )

    map_arrays["z_mm"] = z
    map_arrays["test_indices"] = test
    map_arrays["reference_frame"] = np.asarray([ref], dtype=int)
    return summary, map_arrays


def _axial_curve(summary: dict, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine horizontal and vertical directions at common integer-pitch lengths."""
    h = summary["models"][kind]["directions"]["horizontal"]
    v = summary["models"][kind]["directions"]["vertical"]
    vh = {row["step"]: row for row in h}
    vv = {row["step"]: row for row in v}
    steps = sorted(set(vh).intersection(vv))
    x, exact, linear = [], [], []
    for step in steps:
        a, b = vh[step], vv[step]
        x.append(PITCH_MM * step)
        exact.append(np.sqrt(0.5 * (a["exact_rms_microstrain"] ** 2 + b["exact_rms_microstrain"] ** 2)))
        linear.append(np.sqrt(0.5 * (a["linearised_rms_microstrain"] ** 2 + b["linearised_rms_microstrain"] ** 2)))
    return np.asarray(x), np.asarray(exact), np.asarray(linear)


def _savefig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_figures(all_summary: dict, maps: dict[str, dict[str, np.ndarray]], figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, series in zip(axes, SERIES):
        summary = all_summary[series]
        for kind in KINDS:
            x, exact, _ = _axial_curve(summary, kind)
            ax.plot(x, exact, marker="o", label=LABELS[kind])
        ax.set_title("Wide sweep" if series == "calibration" else "Fine sweep")
        ax.set_xlabel("Virtual gauge length (mm)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"Rigid-motion apparent strain RMS ($\mu\varepsilon$)")
    axes[1].legend()
    fig.suptitle("Calibration ranking as a function of mechanical gauge length")
    _savefig(fig, figures / "gauge_length_dependence")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, series in zip(axes, SERIES):
        summary = all_summary[series]
        for kind in KINDS:
            x, exact, linear = _axial_curve(summary, kind)
            ax.plot(x, exact, marker="o", label=f"{LABELS[kind]} exact")
            ax.plot(x, linear, linestyle="--", label=f"{LABELS[kind]} linearised")
        ax.set_title("Wide sweep" if series == "calibration" else "Fine sweep")
        ax.set_xlabel("Virtual gauge length (mm)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"Apparent strain RMS ($\mu\varepsilon$)")
    axes[1].legend(fontsize=7)
    fig.suptitle("Measured gauge error and prediction from the rigid-residual structure")
    _savefig(fig, figures / "structure_prediction")

    series = "calibration2"
    mapset = maps[series]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), sharex=True, sharey=True)
    values = []
    for kind in KINDS:
        for direction in ("horizontal", "vertical"):
            values.extend(mapset[f"{kind}_{direction}_exact_microstrain"].tolist())
    vmax = float(np.quantile(np.abs(values), 0.98)) if values else 1.0
    for col, kind in enumerate(KINDS):
        for row, direction in enumerate(("horizontal", "vertical")):
            pts = mapset[f"{kind}_{direction}_midpoints_mm"]
            val = mapset[f"{kind}_{direction}_exact_microstrain"]
            sc = axes[row, col].scatter(pts[:, 0], pts[:, 1], c=val, s=18, vmin=-vmax, vmax=vmax, cmap="coolwarm")
            axes[row, col].set_aspect("equal", adjustable="box")
            axes[row, col].set_title(LABELS[kind] if row == 0 else "")
            if col == 0:
                axes[row, col].set_ylabel(("Horizontal" if row == 0 else "Vertical") + "\nY (mm)")
            if row == 1:
                axes[row, col].set_xlabel("X (mm)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label=r"Apparent strain ($\mu\varepsilon$)", shrink=0.82)
    frame = int(mapset["ray_representative_frame"][0])
    dz = float(mapset["ray_representative_nominal_dz_mm"][0])
    fig.suptitle(f"Fine sweep: spatial structure of 1.2 mm gauge error (frame {frame}, nominal Δz={dz:.4f} mm)")
    _savefig(fig, figures / "pseudo_strain_maps")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    values = np.concatenate([mapset[f"{kind}_residual_xyz_mm"][:, 2][np.isfinite(mapset[f"{kind}_residual_xyz_mm"][:, 2])] for kind in KINDS])
    vmax = float(np.quantile(np.abs(values), 0.98)) if len(values) else 1.0
    for ax, kind in zip(axes, KINDS):
        residual = mapset[f"{kind}_residual_xyz_mm"]
        xx, yy = np.meshgrid(np.arange(GRID_COLS) * PITCH_MM, np.arange(GRID_ROWS) * PITCH_MM)
        val = residual[:, 2].reshape(GRID_ROWS, GRID_COLS)
        sc = ax.scatter(xx.ravel(), yy.ravel(), c=val.ravel() * 1e3, s=22, cmap="coolwarm",
                        vmin=-vmax * 1e3, vmax=vmax * 1e3)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(LABELS[kind])
        ax.set_xlabel("Nominal X (mm)")
    axes[0].set_ylabel("Nominal Y (mm)")
    fig.colorbar(sc, ax=axes.ravel().tolist(), label=r"Residual $u_z$ after rigid fit ($\mu$m)", shrink=0.82)
    fig.suptitle("Fine sweep: non-rigid reconstruction residual after best-fit SE(3)")
    _savefig(fig, figures / "rigid_residual_maps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("paper/strain/results"))
    parser.add_argument("--figures", type=Path, default=Path("paper/strain/figures"))
    args = parser.parse_args()

    summaries: dict[str, dict] = {}
    all_maps: dict[str, dict[str, np.ndarray]] = {}
    merged_npz: dict[str, np.ndarray] = {}

    for series in SERIES:
        summary, maps = analyse_file(args.results / f"predictions_{series}.npz")
        summaries[series] = summary
        all_maps[series] = maps
        for key, value in maps.items():
            merged_npz[f"{series}_{key}"] = value

    (args.results / "mechanical_observables.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False)
    )
    np.savez_compressed(args.results / "mechanical_observables_maps.npz", **merged_npz)

    figure_maps = {}
    for series in SERIES:
        figure_maps[series] = {
            key: merged_npz[f"{series}_{key}"]
            for key in all_maps[series]
        }
    make_figures(summaries, figure_maps, args.figures)

    for series in SERIES:
        tag = "wide" if series == "calibration" else "fine"
        print(f"[{tag}]")
        for kind in KINDS:
            rows = summaries[series]["models"][kind]["directions"]["horizontal"]
            row = next(r for r in rows if r["step"] == 4)
            print(
                f"  {kind:7s} L=1.2 mm: exact={row['exact_rms_microstrain']:.1f} µε, "
                f"linear={row['linearised_rms_microstrain']:.1f} µε, "
                f"corr={row['exact_linear_correlation']:.4f}"
            )


if __name__ == "__main__":
    main()
