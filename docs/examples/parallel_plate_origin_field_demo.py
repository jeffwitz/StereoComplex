from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import stereocomplex as sc
from stereocomplex.benchmarks.parallel_plate_origin_field import make_default_parallel_plate_dataset
from stereocomplex.calibration.fit_zernike_origin_field import fit_stereo_zernike_origin_field
from stereocomplex.metrics.rayfield_metrics import compare_rayfields_on_planes, intersect_rays_with_z_plane
from stereocomplex.metrics.reconstruction_metrics import (
    compare_3d_reconstruction_with_without_origin_field,
    reconstruct_points_with_parallel_plate_oracle,
    reconstruct_points_central_stereo,
    reconstruct_points_with_origin_fields,
)
from stereocomplex.rayfields.zernike_origin_field import ZernikeOriginFieldConfig
from stereocomplex.synthetic.parallel_plate import transform_points


_PUBLIC_API_NOTE = (
    "Notebook version: examples/notebooks/04_parallel_plate_origin_field.ipynb. "
    "Theory page: docs/PARALLEL_PLATE_ORIGIN_FIELD.md. "
    "This script uses the same experimental API exposed through `import stereocomplex as sc`."
)

assert sc.ParallelPlateSyntheticParams is not None


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "parallel_plate_origin_field"


def _left_camera_truth(dataset):
    pts = []
    for pose in dataset.board_poses:
        pts.append(transform_points(dataset.T_left_world, transform_points(pose, dataset.object_points)))
    return np.concatenate(pts, axis=0)


def _fit_case(noise_std_px: float):
    dataset = make_default_parallel_plate_dataset(noise_std_px=noise_std_px)
    dataset_clean = make_default_parallel_plate_dataset(noise_std_px=0.0)
    config = ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=4)
    fit = fit_stereo_zernike_origin_field(
        observations=dataset,
        K_left=dataset.K_left,
        K_right=dataset.K_right,
        T_right_left_initial=dataset.T_right_left,
        board_poses_initial=dataset.board_poses,
        config_left=config,
        config_right=config,
        regularization=1e-3,
    )
    config_ba = ZernikeOriginFieldConfig(image_size=dataset.image_size, max_order=3)
    fit_ba = fit_stereo_zernike_origin_field(
        observations=dataset,
        K_left=dataset.K_left,
        K_right=dataset.K_right,
        T_right_left_initial=dataset.T_right_left,
        board_poses_initial=dataset.board_poses,
        config_left=config_ba,
        config_right=config_ba,
        optimize_board_poses=True,
        optimize_directions=True,
        optimize_stereo_extrinsics=True,
        regularization=1e-5,
        direction_regularization=1e-2,
        pose_regularization=10.0,
        rig_regularization=100.0,
        max_nfev=100,
    )
    comparison = compare_3d_reconstruction_with_without_origin_field(dataset, None, fit)
    uv_left = np.concatenate(dataset.left_pixels, axis=0)
    uv_right = np.concatenate(dataset.right_pixels, axis=0)
    uv_left_clean = np.concatenate(dataset_clean.left_pixels, axis=0)
    uv_right_clean = np.concatenate(dataset_clean.right_pixels, axis=0)
    truth = _left_camera_truth(dataset)
    central = reconstruct_points_central_stereo(
        uv_left,
        uv_right,
        dataset.K_left,
        dataset.K_right,
        dataset.T_right_left,
    )
    origin = reconstruct_points_with_origin_fields(
        uv_left,
        uv_right,
        fit.left_field,
        fit.right_field,
        dataset.T_right_left,
    )
    full_ba = reconstruct_points_with_origin_fields(
        uv_left,
        uv_right,
        fit_ba.left_field,
        fit_ba.right_field,
        fit_ba.stereo_transform,
    )
    oracle_clean = reconstruct_points_with_parallel_plate_oracle(
        uv_left_clean,
        uv_right_clean,
        dataset,
    )
    oracle_observed = reconstruct_points_with_parallel_plate_oracle(
        uv_left,
        uv_right,
        dataset,
    )
    left_ray = compare_rayfields_on_planes(
        fit.left_field,
        dataset.oracle_left_ray_function,
        dataset.image_size,
        z_planes=(100.0, 1000.0),
    )
    right_ray = compare_rayfields_on_planes(
        fit.right_field,
        dataset.oracle_right_ray_function,
        dataset.image_size,
        z_planes=(100.0, 1000.0),
    )
    return {
        "dataset": dataset,
        "fit": fit,
        "fit_ba": fit_ba,
        "comparison": comparison,
        "truth": truth,
        "central": central,
        "oracle_clean": oracle_clean,
        "oracle_observed": oracle_observed,
        "origin": origin,
        "full_ba": full_ba,
        "left_ray": left_ray,
        "right_ray": right_ray,
    }


def _norm_errors(result, truth):
    return np.linalg.norm(result.points_3d - truth, axis=1)


def _plot_error_distributions(cases):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, (title, case) in zip(axes, cases.items(), strict=True):
        central_err = _norm_errors(case["central"], case["truth"])
        oracle_err = _norm_errors(case["oracle_observed"], case["truth"])
        origin_err = _norm_errors(case["origin"], case["truth"])
        full_ba_err = _norm_errors(case["full_ba"], case["truth"])
        data = [central_err, oracle_err, origin_err, full_ba_err]
        parts = ax.violinplot(data, showmeans=True, showextrema=True)
        for body, color in zip(parts["bodies"], ["#d55e00", "#009e73", "#0072b2", "#cc79a7"], strict=True):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("black")
            parts[key].set_linewidth(1.0)
        ax.set_title(title)
        ax.set_xticks(
            [1, 2, 3, 4],
            ["central\nwrong model", "oracle\npixel floor", "fitted\nO(u,v)", "BA\nO+d+poses+rig"],
        )
        ax.set_ylabel("3D error norm (mm)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_yscale("log")
    fig.suptitle("Reconstruction error: central model, oracle floor, O-only fit, and full BA")
    fig.tight_layout()
    fig.savefig(OUT / "reconstruction_error_distributions.png", dpi=180)
    plt.close(fig)


def _plot_depth_error_maps(case, filename):
    truth = case["truth"]
    central_err_z = np.abs(case["central"].points_3d[:, 2] - truth[:, 2])
    oracle_err_z = np.abs(case["oracle_observed"].points_3d[:, 2] - truth[:, 2])
    origin_err_z = np.abs(case["origin"].points_3d[:, 2] - truth[:, 2])
    full_ba_err_z = np.abs(case["full_ba"].points_3d[:, 2] - truth[:, 2])
    vmax = max(float(np.percentile(central_err_z, 98)), 1e-6)
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.5), sharex=True, sharey=True)
    for ax, values, title in [
        (axes[0], central_err_z, "Central stereo"),
        (axes[1], oracle_err_z, "Oracle rayfield"),
        (axes[2], origin_err_z, "Fitted O(u,v)"),
        (axes[3], full_ba_err_z, "BA O+d+poses+rig"),
    ]:
        sc = ax.scatter(truth[:, 0], truth[:, 2], c=values, s=32, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("X in left camera (mm)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Z in left camera (mm)")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("|Z error| (mm)")
    fig.suptitle("Depth error on the synthetic board points")
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plane_error_grid(field, oracle_ray_function, image_size, z_plane=1000.0, grid_shape=(31, 23)):
    width, height = image_size
    nx, ny = grid_shape
    u = np.linspace(0.0, width - 1, nx)
    v = np.linspace(0.0, height - 1, ny)
    uu, vv = np.meshgrid(u, v)
    uf = uu.reshape(-1)
    vf = vv.reshape(-1)
    O_fit, d_fit = field.ray(uf, vf)
    O_true, d_true = oracle_ray_function(uf, vf)
    P_fit = intersect_rays_with_z_plane(O_fit, d_fit, z_plane)
    P_true = intersect_rays_with_z_plane(O_true, d_true, z_plane)
    err = np.linalg.norm(P_fit - P_true, axis=1).reshape(ny, nx)
    return u, v, err


def _plot_rayfield_plane_errors(case, filename):
    dataset = case["dataset"]
    left_u, left_v, left_err = _plane_error_grid(
        case["fit"].left_field,
        dataset.oracle_left_ray_function,
        dataset.image_size,
    )
    right_u, right_v, right_err = _plane_error_grid(
        case["fit"].right_field,
        dataset.oracle_right_ray_function,
        dataset.image_size,
    )
    vmax = max(float(np.percentile(left_err, 98)), float(np.percentile(right_err, 98)), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), sharex=True, sharey=True)
    for ax, u, v, err, title in [
        (axes[0], left_u, left_v, left_err, "Left rayfield"),
        (axes[1], right_u, right_v, right_err, "Right rayfield"),
    ]:
        im = ax.imshow(
            err,
            extent=[u.min(), u.max(), v.max(), v.min()],
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(title)
        ax.set_xlabel("u (px)")
        ax.grid(False)
    axes[0].set_ylabel("v (px)")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("ray intersection error at z=1000 mm (mm)")
    fig.suptitle("Rayfield oracle comparison on a reference plane")
    fig.savefig(OUT / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_ray_gap(cases):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    bins = np.geomspace(1e-6, 1.0, 45)
    for ax, (title, case) in zip(axes, cases.items(), strict=True):
        ax.hist(case["central"].ray_gap + 1e-9, bins=bins, alpha=0.65, label="central", color="#d55e00")
        ax.hist(case["oracle_observed"].ray_gap + 1e-9, bins=bins, alpha=0.65, label="oracle floor", color="#009e73")
        ax.hist(case["origin"].ray_gap + 1e-9, bins=bins, alpha=0.65, label="fitted O(u,v)", color="#0072b2")
        ax.hist(case["full_ba"].ray_gap + 1e-9, bins=bins, alpha=0.65, label="BA O+d+poses+rig", color="#cc79a7")
        ax.set_title(title)
        ax.set_xlabel("ray gap (mm)")
        ax.set_xscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("count")
    fig.suptitle("Stereo ray consistency")
    fig.tight_layout()
    fig.savefig(OUT / "ray_gap_histograms.png", dpi=180)
    plt.close(fig)


def _rendered_payload(report):
    central = report.reconstruction_comparison.central
    oracle = report.oracle_detected
    fitted = report.reconstruction_comparison.with_origin_field
    return {
        "method2d": str(report.method2d),
        "n_frames": int(report.n_frames),
        "n_common_corners": int(report.n_common_corners),
        "n_points_total": int(report.n_points_total),
        "fit_success": bool(report.fit_result.success),
        "fit_residual_rms_mm": float(report.fit_result.residual_rms),
        "fit_residual_p95_mm": float(report.fit_result.residual_p95),
        "central_rms_3d_mm": float(central.rms_3d),
        "central_p95_3d_mm": float(central.p95_3d),
        "oracle_detected_rms_3d_mm": float(oracle.rms_3d),
        "oracle_detected_p95_3d_mm": float(oracle.p95_3d),
        "oracle_detected_ray_gap_rms_mm": float(oracle.ray_gap_rms),
        "ba_rms_3d_mm": float(fitted.rms_3d),
        "ba_p95_3d_mm": float(fitted.p95_3d),
        "ba_ray_gap_rms_mm": float(fitted.ray_gap_rms),
        "improvement_rms_factor": float(report.reconstruction_comparison.improvement_rms_factor),
    }


def _plot_rendered_image_benchmarks(reports):
    out = OUT / "rendered_image_ba"
    out.mkdir(parents=True, exist_ok=True)

    raw_report = reports["raw"]
    img_l = plt.imread(raw_report.rendered.left_images[0])
    img_r = plt.imread(raw_report.rendered.right_images[0])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharex=True, sharey=True)
    for ax, img, title in [
        (axes[0], img_l, "Rendered left image"),
        (axes[1], img_r, "Rendered right image"),
    ]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1 if img.dtype.kind == "f" else 255)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Non-central ChArUco render with vignetting, blur and noise")
    fig.tight_layout()
    fig.savefig(out / "rendered_pair.png", dpi=180)
    plt.close(fig)

    labels = ["OpenCV raw", "Ray2D refined"]
    central_values = [reports["raw"].reconstruction_comparison.central.rms_3d, reports["ray2d"].reconstruction_comparison.central.rms_3d]
    oracle_values = [reports["raw"].oracle_detected.rms_3d, reports["ray2d"].oracle_detected.rms_3d]
    ba_values = [
        reports["raw"].reconstruction_comparison.with_origin_field.rms_3d,
        reports["ray2d"].reconstruction_comparison.with_origin_field.rms_3d,
    ]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = [
        ax.bar(x - width, central_values, width, label="central", color="#d55e00", edgecolor="black", alpha=0.85),
        ax.bar(x, oracle_values, width, label="oracle detected", color="#009e73", edgecolor="black", alpha=0.85),
        ax.bar(x + width, ba_values, width, label="BA O+d+poses+rig", color="#cc79a7", edgecolor="black", alpha=0.85),
    ]
    ax.set_ylabel("3D RMS error (mm)")
    ax.set_title("Rendered ChArUco detections: raw OpenCV vs Ray2D-refined front-end")
    ax.set_xticks(x, labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ax.set_ylim(0.0, max([*central_values, *oracle_values, *ba_values]) * 1.22)
    for group in bars:
        for rect in group:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, h, f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "detected_image_ba_rms.png", dpi=180)
    plt.close(fig)

    payload = {name: _rendered_payload(report) for name, report in reports.items()}
    (out / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _summary(cases):
    out = {}
    for name, case in cases.items():
        comp = case["comparison"]
        out[name] = {
            "central_rms_3d_mm": comp.central.rms_3d,
            "central_median_3d_mm": comp.central.median_3d,
            "central_p95_3d_mm": comp.central.p95_3d,
            "central_ray_gap_rms_mm": comp.central.ray_gap_rms,
            "oracle_clean_rms_3d_mm": float(np.sqrt(np.mean(_norm_errors(case["oracle_clean"], case["truth"]) ** 2))),
            "oracle_clean_median_3d_mm": float(np.median(_norm_errors(case["oracle_clean"], case["truth"]))),
            "oracle_clean_p95_3d_mm": float(np.percentile(_norm_errors(case["oracle_clean"], case["truth"]), 95)),
            "oracle_observed_rms_3d_mm": float(np.sqrt(np.mean(_norm_errors(case["oracle_observed"], case["truth"]) ** 2))),
            "oracle_observed_median_3d_mm": float(np.median(_norm_errors(case["oracle_observed"], case["truth"]))),
            "oracle_observed_p95_3d_mm": float(np.percentile(_norm_errors(case["oracle_observed"], case["truth"]), 95)),
            "oracle_observed_ray_gap_rms_mm": float(np.sqrt(np.mean(case["oracle_observed"].ray_gap**2))),
            "origin_rms_3d_mm": comp.with_origin_field.rms_3d,
            "origin_median_3d_mm": comp.with_origin_field.median_3d,
            "origin_p95_3d_mm": comp.with_origin_field.p95_3d,
            "origin_ray_gap_rms_mm": comp.with_origin_field.ray_gap_rms,
            "full_ba_rms_3d_mm": float(np.sqrt(np.mean(_norm_errors(case["full_ba"], case["truth"]) ** 2))),
            "full_ba_median_3d_mm": float(np.median(_norm_errors(case["full_ba"], case["truth"]))),
            "full_ba_p95_3d_mm": float(np.percentile(_norm_errors(case["full_ba"], case["truth"]), 95)),
            "full_ba_ray_gap_rms_mm": float(np.sqrt(np.mean(case["full_ba"].ray_gap**2))),
            "improvement_rms_factor": comp.improvement_rms_factor,
            "improvement_median_factor": comp.improvement_median_factor,
            "improvement_p95_factor": comp.improvement_p95_factor,
            "left_rayfield_plane_rms_mm": case["left_ray"].plane_intersection_rms,
            "right_rayfield_plane_rms_mm": case["right_ray"].plane_intersection_rms,
            "fit_residual_rms_mm": case["fit"].residual_rms,
            "fit_residual_p95_mm": case["fit"].residual_p95,
            "full_ba_residual_rms_mm": case["fit_ba"].residual_rms,
            "full_ba_residual_p95_mm": case["fit_ba"].residual_p95,
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {
        "noise-free oracle": _fit_case(0.0),
        "0.05 px observation noise": _fit_case(0.05),
    }
    _plot_error_distributions(cases)
    _plot_depth_error_maps(cases["noise-free oracle"], "depth_error_map_noise_free.png")
    _plot_depth_error_maps(cases["0.05 px observation noise"], "depth_error_map_noise_005px.png")
    _plot_rayfield_plane_errors(cases["noise-free oracle"], "rayfield_plane_error_noise_free.png")
    _plot_rayfield_plane_errors(cases["0.05 px observation noise"], "rayfield_plane_error_noise_005px.png")
    _plot_ray_gap(cases)
    summary = _summary(cases)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rendered_reports = {
        "raw": sc.run_parallel_plate_rendered_image_benchmark(
            OUT / "rendered_image_ba" / "images_raw",
            max_order=3,
            method2d="raw",
        ),
        "ray2d": sc.run_parallel_plate_rendered_image_benchmark(
            OUT / "rendered_image_ba" / "images_ray2d",
            max_order=3,
            method2d="rayfield_tps_robust",
        ),
    }
    rendered_summary = _plot_rendered_image_benchmarks(rendered_reports)
    print(json.dumps({"geometric": summary, "rendered_image_ba": rendered_summary}, indent=2))


if __name__ == "__main__":
    main()
