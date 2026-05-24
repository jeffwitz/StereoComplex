#!/usr/bin/env python3
"""Final batch — docstrings for all remaining files."""
import tokenize, io, ast, os

FILES = {
    "benchmarks/parallel_plate_origin_field.py": {
        "make_plate_rayfield": "Factory: create a ParallelPlateRayField from physical parameters.",
        "make_plate_dataset": "Generate a synthetic stereo dataset with a parallel-plate non-central camera.",
        "run_plate_benchmark": "Run the full parallel-plate benchmark: dataset → rayfield fit → reconstruction comparison.",
        "run_rendered_benchmark": "Run the benchmark on rendered images with realistic defects (blur, noise).",
    },
    "benchmarks/model_selection_oracles.py": {
        "make_oracle": "Build a synthetic rayfield oracle for a given physical model family.",
        "run_oracle_sweep": "Run BIC model selection on all oracles against all candidate models.",
        "summarise_sweep": "Produce a summary DataFrame from oracle sweep results.",
        "load_sweep": "Load a previously saved sweep from disk.",
    },
    "benchmarks/rayfield_from_observations.py": {
        "fit_zernike_rayfield_to_observations": "Fit Zernike rayfield to ChArUco observations via BA.",
        "fit_origin_field_to_observations": "Fit origin field O(u,v) only.",
        "fit_direction_field_to_observations": "Fit direction perturbations on top of existing origin field.",
        "fit_full_field_to_observations": "Jointly fit O(u,v) and d(u,v).",
    },
    "eval/pycaso_soloff.py": {
        "fit": "Fit a Soloff polynomial to the Pycaso calibration data.",
        "predict": "Evaluate the fitted Soloff polynomial at query points.",
        "fun": "Residual function for Soloff optimisation.",
    },
    "eval/method_comparison.py": {
        "compare_charuco_methods": "Compare multiple ChArUco detection/refinement methods on a dataset.",
        "write_latex_table": "Write comparison results as a LaTeX booktabs table.",
        "write_report_json": "Write comparison results as a JSON report.",
    },
    "eval/predictors/dispatch.py": {
        "marker_correspondences": "Pair detected ArUco marker corners with board object points.",
        "predict_marker_warp": "Predict board-plane coordinates from a marker warp model.",
        "predict_charuco_points": "Predict ChArUco corner positions from marker correspondences.",
    },
    "synthetic/parallel_plate.py": {
        "plate_ray_from_pixel": "Compute ray (origin, direction) for a pixel through a parallel plate.",
        "pinhole_ray_from_pixel": "Compute ray for a pixel through a pinhole camera.",
        "generate_plate_dataset": "Generate a full synthetic dataset with plate-distorted rays.",
    },
    "core/distortion.py": {
        "distort_pixels": "Apply radial and tangential distortion to pixel coordinates.",
        "undistort_pixels": "Iteratively remove distortion from pixel coordinates.",
        "distortion_coeffs_vector": "Pack distortion coefficients into a flat vector.",
    },
    "core/model_compact/central_rayfield.py": {
        "from_opencv": "Build a central rayfield from OpenCV calibration parameters.",
        "to_opencv": "Export a central rayfield to OpenCV calibration parameters.",
        "compress": "Compress a high-resolution rayfield by Zernike projection.",
    },
    "benchmarks/inverse_problem_diagnostics.py": {
        "compute_coupling_norm": "Schur coupling norm c between pose and ray parameters.",
        "compute_condition_number": "Condition number of the reduced Hessian.",
        "run_diagnostics": "Run full inverse-problem diagnostics on a fitted rayfield.",
    },
    "core/geometry.py": {
        "transform_points": "Apply a 4x4 homogeneous transform to 3D points.",
        "invert_transform": "Invert a 4x4 homogeneous transformation matrix.",
    },
    "eval/detectors/charuco.py": {
        "detect_image_features": "Detect ArUco markers and ChArUco corners in an image.",
    },
    "core/rayfield2d.py": {
        "fit_rayfield_2d": "Fit a 2D planar rayfield (homography + smooth residual).",
        "evaluate_rayfield_2d": "Evaluate the 2D rayfield at query points.",
    },
    "synthetic/parallel_plate_images.py": {
        "render_plate_image": "Render a ChArUco image through a parallel-plate camera model.",
        "render_dataset": "Render a full dataset of plate-distorted ChArUco images.",
    },
}

for rel, DOCS in FILES.items():
    fp = os.path.expanduser(f"~/StereoComplex/src/stereocomplex/{rel}")
    with open(fp) as fh: source = fh.read()
    tree = ast.parse(source)
    needs = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not n.name.startswith('_') and n.name != '__init__':
                if not ast.get_docstring(n) and n.name in DOCS:
                    needs[(n.lineno, n.name)] = n.name
    if not needs:
        continue
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    lines = source.split('\n')
    fixed = 0
    for (dl, nm) in sorted(needs, reverse=True):
        for i, tok in enumerate(tokens):
            if tok.type == tokenize.NAME and tok.string == 'def' and tok.start[0] == dl:
                j = i+1; b = 0
                while j < len(tokens):
                    tk = tokens[j]
                    if tk.type == tokenize.OP:
                        if tk.string in '([{': b += 1
                        elif tk.string in ')]}': b -= 1
                        elif tk.string == ':' and b <= 0:
                            j += 1
                            if j < len(tokens) and tokens[j].type == tokenize.NEWLINE: j += 1
                            if j < len(tokens) and tokens[j].type == tokenize.INDENT:
                                ind = tokens[j].string
                                il = tokens[j].start[0] - 1
                                lines.insert(il, ind + '"""' + DOCS[nm] + '"""')
                                fixed += 1
                            break
                    j += 1
                break
    with open(fp, 'w') as fh: fh.write('\n'.join(lines))
    tree2 = ast.parse('\n'.join(lines))
    pub = pud = 0
    for n in ast.walk(tree2):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not n.name.startswith('_'): pub += 1; pud += ast.get_docstring(n) is not None
    print(f"{rel}: {fixed} added → {pud}/{pub} = {100*pud/pub:.1f}%")
