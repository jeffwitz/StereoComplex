#!/usr/bin/env python3
"""Final batch — all remaining undocumented public functions."""
import tokenize, io, ast, os

DOCS_ALL = {
    "calibration/fit_zernike_origin_field.py": {
        "unpack": "Unpack the parameter vector into origin, direction, poses, and rig components.",
        "make_fields": "Build left and right ZernikeRayField from coefficient arrays.",
        "residuals": "Compute ray-space residuals for the current BA state.",
    },
    "meta.py": {
        "load_view_meta": "Load view metadata from a JSON file.",
        "parse_view_meta": "Parse view metadata dict into a structured object.",
    },
    "ray3d/central_ba.py": {
        "fun": "Residual function for central rayfield bundle adjustment.",
    },
    "ray3d/central_stereo_ba.py": {
        "fun": "Residual function for central stereo bundle adjustment.",
    },
    "eval/soloff_poly.py": {
        "fit": "Fit a Soloff polynomial to calibration data.",
        "predict": "Evaluate the fitted polynomial at query points.",
    },
    "sim/cpu/generate_dataset.py": {
        "generate_cpu_dataset": "Generate a synthetic ChArUco dataset with CPU rendering.",
        "sample_one": "Draw a single random sample from the dataset generator.",
    },
    "core/rayfield2d.py": {
        "U": "Thin-plate spline kernel: U(r) = r^2 log(r^2).",
        "proj": "Project 2D points using the fitted planar rayfield.",
    },
    "benchmarks/direct_inversion.py": {
        "make_optical_model_right": "Build the right-channel optical model from parameters.",
        "residuals": "Compute residual between direct inversion and rayfield prediction.",
    },
    "optical_ba/regularized_ba.py": {
        "residual_fun": "Regularised residual function for bundle adjustment.",
    },
    "optical_ba/residuals.py": {
        "n_frames": "Number of frames in the observation set.",
        "n_corners": "Total number of corner observations across all frames.",
    },
    "eval/pycaso_soloff.py": {
        "jac": "Analytical Jacobian of the Soloff polynomial residual.",
    },
    "eval/method_comparison.py": {
        "esc": "Escape special LaTeX characters in a string.",
    },
    "eval/refiners/dispatch.py": {
        "refine_detected_points": "Dispatch to the appropriate corner refinement method.",
    },
    "eval/detectors/charuco.py": {
        "method_requires_markers": "Check whether a detection method needs ArUco marker data.",
    },
    "sim/reencode_dataset.py": {
        "reencode_dataset": "Re-encode a dataset to a different image format.",
    },
    "sim/dataset_validate.py": {
        "validate_dataset": "Validate a dataset against the expected schema.",
    },
    "sim/cpu/effects.py": {
        "fwhm_to_sigma": "Convert FWHM (full width at half maximum) to Gaussian sigma.",
    },
    "sim/patterns/charuco.py": {
        "size_px": "Board dimensions in pixels at the current DPI.",
    },
    "cli/main.py": {
        "main": "CLI entry point for the stereocomplex command-line tool.",
    },
    "core/geometry.py": {
        "ray_directions_cam": "Central ray directions in camera frame for a pixel grid.",
    },
    "core/model_compact/zernike.py": {
        "eval_real_zernike": "Evaluate real-valued Zernike polynomials at (rho, theta).",
    },
    "benchmarks/rayfield_projection.py": {
        "fun": "Residual function minimising the 2D reprojection error.",
    },
}

for rel, DOCS in DOCS_ALL.items():
    fp = os.path.expanduser(f"~/StereoComplex/src/stereocomplex/{rel}")
    if not os.path.exists(fp):
        print(f"{rel}: NOT FOUND")
        continue
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
    print(f"{rel}: {fixed} → {pud}/{pub}")
