#!/usr/bin/env python3
"""Add docstrings to 3 remaining files."""
import tokenize, io, ast, os

FILES = {
    "benchmarks/rayfield_from_observations.py": {
        "fit_zernike_rayfield_to_observations": "Fit a Zernike rayfield to ChArUco corner observations via bundle adjustment.",
        "fit_origin_field_to_observations": "Fit the origin-field O(u,v) only (fixed directions).",
        "fit_direction_field_to_observations": "Fit direction perturbations on top of existing origin field.",
        "fit_full_field_to_observations": "Jointly fit O(u,v) and d(u,v) in full BA.",
    },
    "eval/predictors/warps.py": {
        "predict_points_affine_field": "Predict board-plane points from an affine field model.",
        "predict_points_homography": "Predict board-plane points from a homography warp.",
        "predict_points_mls_affine": "Moving Least Squares with affine local model.",
        "predict_points_mls_homography": "Moving Least Squares with homography local model.",
        "predict_points_piecewise_affine": "Piecewise affine interpolation on Delaunay triangulation.",
        "predict_points_rayfield": "Predict via smooth rayfield (TPS on board plane).",
    },
    "physics/base.py": {
        "n_parameters": "Number of free parameters for model selection.",
        "parameter_vector": "Pack model parameters into a flat vector for optimisation.",
        "from_parameter_vector": "Reconstruct model from a parameter vector.",
        "parameter_dict": "Model parameters as a dict keyed by coefficient name.",
        "ray": "Compute ray (origin, direction) for a pixel.",
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
        print(f"{rel}: already 100%")
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
