#!/usr/bin/env python3
"""Batch 5 — 24 docstrings across 7 files."""
import tokenize, io, ast, os

FILES = {
    "benchmarks/parallel_plate_origin_field.py": {
        "make_grid_board": "Build a ChArUco board geometry as a grid of object points.",
        "make_transform": "Create a 4x4 homogeneous transform from rotation vector and translation.",
        "make_default_parallel_plate_dataset": "Generate the default synthetic parallel-plate dataset.",
        "run_parallel_plate_origin_field_benchmark": "Run the full origin-field benchmark end-to-end.",
    },
    "benchmarks/model_selection_oracles.py": {
        "channel_names": "Channel names in insertion order.",
        "n_channels": "Number of channels in the oracle dataset.",
        "field": "Access the rayfield for a named channel.",
        "K": "Camera matrix K (3,3) for a named channel.",
    },
    "benchmarks/rayfield_from_observations.py": {
        "n_channels": "Number of channels in the observation dataset.",
        "residuals": "Compute ray-space residuals for the current BA state.",
        "residuals_reg": "Compute regularised ray-space residuals.",
    },
    "synthetic/parallel_plate.py": {
        "oracle_left_ray_function": "Exact ray function for the left channel (no noise).",
        "oracle_right_ray_function": "Exact ray function for the right channel (no noise).",
        "fun": "Residual function for plate parameter optimisation.",
    },
    "core/distortion.py": {
        "distort": "Apply radial and tangential distortion to normalised coordinates.",
        "brown_from_dict": "Build Brown-Conrady model from a configuration dict.",
        "brown_to_dict": "Export Brown-Conrady model to a configuration dict.",
    },
    "core/model_compact/central_rayfield.py": {
        "default_disk": "Default normalisation disk (radius) for the central rayfield.",
        "ray_directions_cam": "Central ray directions in camera frame for all pixels.",
        "ray_origins_cam_mm": "Central ray origins in camera frame (all at zero for pinhole).",
    },
    "benchmarks/inverse_problem_diagnostics.py": {
        "full_residual": "Full residual vector for the inverse problem diagnostic.",
        "full_res": "Concatenated residual for all observations.",
        "opt_res": "Optimised residual after BA convergence.",
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
