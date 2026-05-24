#!/usr/bin/env python3
"""Batch docstrings for remaining files."""
import tokenize, io, ast, os

FILES = {
    "eval/predictors/warps.py": {
        "lerp": "Linear interpolation between two values or arrays.",
        "node_index": "Return the index of the node associated with a point.",
        "weights_for_points": "Compute interpolation weights for query points.",
        "proj": "Project 2D points using a fitted warp model.",
        "U": "Matrix of polynomial basis functions evaluated at query points.",
    },
    "cli/refine_corners.py": {
        "load_json": "Load and parse a JSON file with error handling.",
        "load_frames": "Load frame metadata from a ChArUco frames file.",
        "build_charuco_from_meta": "Build a CharucoBoardSpec from metadata dictionary.",
        "detect_view": "Detect and refine ChArUco corners for a single view.",
        "refine_dataset_scene": "Refine corners for all frames in a dataset scene.",
        "run_refine_corners": "CLI entry point for the refine-corners subcommand.",
    },
    "core/pinhole_fit.py": {
        "K": "Camera matrix K (3,3) from intrinsics.",
        "dist": "Distortion coefficients as a flat vector.",
        "distortion": "Distortion coefficients as a flat vector (alias).",
        "project_brown_pinhole": "Project 3D points to pixels with Brown distortion.",
        "fun": "Residual function for pinhole parameter optimisation.",
    },
    "benchmarks/charuco_observation_simulator.py": {
        "to_multi_camera": "Convert stereo observations to multi-camera format.",
        "channel_names": "Channel names in insertion order.",
        "n_channels": "Total number of channels in the observations.",
        "n_poses": "Number of board poses captured.",
        "pixels": "Pixel coordinates for a given channel and pose.",
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
