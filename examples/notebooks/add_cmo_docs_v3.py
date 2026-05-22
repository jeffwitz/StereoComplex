#!/usr/bin/env python3
"""Add docstrings to cmo.py using tokenize for safe indentation."""
import tokenize, io, ast, os

fp = os.path.expanduser("~/StereoComplex/src/stereocomplex/physics/cmo.py")
with open(fp) as fh: source = fh.read()

DOCS = {
    "rotx": "Active rotation matrix about the world X axis (right-hand rule).",
    "roty": "Active rotation matrix about the world Y axis (right-hand rule).",
    "rotz": "Active rotation matrix about the world Z axis (right-hand rule).",
    "local_to_world": "Transform 2D board-plane coordinates (mm) to 3D world coordinates (mm).",
    "world_to_local": "Transform 3D world coordinates back to local board frame.",
    "normal_world": "World-frame normal (Z axis) of the calibration plane.",
    "from_focal_and_pitch": "Build intrinsics from focal length (mm) and pixel pitch (um).",
    "as_K": "Camera matrix K (3x3).",
    "pixel_grid": "Pixel-centre meshgrid (u,v) for the full image.",
    "pixel_to_norm": "Convert pixel coords to normalised image coords (unitless).",
    "norm_to_pixel": "Convert normalised coords to pixel coords, shape (N,2).",
    "distort": "Apply Brown-Conrady distortion (k1-k3 radial, p1-p2 tangential).",
    "undistort": "Iteratively remove Brown-Conrady distortion.",
    "delta": "Polynomial ray-direction perturbation (dx,dy) at (x,y).",
    "add": "Add coefficients of another aberration (same level required).",
    "delta_px": "Sensor-plane warp offset in pixels at given pixel coordinates.",
    "gain": "Pixel-wise vignetting gain map (0-1) for the full image.",
    "origin": "Optical centre (sub-pupil) in world coordinates (mm).",
    "symmetric_default": "Default stereo spec: symmetric channels, centred on screen.",
    "channels": "All channels in this stereo spec (left then right).",
    "n_parameters": "Total number of free parameters for model selection.",
    "default_terms": "Parameter names in canonical order for packing/unpacking.",
    "parameter_vector": "All parameters packed into a flat vector for optimisation.",
    "from_parameter_vector": "Reconstruct spec from a parameter vector.",
    "parameter_dict": "All parameters as a dict keyed by channel and name.",
    "ray": "Compute ray (origin, direction) for a channel and pixel.",
    "parameter_summary": "Human-readable string listing all model parameters with units.",
    "unpack": "Unpack stereo spec into (origins, directions, poses, aux).",
    "residual": "Plucker distance between a predicted ray and a 3D point (mm).",
    "residuals": "Ray-space residuals (Plucker) for left and right pixel pairs.",
    "width_mm": "Board width in mm.",
    "height_mm": "Board height in mm.",
    "inner_corners_local_mm": "Inner ChArUco corner positions in board-local mm.",
    "make_texture_u8": "Generate synthetic speckle texture for ChArUco board rendering.",
    "in_image": "Boolean mask: which pixel coords lie inside the image bounds.",
    "apply_blur_noise": "Simulate acquisition defects: Gaussian blur + Gaussian noise.",
    "save_gray": "Save a float64 array as 8-bit grayscale PNG.",
}

# Find undocumented functions
tree = ast.parse(source)
needs_doc = {}
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if n.name.startswith("_") or n.name == "__init__":
            continue
        if not ast.get_docstring(n) and n.name in DOCS:
            needs_doc[n.name] = n.lineno

# Tokenize
tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
lines = source.split("\n")

fixed = 0
for name, def_lineno in sorted(needs_doc.items(), key=lambda x: -x[1]):
    for i, tok in enumerate(tokens):
        if tok.type == tokenize.NAME and tok.string == "def" and tok.start[0] == def_lineno:
            j = i + 1
            brackets = 0
            while j < len(tokens):
                t = tokens[j]
                if t.type == tokenize.OP:
                    if t.string in "([{": brackets += 1
                    elif t.string in ")]}": brackets -= 1
                    elif t.string == ":" and brackets <= 0:
                        j += 1
                        if j < len(tokens) and tokens[j].type == tokenize.NEWLINE:
                            j += 1
                        if j < len(tokens) and tokens[j].type == tokenize.INDENT:
                            indent = tokens[j].string
                            insert_line = tokens[j].start[0] - 1
                            doc = DOCS[name]
                            doc_line = indent + '"""' + doc + '"""'
                            lines.insert(insert_line, doc_line)
                            fixed += 1
                        break
                j += 1
            break

new_source = "\n".join(lines)
with open(fp, "w") as fh:
    fh.write(new_source)

tree2 = ast.parse(new_source)
pub = pud = 0
for n in ast.walk(tree2):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not n.name.startswith("_"):
            pub += 1
            pud += ast.get_docstring(n) is not None

print(f"Fixed: {fixed}")
print(f"cmo.py: {pud}/{pub} = {100*pud/pub:.1f}%")
