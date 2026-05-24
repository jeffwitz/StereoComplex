#!/usr/bin/env python3
"""Add docstrings to all undocumented public functions in cmo.py using AST positions."""
import ast, os

fp = os.path.expanduser("~/StereoComplex/src/stereocomplex/physics/cmo.py")
with open(fp) as fh:
    source = fh.read()

tree = ast.parse(source)

# Docstring database
DOCS = {
    "rotx": '"""Active rotation matrix about the world X axis (right-hand rule)."""',
    "roty": '"""Active rotation matrix about the world Y axis (right-hand rule)."""',
    "rotz": '"""Active rotation matrix about the world Z axis (right-hand rule)."""',
    "local_to_world": '"""Transform 2D board-plane coordinates (mm) to 3D world coordinates (mm)."""',
    "world_to_local": '"""Transform 3D world coordinates back to local board frame."""',
    "normal_world": '        """World-frame normal (Z axis) of the calibration plane."""',
    "from_focal_and_pitch": '        """Build intrinsics from focal length (mm) and pixel pitch (µm)."""',
    "as_K": '        """Camera matrix K (3,3) — [[fx,0,cx],[0,fy,cy],[0,0,1]]."""',
    "pixel_grid": '        """Pixel-centre meshgrid (u,v) for the full image, shape (H,W)."""',
    "pixel_to_norm": '        """Convert pixel coords to normalised image coords (unitless)."""',
    "norm_to_pixel": '        """Convert normalised coords to pixel coords, shape (N,2)."""',
    "distort": '        """Apply Brown-Conrady distortion (k1-k3 radial, p1-p2 tangential)."""',
    "undistort": '        """Iteratively remove Brown-Conrady distortion (10 fixed-point iters)."""',
    "delta": '        """Polynomial ray-direction perturbation (dx,dy) at (x,y)."""',
    "add": '        """Add coefficients of another aberration (same level required)."""',
    "delta_px": '        """Sensor-plane warp offset in pixels at given pixel coordinates."""',
    "gain": '        """Pixel-wise vignetting gain map (0-1) for the full image."""',
    "origin": '        """Optical centre (sub-pupil) in world coordinates (mm)."""',
    "symmetric_default": '        """Default stereo spec: symmetric channels, centred on screen."""',
    "channels": '        """All channels in this stereo spec (left then right)."""',
    "n_parameters": "        \"\"\"Total number of free parameters for model selection.\"\"\"",
    "default_terms": '        """Parameter names in canonical order for packing/unpacking."""',
    "parameter_vector": "        \"\"\"All parameters packed into a flat vector for optimisation.\"\"\"",
    "from_parameter_vector": "        \"\"\"Reconstruct spec from a parameter vector.\"\"\"",
    "parameter_dict": '        """All parameters as a dict keyed by channel and name."""',
    "ray": '        """Compute ray (origin, direction) for a channel and pixel."""',
    "parameter_summary": '        """Human-readable string listing all model parameters with units."""',
    "unpack": '        """Unpack stereo spec into (origins, directions, poses, aux)."""',
    "residual": '    """Plücker distance between a predicted ray and a 3D point (mm)."""',
    "residuals": '        """Ray-space residuals (Plücker) for left and right pixel pairs."""',
    "width_mm": "        \"\"\"Board width in mm.\"\"\"",
    "height_mm": "        \"\"\"Board height in mm.\"\"\"",
    "inner_corners_local_mm": '        """Inner ChArUco corner positions in board-local mm."""',
    "make_texture_u8": '        """Generate synthetic speckle texture for ChArUco board rendering."""',
    "in_image": '        """Boolean mask: which pixel coords lie inside the image bounds."""',
    "apply_blur_noise": '    """Simulate acquisition defects: Gaussian blur + Gaussian noise."""',
    "save_gray": '    """Save a float64 array as 8-bit grayscale PNG."""',
}

# Collect functions needing docstrings
insertions = []
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name.startswith('_') or node.name == '__init__':
            continue
        if ast.get_docstring(node):
            continue
        if node.name not in DOCS:
            continue
        
        # Get position of first statement in body
        if not node.body:
            continue
        first_stmt = node.body[0]
        insert_pos = first_stmt.col_offset  # not useful for insertion
        
        # Use lineno to find the position in source
        # The docstring goes right after the signature and before the first body statement
        # Find the line of the first body statement and insert before it
        insertions.append((node.name, first_stmt.lineno, DOCS[node.name]))

# Sort by line number (descending so inserts don't shift positions)
insertions.sort(key=lambda x: -x[1])

lines = source.split('\n')
for name, lineno, doc in insertions:
    idx = lineno - 1  # 0-indexed
    # Find the indent level of the line we're inserting before
    if idx < len(lines):
        before_line = lines[idx]
        indent = len(before_line) - len(before_line.lstrip())
        # Add one more indent level (4 spaces) for the method body
        doc_indented = ' ' * (indent) + doc
        lines.insert(idx, doc_indented)
        print(f"  ✓ L{lineno}: {name}")

new_source = '\n'.join(lines)

with open(fp, "w") as fh:
    fh.write(new_source)

# Verify syntax and count
tree2 = ast.parse(new_source)
pub = pud = 0
missing = []
for n in ast.walk(tree2):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not n.name.startswith('_'):
            pub += 1
            if ast.get_docstring(n):
                pud += 1
            else:
                missing.append(f"L{n.lineno}: {n.name}")

print(f"\ncmo.py: {pud}/{pub} = {100*pud/pub:.1f}%")
if missing:
    print(f"Still missing ({len(missing)}):")
    for m in missing: print(f"  {m}")
