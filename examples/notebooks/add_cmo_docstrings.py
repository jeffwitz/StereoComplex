#!/usr/bin/env python3
"""Add docstrings to all undocumented public functions in cmo.py — safe bulk approach."""
import ast, os

fp = os.path.expanduser("~/StereoComplex/src/stereocomplex/physics/cmo.py")
with open(fp) as fh:
    content = fh.read()

# First pass: identify all undocumented public functions via AST
tree = ast.parse(content)
missing = []
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not n.name.startswith('_') and n.name != '__init__':
            if not ast.get_docstring(n):
                missing.append((n.lineno, n.name))

print(f"Undocumented: {len(missing)}")
for lineno, name in missing:
    print(f"  L{lineno}: {name}")

# Add docstrings by name — use the function definition as anchor
# Each entry: (function_name, new_first_body_line, docstring_text)
# We find "def name(" and insert docstring after the signature line (the one ending with ':')
import re

docstrings = {
    "apply_blur_noise": """Simulate acquisition defects on a synthetic image.

Parameters
----------
img : np.ndarray
    Input image (grayscale or RGB).
blur_sigma : float
    Gaussian blur sigma in pixels.
noise_std : float
    Additive Gaussian noise standard deviation.
seed : int | None
    RNG seed for reproducibility.

Returns
-------
np.ndarray
    Degraded image.
""",
    "save_gray": """Save a float64 array as an 8-bit grayscale PNG.

Parameters
----------
arr : np.ndarray
    Image data, any shape and dtype.
path : Path
    Output file path.
""",
    "normal_world": """World-frame normal (Z axis) of the calibration plane.

Returns
-------
np.ndarray
    Unit vector (3,), dtype float64.
""",
    "from_focal_and_pitch": """Build intrinsics from focal length and pixel pitch.

The principal point defaults to the image centre.

Parameters
----------
width : int
    Image width in pixels.
height : int
    Image height in pixels.
focal_mm : float
    Focal length in mm.
pixel_pitch_um : float
    Pixel pitch in µm.

Returns
-------
CMOIntrinsics
""",
    "pixel_to_norm": """Convert pixel coordinates to normalised image coordinates.

Parameters
----------
u, v : np.ndarray
    Pixel coordinates.

Returns
-------
(x, y) : tuple of np.ndarray
    Normalised coordinates (unitless).
""",
    "norm_to_pixel": """Convert normalised coordinates to pixel coordinates.

Parameters
----------
x, y : np.ndarray
    Normalised image coordinates (unitless).

Returns
-------
np.ndarray
    Pixel coordinates, shape (N, 2).
""",
    "delta": """Polynomial perturbation (dx, dy) at given normalised coordinates.

Parameters
----------
x, y : np.ndarray
    Normalised image coordinates.

Returns
-------
(dx, dy) : tuple of np.ndarray
    Ray-direction perturbations (unitless).
""",
    "delta_px": """Compute the sensor-plane warp offset in pixels.

Parameters
----------
uv : np.ndarray
    Pixel coordinates, shape (N, 2).
intr : CMOIntrinsics
    Intrinsics used for normalisation of input coordinates.

Returns
-------
np.ndarray
    (du, dv) pixel offset, shape (N, 2).
""",
    "gain": """Pixel-wise vignetting gain (0–1) for the full image.

Parameters
----------
intr : CMOIntrinsics

Returns
-------
np.ndarray
    Gain map, shape (height, width).
""",
    "origin": """Optical centre in world coordinates (mm).

For a telecentric CMO, this is the sub-pupil centre.
""",
    "symmetric_default": """Default stereo spec with symmetric channels centred on screen.

Parameters
----------
width, height : int
    Image dimensions in pixels.
focal_mm : float
    Focal length in mm.
pixel_pitch_um : float
    Pixel pitch in µm.
baseline_mm : float
    Stereo baseline in mm.
working_distance_mm : float
    Distance from objective plane to specimen in mm.

Returns
-------
CMOStereoSpec
""",
    "channels": """All channels in this stereo specification.

Returns
-------
list of CMOChannelSpec
    Left channel first, then right.
""",
    "n_parameters": "Total number of free parameters for model selection.",
    "default_terms": """Parameter names in canonical order for a parameter vector.

This list defines the packing/unpacking convention.
""",
    "parameter_vector": "All parameters packed into a flat vector for optimisation.",
    "from_parameter_vector": "Reconstruct spec from a parameter vector.",
    "parameter_dict": "All parameters as a dict keyed by channel and parameter name.",
    "ray": "Compute a ray (origin, direction) for a given channel and pixel.",
    "parameter_summary": "Human-readable string listing all model parameters with units.",
    "unpack": """Unpack the stereo spec into individual component arrays.

Returns
-------
tuple
    left_origin, left_dir, left_poses, right_origin, right_dir, right_poses, aux
""",
    "residual": """Plücker distance between a predicted ray and a known 3D point.

Used as the scalar objective for single-ray optimisation.

Parameters
----------
arg : dict
    Ray parameter dict (origin, direction).
point_3d_local_mm : np.ndarray
    3D point in world coordinates (mm).

Returns
-------
(distance_mm, jacobian) : tuple
    Residual distance in mm and its gradient.
""",
    "residuals": """Compute ray-space residuals for left and right pixels.

The residual is the Plücker line distance between the rayfield
prediction and the observed ray for each pixel.

Parameters
----------
u_L, v_L : np.ndarray
    Left pixel coordinates.
u_R, v_R : np.ndarray
    Right pixel coordinates.

Returns
-------
np.ndarray
    Ray residuals, shape (N*6,) — 3 per channel.
""",
    "width_mm": "Board width in mm.",
    "height_mm": "Board height in mm.",
    "inner_corners_local_mm": "Inner ChArUco corner positions in board-local mm coordinates.",
    "make_texture_u8": """Generate a synthetic speckle texture matching the board dimensions.

Used to render synthetic ChArUco images for benchmarks.

Parameters
----------
seed : int
    RNG seed for reproducible textures.

Returns
-------
np.ndarray
    uint8 image of the pattern texture.
""",
    "in_image": """Boolean mask: which pixel coordinates lie inside the image.

Parameters
----------
uv : np.ndarray
    Pixel coordinates, shape (N, 2).

Returns
-------
np.ndarray
    Boolean array of shape (N,).
""",
}

# Insert docstrings
fixed = 0
for name, doc in docstrings.items():
    doc_formatted = f'    """{doc}"""\n'
    # Find the def line pattern: "    def name(" or "def name("
    pattern = re.compile(rf'^(\s*)def {name}\(', re.MULTILINE)
    m = pattern.search(content)
    if not m:
        print(f"  ✗ NOT FOUND: def {name}")
        continue
    
    # Find the next line after the function signature
    # The signature may span multiple lines — find the line ending with ':'
    pos = m.end()
    lines = content[pos:].split('\n')
    sig_end = 0
    for i, line in enumerate(lines):
        sig_end += len(line) + 1
        if line.rstrip().endswith(':'):
            break
    
    insert_pos = pos + sig_end
    
    # Check if there's already a docstring
    after = content[insert_pos:insert_pos+50]
    if '"""' in after:
        print(f"  ✓ {name} — already has docstring")
        continue
    
    content = content[:insert_pos] + '\n' + doc_formatted + content[insert_pos:]
    fixed += 1
    print(f"  ✓ {name}")

with open(fp, "w") as fh:
    fh.write(content)

# Verify syntax
tree2 = ast.parse(content)
pub = pud = 0
for n in ast.walk(tree2):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not n.name.startswith('_'):
            pub += 1
            pud += ast.get_docstring(n) is not None

print(f"\nResult: {pud}/{pub} = {100*pud/pub:.1f}% documented")
print(f"Fixed this batch: {fixed}")
