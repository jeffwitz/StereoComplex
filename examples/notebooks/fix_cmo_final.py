#!/usr/bin/env python3
"""Fix remaining docstrings in cmo.py"""
import ast
fp = "/home/jeff/StereoComplex/src/stereocomplex/physics/cmo.py"
with open(fp) as fh: content = fh.read()

# Fix @property normal_world
content = content.replace(
    '    def normal_world(self) -> Array:\n\n    """World-frame normal (Z axis) of the calibration plane.',
    '    def normal_world(self) -> Array:\n        """World-frame normal (Z axis) of the calibration plane.'
)

# Fix as_K
old = '    def as_K(self) -> Array:\n        return np.array('
new = '    def as_K(self) -> Array:\n        """Camera matrix K (3, 3). Layout: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""\n        return np.array('
content = content.replace(old, new)

# Fix pixel_grid
old = '    def pixel_grid(self) -> tuple[Array, Array]:\n        u, v = np.meshgrid('
new = '    def pixel_grid(self) -> tuple[Array, Array]:\n        """Pixel-centre grid (u, v) for the full image."""\n        u, v = np.meshgrid('
content = content.replace(old, new)

# Fix distort
old = '    def distort(self, x: Array, y: Array) -> tuple[Array, Array]:\n        return brown_conrady_distort_normalized('
new = '    def distort(self, x: Array, y: Array) -> tuple[Array, Array]:\n        """Apply Brown-Conrady distortion (radial k1-k3, tangential p1-p2)."""\n        return brown_conrady_distort_normalized('
content = content.replace(old, new)

# Fix undistort
old = '    def undistort(self, xd: Array, yd: Array, iterations: int = 10) -> tuple[Array, Array]:\n        return undistort_brown_normalized('
new = '    def undistort(self, xd: Array, yd: Array, iterations: int = 10) -> tuple[Array, Array]:\n        """Iteratively remove Brown-Conrady distortion."""\n        return undistort_brown_normalized('
content = content.replace(old, new)

# Fix add
old = '    def add(self, other: PolynomialRayAberration) -> PolynomialRayAberration:\n        if self.level != other.level:'
new = '    def add(self, other: PolynomialRayAberration) -> PolynomialRayAberration:\n        """Add coefficients of another aberration model (same level required)."""\n        if self.level != other.level:'
content = content.replace(old, new)

with open(fp, "w") as fh: fh.write(content)
tree = ast.parse(content)

# Count
pub = pud = 0
missing = []
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not n.name.startswith('_'):
            pub += 1
            if ast.get_docstring(n):
                pud += 1
            else:
                missing.append(f"L{n.lineno}: {n.name}")

print(f"cmo.py: {pud}/{pub} = {100*pud/pub:.1f}%")
if missing:
    print(f"Still missing ({len(missing)}):")
    for m in missing: print(f"  {m}")
else:
    print("ALL DOCUMENTED!")
