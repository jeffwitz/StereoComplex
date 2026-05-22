#!/usr/bin/env python3
"""Fix PERF401 — manual list comprehension"""
path = "/home/jeff/StereoComplex/src/stereocomplex/benchmarks/charuco_observation_simulator.py"
with open(path) as f: content = f.read()
old = """    pts = []
    for iy in range(squares_y - 1):
        for ix in range(squares_x - 1):
            pts.append([float(ix) * square_size_mm, float(iy) * square_size_mm, 0.0])
    return np.array(pts, dtype=np.float64)"""
new = """    pts = [[float(ix) * square_size_mm, float(iy) * square_size_mm, 0.0]
           for iy in range(squares_y - 1) for ix in range(squares_x - 1)]
    return np.array(pts, dtype=np.float64)"""
content = content.replace(old, new)
with open(path, 'w') as f: f.write(content)
print("Fixed")
