#!/usr/bin/env python3
"""Safely move common lazy imports to top-level."""
import os, re

SAFE_IMPORTS = [
    'import math', 'import json', 'import time', 'import sys', 'import os',
    'from pathlib import Path',
    'from scipy.optimize import least_squares',
    'from scipy.spatial.transform import Rotation',
    'from scipy.interpolate import',
    'from matplotlib.patches import',
    'import matplotlib', 'import matplotlib.pyplot as plt',
    'from stereocomplex.eval.detectors.charuco import detect_image_features',
    'from stereocomplex.eval.predictors.dispatch import predict_charuco_points',
    'from stereocomplex.eval.refiners.dispatch import refine_detected_points',
]

src = "/home/jeff/StereoComplex/src/stereocomplex"
fixed_count = 0

# Get list of files from ruff
import subprocess
result = subprocess.run(['ruff', 'check', src, '--select', 'PLC0415'],
                       capture_output=True, text=True)
files = set()
for line in result.stdout.split('\n'):
    m = re.search(r'--> (src/stereocomplex/.*?\.py):(\d+)', line)
    if m:
        files.add((m.group(1), int(m.group(2))))

for filepath, lineno in sorted(files, key=lambda x: (-x[1], x[0])):
    if '/api/' in filepath:
        continue
    fullpath = f"/home/jeff/StereoComplex/{filepath.replace('src/stereocomplex/', 'src/stereocomplex/')}"
    if not os.path.exists(fullpath.replace('src/stereocomplex/', 'src/stereocomplex/')):
        # Fix path
        fullpath = os.path.join("/home/jeff/StereoComplex", filepath.replace('src/', ''))
    
    with open(fullpath) as f:
        lines = f.readlines()
    
    # Get the offending line
    if lineno > len(lines):
        continue
    offending = lines[lineno-1].strip()
    
    # Check if it's a safe import to move
    is_safe = any(offending.startswith(si) for si in SAFE_IMPORTS)
    if not is_safe:
        continue
    
    # Check if already at top level
    already_top = any(l.strip() == offending for l in lines[:10])
    if already_top:
        # Remove inline duplicate
        lines[lineno-1] = ''
        fixed_count += 1
        with open(fullpath, 'w') as f:
            f.writelines(lines)
        continue
    
    # Insert at top after existing imports
    insert_at = 0
    for i, l in enumerate(lines):
        if l.startswith('import ') or l.startswith('from '):
            insert_at = i + 1
        if l.startswith('#') or l.strip() == '':
            continue
    
    lines.insert(insert_at, offending + '\n')
    # Remove the inline import (shifted by 1 due to insertion)
    del lines[lineno]  # lineno is 1-indexed, lines is 0-indexed after insertion
    
    fixed_count += 1
    with open(fullpath, 'w') as f:
        f.writelines(lines)
    print(f"  Fixed {filepath}:{lineno} → {offending[:60]}")

print(f"Total fixed: {fixed_count}")
