#!/usr/bin/env python3
"""CMO stereo microscope optical layout diagram (Figure 1).

Reproduces the pedagogical 2D optical-cut diagram showing the shared
common main objective, two off-axis sub-pupils, tube lenses, sensors,
and chief-ray paths converging at the object plane.
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')
from stereocomplex.viz.figures import diagram_cmo_physical

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})

fig, ax = plt.subplots(figsize=(5.5, 4.0))
diagram_cmo_physical(ax=ax, f_obj=80, working_distance=120, b=8, exaggerated=True)
fig.savefig(Path('paper/cmo/figures/cmo_physical.pdf'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('cmo_physical.pdf OK')
