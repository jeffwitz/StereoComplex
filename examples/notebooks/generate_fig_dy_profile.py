#!/usr/bin/env python3
"""d_y profile comparison across the sensor (Figure 5).

Compares the horizontal ray-direction component d_y(u,v) along the
sensor centre column for three models: Zernike rayfield (measured),
telecentric CMO, and perspective CMO.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})
OUT = Path('paper/cmo/figures')

with open('docs/assets/pycaso_real_data/dy_profile_data.json') as f:
    dd = json.load(f)

v_px = np.array(dd['v_px'])
zernike = np.array(dd['zernike'])
telecentric = np.array(dd['telecentric'])
perspective = np.array(dd['perspective'])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(v_px, zernike, 'ko-', label='Zernike (measured)', ms=6)
ax.plot(v_px, telecentric, 's--', color='darkgreen', label='Telecentric CMO', ms=7)
ax.plot(v_px, perspective, '^:', color='darkred', label='Perspective CMO', ms=7)
ax.axhline(y=0, color='gray', ls='--', alpha=0.3)
ax.set_xlabel('v (px)')
ax.set_ylabel(r'$d_y$')
ax.set_title(r'$d_y(u,v)$ profiles across sensor centre column')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

range_z = np.ptp(zernike)
range_t = np.ptp(telecentric)
range_p = np.ptp(perspective)
ax.text(0.98, 0.98, f'Range: Zernike={range_z:.3f}, Telecentric={range_t:.3f}, Perspective={range_p:.3f}',
        transform=ax.transAxes, ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

fig.tight_layout()
fig.savefig(OUT / 'dy_profile_comparison.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('dy_profile_comparison.pdf OK')
