#!/usr/bin/env python3
"""BIC vs Zernike order sweep (Figure 3).

Dual-axis plot: BIC (left axis) and pixel RMS (right axis) versus total
Zernike parameters. The BIC minimum is at O(1)+d(3) (93 params), though
O(0)+d(2) (57 params) is within ~10% RMS at 40% fewer parameters and is
selected for physical model construction.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})
OUT = Path('paper/cmo/figures')

with open('docs/assets/pycaso_real_data/zernike_order_sweep.json') as f:
    sweep_data = json.load(f)

entries = sweep_data  # it's a flat array
n_params = [e['p'] for e in entries]
rms_vals = [e['rms'] for e in entries]
labels = [f"O({e['O']})+d({e['d']})" for e in entries]

# Compute BIC: n_params * log(n_obs) + n_obs * log(rms^2)
# n_obs ≈ 10 frames × 165 corners × 2 channels × 2 coords ≈ 6600 residuals
n_obs = 10 * 165 * 2 * 2
bic_vals = [p * np.log(n_obs) + n_obs * np.log(r**2) for p, r in zip(n_params, rms_vals)]

fig, ax1 = plt.subplots(figsize=(9, 6))

color1 = '#2a55c7'
color2 = '#c4332a'

ax1.plot(n_params, bic_vals, 'o-', color=color1, lw=2, ms=8, label='BIC')
ax1.set_xlabel('Total parameters')
ax1.set_ylabel('BIC', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# Mark BIC minimum
bic_min_idx = np.argmin(bic_vals)
ax1.annotate(f"BIC min: {labels[bic_min_idx]}\n({n_params[bic_min_idx]}p, {rms_vals[bic_min_idx]:.2f} px)",
             (n_params[bic_min_idx], bic_vals[bic_min_idx]),
             xytext=(n_params[bic_min_idx] + 10, bic_vals[bic_min_idx] - 50),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, color='black')

# Mark selected O(0)+d(2)
idx_O0d2 = next(i for i, lbl in enumerate(labels) if lbl == 'O(0)+d(2)')
ax1.annotate(f"Selected: {labels[idx_O0d2]}\n({n_params[idx_O0d2]}p, {rms_vals[idx_O0d2]:.2f} px)",
             (n_params[idx_O0d2], bic_vals[idx_O0d2]),
             xytext=(n_params[idx_O0d2] - 15, bic_vals[idx_O0d2] + 80),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, color='black')

# Annotate each point with label
for i, lbl in enumerate(labels):
    ax1.annotate(lbl, (n_params[i], bic_vals[i]),
                 textcoords="offset points", xytext=(0, 10),
                 fontsize=9, ha='center', color=color1)

ax2 = ax1.twinx()
ax2.plot(n_params, rms_vals, 's--', color=color2, lw=2, ms=8, label='Pixel RMS')
ax2.set_ylabel('Pixel RMS (px)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(bottom=0.38, top=0.50)

ax1.set_title('Zernike order selection by BIC')
ax1.grid(alpha=0.2)

lines1, lbls1 = ax1.get_legend_handles_labels()
lines2, lbls2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbls1 + lbls2, loc='upper right', fontsize=9)

fig.tight_layout()
fig.savefig(OUT / 'BIC_vs_order.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('BIC_vs_order.pdf OK')
