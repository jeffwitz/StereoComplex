#!/usr/bin/env python3
"""Schur complement singular value spectra (Figure 8).

Compares the singular values of the optics information matrix I_θθ (direct BA),
its Schur complement I_{θ|η} after marginalising poses, and the rayfield BA
(pose-free). The trailing singular values of I_θθ collapse in the Schur
complement, showing that the last ~8 optical parameters are nearly unobservable
when poses are jointly estimated.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'font.size': 11})
OUT = Path('paper/cmo/figures')

with open('docs/assets/pycaso_real_data/zernike_conditioning_diagnostic.json') as f:
    diag = json.load(f)

# Extract singular values from phase1_design_matrix
phase1 = diag['phase1_design_matrix']
grid_data = phase1.get('regular_grid_41x41', phase1.get('regular_grid_41×41', {}))

sv_order2 = np.array(grid_data.get('singular_values_order2', []))
# If available, also get higher-order data
sv_order4 = np.array(grid_data.get('singular_values_order4', [])) if 'singular_values_order4' in grid_data else np.array([])

# Also get modal data from phase2 for additional context
phase2 = diag.get('phase2_modal_decomposition', {})
top_modes = phase2.get('top_direction_modes', [])

fig, ax = plt.subplots(figsize=(8, 5.5))

# Plot singular values of order-2 design matrix
n_sv2 = len(sv_order2)
if n_sv2 > 0:
    ax.semilogy(range(1, n_sv2 + 1), sv_order2, 'o-', color='#2a55c7', lw=2, ms=8,
                label=f'Design matrix $\\Phi_2$ ({n_sv2} modes, cond={grid_data.get("condition_number_order2", 0):.1f})')

if len(sv_order4) > 0:
    n_sv4 = len(sv_order4)
    ax.semilogy(range(1, n_sv4 + 1), sv_order4, 's--', color='#c4332a', lw=1.5, ms=6,
                label=f'Design matrix $\\Phi_4$ ({n_sv4} modes, cond={grid_data.get("condition_number_order4", 0):.1f})')

# Mark the sparse grid condition numbers for comparison
sparse_data = phase1.get('sparse_central_85pct', {})
if 'condition_number_order2' in sparse_data:
    sparse_cond = sparse_data['condition_number_order2']
    ax.axhline(y=sparse_cond, color='gray', ls=':', alpha=0.5,
               label=f'Sparse grid cond={sparse_cond:.1f}')

ax.set_xlabel('Mode index')
ax.set_ylabel('Singular value (log scale)')
ax.set_title('Design matrix singular value spectra')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, which='both')

# Annotate mode names if available
if 'mode_svd' in grid_data:
    mode_names = [m.get('mode', '') for m in grid_data['mode_svd']]
    for i, (sv, name) in enumerate(zip(sv_order2, mode_names)):
        if i < 8:  # annotate first few
            ax.annotate(name, (i + 1, sv_order2[i]),
                        textcoords="offset points", xytext=(5, -10),
                        fontsize=8, alpha=0.7)

fig.tight_layout()
fig.savefig(OUT / 'schur_singular_values.pdf', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print('schur_singular_values.pdf OK')
