#!/usr/bin/env python3
"""Generate figures, tables and numerical macros from the recorded audit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT=Path(__file__).resolve().parents[3]
PAPER=ROOT/'paper/cmo'
NAMES={'compact_14':'Shared affine (14)', 'compact_26':'Shared affine + arms (26)',
       'central_projective_22':'Independent central (22)'}


def text_assets():
    r=json.loads((PAPER/'results/diagnostic_audit.json').read_text())
    v=json.loads((ROOT/'docs/assets/pycaso_real_data/validation_experiments.json').read_text())
    macros={}
    for key,prefix in [('compact_14','Shared'),('compact_26','Aligned'),('central_projective_22','Central')]:
        row=r['models'][key]
        for suffix,val in [('Plane',row['reserved']['plane_xy_rms_um']),
                           ('Angle',row['reserved']['angular_rms_deg']),
                           ('Px',row['inverse_pixel_rms'])]:
            macros[prefix+suffix]=f'{val:.4f}' if suffix=='Angle' else f'{val:.3f}'
    macros['RayPx']=f"{r['models']['reference_field']['inverse_pixel_rms']:.3f}"
    macros['PistonPct']=f"{100*r['models']['compact_14']['reserved']['direction_energy_fraction_constant_linear_quadratic_remainder'][0]:.1f}"
    macros['CoinDifference']=f"{r['coin']['median_distance_after_SE3_um']:.1f}"
    macros['CoinRayMAD']=f"{r['coin']['detrended_height_MAD_um'][0]:.1f}"
    macros['CoinCompactMAD']=f"{r['coin']['detrended_height_MAD_um'][1]:.1f}"
    for case,prefix in [('rigid','Rigid'),('rigid_and_quadratic','Quadratic')]:
        for fit,suffix in [('rigid','Rigid'),('rigid_quadratic','Enriched')]:
            macros[prefix+suffix]=f"{r['simulation']['summary'][case][fit]:.3f}"
    products={'tables/diagnostic_numbers.tex':'% Generated; do not edit.\n'+''.join(
        f'\\newcommand{{\\{k}}}{{{value}}}\n' for k,value in macros.items())}
    rows=[]
    for key,label in NAMES.items():
        d=r['models'][key]
        rows.append(f"{label} & {d['train']['plane_xy_rms_um']:.3f} & {d['reserved']['plane_xy_rms_um']:.3f} & {d['reserved']['angular_rms_deg']:.4f} & {d['inverse_pixel_rms']:.3f} \\\\")
    products['tables/diagnostic_comparison.tex']='\n'.join(rows)+'\n'
    orders=json.loads((ROOT/'docs/assets/pycaso_real_data/zernike_order_sweep.json').read_text())
    products['tables/diagnostic_orders.tex']='\n'.join(
        f"{o['O']} & {o['d']} & {o['p']} & {o['rms']:.3f} \\\\" for o in orders)+'\n'
    products['tables/diagnostic_bootstrap.tex']='\n'.join(
        f"{label} & {v['bootstrap'][key][0]:.2f} & {v['bootstrap'][key][1]:.2f} \\\\"
        for key,label in [('b_95ci','$b$ (mm)'),('wd_95ci','$WD$ (mm)'),('theta_95ci',r'$\theta$ (degrees)')])+'\n'
    products['tables/diagnostic_fx.tex']='\n'.join(
        f"{o['fx']} & {o['rms']:.3f} & {o['b']:.2f} & {o['wd']:.2f} & {o['theta']:.2f} \\\\" for o in v['fx_sensitivity'])+'\n'
    # Keep each tabular environment in one input file; booktabs' noalign must
    # immediately follow the row terminator, without an input-file boundary.
    headers={
        'diagnostic_comparison':('lrrrr',r'Model & Training & Reserved & Angle ($^\circ$) & Inverse (px)'),
        'diagnostic_orders':('rrrr','Origin order & Direction order & Total coefficients & Local equivalent (px)'),
        'diagnostic_bootstrap':('lrr','Descriptor & Lower & Upper'),
        'diagnostic_fx':('rrrrr',r'$f_x$ (px) & RMS (px) & $b$ (mm) & $WD$ (mm) & $\theta$ ($^\circ$)'),
    }
    for name,(fmt,header) in headers.items():
        key=f'tables/{name}.tex'
        products[key]='\\begin{tabular}{'+fmt+'}\\toprule\n'+header+'\\\\\\midrule\n'+products[key]+'\\bottomrule\n\\end{tabular}\n'
    return products


def main():
    for rel,content in text_assets().items():
        (PAPER/rel).write_text(content)
    r=json.loads((PAPER/'results/diagnostic_audit.json').read_text())
    m=np.load(PAPER/'results/diagnostic_maps.npz')
    figdir=PAPER/'figures'
    plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    def save(fig,name):
        fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(7.2,4.7),layout='constrained')
    for j,(key,title) in enumerate(NAMES.items()):
        for c in range(2):
            ax=axs[c,j]
            im=ax.imshow(m[key+'_angular'][c].reshape(24,32),origin='lower',extent=[0,2047,0,2047],
                         norm=LogNorm(vmin=1e-5,vmax=.5),cmap='viridis',aspect='equal')
            ax.set_title(title if c==0 else '')
            ax.set_xlabel('u (pixel)');ax.set_ylabel(('Left: ' if c==0 else 'Right: ')+'v (pixel)')
            ax.set_xticks([0,1024,2047]);ax.set_yticks([0,1024,2047])
    fig.colorbar(im,ax=axs,label='Angular mismatch (degrees)',shrink=.8,pad=.02)
    save(fig,'diagnostic_residuals')
    fig,ax=plt.subplots(figsize=(7.2,2.2),layout='constrained')
    labels=['Constant','Linear','Quadratic','Remainder']; colors=['#243f61','#5586a4','#c18a3b','#b6b6b6']
    left=np.zeros(3)
    values=np.array([r['models'][k]['reserved']['direction_energy_fraction_constant_linear_quadratic_remainder'] for k in NAMES])*100
    for j in range(4):
        ax.barh(list(NAMES.values()),values[:,j],left=left,label=labels[j],color=colors[j]);left+=values[:,j]
    ax.set_xlim(0,100);ax.set_xlabel('Fraction of direction-residual energy (%)')
    ax.legend(ncols=4,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.3));ax.invert_yaxis()
    save(fig,'diagnostic_modes')
    fig,axs=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
    records=r['simulation']['records']
    for ax,case,title in zip(axs,['rigid','rigid_and_quadratic'],['Injected arm displacement','Arm displacement + quadratic field']):
        data=[[rec[label+'_reserved_um'] for rec in records if rec['case']==case] for label in ['rigid','rigid_quadratic']]
        ax.boxplot(data,tick_labels=['Rigid correction','Rigid + quadratic'],widths=.5)
        ax.set_yscale('log');ax.set_ylim(.003,15);ax.set_ylabel('Reserved-grid RMS (µm)');ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
    save(fig,'diagnostic_simulation')
    fig,axs=plt.subplots(1,3,figsize=(7.2,2.5),layout='constrained')
    vals=[m['coin_Z_ref'],m['coin_Z_compact'],m['coin_Z_compact']-m['coin_Z_ref']]
    for ax,val,title in zip(axs,vals,['Stored ray-field surface','Stored compact surface','Compact minus ray field']):
        im=ax.scatter(m['coin_X'][::5],m['coin_Y'][::5],c=1000*val[::5],s=1,rasterized=True,cmap='coolwarm',vmin=-80,vmax=80)
        ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('X (mm)');ax.set_ylabel('Y (mm)')
    fig.colorbar(im,ax=axs,label='Detrended height (µm)',shrink=.7,pad=.02)
    save(fig,'diagnostic_coin')


if __name__=='__main__':
    main()
