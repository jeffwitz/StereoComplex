"""Generate all manuscript figures and numeric tables from saved computations."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'results';OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8})
KINDS=['ray','soloff','direct'];COLORS={'ray':'#0072B2','soloff':'#D55E00','direct':'#009E73'}
NAMES={'ray':'Ray field','soloff':'Soloff-type','direct':'Direct polynomial'}
SERIES=['calibration','calibration2'];NAMES_S=['Wide sweep (0.70 mm)','Fine sweep (0.08 mm)']
E={s:json.loads((DATA/f'evaluation_{s}.json').read_text()) for s in SERIES}

def save(fig,name):
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)


fig=plt.figure(figsize=(7.1,4.7),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[3,1])
pycaso=ROOT.parents[2]/'Pycaso/Exemple/Images_example'
for j,s in enumerate(SERIES):
    d=np.load(DATA/f'observations_{s}.npz');ref=int(np.argmin(abs(d['z']-3)))
    files=sorted((pycaso/f'left_{s}').glob('*.png'),key=lambda p:float(p.stem));ax=fig.add_subplot(gs[0,j])
    ax.imshow(Image.open(files[ref]),cmap='gray');ax.set_title(f'({chr(97+j)}) {NAMES_S[j]}');ax.axis('off')
    ax=fig.add_subplot(gs[1,j]);z=d['z'];train=E[s]['train_indices'];test=E[s]['test_indices']
    ax.scatter(z[test],np.zeros(len(test)),s=8,color='#888888',label='91 test planes')
    ax.scatter(z[train],np.ones(len(train)),s=22,color='#0072B2',label='10 calibration planes')
    ax.set_yticks([0,1],['Test','Calibration']);ax.set_ylim(-.5,1.5);ax.set_xlabel('Nominal z (mm)');ax.spines['left'].set_visible(False)
save(fig,'data_protocol')

rows=json.loads((DATA/'synthetic.json').read_text())['runs'];fig,axes=plt.subplots(2,2,figsize=(7.1,4.9),layout='constrained')
styles=[('ray',3,'-','o'),('soloff',332,'--','s'),('direct',3,':','^'),('soloff',3,'-.','x')]
for j,extent in enumerate([.7,.08]):
    for kind,degree,ls,marker in styles:
        label=NAMES[kind]+(' 333' if kind=='soloff' and degree==3 else ' 332' if kind=='soloff' else '')
        for i,key in enumerate(['rmse_xyz_mm','strain_x_rmse_microstrain']):
            ns=[0,.05,.2,.5];means=[];sd=[]
            for n in ns:
                values=[r[key][2]*1000 if i==0 else r[key] for r in rows if r['extent_mm']==extent and r['noise_px']==n and r['model']==kind and r['degree']==degree]
                means.append(np.mean(values));sd.append(np.std(values,ddof=1) if len(values)>1 else 0)
            color='#555555' if kind=='soloff' and degree==3 else COLORS[kind]
            axes[i,j].errorbar(ns,means,yerr=sd,color=color,linestyle=ls,marker=marker,markersize=3,capsize=2,label=label,linewidth=1.2)
    axes[0,j].set_title(NAMES_S[j]);axes[0,j].set_ylabel('Depth RMSE (µm)');axes[1,j].set_ylabel('Gauge-strain RMSE (µε)')
    for i in range(2): axes[i,j].set_xlabel('Coordinate-noise standard deviation (px)');axes[i,j].grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=4)
save(fig,'synthetic')

fig,axes=plt.subplots(2,2,figsize=(7.1,5.1),layout='constrained')
for j,s in enumerate(SERIES):
    d=np.load(DATA/f'predictions_{s}.npz');z=d['z'];f=d['frames'];test=E[s]['test_indices']
    for kind in KINDS:
        means=[np.mean(d[kind][f==v,2]-d['xyz'][f==v,2])*1000 for v in test]
        axes[0,j].plot(z[test],means,color=COLORS[kind],label=NAMES[kind],linewidth=1.4)
        rr=E[s]['models'][kind]['motion']['frames']; axes[1,j].plot([r['z_mm'] for r in rr],[r['gauge_rms_microstrain'] for r in rr],color=COLORS[kind],linewidth=1.4)
    axes[0,j].axhline(0,color='#888',lw=.6);axes[0,j].set_title(NAMES_S[j]);axes[0,j].set_ylabel('Frame-mean depth discrepancy (µm)')
    axes[1,j].axvline(3,color='#999',ls=':',lw=.8);axes[1,j].set_ylabel('Apparent gauge strain, RMS (µε)')
    for i in range(2):axes[i,j].set_xlabel('Nominal z (mm)');axes[i,j].grid(alpha=.15)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3)
save(fig,'real_observables')

def fmt(v,d=2):return f'{v:.{d}f}'
table=[]
for s in SERIES:
    for kind in KINDS:
        m=E[s]['models'][kind];degree='332' if m['degree']==332 else str(m['degree']);px='--' if m['train_pixel_rms'] is None else fmt(m['train_pixel_rms'],3)
        table.append(' & '.join(['Wide' if s=='calibration' else 'Fine',NAMES[kind],degree,str(m['nparams']),px,fmt(1000*m['test']['rmse_xyz_mm'][2]),fmt(1000*m['motion']['reference_motion_rmse_mm']),fmt(m['motion']['gauge_rms_microstrain'],0)])+r' \\')
(ROOT/'table_main.tex').write_text('\n'.join(table)+'\n')
a=json.loads((DATA/'additional.json').read_text());table=[]
for s in SERIES:
    for kind in KINDS:
        m=a[s][kind];table.append(' & '.join(['Wide' if s=='calibration' else 'Fine',NAMES[kind],str(m['train_points']),str(m['spatial_holdout']['n']),fmt(m['spatial_holdout']['rmse_xyz_mm'][2]*1000),fmt(m['timing_median_s']*1000,2)])+r' \\')
(ROOT/'table_additional.tex').write_text('\n'.join(table)+'\n')
summary={'synthetic_runs':len(rows),'images':404,'pairs':202,'real_results':{s:{k:{'depth_um':E[s]['models'][k]['test']['rmse_xyz_mm'][2]*1000,'gauge_microstrain':E[s]['models'][k]['motion']['gauge_rms_microstrain']} for k in KINDS} for s in SERIES}}
(DATA/'article_numbers.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
