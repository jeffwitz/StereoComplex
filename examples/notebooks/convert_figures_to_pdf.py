#!/usr/bin/env python3
"""Regenerate specimen_reconstruction.pdf and zernike_cmo_rigid_removed.pdf"""
import matplotlib, os, json, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.spatial.transform as st

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
out_dir = os.path.expanduser("~/StereoComplex/docs/assets/pycaso_real_data")
fig_dir = os.path.expanduser("~/StereoComplex/paper/cmo/figures")

# ---- specimen_reconstruction.pdf ----
corr = np.load(f'{out_dir}/specimen_correspondences.npz')
cmo = np.load(f'{out_dir}/specimen_reconstruction_cmo26.npz')
zernike = np.load(f'{out_dir}/specimen_reconstruction_zernike.npz')
sz = tuple(corr['image_size'].astype(int))

def to_2d(d):
    m = np.full(sz, np.nan)
    u = np.clip(corr['uL'].astype(int), 0, sz[1]-1); v = np.clip(corr['vL'].astype(int), 0, sz[0]-1)
    m[v, u] = d; return m

Zc = to_2d(cmo['Z'])[300:1700, 300:1700]
Zz_raw = to_2d(zernike['Z'])[300:1700, 300:1700]
Vc = to_2d(cmo['valid'].astype(float))[300:1700, 300:1700] > 0.5
valid = Vc & np.isfinite(Zc) & np.isfinite(Zz_raw)

left_roi = plt.imread(f'{fig_dir}/specimen_left_with_roi.png')
du = corr['uR'] - corr['uL']; dv = corr['vR'] - corr['vL']
disparity = np.sqrt(du**2 + dv**2)
disp_map = to_2d(disparity)[300:1700, 300:1700]
valid_map = Vc.copy()
gap_map = to_2d(cmo['gap'])[300:1700, 300:1700]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0,0].imshow(left_roi); axes[0,0].set_title('Left image + ROI'); axes[0,0].axis('off')
vmax_d = np.nanpercentile(disp_map[valid], 80)
im = axes[0,1].imshow(disp_map, cmap='inferno', vmin=0, vmax=vmax_d)
axes[0,1].set_title('Disparity ||(U,V)|| [px]'); plt.colorbar(im, ax=axes[0,1], fraction=0.046)
axes[0,2].imshow(valid_map, cmap='gray'); axes[0,2].set_title(f'Valid {valid_map.mean()*100:.1f}%'); axes[0,2].axis('off')
vm_z = np.nanpercentile(np.concatenate([Zc[valid], Zz_raw[valid]]), [1,99])
im = axes[1,0].imshow(Zc, cmap='viridis', vmin=vm_z[0], vmax=vm_z[1])
axes[1,0].set_title('CMO 26p Z [mm]'); plt.colorbar(im, ax=axes[1,0], fraction=0.046)
im = axes[1,1].imshow(Zz_raw, cmap='viridis', vmin=vm_z[0], vmax=vm_z[1])
axes[1,1].set_title('Zernike 57p Z [mm]'); plt.colorbar(im, ax=axes[1,1], fraction=0.046)
gap_v = gap_map[valid_map]; gap_v = gap_v[gap_v < 1.0]
xmax = np.percentile(gap_v, 99.5)*1.2
axes[1,2].hist(gap_v, bins=100, color='steelblue', alpha=0.85)
axes[1,2].axvline(np.median(gap_v), color='red', linestyle='--', label=f'med={np.median(gap_v):.4f}')
axes[1,2].set_xlabel('Ray gap [mm]'); axes[1,2].set_xlim(0, xmax)
axes[1,2].legend(fontsize=8); axes[1,2].set_title('Ray gap histogram')
plt.tight_layout()
plt.savefig(f'{fig_dir}/specimen_reconstruction.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('specimen_reconstruction.pdf OK')

# ---- zernike_cmo_rigid_removed.pdf ----
rigid = json.load(open(f'{out_dir}/zernike_cmo_rigid_comparison.json'))
R = st.Rotation.from_rotvec(rigid['kabsch_se3']['rotation_vec']).as_matrix()
t = np.array(rigid['kabsch_se3']['translation'])
Zz_se3 = np.full_like(Zz_raw, np.nan)
for i in range(1400):
    for j in range(1400):
        if Vc[i,j] and np.isfinite(Zz_raw[i,j]):
            Zz_se3[i,j] = -(R @ np.array([j, i, Zz_raw[i,j]]) + t)[2]
valid2 = Vc & np.isfinite(Zc) & np.isfinite(Zz_se3)
yi, xi = np.where(valid2)
Xf = np.column_stack([xi, yi, np.ones_like(xi)])
cc = np.linalg.lstsq(Xf, Zc[valid2], rcond=None)[0]
Zc_norm = Zc - (cc[0]*np.arange(1400)[None,:] + cc[1]*np.arange(1400)[:,None] + cc[2])
cz = np.linalg.lstsq(Xf, Zz_se3[valid2], rcond=None)[0]
Zz_norm = Zz_se3 - (cz[0]*np.arange(1400)[None,:] + cz[1]*np.arange(1400)[:,None] + cz[2])
Zz_plane = cz[0]*np.arange(1400)[None,:] + cz[1]*np.arange(1400)[:,None] + cz[2]
dZ_before = Zz_se3 - Zc; dZ_norm = Zz_norm - Zc_norm; dZ_pre = Zz_raw - Zc

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
z_p2 = np.nanpercentile(Zz_norm[valid2], 2); z_p98 = np.nanpercentile(Zz_norm[valid2], 98)
im = axes[0,0].imshow(Zc_norm, cmap='viridis', vmin=z_p2, vmax=z_p98)
axes[0,0].set_title(f'CMO 26p Z (plane-norm)\nstd={np.nanstd(Zc_norm[valid2]):.4f} mm'); axes[0,0].axis('off')
plt.colorbar(im, ax=axes[0,0], fraction=0.046)
im = axes[0,1].imshow(Zz_norm, cmap='viridis', vmin=z_p2, vmax=z_p98)
axes[0,1].set_title(f'Zernike 57p Z (SE3+plane-norm)\nstd={np.nanstd(Zz_norm[valid2]):.4f} mm'); axes[0,1].axis('off')
plt.colorbar(im, ax=axes[0,1], fraction=0.046)
vdz = np.nanpercentile(np.abs(dZ_before[valid2]), 99)
im = axes[0,2].imshow(dZ_before, cmap='RdBu_r', vmin=-vdz, vmax=vdz)
axes[0,2].set_title(f'dZ raw\nmed={np.nanmedian(np.abs(dZ_before[valid2])):.3f} mm'); axes[0,2].axis('off')
plt.colorbar(im, ax=axes[0,2], fraction=0.046)
vdz_n = np.nanpercentile(np.abs(dZ_norm[valid2]), 99)
im = axes[0,3].imshow(dZ_norm, cmap='RdBu_r', vmin=-vdz_n, vmax=vdz_n)
axes[0,3].set_title(f'dZ plane-norm\nmed={np.nanmedian(np.abs(dZ_norm[valid2])):.3f} mm'); axes[0,3].axis('off')
plt.colorbar(im, ax=axes[0,3], fraction=0.046)
vdz_p = np.nanpercentile(np.abs(dZ_pre[valid2]), 99)
im = axes[1,0].imshow(dZ_pre, cmap='RdBu_r', vmin=-vdz_p, vmax=vdz_p)
axes[1,0].set_title(f'dZ before SE(3)\nR²={rigid["affine_plane"]["before_se3"]["r2"]:.3f}'); axes[1,0].axis('off')
plt.colorbar(im, ax=axes[1,0], fraction=0.046)
im = axes[1,1].imshow(Zz_plane, cmap='viridis'); axes[1,1].set_title('Affine plane'); axes[1,1].axis('off')
plt.colorbar(im, ax=axes[1,1], fraction=0.046)
dPr = np.abs(dZ_pre[valid2]); dPo = np.sqrt((dZ_norm[valid2])**2)
axes[1,2].hist(dPr, bins=100, alpha=0.6, label=f'before (med={np.median(dPr):.2f})', color='red')
axes[1,2].hist(dPo, bins=100, alpha=0.6, label=f'after (med={np.median(dPo):.3f})', color='green')
axes[1,2].set_xlabel('3D residual [mm]'); axes[1,2].set_xlim(0, 1)
axes[1,2].legend(fontsize=8)
scale_ratio = np.nanstd(Zz_norm[valid2])/np.nanstd(Zc_norm[valid2])
axes[1,3].text(0.5, 0.8, 'CASE A', ha='center', va='center', fontsize=20, fontweight='bold', transform=axes[1,3].transAxes)
axes[1,3].text(0.5, 0.55, f'Global frame change\n{rigid["kabsch_se3"]["rotation_angle_deg"]:.1f}° rotation\nResidual: {rigid["kabsch_se3"]["median_dP_before_mm"]:.1f}->{np.median(dPo):.2f} mm\nPlane R²={rigid["affine_plane"]["before_se3"]["r2"]:.3f}', ha='center', va='center', fontsize=10, transform=axes[1,3].transAxes)
axes[1,3].text(0.5, 0.15, f'Zern amp / CMO amp = {scale_ratio:.2f}x', ha='center', va='center', fontsize=11, fontweight='bold', color='darkblue', transform=axes[1,3].transAxes)
axes[1,3].axis('off')
plt.tight_layout()
plt.savefig(f'{fig_dir}/zernike_cmo_rigid_removed.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('zernike_cmo_rigid_removed.pdf OK')

# Update manuscript
tex_path = os.path.expanduser("~/StereoComplex/paper/cmo/manuscript.tex")
with open(tex_path) as f: t = f.read()
t = t.replace('figures/specimen_reconstruction.png', 'figures/specimen_reconstruction.pdf')
t = t.replace('figures/zernike_cmo_rigid_removed.png', 'figures/zernike_cmo_rigid_removed.pdf')
with open(tex_path, 'w') as f: f.write(t)
print('manuscript.tex updated')
