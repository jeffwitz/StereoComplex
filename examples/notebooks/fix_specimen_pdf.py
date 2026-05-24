#!/usr/bin/env python3
"""Regenerate specimen_reconstruction.pdf with downsampled rasters (numpy-only)."""
import matplotlib, os, numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
out_dir = os.path.expanduser("~/StereoComplex/docs/assets/pycaso_real_data")
fig_dir = os.path.expanduser("~/StereoComplex/paper/cmo/figures")

def downsample(arr, factor=4):
    """Simple block-mean downsampling, handles non-exact shapes."""
    h, w = arr.shape[:2]
    h2, w2 = h // factor, w // factor
    arr = arr[:h2*factor, :w2*factor]
    if arr.ndim == 3:
        return arr.reshape(h2, factor, w2, factor, arr.shape[2]).mean(axis=(1,3))
    return arr.reshape(h2, factor, w2, factor).mean(axis=(1,3))

corr = np.load(f'{out_dir}/specimen_correspondences.npz')
cmo = np.load(f'{out_dir}/specimen_reconstruction_cmo26.npz')
zernike = np.load(f'{out_dir}/specimen_reconstruction_zernike.npz')
sz = tuple(corr['image_size'].astype(int))

def to_2d(d):
    m = np.full(sz, np.nan)
    u = np.clip(corr['uL'].astype(int), 0, sz[1]-1); v = np.clip(corr['vL'].astype(int), 0, sz[0]-1)
    m[v, u] = d; return m

left = plt.imread(f'{fig_dir}/specimen_left_with_roi.png')
Zc = to_2d(cmo['Z'])[300:1700, 300:1700]
Zz = to_2d(zernike['Z'])[300:1700, 300:1700]
Vc = to_2d(cmo['valid'].astype(float))[300:1700, 300:1700] > 0.5
du = corr['uR'] - corr['uL']; dv = corr['vR'] - corr['vL']
disp_map = to_2d(np.sqrt(du**2 + dv**2))[300:1700, 300:1700]
gap_map = to_2d(cmo['gap'])[300:1700, 300:1700]

# Downsample 4x
Zc_s = downsample(Zc, 4); Zz_s = downsample(Zz, 4)
disp_s = downsample(disp_map, 4)
valid_s = downsample(Vc.astype(float), 4) > 0.5
left_s = downsample(left[300:1700, 300:1700], 4)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes[0,0].imshow(left_s); axes[0,0].set_title('Left image + ROI'); axes[0,0].axis('off')
vmax_d = np.nanpercentile(disp_s[valid_s], 80)
im = axes[0,1].imshow(disp_s, cmap='inferno', vmin=0, vmax=vmax_d)
axes[0,1].set_title('Disparity ||(U,V)|| [px]'); plt.colorbar(im, ax=axes[0,1], fraction=0.046)
axes[0,2].imshow(valid_s, cmap='gray'); axes[0,2].set_title(f'Valid {valid_s.mean()*100:.1f}%'); axes[0,2].axis('off')
vm_z = np.nanpercentile(np.concatenate([Zc_s[valid_s], Zz_s[valid_s]]), [1,99])
im = axes[1,0].imshow(Zc_s, cmap='viridis', vmin=vm_z[0], vmax=vm_z[1])
axes[1,0].set_title('CMO 26p Z [mm]'); plt.colorbar(im, ax=axes[1,0], fraction=0.046)
im = axes[1,1].imshow(Zz_s, cmap='viridis', vmin=vm_z[0], vmax=vm_z[1])
axes[1,1].set_title('Zernike 57p Z [mm]'); plt.colorbar(im, ax=axes[1,1], fraction=0.046)
gap_v = gap_map[Vc]; gap_v = gap_v[gap_v < 1.0]
xmax = np.percentile(gap_v, 99.5)*1.2
axes[1,2].hist(gap_v, bins=100, color='steelblue', alpha=0.85)
axes[1,2].axvline(np.median(gap_v), color='red', linestyle='--', label=f'med={np.median(gap_v):.4f}')
axes[1,2].set_xlabel('Ray gap [mm]'); axes[1,2].set_xlim(0, xmax)
axes[1,2].legend(fontsize=8); axes[1,2].set_title('Ray gap histogram')
plt.tight_layout()
out = f'{fig_dir}/specimen_reconstruction.pdf'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'specimen_reconstruction.pdf: {os.path.getsize(out)} bytes')
