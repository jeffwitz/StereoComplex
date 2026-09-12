#!/usr/bin/env python3
"""Reproducible ray-space audit; no new images or physical measurements.

Run from any directory. Fits use identical XY intersections at two planes.
Reserved pixels test compression of the SAME fitted field, not calibration
generalisation. All inverse-pixel scores use archived processed corners.
"""
from pathlib import Path
import hashlib
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from stereocomplex.rayfields.zernike_origin_field import (  # noqa: E402
    ZernikeOriginFieldConfig, ZernikeRayField, ZernikeRayFieldCoefficients,
)
from stereocomplex.physics.cmo_physical import CMOTelecentricStereoModel  # noqa: E402

OUT = ROOT / 'paper/cmo/results'
ASSETS = ROOT / 'docs/assets/pycaso_real_data'
CACHE = ROOT / 'docs/assets/cmo_paper/figure4_subpupil_3d/zernike_rayfield_canonical.npz'
PLANES = (50., 80.)


def unit(d):
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def grid(nx, ny, midpoint=False):
    if midpoint:
        x, y = (np.arange(nx)+.5)/nx*2047, (np.arange(ny)+.5)/ny*2047
    else:
        x, y = np.linspace(0, 2047, nx), np.linspace(0, 2047, ny)
    u, v = np.meshgrid(x, y)
    return np.c_[u.ravel(), v.ravel()]


def intersections(ray, z=PLANES):
    o, d = ray
    z = np.asarray(z)
    if z.ndim == 0 or z.shape == (len(o),):
        return o[:, :2] + ((z-o[:, 2])/d[:, 2])[:, None]*d[:, :2]
    return np.stack([intersections(ray, float(zz)) for zz in z], axis=1)


def transform(ray, x):
    o, d = ray
    R = Rotation.from_rotvec(x[:3]).as_matrix()
    return o @ R.T + x[3:6], d @ R.T


def projective(x, p):
    xy = (p-1023.5)/1023.5
    A = np.c_[np.ones(len(p)), xy]
    d = np.c_[A@x[3:6], A@x[6:9], 1+xy@x[9:11]]
    return np.broadcast_to(x[:3], d.shape), unit(d)


def projective_start(ray, p):
    o, d = ray
    P = np.eye(3)[None] - d[:, :, None]*d[:, None, :]
    centre = np.linalg.solve(P.sum(0), np.einsum('nij,nj->i', P, o))
    xy = (p-1023.5)/1023.5
    A = np.c_[np.ones(len(p)), xy]
    q = d[:, :2]/d[:, 2:]
    D = np.zeros((2*len(p), 8))
    D[0::2, :3], D[1::2, 3:6] = A, A
    D[0::2, 6:], D[1::2, 6:] = -q[:, :1]*xy, -q[:, 1:]*xy
    h = np.linalg.lstsq(D, q.ravel(), rcond=None)[0]
    return np.r_[centre, h]


def compact(x, p, channel):
    m = CMOTelecentricStereoModel.from_parameter_vector(
        x[:14], pixel_pitch_mm=.0055, image_size=(2048, 2048))
    r = m.ray(p[:, 0], p[:, 1], ('left', 'right')[channel])
    return transform(r, x[14+channel*6:20+channel*6]) if len(x)==26 else r


def numerical_projection(fun, pixels, X):
    """Invert the ray intersection at each observed point's own depth.

    Unbounded local Newton inverse; report divergence explicitly. Solving the
    two XY equations gives the ray through X when a local inverse exists.
    """
    p = pixels.copy()
    step = .01
    for _ in range(20):
        base = intersections(fun(p), X[:, 2])
        e = base-X[:, :2]
        Ju = (intersections(fun(p+[step, 0]), X[:, 2])-base)/step
        Jv = (intersections(fun(p+[0, step]), X[:, 2])-base)/step
        J = np.stack([Ju, Jv], axis=2)
        delta = np.linalg.solve(J, e[..., None])[..., 0]
        p -= delta
        if np.max(np.linalg.norm(delta, axis=1)) < 1e-8:
            break
    miss = np.linalg.norm(intersections(fun(p), X[:, 2])-X[:, :2], axis=1)
    return p-pixels, float(miss.max())


def measure(funs, refs, pixels):
    residual = np.stack([intersections(f(pixels))-intersections(r(pixels))
                         for f, r in zip(funs, refs)])
    angular = np.stack([np.degrees(np.arctan2(
        np.linalg.norm(np.cross(f(pixels)[1], r(pixels)[1]), axis=1),
        np.sum(f(pixels)[1]*r(pixels)[1], axis=1))) for f,r in zip(funs, refs)])
    xy=(pixels-1023.5)/1023.5
    A=np.c_[np.ones(len(pixels)),xy,xy**2,xy[:,0]*xy[:,1]]
    Q,_=np.linalg.qr(A)
    energy=np.zeros(4)
    for f,r in zip(funs,refs):
        delta=f(pixels)[1]-r(pixels)[1]
        coef=Q.T@delta
        energy+=np.array([np.sum(coef[:1]**2),np.sum(coef[1:3]**2),
                          np.sum(coef[3:]**2),np.sum((delta-Q@coef)**2)])
    fractions=(energy/max(energy.sum(),1e-300)).tolist()
    return {'plane_xy_rms_um': float(1000*np.sqrt(np.mean(np.sum(residual**2, axis=-1)))),
            'angular_rms_deg': float(np.sqrt(np.mean(angular**2))),
            'direction_energy_fraction_constant_linear_quadratic_remainder':fractions}, angular, residual


def fit(fun, x0, bounds=(-np.inf, np.inf), max_nfev=800):
    return least_squares(fun, x0, bounds=bounds, x_scale='jac',
                         max_nfev=max_nfev, ftol=1e-11, xtol=1e-11, gtol=1e-11)


def main():
    start = time.time()
    OUT.mkdir(exist_ok=True)
    raw = np.load(CACHE)
    fields = [ZernikeRayField(K=raw['K'],
        config=ZernikeOriginFieldConfig(image_size=(2048,2048), max_order=2),
        coefficients=ZernikeRayFieldCoefficients(
            origin_coeffs=raw[c+'_origin_coeffs'], direction_coeffs=raw[c+'_direction_coeffs']))
              for c in ('left','right')]
    refs = [(lambda p, field=f: field.ray(p[:,0],p[:,1])) for f in fields]
    train, test = grid(17,13), grid(32,24,True)
    target = [intersections(r(train)) for r in refs]
    report = {'base_commit':'ed422d6581d841771fbeeb7c0f596e5d05875766',
        'protocol': {'planes_mm':PLANES, 'train_shape':[17,13], 'reserved_shape':[32,24],
                     'residual_scalars':int(2*len(train)*2*2),
                     'loss':'linear', 'random_seed':20260911,
                     'scope':'compression of one fixed fitted field; not independent measurement validation'},
        'sources': {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [CACHE, ASSETS/'intermediate_state.npz',
                              ROOT/'src/stereocomplex/physics/cmo_physical.py',
                              ROOT/'src/stereocomplex/rayfields/zernike_origin_field.py']},
        'models':{}, 'centrality':{}}
    centre_params = []
    for c, r in enumerate(refs):
        raw_centre = raw[('left','right')[c]+'_origin_coeffs'][0]
        o, d = r(test)
        report['centrality'][('left','right')[c]] = {
            'raw_origin_mm':raw_centre.tolist(),
            'max_distance_to_common_centre_mm':float(np.max(np.linalg.norm(np.cross(raw_centre-o,d),axis=1))),
            'nonconstant_origin_coeff_max':float(np.max(np.abs(raw[('left','right')[c]+'_origin_coeffs'][1:])))}
        sol = fit(lambda x: (intersections(projective(x,train))-target[c]).ravel(),
                  projective_start(r(train),train))
        centre_params.append(sol.x)
        print('central channel',c,sol.success,sol.nfev,flush=True)
    models = {'central_projective_22': [(lambda p,x=x:projective(x,p)) for x in centre_params]}
    parameters = {'central_projective_22': np.concatenate(centre_params)}
    # Remove two exact gauges: hold f_obj and f_angular at a declared convention.
    f0=62.208914898071626
    dmid = [r(np.array([[1024.,1024.]]))[1][0] for r in refs]
    x14 = np.array([f0, f0-.845, 26.0756, 1024,1024, f0,
                    .197, np.mean([d[1] for d in dmid]), .43,-.44,.43,-.44,0,0])
    lo=np.array([1,1,0,0,0,20,0,-.3,-10,-10,-10,-10,-10,-10.])
    hi=np.array([500,1000,200,2048,2048,200,.5,.3,10,10,10,10,10,10.])
    diagnostics={}
    for n, x0 in [(14,x14),(26,None)]:
        if n==26:
            x0=np.r_[parameters['compact_14'],np.zeros(12)]
            lo=np.r_[lo, np.tile(np.r_[[-.15]*3,[-5.]*3],2)]
            hi=np.r_[hi, np.tile(np.r_[[.15]*3,[5.]*3],2)]
        active=np.array([i for i in range(n) if i not in (0,5)])
        def unpack(a):
            x=x0.copy(); x[active]=a; return x
        def residual(a):
            x=unpack(a)
            return np.concatenate([(intersections(compact(x,train,c))-target[c]).ravel() for c in range(2)])
        sol=fit(residual,x0[active],(lo[active],hi[active]))
        trials=[{'cost':float(sol.cost),'success':bool(sol.success),'nfev':sol.nfev}]
        if n==26:
            rng_start=np.random.default_rng(6413)
            for _ in range(3):
                alternative=x0.copy()
                alternative[14:]=rng_start.normal(0,np.tile([.01]*3+[.2]*3,2))
                trial=fit(residual,alternative[active],(lo[active],hi[active]))
                trials.append({'cost':float(trial.cost),'success':bool(trial.success),'nfev':trial.nfev})
                if trial.cost<sol.cost:
                    sol=trial
        name=f'compact_{n}'; parameters[name]=unpack(sol.x)
        models[name]=[(lambda p,c=c,x=unpack(sol.x):compact(x,p,c)) for c in range(2)]
        singular=np.linalg.svd(sol.jac/np.maximum(np.linalg.norm(sol.jac,axis=0),1e-30),compute_uv=False)
        diagnostics[name]={'success':bool(sol.success),'nfev':sol.nfev,'message':sol.message,
                           'cost':sol.cost,'active_coefficients':len(active),
                           'trials':trials,
                           'scaled_jacobian_singular_values':singular.tolist(),
                           'rank_at_relative_1e_6':int(np.sum(singular>singular[0]*1e-6))}
        print(name,diagnostics[name],flush=True)
    inter=np.load(ASSETS/'intermediate_state.npz')
    X=np.stack([inter['obj_pts']@R.T+t for R,t in zip(inter['opt_R'],inter['opt_t'])]).reshape(-1,3)
    pixels=[inter[c+'_pixels'].reshape(-1,2) for c in ('left','right')]
    maps={}
    for name, funcs in {**models,'reference_field':refs}.items():
        tr,_,_=measure(funcs,refs,train)
        te,angular,res=measure(funcs,refs,test)
        errors=[]; closure=[]
        for f,p in zip(funcs,pixels):
            e,cl=numerical_projection(f,p,X);errors.append(e);closure.append(cl)
        ee=np.concatenate(errors)
        report['models'][name]={'train':tr,'reserved':te,
            'inverse_pixel_rms':float(np.sqrt(np.mean(np.sum(ee**2,axis=1)))),
            'inverse_max_closure_mm':max(closure),
            'note':'fixed archived board poses; processed/completed corners, fitted observations'}
        maps[name+'_angular']=angular;maps[name+'_xy']=res
    report['optimisation']=diagnostics
    # Exact analytic reparameterisations checked through the implemented rays.
    x=parameters['compact_26']; shifted=x.copy(); shifted[:2]+=10
    scaled=x.copy(); scaled[5]*=1.2; scaled[8:14]*=1.2
    report['gauges']={}
    for label,other in [('f_obj_WD_plus_10_mm',shifted),('angular_scale_times_1p2',scaled)]:
        diffs=[intersections(compact(other,test,c))-intersections(compact(x,test,c)) for c in range(2)]
        report['gauges'][label]={'max_intersection_difference_mm':float(np.max(np.abs(diffs)))}
    # Controlled ablation: fixed central baseline, then a known arm transform
    # with or without spatially quadratic angular perturbations. This verifies
    # the residual diagnostic, not the microscope's optical construction.
    rng=np.random.default_rng(20260911)
    sim=[]
    basis=lambda p: np.c_[((p[:,0]-1023.5)/1023.5)**2-1/3,
                          ((p[:,1]-1023.5)/1023.5)**2-1/3]
    def simray(p, x):
        o,d=transform(models['central_projective_22'][0](p),x[:6])
        if len(x)>6:
            d=unit(d+np.c_[basis(p)*x[6:8],np.zeros(len(p))])
        return o,d
    for case, amplitudes in [('rigid',[0.,0.]),('rigid_and_quadratic',[.0002,-.00015])]:
        for seed in range(20):
            truth=np.r_[rng.normal(0,.008,3),rng.normal(0,.15,3),amplitudes]
            o,d=simray(train,truth)
            noisy=intersections((o,unit(d+rng.normal(0,2e-6,d.shape))))
            record={'case':case,'replicate':seed}
            for label,n in [('rigid',6),('rigid_quadratic',8)]:
                sol=fit(lambda a:(intersections(simray(train,a))-noisy).ravel(),np.zeros(n),max_nfev=100)
                err=intersections(simray(test,sol.x))-intersections(simray(test,truth))
                record[label+'_reserved_um']=float(1000*np.sqrt(np.mean(np.sum(err**2,axis=-1))))
                record[label+'_success']=bool(sol.success)
            sim.append(record)
    report['simulation']={'angular_noise_component_sd':2e-6,'replicates_per_case':20,
        'quadratic_coefficients': [.0002,-.00015], 'records':sim,
        'summary': {case:{label:float(np.median([r[label+'_reserved_um'] for r in sim if r['case']==case]))
                         for label in ('rigid','rigid_quadratic')}
                    for case in ('rigid','rigid_and_quadratic')}}
    # Compare stored coin reconstructions after ONE common Euclidean alignment;
    # report shape differences, never accuracy or noise from whole-surface MAD.
    zn=np.load(ASSETS/'specimen_reconstruction_zernike.npz')
    cm=np.load(ASSETS/'specimen_reconstruction_cmo26.npz')
    mask=zn['valid'].astype(bool)&cm['valid'].astype(bool)
    idx=np.flatnonzero(mask); idx=idx[np.linspace(0,len(idx)-1,min(len(idx),100000)).astype(int)]
    A=np.column_stack([zn[k].ravel()[idx] for k in ('X','Y','Z')]).astype(float)
    B=np.column_stack([cm[k].ravel()[idx] for k in ('X','Y','Z')]).astype(float)
    a,b=A.mean(0),B.mean(0); U,_,Vt=np.linalg.svd((B-b).T@(A-a))
    R=U@np.diag([1,1,np.linalg.det(U@Vt)])@Vt
    Bfit=(B-b)@R+a
    plane=np.c_[np.ones(len(A)),A[:,:2]]
    height=[]
    for pts in (A,Bfit):
        height.append(pts[:,2]-plane@np.linalg.lstsq(plane,pts[:,2],rcond=None)[0])
    report['coin']={'common_points':int(mask.sum()),'deterministic_subsample':len(A),
        'median_distance_after_SE3_um':float(1000*np.median(np.linalg.norm(Bfit-A,axis=1))),
        'detrended_height_MAD_um':[float(1000*np.median(np.abs(h-np.median(h)))) for h in height],
        'note':'stored two-model reconstructions, common mask and rigid alignment; no independent height truth'}
    np.savez_compressed(OUT/'diagnostic_maps.npz',pixels=test,**maps,
                        coin_X=A[:,0],coin_Y=A[:,1],coin_Z_ref=height[0],coin_Z_compact=height[1])
    np.savez_compressed(OUT/'fitted_models.npz',**parameters)
    report['runtime_seconds']=time.time()-start
    (OUT/'diagnostic_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'models':report['models'],'gauges':report['gauges'],
                      'simulation':report['simulation']['summary'],'coin':report['coin']},indent=2))


if __name__=='__main__':
    main()
