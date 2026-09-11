"""Depth-blocked selection and evaluation on actual, uncompleted corner observations."""
from pathlib import Path
import argparse,json,time
import numpy as np
from models import model

WIDE_TRAIN=np.array([0,10,20,30,40,50,60,80,90,100])
CANDIDATES={'ray':[1,2,3,4],'soloff':[1,2,332,3,4],'direct':[1,2,3,4]}


def load(path):
    d=np.load(path); pixels=d['pixels']; good=np.isfinite(pixels).all(axis=(1,3)); frames,ids=np.where(good)
    p=pixels.transpose(0,2,1,3)[good].reshape(-1,4)
    xyz=np.c_[d['xy'][ids],d['z'][frames]]
    return xyz,p,frames,ids,d['z'],d['xy']


def metrics(pred,true):
    e=pred-true
    return dict(n=len(e),rmse_xyz_mm=np.sqrt(np.mean(e*e,axis=0)).tolist(),bias_xyz_mm=e.mean(axis=0).tolist(),p95_abs_z_mm=float(np.quantile(abs(e[:,2]),.95)))


def motion(pred,frames,ids,z,test):
    clouds=np.full((len(z),165,3),np.nan); clouds[frames,ids]=pred
    ref=int(np.argmin(abs(z-3))); pairs=[]
    for i in range(165):
        if i%15+4<15: pairs.append((i,i+4))
        if i//15+4<11: pairs.append((i,i+60))
    pairs=np.array(pairs); rows=[]
    for f in test:
        if f==ref: continue  # A cloud compared with itself is not a measurement.
        ok=np.isfinite(clouds[f]).all(axis=1)&np.isfinite(clouds[ref]).all(axis=1)
        a,b=clouds[f,ok],clouds[ref,ok]; delta=(a-b).mean(axis=0)
        aa=a-a.mean(axis=0); bb=b-b.mean(axis=0); u,_,vt=np.linalg.svd(aa.T@bb)
        rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
        rg=np.sqrt(np.mean(np.sum((aa@rot-bb)**2,axis=1)))
        pp=pairs[ok[pairs].all(axis=1)]
        lengths=lambda c:np.linalg.norm(c[pp[:,1]]-c[pp[:,0]],axis=1)
        strain=1e6*(lengths(clouds[f])/lengths(clouds[ref])-1)
        rows.append(dict(frame=int(f),z_mm=float(z[f]),nominal_dz_mm=float(z[f]-z[ref]),dz_mm=float(delta[2]),
            motion_error_mm=float(delta[2]-(z[f]-z[ref])),nonrigid_rms_mm=float(rg),
            gauge_rms_microstrain=float(np.sqrt(np.mean(strain**2))),gauge_bias_microstrain=float(strain.mean()),gauges=len(pp)))
    increments=[]
    for f in range(len(z)-1):
        if f not in test or f+1 not in test: continue
        ok=np.isfinite(clouds[f:f+2]).all(axis=(0,2)); delta=(clouds[f+1,ok]-clouds[f,ok]).mean(axis=0)
        increments.append(float(delta[2]-(z[f+1]-z[f])))
    return dict(reference_frame=ref,frames=rows,reference_motion_rmse_mm=float(np.sqrt(np.mean([r['motion_error_mm']**2 for r in rows]))),
        gauge_rms_microstrain=float(np.sqrt(np.mean([r['gauge_rms_microstrain']**2 for r in rows]))),
        adjacent_test_pairs=len(increments),adjacent_dz_error_rmse_mm=float(np.sqrt(np.mean(np.array(increments)**2))))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=Path('paper/strain/results')); a=parser.parse_args()
    for series in ('calibration','calibration2'):
        xyz,p,f,ids,z,xy=load(a.out/f'observations_{series}.npz')
        train=WIDE_TRAIN if series=='calibration' else np.rint(np.linspace(0,100,10)).astype(int)
        test=np.setdiff1d(np.arange(101),train); mask=np.isin(f,train)
        results=dict(series=series,train_indices=train.tolist(),test_indices=test.tolist(),train_points=int(mask.sum()),test_points=int((~mask).sum()),models={})
        arrays=dict(xyz=xyz,pixels=p,frames=f,ids=ids,z=z,train_indices=train,test_indices=test)
        for kind,degrees in CANDIDATES.items():
            cv=[]
            for degree in degrees:
                errs=[]
                for held in train:
                    fit=np.isin(f,train[train!=held]); hold=f==held
                    m=model(kind,degree).fit(xyz[fit],p[fit]); error=m.predict(p[hold])[:,2]-xyz[hold,2]
                    errs.append(float(np.mean(error**2)))
                score=float(np.sqrt(np.mean(errs))); cv.append(dict(degree=degree,z_rmse_mm=score,fold_mse_mm2=errs))
                print(series,kind,degree,'CV um',score*1000,flush=True)
            selected=min(cv,key=lambda v:v['z_rmse_mm'])['degree']; m=model(kind,selected).fit(xyz[mask],p[mask])
            start=time.perf_counter(); pred=m.predict(p); elapsed=time.perf_counter()-start
            residual=m.pixel_residual(xyz[mask],p[mask])
            result=dict(degree=selected,nparams=m.nparams,cv=cv,fit_diagnostics=m.diagnostics,
                train_pixel_rms=None if kind=='direct' else float(np.sqrt(np.mean(residual**2))),
                test=metrics(pred[~mask],xyz[~mask]),motion=motion(pred,f,ids,z,test),prediction_seconds=elapsed)
            results['models'][kind]=result; arrays[kind]=pred
            print(series,kind,'TEST um',1000*result['test']['rmse_xyz_mm'][2],'strain',result['motion']['gauge_rms_microstrain'],flush=True)
        (a.out/f'evaluation_{series}.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        np.savez_compressed(a.out/f'predictions_{series}.npz',**arrays)

if __name__=='__main__': main()
