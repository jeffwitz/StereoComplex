"""Scientific numerical checks: exact geometry, basis and nonlinear inversion."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares
from models import model,Soloff,RayField
from evaluate import load,WIDE_TRAIN


def main():
    rng=np.random.default_rng(19073); xyz=rng.uniform([-2,-1,2.7],[2,1,3.3],(500,3))
    def project(x): return np.c_[1023+400*x[:,0]+70*x[:,2],1023+410*x[:,1],1023+400*x[:,0]-70*x[:,2],1023+410*x[:,1]]
    p=project(xyz); target=rng.uniform([-1.8,-.9,2.72],[1.8,.9,3.28],(100,3)); out={}
    for kind in ('ray','soloff','direct'):
        m=model(kind,2).fit(xyz,p); error=np.max(abs(m.predict(project(target))-target)); assert error<1e-9,(kind,error)
        out[kind+'_affine_max_mm']=float(error)
    # Soloff 332 is the full cubic basis with pure z^3 removed.
    s=Soloff(332); assert len(s.p)==19 and not np.any(np.all(s.p==[0,0,3],axis=1))
    # Moving the ray reference plane changes coefficients but not lines.
    ray=RayField(3).fit(xyz,p); before=ray.predict(project(target)); dz=.12
    for c in ray.c: c[:len(ray.modes)]+=dz*c[len(ray.modes):]/ray.zscale
    ray.z0+=dz; gauge_error=float(np.max(abs(ray.predict(project(target))-before))); assert gauge_error<1e-9
    out['ray_reference_plane_invariance_max_mm']=gauge_error
    root=Path('paper/strain/results'); real=root/'observations_calibration.npz'
    if real.exists():
        x,p,f,*_=load(real); train=np.isin(f,WIDE_TRAIN); s=Soloff(4).fit(x[train],p[train]); pp=p[~train][::1700]
        fast=s.predict(pp); slow=[]
        for pix in pp:
            res=least_squares(lambda q:s.forward(q[None])[0]-pix,(pix-s.offset)@s.inverse,
                jac=lambda q:s.jac(q[None])[0],xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=100)
            slow.append(res.x*s.scale+s.center)
        err=float(np.max(abs(fast-np.array(slow)))); assert err<1e-6,err
        out['soloff_vector_vs_scipy_max_mm']=err;out['soloff_scipy_points']=len(pp)
    out['passed']=True; (root/'verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__': main()
