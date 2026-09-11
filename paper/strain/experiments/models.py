"""Comparable finite-plane ray, forward Soloff-type and direct polynomial fits.

All fits use the same nominal coordinates, unweighted least squares, training-only
normalisation, column RMS scaling and SVD cutoff 1e-12. No target-pose refinement.
"""
import itertools, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from stereocomplex.core.model_compact.zernike import zernike_modes, eval_real_zernike


def powers(dim, degree):
    return np.array([p for p in itertools.product(range(degree+1),repeat=dim) if sum(p)<=degree],int)


def basis(x,p):
    out=np.ones((len(x),len(p)))
    for k in range(x.shape[1]):
        xp=np.ones((len(x),int(p[:,k].max())+1))
        for d in range(1,xp.shape[1]): xp[:,d]=xp[:,d-1]*x[:,k]
        out*=xp[:,p[:,k]]
    return out


def linear_fit(a,b):
    scale=np.sqrt(np.mean(a*a,axis=0)); scale=np.maximum(scale,1e-30)
    c,_,rank,s=np.linalg.lstsq(a/scale,b,rcond=1e-12)
    return c/scale[:,None],dict(rank=int(rank),columns=a.shape[1],condition=float(s[0]/s[-1]))


class RayField:
    def __init__(self,degree): self.degree=degree; self.modes=zernike_modes(degree); self.nparams=8*len(self.modes)
    def b(self,uv):
        q=(uv-1023.5)/(1023.5*np.sqrt(2)); r=np.linalg.norm(q,axis=1); t=np.arctan2(q[:,1],q[:,0])
        return np.column_stack([eval_real_zernike(m,r,t) for m in self.modes])
    def fit(self,xyz,pixels):
        self.z0=xyz[:,2].mean(); self.zscale=xyz[:,2].std(); self.c=[]; self.diagnostics=[]
        dz=(xyz[:,2]-self.z0)/self.zscale
        for cam in range(2):
            b=self.b(pixels[:,2*cam:2*cam+2]); c,d=linear_fit(np.c_[b,dz[:,None]*b],xyz[:,:2]); self.c.append(c); self.diagnostics.append(d)
        return self
    def ray(self,uv,cam):
        b=self.b(uv); m=len(self.modes); a=b@self.c[cam][:m]; slope=b@self.c[cam][m:]/self.zscale
        origin=np.c_[a,np.full(len(a),self.z0)]; direction=np.c_[slope,np.ones(len(a))]
        direction/=np.linalg.norm(direction,axis=1)[:,None]
        return origin,direction
    def predict(self,pixels):
        o,d=self.ray(pixels[:,:2],0); q,e=self.ray(pixels[:,2:],1); w=o-q
        dot=lambda a,b:np.sum(a*b,axis=1)
        b=dot(d,e); dd=dot(d,w); ee=dot(e,w); den=1-b*b
        if np.any(den<1e-12): raise ValueError('Degenerate stereo rays')
        s=(b*ee-dd)/den; t=(ee-b*dd)/den
        return .5*(o+s[:,None]*d+q+t[:,None]*e)
    def pixel_residual(self,xyz,pixels):
        result=[]; dz=(xyz[:,2]-self.z0)/self.zscale
        for cam in range(2):
            uv=pixels[:,2*cam:2*cam+2].copy()
            def f(q):
                b=self.b(q); return np.c_[b,dz[:,None]*b]@self.c[cam]
            for _ in range(8):
                y=f(uv); j=np.stack([(f(uv+np.eye(2)[k]*.01)-y)/.01 for k in range(2)],axis=2)
                uv-=np.linalg.solve(j,(y-xyz[:,:2])[...,None])[...,0]
            result.append(uv-pixels[:,2*cam:2*cam+2])
        return np.concatenate(result,axis=1)


class Soloff:
    def __init__(self,degree):
        self.degree=degree; self.p=powers(3,3 if degree==332 else degree)
        if degree==332: self.p=self.p[~np.all(self.p==[0,0,3],axis=1)]
        self.nparams=4*len(self.p)
    def fit(self,xyz,pixels):
        self.center=xyz.mean(axis=0); self.scale=xyz.std(axis=0); x=(xyz-self.center)/self.scale
        self.c,self.diagnostics=linear_fit(basis(x,self.p),pixels)
        affine,_=linear_fit(np.c_[np.ones(len(x)),x],pixels)
        self.offset=affine[0]; self.inverse=np.linalg.pinv(affine[1:]); return self
    def forward(self,x): return basis(x,self.p)@self.c
    def jac(self,x):
        js=[]
        for k in range(3):
            active=self.p[:,k]>0; pp=self.p[active].copy(); weight=pp[:,k].copy(); pp[:,k]-=1
            js.append((basis(x,pp)*weight)@self.c[active])
        return np.stack(js,axis=2)
    def predict(self,pixels):
        x=(pixels-self.offset)@self.inverse
        for _ in range(20):
            r=self.forward(x)-pixels; j=self.jac(x); h=np.einsum('nki,nkj->nij',j,j)
            h+=np.eye(3)[None]*np.maximum(np.trace(h,axis1=1,axis2=2),1)[:,None,None]*1e-10
            step=np.linalg.solve(h,np.einsum('nki,nk->ni',j,r)[...,None])[...,0]
            old=np.sum(r*r,axis=1); factor=np.ones(len(x))
            for _ in range(8):
                candidate=x-factor[:,None]*step; new=np.sum((self.forward(candidate)-pixels)**2,axis=1)
                bad=new>old+1e-12
                if not bad.any(): break
                factor[bad]*=.5
            x-=factor[:,None]*step
            if np.max(np.linalg.norm(factor[:,None]*step,axis=1))<1e-9: break
        return x*self.scale+self.center
    def pixel_residual(self,xyz,pixels): return self.forward((xyz-self.center)/self.scale)-pixels


class DirectPolynomial:
    def __init__(self,degree): self.degree=degree; self.p=powers(4,degree); self.nparams=3*len(self.p)
    def fit(self,xyz,pixels):
        self.center=pixels.mean(axis=0); self.scale=pixels.std(axis=0)
        self.c,self.diagnostics=linear_fit(basis((pixels-self.center)/self.scale,self.p),xyz); return self
    def predict(self,pixels): return basis((pixels-self.center)/self.scale,self.p)@self.c
    def pixel_residual(self,xyz,pixels): return np.full_like(pixels,np.nan)


def model(kind,degree): return {'ray':RayField,'soloff':Soloff,'direct':DirectPolynomial}[kind](degree)
