"""Independent Brown/perspective generator; coordinate noise, not rendered DIC images."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from models import model
from evaluate import WIDE_TRAIN,metrics


def project(xyz):
    world=xyz+[-2.4,-1.8,62]; pixels=[]
    for sign in (-1,1):
        center=np.array([sign*12.5,0,0.]); forward=np.array([0.,0.,65])-center; forward/=np.linalg.norm(forward)
        right=np.cross([0.,1.,0.],forward); right/=np.linalg.norm(right); up=np.cross(forward,right)
        rot=np.vstack([right,up,forward]); rvec,_=cv2.Rodrigues(rot)
        k=np.array([[25600.,0,1023.5],[0,25600.,1023.5],[0,0,1.]])
        pix,_=cv2.projectPoints(world,rvec,-rot@center,k,np.array([.8,-2.,.002*sign,-.003,.5]))
        pixels.append(pix.reshape(-1,2))
    return np.c_[pixels[0],pixels[1]]


def generate(extent):
    xy=np.array([(x*.3,y*.3) for y in range(1,12) for x in range(1,16)])
    z=np.linspace(3-extent/2,3+extent/2,101); f=np.repeat(np.arange(101),165)
    xyz=np.c_[np.tile(xy,(101,1)),np.repeat(z,165)]
    return xyz,project(xyz),f


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('paper/strain/results'));p.add_argument('--repeats',type=int,default=20);a=p.parse_args()
    rows=[]
    for extent in (.7,.08):
        xyz,clean,f=generate(extent); train=np.isin(f,WIDE_TRAIN)
        # Three unused interior planes; true x-extension 1000 microstrain.
        smask=np.isin(f,[25,45,75]); sx=xyz[smask]; deformed=sx.copy();deformed[:,0]=2.4+1.001*(sx[:,0]-2.4)
        sp=project(deformed); h=np.array([(i,i+4) for i in range(165) if i%15<11])
        v=np.array([(i,i+60) for i in range(165) if i//15<7])
        for noise in (0.,.05,.2,.5):
            for seed in range(1 if noise==0 else a.repeats):
                rng=np.random.default_rng(12000+seed); pix=clean+rng.normal(0,noise,clean.shape)
                dp=sp+rng.normal(0,noise,sp.shape)
                for kind,degree in [('ray',3),('soloff',332),('soloff',3),('direct',3)]:
                    m=model(kind,degree).fit(xyz[train],pix[train]); pred=m.predict(pix[~train])
                    row=dict(extent_mm=extent,noise_px=noise,repetition=seed,model=kind,degree=degree,nparams=m.nparams,**metrics(pred,xyz[~train]))
                    before=m.predict(pix[smask]).reshape(3,165,3);after=m.predict(dp).reshape(3,165,3)
                    for label,pairs,true in [('x',h,1000.),('y',v,0.)]:
                        length=lambda a:np.linalg.norm(a[:,pairs[:,1]]-a[:,pairs[:,0]],axis=2)
                        strain=1e6*(length(after)/length(before)-1)
                        row[f'strain_{label}_bias_microstrain']=float(strain.mean()-true)
                        row[f'strain_{label}_rmse_microstrain']=float(np.sqrt(np.mean((strain-true)**2)))
                    rows.append(row)
            print('completed',extent,noise,'runs',len(rows),flush=True)
            (a.out/'synthetic.json').write_text(json.dumps(dict(description='Brown-distorted perspective stereo; independent Gaussian coordinate noise in calibration and test; paired seeds; fixed cubic models',runs=rows),indent=2))

if __name__=='__main__':main()
