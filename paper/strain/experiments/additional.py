"""Spatial holdout and repeat timing, with model orders fixed by depth-only CV."""
from pathlib import Path
import json,time,platform
import numpy as np
from models import model
from evaluate import load,metrics

root=Path('paper/strain/results');results={}
for series in ('calibration','calibration2'):
    e=json.loads((root/f'evaluation_{series}.json').read_text()); x,p,f,ids,z,xy=load(root/f'observations_{series}.npz')
    train=np.isin(f,e['train_indices']); spatial=(ids//15+ids%15)%2==0
    hold=(~train)&(~spatial); out={}
    for kind,entry in e['models'].items():
        m=model(kind,entry['degree']).fit(x[train&spatial],p[train&spatial]); out[kind]={'spatial_holdout':metrics(m.predict(p[hold]),x[hold]),'train_points':int((train&spatial).sum())}
        m=model(kind,entry['degree']).fit(x[train],p[train]);batch=p[~train][:10000];m.predict(batch)
        times=[]
        for _ in range(7):
            t=time.perf_counter();m.predict(batch);times.append(time.perf_counter()-t)
        out[kind]['timing_10000_points_s']=times;out[kind]['timing_median_s']=float(np.median(times))
    results[series]=out
results['environment']=dict(python=platform.python_version(),numpy=np.__version__,machine=platform.machine(),platform=platform.platform(),blas_threads=1)
(root/'additional.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
