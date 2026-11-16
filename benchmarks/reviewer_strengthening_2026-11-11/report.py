from pathlib import Path
import json,math
import numpy as np
from probe import HERE,affine,save
from mfd_deadline_certificates import inverse_interval,I,imat

def bounds(c):
    inv=inverse_interval(np.array(c['proof_descriptor']['initial_L']))
    G=np.array(c['affine_terminal_generators']);rho=np.array(c['affine_terminal_remainder'])
    r=(I(affine.rowsum_upper(imat(inv,G)))+imat(I(inv.mag()),I(rho)))
    p=I(c['terminal_shape']['center'])
    lo=((p-r)*I(100.)).lo;hi=((p+r)*I(100.)).hi
    queue_upper=float((I(20.)-I(c['continuous_constraint_margins_lower']['exterior_queue_upper'])) .hi)*100
    return dict(lower=lo.tolist(),upper=hi.tolist(),width=(hi-lo).tolist(),queue_upper=queue_upper)

def union(records):
    bs=[bounds(c) for c in records]
    lo=np.min([b['lower'] for b in bs],axis=0);hi=np.max([b['upper'] for b in bs],axis=0)
    return dict(lower=lo.tolist(),upper=hi.tolist(),width=(hi-lo).tolist(),queue_upper=max(b['queue_upper'] for b in bs))

if __name__=='__main__':
    data={}
    for path in sorted((HERE/'results').glob('arrival_w*.json')):
        r=json.loads(path.read_text());data[path.stem]=dict(accepted=r['summary']['accepted'],bounds=bounds(r['certificate']))
    for plan in ('backup','repair'):
        for d in (25,30):
            path=HERE/'comparison'/f'congestion_{plan}_d{d}.json';r=json.loads(path.read_text())
            if r['summary']['accepted']:
                winner=next(a for a in r['attempts'] if a['passed'])
                data[f'{plan}_d{d}']=dict(accepted=True,bounds=union([ch['record']['certificate'] for ch in winner['children']]))
    for name,row in data.items():print(name,row['accepted'],{k:row['bounds'][k][:2] if isinstance(row['bounds'][k],list) else row['bounds'][k] for k in ('upper','width','queue_upper')})
    save(HERE/'bound_summary.json',data)
