"""Fixed follow-up configuration and frozen plans; every attempted outcome retained."""
from pathlib import Path
import argparse, hashlib, itertools, json, time
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from probe import HERE,get_job,check,save,common

CONFIGS=[('conservation_box',1),('class_box',1),('class_box',2),('conservation_box',2)]

def children(job,divisions):
    for cells in itertools.product(range(divisions),repeat=2):
        child=dict(job);x=list(job['center']);w=list(job['width'])
        for k,cell in enumerate(cells):
            center=Fraction.from_float(x[k]);radius=Fraction.from_float(w[k])
            low=center-radius+2*radius*cell/divisions;high=center-radius+2*radius*(cell+1)/divisions
            middle=float((low+high)/2);needed=max(Fraction.from_float(middle)-low,high-Fraction.from_float(middle))
            bound=float(needed)
            if Fraction.from_float(bound)<needed:bound=np.nextafter(bound,np.inf)
            assert Fraction.from_float(middle)-Fraction.from_float(bound)<=low
            assert Fraction.from_float(middle)+Fraction.from_float(bound)>=high
            x[k]=middle;w[k]=float(bound)
        child.update(center=x,width=w)
        yield child,cells

def run(job):
    out=HERE/'comparison';out.mkdir(exist_ok=True)
    path=out/(job['id']+'.json')
    if path.exists():return json.loads(path.read_text())['summary']
    start=time.perf_counter();attempts=[];winner=None
    for metric,divisions in CONFIGS:
        rows=[]
        for child,cells in children(job,divisions):
            r=check(child,'tight12',metric,1.01,retain=False)
            rows.append(dict(cells=cells,record=r))
            if not r['summary']['accepted']:break
        passed=len(rows)==divisions**2 and all(r['record']['summary']['accepted'] for r in rows)
        attempts.append(dict(metric=metric,divisions=divisions,passed=passed,children=rows))
        if passed:winner=f'{metric}_{divisions**2}';break
    summary=dict(id=job['id'],group=job['group'],accepted=winner is not None,winner=winner,
        seconds=time.perf_counter()-start,attempted_checks=sum(len(r['children']) for r in attempts))
    save(path,dict(job=job,summary=summary,attempts=attempts))
    return summary

def jobs():
    frozen=json.loads((HERE.parent/'fixed_portfolio_2026-11-10/protocol_final.json').read_text())['jobs']
    selected=[j for j in frozen if j['group'] in ('full','nominal','structure')]
    for plan in ('backup','repair'):
        for d in (25,50,100,200):
            j=get_job(f'depth_{plan}_4500');j['id']=f'congestion_{plan}_d{d}';j['group']='congestion'
            j['width']=[d/100]*6+[.1];selected.append(j)
    return selected

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--group',default='all');a=p.parse_args()
    js=jobs()
    if a.group!='all':js=[j for j in js if j['group']==a.group]
    protocol=dict(purpose='Refine interval evaluation and enclosure search on frozen plans; no plan tuning or revised limits.',
        variants='tight12: exact even-power ranges; 1% trial padding; 12-term verified series including explicit tail; vectorized products with unchanged summation and directed rounding.',
        configs=CONFIGS,jobs=js,workers=a.workers,
        timing='Concurrent exploratory batch. Separate serial paired timing is required; these times are not a deadline result.',
        sources=common.portable_source_hashes(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(HERE/f'protocol_{a.group}.json',protocol)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,j) for j in js]):print(json.dumps(f.result()),flush=True)

if __name__=='__main__':main()
