"""Record follow-up widths between a passing 25 and failing 50 vehicles."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from probe import HERE,get_job,save
from run_comparison import run

if __name__=='__main__':
    jobs=[]
    for plan in ('backup','repair'):
        for d in (30,35,40):
            j=get_job(f'depth_{plan}_4500');j.update(id=f'congestion_{plan}_d{d}',group='congestion')
            j['width']=[d/100]*6+[.1];jobs.append(j)
    save(HERE/'uncertainty_followup_protocol.json',dict(jobs=jobs,
        rule='Same four configurations and refinement as frozen comparison; grid refined after 25 passes and 50 fails. Exploratory, not held-out.',
        sources_fixed=True))
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
