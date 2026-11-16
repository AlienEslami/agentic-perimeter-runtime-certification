"""Paired serial runtime study; no other study workers run concurrently."""
import hashlib,json,time
from probe import HERE,get_job,check,save
from run_comparison import CONFIGS,children

def one(job,variant,rep):
    start=time.perf_counter();attempts=[];winner=None
    for metric,divisions in CONFIGS:
        checks=[]
        for child,cells in children(job,divisions):
            r=check(child,variant,metric,1.01,False)
            checks.append(dict(cells=cells,record=r))
            if not r['summary']['accepted']:break
        passed=len(checks)==divisions**2 and all(x['record']['summary']['accepted'] for x in checks)
        attempts.append(dict(metric=metric,divisions=divisions,passed=passed,children=checks))
        if passed:winner=f'{metric}_{divisions**2}';break
    seconds=time.perf_counter()-start
    result=dict(job=job,variant=variant,repetition=rep,accepted=winner is not None,winner=winner,
        seconds=seconds,checks=sum(len(a['children']) for a in attempts),attempts=attempts)
    save(HERE/'timing'/f'{job["id"]}_{variant}_r{rep}.json',result)
    print(json.dumps({k:result[k] for k in ('variant','repetition','accepted','winner','seconds','checks')},ensure_ascii=True),flush=True)
    return {k:result[k] for k in ('variant','repetition','accepted','winner','seconds','checks')}|dict(id=job['id'])

if __name__=='__main__':
    ids=['depth_backup_4500','depth_repair_4500']
    save(HERE/'timing_protocol.json',dict(ids=ids,repetitions=2,
        ordering='baseline then refined in repetition 1; refined then baseline in repetition 2; serial processes',
        includes='All attempted configurations and sub-box checks, setup and result assembly.',
        excludes='Planning, observation, state estimation, actuation, disk serialization, Python import.',
        claim='Observed local verification time only; does not update the archived delayed-activation experiment.'))
    rows=[]
    for rep in (1,2):
        for case in ids:
            for variant in (('baseline','tight12') if rep==1 else ('tight12','baseline')):
                rows.append(one(get_job(case),variant,rep))
    save(HERE/'timing_summary.json',rows)
