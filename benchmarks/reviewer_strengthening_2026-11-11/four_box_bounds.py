"""Like-for-like four-box enclosure comparison, not a timing experiment."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from probe import HERE,get_job,check,save
from run_comparison import children
from report import union

def one(item):
    job,cells=item
    r=check(job,'tight12','class_box',1.01,False)
    name=job['id']+'_cell'+''.join(map(str,cells))
    save(HERE/'four_box'/f'{name}.json',r)
    return name,r

if __name__=='__main__':
    jobs=[]
    for plan in ('backup','repair'):
        j=get_job(f'depth_{plan}_4500');jobs.extend(children(j,2))
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,jobs))
    result={}
    for plan in ('backup','repair'):
        cs=[r['certificate'] for name,r in rows if name.startswith(f'depth_{plan}_')]
        assert len(cs)==4 and all(c['accepted'] for c in cs)
        result[plan]=union(cs)
    save(HERE/'four_box_bound_summary.json',result)
    print(json.dumps(result))
