"""Larger arrival uncertainty, with the original proposal/backup and repair grid."""
import json
import numpy as np
from probe import HERE,get_job,check,save,common

if __name__=='__main__':
    old=HERE.parent/'congested_recovery_2026-11-10/results'
    proposal=np.array(json.loads((old/'proposal_planning.json').read_text())['commands'])
    backup=np.array(json.loads((old/'backup_planning.json').read_text())['commands'])
    jobs=[]
    for w,weight in [(600,1.),(600,.65),(900,1.),(900,.95)]:
        job=get_job('depth_backup_4500');job['w']=w
        job['commands']=common.compile_commands((1-weight)*proposal+weight*backup)[0].tolist()
        job['id']=f'arrival_w{w}_lambda{weight}';job['group']='arrival_extension';job['weight']=weight
        jobs.append(job)
    save(HERE/'arrival_extension_protocol.json',dict(jobs=jobs,
        choice='First grid weight that can satisfy the necessary one-hour queue balance, plus backup; unchanged 0.05 grid and endpoints.',
        regional_bounds=25,queue_bound=10,scope='Exploratory extension at wbar=600 and 900 veh/h; full-hour check required.'))
    for job in jobs:
        r=check(job,'tight12','class_box',1.01,False)
        save(HERE/'results'/(job['id']+'.json'),r)
        print(json.dumps(dict(id=job['id'],**{k:r['summary'][k] for k in ('accepted','minutes','seconds','service')})),flush=True)
