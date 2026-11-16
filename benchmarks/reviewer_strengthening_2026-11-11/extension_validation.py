import hashlib,json
from probe import HERE,check,save
from plant_sensitivity import simulate

if __name__=='__main__':
    rows=[]
    for name in ('arrival_w600_lambda1.0','arrival_w600_lambda0.65'):
        old=json.loads((HERE/'results'/(name+'.json')).read_text());j=old['job']
        r=check(j,'tight12','class_box',1.01,True)
        c=r['certificate'];assert c['accepted']
        for st in c['interval_steps']:
            a=st['attempts'][-1]
            assert all(v<t for v,t in zip(a['continuous_error_upper'],a['trial_radius']))
            assert min(st['continuous_constraint_margins_lower'])>=0
            for k in ('exponential_tail','metzler_tail','flow_tail','remainder_flow_tail'):
                assert st['affine'][k]['order']==12
        save(HERE/'full_certificates'/(name+'.json'),r)
        s=simulate(name,1.,1.,2.5,j)
        assert s['physical_passes']==s['service_passes']==640
        rows.append(dict(id=name,simulation=s,intervals=len(c['interval_steps']),proof_id=c['proof_id']))
        print(json.dumps(rows[-1]),flush=True)
    save(HERE/'extension_validation.json',rows)
