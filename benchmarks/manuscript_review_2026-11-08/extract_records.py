"""Reproduce manuscript summaries from unchanged archived study records."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
RESULTS=SUITE/'results'
hashes={}

def read(path):
    hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf-8'))

def summary(a):
    return dict(count=len(a),minimum=float(min(a)),median=float(np.median(a)),maximum=float(max(a)))

rows=[]; repair_times=[]; illustrative={}; counts={'nominal':0,'with_repair':0}
for path in sorted((RESULTS/'contracts').glob('D*.json')):
    d=read(path); c=d['contract']; p=d['policies']['nominal_mpc']
    rows.append(dict(case=c['id'],
        queue_excess_vehicles=max(0,p['summary']['peak_queue_vehicles']['maximum']-2000),
        deadline_excess_vehicles=max(0,p['summary']['deadline_n1_vehicles']['maximum']-c['deadline_vehicles'])))
    nominal=p['certificate']['accepted']
    repaired=d['policies'].get('grid_repair',{}).get('certificate',{}).get('accepted',False)
    counts['nominal']+=nominal
    counts['with_repair']+=nominal or repaired
    if repaired and not nominal:
        repair_times.append(sum(a['certificate']['wall_seconds'] for a in d['repair_attempts']))
    if c['id']=='D027':
        for name in ('nominal_mpc','grid_repair','continuation'):
            replay=next(r for r in d['policies'][name]['realizations']
                        if r['corner']=='+++++++' and r['profile']=='constant_high')
            illustrative[name]={k:replay[k] for k in ('completed_trips','tts_veh_h',
                'peak_queue_vehicles','deadline_n1_vehicles','conservation_residual_vehicles')}
assert len(rows)==80 and counts=={'nominal':36,'with_repair':74}
queue=[r['queue_excess_vehicles'] for r in rows if r['queue_excess_vehicles']>1e-6]
deadline=[r['deadline_excess_vehicles'] for r in rows if r['deadline_excess_vehicles']>1e-6]
assert len(queue)==44 and len(deadline)==36 and len(repair_times)==38

scan=read(RESULTS/'sequential_authority_2026-10-01/full_grid_scan.json')
violations=[dict(pair=r['pair'],queue_excess_vehicles=-100*r['minimum_physical_margin'])
            for r in scan if r.get('physical_violations',0)]
# The archived report and independent witness identify these negative margins as queue overflow.
assert len(violations)==3
pilot_errors=[]
for path in sorted((RESULTS/'api').glob('*.json')):
    d=json.loads(path.read_text(encoding='utf-8'))
    if 'call_id' not in d: continue
    if d['interface']=='history_only' and not d['attack'] and not d['tool_correct']:
        d=read(path)
        pilot_errors.append({k:d[k] for k in ('call_id','expected_objective','expected_tool','arguments')})
assert len(pilot_errors)==2
assert all(r['expected_tool']=='repair' and r['arguments']['tool']=='retain' for r in pilot_errors)

out=dict(per_contract=rows,queue_excess_summary=summary(queue),
    deadline_excess_summary=summary(deadline),archived_additional_repair_time_seconds=summary(repair_times),
    certified_counts=counts,illustrative_realization=illustrative,
    sequential_queue_violations=violations,pilot_clean_tool_errors=pilot_errors,source_sha256=hashes)
(HERE/'results/archived_record_checks.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k not in ('per_contract','source_sha256','pilot_clean_tool_errors')},indent=2))
