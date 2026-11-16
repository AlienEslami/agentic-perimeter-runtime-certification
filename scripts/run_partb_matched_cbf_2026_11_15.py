"""Complete the existing D027 physical-CBF comparison on the fixed 640 samples."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
import sys, json, itertools, time, hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
sys.path.insert(0,str(SUITE))
import common
OUT=ROOT/'benchmarks/partb_traffic_presentation_2026-11-15'

def work(item):
    signs,profile=item
    d=json.loads((SUITE/'results/contracts/D027.json').read_text())
    return common.feedback_run(np.asarray(d['policies']['nominal_mpc']['commands']),
                               d['contract'],'cbf',profile=profile,signs=signs)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    protocol={'case':'D027','source':'reusable_perimeter_suite_2026-09-26/results/contracts/D027.json',
      'samples':'all 128 initial-box corners paired with the five existing arrival profiles',
      'controller':'unchanged common.cbf, alpha=6 per hour, exact simulated state, robust arrival bound',
      'control_update_seconds':10,'integrator':'unchanged RK4 at ten-second steps',
      'planning':'unchanged stored nominal proposal; no retuning or optimization',
      'purpose':'matched simulation comparison, not a certificate for the feedback controller',
      'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [SUITE/'common.py',SUITE/'results/contracts/D027.json']}}
    common.save(OUT/'cbf_protocol.json',protocol)
    jobs=list(itertools.product(list(itertools.product((-1.,1.),repeat=7)),common.PROFILES))
    begin=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(work,j) for j in jobs]
        for f in as_completed(fs):
            rows.append(f.result())
            if len(rows)%64==0:
                print(f'{len(rows)}/640 complete, elapsed {time.perf_counter()-begin:.1f} s',flush=True)
                common.save(OUT/'cbf_progress.json',{'completed':len(rows),'rows':rows})
    assert all(r['completed'] for r in rows),[r for r in rows if not r['completed']]
    rows.sort(key=lambda r:(r['corner'],r['profile']))
    old=json.loads((SUITE/'results/contracts/D027.json').read_text())['cbf_adverse_realization']
    check=next(r for r in rows if r['corner']=='+++++++' and r['profile']=='constant_high')
    for key in ['tts_veh_h','priority_cost','peak_queue_vehicles','deadline_n1_vehicles']:
        assert abs(check[key]-old[key])<1e-7,(key,check[key],old[key])
    summary=common.summarize(rows)
    assert summary['maximum_conservation_error_vehicles']<1e-6
    common.save(OUT/'cbf_ensemble.json',{'protocol':protocol,'summary':summary,'realizations':rows,
                                       'elapsed_seconds':time.perf_counter()-begin})
    print(json.dumps(summary,indent=2),flush=True)
