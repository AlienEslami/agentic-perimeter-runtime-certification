from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = ROOT / 'benchmarks/reusable_perimeter_suite_2026-09-26'
ARCHIVE = ROOT / 'output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK'
OUT = HERE / 'results'
sys.path.insert(0, str(SUITE))
import common
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_uncertainty_frontier import MetricConstantVerifier
from mfd_incident_positive_reachability import PositiveVerifier
import mfd_incident_positive_reachability as positive
from mfd_incident_affine_reachability import AffineGeneratorVerifier

def sha(path):
    path = Path(path)
    if not path.exists():
        candidates = [SUITE/'vendor'/path.name, HERE/path.name, ARCHIVE/path.name]
        path = next(p for p in candidates if p.exists())
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Adapt archived provenance paths only. Numerical kernels are unchanged.
positive.sha = sha

def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def fixture():
    path = HERE/'coordinate_fixture.json'
    return json.loads(path.read_text())

def attempt(method, width):
    path = OUT / f'coordinates_{method}_d{width}.json'
    if path.exists():
        return json.loads(path.read_text())['summary']
    src = fixture()
    verifier = (PositiveVerifier('positive_queue') if method == 'positive_queue' else
                AffineGeneratorVerifier() if method == 'signed' else MetricConstantVerifier(method))
    result = verify_command_plan(np.array(src['initial_state']), src['commands'],
            [Fraction(1,12), Fraction(11,12)], initial_box=np.array([width]*6+[min(width,10)])/100,
            residual_bound=.6, verifier=verifier, retain_steps=False)
    checks = result['service_check_results']
    upper = 100*checks[0]['upper'] if checks else None
    row = dict(method=method, d=width, accepted=result['accepted'],
               deadline_bound_vehicles=upper, certified_until_h=result['certified_until_h'],
               wall_seconds=result['wall_seconds'], proof_id=result['proof_id'])
    save(path, dict(summary=row, certificate=result, source_hashes=common.portable_source_hashes()))
    return row

def sweep():
    source = ARCHIVE/'reproducibility/results/mfd_incident_uncertainty_frontier_results.json'
    if not (HERE/'coordinate_fixture.json').exists():
        src=json.loads(source.read_text())
        save(HERE/'coordinate_fixture.json',dict(initial_state=src['initial_state'],commands=src['commands'],
             archived_source=str(source.relative_to(ROOT)),source_sha256=sha(source)))
    jobs=[(m,d) for m in ('physical_box','class_box','conservation_box','positive_queue','signed')
          for d in (1,5,10,25,50,100)]
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(attempt,*job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result(); rows.append(row); print(json.dumps(row),flush=True)
    save(OUT/'coordinate_comparison.json',dict(protocol_sha256=sha(HERE/'PROTOCOL.txt'),
        rows=sorted(rows,key=lambda r:(r['method'],r['d']))))

def trace():
    src=SUITE/'results/contracts/D027.json'
    record=json.loads(src.read_text()); c=record['contract']
    result=verify_command_plan(np.array(common.fixture()['center']),
        record['policies']['grid_repair']['commands'],common.DURATIONS,
        initial_time_h=common.START, amplitude=c['amplitude'],duration=c['duration_h'],
        initial_box=c['width_vehicles']/100,residual_bound=c['residual_veh_h']/100,
        step_seconds=15,verifier=AffineGeneratorVerifier(),retain_steps=True,
        service_checks=[dict(time_h=common.END,coordinate=0,upper=c['deadline_vehicles']/100)])
    assert result['accepted'] and result['verified_continuous_physical']
    save(OUT/'D027_repair_continuous_certificate.json',dict(source_sha256=sha(src),certificate=result))
    print(json.dumps(dict(trace_accepted=result['accepted'],steps=len(result['interval_steps']),
                         deadline=result['service_check_results'])),flush=True)

def runtime():
    import platform
    import winreg
    key=winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
    processor=winreg.QueryValueEx(key,'ProcessorNameString')[0]
    src=SUITE/'results/contracts/D027.json'
    rec=json.loads(src.read_text()); c=rec['contract']
    nominal=np.array(rec['policies']['nominal_mpc']['commands'])
    backup=np.array(rec['policies']['continuation']['commands'])
    records=[]
    for repetition in range(3):
        t=time.perf_counter(); bc=common.certify(backup,c); backup_seconds=time.perf_counter()-t
        assert bc['accepted']
        t=time.perf_counter(); nc=common.certify(nominal,c); nominal_seconds=time.perf_counter()-t
        assert not nc['accepted']
        attempts=[]; t=time.perf_counter()
        for j in range(1,21):
            weight=j/20
            commands,_=common.compile_commands((1-weight)*nominal+weight*backup)
            start=time.perf_counter(); cert=common.certify(commands,c)
            attempts.append(dict(weight=weight,accepted=cert['accepted'],seconds=time.perf_counter()-start))
            if cert['accepted']:break
        repair_seconds=time.perf_counter()-t
        assert attempts[-1]['weight']==rec['selected_theta']==.25
        row=dict(repetition=repetition+1,backup_seconds=backup_seconds,
                 nominal_seconds=nominal_seconds,repair_candidate_seconds=repair_seconds,
                 proposal_and_repair_seconds=nominal_seconds+repair_seconds,
                 including_backup_seconds=backup_seconds+nominal_seconds+repair_seconds,attempts=attempts)
        records.append(row); print(json.dumps(row),flush=True)
        save(OUT/'runtime.json',dict(platform=platform.platform(),python=sys.version,
             processor=processor,source_sha256=sha(src),records=records,
             timing_scope='Fresh verifier per candidate; excludes imports, planning, simulations and plotting.'))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['sweep','trace','runtime'])
    args=parser.parse_args();globals()[args.mode]()
