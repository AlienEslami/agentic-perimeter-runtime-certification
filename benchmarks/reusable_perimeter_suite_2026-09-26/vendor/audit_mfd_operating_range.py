"""Cross-study audit and compact aggregation; no new live calls or optimization."""
from __future__ import annotations
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import mfd_incident_model as model
from mfd_incident_agent_runtime import audit
from mfd_incident_mpc_certification import constant_command_mode, exact_command_constraints
from mfd_incident_verifier import I, imat

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'reproducibility/results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((RESULTS/name).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-package', type=Path)
    args = parser.parse_args()
    old = read('streamlined_prior_live_manifest.json')
    for name, expected in old['sha256'].items():
        assert digest(ROOT/name)==expected, name
    preserved_paper = []
    if args.prior_package:
        names = ['FOUNDATION_MFD_PARTB_ELSEVIER.tex',
                 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY.tex',
                 'FOUNDATION_REFERENCES.bib']
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'manuscript_streamlined').glob('*.tex')]
        for name in names:
            assert digest(ROOT/name)==digest(args.prior_package/name), name
            preserved_paper.append(name)
        for name in ['mfd_incident_mpc_grid_results.json', 'mfd_incident_mpc_certified_grid_results.json']:
            assert digest(RESULTS/name)==digest(args.prior_package/'reproducibility/results'/name), name
    expected_model = dict(equations='seven_state_class_conservation_with_min_admission_v1',
        state_order=list(model.STATE_ORDER),
        MFD_coefficients=[model.original.MFD_A,model.original.MFD_B,model.original.MFD_C],
        internal_demand=model.INTERNAL_DEMAND.tolist(), admission_capacity=model.CAPACITY,
        release_rate=model.RELEASE_RATE,base_arrival=model.BASE_ARRIVAL,queue_limit=model.QUEUE_LIMIT,
        regional_jam_limits=[100.,100.],command_box=[0.,1.],directional_limit=.75,admission_split=.5)
    names = [f'mfd_incident_agent_live_p{p}_{v}_results.json' for p in (0,1)
             for v in ('stateful','memoryless')]
    names += ['mfd_incident_agent_oracle_final_results.json','mfd_incident_agent_network_results.json']
    rows, hashes = [], {}
    proof_count = snapshots = 0
    for name in names:
        result = read(name)
        hashes[name] = digest(RESULTS/name)
        c,n = result['certified'],result['nominal_matched_requests']
        assert c['metrics']['complete'] and n['metrics']['complete']
        assert len(c['decisions'])==len(n['decisions'])==12
        for case in (c,n):
            fresh_audit = audit(case)
            assert fresh_audit==case['independent_audit']
            for source, expected in case['source_sha256'].items():
                snapshot=RESULTS/'ors'/expected[:16]/Path(source).name
                assert digest(snapshot)==expected, snapshot
                snapshots += 1
            for event in case['decisions']:
                cert = event['certificate']
                if cert is None: continue
                desc = cert['proof_descriptor']
                assert digest_bytes(desc)==cert['proof_id']
                assert desc['model_contract']==expected_model
                assert desc['initial_state']==event['initial_state']
                assert desc['pulse']==[str(Fraction.from_float(.3)),'1','1/4',str(Fraction.from_float(1/12))]
                widths=np.array(case['configuration']['state_half_widths_vehicles'])/100
                radius=imat(np.array(desc['initial_L']),I(-widths,widths)).mag()
                assert np.array_equal(radius,desc['initial_radius'])
                assert Fraction(desc['initial_time'])==Fraction(event['exact_time_h'])
                assert sum(map(Fraction,desc['durations']))+Fraction(desc['initial_time'])==1
                for index, command in enumerate(event['candidate_commands']):
                    mode=constant_command_mode(command)
                    fingerprint=hashlib.sha256(b''.join(np.asarray(value,dtype=float).tobytes()
                        for value in (mode.center,mode.feedforward,mode.gain,mode.A,mode.P))).hexdigest()
                    assert fingerprint==desc['mode_fingerprints'][index]
                    assert cert['adapter_audit']['exact_actuator_checks'][index]==exact_command_constraints(command)
                    if cert['accepted']: assert exact_command_constraints(command)['passed']
                proof_count += 1
        for ce,ne in zip(c['decisions'],n['decisions']):
            assert ce['tool_request']==ne['tool_request'] and ce['request_valid']==ne['request_valid']
        rejects=[]
        for e in c['decisions']:
            if e['candidate_admitted']: continue
            samples=np.array(c['trajectory']['state'][30*e['epoch']:30*(e['epoch']+1)+1])
            candidate=np.array(e['candidate_commands'][0]); retained=np.array(e['executed_constant_command'])
            rejects.append(dict(minute=60*e['time_h'],failures=[{k:f[k] for k in
                ('kind','constraint','margin_lower','time_h') if k in f} for f in e['certificate']['failures']],
                proposed_first_vs_executed_max_difference=float(np.max(np.abs(
                    candidate-retained))),
                sampled_first_interval_field_difference_units_h=float(max(np.max(abs(
                    model.field(x,candidate)-model.field(x,retained))) for x in samples)),
                candidate_admission_capacity_minus_availability_min=float(min(model.CAPACITY*candidate[2]-model.RELEASE_RATE*x[6] for x in samples)),
                retained_admission_capacity_minus_availability_min=float(min(model.CAPACITY*retained[2]-model.RELEASE_RATE*x[6] for x in samples))))
        rows.append(dict(file=name,accepted_plans=c['metrics']['accepted_plans'],
            rejected_plans=c['metrics']['rejected_certificates'],rejections=rejects,
            certified_metrics=c['independent_audit'],nominal_metrics=n['independent_audit'],
            max_command_difference=float(np.max(np.abs(np.array(c['trajectory']['command'])-n['trajectory']['command']))),
            max_state_difference_vehicles=float(100*np.max(np.abs(np.array(c['trajectory']['state'])-n['trajectory']['state']))),
            **result['comparison']))
    output=dict(passed=True,prior_live_files_unchanged=len(old['sha256']),
        prior_paper_files_unchanged=preserved_paper,archived_grids_unchanged=bool(args.prior_package),
        proof_descriptors_and_model_bindings_checked=proof_count,source_snapshots_checked=snapshots,
        independent_class_balance_episode_replays=2*len(rows),source_results_sha256=hashes,episodes=rows)
    path=RESULTS/'mfd_operating_range_aggregate_audit_results.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in ('episodes','source_results_sha256','prior_paper_files_unchanged')}))
    print(json.dumps(rows,indent=2))


def digest_bytes(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


if __name__=='__main__': main()
