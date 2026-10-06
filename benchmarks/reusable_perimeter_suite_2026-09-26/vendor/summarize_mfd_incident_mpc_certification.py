"""Bind the complete MPC certificate grid to its independent audits/comparators."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'reproducibility/results'


def read(name):
    return json.loads((RESULTS/name).read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    a = np.asarray(values)
    return dict(minimum=float(a.min()),median=float(np.median(a)),maximum=float(a.max()),mean=float(a.mean()))


def main():
    grid_name='mfd_incident_mpc_certified_grid_results.json'
    audit_name='mfd_incident_mpc_certified_grid_audit_results.json'
    grid,audit=read(grid_name),read(audit_name)
    old=read('mfd_incident_results_comparison.json')
    assert len(grid['runs']) == len(audit['cases']) == 8
    assert audit['passed'] and audit['source_sha256'] == digest(RESULTS/grid_name)
    regression=read('mfd_incident_mpc_adapter_regression_results.json')
    retention=read('mfd_incident_mpc_retention_audit_results.json')
    adverse=read('mfd_incident_mpc_adverse_fixed_plan_results.json')
    assert regression['passed'] and retention['passed'] and adverse['passed']
    assert len(adverse['cases']) == 48 and adverse['source_grid_sha256'] == digest(RESULTS/grid_name)
    cases=[]
    for case,checked in zip(grid['runs'],audit['cases']):
        key=(case['amplitude'],case['duration_h'])
        assert key == (checked['amplitude'],checked['duration_h'])
        comparator=next(c for c in old['cases'] if (c['amplitude'],c['duration_h']) == key)
        complete=case['metrics']['full_horizon_completed']
        tts=case['independent_replay']['total_time_spent_veh_h'] if complete else None
        nominal=comparator['mpc']['tts_veh_h']
        library=comparator['library']['tts_veh_h']
        # Both old MPC and new MPC use the same independent adaptive integrator
        # here; separately implemented class-balance replay is audited below.
        cases.append(dict(amplitude=key[0],duration_h=key[1],complete=complete,
                          nominal_mpc_tts_veh_h=nominal,certified_mpc_tts_veh_h=tts,
                          certified_library_tts_veh_h=library,
                          certification_tts_cost_percent=None if tts is None else 100*(tts/nominal-1),
                          tts_reduction_percent_of_library=None if tts is None else 100*(1-tts/library),
                          queue_time_veh_h=case['metrics']['queue_time_veh_h'],
                          peak_queue_vehicles=case['metrics']['maximum_queue_vehicles'],
                          deadline_n1_vehicles=None if case['metrics']['deadline_n1'] is None else 100*case['metrics']['deadline_n1'],
                          successful_certificates=case['metrics']['successful_certificates'],
                          failed_certificates=case['metrics']['failed_certificates'],
                          command_changes_including_initial=case['metrics']['first_command_changes_including_initial'],
                          admitted_unchanged_first_commands=case['metrics']['admitted_plans_with_unchanged_first_command'],
                          matched_archived_commands=case['comparison_to_archived_nominal_mpc'].get('maximum_command_difference') == 0.,
                          matched_archived_states=case['comparison_to_archived_nominal_mpc'].get('maximum_state_difference_units') == 0.,
                          independent_class_replay=checked['replay']))
    events=[e for c in grid['runs'] for e in c['decisions']]
    primary=next(c for c in grid['runs'] if c['amplitude'] == .3 and c['duration_h'] == 1.)
    first=primary['decisions'][0]['certificate']
    report=dict(scope='Nominal economic MPC performance plus finite-horizon execution certification under the unchanged uncertainty contract',
                source_sha256={name:digest(RESULTS/name) for name in (grid_name,audit_name,
                    'mfd_incident_results_comparison.json','mfd_incident_mpc_adapter_regression_results.json',
                    'mfd_incident_mpc_retention_audit_results.json','mfd_incident_mpc_adverse_fixed_plan_results.json')},
                cases=cases,complete_cases=sum(c['complete'] for c in cases),
                certificate_attempts=sum(c['metrics']['certificate_attempts'] for c in grid['runs']),
                accepted_certificates=sum(c['successful_certificates'] for c in cases),
                rejected_certificates=sum(c['failed_certificates'] for c in cases),
                all_nominal_commands_and_states_identical=all(c['matched_archived_commands'] and c['matched_archived_states'] for c in cases),
                total_decision_seconds=statistics([e['total_decision_seconds'] for e in events]),
                verification_seconds=statistics([e['certificate_seconds'] for e in events]),
                optimization_seconds=statistics([e['solve_seconds'] for e in events]),
                primary_initial_plan_deadline_upper_vehicles=100*first['service_check_results'][0]['upper'],
                adverse_fixed_plan_checks=dict(cases=48,deadline_n1_maximum_vehicles=max(c['deadline_n1_vehicles'] for c in adverse['cases']),
                                                maximum_queue_vehicles=max(c['maximum_queue_vehicles'] for c in adverse['cases']),
                                                all_checkpoint_memberships_passed=True),
                environment=dict(python=platform.python_version(),platform=platform.platform(),
                                 packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','mpmath')}),
                timing_scope='single-host prototype, some background validation overlapped; not an isolated performance benchmark or latency-aware physical proof',
                trajectory_scope='fresh state-dependent optimization, same nominal realizations; robust finite-plan certificates do not prove worst-case TTS optimality',
                paper_status='experiment supplement; preceding INCIDENT PDFs are unchanged',
                theorem_status='same finite-plan comparison proof; stable interval expm1 and exact zero-gain actuator supports',
                llm_status='deterministic MPC proposals; no new live calls')
    output=RESULTS/'mfd_incident_mpc_certification_summary_results.json'
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','source_sha256')},indent=2))
    for case in cases:
        print(case['amplitude'],case['duration_h'],case['certified_mpc_tts_veh_h'],case['tts_reduction_percent_of_library'])


if __name__ == '__main__':
    main()
