"""Bounded independent semantics probes, not an online performance experiment.

Three predeclared fresh-plan tests: nonzero time crossing an absolute pulse knot,
sound biased sensing, and an out-of-contract sensor bias. Each has the same
one-vehicle information width, 60veh/h residual and five-minute constant plan.
Also test exact correlated-set/outer-hull and rejected-initial-shape semantics.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import mfd_incident_model as model
import mfd_incident_affine_reachability as affine
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_service_frontier import independent_fixed_replay

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'reproducibility/results'
OUTPUT = RESULTS/'mfd_online_structure_semantics_audit_results.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_member(point, center, width):
    return all(abs(Fraction.from_float(float(x))-Fraction.from_float(float(c))) <=
               Fraction.from_float(float(w)) for x,c,w in zip(point,center,width))


def run():
    if OUTPUT.exists():
        raise FileExistsError('Preserve the existing semantics probe result')
    source_hashes = affine.source_hashes()
    source_hashes[Path(__file__).relative_to(ROOT).as_posix()] = digest(Path(__file__))
    source_hashes['ONLINE_STRUCTURE_PROOF_REVIEW.md'] = digest(ROOT/'ONLINE_STRUCTURE_PROOF_REVIEW.md')
    result = dict(scope='bounded independent semantic probes; not an online performance study',
        source_sha256=source_hashes,
        predeclared_cases=[dict(name='absolute_knot', now='7/24', bias_units=0.),
                           dict(name='sound_biased_sensor', now='5/12', bias_units=.005),
                           dict(name='out_of_contract_sensor', now='5/12', bias_units=.02)],
        cases=[])
    widths = np.full(7,.01)
    for spec in result['predeclared_cases']:
        now = Fraction(spec['now'])
        end = now+Fraction(1,12)
        actual = model.BASE_STATE.copy()
        center = actual.copy()
        center[0] += spec['bias_units']
        commands = [model.BASE_COMMAND.copy()]
        certificate = verify_command_plan(center, commands, [Fraction(1,12)],
            initial_time_h=now, initial_box=widths, residual_bound=.6,
            verifier=affine.AffineGeneratorVerifier(), retain_steps=True,
            service_checks=[dict(time_h=now,coordinate=0,upper=100.),
                            dict(time_h=end,coordinate=0,upper=100.)])
        assert certificate['accepted']
        desc = certificate['proof_descriptor']
        assert Fraction(desc['initial_time']) == now
        assert list(map(Fraction,desc['durations'])) == [Fraction(1,12)]
        assert [Fraction(v['exact_time']) for v in desc['service_checks']] == [now,end]
        assert len(certificate['service_check_results']) == 2
        assert all(r['passed'] for r in certificate['service_check_results'])
        assert certificate['adapter_audit']['commands'] == [model.BASE_COMMAND.tolist()]
        assert hashlib.sha256(json.dumps(desc,sort_keys=True).encode()).hexdigest() == certificate['proof_id']
        membership = exact_member(actual,center,widths)
        assert membership == (spec['name'] != 'out_of_contract_sensor')
        independent = independent_fixed_replay(actual, np.tile(model.BASE_COMMAND,(2,1)),
            now_h=float(now),end_h=float(end))
        row = dict(**spec, exact_end_h=str(end), actual_initial_state=actual.tolist(),
            measurement_center=center.tolist(), physical_information_half_widths=widths.tolist(),
            exact_actual_state_in_information_box=membership,
            certificate_proves_declared_box=True,
            certificate_applies_to_actual_initial_state_under_sensor_premise=membership,
            independent_nominal_replay=independent,
            certificate=certificate)
        if spec['name']=='absolute_knot':
            step_edges=[r['left_h'] for r in certificate['interval_steps']]+[certificate['interval_steps'][-1]['right_h']]
            actual_knot=Fraction(1,4)+Fraction.from_float(1/12)
            assert any(abs(t-float(actual_knot))<1e-15 for t in step_edges)
            row['absolute_pulse_knot_present']=str(actual_knot)
        result['cases'].append(row)
    # Exact outer-hull counterexample: original correlation x1=x2 is lost.
    point=(Fraction(3),Fraction(1))
    outer_member=all(Fraction(1)<=v<=Fraction(3) for v in point)
    correlated_member=point[0]==point[1]
    violated_row=point[0]-point[1]
    assert outer_member and not correlated_member and violated_row>0
    result['outer_hull_counterexample']=dict(outer_hull_member=outer_member,
        correlated_set_member=correlated_member, row_excess_exact=str(violated_row),
        conclusion='Outer-hull membership does not authorize correlated-set restart')
    try:
        affine.AffineGeneratorVerifier().verify_plan(model.BASE_STATE, [], initial_shape={})
    except ValueError as exc:
        assert 'initial box' in str(exc)
        result['initial_shape_restart_rejected']=True
    else:
        raise AssertionError('Correlated initial-shape restart unexpectedly accepted')
    for name,sha in source_hashes.items():
        assert digest(ROOT/name)==sha, ('source_drift',name)
    result['passed']=True
    OUTPUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True, fresh_plan_proofs=3, nonzero_exact_clock_tests=3,
        sound_biased_sensor=True, out_of_contract_sensor_identified=True,
        outer_hull_counterexample=True, correlated_restart_rejected=True,
        result_sha256=digest(OUTPUT)),indent=2))
    return result


if __name__=='__main__':
    run()
