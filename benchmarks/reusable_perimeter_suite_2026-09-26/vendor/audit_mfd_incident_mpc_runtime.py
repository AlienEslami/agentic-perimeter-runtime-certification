"""Independent exact-contract, numerical replay and failure-path MPC audits.

This is not a second implementation of interval integration. It reconstructs
controller/constraint/time bindings, checks outward records, and uses a separate
class-balance ODE implementation to audit the actual executed schedule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from mpmath import iv, mp
from scipy.integrate import solve_ivp

import mfd_incident_mpc_runtime as runtime
import mfd_incident_mpc_certification as adapter
import mfd_incident_verifier as verifier
from audit_mfd_incident_certified_runtime import exact_affine_contract, mode_fingerprint
from audit_mfd_incident_model import reference_field
from audit_mfd_incident_library_study import pulse, INTERNAL_TOTAL
from audit_mfd_incident_operational_study import constraints

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'reproducibility/results'
F = lambda v: Fraction.from_float(float(v))
L = np.array([[0,0,1,0,0,0,0], [1,0,-1,0,-1,0,0], [0,0,0,0,1,0,0],
              [0,0,0,1,0,0,.5], [0,1,0,-1,0,-1,.5], [0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1]], dtype=float)


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def exact_gate_margins(command):
    u = list(map(F, command))
    residuals = [r for v in u for r in (1-v, v)]
    residuals += [r for a,b in ((1,0),(3,2)) for r in (Fraction(3,4)-u[a]+u[b], Fraction(3,4)+u[a]-u[b])]
    names = [f'{name}_{side}' for name in ('u12','u21','u02','u20') for side in ('upper','lower')]
    names += [f'direction_{a}_{b}_{side}' for a,b in ((1,0),(3,2)) for side in ('upper','lower')]
    return dict(zip(names, residuals))


def audit_certificate(cert, decision, case):
    assert cert['accepted'] and cert['verified_continuous_physical'] and cert['verified_scheduled_service']
    desc = cert['proof_descriptor']
    assert hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest() == cert['proof_id']
    now = Fraction(decision['exact_time_h'])
    end = min(Fraction(2), now+1)
    lengths = list(map(Fraction, desc['durations']))
    expected = [d for d in (Fraction(1,12), end-now-Fraction(1,12)) if d > 0]
    assert lengths == expected and Fraction(desc['initial_time']) == now
    assert cert['certified_until_h'] == cert['enclosed_until_h'] == float(end)
    assert not cert['failures'] and cert['initial_box_units'] == [.01]*7
    assert desc['initial_state'] == decision['initial_state']
    assert desc['initial_L'] == L.tolist()
    for row, radius in zip(L, desc['initial_radius']):
        assert F(radius) >= sum(abs(F(a))*F(.01) for a in row)
    assert desc['residual_bound'] == .6 and desc['step_seconds'] == 15.
    assert desc['metric_strategy'] == 'conservation_box'
    assert list(map(Fraction, desc['pulse'])) == [F(case['amplitude']), F(case['duration_h']), Fraction(1,4), F(1/12)]
    contract = desc['model_contract']
    expected_contract = dict(equations='seven_state_class_conservation_with_min_admission_v1',
                             state_order=['n1','n2','n11','n22','n10','n20','exterior_queue'],
                             MFD_coefficients=[.0014877,-.29815,15.0912],
                             # Exact installed binary values, frozen by the preceding
                             # incident certificates (not rounded displayed demands).
                             internal_demand=[74.85629999999998,37.42815000000003,18.714074999999998,
                                              44.856299999999976,7.428150000000002,.7485630000000008],
                             admission_capacity=120., release_rate=60., base_arrival=60., queue_limit=20.,
                             regional_jam_limits=[100.,100.], command_box=[0.,1.],
                             directional_limit=.75, admission_split=.5)
    assert contract == expected_contract, contract
    scheduled = [dict(exact_time='1',coordinate=0,upper=28.5)] if now < 1 else []
    assert desc['service_checks'] == scheduled
    assert len(cert['service_check_results']) == len(scheduled)
    assert all(s['passed'] and s['upper'] <= 28.5 for s in cert['service_check_results'])
    expected_cuts = [Fraction(k,12) for k in range(25) if now < Fraction(k,12) < end]+([Fraction(1)] if scheduled else [])
    assert list(map(Fraction, desc['timecuts'])) == expected_cuts
    commands = decision['candidate_commands'][:len(lengths)]
    assert cert['adapter_audit']['commands'] == commands
    support = desc['exact_constant_command_support']
    assert support['rule'] == 'zero_gain_constant_actuator_residuals_v1'
    assert support['zero_gain_checked'] and support['physical_state_supports_unchanged']
    equalities = 0
    for j, command in enumerate(commands):
        record = dict(center=[0.]*7, feedforward=command, gain=np.zeros((4,7)).tolist(),
                      A=np.zeros((7,7)).tolist(), P=np.eye(7).tolist())
        fingerprint = mode_fingerprint(record)
        assert desc['mode_fingerprints'][j] == fingerprint
        affine = desc['affine_constraint_contracts'][j]
        assert affine['mode_fingerprint'] == fingerprint
        exact_affine_contract(affine, record)
        exact = exact_gate_margins(command)
        stored = support['controller_contracts'][j]
        assert stored['mode_fingerprint'] == fingerprint and stored['passed']
        assert {key:Fraction(value) for key,value in stored['exact_margins'].items()} == exact
        assert all(v >= 0 for v in exact.values())
        equalities += sum(v == 0 for v in exact.values())
    assert len(cert['continuous_constraint_margins_lower']) == 24
    assert min(cert['continuous_constraint_margins_lower'].values()) >= 0
    checkpoints = {Fraction(c['exact_time_fraction']):c for c in cert['reachable_checkpoints']}
    assert all(Fraction(k,12) in checkpoints for k in range(25) if now <= Fraction(k,12) <= end)
    for at, shape in checkpoints.items():
        assert now <= at <= end and shape['proof_id'] == cert['proof_id']
        assert shape['metric_L'] == L.tolist() and min(shape['radius']) >= 0
    return dict(proof_id=cert['proof_id'], command_blocks=len(commands), exact_gate_equalities=equalities,
                affine_contract_rows=24*len(commands), checkpoints=len(checkpoints))


def audit_binding(case):
    records, retained = [], []
    by_epoch = {}
    for k, event in enumerate(case['decisions']):
        now = Fraction(event['exact_time_h'])
        following = Fraction(event['exact_next_decision_h'])
        assert now == Fraction(k,12) and following == now+Fraction(1,12)
        assert event['initial_state'] == case['trajectory']['state'][30*k]
        cert = event['certificate']
        if cert and cert['accepted']:
            records.append(audit_certificate(cert, event, case))
        if event['candidate_admitted']:
            assert cert['accepted'] and not event['forced_verifier_outage']
            assert event['executed_constant_command'] == event['candidate_commands'][0]
            assert event['retained_plan_origin_epoch'] == k
            assert event['retained_plan_proof_id'] == cert['proof_id']
            assert Fraction(event['retained_plan_exact_expiry_h']) == min(Fraction(2), now+1)
            by_epoch[k] = event
        elif event['execution_source'] == 'retained_certified_suffix':
            origin = by_epoch[event['retained_plan_origin_epoch']]
            parent = origin['certificate']
            assert event['executed_constant_command'] == origin['candidate_commands'][1]
            expiry = Fraction(origin['exact_time_h'])+sum(map(Fraction,parent['proof_descriptor']['durations']))
            assert Fraction(event['retained_plan_exact_expiry_h']) == expiry >= following
            assert event['retained_plan_proof_id'] == parent['proof_id']
            shapes = {Fraction(c['exact_time_fraction']):c for c in parent['reachable_checkpoints']}
            for at, index in ((now,30*k),(following,30*(k+1))):
                check = verifier.contains_information_box(shapes[at],case['trajectory']['state'][index],0.)
                assert check['passed'], (k,at,check)
            retained.append(dict(epoch=k, original_epoch=origin['epoch'], proof_id=parent['proof_id'], expiry=str(expiry)))
        else:
            assert event['execution_source'] == 'no_authorized_initial_plan_or_suffix'
            assert event['executed_constant_command'] is None
            assert case['metrics']['stopped']['reason'] == 'unresolved_continuation'
            continue
        executed = np.asarray(case['trajectory']['command'][30*k:30*(k+1)])
        assert executed.shape == (30,4) and np.array_equal(executed,np.tile(event['executed_constant_command'],(30,1)))
    return dict(certificates=records, retained=retained)


def independent_replay(case):
    y = np.r_[30.,30.,15.,15.,5.,5.,2.,np.zeros(4)]
    maximum_error = minimum = 0.
    max_queue, service = 200., None
    for event in case['decisions']:
        if event['executed_constant_command'] is None:
            continue
        a,b = float(Fraction(event['exact_time_h'])),float(Fraction(event['exact_next_decision_h']))
        command = np.asarray(event['executed_constant_command'])
        def rhs(t,z):
            arrival = pulse(t,case['amplitude'],case['duration_h'])
            dx,out = reference_field(z[:7],command,arrival)
            return np.r_[dx,100*sum(z[[0,1,6]]),100*z[6],100*out,100*(INTERNAL_TOTAL+arrival)]
        solution = solve_ivp(rhs,(a,b),y,method='DOP853',rtol=1e-11,atol=1e-12,max_step=1/360,dense_output=True)
        assert solution.success
        samples = solution.sol(np.linspace(a,b,31))[:7].T
        stored = np.asarray(case['trajectory']['state'][30*event['epoch']:30*(event['epoch']+1)+1])
        maximum_error = max(maximum_error,float(np.max(abs(samples-stored))))
        for state in solution.sol(np.linspace(a,b,151))[:7].T:
            minimum = min(minimum,*(float(np.min(v)) for v in constraints(state,command)))
            max_queue = max(max_queue,100*state[6])
        if a < 1 <= b:
            service = float(solution.sol(1.)[0])*100
        y = solution.y[:,-1]
    balance = 6200+y[10]-100*sum(y[[0,1,6]])-y[9]
    assert abs(balance) < 1e-6 and minimum >= -1e-9
    assert maximum_error < 2e-4, maximum_error
    metrics = case['metrics']
    difference = dict(tts_veh_h=float(y[7]-metrics['total_time_spent_veh_h']),
                      queue_veh_h=float(y[8]-metrics['queue_time_veh_h']),
                      departures_vehicles=float(y[9]-metrics['completed_trips_vehicles']))
    assert abs(difference['tts_veh_h']) < .02 and abs(difference['queue_veh_h']) < .02
    assert abs(difference['departures_vehicles']) < .1
    return dict(total_time_spent_veh_h=float(y[7]),queue_time_veh_h=float(y[8]),
                completed_trips_vehicles=float(y[9]),maximum_queue_vehicles=float(max_queue),
                deadline_n1_vehicles=service,minimum_sampled_margin=minimum,
                maximum_state_error_units=maximum_error,conservation_residual_vehicles=float(balance),
                difference_from_runtime=difference, independent_class_balance_rhs=True)


def regression_checks():
    mp.dps = 420
    rates = [0.,7.20185e-318,-7.20185e-318,1e-20,-1e-20,1.,-60.]
    records = []
    h = 1/240
    for rate in rates:
        value = iv.mpf(h) if rate == 0 else iv.expm1(iv.mpf(rate)*iv.mpf(h))/iv.mpf(rate)
        exact = mp.mpf(h) if rate == 0 else mp.expm1(mp.mpf(rate)*mp.mpf(h))/mp.mpf(rate)
        lo,hi = float(value.a),float(value.b)
        assert mp.mpf(lo) <= exact <= mp.mpf(hi) and np.isfinite([lo,hi]).all()
        records.append(dict(rate=rate,interval=[lo,hi],width=hi-lo))
    # Exact equality is feasible; a one-ULP violation is rejected, not rounded in.
    checks = []
    for command, expected in (([1.,1.,1.,1.],True),([0.,0.,0.,0.],True),
                              ([1.,.25,.25,1.],True),([np.nextafter(1.,np.inf),1.,1.,1.],False),
                              ([1.,np.nextafter(.25,0.),.25,1.],False)):
        independent = exact_gate_margins(command)
        actual = adapter.exact_command_constraints(command)
        assert all(v>=0 for v in independent.values()) == expected == actual['passed']
        assert all(F(adapter.rational_lower(v)) <= v for v in independent.values())
        cert = adapter.verify_command_plan([30.,30.,15.,15.,5.,5.,2.],[command],[Fraction(1,3600)],
                                           initial_time_h=Fraction(1),retain_steps=True)
        assert cert['accepted'] == expected, (command,cert['failures'])
        # Verify only actuator supports change in adapter step.
        if cert['interval_steps']:
            step = cert['interval_steps'][0]
            modes = [adapter.constant_command_mode(command)]
            raw = verifier.PlanVerifier(modes,metric_strategy='conservation_box')
            radius = np.asarray(cert['proof_descriptor']['initial_radius'])
            nc,nr,original = raw.step(np.array([30.,30.,15.,15.,5.,5.,2.]),radius,modes[0],Fraction(1),Fraction(3601,3600),verifier.Pulse(.3,1.),.6)
            assert step['continuous_constraint_margins_lower'][:12] == original['continuous_constraint_margins_lower'][:12]
            assert step['radius'] == nr.tolist()
        checks.append(dict(command=command,accepted=cert['accepted'],expected=expected))
    installed = runtime.operating.library()
    old_profile = verifier.PlanVerifier(installed,metric_strategy='conservation_box').verify_plan(
        [30.,30.,15.,15.,5.,5.,2.],[installed[0]],[Fraction(1)],amplitude=.3,duration=1.,
        initial_box=.01,residual_bound=.6,step_seconds=15.,
        service_checks=[dict(time_h=Fraction(1),coordinate=0,upper=28.5)])
    assert old_profile['accepted']
    return dict(comparison_integral=records, exact_gate_cases=checks,
                state_enclosure_unchanged_by_exact_gate_adapter=True,
                original_feedback_profile_regression=old_profile)


def retained_tests():
    def injected(epochs):
        def verify(state,commands,durations,**kwargs):
            epoch = int(kwargs['initial_time_h']*12)
            if epoch in epochs:
                return dict(accepted=False,proof_id=None,test_only_verifier_outage=True)
            return adapter.verify_command_plan(state,commands,durations,**kwargs)
        return verify
    tests = []
    for label,epochs,horizon in (('recovery',set(range(1,4)),5),('expiry',set(range(1,24)),24)):
        case = runtime.run(verify_fn=injected(epochs),force_rejection_epochs=epochs,max_epochs=horizon)
        check = audit_binding(case)
        if label == 'expiry':
            assert abs(case['metrics']['completed_horizon_h']-1.) < 1e-12
            assert case['metrics']['retained_suffix_epochs'] == 11
            assert case['metrics']['stopped']['reason'] == 'unresolved_continuation'
        else:
            assert case['decisions'][4]['candidate_admitted'] and len(check['retained']) == 3
        tests.append(dict(test=label,run=case,binding_audit=check,numerical_replay=independent_replay(case)))
    output = RESULTS/'mfd_incident_mpc_retention_audit_results.json'
    save(output,dict(passed=True,tests=tests,scope='forced verification outage; no physical safe-stop claim beyond expiry'))
    print(output)


def adverse_fixed_plan():
    """48 independent uncertain replays of the primary initial prescribed plan."""
    grid = json.loads((RESULTS/'mfd_incident_mpc_certified_grid_results.json').read_text())
    case = next(c for c in grid['runs'] if c['amplitude'] == .3 and c['duration_h'] == 1.)
    cert = case['decisions'][0]['certificate']
    assert cert['accepted']
    commands = case['decisions'][0]['candidate_commands']
    shapes = {Fraction(c['exact_time_fraction']):c for c in cert['reachable_checkpoints']}
    base = np.array([30.,30.,15.,15.,5.,5.,2.])
    records = []
    for corner in range(16):
        signs = np.array([1. if corner & (1 << (j % 4)) else -1. for j in range(7)])
        # Round each exact box corner toward the box interior for numerical ODEs.
        initial = np.array([float(F(b)+F(s)*F(.01)) for b,s in zip(base,signs)])
        for j in range(7):
            if abs(F(initial[j])-F(base[j])) > F(.01):
                initial[j] = np.nextafter(initial[j],base[j])
            assert abs(F(initial[j])-F(base[j])) <= F(.01)
        for kind in ('positive','negative','sinusoidal'):
            x = initial.copy()
            maximum_queue = 100*x[6]
            minimum = np.inf
            checked = 0
            for k in range(12):
                a,b = Fraction(k,12),Fraction(k+1,12)
                u = np.array(commands[0 if k == 0 else 1])
                def rhs(t,z):
                    residual = .6 if kind == 'positive' else -.6 if kind == 'negative' else .6*np.sin(8*np.pi*t)
                    return reference_field(z,u,pulse(t,.3,1.)+residual)[0]
                solution = solve_ivp(rhs,(float(a),float(b)),x,method='DOP853',rtol=1e-10,atol=1e-12,
                                     max_step=1/180,dense_output=True)
                assert solution.success
                for state in solution.sol(np.linspace(float(a),float(b),31)).T:
                    minimum = min(minimum,*(float(np.min(v)) for v in constraints(state,u)))
                    maximum_queue = max(maximum_queue,100*state[6])
                for at,point in ((a,x),(b,solution.y[:,-1])):
                    membership = verifier.contains_information_box(shapes[at],point,0.)
                    assert membership['passed'], (corner,kind,at,membership)
                    checked += 1
                x = solution.y[:,-1]
            assert minimum >= -1e-9 and 100*x[0] <= 2850
            records.append(dict(corner=corner,initial_state=initial.tolist(),residual_pattern=kind,
                                deadline_n1_vehicles=float(100*x[0]),maximum_queue_vehicles=float(maximum_queue),
                                minimum_sampled_margin=float(minimum),checkpoint_membership_checks=checked))
    output=RESULTS/'mfd_incident_mpc_adverse_fixed_plan_results.json'
    save(output,dict(passed=True,scope='16 inward-rounded box vertices x 3 residual patterns; prescribed first plan, not repeated MPC optimization',
                     certificate_id=cert['proof_id'],source_grid_sha256=runtime.digest(RESULTS/'mfd_incident_mpc_certified_grid_results.json'),
                     independent_class_balance_rhs=True,numerical_falsification_supplements_interval_proof=True,cases=records))
    print('PASS adverse fixed-plan replays',len(records),output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary',action='store_true')
    parser.add_argument('--regression',action='store_true')
    parser.add_argument('--retention',action='store_true')
    parser.add_argument('--adverse',action='store_true')
    args = parser.parse_args()
    if args.regression:
        result = regression_checks()
        save(RESULTS/'mfd_incident_mpc_adapter_regression_results.json',dict(passed=True,**result))
        print('PASS exact actuator / stable comparison regression')
        return
    if args.retention:
        retained_tests()
        return
    if args.adverse:
        adverse_fixed_plan()
        return
    path = RESULTS/f'mfd_incident_mpc_certified_{"primary" if args.primary else "grid"}_results.json'
    payload = json.loads(path.read_text())
    for name,expected in payload['source_sha256'].items():
        assert runtime.digest(ROOT/name) == expected, name
    cases=[]
    for case in payload['runs']:
        assert case['configuration']['amplitude'] == case['amplitude']
        assert case['configuration']['incident_duration_h'] == case['duration_h']
        check = audit_binding(case)
        replay = independent_replay(case)
        cases.append(dict(amplitude=case['amplitude'],duration_h=case['duration_h'],binding=check,replay=replay))
    report=dict(passed=True,source_sha256=runtime.digest(path),cases=cases,
                accepted_certificates=sum(len(c['binding']['certificates']) for c in cases),
                guarantee_scope='finite-horizon source proof, independent exact-contract and numerical trajectory audits')
    output=RESULTS/f'mfd_incident_mpc_certified_{"primary" if args.primary else "grid"}_audit_results.json'
    save(output,report)
    print(json.dumps(dict(passed=True,cases=len(cases),certificates=report['accepted_certificates'],output=str(output))))


if __name__ == '__main__':
    main()
