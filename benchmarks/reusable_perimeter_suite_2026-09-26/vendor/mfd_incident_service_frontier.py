"""Predeclared first-plan service frontier, with an unmodified plant/verifier.

Only the regional service threshold/deadline is parameterized relative to the
archived MPC. The default predictor regression checks every computed array.
No original module globals or result files are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import mfd_incident_model as plant
import mfd_incident_mpc as base
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_verifier import contains_information_box

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reproducibility/results/mfd_incident_service_frontier_results.json'
RETENTION_OUTPUT = ROOT / 'reproducibility/results/mfd_incident_service_retention_results.json'
ADVERSE_OUTPUT = ROOT / 'reproducibility/results/mfd_incident_service_adverse_results.json'
AUDIT_OUTPUT = ROOT / 'reproducibility/results/mfd_incident_service_frontier_audit_results.json'
PROTOCOL = ROOT / 'SERVICE_FRONTIER_PROTOCOL.md'
TARGETS_VEHICLES = [None, 3000, 2850, 2400, 2000, 1800, 1700, 1600, 1500]
PERIOD = Fraction(1, 12)


class ServicePrediction(base.Prediction):
    """Original prediction arithmetic; only the optional service row differs."""
    def __init__(self, state, now, horizon, arrival_fun=None, *,
                 service_target_vehicles=2850., deadline_h=1., cost_row=None,
                 priority_minutes=None):
        super().__init__(state, now, horizon, arrival_fun)
        self.service_target = None if service_target_vehicles is None else service_target_vehicles/100.
        self.deadline_h = float(deadline_h)
        self.cost_row = base.COST_ROW.copy() if cost_row is None else np.asarray(cost_row, dtype=float).copy()
        if self.cost_row.shape != (7,) or not np.all(np.isfinite(self.cost_row)) or np.any(self.cost_row < 0):
            raise ValueError('cost_row requires seven finite nonnegative coefficients')
        self.priority_minutes = priority_minutes
        if priority_minutes is not None and (priority_minutes <= 0 or
                abs(priority_minutes/60/self.step-round(priority_minutes/60/self.step)) > 1e-8):
            raise ValueError('priority duration must be positive and on the prediction grid')
        self.service_required = (self.service_target is not None and
                                 now < self.deadline_h-1e-10 and
                                 now+horizon >= self.deadline_h-1e-10)

    def evaluate(self, vector):
        if self.key is not None and np.array_equal(vector, self.key):
            return self.value
        self.calls += 1
        self.key = np.array(vector).copy()
        controls = np.asarray(vector).reshape(2, 4)
        x, sensitivity = self.state.copy(), np.zeros((7, 8))
        cost, grad = 0., np.zeros(8)
        constraints = [base.STATE_A@x+base.STATE_B, .75+base.CONTROL_A@vector]
        jacobians = [np.zeros((12, 8)), base.CONTROL_A]
        states = [x.copy()]
        target_prediction = None
        min_stage_margin = np.inf
        for k in range(self.nsteps):
            t = self.now+k*self.step
            block = 0 if k*self.step < base.DECISION-1e-10 else 1
            u = controls[block]
            embed = np.zeros((4, 8))
            embed[:, block*4:block*4+4] = np.eye(4)

            def rhs(y, sy, clock):
                f, fx, fu = base.field_jacobians(y, u, self.arrival_fun(clock))
                return f, fx@sy+fu@embed

            h = self.step
            cost_row = (self.cost_row if self.priority_minutes is None or
                        k*self.step < self.priority_minutes/60-1e-10 else base.COST_ROW)
            f1, s1 = rhs(x, sensitivity, t)
            x2, ss2 = x+h*f1/2, sensitivity+h*s1/2
            f2, s2 = rhs(x2, ss2, t+h/2)
            x3, ss3 = x+h*f2/2, sensitivity+h*s2/2
            f3, s3 = rhs(x3, ss3, t+h/2)
            x4, ss4 = x+h*f3, sensitivity+h*s3
            f4, s4 = rhs(x4, ss4, t+h)
            for ys, dys in ((x2, ss2), (x3, ss3), (x4, ss4)):
                sm = base.STATE_A@ys+base.STATE_B
                constraints.append(sm)
                jacobians.append(base.STATE_A@dys)
                min_stage_margin = min(min_stage_margin, float(sm.min()))
            cost += h/6*(cost_row@(x+2*x2+2*x3+x4))
            grad += h/6*(cost_row@(sensitivity+2*ss2+2*ss3+ss4))
            x = x+h/6*(f1+2*f2+2*f3+f4)
            sensitivity = sensitivity+h/6*(s1+2*s2+2*s3+s4)
            states.append(x.copy())
            constraints.append(base.STATE_A@x+base.STATE_B)
            jacobians.append(base.STATE_A@sensitivity)
            if self.service_required and abs(t+h-self.deadline_h) < 1e-8:
                target_prediction = float(x[0])
                constraints.append(np.array([self.service_target-x[0]]))
                jacobians.append(-sensitivity[[0]])
        if self.service_required and target_prediction is None:
            raise RuntimeError('The service deadline was not on the prediction grid')
        self.value = dict(cost=cost, gradient=grad, constraints=np.concatenate(constraints),
                          jacobian=np.vstack(jacobians), states=np.array(states),
                          target_n1=target_prediction, minimum_stage_margin=min_stage_margin)
        return self.value


def solve_service(state, now=0., library=None, warm=None, arrival_fun=None, *,
                  service_target_vehicles=2850., deadline_h=1., horizon_h=None,
                  cost_row=None, priority_minutes=None):
    """Original fixed three-start SLSQP; service contract is explicitly supplied."""
    library = plant.default_library() if library is None else library
    horizon = min(base.PREDICTION_HORIZON, base.EVALUATION_HORIZON-now) if horizon_h is None else horizon_h
    predictor = ServicePrediction(state, now, horizon, arrival_fun,
                                  service_target_vehicles=service_target_vehicles,
                                  deadline_h=deadline_h, cost_row=cost_row,
                                  priority_minutes=priority_minutes)
    starts = [np.tile(plant.BASE_COMMAND, 2), np.tile(library[1].feedforward, 2)]
    starts.insert(0, np.tile(warm if warm is not None else library[1].command(state), 2))
    candidates, attempts = [], []
    started = time.perf_counter()
    for guess in starts:
        begin = time.perf_counter()
        result = minimize(predictor.objective, guess, method='SLSQP', jac=True,
                          bounds=[(0., 1.)]*8,
                          constraints=[dict(type='ineq', fun=predictor.constraints,
                                            jac=predictor.constraint_jacobian)],
                          options=dict(maxiter=base.MAXITER, ftol=base.FTOL, disp=False))
        rec = predictor.evaluate(result.x)
        feasible = bool(np.all(np.isfinite(result.x)) and
                        min(rec['constraints']) >= -base.FEASIBILITY_TOLERANCE and
                        min(result.x) >= -base.FEASIBILITY_TOLERANCE and
                        max(result.x) <= 1+base.FEASIBILITY_TOLERANCE)
        attempts.append(dict(success=bool(result.success), status=int(result.status),
                             message=str(result.message), iterations=int(result.nit),
                             controls=result.x.reshape(2, 4).tolist(),
                             cost_veh_h=100*float(rec['cost']), sampled_feasible=feasible,
                             minimum_constraint_margin=float(min(rec['constraints'])),
                             solve_seconds=time.perf_counter()-begin))
        if result.success and feasible:
            candidates.append((float(rec['cost']), result.x.copy(), rec))
    selected = min(candidates, key=lambda r:r[0]) if candidates else None
    record = dict(time_h=now, horizon_h=horizon, solve_seconds=time.perf_counter()-started,
                  service_target_vehicles=service_target_vehicles, deadline_h=deadline_h,
                  objective_cost_row=predictor.cost_row.tolist(), priority_minutes=priority_minutes,
                  attempts=attempts, any_solver_success=any(r['success'] for r in attempts),
                  feasible_solution_found=selected is not None, rollout_evaluations=predictor.calls)
    if selected is not None:
        record.update(predicted_cost_veh_h=100*selected[0], controls=selected[1].reshape(2,4).tolist(),
                      predicted_target_n1=selected[2]['target_n1'],
                      minimum_prediction_margin=float(min(selected[2]['constraints'])))
    return selected, record


def regression_audit():
    checks = []
    rng = np.random.default_rng(92817)
    for now in [0., 1/12, .5, 1.]:
        for vector in [np.tile(plant.BASE_COMMAND, 2), rng.uniform(.3, .7, 8)]:
            old = base.Prediction(plant.BASE_STATE, now, 1.).evaluate(vector)
            new = ServicePrediction(plant.BASE_STATE, now, 1.).evaluate(vector)
            for key in old:
                assert np.array_equal(old[key], new[key]), (now, key)
            checks.append(dict(now_h=now, exact_array_equality=True))
    # Removing the service row changes no physics, cost, or sensitivities.
    vector = np.tile(plant.BASE_COMMAND, 2)
    default = ServicePrediction(plant.BASE_STATE, 0., 1.).evaluate(vector)
    none = ServicePrediction(plant.BASE_STATE, 0., 1., service_target_vehicles=None).evaluate(vector)
    for key in ['cost', 'gradient', 'states', 'minimum_stage_margin']:
        assert np.array_equal(default[key], none[key])
    assert np.array_equal(default['constraints'][:-1], none['constraints'])
    assert np.array_equal(default['jacobian'][:-1], none['jacobian'])
    priority_errors = []
    for row in ([4., 1., 0., 0., 0., 0., 1.], [1., 4., 0., 0., 0., 0., 1.]):
        for minutes in [15, 30, 45]:
            predictor = ServicePrediction(plant.BASE_STATE, 0., 1., cost_row=row, priority_minutes=minutes)
            analytic = predictor.evaluate(vector)['gradient'].copy()
            numerical = [(predictor.evaluate(vector+np.eye(8)[k]*1e-6)['cost']-
                          predictor.evaluate(vector-np.eye(8)[k]*1e-6)['cost'])/2e-6 for k in range(8)]
            error = float(np.max(abs(analytic-numerical)))
            assert error < 1e-4
            priority_errors.append(dict(cost_row=row, minutes=minutes, gradient_absolute_error=error))
    return dict(passed=True, exact_default_prediction_checks=checks,
                service_removal_changes_only_one_constraint_row=True,
                priority_gradient_checks=priority_errors,
                original_modules_unmodified=True)


def replay_plan(initial_state, commands, *, now_h=0., horizon_h=1., deadline_h=1.,
                service_target_vehicles=None, arrival_fun=None):
    """Adaptive physical replay of fixed commands; no continuous-time proof claim."""
    arrival_fun = base.arrival if arrival_fun is None else arrival_fun
    state = np.r_[initial_state, np.zeros(4)]
    times, states, margins = [], [], []
    deadline_n1 = None
    first_end = min(now_h+base.DECISION, now_h+horizon_h)
    edges = [(now_h, first_end, commands[0]), (first_end, now_h+horizon_h, commands[1])]
    for begin, end, command in edges:
        if end <= begin:
            continue
        u = np.asarray(command)

        def rhs(t, y):
            x = y[:7]
            return np.r_[plant.field(x, u, arrival_fun(t)), 100*(base.COST_ROW@x),
                         100*x[6], 100*plant.departure_rate(x, u),
                         100*(sum(plant.INTERNAL_DEMAND)+arrival_fun(t))]

        sol = solve_ivp(rhs, (begin, end), state, method='DOP853', rtol=1e-11,
                        atol=1e-12, dense_output=True, max_step=1/360)
        assert sol.success, sol.message
        grid = np.linspace(begin, end, max(2, round((end-begin)*3600/2)+1))
        xs = sol.sol(grid)[:7].T
        times.extend(grid.tolist())
        states.extend(xs.tolist())
        margins.extend([float(min(plant.margins(x, u))) for x in xs])
        if begin <= deadline_h <= end:
            deadline_n1 = float(100*sol.sol(deadline_h)[0])
        state = sol.y[:, -1]
    balance = 100*(base.COST_ROW@initial_state)+state[10]-100*(base.COST_ROW@state[:7])-state[9]
    return dict(scope='adaptive fixed-plan physical replay; physical margins sampled every two seconds',
                initial_time_h=now_h, horizon_h=horizon_h, deadline_h=deadline_h,
                total_time_spent_veh_h=float(state[7]), queue_time_veh_h=float(state[8]),
                completed_trips_vehicles=float(state[9]), maximum_queue_vehicles=float(100*np.max(np.array(states)[:, 6])),
                minimum_sampled_physical_margin=float(min(margins)),
                deadline_n1_vehicles=deadline_n1,
                nominal_service_pass=None if service_target_vehicles is None else bool(deadline_n1 <= service_target_vehicles+1e-5),
                nominal_service_excess_vehicles=None if service_target_vehicles is None else float(deadline_n1-service_target_vehicles),
                conservation_residual_vehicles=float(balance), final_state=state[:7].tolist(),
                trajectory=dict(time_h=times, state=states))


def run(targets=TARGETS_VEHICLES, deadline=Fraction(1), output=OUTPUT):
    frozen = ['reproducibility/mfd_incident_model.py', 'reproducibility/mfd_incident_mpc.py',
              'reproducibility/mfd_incident_verifier.py', 'reproducibility/mfd_incident_mpc_certification.py']
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    payload = dict(scope='predeclared first-plan regional service study',
                   protocol=str(PROTOCOL.relative_to(ROOT)), protocol_sha256=sha(PROTOCOL),
                   source_sha256=sha(Path(__file__)), preserved_sources={p: sha(ROOT/p) for p in frozen},
                   configuration=dict(arrival_amplitude=.3, arrival_duration_h=1.,
                                      horizon_h=1., deadline_h=str(deadline),
                                      service_targets_vehicles=targets, initial_box_vehicles=1.,
                                      arrival_residual_vehicles_per_hour=60., proof_step_seconds=15.,
                                      objective='one-hour queue-inclusive TTS', starts='unchanged original three-start protocol'),
                   implementation_audit=regression_audit(), runs=[])
    for target in targets:
        started = time.perf_counter()
        selected, report = solve_service(plant.BASE_STATE, service_target_vehicles=target,
                                         deadline_h=float(deadline))
        record = dict(target_vehicles=target, optimizer=report)
        if selected is None:
            record['classification'] = 'solver_failed_to_find_feasible_plan'
        else:
            commands = selected[1].reshape(2, 4)
            record['replay'] = replay_plan(plant.BASE_STATE, commands,
                                            deadline_h=float(deadline), service_target_vehicles=target)
            checks = [] if target is None else [dict(time_h=deadline, coordinate=0, upper=target/100.)]
            record['certificate'] = verify_command_plan(plant.BASE_STATE, commands,
                [PERIOD, Fraction(1)-PERIOD], service_checks=checks, retain_steps=True)
            certificate = record['certificate']
            predicted = report['predicted_target_n1']
            record['prediction_replay_deadline_difference_vehicles'] = (None if predicted is None else
                record['replay']['deadline_n1_vehicles']-100*predicted)
            record['classification'] = ('certified' if certificate['accepted'] else
                                        'nominally_predicted_feasible_but_not_certified')
        record['wall_seconds'] = time.perf_counter()-started
        payload['runs'].append(record)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')
        print(json.dumps(dict(target_vehicles=target, classification=record['classification'],
                              nominal_TTS=record.get('replay', {}).get('total_time_spent_veh_h'),
                              nominal_deadline=record.get('replay', {}).get('deadline_n1_vehicles'),
                              proof_service=record.get('certificate', {}).get('service_check_results'),
                              failures=record.get('certificate', {}).get('failures'),
                              wall_seconds=record['wall_seconds']), default=str), flush=True)
    baseline = next((r['replay']['total_time_spent_veh_h'] for r in payload['runs']
                     if r['target_vehicles'] is None and 'replay' in r), None)
    if baseline is not None:
        for record in payload['runs']:
            if 'replay' in record:
                delta = record['replay']['total_time_spent_veh_h']-baseline
                record['cost_against_no_service'] = dict(extra_TTS_veh_h=delta, extra_TTS_percent=100*delta/baseline)
    assert payload['preserved_sources'] == {p: sha(ROOT/p) for p in frozen}
    payload['all_original_source_hashes_unchanged'] = True
    output.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')
    return payload


def run_retention():
    """A real service-proof rejection under a single unchanged trusted contract.

    The more stringent 1600-vehicle nominal initialization is commissioned to
    certify the trusted 1700-vehicle task. This margin is disclosed, not free.
    The candidate replans after five minutes for the same trusted task. The
    retained branch executes the already certified tail only to its 1h expiry.
    """
    raw = OUTPUT.read_bytes()
    archive = json.loads(raw)
    backup = next(r for r in archive['runs'] if r['target_vehicles'] == 1600)
    commands = np.asarray(backup['optimizer']['controls'])
    checks = [dict(time_h=Fraction(1), coordinate=0, upper=17.)]
    cert = verify_command_plan(plant.BASE_STATE, commands, [PERIOD, Fraction(1)-PERIOD],
                               service_checks=checks, retain_steps=True)
    assert cert['accepted'], 'The commissioned strict backup must certify the trusted task'
    prefix = replay_plan(plant.BASE_STATE, commands, horizon_h=float(PERIOD))
    x = np.array(prefix['final_state'])
    shape = next(c for c in cert['reachable_checkpoints'] if c['exact_time_fraction'] == '1/12')
    membership = contains_information_box(shape, x, 0.)
    assert membership['passed']
    selected, report = solve_service(x, now=float(PERIOD), warm=commands[1],
                                     service_target_vehicles=1700., horizon_h=1.)
    result = dict(scope='single natural certificate-rejection and retained-suffix study',
        protocol=str(PROTOCOL.relative_to(ROOT)), protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frontier_source_sha256=hashlib.sha256(raw).hexdigest(),
        trusted_contract=dict(service_limit_vehicles=1700., deadline_h='1', initial_box_vehicles=1.,
                              residual_vehicles_per_hour=60., original_expiry_h='1',
                              same_contract_before_and_after_rejection=True),
        commissioning=dict(nominal_target_vehicles=1600., source='predeclared service-frontier target',
                           commands=commands.tolist(), certificate=cert),
        prefix=prefix, replacement_state=x.tolist(),
        membership_in_original_reachable_checkpoint=membership,
        replacement_optimizer=report)
    tail = np.tile(commands[1], (2, 1))
    result['retained_suffix_replay'] = replay_plan(x, tail, now_h=float(PERIOD),
        horizon_h=float(Fraction(1)-PERIOD), service_target_vehicles=1700.)
    result['retained_total_TTS_veh_h'] = prefix['total_time_spent_veh_h']+result['retained_suffix_replay']['total_time_spent_veh_h']
    # All exact five-minute checkpoints are checked against the same original
    # proof, using the adaptive physical trajectory; no fresh proof is substituted.
    membership_records = []
    retained = result['retained_suffix_replay']['trajectory']
    tt = np.asarray(retained['time_h'])
    for checkpoint in cert['reachable_checkpoints']:
        at = Fraction(checkpoint['exact_time_fraction'])
        if at < PERIOD:
            continue
        index = int(np.argmin(abs(tt-float(at))))
        assert abs(tt[index]-float(at)) < 1e-12
        audit = contains_information_box(checkpoint, retained['state'][index], 0.)
        assert audit['passed'], (at, audit)
        membership_records.append(dict(exact_time_h=str(at), proof_id=cert['proof_id'], **audit))
    result['retained_checkpoint_membership'] = membership_records
    if selected is None:
        result['outcome'] = 'replacement_optimizer_failed_to_find_feasible_plan'
    else:
        new = selected[1].reshape(2, 4)
        candidate = verify_command_plan(x, new, [PERIOD, Fraction(1)-PERIOD],
            initial_time_h=PERIOD, service_checks=checks, retain_steps=True)
        result['replacement_certificate'] = candidate
        result['replacement_replay_to_common_deadline'] = replay_plan(x, new, now_h=float(PERIOD),
            horizon_h=float(Fraction(1)-PERIOD), service_target_vehicles=1700.)
        nominal_tts = prefix['total_time_spent_veh_h']+result['replacement_replay_to_common_deadline']['total_time_spent_veh_h']
        result['nominal_replacement_total_TTS_veh_h'] = nominal_tts
        result['retention_TTS_premium_veh_h'] = result['retained_total_TTS_veh_h']-nominal_tts
        result['retention_TTS_premium_percent'] = 100*result['retention_TTS_premium_veh_h']/nominal_tts
        result['outcome'] = ('replacement_certified' if candidate['accepted'] else
                             'natural_certificate_rejection_retained_original_suffix')
    result['no_forced_outage'] = True
    result['no_unsafe_witness_claim'] = True
    result['original_expiry_preserved'] = True
    RETENTION_OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('outcome', 'retained_total_TTS_veh_h',
                                            'nominal_replacement_total_TTS_veh_h',
                                            'retention_TTS_premium_veh_h',
                                            'retention_TTS_premium_percent') if k in result}), flush=True)
    return result


def run_adverse():
    """Eight physically propagated perturbation pairs; no extra proof calls."""
    raw = RETENTION_OUTPUT.read_bytes()
    retained = json.loads(raw)
    cert = retained['commissioning']['certificate']
    shape = next(c for c in cert['reachable_checkpoints'] if c['exact_time_fraction'] == '1/12')
    nominal = np.asarray(retained['replacement_state'])
    candidate = np.asarray(retained['replacement_optimizer']['controls'])
    backup_commands = np.asarray(retained['commissioning']['commands'])
    tail = np.tile(retained['commissioning']['commands'][1], (2, 1))
    directions = [('zero', np.zeros(7)), ('n1', np.eye(7)[0]), ('n2', np.eye(7)[1]), ('all_coordinates', np.ones(7))]
    result = dict(scope='matched fixed-plan perturbation witnesses from initial physical errors at time zero',
                   retention_source_sha256=hashlib.sha256(raw).hexdigest(),
                   protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   interpretation='Same trusted deadline and disturbance family; sampled witnesses supplement, not replace, the proof',
                   nominal_replacement_service_excess_vehicles=retained['replacement_replay_to_common_deadline']['nominal_service_excess_vehicles'],
                   residual_schedule='zero before 5min, signed60veh/h from5min through1h',
                   original_proof_id=cert['proof_id'], original_expiry_h='1', runs=[])
    for name, direction in directions:
        for sign in [1., -1.]:
            delta = sign*.01*direction
            # A rounded x+0.01 can lie one ulp outside the exact binary 0.01
            # radius. Choose an inward adjacent representable corner instead.
            initial = np.where(delta == 0., plant.BASE_STATE,
                               np.nextafter(plant.BASE_STATE+delta, plant.BASE_STATE))
            exact_initial_membership = all(abs(Fraction.from_float(float(a))-Fraction.from_float(float(b))) <=
                                           Fraction.from_float(.01) for a, b in zip(initial, plant.BASE_STATE))
            assert exact_initial_membership
            prefix = replay_plan(initial, backup_commands, horizon_h=float(PERIOD))
            x = np.array(prefix['final_state'])
            membership = contains_information_box(shape, x, 0.)
            fresh = contains_information_box(dict(center=nominal.tolist(), metric_L=np.eye(7).tolist(),
                                                   radius=[.01]*7), x, 0.)
            record = dict(direction=name, sign=sign, requested_initial_error_vehicles=(100*delta).tolist(),
                           initial_state=initial.tolist(),
                           initial_error_vehicles=(100*(initial-plant.BASE_STATE)).tolist(),
                           exact_initial_information_box_membership=exact_initial_membership,
                           arrival_residual_vehicles_per_hour=60*sign,
                           prefix=prefix, actual_replacement_state=x.tolist(),
                           replacement_state_error_vehicles=(100*(x-nominal)).tolist(),
                           original_checkpoint_membership=membership,
                           fresh_information_box_membership=fresh)
            if not membership['passed']:
                record['excluded_reason'] = 'outside_original_checkpoint'
            elif not fresh['passed']:
                record['excluded_reason'] = 'outside_fresh_candidate_information_box'
            else:
                arrival_fun = lambda t, sign=sign: base.arrival(t)+sign*.6
                for label, commands in [('candidate', candidate), ('retained', tail)]:
                    replay = replay_plan(x, commands, now_h=float(PERIOD),
                        horizon_h=float(Fraction(1)-PERIOD), service_target_vehicles=1700., arrival_fun=arrival_fun)
                    record[label] = replay
                    record[label+'_whole_hour_TTS_veh_h'] = prefix['total_time_spent_veh_h']+replay['total_time_spent_veh_h']
                assert record['retained']['nominal_service_pass']
                assert record['retained']['minimum_sampled_physical_margin'] >= -1e-8
            result['runs'].append(record)
            print(json.dumps(dict(direction=name, sign=sign, eligible='excluded_reason' not in record,
                candidate_deadline=record.get('candidate', {}).get('deadline_n1_vehicles'),
                retained_deadline=record.get('retained', {}).get('deadline_n1_vehicles'))), flush=True)
    ADVERSE_OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def independent_fixed_replay(initial_state, commands, *, now_h=0., end_h=1., residual=0.):
    """Separate class-balance field, pulse construction, tight adaptive solve."""
    from audit_mfd_incident_model import reference_field
    from audit_mfd_incident_library_study import pulse
    current = np.r_[initial_state, np.zeros(3)]
    first_end = now_h+1/12
    cuts = sorted(set([now_h, end_h]+[v for v in [first_end, .25, .25+1/12, .25+1-1/12, 1.25]
                                      if now_h < v < end_h]))
    for left, right in zip(cuts[:-1], cuts[1:]):
        u = np.asarray(commands[0 if left < first_end-1e-10 else 1])

        def rhs(t, y):
            derivative, departures = reference_field(y[:7], u, pulse(t, .3, 1.)+residual)
            return np.r_[derivative, 100*sum(y[[0, 1, 6]]), 100*y[6], 100*departures]

        sol = solve_ivp(rhs, (left, right), current, method='DOP853', rtol=2e-12,
                        atol=2e-13, max_step=1/720)
        assert sol.success
        current = sol.y[:, -1]
    return dict(final_state=current[:7].tolist(), total_time_spent_veh_h=float(current[7]),
                queue_time_veh_h=float(current[8]), completed_trips_vehicles=float(current[9]),
                deadline_n1_vehicles=float(100*current[0]))


def audit_results():
    """Independent numerical replay and exact identity/provenance checks."""
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    records = [json.loads(path.read_bytes()) for path in [OUTPUT, RETENTION_OUTPUT, ADVERSE_OUTPUT]]
    frontier, retained, adverse = records
    for record in records:
        assert record['source_sha256'] == sha(Path(__file__))
        assert record['protocol_sha256'] == sha(PROTOCOL)
    assert retained['frontier_source_sha256'] == sha(OUTPUT)
    assert adverse['retention_source_sha256'] == sha(RETENTION_OUTPUT)
    assert frontier['preserved_sources'] == {p: sha(ROOT/p) for p in frontier['preserved_sources']}
    assert [r['target_vehicles'] for r in frontier['runs']] == TARGETS_VEHICLES
    proofs = [r['certificate'] for r in frontier['runs'] if 'certificate' in r]
    proofs += [retained['commissioning']['certificate'], retained['replacement_certificate']]
    for certificate in proofs:
        desc = certificate['proof_descriptor']
        assert hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest() == certificate['proof_id']
        assert desc['residual_bound'] == .6
        assert certificate['initial_box_units'] == [.01]*7
        assert certificate['adapter_audit']['zero_feedback_gain']
        assert all(a['passed'] for a in certificate['adapter_audit']['exact_actuator_checks'])
    errors = []

    def compare(label, computed, stored):
        error = dict(label=label, final_state_vehicles=100*float(np.max(abs(np.array(computed['final_state'])-stored['final_state']))),
                     total_time_spent_veh_h=abs(computed['total_time_spent_veh_h']-stored['total_time_spent_veh_h']),
                     deadline_vehicles=abs(computed['deadline_n1_vehicles']-stored['deadline_n1_vehicles']))
        assert max(value for key, value in error.items() if key != 'label') < 1e-4, error
        errors.append(error)

    for row in frontier['runs']:
        if 'replay' not in row:
            assert not row['optimizer']['feasible_solution_found']
            continue
        computed = independent_fixed_replay(plant.BASE_STATE, row['optimizer']['controls'])
        compare('primary_'+str(row['target_vehicles']), computed, row['replay'])
        assert abs(row['replay']['conservation_residual_vehicles']) < 1e-6
    contract = retained['trusted_contract']
    assert contract['same_contract_before_and_after_rejection'] and contract['original_expiry_h'] == '1'
    old, new = retained['commissioning']['certificate'], retained['replacement_certificate']
    assert old['accepted'] and not new['accepted']
    assert old['proof_descriptor']['service_checks'] == new['proof_descriptor']['service_checks']
    assert old['proof_descriptor']['model_contract'] == new['proof_descriptor']['model_contract']
    assert old['proof_descriptor']['pulse'] == new['proof_descriptor']['pulse']
    assert all(record['passed'] and record['proof_id'] == old['proof_id']
               for record in retained['retained_checkpoint_membership'])
    assert Fraction(old['proof_descriptor']['initial_time'])+sum(map(Fraction, old['proof_descriptor']['durations'])) == 1
    x = np.array(retained['replacement_state'])
    nominal = independent_fixed_replay(x, retained['replacement_optimizer']['controls'], now_h=1/12)
    compare('replacement_nominal', nominal, retained['replacement_replay_to_common_deadline'])
    tail = np.tile(retained['commissioning']['commands'][1], (2, 1))
    computed = independent_fixed_replay(x, tail, now_h=1/12)
    compare('retained_nominal', computed, retained['retained_suffix_replay'])
    usable = [r for r in adverse['runs'] if 'excluded_reason' not in r]
    assert len(adverse['runs']) == 8
    for index, row in enumerate(usable):
        assert all(abs(Fraction.from_float(float(a))-Fraction.from_float(float(b))) <= Fraction.from_float(.01)
                   for a, b in zip(row['initial_state'], plant.BASE_STATE))
        assert row['original_checkpoint_membership']['passed'] and row['fresh_information_box_membership']['passed']
        for label, commands in [('candidate', retained['replacement_optimizer']['controls']), ('retained', tail)]:
            computed = independent_fixed_replay(row['actual_replacement_state'], commands, now_h=1/12,
                                               residual=row['arrival_residual_vehicles_per_hour']/100.)
            compare('adverse_'+str(index)+'_'+label, computed, row[label])
            assert row[label]['minimum_sampled_physical_margin'] >= -1e-8
            assert abs(row[label]['conservation_residual_vehicles']) < 1e-6
    positive = [row['candidate']['nominal_service_excess_vehicles'] for row in usable if row['sign'] > 0]
    report = dict(passed=True, source_sha256=sha(Path(__file__)),
                   result_sha256={str(path.relative_to(ROOT)): sha(path) for path in [OUTPUT, RETENTION_OUTPUT, ADVERSE_OUTPUT]},
                   implementation_regression=regression_audit(), exact_proof_descriptors_checked=len(proofs),
                   source_provenance_and_same_contract_checked=True,
                   independent_replays=errors, adverse_attempts=len(adverse['runs']),
                   adverse_eligible=len(usable), adverse_excluded=len(adverse['runs'])-len(usable),
                   adverse_positive_excess_vehicles=positive,
                   unsafe_nominal_counterexample_scope='bounded-disturbance deadline failure, not physical-state or queue failure',
                   retained_nominal_TTS_premium_percent=retained['retention_TTS_premium_percent'])
    AUDIT_OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['passed', 'exact_proof_descriptors_checked',
                     'adverse_eligible', 'adverse_excluded', 'adverse_positive_excess_vehicles',
                     'retained_nominal_TTS_premium_percent']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--retention', action='store_true')
    parser.add_argument('--adverse', action='store_true')
    parser.add_argument('--audit-results', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(regression_audit(), indent=2))
    elif args.retention:
        run_retention()
    elif args.adverse:
        run_adverse()
    elif args.audit_results:
        audit_results()
    else:
        run()
