"""Constrained continuous-input economic MPC for the new seven-state plant.

Frozen pilot: 5-minute decisions, 60-minute horizon, two control blocks (the
first 5 minutes and remaining horizon), 1-minute RK4 prediction, 10-second
physical integration, known 30% one-hour trapezoidal arrival, n1(1h)<=28.5.
Constraints are numerically checked, not robust or continuous-time proofs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import mfd_incident_model as plant

DECISION = 1.0 / 12.0
PREDICTION_HORIZON = 1.0
PREDICTION_STEP = 1.0 / 60.0
EXECUTION_STEP = 1.0 / 360.0
EVALUATION_HORIZON = 2.0
AMPLITUDE = 0.3
DURATION = 1.0
DEADLINE = 1.0
TARGET_N1 = 28.5
MAXITER = 100
FTOL = 1e-8
FEASIBILITY_TOLERANCE = 1e-7
STATE_A = np.vstack((np.eye(7), [1, 0, -1, 0, -1, 0, 0],
                     [0, 1, 0, -1, 0, -1, 0],
                     [-1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, -1]))
STATE_B = np.r_[np.zeros(9), 100., 100., plant.QUEUE_LIMIT]
PAIR_A = np.array([[1, -1, 0, 0], [-1, 1, 0, 0],
                   [0, 0, 1, -1], [0, 0, -1, 1]], dtype=float)
CONTROL_A = np.zeros((8, 8))
CONTROL_A[:4, :4] = PAIR_A
CONTROL_A[4:, 4:] = PAIR_A
COST_ROW = np.array([1., 1., 0., 0., 0., 0., 1.])


def arrival(t):
    return plant.arrival(t, amplitude=AMPLITUDE, duration=DURATION)


def field_jacobians(x, u, offered):
    """Exact branchwise derivatives, with the plant's capacity-side tie rule."""
    n1, n2, n11, n22, n10, n20, z = x
    h1 = .0014877*n1*n1-.29815*n1+15.0912
    h2 = .0014877*n2*n2-.29815*n2+15.0912
    hp1, hp2 = .0029754*n1-.29815, .0029754*n2-.29815
    n12, n21 = n1-n11-n10, n2-n22-n20
    q11, q12, q10 = n11*h1, n12*h1, n10*h1
    q22, q21, q20 = n22*h2, n21*h2, n20*h2
    a11 = np.array([n11*hp1, 0, h1, 0, 0, 0, 0.])
    a12 = np.array([h1+n12*hp1, 0, -h1, 0, -h1, 0, 0.])
    a10 = np.array([n10*hp1, 0, 0, 0, h1, 0, 0.])
    a22 = np.array([0, n22*hp2, 0, h2, 0, 0, 0.])
    a21 = np.array([0, h2+n21*hp2, 0, -h2, 0, -h2, 0.])
    a20 = np.array([0, n20*hp2, 0, 0, 0, h2, 0.])
    u12, u21, u02, u20 = u
    if plant.RELEASE_RATE*z < plant.CAPACITY*u02:
        admitted, az, au = plant.RELEASE_RATE*z, plant.RELEASE_RATE, 0.
    else:
        admitted, az, au = plant.CAPACITY*u02, 0., plant.CAPACITY
    adm_x = np.array([0., 0, 0, 0, 0, 0, az])
    d11, d12, d10, d22, d21, d20 = plant.INTERNAL_DEMAND
    f = np.array([-q11-(q12+q10)*u12+q21*u21+d11+d12+d10,
                  -q22+(q12+q10)*u12-q21*u21+admitted-q20*u20+d22+d21+d20,
                  -q11+q21*u21+d11, -q22+q12*u12+.5*admitted+d22,
                  -q10*u12+d10, -q20*u20+q10*u12+d20, offered-admitted])
    fx = np.array([-a11-(a12+a10)*u12+a21*u21,
                   -a22+(a12+a10)*u12-a21*u21+adm_x-a20*u20,
                   -a11+a21*u21, -a22+a12*u12+.5*adm_x,
                   -a10*u12, -a20*u20+a10*u12, -adm_x])
    fu = np.array([[-q12-q10, q21, 0., 0], [q12+q10, -q21, au, -q20],
                   [0, q21, 0, 0], [q12, 0, .5*au, 0], [-q10, 0, 0, 0],
                   [q10, 0, 0, -q20], [0, 0, -au, 0]])
    return f, fx, fu


class Prediction:
    """Cached RK4 state, cost, and exact forward-sensitivity rollout."""
    def __init__(self, state, now, horizon, arrival_fun=None):
        self.state = np.asarray(state).copy()
        self.now, self.horizon = now, horizon
        self.arrival_fun = arrival if arrival_fun is None else arrival_fun
        self.key = None
        self.value = None
        self.calls = 0
        self.nsteps = round(horizon/PREDICTION_STEP)
        self.step = horizon/self.nsteps
        self.service_required = now < DEADLINE-1e-10 and now+horizon >= DEADLINE-1e-10

    def evaluate(self, vector):
        if self.key is not None and np.array_equal(vector, self.key):
            return self.value
        self.calls += 1
        self.key = np.array(vector).copy()
        controls = np.asarray(vector).reshape(2, 4)
        x, sensitivity = self.state.copy(), np.zeros((7, 8))
        cost, grad = 0., np.zeros(8)
        constraints = [STATE_A@x+STATE_B, .75+CONTROL_A@vector]
        jacobians = [np.zeros((12, 8)), CONTROL_A]
        states = [x.copy()]
        target_prediction = None
        min_stage_margin = np.inf
        for k in range(self.nsteps):
            t = self.now+k*self.step
            block = 0 if k*self.step < DECISION-1e-10 else 1
            u = controls[block]
            embed = np.zeros((4, 8))
            embed[:, block*4:block*4+4] = np.eye(4)
            def rhs(y, sy, clock):
                f, fx, fu = field_jacobians(y, u, self.arrival_fun(clock))
                return f, fx@sy+fu@embed
            h = self.step
            f1, s1 = rhs(x, sensitivity, t)
            x2, ss2 = x+h*f1/2, sensitivity+h*s1/2
            f2, s2 = rhs(x2, ss2, t+h/2)
            x3, ss3 = x+h*f2/2, sensitivity+h*s2/2
            f3, s3 = rhs(x3, ss3, t+h/2)
            x4, ss4 = x+h*f3, sensitivity+h*s3
            f4, s4 = rhs(x4, ss4, t+h)
            # RK stages and accepted nodes are constrained; still not an enclosure.
            for ys, dys in ((x2, ss2), (x3, ss3), (x4, ss4)):
                sm = STATE_A@ys+STATE_B
                constraints.append(sm)
                jacobians.append(STATE_A@dys)
                min_stage_margin = min(min_stage_margin, float(sm.min()))
            cost += h/6*(COST_ROW@(x+2*x2+2*x3+x4))
            grad += h/6*(COST_ROW@(sensitivity+2*ss2+2*ss3+ss4))
            x = x+h/6*(f1+2*f2+2*f3+f4)
            sensitivity = sensitivity+h/6*(s1+2*s2+2*s3+s4)
            states.append(x.copy())
            constraints.append(STATE_A@x+STATE_B)
            jacobians.append(STATE_A@sensitivity)
            if self.service_required and abs(t+h-DEADLINE) < 1e-8:
                target_prediction = float(x[0])
                constraints.append(np.array([TARGET_N1-x[0]]))
                jacobians.append(-sensitivity[[0]])
        if self.service_required and target_prediction is None:
            raise RuntimeError('The fixed service deadline was not on the prediction grid')
        self.value = dict(cost=cost, gradient=grad, constraints=np.concatenate(constraints),
                          jacobian=np.vstack(jacobians), states=np.array(states),
                          target_n1=target_prediction, minimum_stage_margin=min_stage_margin)
        return self.value

    def objective(self, vector):
        r = self.evaluate(vector)
        return r['cost'], r['gradient']

    def constraints(self, vector):
        return self.evaluate(vector)['constraints']

    def constraint_jacobian(self, vector):
        return self.evaluate(vector)['jacobian']


def derivative_audit():
    x = plant.BASE_STATE.copy()+np.array([.1, -.15, .03, .02, .01, -.01, .05])
    u = plant.BASE_COMMAND.copy()+np.array([.03, -.01, .02, 0])
    f, a, b = field_jacobians(x, u, 67.)
    a_ref = plant.original.complex_step_jacobian(lambda y: plant.field(y, u, 67.), x)
    b_ref = plant.original.complex_step_jacobian(lambda v: plant.field(x, v, 67.), u)
    prediction = Prediction(plant.BASE_STATE, 0., 1.)
    v = np.tile(plant.BASE_COMMAND, 2)
    base = prediction.evaluate(v)
    grad, jac = base['gradient'].copy(), base['jacobian'].copy()
    numgrad, numjac = [], []
    step = 1e-6
    for k in range(8):
        delta = np.eye(8)[k]*step
        rp, rm = prediction.evaluate(v+delta), prediction.evaluate(v-delta)
        numgrad.append((rp['cost']-rm['cost'])/(2*step))
        numjac.append((rp['constraints']-rm['constraints'])/(2*step))
    errors = dict(field_error=float(np.max(abs(f-plant.field(x, u, 67.)))),
                  state_jacobian_error=float(np.max(abs(a-a_ref))),
                  input_jacobian_error=float(np.max(abs(b-b_ref))),
                  cost_gradient_absolute_error=float(np.max(abs(grad-numgrad))),
                  constraint_jacobian_absolute_error=float(np.max(abs(jac-np.array(numjac).T))))
    assert errors['field_error'] < 1e-10 and errors['state_jacobian_error'] < 1e-9
    assert errors['input_jacobian_error'] < 1e-9
    assert errors['cost_gradient_absolute_error'] < 1e-4
    assert errors['constraint_jacobian_absolute_error'] < 1e-3
    supply_state = x.copy()
    supply_state[6] = .1
    sf, sa, sb = field_jacobians(supply_state, u, 67.)
    sa_ref = plant.original.complex_step_jacobian(lambda y: plant.field(y, u, 67.), supply_state)
    sb_ref = plant.original.complex_step_jacobian(lambda v: plant.field(supply_state, v, 67.), u)
    errors['supply_branch_field_error'] = float(max(abs(sf-plant.field(supply_state,u,67.))))
    errors['supply_branch_state_jacobian_error'] = float(np.max(abs(sa-sa_ref)))
    errors['supply_branch_input_jacobian_error'] = float(np.max(abs(sb-sb_ref)))
    assert max(errors[key] for key in errors if key.startswith('supply_branch')) < 1e-9
    return errors


def independent_replay(decisions, metrics, arrival_fun=arrival):
    """Adaptive DOP853 replay with integrated costs and a dense sampled audit.

    This audits the realized command sequence, not receding-horizon decisions
    under a perturbed state and not a formal continuous-time certificate.
    """
    library = plant.default_library()
    y = np.r_[plant.BASE_STATE, np.zeros(4)]
    minimum = np.inf
    deadline_n1 = None
    samples, nfev = 0, 0
    for report in decisions:
        begin, end = report['time_h'], report['executed_end_time_h']
        if end <= begin: continue
        constant = report['executed_constant_command']
        controller = library[0].command if constant is None else lambda _x,u=np.array(constant):u
        def rhs(t, augmented):
            x = augmented[:7]
            u = controller(x)
            return np.r_[plant.field(x,u,arrival_fun(t)),100*(COST_ROW@x),100*x[6],
                         100*plant.departure_rate(x,u),100*(sum(plant.INTERNAL_DEMAND)+arrival_fun(t))]
        solution = solve_ivp(rhs,(begin,end),y,method='DOP853',rtol=1e-11,atol=1e-12,
                             dense_output=True,max_step=EXECUTION_STEP)
        if not solution.success: raise RuntimeError(solution.message)
        nfev += solution.nfev
        grid = np.linspace(begin,end,max(2,round((end-begin)*3600/2)+1))
        xs = solution.sol(grid)[:7].T
        minimum = min(minimum,min(float(min(plant.margins(x,controller(x)))) for x in xs))
        samples += len(grid)
        if begin < DEADLINE+1e-10 and end >= DEADLINE-1e-10:
            deadline_n1 = float(solution.sol(DEADLINE)[0])
        y = solution.y[:,-1]
    balance = 100*(COST_ROW@plant.BASE_STATE)+y[10]-100*(COST_ROW@y[:7])-y[9]
    return dict(scope='adaptive replay of executed controls; sampled physical audit, not formal proof',
                integrator='DOP853',rtol=1e-11,atol=1e-12,dense_audit_step_seconds=2,
                function_evaluations=nfev,sampled_states=samples,minimum_physical_margin=minimum,
                total_time_spent_veh_h=float(y[7]),queue_time_veh_h=float(y[8]),
                completed_trips_vehicles=float(y[9]),final_state=y[:7].tolist(),
                deadline_n1=deadline_n1,conservation_residual_vehicles=float(balance),
                differences_from_primary=dict(total_time_spent_veh_h=float(y[7]-metrics['total_time_spent_veh_h']),
                                              queue_time_veh_h=float(y[8]-metrics['queue_time_veh_h']),
                                              completed_trips_vehicles=float(y[9]-metrics['completed_trips_vehicles']),
                                              deadline_n1=None if deadline_n1 is None else deadline_n1-metrics['deadline_n1']))


def solve(state, now, library, warm=None, arrival_fun=None):
    horizon = min(PREDICTION_HORIZON, EVALUATION_HORIZON-now)
    predictor = Prediction(state, now, horizon, arrival_fun)
    # Fixed initialization protocol; these are solver guesses, not executed inputs.
    starts = [np.tile(plant.BASE_COMMAND, 2), np.tile(library[1].feedforward, 2)]
    if warm is not None:
        starts.insert(0, np.tile(warm, 2))
    else:
        starts.insert(0, np.tile(library[1].command(state), 2))
    candidates, attempts = [], []
    started = time.perf_counter()
    for guess in starts:
        begin = time.perf_counter()
        result = minimize(predictor.objective, guess, method='SLSQP', jac=True,
                          bounds=[(0., 1.)]*8,
                          constraints=[dict(type='ineq', fun=predictor.constraints,
                                            jac=predictor.constraint_jacobian)],
                          options=dict(maxiter=MAXITER, ftol=FTOL, disp=False))
        rec = predictor.evaluate(result.x)
        feasible = bool(np.all(np.isfinite(result.x)) and
                        min(rec['constraints']) >= -FEASIBILITY_TOLERANCE and
                        min(result.x) >= -FEASIBILITY_TOLERANCE and
                        max(result.x) <= 1+FEASIBILITY_TOLERANCE)
        attempts.append(dict(success=bool(result.success), status=int(result.status),
                             message=str(result.message), iterations=int(result.nit),
                             cost_veh_h=100*float(rec['cost']), sampled_feasible=feasible,
                             minimum_constraint_margin=float(min(rec['constraints'])),
                             solve_seconds=time.perf_counter()-begin))
        if result.success and feasible:
            candidates.append((float(rec['cost']), result.x.copy(), rec))
    elapsed = time.perf_counter()-started
    selected = min(candidates, key=lambda r:r[0]) if candidates else None
    record = dict(time_h=now, horizon_h=horizon, solve_seconds=elapsed,
                  attempts=attempts, any_solver_success=any(r['success'] for r in attempts),
                  feasible_solution_found=selected is not None, rollout_evaluations=predictor.calls)
    if selected is not None:
        record.update(predicted_cost_veh_h=100*selected[0], controls=selected[1].reshape(2,4).tolist(),
                      predicted_target_n1=selected[2]['target_n1'],
                      minimum_prediction_margin=float(min(selected[2]['constraints'])))
    return selected, record


def run(first_step=False, amplitude=AMPLITUDE, duration=DURATION):
    audit = derivative_audit()
    library = plant.default_library()
    scenario_arrival = lambda t: plant.arrival(t, amplitude=amplitude, duration=duration)
    x = plant.BASE_STATE.copy()
    times, states, commands, decisions = [0.], [x.copy()], [], []
    total = queue = departures = offered_integral = 0.
    minimum_margin = np.inf
    previous = None
    stopped = None
    for epoch in range(round(EVALUATION_HORIZON/DECISION)):
        now = epoch*DECISION
        selected, report = solve(x, now, library, None if previous is None else previous['tail'], scenario_arrival)
        decision_state = x.copy()
        if selected is not None:
            controls = selected[1].reshape(2, 4)
            command = controls[0].copy()
            controller = lambda _x, u=command: u
            previous = dict(tail=controls[1].copy(), expires=now+report['horizon_h'])
            report['execution_source'] = 'successful_sampled_feasible_mpc'
            prediction_next = selected[2]['states'][round(DECISION/PREDICTION_STEP)].copy()
        else:
            usable = previous is not None and previous['expires'] >= now+DECISION-1e-10
            if usable:
                available = previous['expires']-now
                fallback = Prediction(x, now, available, scenario_arrival).evaluate(np.tile(previous['tail'],2))
                usable = min(fallback['constraints']) >= -FEASIBILITY_TOLERANCE
            if usable:
                controller = lambda _x, u=previous['tail'].copy(): u
                report['execution_source'] = 'revalidated_previous_remaining_plan'
            else:
                controller = library[0].command
                report['execution_source'] = 'balanced_feedback_unverified_fallback'
            prediction_next = None
        print(f"decision={epoch:02d} t={now:.4f} solve_s={report['solve_seconds']:.3f} "
              f"feasible={report['feasible_solution_found']} source={report['execution_source']} "
              f"predicted_target={report.get('predicted_target_n1')}", flush=True)
        if first_step:
            return dict(scope='first-decision diagnostic only', configuration=configuration(amplitude,duration),
                        derivative_audit=audit, decisions=[report])
        for step in range(round(DECISION/EXECUTION_STEP)):
            t = now+step*EXECUTION_STEP
            u = controller(x)
            margin = float(min(plant.margins(x,u)))
            minimum_margin = min(minimum_margin,margin)
            if margin < -1e-6 or not np.all(np.isfinite(x)):
                stopped = dict(time_h=t, reason='physical_constraint_violation', margin=margin)
                break
            new = plant.rk4(x,controller,t,EXECUTION_STEP,scenario_arrival)
            new_u = controller(new)
            new_margin = float(min(plant.margins(new,new_u)))
            minimum_margin = min(minimum_margin,new_margin)
            total += EXECUTION_STEP/2*100*(COST_ROW@(x+new))
            queue += EXECUTION_STEP/2*100*(x[6]+new[6])
            departures += EXECUTION_STEP/2*100*(plant.departure_rate(x,u)+plant.departure_rate(new,new_u))
            offered_integral += EXECUTION_STEP/2*100*(sum(plant.INTERNAL_DEMAND)*2+scenario_arrival(t)+scenario_arrival(t+EXECUTION_STEP))
            commands.append(u.tolist())
            x = new
            states.append(x.copy()); times.append(t+EXECUTION_STEP)
            if new_margin < -1e-6 or not np.all(np.isfinite(x)):
                stopped = dict(time_h=t+EXECUTION_STEP, reason='physical_constraint_violation', margin=new_margin)
                break
        report['initial_state']=decision_state.tolist()
        report['executed_final_state']=x.tolist()
        report['executed_end_time_h']=float(times[-1])
        report['executed_constant_command']=None if report['execution_source']=='balanced_feedback_unverified_fallback' else controller(decision_state).tolist()
        report['one_decision_prediction_error_inf']=None if prediction_next is None else float(max(abs(x-prediction_next)))
        decisions.append(report)
        if stopped is not None: break
    arrays = np.array(states)
    deadline_indices = np.flatnonzero(np.isclose(times,DEADLINE,atol=1e-9,rtol=0))
    deadline_n1 = None if not len(deadline_indices) else float(arrays[deadline_indices[0],0])
    timing = np.array([d['solve_seconds'] for d in decisions])
    balance = 100*(COST_ROW@plant.BASE_STATE)+offered_integral-100*(COST_ROW@x)-departures
    result = dict(scope='sampled constrained numerical MPC baseline, not a robust certificate',
                configuration=configuration(amplitude,duration), derivative_audit=audit, decisions=decisions,
                metrics=dict(total_time_spent_veh_h=total,queue_time_veh_h=queue,
                             completed_trips_vehicles=departures,maximum_queue_vehicles=100*float(max(arrays[:,6])),
                             final_queue_vehicles=100*float(x[6]),minimum_physical_margin=minimum_margin,
                             deadline_n1=deadline_n1,deadline_pass=deadline_n1 is not None and deadline_n1<=TARGET_N1+1e-6,
                             completed_horizon_h=float(times[-1]),stopped=stopped,
                             conservation_residual_vehicles=float(balance),
                             mean_solve_seconds=float(timing.mean()),median_solve_seconds=float(np.median(timing)),
                             maximum_solve_seconds=float(timing.max()),
                             optimization_epochs=len(decisions),
                             epochs_without_feasible_mpc=sum(not d['feasible_solution_found'] for d in decisions),
                             unsuccessful_solver_attempts=sum(not a['success'] for d in decisions for a in d['attempts'])),
                trajectory=dict(time_h=times,state=arrays.tolist(),command=commands))
    result['independent_replay']=independent_replay(decisions,result['metrics'],scenario_arrival)
    return result


def configuration(amplitude=AMPLITUDE,duration=DURATION):
    return dict(proposal_minutes=60*DECISION,horizon_minutes=60*PREDICTION_HORIZON,
                horizon_truncation='min(1 hour, evaluation_end-current_time)',
                control_blocks='first 5 minutes; remainder of prediction horizon',
                control_parameterization='two constant four-dimensional command blocks, not feedback gains',
                prediction_step_seconds=3600*PREDICTION_STEP,execution_step_seconds=3600*EXECUTION_STEP,
                amplitude=amplitude,incident_duration_h=duration,evaluation_horizon_h=EVALUATION_HORIZON,
                fixed_deadline_h=DEADLINE,target_n1=TARGET_N1,queue_limit=plant.QUEUE_LIMIT,
                objective='100 integral(n1+n2+external_queue) dt',terminal_cost='none',
                known_arrival_forecast=True,uncertainty='none in this nominal pilot',
                maxiter=MAXITER,ftol=FTOL,feasibility_tolerance=FEASIBILITY_TOLERANCE,
                multistart='warm previous tail (or initial protection feedback), balanced feedforward, protection feedforward',
                certificate_scope='numerical RK4 stage/node constraints only; no continuous-time or robust guarantee')


def run_grid():
    import mfd_incident_operational_study as operational
    cases=operational.DEVELOPMENT_CASES+operational.HOLDOUT_CASES
    out=Path(__file__).resolve().parent/'results/mfd_incident_mpc_grid_results.json'
    payload=dict(scope='frozen continuous-input MPC across the common operational scenario grid',
                 scenario_source='mfd_incident_operational_study.DEVELOPMENT_CASES + HOLDOUT_CASES',
                 development_cases=operational.DEVELOPMENT_CASES,holdout_cases=operational.HOLDOUT_CASES,
                 holdout_scope='same synthetic plant; amplitude-duration combinations held out from library selection',
                 fixed_mpc_configuration=configuration(),
                 baseline_scope='known scenario forecast, numerical constraints, no robust certificate',runs=[])
    for amplitude,duration in cases:
        started=time.perf_counter()
        case=dict(amplitude=amplitude,duration_h=duration,
                  held_out=(amplitude,duration) in operational.HOLDOUT_CASES)
        try:
            case.update(run(amplitude=amplitude,duration=duration))
            case['run_completed']=True
        except Exception as exc:
            # Retain failed cases rather than silently dropping them from the grid.
            case.update(run_completed=False,failure=dict(type=type(exc).__name__,message=str(exc)))
        case['wall_seconds']=time.perf_counter()-started
        payload['runs'].append(case)
        out.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print('GRID_CASE',json.dumps({key:value for key,value in case.items() if key in
                                     ('amplitude','duration_h','held_out','run_completed','metrics','failure','wall_seconds')}),flush=True)
    return payload


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--first-step',action='store_true')
    modes.add_argument('--grid',action='store_true')
    args=parser.parse_args()
    if args.grid:
        run_grid()
    else:
        result=run(args.first_step)
        out=Path(__file__).resolve().parent/'results'/('mfd_incident_mpc_first_step_results.json' if args.first_step else 'mfd_incident_mpc_pilot_results.json')
        out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(json.dumps(result.get('metrics',{}),indent=2),flush=True)
