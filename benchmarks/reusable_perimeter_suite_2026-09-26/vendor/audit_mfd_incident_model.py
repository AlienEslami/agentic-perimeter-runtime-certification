"""Independent model/conservation/numerical audit; not a robust certificate.

Run from the repository root. This audit does not modify the incident model,
the legacy benchmark, any gains, or any certificate. All error thresholds in
this script concern algebraic/numerical consistency, not continuous safety.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import mfd_incident_model as model


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_incident_model_audit_results.json"


def reference_field(x, u, arrival, scale=1.0):
    """Independent class-coordinate construction, including both residuals."""
    n1, n2, n11, n22, n10, n20, z = x
    u12, u21, u02, u20 = u
    classes1 = np.array([n11, n1 - n11 - n10, n10])
    classes2 = np.array([n22, n2 - n22 - n20, n20])
    speed1 = 0.0014877 * n1**2 - 0.29815 * n1 + 15.0912
    speed2 = 0.0014877 * n2**2 - 0.29815 * n2 + 15.0912
    q11, q12, q10 = scale * speed1 * classes1
    q22, q21, q20 = scale * speed2 * classes2
    admission = min(60.0 * z, 120.0 * u02)
    d11, d12, d10, d22, d21, d20 = (
        74.85630, 37.42815, 18.714075, 44.85630, 7.42815, 0.748563
    )
    class_derivatives1 = np.array([
        d11 + q21 * u21 - q11,
        d12 - q12 * u12,
        d10 - q10 * u12,
    ])
    class_derivatives2 = np.array([
        d22 + q12 * u12 + admission / 2 - q22,
        d21 + admission / 2 - q21 * u21,
        d20 + q10 * u12 - q20 * u20,
    ])
    derivative = np.array([
        sum(class_derivatives1), sum(class_derivatives2),
        class_derivatives1[0], class_derivatives2[0],
        class_derivatives1[2], class_derivatives2[2], arrival - admission,
    ])
    departures = q11 + q22 + q20 * u20
    return derivative, departures


def high_accuracy_held(mode, amplitude=.4, duration=1.0, horizon=2.0):
    """DOP853 integration of states and metrics, split at every pulse knot."""
    start, ramp = .25, 1 / 12
    knots = sorted(set([0.0, start, start + ramp, start + duration - ramp,
                        start + duration, horizon]))
    current = np.r_[model.BASE_STATE, 0.0, 0.0, 0.0]
    samples = []

    def rhs(t, y):
        x = y[:7]
        command = mode.command(x)
        derivative, departures = reference_field(
            x, command, model.arrival(t, amplitude, duration)
        )
        return np.r_[derivative, 100 * sum(x[[0, 1, 6]]),
                     100 * x[6], 100 * departures]

    for left, right in zip(knots[:-1], knots[1:]):
        solution = solve_ivp(rhs, (left, right), current, method="DOP853",
                             rtol=2e-12, atol=2e-13, dense_output=True,
                             max_step=1 / 240)
        if not solution.success:
            raise RuntimeError(solution.message)
        samples.extend(solution.sol(np.linspace(left, right, 1001))[:7].T)
        current = solution.y[:, -1]
    physical_margins = [float(min(model.margins(x, mode.command(x)))) for x in samples]
    offered = float(sum(model.INTERNAL_DEMAND)) * horizon
    offered += model.BASE_ARRIVAL * (horizon + amplitude * (duration - ramp))
    inventory_change = sum(current[[0, 1, 6]]) - sum(model.BASE_STATE[[0, 1, 6]])
    balance_error = 100 * (offered - inventory_change) - current[9]
    return dict(final_state=current[:7].tolist(), total_time_spent_veh_h=float(current[7]),
                queue_time_veh_h=float(current[8]), departures=float(current[9]),
                minimum_sampled_margin_including_endpoints=min(physical_margins),
                maximum_sampled_queue_vehicles=100 * float(np.max(np.array(samples)[:, 6])),
                completion_balance_error_vehicles=float(balance_error),
                sample_safety_is_not_a_certificate=True)


def algebraic_audit(library):
    rng = np.random.default_rng(17032017)
    field_errors, departure_errors, balance_errors, incidence_errors = [], [], [], []
    constraint_errors = []
    branch_counts = [0, 0]
    for _ in range(2000):
        n1, n2 = rng.uniform(0, 100, 2)
        a1, a2 = rng.dirichlet(np.ones(3)) * n1, rng.dirichlet(np.ones(3)) * n2
        x = np.array([n1, n2, a1[0], a2[0], a1[2], a2[2], rng.uniform(0, 20)])
        u = rng.uniform(0, 1, 4)
        arrival, scale = rng.uniform(0, 110), rng.uniform(.8, 1.2)
        branch_counts[int(60 * x[6] >= 120 * u[2])] += 1
        actual = model.field(x, u, arrival, scale)
        expected, departures = reference_field(x, u, arrival, scale)
        field_errors.append(float(max(abs(actual - expected))))
        departure_errors.append(abs(float(model.departure_rate(x, u, scale) - departures)))
        balance_errors.append(abs(float(sum(actual[[0, 1, 6]]) -
                                        (sum(model.INTERNAL_DEMAND) + arrival - departures))))
        incidence = model.field(x, u, arrival + .25, scale) - actual
        incidence_errors.append(float(max(abs(incidence - np.r_[np.zeros(6), .25]))))
        for mode in library:
            rows, bounds = model.affine_constraints(mode.center, mode.gain, mode.feedforward)
            # The two APIs intentionally order input-box margins differently.
            margin_difference = np.sort(bounds - rows @ x) - np.sort(
                model.margins(x, mode.command(x)))
            constraint_errors.append(float(max(abs(margin_difference))))
    equilibrium_errors = [float(max(abs(model.field(mode.center, mode.feedforward))))
                          for mode in library]
    branch_point = model.BASE_STATE.copy()
    branch_point[6] = 1.0
    epsilon = 1e-9
    left, right = branch_point.copy(), branch_point.copy()
    left[6] -= epsilon
    right[6] += epsilon
    continuity_jump = float(max(abs(model.field(left, model.BASE_COMMAND) -
                                   model.field(right, model.BASE_COMMAND))))
    empty = model.BASE_STATE.copy()
    empty[6] = 0.0
    empty_drift = float(model.field(empty, model.BASE_COMMAND, 84)[6])
    full = model.BASE_STATE.copy()
    full[6] = model.QUEUE_LIMIT
    full_drift = float(model.field(full, model.BASE_COMMAND, 84)[6])
    record = dict(random_samples=2000, branch_sample_counts=branch_counts,
                  independent_field_max_error=max(field_errors),
                  independent_departure_max_error=max(departure_errors),
                  differential_conservation_max_error=max(balance_errors),
                  arrival_incidence_max_error=max(incidence_errors),
                  affine_constraint_margin_max_error=max(constraint_errors),
                  equilibrium_max_errors=equilibrium_errors,
                  branch_surface_two_sided_field_difference=continuity_jump,
                  branch_surface_state_separation=2 * epsilon,
                  empty_queue_drift_with_arrival_84=empty_drift,
                  full_queue_drift_with_nominal_gate_and_arrival_84=full_drift,
                  finite_storage_is_not_automatic=full_drift > 0)
    assert max(field_errors) < 2e-11
    assert max(departure_errors) < 2e-11
    assert max(balance_errors) < 2e-11
    assert max(incidence_errors) < 2e-11
    assert max(constraint_errors) < 2e-11
    assert max(equilibrium_errors) < 2e-11
    assert continuity_jump < 1e-6
    assert empty_drift == 84 and full_drift == 24
    return record


def rectangular_queue_blind_audit():
    current = model.BASE_STATE.copy()
    for left, right, arrival in [(0, .25, 60), (.25, .75, 84), (.75, 2, 60)]:
        solution = solve_ivp(lambda t, x: model.field(x, model.BASE_COMMAND, arrival),
                             (left, right), current, method="DOP853", rtol=1e-12, atol=1e-13)
        assert solution.success
        current = solution.y[:, -1]
    expected = model.BASE_STATE.copy()
    expected[6] = 14
    error = float(max(abs(current - expected)))
    assert error < 1e-9
    return dict(pulse_shape="rectangle", amplitude=.4, duration_h=.5,
                fixed_command=model.BASE_COMMAND.tolist(), expected_final_queue_vehicles=1400,
                numerical_final_state=current.tolist(), maximum_state_error=error,
                interpretation="Six traffic states remain nominal; excess exterior backlog persists.")


def main():
    library = model.default_library()
    algebra = algebraic_audit(library)
    rectangle = rectangular_queue_blind_audit()
    convergence = []
    for mode in library:
        reference = high_accuracy_held(mode)
        steps = []
        for step in [1 / 360, 1 / 720, 1 / 1440]:
            candidate = model.simulate_held(mode, amplitude=.4, duration=1., step=step)
            state_error = float(max(abs(np.array(candidate["final_state"]) -
                                        reference["final_state"])))
            steps.append(dict(step_seconds=3600 * step, final_state_max_error=state_error,
                              tts_error_veh_h=candidate["total_time_spent_veh_h"] -
                                             reference["total_time_spent_veh_h"],
                              queue_time_error_veh_h=candidate["queue_time_veh_h"] -
                                                    reference["queue_time_veh_h"],
                              departures_error_vehicles=candidate["departures"] -
                                                        reference["departures"]))
        assert abs(reference["completion_balance_error_vehicles"]) < 1e-7
        assert steps[-1]["final_state_max_error"] < 1e-6
        assert abs(steps[-1]["tts_error_veh_h"]) < .01
        convergence.append(dict(mode=mode.name, independent_reference=reference, refinements=steps))
    output = dict(audit="independent seven-state model and numerical consistency",
                  not_a_robust_or_continuous_time_safety_certificate=True,
                  algebra=algebra, rectangular_queue_blind_test=rectangle,
                  held_policy_numerical_convergence=convergence,
                  pulse_record=dict(primary_shape="trapezoid", amplitude=.4, duration_h=1.,
                                    ramp_h=1/12, exact_excess_offered_vehicles=2200.0),
                  status="pass")
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="pass", output=str(OUTPUT),
                          field_error=algebra["independent_field_max_error"],
                          balance_error=algebra["differential_conservation_max_error"],
                          max_reference_completion_error=max(abs(r["independent_reference"][
                              "completion_balance_error_vehicles"]) for r in convergence)), indent=2))


if __name__ == "__main__":
    main()
