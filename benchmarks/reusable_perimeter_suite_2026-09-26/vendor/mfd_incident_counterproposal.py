#!/usr/bin/env python3
"""Semantic service counterproposal, separate from the commissioned library.

The diagnostic incumbent lifts the original six-state Mode 1 feedback to the
seven-state plant by adding an exactly zero queue-gain column. It is a well-
formed controller/hold plan, not corrupted syntax and not a new LLM response.
It is NOT added to the operational three-mode registry.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import mfd_incident_model as model
from mfd_incident_library_design import synthesize
from mfd_incident_verifier import PlanVerifier


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reproducibility/results/mfd_cubic_pilot_results.json"
OUTPUT = ROOT / "results/mfd_incident_counterproposal_results.json"


def queue_blind_incumbent():
    record = json.loads(SOURCE.read_text(encoding="utf-8"))
    six_gain = np.asarray(record["controller_search"]["selected_gains"][0])
    gain = np.column_stack((six_gain, np.zeros(4)))
    closed = model.original.complex_step_jacobian(
        lambda x: model.field(x, model.BASE_COMMAND + gain @ (x - model.BASE_STATE)), model.BASE_STATE)
    # P only supplies an invertible fallback geometry. No seven-state
    # Lyapunov claim is made for this queue-blind, integrating incumbent.
    geometry = np.eye(7)
    geometry[:6, :6] = np.asarray(record["certificate"]["matrices"][0])
    mode = model.Mode("diagnostic_queue_blind_original_mode1", model.BASE_STATE.copy(),
                      model.BASE_COMMAND.copy(), gain, closed, geometry)
    assert np.array_equal(mode.gain[:, 6], np.zeros(4))
    assert np.array_equal(mode.command(model.BASE_STATE), model.BASE_COMMAND)
    return mode


def exact_regional_residual(decimal=False):
    """Distinguish ideal decimal equilibrium from installed binary coefficients."""
    convert = (lambda value: Fraction(str(round(float(value), 9)))) if decimal else Fraction.from_float
    a, b, c = map(convert, (model.original.MFD_A, model.original.MFD_B, model.original.MFD_C))
    d11, d12, d10, d22, d21, d20 = map(convert, model.INTERNAL_DEMAND)
    n1, n2, n11, n22, n10, n20 = map(Fraction, (30, 30, 15, 15, 5, 5))
    u12, u21, u02, u20 = map(convert, model.BASE_COMMAND)
    h1, h2 = a * n1**2 + b * n1 + c, a * n2**2 + b * n2 + c
    q11, q12, q10 = n11 * h1, (n1 - n11 - n10) * h1, n10 * h1
    q22, q21, q20 = n22 * h2, (n2 - n22 - n20) * h2, n20 * h2
    admission = Fraction(120) * u02
    return [-q11 - (q12 + q10) * u12 + q21 * u21 + d11 + d12 + d10,
            -q22 + (q12 + q10) * u12 - q21 * u21 + admission - q20 * u20 + d22 + d21 + d20,
            -q11 + q21 * u21 + d11, -q22 + q12 * u12 + admission / 2 + d22,
            -q10 * u12 + d10, -q20 * u20 + q10 * u12 + d20]


def pulse_area(t, duration=Fraction(1), start=Fraction(1, 4), ramp=Fraction(1, 12)):
    local = t - start
    if local <= 0:
        return Fraction(0)
    if local >= duration:
        return duration - ramp
    if local < ramp:
        return local**2 / (2 * ramp)
    if local <= duration - ramp:
        return local - ramp / 2
    return duration - ramp - (duration - local)**2 / (2 * ramp)


def analytic_record(amplitude):
    duration, start, ramp = Fraction(1), Fraction(1, 4), Fraction(1, 12)
    at_deadline = Fraction(2) + 60 * amplitude * pulse_area(Fraction(1))
    peak = Fraction(2) + 60 * amplitude * (duration - ramp)
    candidate = start + ramp / 2 + Fraction(18) / (60 * amplitude)
    crossing = candidate if start + ramp <= candidate <= start + duration - ramp else None
    return dict(amplitude=str(amplitude), regional_accumulations_vehicles=[3000, 3000],
                admission_units_per_hour=60, queue_at_deadline_units=str(at_deadline),
                queue_at_deadline_vehicles=float(100 * at_deadline), peak_queue_units=str(peak),
                peak_queue_vehicles=float(100 * peak), queue_limit_vehicles=2000,
                first_queue_limit_crossing_h=None if crossing is None else str(crossing),
                first_queue_limit_crossing_minutes=None if crossing is None else float(60 * crossing),
                deadline_service_limit_vehicles=2850, deadline_service_violation_vehicles=150,
                exact_identity_scope="ideal displayed-decimal coefficients; installed binary model checked independently below")


def main():
    incumbent = queue_blind_incumbent()
    backup = synthesize(20, 22, 72, 3000)
    verifier = PlanVerifier([incumbent, backup], metric_strategy="conservation_box")
    decimal_residual = exact_regional_residual(decimal=True)
    binary_residual = exact_regional_residual(decimal=False)
    assert all(value == 0 for value in decimal_residual)
    assert max(map(lambda value: abs(float(value)), binary_residual)) < 1.e-10
    service = [dict(time_h=Fraction(1), coordinate=0, upper=28.5)]

    nominal = verifier.verify_plan(model.BASE_STATE, [incumbent], [Fraction(1)], duration=1., amplitude=.3,
                                    initial_box=0., residual_bound=0., service_checks=service)
    nominal_lower = nominal["terminal_coordinate_intervals"]["n1"][0]
    assert nominal["verified_continuous_physical"] and nominal_lower > 28.5 and not nominal["accepted"]
    print("nominal semantic counterexample", nominal_lower, nominal["failures"][0]["kind"], flush=True)
    robust_attempt = verifier.verify_plan(model.BASE_STATE, [incumbent], [Fraction(1)], duration=1., amplitude=.3,
                                           initial_box=.01, residual_bound=.6, service_checks=service)
    backup_certificate = verifier.verify_plan(model.BASE_STATE, [backup], [Fraction(1)], duration=1., amplitude=.3,
                                               initial_box=.01, residual_bound=.6, service_checks=service)
    assert backup_certificate["accepted"]
    print("robust incumbent accepted", robust_attempt["accepted"], "backup accepted", backup_certificate["accepted"], flush=True)

    crossing = Fraction(25, 24)
    before = verifier.verify_plan(model.BASE_STATE, [incumbent], [crossing - Fraction(1, 3600)],
                                   duration=1., amplitude=.4, initial_box=0., residual_bound=0.)
    after = verifier.verify_plan(model.BASE_STATE, [incumbent], [crossing + Fraction(1, 3600)],
                                  duration=1., amplitude=.4, initial_box=0., residual_bound=0.)
    before_upper = before["terminal_coordinate_intervals"]["exterior_queue"][1]
    after_lower = after["terminal_coordinate_intervals"]["exterior_queue"][0]
    installed_overflow_bracketed = (before["verified_continuous_physical"] and before_upper < 20.
                                     and after_lower > 20. and after["enclosed_until_h"] >= float(crossing + Fraction(1, 3600)))
    print("installed 40% overflow bracket", before_upper, after_lower, installed_overflow_bracketed, flush=True)
    result = dict(
        status="separate_semantic_counterproposal_diagnostic",
        proposal=dict(kind="diagnostic_incumbent_hold", controller=incumbent.name, hold_minutes=60,
                      well_formed_typed_controller=True, commissioned_library_member=False,
                      new_llm_call=False, parser_corruption=False,
                      scope="offline controller/plan acceptance probe; not inserted into the operational three-mode protocol"),
        incumbent=dict(**incumbent.record(), source=str(SOURCE),
                       source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), original_six_state_mode=1,
                       queue_gain_exactly_zero=True,
                       stability_warning="The added queue is an integrator: no seven-state asymptotic-stability certificate is claimed."),
        analytic_nominal_identity=dict(
            statement="At the balanced regional equilibrium, zero queue gain keeps all commands at their feedforward values. Since z>=2, admission=min(60*z,120*.5)=60. The six regional derivatives vanish for the displayed-decimal equilibrium, while z(t)=2+60*amplitude*integral(trapezoid).",
            decimal_regional_derivatives=[str(value) for value in decimal_residual],
            installed_binary_regional_derivatives=[dict(exact_fraction=str(value), units_per_hour=float(value))
                                                   for value in binary_residual],
            binary_scope="The mathematical installed binary coefficients have tiny equilibrium residuals. Exact constancy is not asserted for that model; outward nominal integration below supplies the counterexample lower bound.",
            pulse30=analytic_record(Fraction(3, 10)), pulse40=analytic_record(Fraction(2, 5))),
        nominal_installed_certificate=nominal, robust_incumbent_attempt=robust_attempt,
        commissioned_backup_certificate=backup_certificate,
        forty_percent_queue_counterexample=dict(exact_decimal_crossing_h="25/24", exact_decimal_crossing_minutes=62.5,
                                                installed_binary_crossing_bracket_minutes=[62.5 - 1 / 60, 62.5 + 1 / 60],
                                                installed_binary_overflow_bracket_verified=installed_overflow_bracketed,
                                                before_certificate=before, after_enclosure=after),
        conclusion=dict(
            substantive_service_counterexample_verified=True,
            installed_nominal_n1_lower_vehicles=100 * nominal_lower,
            installed_nominal_queue_upper_at_deadline_vehicles=100 * nominal["terminal_coordinate_intervals"]["exterior_queue"][1],
            reason_robust_service_cannot_pass="The exact initial state and zero residual signal belong to the declared robust uncertainty set, and their verified n1 deadline lower bound exceeds 28.5. This single admissible trajectory refutes the universal robust service requirement.",
            robust_attempt_interpretation="An unaccepted robust enclosure is retained as inconclusive on its own. The separate nominal lower-bound counterexample, not absence of an upper-bound certificate, establishes service infeasibility.",
            commissioned_backup_passes_same_robust_contract=backup_certificate["accepted"],
            backup_n1_upper_vehicles=100 * backup_certificate["terminal_coordinate_intervals"]["n1"][1],
            library_unchanged=True, main_verifier_unchanged=True),
        proof_text="For the queue-blind incumbent, the nominal region state remains at the balanced equilibrium in the exact decimal model and z'=lambda-60. Thus n1(1h)=3000>2850 although the30% pulse has z(1h)=1475 vehicles and peak1850<2000. Under the installed binary coefficients, the outward nominal certificate independently places n1(1h) strictly above2850, giving an admissible trajectory that falsifies any robust service guarantee. At40%, the decimal queue reaches2000 at62.5min; the installed model is separately bracketed using outward before/after enclosures. None of these claims comes from a parser failure or a numerical tolerance.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["conclusion"], indent=2), flush=True)


if __name__ == "__main__":
    main()
