#!/usr/bin/env python3
"""Archived MPC command-plan certification pilot, not a closed-loop study.

Exact archived binary commands are represented by zero-gain affine controllers.
The original MPC results and physical model are unchanged. The shared verifier
uses the equivalent expm1 evaluation of the scalar comparison integral.
Every attempted bound is retained; a failed enclosure is not a counterexample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

import mfd_incident_model as model
from mfd_incident_verifier import PlanVerifier


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reproducibility/results/mfd_incident_mpc_grid_results.json"
OUTPUT = ROOT / "results/mfd_incident_mpc_certification_archived_primary_pilot_results.json"
RAW_OUTPUT = ROOT / "results/mfd_incident_mpc_certification_raw_binary_pulse_pilot_results.json"
EXACT_OUTPUT = ROOT / "results/mfd_incident_mpc_certification_exact_actuator_pilot_results.json"
PERIOD = Fraction(1, 12)
END = Fraction(2)
DEADLINE = Fraction(1)


def constant_command_mode(command, name="constant_command"):
    """No equilibrium/linear stability claim; A and P only satisfy geometry API."""
    command = np.asarray(command, dtype=float)
    if command.shape != (4,) or not np.all(np.isfinite(command)):
        raise ValueError("a finite four-command vector is required")
    return model.Mode(name, np.zeros(7), command.copy(), np.zeros((4, 7)),
                      np.zeros((7, 7)), np.eye(7))


def exact_command_constraints(command):
    """Exact rational actuator audit, independent of trajectory or ODE arithmetic."""
    values = list(map(lambda value: Fraction.from_float(float(value)), command))
    names, margins = [], []
    for name, value in zip(model.original.INPUT_ORDER, values):
        names.extend([name + "_upper", name + "_lower"])
        margins.extend([1 - value, value])
    for left, right in ((1, 0), (3, 2)):
        delta = values[left] - values[right]
        names.extend([f"direction_{left}_{right}_upper", f"direction_{left}_{right}_lower"])
        margins.extend([Fraction(3, 4) - delta, Fraction(3, 4) + delta])
    return dict(passed=all(value >= 0 for value in margins),
                exact_binary_commands=[str(value) for value in values],
                exact_margins={name: str(value) for name, value in zip(names, margins)},
                margins_float={name: float(value) for name, value in zip(names, margins)},
                equalities=[name for name, value in zip(names, margins) if value == 0])


def rational_lower(value):
    """Downward-rounded float of an exact dyadic residual, keeping exact zero."""
    value = Fraction(value)
    near = float(value)
    return float(np.nextafter(near, -np.inf)) if Fraction.from_float(near) > value else near


class ConstantCommandVerifier(PlanVerifier):
    """Same state enclosure; exact algebra replaces only zero-gain actuator rows.

    If K=0, each actuator inequality is a constant exact binary-rational
    residual. Its value is independent of every state, prediction error, and
    disturbance. These 12 supports are therefore known exactly, including
    equality. No state/class/regional/storage support is modified.
    """
    def __init__(self):
        super().__init__([], metric_strategy="conservation_box")

    def step(self, center, radius, mode, left, right, pulse, residual_bound):
        if not np.array_equal(mode.gain, np.zeros((4, 7))):
            raise ValueError("exact constant-command proof requires identically zero feedback gain")
        next_center, next_radius, record = super().step(center, radius, mode, left, right, pulse, residual_bound)
        audit = exact_command_constraints(mode.feedforward)
        names = self.geometry(mode)["names"]
        raw, replacement = {}, {}
        for name, value in audit["exact_margins"].items():
            index = names.index(name)
            raw[name] = record["continuous_constraint_margins_lower"][index]
            replacement[name] = rational_lower(value)
            record["continuous_constraint_margins_lower"][index] = replacement[name]
        record["constant_command_support_audit"] = dict(
            exact_margins=audit["exact_margins"], original_interval_lower=raw,
            exact_downward_lower=replacement, physical_state_rows_unchanged=True)
        return next_center, next_radius, record

    def verify_plan(self, initial_state, modes, durations_h=None, **kwargs):
        selected = [self.library[int(mode)] if isinstance(mode, (int, np.integer)) else mode for mode in modes]
        for mode in selected:
            if not np.array_equal(mode.gain, np.zeros((4, 7))):
                raise ValueError("constant-command verifier does not accept nonzero feedback")
        result = super().verify_plan(initial_state, selected, durations_h, **kwargs)
        result["pre_adapter_proof_id"] = result["proof_id"]
        result["proof_descriptor"]["exact_constant_command_support"] = dict(
            rule="zero_gain_constant_actuator_residuals_v1",
            zero_gain_checked=True, physical_state_supports_unchanged=True,
            arithmetic="Fraction.from_float exact binary coefficients; downward float residuals, no tolerance",
            controller_contracts=[dict(mode_fingerprint=self.geometry(mode)["fingerprint"],
                                       **exact_command_constraints(mode.feedforward)) for mode in selected])
        result["proof_id"] = hashlib.sha256(json.dumps(result["proof_descriptor"], sort_keys=True).encode("utf-8")).hexdigest()
        for checkpoint in result["reachable_checkpoints"]:
            checkpoint["proof_id"] = result["proof_id"]
        return result


def verify_command_plan(initial_state, commands, durations_h, *, initial_time_h=Fraction(0),
                        amplitude=.3, duration=1., initial_box=.01, residual_bound=.6,
                        step_seconds=15., verifier=None, retain_steps=False,
                        exact_actuator_support=True, service_checks=None, timecuts_h=None):
    """Compile supplied commands verbatim and apply the unchanged plan verifier.

    Positive durations are required; omit an unused zero-duration final block.
    Supply Fraction times when the execution clock has exact rational epochs.
    """
    commands = [np.asarray(command, dtype=float) for command in commands]
    modes = [constant_command_mode(command, f"constant_block_{index}")
             for index, command in enumerate(commands)]
    if verifier is None:
        verifier = ConstantCommandVerifier() if exact_actuator_support else PlanVerifier(modes, metric_strategy="conservation_box")
    initial_time_h = (initial_time_h if isinstance(initial_time_h, Fraction)
                      else Fraction.from_float(float(initial_time_h)))
    durations = [item if isinstance(item, Fraction) else Fraction.from_float(float(item)) for item in durations_h]
    final_time = initial_time_h + sum(durations)
    checks = ([dict(time_h=DEADLINE, coordinate=0, upper=28.5)]
              if initial_time_h < DEADLINE <= final_time else []) if service_checks is None else service_checks
    cuts = [PERIOD * index for index in range(25) if initial_time_h < PERIOD * index < final_time]
    if timecuts_h is not None:
        cuts = timecuts_h
    result = verifier.verify_plan(initial_state, modes, durations, initial_time_h=initial_time_h,
                                  amplitude=amplitude, duration=duration, pulse_start=.25,
                                  ramp=1 / 12, residual_bound=residual_bound, initial_box=initial_box,
                                  step_seconds=step_seconds, service_checks=checks,
                                  timecuts_h=cuts, retain_steps=retain_steps)
    encoded = json.dumps(result["proof_descriptor"], sort_keys=True).encode("utf-8")
    result["adapter_audit"] = dict(
        kind="exact_binary_piecewise_constant_command_adapter",
        commands=[value.tolist() for value in commands], zero_feedback_gain=True,
        geometry="center=0, A=0, P=identity are finite placeholders; common conservation_box coordinates are installed",
        stability_claim=False, equilibrium_claim=False,
        exact_actuator_checks=[exact_command_constraints(value) for value in commands],
        proof_hash_recomputed=hashlib.sha256(encoded).hexdigest() == result["proof_id"],
        commands_bound_by_mode_fingerprints=True)
    return result


def summary(record):
    certificate = record["certificate"]
    return dict(epoch=record["epoch"], accepted=certificate["accepted"],
                certified_until_h=certificate["certified_until_h"],
                failure=[{key: failure[key] for key in ("kind", "constraint", "margin_lower") if key in failure}
                         for failure in certificate["failures"]],
                service=certificate["service_check_results"], wall_seconds=certificate["wall_seconds"])


def run(epochs=(0, 1, 6, 11, 12, 18, 23), *, raw_actuator=False, step_seconds=15.):
    raw = SOURCE.read_bytes()
    archive = json.loads(raw)
    primary = next(case for case in archive["runs"] if case["amplitude"] == .3 and case["duration_h"] == 1.)
    payload = dict(scope="archived primary MPC proposal plans; not executed robust MPC closed loop",
                   source=str(SOURCE), source_sha256=hashlib.sha256(raw).hexdigest(),
                   verifier_source_sha256=hashlib.sha256((ROOT / "reproducibility/mfd_incident_verifier.py").read_bytes()).hexdigest(),
                   preserved_initial_rational_ramp_probe=dict(path=str(OUTPUT),
                       sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest() if OUTPUT.exists() else None,
                       verifier_source_sha256="a8248a5ffe620004675c45f991a3a6826a00175944f5b444a6c4ef6a102386fd",
                       scope="first failed probe, rational ramp differed infinitesimally; retained, not evidence for matched forecast"),
                   commands_unchanged=True, initial_box_units=.01, residual_bound_units_per_hour=.6,
                   exact_clock="epoch/12 hour, horizon min(1,2-epoch/12); five-minute first block",
                   pulse_contract="unchanged exact binary float ramp=1/12, amplitude=.3, start=.25, duration=1",
                   exact_actuator_support=not raw_actuator,
                   step_seconds=step_seconds,
                   preserved_source_files=["mfd_incident_model.py", "mfd_incident_mpc.py"],
                   verifier_numerical_refinement="interval expm1(a*h)/a, same comparison integral; original source archived",
                   records=[])
    verifier = PlanVerifier([], metric_strategy="conservation_box") if raw_actuator else ConstantCommandVerifier()
    output = RAW_OUTPUT if raw_actuator else EXACT_OUTPUT
    for epoch in epochs:
        archived = primary["decisions"][epoch]
        now = epoch * PERIOD
        horizon = min(Fraction(1), END - now)
        blocks = [min(PERIOD, horizon), horizon - min(PERIOD, horizon)]
        commands = [command for command, length in zip(archived["controls"], blocks) if length > 0]
        blocks = [length for length in blocks if length > 0]
        certificate = verify_command_plan(archived["initial_state"], commands, blocks,
                                          initial_time_h=now, verifier=verifier, step_seconds=step_seconds,
                                          exact_actuator_support=not raw_actuator, retain_steps=True)
        record = dict(epoch=epoch, archived_time_h=archived["time_h"], exact_time_fraction=str(now),
                      archived_horizon_h=archived["horizon_h"], exact_durations=[str(value) for value in blocks],
                      initial_state=archived["initial_state"], commands=commands, certificate=certificate)
        payload["records"].append(record)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
        print(json.dumps(summary(record)), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, nargs="+", default=[0, 1, 6, 11, 12, 18, 23])
    parser.add_argument("--raw-actuator", action="store_true")
    parser.add_argument("--step-seconds", type=float, default=15.)
    arguments = parser.parse_args()
    run(arguments.epochs, raw_actuator=arguments.raw_actuator, step_seconds=arguments.step_seconds)
