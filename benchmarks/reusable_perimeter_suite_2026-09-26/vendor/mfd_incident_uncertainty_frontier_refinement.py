"""Four frozen step refinements and twelve independent adverse-plan replays."""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

import mfd_incident_model as model
from audit_mfd_incident_model import reference_field
from mfd_incident_mpc_certification import verify_command_plan
from mfd_incident_uncertainty_frontier import initial_supports
from mfd_incident_verifier import affine_constraints
from mfd_incident_mpc_certification import constant_command_mode

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PREFIX = "mfd_incident_uncertainty_frontier_refinement"
PROTOCOL = ROOT / "UNCERTAINTY_FRONTIER_REFINEMENT_PROTOCOL.md"
FRONTIER = RESULTS / "mfd_incident_uncertainty_frontier_results.json"
ARCHIVE = RESULTS / "mfd_incident_mpc_grid_results.json"
F = lambda value: Fraction.from_float(float(value))
CASES = ((25., 60.), (1., 300.))
CORNERS = {
    "derived_classes_low": [-1, -1, 1, 1, 1, 1, -1],
    "stored_classes_low": [-1] * 7,
    "regional_totals_high": [1, 1, -1, -1, -1, -1, 1],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def corner_inside(center, widths, signs):
    initial = []
    for base, width, sign in zip(center, widths, signs):
        exact = F(base) + sign * F(width)
        value = float(exact)
        if abs(F(value) - F(base)) > F(width):
            value = np.nextafter(value, base)
        assert abs(F(value) - F(base)) <= F(width)
        initial.append(value)
    return np.array(initial)


def replay(initial, commands, residual, *, tight=False):
    """Independent ODE observations; sampled constraints are not certificates."""
    start = time.perf_counter()
    current = np.r_[initial, np.zeros(4)]  # x,TTS,queue-TTS,departures,arrivals
    cuts = [0., 1/12, .25, .25 + 1/12, 1.]
    times, states = [], []
    min_states = np.full(12, np.inf)
    min_commands = np.inf
    command_row_names = None
    max_rhs_difference = 0.
    for left, right in zip(cuts[:-1], cuts[1:]):
        u = np.asarray(commands[0 if left == 0 else 1])
        names, _, _ = affine_constraints(constant_command_mode(u))
        command_row_names = names

        def rhs(t, y):
            offered = model.arrival(t, amplitude=.3, duration=1.) + residual
            dx, departures = reference_field(y[:7], u, offered)
            return np.r_[dx, 100. * sum(y[[0, 1, 6]]), 100. * y[6],
                         100. * departures, 100. * (sum(model.INTERNAL_DEMAND) + offered)]

        solution = solve_ivp(rhs, (left, right), current, method="DOP853",
                             rtol=2e-12 if tight else 1e-11,
                             atol=2e-13 if tight else 1e-12,
                             max_step=(2.5 if tight else 5.) / 3600., dense_output=True)
        assert solution.success, solution.message
        grid = np.linspace(left, right, round((right-left) * 3600.) + 1)
        sample = solution.sol(grid).T
        for y in sample:
            x = y[:7]
            n1, n2, n11, n22, n10, n20, z = x
            # Match the verifier's state-row order, including both derived classes.
            slacks = np.array([n1, n2, n11, n22, n10, n20, z,
                               n1-n11-n10, n2-n22-n20, 100.-n1, 100.-n2, 20.-z])
            min_states = np.minimum(min_states, slacks)
            min_commands = min(min_commands, float(min(model.margins(x, u)[12:])))
        offered_mid = model.arrival((left+right)/2, amplitude=.3, duration=1.) + residual
        midpoint = solution.sol((left+right)/2)[:7]
        max_rhs_difference = max(max_rhs_difference, float(max(abs(
            reference_field(midpoint, u, offered_mid)[0] - model.field(midpoint, u, offered_mid)))))
        times.extend(grid.tolist() if not times else grid[1:].tolist())
        states.extend(sample[:, :7].tolist() if not states else sample[1:, :7].tolist())
        current = solution.y[:, -1]
    states = np.asarray(states)
    binding = int(np.argmin(min_states))
    conservation = 100. * (sum(initial[[0, 1, 6]]) - sum(current[[0, 1, 6]])) + current[10] - current[9]
    return dict(initial_state=initial.tolist(), residual_units_per_hour=residual,
                final_state=current[:7].tolist(), deadline_n1_vehicles=100. * current[0],
                deadline_pass=bool(100. * current[0] <= 2850.),
                total_time_spent_veh_h=current[7], queue_time_veh_h=current[8],
                completed_trips_vehicles=current[9], offered_arrivals_vehicles=current[10],
                maximum_queue_vehicles=100. * float(max(states[:, 6])),
                minimum_sampled_state_margin_vehicles=100. * float(min(min_states)),
                limiting_sampled_state_row=command_row_names[binding],
                minimum_sampled_command_margin=min_commands,
                sampled_physical_pass=bool(min(min_states) >= 0 and min_commands >= 0),
                conservation_residual_vehicles=float(conservation),
                maximum_independent_rhs_difference_units_per_hour=max_rhs_difference,
                integrator="independent-class-balance DOP853", tight=tight,
                dense_audit_spacing_seconds=1., wall_seconds=time.perf_counter()-start,
                trajectory=dict(time_h=times, state=states.tolist()))


def main():
    frontier_hash = digest(FRONTIER)
    frontier = json.loads(FRONTIER.read_text())
    archive = json.loads(ARCHIVE.read_text())
    primary = next(row for row in archive["runs"] if row["amplitude"] == .3 and row["duration_h"] == 1.)
    base = primary["decisions"][0]
    center, commands = np.asarray(base["initial_state"]), base["controls"]
    files = [Path(__file__), PROTOCOL, FRONTIER, ARCHIVE,
             ROOT / "reproducibility/mfd_incident_uncertainty_frontier.py",
             ROOT / "reproducibility/mfd_incident_model.py",
             ROOT / "reproducibility/mfd_incident_verifier.py",
             ROOT / "reproducibility/mfd_incident_mpc_certification.py",
             ROOT / "reproducibility/audit_mfd_incident_model.py"]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    output = RESULTS / f"{PREFIX}_results.json"
    if output.exists():
        raise RuntimeError("Refusing to overwrite an existing frozen refinement run")
    payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
                   fixed_commands=commands, refinements=[], adverse_replays=[])
    for width, residual in CASES:
        widths = np.array([width] * 6 + [min(width, 10.)]) / 100.
        old = next(row for row in frontier["rows"] if row["row_id"] == f"v{width:g}_r{residual:g}_conservation_box")
        for step in (7.5, 3.75):
            row_id = f"v{width:g}_r{residual:g}_step{step:g}"
            cert = verify_command_plan(center, commands, [Fraction(1, 12), Fraction(11, 12)],
                                       initial_box=widths, residual_bound=residual / 100.,
                                       step_seconds=step, retain_steps=True)
            proof = RESULTS / f"{PREFIX}_{row_id}_proof.json"
            write(proof, dict(source_sha256=hashes, certificate=cert))
            first = cert["failures"][0] if cert["failures"] else None
            rec = dict(row_id=row_id, state_half_width_vehicles=width,
                       residual_vehicles_per_hour=residual, maximum_step_seconds=step,
                       previous_15second_result=old, accepted=cert["accepted"],
                       certified_until_h=cert["certified_until_h"],
                       first_failure=None if first is None else {k: v for k, v in first.items()
                                                                 if k not in ("failed_step", "attempts")},
                       service_upper_vehicles=100.*cert["service_check_results"][0]["upper"]
                       if cert["service_check_results"] else None,
                       wall_seconds=cert["wall_seconds"], completed_interval_steps=cert["completed_interval_steps"],
                       proof_file=str(proof.relative_to(ROOT)), proof_sha256=digest(proof))
            payload["refinements"].append(rec)
            write(output, payload)
            print("REFINEMENT", json.dumps({k: v for k, v in rec.items() if k != "previous_15second_result"}), flush=True)
        for name, signs in CORNERS.items():
            initial = corner_inside(center, widths, signs)
            initial_check = initial_supports(initial, commands[0], np.zeros(7))
            assert initial_check["valid"]
            for residual_sign in (-1., 1.):
                rec = replay(initial, commands, residual_sign * residual / 100.)
                tight = replay(initial, commands, residual_sign * residual / 100., tight=True)
                row_id = f"v{width:g}_r{residual:g}_{name}_{'positive' if residual_sign > 0 else 'negative'}"
                rec.update(row_id=row_id, corner=name, box_half_width_vehicles=width,
                           queue_half_width_vehicles=min(width, 10.), initial_admissibility=initial_check,
                           numerical_convergence=dict(
                               terminal_maximum_difference_vehicles=100.*float(max(abs(np.asarray(rec["final_state"])-tight["final_state"]))),
                               total_time_spent_difference_veh_h=abs(rec["total_time_spent_veh_h"]-tight["total_time_spent_veh_h"]),
                               sampled_state_maximum_difference_vehicles=100.*float(np.max(abs(np.asarray(rec["trajectory"]["state"])-tight["trajectory"]["state"])))))
                path = RESULTS / f"{PREFIX}_{row_id}_replay.json"
                write(path, dict(source_sha256=hashes, nominal_accuracy=rec, tighter_accuracy=tight))
                summary = {k: v for k, v in rec.items() if k not in ("trajectory", "initial_admissibility")}
                summary.update(replay_file=str(path.relative_to(ROOT)), replay_sha256=digest(path))
                payload["adverse_replays"].append(summary)
                write(output, payload)
                print("REPLAY", json.dumps(summary), flush=True)
    assert digest(FRONTIER) == frontier_hash, "Original frontier changed"
    payload["summary"] = dict(
        refinement_proofs=len(payload["refinements"]), refinements_accepted=sum(r["accepted"] for r in payload["refinements"]),
        adverse_cases=len(payload["adverse_replays"]),
        adverse_cases_passing_sampled_physical_and_service=sum(r["sampled_physical_pass"] and r["deadline_pass"] for r in payload["adverse_replays"]),
        maximum_convergence_state_difference_vehicles=max(r["numerical_convergence"]["sampled_state_maximum_difference_vehicles"] for r in payload["adverse_replays"]),
        original_frontier_sha256_unchanged=True,
        interpretation="Failed finite-plan proofs and selected numerical adverse observations are separate; no worst-case claim is inferred from twelve sampled trajectories.")
    write(output, payload)
    print("SUMMARY", json.dumps(payload["summary"]), flush=True)


if __name__ == "__main__":
    main()
