"""Saved-record audit of uncertainty proofs/replays; no simulation or LLM calls."""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

import mfd_incident_model as model
from mfd_incident_mpc_certification import constant_command_mode, exact_command_constraints
from mfd_incident_uncertainty_frontier import initial_supports
from mfd_incident_verifier import I, imat

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
F = lambda value: Fraction.from_float(float(value))


def local(name):
    return ROOT / str(name).replace("\\", "/")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def descriptor_id(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def main():
    frontier_path = RESULTS / "mfd_incident_uncertainty_frontier_results.json"
    refinement_path = RESULTS / "mfd_incident_uncertainty_frontier_refinement_results.json"
    frontier, refinement = read(frontier_path), read(refinement_path)
    source_checks = []
    for family, payload in (("frontier", frontier), ("refinement", refinement)):
        for name, expected in payload["source_sha256"].items():
            assert digest(local(name)) == expected, name
            source_checks.append(dict(family=family, source=name, sha256=expected))
    archive = read(RESULTS / "mfd_incident_mpc_grid_results.json")
    primary = next(r for r in archive["runs"] if r["amplitude"] == .3 and r["duration_h"] == 1.)["decisions"][0]
    center, commands = primary["initial_state"], primary["controls"]
    assert frontier["initial_state"] == center and frontier["commands"] == commands
    assert refinement["fixed_commands"] == commands
    expected_model = dict(
        equations="seven_state_class_conservation_with_min_admission_v1", state_order=list(model.STATE_ORDER),
        MFD_coefficients=[model.original.MFD_A, model.original.MFD_B, model.original.MFD_C],
        internal_demand=model.INTERNAL_DEMAND.tolist(), admission_capacity=model.CAPACITY,
        release_rate=model.RELEASE_RATE, base_arrival=model.BASE_ARRIVAL, queue_limit=model.QUEUE_LIMIT,
        regional_jam_limits=[100., 100.], command_box=[0., 1.], directional_limit=.75, admission_split=.5)
    proof_checks, certs = [], {}

    def check_proof(row, hashes, width, residual, metric, step):
        proof_path = local(row["proof_file"])
        assert digest(proof_path) == row["proof_sha256"]
        raw = read(proof_path)
        assert raw["source_sha256"] == hashes
        cert = raw["certificate"]
        desc = cert["proof_descriptor"]
        assert descriptor_id(desc) == cert["proof_id"]
        assert desc["model_contract"] == expected_model
        assert desc["initial_state"] == center
        assert desc["initial_time"] == "0" and desc["durations"] == ["1/12", "11/12"]
        assert desc["pulse"] == [str(F(.3)), "1", "1/4", str(F(1/12))]
        assert desc["residual_bound"] == residual / 100.
        assert desc["metric_strategy"] == metric and desc["step_seconds"] == step
        assert desc["service_checks"] == [dict(exact_time="1", coordinate=0, upper=28.5)]
        widths = np.array([width] * 6 + [min(width, 10.)]) / 100.
        assert cert["initial_box_units"] == widths.tolist()
        radius = imat(np.array(desc["initial_L"]), I(-widths, widths)).mag()
        assert np.array_equal(radius, desc["initial_radius"])
        for index, u in enumerate(commands):
            mode = constant_command_mode(u)
            fingerprint = hashlib.sha256(b"".join(np.asarray(v, dtype=float).tobytes()
                for v in (mode.center, mode.feedforward, mode.gain, mode.A, mode.P))).hexdigest()
            assert desc["mode_fingerprints"][index] == fingerprint
            assert cert["adapter_audit"]["exact_actuator_checks"][index] == exact_command_constraints(u)
            assert exact_command_constraints(u)["passed"]
        assert cert["accepted"] == row["accepted"]
        assert cert["verified"] == cert["accepted"]
        assert cert["accepted"] == (cert["verified_continuous_physical"] and
                                     cert["verified_terminal_service"] and cert["verified_scheduled_service"])
        assert len(cert["interval_steps"]) == cert["completed_interval_steps"] == row["completed_interval_steps"]
        assert row["certified_until_h"] == cert["certified_until_h"]
        assert row["wall_seconds"] == cert["wall_seconds"]
        assert row["service_upper_vehicles"] == (100. * cert["service_check_results"][0]["upper"]
                                                  if cert["service_check_results"] else None)
        if cert["accepted"]:
            assert not cert["failures"] and cert["certified_until_h"] == 1.
            assert cert["service_check_results"][0]["passed"]
            assert min(cert["continuous_constraint_margins_lower"].values()) >= 0
        else:
            assert cert["failures"][0]["kind"] == "continuous_physical_constraint_not_certified"
            assert cert["failures"][0]["margin_lower"] < 0.
            assert row["service_upper_vehicles"] is None
            assert 0. <= cert["certified_until_h"] < 1.
            assert row["first_failure"] == {k: v for k, v in cert["failures"][0].items()
                                             if k not in ("failed_step", "attempts")}
        initial = initial_supports(center, commands[0], widths)
        assert initial["valid"]
        if "row_contract" in raw:
            # This is the original pre-execution metadata snapshot, not its outcome.
            pre = raw["row_contract"]
            assert pre["accepted"] is False and pre["proof_file"] is None
            assert pre["initial_physical_set"] == row["initial_physical_set"] == initial
            assert pre["initial_box_units"] == widths.tolist()
            assert pre["metric"] == metric and pre["residual_vehicles_per_hour"] == residual
            assert pre["row_id"] == row["row_id"]
        result = dict(row_id=row["row_id"], proof_file=row["proof_file"], sha256=row["proof_sha256"],
                      proof_id=cert["proof_id"], accepted=cert["accepted"], initial_set_valid=True,
                      certified_until_h=cert["certified_until_h"], completed_steps=cert["completed_interval_steps"],
                      descriptor_model_commands_uncertainty_and_scope_checked=True)
        proof_checks.append(result)
        certs[row["row_id"]] = cert

    for row in frontier["rows"]:
        check_proof(row, frontier["source_sha256"], row["initial_half_width_vehicles"],
                    row["residual_vehicles_per_hour"], row["metric"], 15.)
    for row in refinement["refinements"]:
        check_proof(row, refinement["source_sha256"], row["state_half_width_vehicles"],
                    row["residual_vehicles_per_hour"], "conservation_box", row["maximum_step_seconds"])
        original = next(r for r in frontier["rows"] if r["row_id"] == row["previous_15second_result"]["row_id"])
        assert original == row["previous_15second_result"]
    vector = certs["v1_r60_conservation_box"]
    scalar = certs["v1_r60_conservation_box_scalar"]
    sanity = dict(proof_descriptor_identical=vector["proof_descriptor"] == scalar["proof_descriptor"],
                  terminal_bounds_identical=vector["terminal_coordinate_intervals"] == scalar["terminal_coordinate_intervals"],
                  acceptance_identical=vector["accepted"] == scalar["accepted"])
    assert all(sanity.values()) and sanity == frontier["scalar_vector_sanity"]
    summary = frontier["summary"]
    assert summary["completed_rows"] == summary["full_proofs"] == len(frontier["rows"]) == 19
    assert summary["accepted_rows"] == sum(r["accepted"] for r in frontier["rows"]) == 8
    assert summary["invalid_initial_sets"] == 0
    assert summary["largest_tested_accepted_state_half_width_vehicles"] == max(
        r["initial_half_width_vehicles"] for r in frontier["rows"] if r["metric"] == "conservation_box"
        and r["residual_vehicles_per_hour"] == 60. and r["accepted"])
    assert summary["largest_tested_accepted_residual_vehicles_per_hour"] == max(
        r["residual_vehicles_per_hour"] for r in frontier["rows"] if r["metric"] == "conservation_box"
        and r["initial_half_width_vehicles"] == 1. and r["accepted"])

    replay_checks = []
    for row in refinement["adverse_replays"]:
        path = local(row["replay_file"])
        assert digest(path) == row["replay_sha256"]
        raw = read(path)
        assert raw["source_sha256"] == refinement["source_sha256"]
        nominal, tighter = raw["nominal_accuracy"], raw["tighter_accuracy"]
        for key, value in row.items():
            if key not in ("replay_file", "replay_sha256"):
                assert nominal[key] == value, (row["row_id"], key)
        initial = nominal["initial_state"]
        widths = np.array([row["box_half_width_vehicles"]] * 6 + [row["queue_half_width_vehicles"]]) / 100.
        assert all(abs(F(x)-F(c)) <= F(w) for x, c, w in zip(initial, center, widths))
        assert initial_supports(initial, commands[0], np.zeros(7)) == nominal["initial_admissibility"]
        assert nominal["initial_admissibility"]["valid"]
        for record in (nominal, tighter):
            states = np.asarray(record["trajectory"]["state"])
            times = np.asarray(record["trajectory"]["time_h"])
            assert len(states) == len(times) == 3601 and times[0] == 0. and times[-1] == 1.
            assert np.max(np.diff(times)) <= 1/3600 + 1e-15
            assert np.array_equal(states[0], initial) and np.array_equal(states[-1], record["final_state"])
            assert record["deadline_n1_vehicles"] == 100. * states[-1, 0]
            assert record["deadline_pass"] == bool(record["deadline_n1_vehicles"] <= 2850.)
            slacks = np.column_stack((states, states[:, 0]-states[:, 2]-states[:, 4],
                states[:, 1]-states[:, 3]-states[:, 5], 100.-states[:, 0],
                100.-states[:, 1], 20.-states[:, 6]))
            assert record["minimum_sampled_state_margin_vehicles"] == 100. * float(np.min(slacks))
            assert record["maximum_queue_vehicles"] == 100. * float(max(states[:, 6]))
            assert record["sampled_physical_pass"] == bool(np.min(slacks) >= 0 and
                                                           record["minimum_sampled_command_margin"] >= 0)
            conservation = 100. * (sum(np.array(initial)[[0, 1, 6]]) - sum(states[-1, [0, 1, 6]]))
            conservation += record["offered_arrivals_vehicles"] - record["completed_trips_vehicles"]
            assert abs(conservation - record["conservation_residual_vehicles"]) < 1e-8
        assert nominal["trajectory"]["time_h"] == tighter["trajectory"]["time_h"]
        convergence = dict(
            terminal_maximum_difference_vehicles=100. * float(max(abs(np.array(nominal["final_state"])-tighter["final_state"]))),
            total_time_spent_difference_veh_h=abs(nominal["total_time_spent_veh_h"]-tighter["total_time_spent_veh_h"]),
            sampled_state_maximum_difference_vehicles=100. * float(np.max(abs(np.array(nominal["trajectory"]["state"])-tighter["trajectory"]["state"]))))
        assert convergence == row["numerical_convergence"]
        replay_checks.append(dict(row_id=row["row_id"], replay_file=row["replay_file"], sha256=row["replay_sha256"],
                                  initial_corner_inside_original_box=True, initial_set_valid=True,
                                  sampled_physical_and_service_pass=nominal["sampled_physical_pass"] and nominal["deadline_pass"],
                                  convergence=convergence))
    refined_summary = refinement["summary"]
    assert refined_summary["refinement_proofs"] == len(refinement["refinements"]) == 4
    assert refined_summary["refinements_accepted"] == sum(r["accepted"] for r in refinement["refinements"]) == 0
    assert refined_summary["adverse_cases"] == len(replay_checks) == 12
    assert refined_summary["adverse_cases_passing_sampled_physical_and_service"] == sum(
        r["sampled_physical_and_service_pass"] for r in replay_checks) == 12
    assert refined_summary["maximum_convergence_state_difference_vehicles"] == max(
        r["convergence"]["sampled_state_maximum_difference_vehicles"] for r in replay_checks)
    assert refined_summary["original_frontier_sha256_unchanged"]
    prior_frontier_hash = next(expected for name, expected in refinement["source_sha256"].items()
                               if local(name) == frontier_path)
    assert digest(frontier_path) == prior_frontier_hash
    output = dict(
        passed=True, created_utc=datetime.now(timezone.utc).isoformat(), audit_script_sha256=digest(Path(__file__)),
        source_results_sha256={frontier_path.name: digest(frontier_path), refinement_path.name: digest(refinement_path)},
        simulation_runs=0, live_model_calls=0, proof_descriptors_checked=len(proof_checks),
        proof_file_hashes_checked=len(proof_checks), replay_file_hashes_checked=len(replay_checks),
        source_hash_checks=len(source_checks), initial_certificate_sets_checked=len(proof_checks),
        adverse_initial_corners_checked=len(replay_checks), scalar_vector_sanity=sanity,
        summary_counts_and_tested_axis_maxima_checked=True,
        metadata_note="Original frontier row_contract is pre-execution metadata with accepted=false/proof_file=null placeholders. Actual certificate.accepted and completed summary rows are authoritative; their agreement is checked for all19 rows. Original files are preserved.",
        scope="Saved record and mathematical contract consistency audit, not an independent rerun of interval integration or a worst-case proof from sampled replays.",
        sources=source_checks, proofs=proof_checks, replays=replay_checks)
    path = RESULTS / "mfd_incident_uncertainty_frontiers_audit_results.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ("sources", "proofs", "replays")}, indent=2))
    print(path)


if __name__ == "__main__":
    main()
