"""Frozen finite-plan uncertainty sweep; all old experiments remain unchanged."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

import mfd_incident_model as model
from mfd_incident_mpc_certification import ConstantCommandVerifier, verify_command_plan
from mfd_incident_verifier import PlanVerifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PREFIX = "mfd_incident_uncertainty_frontier"
ARCHIVE = RESULTS / "mfd_incident_mpc_grid_results.json"
PROTOCOL = ROOT / "UNCERTAINTY_FRONTIER_PROTOCOL.md"
WIDTHS = [1., 5., 10., 25., 50., 100.]
RESIDUALS = [60., 120., 300., 600., 1200.]
MAX_PROOFS = 35


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class MetricConstantVerifier(ConstantCommandVerifier):
    """Select coordinates while retaining the exact zero-gain actuator adapter."""

    def __init__(self, strategy):
        PlanVerifier.__init__(self, [], metric_strategy=strategy)


def initial_supports(center, command, widths):
    """Exact binary-rational supports for the initial physical affine polytope."""
    rows, bounds = model.affine_constraints(np.zeros(7), np.zeros((4, 7)), np.asarray(command))
    rational = lambda value: Fraction.from_float(float(value))
    exact = []
    for row, bound in zip(rows, bounds):
        slack = rational(bound) - sum(rational(a) * rational(x) for a, x in zip(row, center))
        slack -= sum(abs(rational(a)) * rational(w) for a, w in zip(row, widths))
        exact.append(slack)
    return dict(valid=all(value >= 0 for value in exact),
                rule="b-a*center-sum(abs(a_i)*width_i), exact binary rational arithmetic",
                minimum_margin=float(min(exact)),
                exact_margins=[str(value) for value in exact],
                negative_rows=[index for index, value in enumerate(exact) if value < 0])


def near_axis(rows, values, name):
    passed = [value for value in values if rows[value]["accepted"]]
    if not passed:
        return values[:2]
    best = max(passed)
    greater_failed = [value for value in values if value > best and not rows[value]["accepted"]]
    return [best, min(greater_failed)] if greater_failed else values[-2:]


def run(stage="all"):
    archive = json.loads(ARCHIVE.read_text())
    primary = next(row for row in archive["runs"] if row["amplitude"] == .3 and row["duration_h"] == 1.)
    record = primary["decisions"][0]
    center = np.asarray(record["initial_state"], dtype=float)
    commands = record["controls"]
    sources = [Path(__file__), PROTOCOL, ARCHIVE,
               ROOT / "reproducibility/mfd_incident_model.py",
               ROOT / "reproducibility/mfd_incident_verifier.py",
               ROOT / "reproducibility/mfd_incident_mpc_certification.py"]
    output = RESULTS / f"{PREFIX}_results.json"
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    if output.exists():
        payload = json.loads(output.read_text())
        if payload["source_sha256"] != hashes:
            raise RuntimeError("Refusing to mix source/protocol revisions in an existing sweep")
    else:
        payload = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=hashes,
                       scope="fixed archived first MPC plan; finite-plan certificates only",
                       protocol=str(PROTOCOL.relative_to(ROOT)), initial_state=center.tolist(),
                       commands=commands, exact_durations_h=["1/12", "11/12"],
                       service_limit_vehicles=2850., maximum_proofs=MAX_PROOFS, rows=[])

    def test(width, residual, metric="conservation_box", scalar=False, purpose="axis"):
        row_id = f"v{width:g}_r{residual:g}_{metric}" + ("_scalar" if scalar else "")
        old = next((row for row in payload["rows"] if row["row_id"] == row_id), None)
        if old:
            return old
        if sum(row["proof_file"] is not None for row in payload["rows"]) >= MAX_PROOFS:
            raise RuntimeError("Frozen full-proof budget reached")
        widths = np.array([width] * 6 + [min(width, 10.)]) / 100.
        valid = initial_supports(center, commands[0], widths)
        row = dict(row_id=row_id, purpose=purpose, initial_half_width_vehicles=width,
                   queue_half_width_vehicles=min(width, 10.), initial_box_units=widths.tolist(),
                   initial_argument="scalar" if scalar else "seven_entry_vector",
                   residual_vehicles_per_hour=residual, metric=metric,
                   initial_physical_set=valid, accepted=False, proof_file=None)
        if not valid["valid"]:
            row.update(failure_category="invalid_initial_set", first_failure=valid)
        else:
            cert = verify_command_plan(center, commands, [Fraction(1, 12), Fraction(11, 12)],
                                       initial_box=width / 100. if scalar else widths,
                                       residual_bound=residual / 100.,
                                       verifier=MetricConstantVerifier(metric), retain_steps=True)
            proof_path = RESULTS / f"{PREFIX}_{row_id}_proof.json"
            write(proof_path, dict(row_contract=row, source_sha256=hashes, certificate=cert))
            first = cert["failures"][0] if cert["failures"] else None
            category = None
            if first:
                kind = first["kind"]
                category = ("continuous_physical_support" if kind == "continuous_physical_constraint_not_certified"
                            else "scheduled_service" if kind == "scheduled_service_not_certified"
                            else "self_inclusion" if "inclusion" in kind or "enclosure" in kind
                            else kind)
            service = cert["service_check_results"]
            row.update(accepted=cert["accepted"], proof_file=str(proof_path.relative_to(ROOT)),
                       proof_sha256=digest(proof_path), proof_id=cert["proof_id"],
                       verified_continuous_physical=cert["verified_continuous_physical"],
                       failure_category=category,
                       first_failure=None if first is None else {key: value for key, value in first.items()
                                                                 if key not in ("failed_step", "attempts")},
                       service_upper_vehicles=100. * service[0]["upper"] if service else None,
                       certified_until_h=cert["certified_until_h"],
                       enclosed_until_h=cert["enclosed_until_h"],
                       completed_interval_steps=cert["completed_interval_steps"],
                       maximum_metric_radius=cert["maximum_metric_radius"],
                       wall_seconds=cert["wall_seconds"])
        payload["rows"].append(row)
        write(output, payload)
        print(json.dumps(row, allow_nan=False), flush=True)
        return row

    state_rows = {width: test(width, 60., purpose="state_error_axis") for width in WIDTHS}
    residual_rows = {residual: test(1., residual, purpose="arrival_residual_axis") for residual in RESIDUALS}
    if stage == "axes":
        return payload
    for width in near_axis(state_rows, WIDTHS, "state"):
        for residual in near_axis(residual_rows, RESIDUALS, "residual"):
            test(width, residual, purpose="deterministic_near_frontier_joint")
    passed_widths = [value for value in WIDTHS if state_rows[value]["accepted"]]
    for width in sorted(set([1., max(passed_widths) if passed_widths else 1.])):
        for metric in ("class_box", "physical_box"):
            test(width, 60., metric=metric, purpose="coordinate_ablation")
    scalar = test(1., 60., scalar=True, purpose="scalar_vector_sanity")
    vector = state_rows[1.]
    scalar_proof = json.loads((ROOT / scalar["proof_file"]).read_text())["certificate"]
    vector_proof = json.loads((ROOT / vector["proof_file"]).read_text())["certificate"]
    sanity = dict(proof_descriptor_identical=scalar_proof["proof_descriptor"] == vector_proof["proof_descriptor"],
                  terminal_bounds_identical=scalar_proof["terminal_coordinate_intervals"] == vector_proof["terminal_coordinate_intervals"],
                  acceptance_identical=scalar_proof["accepted"] == vector_proof["accepted"])
    assert all(sanity.values()), sanity
    payload["scalar_vector_sanity"] = sanity
    payload["summary"] = dict(
        completed_rows=len(payload["rows"]), full_proofs=sum(row["proof_file"] is not None for row in payload["rows"]),
        accepted_rows=sum(row["accepted"] for row in payload["rows"]),
        invalid_initial_sets=sum(not row["initial_physical_set"]["valid"] for row in payload["rows"]),
        largest_tested_accepted_state_half_width_vehicles=max(passed_widths) if passed_widths else None,
        largest_tested_accepted_residual_vehicles_per_hour=max((v for v in RESIDUALS if residual_rows[v]["accepted"]), default=None),
        interpretation="Separate axes, fixed nominal plan; maxima are tested sufficient conditions, not necessary thresholds. Failed proofs are not unsafe trajectories.")
    write(output, payload)
    print("SUMMARY", json.dumps(payload["summary"]), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("axes", "all"), default="all")
    run(parser.parse_args().stage)
