#!/usr/bin/env python3
"""Exact requested E2 binding-service commissioning study, resumable."""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

import mfd_incident_model as model
from mfd_incident_affine_reachability import AffineGeneratorVerifier
from mfd_incident_mpc_certification import ConstantCommandVerifier, verify_command_plan
from mfd_incident_positive_reachability import PositiveVerifier
from mfd_incident_service_frontier import replay_plan, solve_service


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproducibility/results/mfd_uncertainty/required_E2_binding_service.json"
ARCHIVE = (ROOT.parents[1] / "output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK/"
           "reproducibility/results/mfd_reachability_service_margin_results.json")
TARGETS = (1700., 1600., 1575., 1550., 1500., 1450., 1400., 1300.)
DELTAS = (.02, .05)
METHODS = ("original", "positive_queue", "signed_generator")


def make_verifier(method, delta):
    bound = (delta, delta)
    if method == "original":
        return ConstantCommandVerifier(bound)
    if method == "positive_queue":
        return PositiveVerifier("positive_queue", bound)
    return AffineGeneratorVerifier(bound)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.loads(OUT.read_text()) if OUT.exists() else
               dict(protocol="requested E2: true 1700-vehicle service, nominal synthesis target swept downward",
                    targets=list(TARGETS), deltas=list(DELTAS), methods=list(METHODS), plans={}, rows=[]))
    archived = json.loads(ARCHIVE.read_text())
    archived_by_target = {float(row["nominal_target_vehicles"]): row for row in archived["rows"]}
    # The zero-uncertainty commissioning reported in the manuscript first closes
    # for all three methods at the nominal 1600-vehicle target.
    baseline_tts = archived_by_target[1600.]["replay"]["total_time_spent_veh_h"]
    for target in TARGETS:
        key = f"{target:g}"
        if key not in payload["plans"]:
            if target in archived_by_target:
                source = archived_by_target[target]
                plan = dict(feasible=True, commands=source["commands"], optimizer=source["optimizer"],
                            replay=source["replay"], source="archived nominal service-margin study")
            else:
                selected, optimizer = solve_service(model.BASE_STATE, service_target_vehicles=target,
                                                    deadline_h=1.)
                if selected is None:
                    plan = dict(feasible=False, optimizer=optimizer, source="new unchanged nominal planner run")
                else:
                    commands = selected[1].reshape(2, 4).tolist()
                    plan = dict(feasible=True, commands=commands, optimizer=optimizer,
                                replay=replay_plan(model.BASE_STATE, commands, deadline_h=1.,
                                                   service_target_vehicles=target),
                                source="new unchanged nominal planner run")
            if plan["feasible"]:
                plan["nominal_TTS_increase_vs_delta0_commissioning_veh_h"] = (
                    plan["replay"]["total_time_spent_veh_h"] - baseline_tts)
            payload["plans"][key] = plan
            OUT.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")

    complete = {(row["target"], row["delta"], row["method"]) for row in payload["rows"]}
    for delta in DELTAS:
        for target in TARGETS:
            plan = payload["plans"][f"{target:g}"]
            if not plan["feasible"]:
                continue
            for method in METHODS:
                if (target, delta, method) in complete:
                    continue
                started = time.perf_counter()
                cert = verify_command_plan(
                    model.BASE_STATE, plan["commands"], [Fraction(1, 12), Fraction(11, 12)],
                    amplitude=.3, duration=1., initial_box=[.01] * 7, residual_bound=.6,
                    step_seconds=15., verifier=make_verifier(method, delta),
                    service_checks=[dict(time_h=Fraction(1), coordinate=0, upper=17.)],
                )
                failure = cert["failures"][0] if cert["failures"] else {}
                service = cert["service_check_results"][0] if cert["service_check_results"] else None
                row = dict(target=target, delta=delta, method=method, accepted=cert["accepted"],
                           certified_until_h=cert["certified_until_h"],
                           service_upper_vehicles=None if service is None else 100 * service["upper"],
                           internal_service_margin_vehicles=None if service is None else 1700 - 100 * service["upper"],
                           nominal_TTS_veh_h=plan["replay"]["total_time_spent_veh_h"],
                           nominal_TTS_increase_vs_delta0_commissioning_veh_h=
                               plan["nominal_TTS_increase_vs_delta0_commissioning_veh_h"],
                           first_failure=failure.get("kind"), first_constraint=failure.get("constraint"),
                           proof_id=cert["proof_id"], certificate_wall_seconds=cert["wall_seconds"],
                           total_wall_seconds=time.perf_counter() - started)
                payload["rows"].append(row)
                OUT.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")
                print(json.dumps({k: row[k] for k in ("target", "delta", "method", "accepted",
                                  "certified_until_h", "service_upper_vehicles", "first_failure")}), flush=True)
    first = {}
    for delta in DELTAS:
        first[str(delta)] = {}
        for method in METHODS:
            candidates = [row for row in payload["rows"] if row["delta"] == delta
                          and row["method"] == method and row["accepted"]]
            first[str(delta)][method] = None if not candidates else candidates[0]
    payload["first_certified_targets"] = first
    payload["completed"] = True
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n")


if __name__ == "__main__":
    main()
