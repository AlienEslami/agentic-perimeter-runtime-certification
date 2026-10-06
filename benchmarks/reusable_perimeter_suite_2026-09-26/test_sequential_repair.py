"""Exploratory minute-30 grid repair for D025/D024; see protocol extension."""
from __future__ import annotations

import json
import time
from fractions import Fraction

import numpy as np

from common import END, compile_commands, save, summarize
from run_delayed_handover import proof
from run_sequential_authority import OUT, replay


def compile_row(raw):
    return compile_commands(np.vstack((raw, raw)))[0][0]


def brief(cert):
    failures = cert.get("failures") or []
    first = failures[0] if failures else {}
    return dict(accepted=cert["accepted"], proof_id=cert.get("proof_id"),
                failure_kind=first.get("kind"), constraint=first.get("constraint"),
                service_checks=cert.get("service_check_results"),
                wall_seconds=cert.get("wall_seconds"))


def main():
    case = json.loads((OUT/"D025_D024.json").read_text())
    contract = case["contract"]
    assert case["decisions"][0]["accepted"] and not case["decisions"][1]["accepted"]
    center = np.asarray(case["box30"]["center_units"])
    widths = np.asarray(case["box30"]["width_units"])
    incumbent = np.asarray(case["executed_commands"][1])
    rejected = np.asarray(case["decisions"][1]["proposed_command"])
    now = Fraction(1, 2)
    record = dict(pair=["D025", "D024"], active_command=incumbent.tolist(),
                  rejected_command=rejected.tolist(), attempts=[])
    started = time.perf_counter()
    try:
        endpoint = proof(center, widths, [incumbent], [END-now], now, contract)
        record["fresh_endpoint"] = brief(endpoint)
    except (ValueError, ArithmeticError) as exc:
        record["fresh_endpoint"] = dict(accepted=False, information_rejected=str(exc))
    if not record["fresh_endpoint"]["accepted"]:
        record["outcome"] = "retain_original_correlated_proof; no new-box grid completeness"
        record["total_seconds"] = time.perf_counter()-started
        save(OUT/"D025_D024_repair_extension.json", record)
        print(json.dumps(record, indent=2))
        return
    chosen = None
    for n in range(1, 21):
        theta = n/20
        command = compile_row((1-theta)*rejected+theta*incumbent)
        try:
            cert = proof(center, widths, [command], [END-now], now, contract)
            status = brief(cert)
        except (ValueError, ArithmeticError) as exc:
            status = dict(accepted=False, information_rejected=str(exc))
        record["attempts"].append(dict(theta=theta, command=command.tolist(), certificate=status))
        print(theta, status["accepted"], status.get("failure_kind"), flush=True)
        if status["accepted"]:
            chosen = command
            break
    if chosen is None:
        record["outcome"] = "no_grid_repair; retain_original_correlated_proof"
    else:
        record["outcome"] = "certified_grid_repair"
        record["theta"] = theta
        record["repaired_command"] = chosen.tolist()
        commands = np.asarray([case["executed_commands"][0], incumbent, chosen])
        rows = replay(contract, commands)
        record["replay_summary"] = summarize(rows)
        assert record["replay_summary"]["physical_passes"] == 640
        assert record["replay_summary"]["service_passes"] == 640
    record["total_seconds"] = time.perf_counter()-started
    save(OUT/"D025_D024_repair_extension.json", record)
    print(json.dumps(dict(outcome=record["outcome"], theta=record.get("theta"),
                          attempts=len(record["attempts"]),
                          seconds=record["total_seconds"],
                          replay=record.get("replay_summary")), indent=2))


if __name__ == "__main__":
    main()
