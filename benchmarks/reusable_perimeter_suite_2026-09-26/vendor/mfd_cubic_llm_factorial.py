#!/usr/bin/env python3
"""Prompt/scenario-seed factorial for live proposals on the cubic MFD.

Each model receives three semantically equivalent prompt templates over five
balanced, seeded regional-priority schedules.  Every configuration contains 24
actual Ollama calls.  The raw prompts, responses, model records, latencies,
parsed proposals, policy events, and matched execution summaries are retained.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import live_llm_mfd_cubic_validation as live
import mfd_cubic_pilot as mfd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "live_llm_results" / "cubic" / "factorial"
PROMPT_TEMPLATES = ("rule_first", "costs_first", "compact")
SCENARIO_SEEDS = (20260812, 20260813, 20260814, 20260815, 20260816)
NUMBER_OF_DECISIONS = int(round(live.HORIZON_H / live.DECISION_INTERVAL_H))


def priority_schedule(seed: int) -> list[int]:
    """Return a balanced 24-epoch numerical-cost ordering."""

    schedule = np.repeat(np.array([0, 1], dtype=int), NUMBER_OF_DECISIONS // 2)
    np.random.default_rng(seed).shuffle(schedule)
    assert np.count_nonzero(schedule == 0) == NUMBER_OF_DECISIONS // 2
    assert np.count_nonzero(schedule == 1) == NUMBER_OF_DECISIONS // 2
    return schedule.tolist()


def prompt_variant(name: str, observation: dict[str, Any]) -> str:
    if name == "rule_first":
        return live.proposal_prompt(observation)

    costs = observation["rollout_costs"]
    error = np.asarray(observation["state"], dtype=float) - mfd.X_STAR
    active = int(observation["active_mode"]) + 1
    time_h = float(observation["time_h"])
    state_text = np.array2string(error, precision=5, separator=",")
    cost_1 = float(costs[0]["total_rollout_cost"])
    cost_2 = float(costs[1]["total_rollout_cost"])

    if name == "costs_first":
        return f"""Urban perimeter-controller selection at t={time_h:.2f} h.
The deterministic supervisor, not this model, controls execution.
Current controller: mode {active}.
Accumulation deviations [n1,n2,n11,n22,n10,n20]: {state_text}
Candidate costs are intentionally listed in reverse label order:
- mode_2_cost = {cost_2:.8f}
- mode_1_cost = {cost_1:.8f}
Select the smaller numerical cost: choose select_mode_1 exactly when mode_1_cost < mode_2_cost; otherwise choose select_mode_2. Selecting the active mode means staying. There is no separate hold action or switching reward.
Return only JSON matching the supplied schema, with basis lower_rollout_cost."""

    if name == "compact":
        return f"""Choose the installed urban perimeter controller with the lower declared 15-minute rollout cost. A deterministic supervisor separately decides execution.
t={time_h:.2f} h; active_mode={active}; deviations[n1,n2,n11,n22,n10,n20]={state_text}; mode_1_cost={cost_1:.8f}; mode_2_cost={cost_2:.8f}.
Apply the exact numerical argmin: select_mode_1 if mode_1_cost < mode_2_cost, else select_mode_2. The active label is a stay selection, and switching has no extra reward or penalty.
Return only JSON matching the supplied schema, with basis lower_rollout_cost."""

    raise ValueError(name)


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"events", "trajectory"}
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(live.json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def configuration_paths(output_dir: Path, prompt_name: str, seed: int) -> tuple[Path, Path]:
    stem = f"{prompt_name}_scenario_{seed}"
    return output_dir / f"{stem}_summary.json", output_dir / f"{stem}_trace.json"


def validate_resumed_trace(
    trace: dict[str, Any], model: str, prompt_name: str, seed: int
) -> None:
    assert trace["model"] == model
    assert trace["prompt_template"] == prompt_name
    assert trace["scenario_seed"] == seed
    assert trace["number_of_live_calls"] == NUMBER_OF_DECISIONS
    assert len(trace["calls"]) == NUMBER_OF_DECISIONS
    assert all(call["raw_response"].strip() for call in trace["calls"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--base-seed", type=int, default=471)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun configurations even when a complete trace already exists",
    )
    args = parser.parse_args()

    model_key = live.safe_model_name(args.model)
    output_dir = OUTPUT_ROOT / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    gains, certificate, tube = live.load_records()
    client = live.OllamaProposer(args.endpoint, args.model, args.base_seed, args.timeout)
    model_record = client.model_record()
    observation_cache = {
        seed: live.build_reference_observations(
            gains, certificate, tube, priority_schedule(seed)
        )
        for seed in SCENARIO_SEEDS
    }

    summaries: list[dict[str, Any]] = []
    for prompt_name in PROMPT_TEMPLATES:
        for scenario_seed in SCENARIO_SEEDS:
            summary_path, trace_path = configuration_paths(
                output_dir, prompt_name, scenario_seed
            )
            if trace_path.exists() and not args.no_resume:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                validate_resumed_trace(trace, args.model, prompt_name, scenario_seed)
                summaries.append(trace["summary"])
                print(
                    f"RESUME {args.model} {prompt_name} {scenario_seed}: "
                    f"{NUMBER_OF_DECISIONS} calls",
                    flush=True,
                )
                continue

            observations = observation_cache[scenario_seed]
            schedule = priority_schedule(scenario_seed)
            client.seed = args.base_seed + scenario_seed
            calls = []
            for index, observation in enumerate(observations):
                prompt = prompt_variant(prompt_name, observation)
                call = client.propose(prompt, index)
                call["prompt"] = prompt
                call["observation"] = observation
                calls.append(call)
                print(
                    f"CALL {args.model} {prompt_name} {scenario_seed} "
                    f"{index + 1}/{NUMBER_OF_DECISIONS}",
                    flush=True,
                )

            proposed_modes = np.asarray(
                [live.action_mode(call["proposal"]["action"]) for call in calls],
                dtype=int,
            )
            lower_cost_modes = np.asarray(
                [int(call["observation"]["lower_cost_mode"]) for call in calls],
                dtype=int,
            )
            results = {
                policy: live.replay(
                    policy, calls, gains, certificate, tube, priority_modes=schedule
                )
                for policy in (
                    "unrestricted",
                    "landing_only",
                    "time_envelope",
                    "dwell_certified",
                )
            }
            summary = {
                "experiment": "cubic-MFD live prompt/scenario factorial",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "model": args.model,
                "installed_model_record": model_record,
                "temperature": 0.0,
                "base_seed": args.base_seed,
                "prompt_template": prompt_name,
                "scenario_seed": scenario_seed,
                "priority_schedule": schedule,
                "number_of_live_calls": len(calls),
                "prompt_sha256_sequence": hashlib.sha256(
                    "".join(call["prompt_sha256"] for call in calls).encode("ascii")
                ).hexdigest(),
                "lower_cost_matches": int(
                    np.count_nonzero(proposed_modes == lower_cost_modes)
                ),
                "lower_cost_match_rate": float(
                    np.mean(proposed_modes == lower_cost_modes)
                ),
                "proposal_counts": {
                    "mode_1": int(np.count_nonzero(proposed_modes == 0)),
                    "mode_2": int(np.count_nonzero(proposed_modes == 1)),
                },
                "results": {
                    policy: compact_result(result)
                    for policy, result in results.items()
                },
            }
            trace = {
                "summary": summary,
                "model": args.model,
                "prompt_template": prompt_name,
                "scenario_seed": scenario_seed,
                "number_of_live_calls": len(calls),
                "proposal_schema": live.PROPOSAL_SCHEMA,
                "calls": calls,
                "policy_events": {
                    policy: result["events"] for policy, result in results.items()
                },
            }
            write_json_atomic(summary_path, summary)
            write_json_atomic(trace_path, trace)
            summaries.append(summary)
            print(
                f"DONE {args.model} {prompt_name} {scenario_seed}: "
                f"match={summary['lower_cost_matches']}/{NUMBER_OF_DECISIONS}",
                flush=True,
            )

    model_summary = {
        "experiment": "cubic-MFD live prompt/scenario factorial",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "installed_model_record": model_record,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "scenario_seeds": list(SCENARIO_SEEDS),
        "configurations": summaries,
        "aggregate": {
            "configuration_count": len(summaries),
            "total_live_calls": sum(item["number_of_live_calls"] for item in summaries),
            "mean_lower_cost_match_rate": float(
                np.mean([item["lower_cost_match_rate"] for item in summaries])
            ),
            "minimum_lower_cost_match_rate": float(
                min(item["lower_cost_match_rate"] for item in summaries)
            ),
            "maximum_certified_tube_occupancy": float(
                max(
                    item["results"][policy]["maximum_tube_occupancy"]
                    for item in summaries
                    for policy in ("landing_only", "time_envelope", "dwell_certified")
                )
            ),
        },
    }
    model_summary_path = OUTPUT_ROOT / f"mfd_cubic_factorial_{model_key}_summary.json"
    write_json_atomic(model_summary_path, model_summary)
    print(json.dumps(live.json_ready(model_summary["aggregate"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
