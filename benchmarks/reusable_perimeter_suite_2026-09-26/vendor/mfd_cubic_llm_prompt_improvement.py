#!/usr/bin/env python3
"""Theorem-neutral live-LLM interface study for the cubic MFD.

The plant, rollout costs, reference path, certificate, and replay policies are
identical to the principal factorial.  Only the user-facing prompt and response
encoding change.  Development and held-out scenario seeds are disjoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

import live_llm_mfd_cubic_validation as live
import mfd_cubic_llm_factorial as factorial
import mfd_cubic_pilot as mfd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "live_llm_results" / "cubic" / "prompt_improvement"
DEVELOPMENT_SEEDS = (20260817, 20260818)
HELD_OUT_SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)
DEVELOPMENT_VARIANTS = (
    "current_rule_first",
    "minimal_numeric",
    "grounded_numeric",
    "signed_gap_numeric",
)

NUMERIC_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_mode": {"type": "integer", "minimum": 1, "maximum": 2},
        "basis": {"type": "string", "enum": ["lower_rollout_cost"]},
    },
    "required": ["selected_mode", "basis"],
    "additionalProperties": False,
}

GROUNDED_SCHEMA = {
    "type": "object",
    "properties": {
        "mode_1_cost_read": {"type": "number"},
        "mode_2_cost_read": {"type": "number"},
        "selected_mode": {"type": "integer", "minimum": 1, "maximum": 2},
        "basis": {"type": "string", "enum": ["lower_rollout_cost"]},
    },
    "required": [
        "mode_1_cost_read",
        "mode_2_cost_read",
        "selected_mode",
        "basis",
    ],
    "additionalProperties": False,
}


def schema_for(variant: str) -> dict[str, Any]:
    if variant == "current_rule_first":
        return live.PROPOSAL_SCHEMA
    if variant == "grounded_numeric":
        return GROUNDED_SCHEMA
    return NUMERIC_SCHEMA


def prompt_for(variant: str, observation: dict[str, Any]) -> str:
    if variant == "current_rule_first":
        return live.proposal_prompt(observation)

    costs = observation["rollout_costs"]
    cost_1 = float(costs[0]["total_rollout_cost"])
    cost_2 = float(costs[1]["total_rollout_cost"])

    if variant == "minimal_numeric":
        return f"""Compare two decimal rollout costs for installed traffic controllers.
The controller with the strictly smaller number wins.
mode 1 cost: {cost_1:.8f}
mode 2 cost: {cost_2:.8f}
Set selected_mode to 1 when the mode 1 cost is smaller; otherwise set it to 2.
Return only JSON matching the supplied schema with basis lower_rollout_cost."""

    if variant == "grounded_numeric":
        return f"""Read and compare the two decimal rollout costs below.
mode 1 cost: {cost_1:.8f}
mode 2 cost: {cost_2:.8f}
First copy each displayed cost into its corresponding JSON field. Then set
selected_mode to the label whose copied cost is strictly smaller. Do not choose
by label order. Return only JSON matching the supplied schema and set basis to
lower_rollout_cost."""

    if variant == "signed_gap_numeric":
        gap = cost_1 - cost_2
        return f"""Select the lower-cost installed traffic controller.
mode 1 cost: {cost_1:.8f}
mode 2 cost: {cost_2:.8f}
The precomputed signed gap mode_1_cost - mode_2_cost is {gap:+.8f}.
A negative gap means selected_mode=1. A zero or positive gap means
selected_mode=2. Return only JSON matching the supplied schema with basis
lower_rollout_cost."""

    raise ValueError(variant)


def call_model(
    endpoint: str,
    model: str,
    seed: int,
    prompt: str,
    schema: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": schema,
        "options": {"temperature": 0.0, "seed": seed, "num_predict": 100},
        "keep_alive": "10m",
    }
    if model.startswith("qwen3"):
        body["think"] = False
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/generate",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    elapsed = time.perf_counter() - started
    raw = payload["response"]
    decoded = json.loads(raw)
    return {
        "request": body,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_response": raw,
        "decoded_response": decoded,
        "latency_seconds": elapsed,
        "ollama_record": {
            "model": payload.get("model"),
            "created_at": payload.get("created_at"),
            "done_reason": payload.get("done_reason"),
            "prompt_tokens": payload.get("prompt_eval_count"),
            "completion_tokens": payload.get("eval_count"),
            "total_duration_ns": payload.get("total_duration"),
        },
    }


def standardize_call(call: dict[str, Any], variant: str) -> None:
    decoded = call["decoded_response"]
    if variant == "current_rule_first":
        action = decoded.get("action")
        if action not in live.PROPOSAL_SCHEMA["properties"]["action"]["enum"]:
            raise ValueError(f"Invalid action: {decoded!r}")
    else:
        selected = decoded.get("selected_mode")
        if selected not in (1, 2):
            raise ValueError(f"Invalid selected_mode: {decoded!r}")
        action = f"select_mode_{selected}"
    if decoded.get("basis") != "lower_rollout_cost":
        raise ValueError(f"Invalid basis: {decoded!r}")
    call["proposal"] = {"action": action, "basis": "lower_rollout_cost"}


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"events", "trajectory"}}


def phase_definition(phase: str, variants: list[str] | None) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if phase == "development":
        return DEVELOPMENT_SEEDS, tuple(variants or DEVELOPMENT_VARIANTS)
    if not variants:
        raise SystemExit("Held-out phase requires at least one --variant")
    return HELD_OUT_SEEDS, tuple(variants)


def configuration_paths(
    output_dir: Path, variant: str, scenario_seed: int
) -> tuple[Path, Path]:
    stem = f"{variant}_scenario_{scenario_seed}"
    return output_dir / f"{stem}_summary.json", output_dir / f"{stem}_trace.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(live.json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--phase", choices=("development", "held_out"), required=True)
    parser.add_argument("--variant", action="append", choices=DEVELOPMENT_VARIANTS)
    parser.add_argument("--base-seed", type=int, default=941)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    seeds, variants = phase_definition(args.phase, args.variant)
    model_key = live.safe_model_name(args.model)
    output_dir = OUTPUT_ROOT / args.phase / model_key
    output_dir.mkdir(parents=True, exist_ok=True)
    gains, certificate, tube = live.load_records()
    model_record = live.OllamaProposer(
        args.endpoint, args.model, args.base_seed, args.timeout
    ).model_record()
    observations = {
        seed: live.build_reference_observations(
            gains, certificate, tube, factorial.priority_schedule(seed)
        )
        for seed in seeds
    }

    summaries: list[dict[str, Any]] = []
    for variant in variants:
        schema = schema_for(variant)
        for scenario_seed in seeds:
            summary_path, trace_path = configuration_paths(
                output_dir, variant, scenario_seed
            )
            if trace_path.exists() and not args.no_resume:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                if (
                    trace.get("model") == args.model
                    and trace.get("variant") == variant
                    and trace.get("scenario_seed") == scenario_seed
                    and len(trace.get("calls", [])) == factorial.NUMBER_OF_DECISIONS
                ):
                    summaries.append(trace["summary"])
                    print(
                        f"RESUME {args.model} {args.phase} {variant} {scenario_seed}",
                        flush=True,
                    )
                    continue

            calls: list[dict[str, Any]] = []
            for index, observation in enumerate(observations[scenario_seed]):
                prompt = prompt_for(variant, observation)
                call = call_model(
                    args.endpoint,
                    args.model,
                    args.base_seed + scenario_seed + index,
                    prompt,
                    schema,
                    args.timeout,
                )
                standardize_call(call, variant)
                call["prompt"] = prompt
                call["observation"] = observation
                calls.append(call)
                print(
                    f"CALL {args.model} {args.phase} {variant} {scenario_seed} "
                    f"{index + 1}/{factorial.NUMBER_OF_DECISIONS}",
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
            schedule = factorial.priority_schedule(scenario_seed)
            results = {
                policy: live.replay(
                    policy,
                    calls,
                    gains,
                    certificate,
                    tube,
                    priority_modes=schedule,
                )
                for policy in ("unrestricted", "landing_only", "time_envelope", "dwell_certified")
            }
            confusion = {
                f"truth_{truth + 1}_selected_{selected + 1}": int(
                    np.count_nonzero(
                        (lower_cost_modes == truth) & (proposed_modes == selected)
                    )
                )
                for truth in (0, 1)
                for selected in (0, 1)
            }
            grounding_errors: list[float] = []
            if variant == "grounded_numeric":
                for call in calls:
                    decoded = call["decoded_response"]
                    observed = call["observation"]["rollout_costs"]
                    grounding_errors.extend(
                        [
                            abs(float(decoded["mode_1_cost_read"]) - float(observed[0]["total_rollout_cost"])),
                            abs(float(decoded["mode_2_cost_read"]) - float(observed[1]["total_rollout_cost"])),
                        ]
                    )
            summary = {
                "experiment": "cubic-MFD theorem-neutral LLM interface improvement",
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "phase": args.phase,
                "model": args.model,
                "installed_model_record": model_record,
                "temperature": 0.0,
                "base_seed": args.base_seed,
                "variant": variant,
                "scenario_seed": scenario_seed,
                "priority_schedule": schedule,
                "number_of_live_calls": len(calls),
                "lower_cost_matches": int(np.count_nonzero(proposed_modes == lower_cost_modes)),
                "lower_cost_match_rate": float(np.mean(proposed_modes == lower_cost_modes)),
                "confusion": confusion,
                "maximum_grounding_error": max(grounding_errors, default=None),
                "mean_latency_seconds": float(np.mean([call["latency_seconds"] for call in calls])),
                "results": {policy: compact_result(result) for policy, result in results.items()},
            }
            trace = {
                "summary": summary,
                "phase": args.phase,
                "model": args.model,
                "variant": variant,
                "scenario_seed": scenario_seed,
                "number_of_live_calls": len(calls),
                "response_schema": schema,
                "calls": calls,
                "policy_events": {policy: result["events"] for policy, result in results.items()},
            }
            write_json_atomic(summary_path, summary)
            write_json_atomic(trace_path, trace)
            summaries.append(summary)
            print(
                f"DONE {args.model} {args.phase} {variant} {scenario_seed}: "
                f"match={summary['lower_cost_matches']}/{factorial.NUMBER_OF_DECISIONS}",
                flush=True,
            )

    model_summary = {
        "experiment": "cubic-MFD theorem-neutral LLM interface improvement",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "phase": args.phase,
        "model": args.model,
        "installed_model_record": model_record,
        "scenario_seeds": list(seeds),
        "variants": list(variants),
        "configurations": summaries,
        "aggregate": {
            "configuration_count": len(summaries),
            "total_live_calls": sum(item["number_of_live_calls"] for item in summaries),
            "mean_lower_cost_match_rate": float(np.mean([item["lower_cost_match_rate"] for item in summaries])),
            "minimum_lower_cost_match_rate": float(min(item["lower_cost_match_rate"] for item in summaries)),
            "maximum_certified_tube_occupancy": float(
                max(
                    item["results"][policy]["maximum_tube_occupancy"]
                    for item in summaries
                    for policy in ("landing_only", "time_envelope", "dwell_certified")
                )
            ),
        },
    }
    model_summary_path = OUTPUT_ROOT / args.phase / f"{model_key}_summary.json"
    write_json_atomic(model_summary_path, model_summary)
    print(json.dumps(live.json_ready(model_summary["aggregate"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
