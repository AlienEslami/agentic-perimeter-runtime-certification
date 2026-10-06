#!/usr/bin/env python3
"""Available-model prompt/forecast-seed grid for the live MFD proposer."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np

import live_llm_mfd_validation as live
import mfd_fixed_gain_validation as mfd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "live_llm_results" / "prompt_seed_grid"
PROMPTS = ("current", "costs_reversed", "neutral")
FORECAST_SEEDS = (20260812, 20260813, 20260814, 20260815, 20260816)
ORIGINAL_PROMPT = live.proposal_prompt


def prompt_variant(name: str, decision_time: float, state: np.ndarray, current_mode: int, economics: dict) -> str:
    if name == "current":
        return ORIGINAL_PROMPT(decision_time, state, current_mode, economics)
    error = state - mfd.X_STAR
    command = mfd.U_STAR + mfd.GAINS[current_mode] @ error
    forecasts = economics["controller_forecasts"]
    observations = (
        f"time={decision_time:.1f} h; current mode={current_mode + 1}; "
        f"state deviations={np.array2string(error, precision=6, separator=',')}; "
        f"commands={np.array2string(command, precision=6, separator=',')}; "
        f"forecast errors={np.array2string(economics['regional_forecast_errors'], precision=6, separator=',')}."
    )
    if name == "costs_reversed":
        return f"""You select one of two installed perimeter controllers. A separate runtime supervisor controls execution. Minimize the predicted 30-minute net operating cost. Selecting the active mode means stay and there is no switching reward or penalty.
{observations}
Mode 2 predicted net operating cost: ${forecasts[1]['net_operating_cost_dollars']:.2f}
Mode 1 predicted net operating cost: ${forecasts[0]['net_operating_cost_dollars']:.2f}
Return only JSON matching the schema."""
    if name == "neutral":
        return f"""You are the controller-selection agent for a two-region traffic network. A separate runtime supervisor controls execution.
{observations}
The installed alternatives have these 30-minute rollout results:
mode 1 net operating cost = ${forecasts[0]['net_operating_cost_dollars']:.2f}
mode 2 net operating cost = ${forecasts[1]['net_operating_cost_dollars']:.2f}
Select the controller you recommend. Return only JSON matching the schema."""
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:latest")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base-seed", type=int, default=471)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--allow-thinking",
        action="store_true",
        help="leave model-specific reasoning mode enabled; disabled by default for matched decoding",
    )
    parser.add_argument(
        "--output-dir",
        default="live_llm_results/prompt_seed_grid",
        help="repository-relative directory for traces and the grid summary",
    )
    args = parser.parse_args()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    certificate = mfd.independent_certificate()
    tube = mfd.tube_record(certificate)
    client = live.OllamaMFDProposer(
        args.endpoint, args.model, args.temperature, args.base_seed,
        args.timeout, live.PROPOSAL_SCHEMA, None if args.allow_thinking else False,
    )
    model_record = client.model_record()
    original_prompt = live.proposal_prompt
    summaries = []
    number_of_steps = int(round(live.LIVE_HORIZON / mfd.PROPOSAL_INTERVAL))
    try:
        for prompt_name in PROMPTS:
            live.proposal_prompt = lambda time_value, state, mode, economics, selected=prompt_name: prompt_variant(
                selected, time_value, state, mode, economics
            )
            for forecast_seed in FORECAST_SEEDS:
                forecast = live.rolling_forecast_errors(number_of_steps, forecast_seed)
                certified, calls = live.live_certified_run(client, certificate, tube, forecast)
                unrestricted = live.unrestricted_replay(calls, certificate, tube)
                preferred = np.asarray([
                    call["observation"]["economics"]["lower_cost_mode"] for call in calls
                ])
                proposed = np.asarray([
                    live.action_mode(call["proposal"]["action"]) for call in calls
                ])
                record = {
                    "model": args.model,
                    "temperature": args.temperature,
                    "base_seed": args.base_seed,
                    "prompt_template": prompt_name,
                    "forecast_seed": forecast_seed,
                    "calls": len(calls),
                    "lower_cost_match_count": int(np.count_nonzero(preferred == proposed)),
                    "lower_cost_match_rate": float(np.mean(preferred == proposed)),
                    "requested_changes_on_certified_path": int(sum(
                        proposal != call["observation"]["current_mode"]
                        for proposal, call in zip(proposed, calls)
                    )),
                    "certified": live.simulation_summary(certified),
                    "unrestricted": live.simulation_summary(unrestricted),
                }
                summaries.append(record)
                trace_name = f"{prompt_name}_forecast_{forecast_seed}.json"
                (output_dir / trace_name).write_text(
                    json.dumps(live.json_ready({"summary": record, "calls": calls}), indent=2),
                    encoding="utf-8",
                )
                print(json.dumps(record))
    finally:
        live.proposal_prompt = original_prompt
    output = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "installed_model_record": model_record,
        "available_model_count": 1,
        "prompt_templates": list(PROMPTS),
        "forecast_seeds": list(FORECAST_SEEDS),
        "configurations": summaries,
        "aggregate": {
            "mean_lower_cost_match_rate": float(np.mean([item["lower_cost_match_rate"] for item in summaries])),
            "certified_maximum_occupancy": float(max(item["certified"]["maximum_tube_occupancy"] for item in summaries)),
            "unrestricted_maximum_occupancy": float(max(item["unrestricted"]["maximum_tube_occupancy"] for item in summaries)),
            "unrestricted_earliest_tube_exit_h": float(min(
                item["unrestricted"]["first_tube_exit_time_h"]
                for item in summaries if item["unrestricted"]["first_tube_exit_time_h"] is not None
            )),
        },
        "scope": (
            "The grid varies every prompt and forecast-seed factor available locally. "
            "Additional model families require separately installed weights or API access."
        ),
    }
    (output_dir / "mfd_llm_prompt_seed_grid_summary.json").write_text(
        json.dumps(live.json_ready(output), indent=2), encoding="utf-8"
    )
    print(json.dumps(live.json_ready(output["aggregate"]), indent=2))


if __name__ == "__main__":
    main()
