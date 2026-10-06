#!/usr/bin/env python3
"""Audit the theorem-neutral cubic-MFD LLM interface improvement study."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "live_llm_results" / "cubic" / "prompt_improvement"
AGGREGATE = LIVE / "mfd_cubic_prompt_improvement_aggregate.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(value: float, expected: float, atol: float = 5.0e-10) -> None:
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=atol), (value, expected)


def main() -> None:
    record = read_json(AGGREGATE)
    development = record["development"]
    held = record["held_out"]
    assert set(development["scenario_seeds"]).isdisjoint(held["scenario_seeds"])
    assert development["selected_variant"] == "grounded_numeric"
    selected_development = development["variants"]["grounded_numeric"]["overall"]
    assert selected_development["total_live_calls"] == 144
    assert selected_development["lower_cost_matches"] == 144

    baseline = held["variants"]["current_rule_first"]["overall"]
    grounded = held["variants"]["grounded_numeric"]["overall"]
    assert baseline["total_live_calls"] == grounded["total_live_calls"] == 360
    assert baseline["lower_cost_matches"] == 297
    assert grounded["lower_cost_matches"] == 360
    close(baseline["lower_cost_match_rate"], 0.825)
    close(grounded["lower_cost_match_rate"], 1.0)
    close(held["absolute_match_rate_gain"], 0.175)
    close(held["relative_error_reduction"], 1.0)
    close(grounded["envelope_cost_improvement_percent_mean"], 3.9761664537778136)
    close(grounded["envelope_cost_improvement_percent_min"], 0.06297647226517313)
    close(grounded["envelope_cost_improvement_percent_max"], 14.251482971872841)
    close(baseline["envelope_cost_improvement_percent_min"], -4.710766483205558)
    assert grounded["maximum_cost_copy_error"] < 5.1e-9
    assert baseline["all_certified_paths_inside_tube"]
    assert grounded["all_certified_paths_inside_tube"]
    assert grounded["maximum_certified_tube_occupancy"] <= 0.8000000001

    total_traces = 0
    total_calls = 0
    for phase, expected_traces in (("development", 24), ("held_out", 30)):
        traces = sorted((LIVE / phase).glob("*/*_trace.json"))
        assert len(traces) == expected_traces
        total_traces += len(traces)
        for path in traces:
            trace = read_json(path)
            assert trace["number_of_live_calls"] == 24
            assert len(trace["calls"]) == 24
            for call in trace["calls"]:
                assert call["raw_response"].strip()
                assert call["prompt"].strip()
                assert len(call["prompt_sha256"]) == 64
                assert call["decoded_response"]
                assert call["proposal"]["basis"] == "lower_rollout_cost"
                total_calls += 1
                if trace["variant"] == "grounded_numeric":
                    observation = call["observation"]
                    decoded = call["decoded_response"]
                    cost_1 = observation["rollout_costs"][0]["total_rollout_cost"]
                    cost_2 = observation["rollout_costs"][1]["total_rollout_cost"]
                    assert abs(decoded["mode_1_cost_read"] - cost_1) < 5.1e-9
                    assert abs(decoded["mode_2_cost_read"] - cost_2) < 5.1e-9
                    assert decoded["selected_mode"] - 1 == observation["lower_cost_mode"]
    assert total_traces == 54
    assert total_calls == 1_296
    assert (LIVE / "mfd_cubic_prompt_improvement_aggregate.pdf").exists()
    assert (LIVE / "mfd_cubic_prompt_improvement_aggregate.png").exists()
    print("PASS: 576 development calls, frozen interface selection, 720 held-out")
    print("      paired calls, exact grounding, certified replays, and aggregate files.")


if __name__ == "__main__":
    main()
