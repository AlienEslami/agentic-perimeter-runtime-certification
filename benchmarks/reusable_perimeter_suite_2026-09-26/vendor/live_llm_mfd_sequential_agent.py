#!/usr/bin/env python3
"""Live sequential language-to-rollout-tool study for the cubic MFD.

See ``MFD_SEQUENTIAL_LLM_PROTOCOL.md``.  Unlike the archived scalar-choice
audit, this experiment asks the model to maintain an evolving operator
objective, request a deterministic traffic rollout tool, inspect its component
outputs, and form a controller proposal through a deterministic comparison.
The certified supervisor retains sole execution authority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import mfd_cubic_nonstationary_study as study
import mfd_cubic_pilot as pilot
import mfd_cubic_runtime_pilot as runtime


ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "mfd-sequential-semantic-agent-v3-five-trace"
OUTPUT_DIR = ROOT / "live_llm_results" / "cubic" / "sequential_semantic_agent_v3"
AGGREGATE_JSON = OUTPUT_DIR / "mfd_sequential_semantic_agent_aggregate.json"
FIGURE_PDF = OUTPUT_DIR / "mfd_sequential_semantic_agent_aggregate.pdf"
FIGURE_PNG = OUTPUT_DIR / "mfd_sequential_semantic_agent_aggregate.png"

MODELS = ("llama3.1:latest", "qwen3:8b", "mistral:7b")
VARIANTS = ("stateful", "memoryless")
BASE_SEED = 20260902
TEMPERATURE = 0.0
DECISION_MINUTES = 30.0
AUDIT_MINUTES = 1.0
DEMAND_AMPLITUDE = 0.003
ADT_AVERAGE_MINUTES = 105.0
ADT_BURST_CAPACITY = 2
INITIAL_OCCUPANCY = 0.8

INCIDENTS = (
    {
        "directive": (
            "A stadium release is building in Region 1 and should persist for "
            "about 60 minutes. Prioritize clearing Region 1 and assess a "
            "60-minute operating horizon."
        ),
        "objective": "region_1",
        "horizon_minutes": 60,
        "context_dependent": False,
    },
    {
        "directive": "Continue the active incident plan for another interval; conditions have not cleared.",
        "objective": "region_1",
        "horizon_minutes": 60,
        "context_dependent": True,
    },
    {
        "directive": "The stadium release has cleared. Restore balanced operation using a 30-minute horizon.",
        "objective": "balanced",
        "horizon_minutes": 30,
        "context_dependent": False,
    },
    {
        "directive": (
            "A freight-terminal surge is forming in Region 2 and is expected "
            "to last 90 minutes. Prioritize Region 2 discharge over that horizon."
        ),
        "objective": "region_2",
        "horizon_minutes": 90,
        "context_dependent": False,
    },
    {
        "directive": "Continue the active surge plan and keep its current planning horizon.",
        "objective": "region_2",
        "horizon_minutes": 90,
        "context_dependent": True,
    },
    {
        "directive": (
            "Give Region 1 temporary emergency-access priority for one "
            "30-minute interval, then retain the prior freight plan for return."
        ),
        "objective": "region_1",
        "horizon_minutes": 30,
        "context_dependent": False,
    },
    {
        "directive": "The emergency interval is over. Return to the plan that was active before it.",
        "objective": "region_2",
        "horizon_minutes": 90,
        "context_dependent": True,
    },
    {
        "directive": "All special events are cleared. Restore balanced operation using a 30-minute horizon.",
        "objective": "balanced",
        "horizon_minutes": 30,
        "context_dependent": False,
    },
)

SCENARIOS = (
    {
        "id": "stadium_freight",
        "initial_direction": "region_1_heavy",
        "incidents": INCIDENTS,
    },
    {
        "id": "rail_downtown",
        "initial_direction": "region_2_heavy",
        "incidents": (
            {
                "directive": "A rail-terminal disruption is loading Region 2 for about 60 minutes. Prioritize Region 2 and use a 60-minute horizon.",
                "objective": "region_2", "horizon_minutes": 60, "context_dependent": False,
            },
            {
                "directive": "Continue the active disruption response with the same horizon.",
                "objective": "region_2", "horizon_minutes": 60, "context_dependent": True,
            },
            {
                "directive": "Rail operations have recovered. Restore balanced service over 30 minutes.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "A downtown event is releasing traffic into Region 1 and will persist for 90 minutes. Prioritize Region 1 over that horizon.",
                "objective": "region_1", "horizon_minutes": 90, "context_dependent": False,
            },
            {
                "directive": "Continue that event plan and retain its planning horizon.",
                "objective": "region_1", "horizon_minutes": 90, "context_dependent": True,
            },
            {
                "directive": "Give Region 2 temporary emergency-access priority for one 30-minute interval, while retaining the downtown plan for return.",
                "objective": "region_2", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "Emergency access is complete. Return to the plan active immediately before it.",
                "objective": "region_1", "horizon_minutes": 90, "context_dependent": True,
            },
            {
                "directive": "The event has ended. Resume balanced operation with a 30-minute horizon.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
        ),
    },
    {
        "id": "bridge_port",
        "initial_direction": "region_1_heavy",
        "incidents": (
            {
                "directive": "A bridge restriction is creating a Region-1 queue expected to last 90 minutes. Prioritize Region 1 using that horizon.",
                "objective": "region_1", "horizon_minutes": 90, "context_dependent": False,
            },
            {
                "directive": "Keep the current restriction plan running and preserve its horizon.",
                "objective": "region_1", "horizon_minutes": 90, "context_dependent": True,
            },
            {
                "directive": "The bridge queue has dissipated. Balance both regions over the next 30 minutes.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "A port release is now loading Region 2 for roughly 60 minutes. Prioritize Region 2 on a 60-minute horizon.",
                "objective": "region_2", "horizon_minutes": 60, "context_dependent": False,
            },
            {
                "directive": "Continue the port-release plan without changing its horizon.",
                "objective": "region_2", "horizon_minutes": 60, "context_dependent": True,
            },
            {
                "directive": "Apply Region-1 transit-access priority for one 30-minute interval and retain the port plan for later resumption.",
                "objective": "region_1", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "The transit interval is finished. Return to the plan that preceded it.",
                "objective": "region_2", "horizon_minutes": 60, "context_dependent": True,
            },
            {
                "directive": "Port demand has cleared. Restore balanced service over 30 minutes.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
        ),
    },
    {
        "id": "weather_sports",
        "initial_direction": "region_2_heavy",
        "incidents": (
            {
                "directive": "Weather-related queuing is concentrated in Region 2 for the next 30 minutes. Prioritize Region 2 on a 30-minute horizon.",
                "objective": "region_2", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "Continue the weather response for another interval with its current horizon.",
                "objective": "region_2", "horizon_minutes": 30, "context_dependent": True,
            },
            {
                "directive": "Weather impacts have cleared. Use balanced operation for 30 minutes.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "A sports venue in Region 1 is releasing traffic for about 60 minutes. Prioritize Region 1 over a 60-minute horizon.",
                "objective": "region_1", "horizon_minutes": 60, "context_dependent": False,
            },
            {
                "directive": "Maintain the active venue plan and keep the same horizon.",
                "objective": "region_1", "horizon_minutes": 60, "context_dependent": True,
            },
            {
                "directive": "Temporarily prioritize Region 2 evacuation access for one 30-minute interval, retaining the venue plan for return.",
                "objective": "region_2", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "The evacuation interval has ended. Return to the preceding plan.",
                "objective": "region_1", "horizon_minutes": 60, "context_dependent": True,
            },
            {
                "directive": "All special demand is cleared. Restore balance using a 30-minute horizon.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
        ),
    },
    {
        "id": "construction_airport",
        "initial_direction": "region_1_heavy",
        "incidents": (
            {
                "directive": "A construction diversion is loading Region 1 for about 60 minutes. Prioritize Region 1 with a 60-minute horizon.",
                "objective": "region_1", "horizon_minutes": 60, "context_dependent": False,
            },
            {
                "directive": "Continue the active diversion strategy and retain its horizon.",
                "objective": "region_1", "horizon_minutes": 60, "context_dependent": True,
            },
            {
                "directive": "The diversion has ended. Restore balanced service over 30 minutes.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "An airport arrival bank is loading Region 2 for the next 90 minutes. Prioritize Region 2 over that horizon.",
                "objective": "region_2", "horizon_minutes": 90, "context_dependent": False,
            },
            {
                "directive": "Keep the airport plan active and preserve its current horizon.",
                "objective": "region_2", "horizon_minutes": 90, "context_dependent": True,
            },
            {
                "directive": "Give Region 1 one 30-minute emergency-service interval and retain the airport plan for resumption.",
                "objective": "region_1", "horizon_minutes": 30, "context_dependent": False,
            },
            {
                "directive": "Emergency service is complete. Return to the operating plan active before it.",
                "objective": "region_2", "horizon_minutes": 90, "context_dependent": True,
            },
            {
                "directive": "The airport surge is over. Resume balanced service using 30 minutes.",
                "objective": "balanced", "horizon_minutes": 30, "context_dependent": False,
            },
        ),
    },
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {
            "type": "string",
            "enum": ["region_1", "region_2", "balanced"],
        },
        "horizon_minutes": {"type": "integer", "enum": [30, 60, 90]},
        "memory_update": {"type": "string", "maxLength": 120},
    },
    "required": ["objective", "horizon_minutes", "memory_update"],
    "additionalProperties": False,
}

def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


class OllamaStructuredClient:
    def __init__(self, endpoint: str, model: str, timeout: float) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout

    def model_record(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.endpoint}/api/tags")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        return next(
            (item for item in payload.get("models", []) if item.get("name") == self.model),
            {"name": self.model, "installed_record": None},
        )

    def call(
        self,
        prompt: str,
        schema: dict[str, Any],
        call_index: int,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {
                "temperature": TEMPERATURE,
                "seed": BASE_SEED + call_index,
                "num_predict": 220,
            },
            "keep_alive": "15m",
        }
        if self.model.startswith("qwen3"):
            body["think"] = False
        request = urllib.request.Request(
            f"{self.endpoint}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.load(response)
        elapsed = time.perf_counter() - started
        raw = payload.get("response", "")
        parsed = None
        error = None
        try:
            parsed = json.loads(raw)
            required = set(schema["required"])
            if not required.issubset(parsed):
                raise ValueError(f"missing keys {sorted(required - set(parsed))}")
            for key, specification in schema["properties"].items():
                if "enum" in specification and parsed[key] not in specification["enum"]:
                    raise ValueError(f"invalid {key}: {parsed[key]!r}")
        except (json.JSONDecodeError, TypeError, ValueError) as exception:
            error = str(exception)
        return {
            "request": body,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_response": raw,
            "parsed": parsed,
            "valid": error is None,
            "validation_error": error,
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


def directive_delta(objective: str) -> float:
    if objective == "region_1":
        return DEMAND_AMPLITUDE
    if objective == "region_2":
        return -(2.0 / 3.0) * DEMAND_AMPLITUDE
    return 0.0


def demand_for(objective: str) -> np.ndarray:
    return pilot.DEMAND + directive_delta(objective) * study.shift_direction()


def rollout_components(
    state: np.ndarray,
    objective: str,
    horizon_minutes: int,
    gains: tuple[np.ndarray, ...],
) -> dict[str, Any]:
    horizon_h = horizon_minutes / 60.0
    demand = demand_for(objective)
    output = []
    for mode in range(2):
        sample_times = np.linspace(0.0, horizon_h, 61)

        def rhs(_time_h: float, current: np.ndarray) -> np.ndarray:
            command = pilot.U_STAR + gains[mode] @ (current - pilot.X_STAR)
            return pilot.mfd_rhs(current, command, demand=demand)

        solution = solve_ivp(
            rhs,
            (0.0, horizon_h),
            state,
            method="DOP853",
            t_eval=sample_times,
            rtol=1.0e-10,
            atol=1.0e-12,
            max_step=min(0.005, horizon_h / 50.0),
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        states = solution.y.T
        errors = states - pilot.X_STAR
        commands = np.asarray([pilot.U_STAR + gains[mode] @ error for error in errors])
        margins = [study.constraint_margins(item, mode, gains) for item in states]
        rates = [
            pilot.transportation_rates(item, command)
            for item, command in zip(states, commands)
        ]
        output.append(
            {
                "mode": mode + 1,
                "region_1_vehicle_hours": float(
                    pilot.ACCUMULATION_UNIT_VEHICLES
                    * np.trapezoid(states[:, 0], sample_times)
                ),
                "region_2_vehicle_hours": float(
                    pilot.ACCUMULATION_UNIT_VEHICLES
                    * np.trapezoid(states[:, 1], sample_times)
                ),
                "network_vehicle_hours": float(
                    pilot.ACCUMULATION_UNIT_VEHICLES
                    * np.trapezoid(states[:, 0] + states[:, 1], sample_times)
                ),
                "imbalance_vehicle_hours": float(
                    pilot.ACCUMULATION_UNIT_VEHICLES
                    * np.trapezoid(np.abs(states[:, 0] - states[:, 1]), sample_times)
                ),
                "completed_trips_vehicles": float(
                    pilot.ACCUMULATION_UNIT_VEHICLES
                    * np.trapezoid(
                        [item["completed_trips_units_per_hour"] for item in rates],
                        sample_times,
                    )
                ),
                "minimum_state_margin_vehicles": min(item[0] for item in margins),
                "minimum_input_margin": min(item[1] for item in margins),
            }
        )
    primary_key = {
        "region_1": "region_1_vehicle_hours",
        "region_2": "region_2_vehicle_hours",
        "balanced": "imbalance_vehicle_hours",
    }[objective]
    ranked = sorted(
        range(2),
        key=lambda index: (
            output[index][primary_key], output[index]["network_vehicle_hours"]
        ),
    )
    return {
        "objective": objective,
        "horizon_minutes": horizon_minutes,
        "primary_component": primary_key,
        "mode_results": output,
        "lower_primary_component_mode": ranked[0] + 1,
        "primary_component_gap": float(
            output[ranked[1]][primary_key] - output[ranked[0]][primary_key]
        ),
    }


def compact_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "No prior decisions."
    lines = []
    for item in reversed(history[-4:]):
        lines.append(
            f"t={item['time_h']:.1f}h directive={item['directive']} "
            f"tool={item['tool_objective']}/{item['tool_horizon_minutes']}min "
            f"proposal={item['proposed_mode']} outcome={item['supervisor_outcome']}"
        )
    return "MOST RECENT FIRST:\n" + "\n".join(lines)


def tool_prompt(
    directive: str,
    time_h: float,
    state: np.ndarray,
    active_mode: int,
    history: list[dict[str, Any]],
    variant: str,
) -> str:
    error_vehicles = pilot.ACCUMULATION_UNIT_VEHICLES * (state - pilot.X_STAR)
    context = compact_history(history) if variant == "stateful" else "Prior context is not supplied."
    return f"""You are the planning component of an urban traffic operations agent.
Interpret the operator's current instruction and request one deterministic controller-rollout tool. Do not select a controller yet.

Allowed objective labels:
- region_1: prioritize clearing Region 1;
- region_2: prioritize clearing Region 2;
- balanced: restore balanced network service.
Allowed horizons are 30, 60, or 90 minutes. Instructions such as continue or return refer to prior context when it is supplied.
For continue, use the most recent active plan. For return after a temporary intervention, use the plan immediately preceding that intervention.

Current time: {time_h:.1f} h
Active controller: mode {active_mode + 1}
Accumulation deviations in vehicles [n1,n2,n11,n22,n10,n20]: {np.array2string(error_vehicles, precision=2, separator=',')}
Operator directive: {directive}

Prior sequential context:
{context}

Return only JSON matching the supplied schema. memory_update must be a compact record of at most 120 characters useful at the next decision."""


def advance_plant(
    state: np.ndarray,
    mode: int,
    objective: str,
    start_h: float,
    gains: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    end_h = start_h + DECISION_MINUTES / 60.0
    audit_h = AUDIT_MINUTES / 60.0
    sample_times = np.arange(start_h + audit_h, end_h + 0.5 * audit_h, audit_h)
    sample_times[-1] = end_h
    demand = demand_for(objective)

    def rhs(_time_h: float, current: np.ndarray) -> np.ndarray:
        command = pilot.U_STAR + gains[mode] @ (current - pilot.X_STAR)
        return pilot.mfd_rhs(current, command, demand=demand)

    solution = solve_ivp(
        rhs,
        (start_h, end_h),
        state,
        method="DOP853",
        t_eval=sample_times,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_step=0.0025,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1], solution.t, solution.y.T


def run_trace(
    client: OllamaStructuredClient | None,
    variant: str,
    gains: tuple[np.ndarray, ...],
    certificate: dict[str, Any],
    tube: dict[str, Any],
    incidents: tuple[dict[str, Any], ...] = INCIDENTS,
    scenario_id: str = "stadium_freight",
    initial_direction: str = "region_1_heavy",
    call_offset: int = 0,
) -> dict[str, Any]:
    robust = study.robust_record(
        DEMAND_AMPLITUDE, DECISION_MINUTES, certificate, tube
    )
    state = runtime.state_on_tube(
        runtime.structured_directions()[initial_direction],
        INITIAL_OCCUPANCY,
        certificate,
        tube,
        mode=0,
    )
    mode = 0
    initial_error = state - pilot.X_STAR
    initial_level = float(
        initial_error @ certificate["matrices"][mode] @ initial_error
    )
    tokens = float(ADT_BURST_CAPACITY)
    token_update_h = 0.0
    last_switch_h = -1.0e9
    executed_times: list[float] = []
    rejected = 0
    no_ops = 0
    history: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    times = [0.0]
    states = [state.copy()]
    modes = [mode]
    objectives = [incidents[0]["objective"]]

    for index, incident in enumerate(incidents):
        time_h = index * DECISION_MINUTES / 60.0
        tokens = min(
            float(ADT_BURST_CAPACITY),
            tokens + (time_h - token_update_h) / (ADT_AVERAGE_MINUTES / 60.0),
        )
        token_update_h = time_h

        if client is None:
            tool_request = {
                "objective": incident["objective"],
                "horizon_minutes": incident["horizon_minutes"],
                "memory_update": "deterministic oracle context",
            }
            tool_call = {
                "valid": True,
                "parsed": tool_request,
                "prompt": None,
                "raw_response": None,
                "latency_seconds": 0.0,
            }
        else:
            prompt = tool_prompt(
                incident["directive"], time_h, state, mode, history, variant
            )
            tool_call = client.call(prompt, TOOL_SCHEMA, call_offset + index)
            tool_request = (
                tool_call["parsed"]
                if tool_call["valid"]
                else {
                    "objective": "balanced",
                    "horizon_minutes": 30,
                    "memory_update": "invalid call; safe default",
                }
            )

        tool_result = rollout_components(
            state,
            tool_request["objective"],
            int(tool_request["horizon_minutes"]),
            gains,
        )
        true_result = rollout_components(
            state,
            incident["objective"],
            int(incident["horizon_minutes"]),
            gains,
        )

        requested_mode = int(tool_result["lower_primary_component_mode"]) - 1
        proposal_call = {
            "source": "deterministic component comparison",
            "objective": tool_result["objective"],
            "primary_component": tool_result["primary_component"],
            "tie_breaker": "network_vehicle_hours",
            "proposed_mode": requested_mode + 1,
            "is_reselection": requested_mode == mode,
        }
        tokens_before = tokens
        if requested_mode == mode:
            no_ops += 1
            outcome = "no_op_reselection"
        else:
            admissible, diagnostic = study.evaluate_candidate(
                state,
                requested_mode,
                time_h,
                initial_level,
                gains,
                certificate,
                tube,
                robust,
                "robust_adt_envelope",
                last_switch_h,
                tokens,
            )
            if admissible:
                mode = requested_mode
                tokens -= 1.0
                last_switch_h = time_h
                executed_times.append(time_h)
                outcome = "executed"
            else:
                rejected += 1
                outcome = "rejected"
        selected_result = int(tool_result["lower_primary_component_mode"]) - 1
        true_selected = int(true_result["lower_primary_component_mode"]) - 1
        true_key = true_result["primary_component"]
        true_values = true_result["mode_results"]
        regret = float(
            true_values[requested_mode][true_key]
            - true_values[true_selected][true_key]
        )

        record = {
            "decision_index": index,
            "time_h": time_h,
            "directive": incident["directive"],
            "true_objective": incident["objective"],
            "true_horizon_minutes": incident["horizon_minutes"],
            "context_dependent": incident["context_dependent"],
            "active_mode_before": modes[-1] + 1,
            "tool_call": tool_call,
            "tool_objective": tool_request["objective"],
            "tool_horizon_minutes": tool_request["horizon_minutes"],
            "tool_request_correct": bool(
                tool_request["objective"] == incident["objective"]
                and int(tool_request["horizon_minutes"])
                == int(incident["horizon_minutes"])
            ),
            "tool_result": tool_result,
            "true_objective_rollout": true_result,
            "proposal_call": proposal_call,
            "proposed_mode": requested_mode + 1,
            "tool_consistent_proposal": requested_mode == selected_result,
            "true_oracle_proposal": requested_mode == true_selected,
            "true_primary_component_regret": regret,
            "tokens_before": tokens_before,
            "tokens_after": tokens,
            "supervisor_outcome": outcome,
            "active_mode_after": mode + 1,
        }
        history.append(record)
        calls.append(tool_call)

        state, interval_times, interval_states = advance_plant(
            state, mode, incident["objective"], time_h, gains
        )
        for sample_time, sample_state in zip(interval_times, interval_states):
            times.append(float(sample_time))
            states.append(np.asarray(sample_state, dtype=float))
            modes.append(mode)
            objectives.append(incident["objective"])

    times_array = np.asarray(times)
    states_array = np.asarray(states)
    modes_array = np.asarray(modes, dtype=int)
    errors = states_array - pilot.X_STAR
    values = np.asarray(
        [
            error @ certificate["matrices"][active] @ error
            for error, active in zip(errors, modes_array)
        ]
    )
    margins = [
        study.constraint_margins(state_value, int(active), gains)
        for state_value, active in zip(states_array, modes_array)
    ]
    commands = np.asarray(
        [pilot.U_STAR + gains[active] @ error for active, error in zip(modes_array, errors)]
    )
    rates = [
        pilot.transportation_rates(state_value, command)
        for state_value, command in zip(states_array, commands)
    ]

    count_residual = -math.inf
    endpoints = [0.0, *executed_times, len(incidents) * DECISION_MINUTES / 60.0]
    for left in endpoints:
        for right in endpoints:
            if right + 1.0e-12 < left:
                continue
            count = sum(left < item <= right + 1.0e-12 for item in executed_times)
            count_residual = max(
                count_residual,
                count - (
                    ADT_BURST_CAPACITY
                    + (right - left) / (ADT_AVERAGE_MINUTES / 60.0)
                ),
            )

    explicit = [item for item in history if not item["context_dependent"]]
    contextual = [item for item in history if item["context_dependent"]]
    valid_calls = [item for item in calls if "valid" in item]
    return {
        "scenario_id": scenario_id,
        "variant": variant,
        "model": None if client is None else client.model,
        "decisions": history,
        "summary": {
            "valid_call_rate": float(np.mean([item["valid"] for item in valid_calls])),
            "valid_calls": int(sum(item["valid"] for item in valid_calls)),
            "call_count": len(valid_calls),
            "explicit_request_count": len(explicit),
            "context_request_count": len(contextual),
            "correct_explicit_requests": int(
                sum(item["tool_request_correct"] for item in explicit)
            ),
            "correct_context_requests": int(
                sum(item["tool_request_correct"] for item in contextual)
            ),
            "explicit_tool_request_accuracy": float(
                np.mean([item["tool_request_correct"] for item in explicit])
            ),
            "context_tool_request_accuracy": float(
                np.mean([item["tool_request_correct"] for item in contextual])
            ),
            "all_tool_request_accuracy": float(
                np.mean([item["tool_request_correct"] for item in history])
            ),
            "tool_consistent_proposal_rate": float(
                np.mean([item["tool_consistent_proposal"] for item in history])
            ),
            "true_oracle_proposal_rate": float(
                np.mean([item["true_oracle_proposal"] for item in history])
            ),
            "true_oracle_proposals": int(
                sum(item["true_oracle_proposal"] for item in history)
            ),
            "proposal_count": len(history),
            "mean_true_primary_component_regret": float(
                np.mean([item["true_primary_component_regret"] for item in history])
            ),
            "executed_changes": len(executed_times),
            "rejected_changes": rejected,
            "no_op_reselections": no_ops,
            "maximum_tube_occupancy": float(np.max(values) / tube["common_level"]),
            "minimum_state_constraint_margin_vehicles": min(item[0] for item in margins),
            "minimum_input_constraint_margin": min(item[1] for item in margins),
            "maximum_count_bound_residual": count_residual,
            "count_bound_verified": bool(count_residual <= 1.0e-10),
            "vehicle_hours_in_network": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(states_array[:, 0] + states_array[:, 1], times_array)
            ),
            "regional_imbalance_vehicle_hours": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(np.abs(states_array[:, 0] - states_array[:, 1]), times_array)
            ),
            "completed_trips_vehicles": float(
                pilot.ACCUMULATION_UNIT_VEHICLES
                * np.trapezoid(
                    [item["completed_trips_units_per_hour"] for item in rates],
                    times_array,
                )
            ),
            "total_llm_latency_seconds": float(
                sum(item.get("latency_seconds", 0.0) for item in calls)
            ),
        },
        "trajectory": {
            "time_h": times_array,
            "state": states_array,
            "mode": modes_array,
            "true_objective": objectives,
            "tube_occupancy": values / tube["common_level"],
        },
    }


def aggregate_trace_runs(traces: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = [
        decision for trace in traces for decision in trace["decisions"]
    ]
    summaries = [trace["summary"] for trace in traces]
    explicit = [item for item in decisions if not item["context_dependent"]]
    contextual = [item for item in decisions if item["context_dependent"]]
    call_count = sum(item["call_count"] for item in summaries)
    valid_calls = sum(item["valid_calls"] for item in summaries)
    proposal_count = len(decisions)
    return {
        "scenario_count": len(traces),
        "scenarios": {trace["scenario_id"]: trace for trace in traces},
        "summary": {
            "valid_call_rate": valid_calls / call_count,
            "valid_calls": valid_calls,
            "call_count": call_count,
            "explicit_request_count": len(explicit),
            "context_request_count": len(contextual),
            "correct_explicit_requests": int(
                sum(item["tool_request_correct"] for item in explicit)
            ),
            "correct_context_requests": int(
                sum(item["tool_request_correct"] for item in contextual)
            ),
            "explicit_tool_request_accuracy": float(
                np.mean([item["tool_request_correct"] for item in explicit])
            ),
            "context_tool_request_accuracy": float(
                np.mean([item["tool_request_correct"] for item in contextual])
            ),
            "all_tool_request_accuracy": float(
                np.mean([item["tool_request_correct"] for item in decisions])
            ),
            "tool_consistent_proposal_rate": float(
                np.mean([item["tool_consistent_proposal"] for item in decisions])
            ),
            "true_oracle_proposal_rate": float(
                np.mean([item["true_oracle_proposal"] for item in decisions])
            ),
            "true_oracle_proposals": int(
                sum(item["true_oracle_proposal"] for item in decisions)
            ),
            "proposal_count": proposal_count,
            "mean_true_primary_component_regret": float(
                np.mean([item["true_primary_component_regret"] for item in decisions])
            ),
            "executed_changes": sum(item["executed_changes"] for item in summaries),
            "rejected_changes": sum(item["rejected_changes"] for item in summaries),
            "no_op_reselections": sum(item["no_op_reselections"] for item in summaries),
            "maximum_tube_occupancy": max(
                item["maximum_tube_occupancy"] for item in summaries
            ),
            "minimum_state_constraint_margin_vehicles": min(
                item["minimum_state_constraint_margin_vehicles"] for item in summaries
            ),
            "minimum_input_constraint_margin": min(
                item["minimum_input_constraint_margin"] for item in summaries
            ),
            "maximum_count_bound_residual": max(
                item["maximum_count_bound_residual"] for item in summaries
            ),
            "count_bound_verified": all(
                item["count_bound_verified"] for item in summaries
            ),
            "mean_vehicle_hours_in_network": float(
                np.mean([item["vehicle_hours_in_network"] for item in summaries])
            ),
            "mean_regional_imbalance_vehicle_hours": float(
                np.mean(
                    [item["regional_imbalance_vehicle_hours"] for item in summaries]
                )
            ),
            "mean_completed_trips_vehicles": float(
                np.mean([item["completed_trips_vehicles"] for item in summaries])
            ),
            "total_llm_latency_seconds": sum(
                item["total_llm_latency_seconds"] for item in summaries
            ),
        },
        "trajectory": traces[0]["trajectory"],
    }


def make_figure(payload: dict[str, Any]) -> None:
    runs = payload["runs"]
    models = list(payload["models"])
    x = np.arange(len(models))
    width = 0.34
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 6.9))

    for offset, variant, color in ((-0.5, "stateful", "#245a85"), (0.5, "memoryless", "#b36a43")):
        context = [
            100.0 * runs[model][variant]["summary"]["context_tool_request_accuracy"]
            for model in models
        ]
        proposal = [
            100.0 * runs[model][variant]["summary"]["true_oracle_proposal_rate"]
            for model in models
        ]
        axes[0, 0].bar(x + offset * width, context, width, label=variant, color=color)
        axes[0, 1].bar(x + offset * width, proposal, width, label=variant, color=color)
    labels = [item.split(":")[0] for item in models]
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_ylim(0.0, 105.0)
    axes[0, 0].set_ylabel("correct contextual requests (%)")
    axes[0, 0].set_title("Operator context (15 requests/model)")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_ylim(0.0, 105.0)
    axes[0, 1].set_ylabel("true-objective proposals (%)")
    axes[0, 1].set_title("Decision quality (40 proposals/model)")
    axes[0, 1].legend(frameon=False)

    selected_model = models[0]
    for variant, color in (("stateful", "#245a85"), ("memoryless", "#b36a43")):
        trajectory = runs[selected_model][variant]["trajectory"]
        axes[1, 0].plot(
            trajectory["time_h"], trajectory["tube_occupancy"],
            color=color, label=variant, linewidth=1.8,
        )
    axes[1, 0].axhline(1.0, color="#9b2c2c", linestyle="--", linewidth=1.0)
    axes[1, 0].set_xlabel("time (h)")
    axes[1, 0].set_ylabel("tube occupancy")
    axes[1, 0].set_title(f"Representative certified trace: {labels[0]}")
    axes[1, 0].legend(frameon=False)

    outcome_names = ("executed_changes", "rejected_changes", "no_op_reselections")
    outcome_labels = ("executed changes", "rejected changes", "no-op reselections")
    bottom = np.zeros(len(models))
    for name, label, color in zip(outcome_names, outcome_labels, ("#2d7c5b", "#b24745", "#8d8d8d")):
        values = [runs[model]["stateful"]["summary"][name] for model in models]
        axes[1, 1].bar(x, values, bottom=bottom, label=label, color=color)
        bottom += np.asarray(values)
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylabel("40 proposal epochs")
    axes[1, 1].set_title("Five-trace outcomes (stateful)")
    axes[1, 1].legend(frameon=False, fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURE_PDF, bbox_inches="tight")
    figure.savefig(FIGURE_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gains, certificate, tube = study.load_installed()
    model_records: dict[str, Any] = {}
    runs: dict[str, Any] = {}
    for model_index, model in enumerate(args.models):
        client = OllamaStructuredClient(args.endpoint, model, args.timeout)
        model_records[model] = client.model_record()
        model_runs = {}
        for variant_index, variant in enumerate(VARIANTS):
            traces = []
            for scenario_index, scenario in enumerate(SCENARIOS):
                scenario_id = scenario["id"]
                partial_path = OUTPUT_DIR / (
                    f"{model.replace(':', '_')}_{variant}_{scenario_id}.json"
                )
                if args.resume and partial_path.exists():
                    print(f"RESUME {model} {variant} {scenario_id}", flush=True)
                    result = json.loads(partial_path.read_text(encoding="utf-8"))
                else:
                    print(f"RUN {model} {variant} {scenario_id}", flush=True)
                    result = run_trace(
                        client,
                        variant,
                        gains,
                        certificate,
                        tube,
                        incidents=scenario["incidents"],
                        scenario_id=scenario_id,
                        initial_direction=scenario["initial_direction"],
                        call_offset=(
                            model_index * 10_000
                            + variant_index * 1_000
                            + scenario_index * 100
                        ),
                    )
                    partial_path.write_text(
                        json.dumps(json_ready(result), indent=2) + "\n",
                        encoding="utf-8",
                    )
                traces.append(result)
                print(json.dumps(result["summary"], indent=2), flush=True)
            model_runs[variant] = aggregate_trace_runs(traces)
            print(
                f"AGGREGATE {model} {variant}\n"
                + json.dumps(model_runs[variant]["summary"], indent=2),
                flush=True,
            )
        runs[model] = model_runs

    oracle = aggregate_trace_runs(
        [
            run_trace(
                None,
                "stateful",
                gains,
                certificate,
                tube,
                incidents=scenario["incidents"],
                scenario_id=scenario["id"],
                initial_direction=scenario["initial_direction"],
            )
            for scenario in SCENARIOS
        ]
    )
    stateful_context = np.asarray(
        [runs[model]["stateful"]["summary"]["context_tool_request_accuracy"] for model in args.models]
    )
    memoryless_context = np.asarray(
        [runs[model]["memoryless"]["summary"]["context_tool_request_accuracy"] for model in args.models]
    )
    all_calls_valid = all(
        run["summary"]["valid_call_rate"] == 1.0
        for model_runs in runs.values()
        for run in model_runs.values()
    )
    all_safe = all(
        run["summary"]["maximum_tube_occupancy"] <= 1.0 + 1.0e-8
        and run["summary"]["minimum_state_constraint_margin_vehicles"] >= -1.0e-6
        and run["summary"]["minimum_input_constraint_margin"] >= -1.0e-9
        and run["summary"]["count_bound_verified"]
        for model_runs in runs.values()
        for run in model_runs.values()
    )
    material_context_gain = float(np.mean(stateful_context - memoryless_context)) >= 1.0 / 3.0
    stateful_context_correct = sum(
        runs[model]["stateful"]["summary"]["correct_context_requests"]
        for model in args.models
    )
    memoryless_context_correct = sum(
        runs[model]["memoryless"]["summary"]["correct_context_requests"]
        for model in args.models
    )
    context_requests_per_variant = sum(
        runs[model]["stateful"]["summary"]["context_request_count"]
        for model in args.models
    )
    stateful_true_proposals = sum(
        runs[model]["stateful"]["summary"]["true_oracle_proposals"]
        for model in args.models
    )
    memoryless_true_proposals = sum(
        runs[model]["memoryless"]["summary"]["true_oracle_proposals"]
        for model in args.models
    )
    proposals_per_variant = sum(
        runs[model]["stateful"]["summary"]["proposal_count"]
        for model in args.models
    )
    payload = {
        "experiment": "sequential language-to-rollout-tool MFD agent",
        "protocol": "MFD_SEQUENTIAL_LLM_PROTOCOL.md",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "models": args.models,
        "model_records": model_records,
        "temperature": TEMPERATURE,
        "base_seed": BASE_SEED,
        "prompt_version": PROMPT_VERSION,
        "decision_minutes": DECISION_MINUTES,
        "average_dwell": {
            "average_minutes": ADT_AVERAGE_MINUTES,
            "burst_capacity": ADT_BURST_CAPACITY,
        },
        "scenario_scripts": SCENARIOS,
        "runs": runs,
        "deterministic_oracle": oracle,
        "inclusion_audit": {
            "all_calls_valid": all_calls_valid,
            "all_executed_trajectories_safe": all_safe,
            "mean_stateful_context_accuracy": float(np.mean(stateful_context)),
            "mean_memoryless_context_accuracy": float(np.mean(memoryless_context)),
            "mean_context_accuracy_gain": float(np.mean(stateful_context - memoryless_context)),
            "total_live_calls": sum(
                run["summary"]["call_count"]
                for model_runs in runs.values()
                for run in model_runs.values()
            ),
            "context_requests_per_variant": context_requests_per_variant,
            "stateful_context_requests_correct": stateful_context_correct,
            "memoryless_context_requests_correct": memoryless_context_correct,
            "proposals_per_variant": proposals_per_variant,
            "stateful_true_objective_proposals": stateful_true_proposals,
            "memoryless_true_objective_proposals": memoryless_true_proposals,
            "material_context_gain_threshold": 1.0 / 3.0,
            "material_context_gain": material_context_gain,
            "eligible_for_main_paper": bool(all_calls_valid and all_safe and material_context_gain),
        },
    }
    AGGREGATE_JSON.write_text(
        json.dumps(json_ready(payload), indent=2) + "\n", encoding="utf-8"
    )
    make_figure(payload)
    print(json.dumps(json_ready(payload["inclusion_audit"]), indent=2))


if __name__ == "__main__":
    main()
