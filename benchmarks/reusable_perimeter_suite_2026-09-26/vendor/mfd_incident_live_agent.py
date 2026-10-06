"""Append-only live semantic-agent interface for the seven-state queue MFD.

This module does not simulate a plant or fabricate execution feedback.  Its
caller commissions a verified plan, calls ``QueueAgentSession.propose`` once
at each prescribed time, runs the returned MPC tool request, and supplies the
actual result through ``record_execution``.  See QUEUE_AGENT_PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
PROMPT_VERSION = "queue-semantic-agent-v1-20260910"
BASE_SEED = 20260910
MAX_CALLS = 32
DECISION_MINUTES = tuple(range(5, 41, 5))
OBJECTIVES = ("network", "region_1", "region_2")
DURATIONS = (5, 30, 60, 90)
COST_ROWS = {
    "network": [1., 1., 0., 0., 0., 0., 1.],
    "region_1": [4., 1., 0., 0., 0., 0., 1.],
    "region_2": [1., 4., 0., 0., 0., 0., 1.],
}
TRUSTED_CONTRACT = {
    "service_region": 1,
    "service_deadline_minutes": 60,
    "service_maximum_vehicles": 2850,
    "queue_storage_vehicles": 2000,
    "verification_horizon": "all pending service deadlines and complete submitted plan",
    "execution_authority": "independent runtime verifier; retain original-expiry certified suffix on rejection",
}
EXPECTED = (
    ("region_1", 60, False), ("region_1", 60, True),
    ("network", 30, False), ("region_2", 90, False),
    ("region_2", 90, True), ("region_1", 5, False),
    ("region_2", 90, True), ("network", 30, False),
)
PARAPHRASES = (
    (
        "A stadium release is loading Region 1. Prioritize clearing Region 1 over a 60-minute objective window.",
        "The stadium operation is still active. Continue its priority and keep its current objective window.",
        "The stadium response is complete. Restore network-wide operation with a 30-minute objective window.",
        "A freight-terminal surge now requires priority for Region 2. Use a 90-minute objective window.",
        "Continue the active freight response and preserve its objective window.",
        "For the next 5-minute interval, temporarily prioritize Region 1 for emergency access. Remember the preceding freight response so we can return to it.",
        "Emergency access is complete. Return to the response that was active immediately before that temporary override, including its objective window.",
        "All regional-priority requests have ended. Restore network-wide operation with a 30-minute objective window.",
    ),
    (
        "Give Region 1 clearance priority because spectators are leaving the venue. Evaluate this priority across the next 60 minutes.",
        "Keep that venue-clearance response in force with the same regional priority and evaluation window.",
        "Venue clearance no longer needs preferential treatment. Resume the network-wide objective over 30 minutes.",
        "Give Region 2 precedence while a freight release is handled, using a 90-minute objective window.",
        "Maintain that freight-release response; its priority and evaluation window remain unchanged.",
        "Interrupt the current response for one 5-minute emergency-access interval in Region 1. Save the interrupted response for subsequent restoration.",
        "The emergency interruption has finished. Restore the saved response, with the same priority and objective window it had before the interruption.",
        "No preferential regional response is needed now. Resume network-wide operation over a 30-minute objective window.",
    ),
)
SCHEMA = {
    "type": "object",
    "properties": {
        "objective": {"type": "string", "enum": list(OBJECTIVES)},
        "duration_minutes": {"type": "integer", "enum": list(DURATIONS)},
        "memory_update": {"type": "string", "maxLength": 500},
    },
    "required": ["objective", "duration_minutes", "memory_update"],
    "additionalProperties": False,
}
# Only these objective-free fields can enter the next model prompt.  The full
# runtime record is retained separately, including optimizer/certificate data.
FEEDBACK_KEYS = (
    "accepted", "status", "reason", "time_minutes", "remaining_validity_minutes",
    "retained_previous_plan", "region_1_vehicles", "region_2_vehicles",
    "queue_vehicles", "cumulative_tts_vehicle_hours", "minimum_physical_margin",
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    """Use exclusive creation: a repeated run never replaces live evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(json_bytes(value))


def protocol_record() -> dict[str, Any]:
    return {
        "version": PROMPT_VERSION, "model": DEFAULT_MODEL, "temperature": 0.0,
        "base_seed": BASE_SEED, "max_live_calls": MAX_CALLS,
        "decision_minutes": list(DECISION_MINUTES),
        "initial_commissioning": "independent certified network MPC at minute 0",
        "continuation": "no additional live calls after minute 40; reuse latest tool request for 5-minute renewals through minute 60",
        "variants": ["stateful", "memoryless"],
        "paraphrases": [list(trace) for trace in PARAPHRASES],
        "expected_requests": [dict(objective=o, duration_minutes=d, contextual=c) for o, d, c in EXPECTED],
        "schema": SCHEMA, "cost_rows": COST_ROWS, "trusted_contract": TRUSTED_CONTRACT,
        "priority_window": "requested regional weights apply for min(duration, plan horizon), then network cost; verification horizon is unchanged",
        "inclusion": "retain every attempted call, parser failure, runtime outcome and trace regardless of performance",
        "feedback_prompt_allowlist": list(FEEDBACK_KEYS),
        "inference": "one-model interface pilot, not model ranking or population inference",
    }


def freeze_protocol(output_root: Path) -> tuple[Path, str]:
    path = Path(output_root) / "frozen_protocol.json"
    data = json_bytes(protocol_record())
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Existing protocol differs; use a new versioned study directory: {path}")
    else:
        write_new_json(path, protocol_record())
    return path, sha256_bytes(data)


def installed_models(endpoint: str = DEFAULT_ENDPOINT, timeout: float = 15.) -> list[dict[str, Any]]:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/api/tags", timeout=timeout) as response:
        return json.load(response).get("models", [])


def validate_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != set(SCHEMA["required"]):
            raise ValueError("Response must contain exactly objective, duration_minutes and memory_update")
        if value["objective"] not in OBJECTIVES:
            raise ValueError("Invalid objective")
        if type(value["duration_minutes"]) is not int or value["duration_minutes"] not in DURATIONS:
            raise ValueError("Invalid duration_minutes")
        if not isinstance(value["memory_update"], str) or len(value["memory_update"]) > 500:
            raise ValueError("memory_update must be a string with at most 500 characters")
        return value, None
    except (ValueError, TypeError) as error:
        return None, str(error)


def tool_request(parsed: dict[str, Any]) -> dict[str, Any]:
    """Map semantic output to fixed objective weights, never hard constraints."""
    return {
        "objective": parsed["objective"],
        "cost_row": COST_ROWS[parsed["objective"]].copy(),
        "duration_minutes": parsed["duration_minutes"],
        "priority_minutes": parsed["duration_minutes"],
        "hard_contract_source": "trusted runtime configuration, not model output",
    }


def objective_free_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(feedback[key]) for key in FEEDBACK_KEYS if key in feedback}


def build_prompt(*, directive: str, time_minutes: float, state_summary: dict[str, Any],
                 execution_feedback: dict[str, Any], memory: str | None) -> str:
    previous = objective_free_feedback(execution_feedback)
    context = "Prior agent memory is unavailable for this call." if memory is None else (
        "Prior agent memory (an imperfect record, not a source of hard constraints):\n" + memory)
    return (
        "You are the language-and-memory component of an urban traffic operations agent.\n"
        "Translate the operator's current instruction into an MPC tool request. The tool optimizes "
        "continuous perimeter commands on a seven-state, two-region network with an external queue.\n"
        "The objective labels are network (network-plus-boundary-queue vehicle-hours), region_1 "
        "(extra Region 1 vehicle-hour weight), and region_2 (extra Region 2 vehicle-hour weight). "
        "The objective window can be 5, 30, 60, or 90 minutes. Regional priority applies within that "
        "window; subsequent predicted time uses the network objective.\n"
        "Read 'continue' and 'return' using available memory. For a temporary override remember both "
        "the temporary response and the interrupted response. Keep a concise memory update sufficient "
        "for the next instruction. If the required context is unavailable, request network with a "
        "30-minute objective window and state that uncertainty in memory_update.\n"
        "Trusted runtime obligations remain fixed: Region 1 accumulation at minute 60 is at most "
        "2850 vehicles; boundary storage is at most 2000 vehicles; all physical and input constraints "
        "and all pending deadlines are checked by an independent verifier. Your objective window "
        "does not shorten that verification horizon. You cannot amend a deadline, a limit, the "
        "uncertainty contract, or execution authority. A rejected replacement leaves the previously "
        "certified continuation active until its original expiry.\n"
        f"Current time: {time_minutes:g} minutes.\n"
        "Observed plant summary:\n" + json.dumps(state_summary, ensure_ascii=False, allow_nan=False) + "\n"
        "Actual previous execution feedback (objective-independent fields):\n"
        + json.dumps(previous, ensure_ascii=False, allow_nan=False) + "\n" + context + "\n"
        "Current operator instruction:\n" + directive + "\n"
        "Return only JSON with exactly objective, duration_minutes, memory_update.\n"
    )


def call_model(prompt: str, *, model: str, endpoint: str, call_index: int,
               timeout: float = 120.) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": False, "format": SCHEMA,
        "options": {"temperature": 0.0, "seed": BASE_SEED + call_index, "num_predict": 400},
        "keep_alive": "15m",
    }
    if model.startswith("qwen3"):
        body["think"] = False
    request = urllib.request.Request(endpoint.rstrip("/") + "/api/generate",
        data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    payload: dict[str, Any] = {}
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exception:
        error = f"{type(exception).__name__}: {exception}"
    elapsed = time.perf_counter() - started
    raw = payload.get("response", "")
    parsed, validation_error = validate_response(raw)
    if error is None:
        error = validation_error
    return {
        "created_at_utc": utc_now(), "call_index": call_index, "request": body,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")), "raw_response": raw,
        "response_payload": payload, "parsed": parsed, "valid": error is None,
        "error": error, "latency_seconds": elapsed,
    }


class QueueAgentSession:
    """One eight-call trace, coupled by the caller to actual runtime execution."""
    def __init__(self, output_root: str | Path, *, variant: str, paraphrase: int,
                 model: str = DEFAULT_MODEL, endpoint: str = DEFAULT_ENDPOINT,
                 timeout: float = 120., resume_empty: bool = False,
                 resume_reason: str = "Explicit empty-session recovery: previous driver startup failed before any step request was saved.") -> None:
        if variant not in ("stateful", "memoryless") or paraphrase not in (0, 1):
            raise ValueError("variant must be stateful/memoryless and paraphrase must be 0/1")
        self.root = Path(output_root)
        self.protocol_path, self.protocol_sha256 = freeze_protocol(self.root)
        self.variant, self.paraphrase = variant, paraphrase
        self.model, self.endpoint, self.timeout = model, endpoint, timeout
        self.directory = self.root / f"paraphrase_{paraphrase + 1}" / variant
        self.memory = "No active preferential regional response. Network objective, 30-minute window; no interrupted response."
        self.next_step = 0
        self.awaiting_execution: int | None = None
        self.last_execution: dict[str, Any] | None = None
        records = installed_models(endpoint, min(timeout, 15.))
        model_record = next((item for item in records if item.get("name") == model), None)
        if model_record is None:
            raise RuntimeError(f"Requested model {model!r} is not already installed; no download was attempted")
        if model != DEFAULT_MODEL:
            raise ValueError("This frozen pilot fixes qwen3:8b; another model requires a new protocol version")
        metadata = {
            "created_at_utc": utc_now(), "protocol_sha256": self.protocol_sha256,
            "model_record": model_record, "variant": variant, "paraphrase": paraphrase,
            "endpoint": endpoint, "source_sha256": sha256_bytes(Path(__file__).read_bytes()),
        }
        if resume_empty and (self.directory / "session.json").exists():
            _resume_empty_session(self.directory, metadata, resume_reason)
        else:
            write_new_json(self.directory / "session.json", metadata)

    def propose(self, step_index: int, time_minutes: float, state_summary: dict[str, Any],
                execution_feedback: dict[str, Any] | None = None,
                *, memory_override: str | None = None) -> dict[str, Any]:
        if self.awaiting_execution is not None:
            raise RuntimeError("Record actual execution before the next live call")
        if step_index != self.next_step or not 0 <= step_index < 8:
            raise ValueError("Each trace requires exactly sequential step indices 0..7")
        if not math.isclose(time_minutes, DECISION_MINUTES[step_index], abs_tol=1e-9):
            raise ValueError("Live calls must follow the frozen 5..40 minute decision clock")
        feedback = execution_feedback if execution_feedback is not None else self.last_execution
        if feedback is None:
            raise ValueError("The first live call requires actual initial-commissioning feedback")
        if memory_override is not None:
            raise ValueError("The frozen32-call pilot excludes memory corruption; use a separately versioned protocol for an attack case")
        memory_before = self.memory if self.variant == "stateful" else None
        directive = PARAPHRASES[self.paraphrase][step_index]
        prompt = build_prompt(directive=directive, time_minutes=time_minutes,
            state_summary=state_summary, execution_feedback=feedback, memory=memory_before)
        call_index = self.paraphrase * 16 + (0 if self.variant == "stateful" else 8) + step_index
        if not 0 <= call_index < MAX_CALLS:
            raise RuntimeError("Frozen pilot call budget exceeded")
        # Save the intended request before sending any live HTTP request.
        request_path = self.directory / f"step_{step_index:02d}_request.json"
        expected_objective, expected_duration, contextual = EXPECTED[step_index]
        write_new_json(request_path, {
            "created_at_utc": utc_now(), "step_index": step_index, "call_index": call_index,
            "time_minutes": time_minutes, "operator_directive": directive,
            "expected_request_not_shown_to_model": {"objective": expected_objective, "duration_minutes": expected_duration},
            "contextual": contextual, "state_summary": state_summary,
            "execution_feedback_full": feedback, "execution_feedback_prompt": objective_free_feedback(feedback),
            "memory_before": memory_before, "prompt": prompt, "protocol_sha256": self.protocol_sha256,
        })
        record = call_model(prompt, model=self.model, endpoint=self.endpoint,
                            call_index=call_index, timeout=self.timeout)
        request = tool_request(record["parsed"]) if record["valid"] else None
        record.update({
            "step_index": step_index, "time_minutes": time_minutes,
            "tool_request": request, "contextual": contextual,
            "objective_correct": bool(request and request["objective"] == expected_objective),
            "duration_correct": bool(request and request["duration_minutes"] == expected_duration),
            "request_correct": bool(request and request["objective"] == expected_objective and request["duration_minutes"] == expected_duration),
        })
        response_path = self.directory / f"step_{step_index:02d}_response.json"
        write_new_json(response_path, record)
        if record["valid"] and self.variant == "stateful":
            self.memory = record["parsed"]["memory_update"]
        self.next_step += 1
        self.awaiting_execution = step_index
        return {"valid": record["valid"], "tool_request": request,
                "record_path": str(response_path), "request_path": str(request_path),
                "request_correct": record["request_correct"], "error": record["error"],
                "latency_seconds": record["latency_seconds"]}

    def record_execution(self, step_index: int, feedback: dict[str, Any]) -> Path:
        """Persist the runtime's actual outcome; no controller is run here."""
        if self.awaiting_execution != step_index:
            raise ValueError("Feedback must match the most recent uncompleted live decision")
        if not isinstance(feedback, dict) or not feedback:
            raise ValueError("A nonempty actual runtime feedback record is required")
        path = self.directory / f"step_{step_index:02d}_execution.json"
        write_new_json(path, {"recorded_at_utc": utc_now(), "step_index": step_index, "feedback": feedback})
        self.last_execution = copy.deepcopy(feedback)
        self.awaiting_execution = None
        return path


def _resume_empty_session(directory: Path, expected: dict[str, Any], reason: str) -> None:
    """Recover pre-call startup only, keeping original metadata unmodified."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Empty-session recovery requires a recorded startup failure reason")
    step_paths = list(directory.glob("step_*"))
    if step_paths:
        raise RuntimeError("Cannot resume: a step request/response/execution already exists; no live-call retries are permitted")
    path = directory / "session.json"
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    for key in ("variant", "paraphrase", "protocol_sha256", "endpoint"):
        if original.get(key) != expected.get(key):
            raise RuntimeError(f"Empty-session recovery metadata mismatch: {key}")
    for key in ("name", "digest"):
        prior_value = original.get("model_record", {}).get(key)
        current_value = expected.get("model_record", {}).get(key)
        if prior_value is None or prior_value != current_value:
            raise RuntimeError(f"Empty-session recovery installed model mismatch: {key}")
    resume_path = directory / "startup_resume.json"
    suffix = 2
    while resume_path.exists():
        resume_path = directory / f"startup_resume_{suffix:02d}.json"
        suffix += 1
    write_new_json(resume_path, {
        "created_at_utc": utc_now(), "kind": "pre_call_empty_session_startup_recovery",
        "prior_failure_reason": reason,
        "precondition": "No step_* request, response, or execution file exists",
        "original_session_sha256": sha256_bytes(original_bytes),
        "original_source_sha256": original.get("source_sha256"),
        "resumed_source_sha256": expected.get("source_sha256"),
        "protocol_sha256": expected["protocol_sha256"],
        "model_digest": expected["model_record"]["digest"],
        "live_calls_retried": 0,
    })


def self_test() -> dict[str, Any]:
    for objective in OBJECTIVES:
        raw = json.dumps(dict(objective=objective, duration_minutes=30, memory_update="test"))
        value, error = validate_response(raw)
        assert error is None and tool_request(value)["cost_row"] == COST_ROWS[objective]
    invalid = (
        '{"objective":"network","duration_minutes":true,"memory_update":"x"}',
        '{"objective":"network","duration_minutes":30,"memory_update":"x","service_maximum_vehicles":5000}',
        '{"objective":"unconstrained","duration_minutes":30,"memory_update":"x"}',
    )
    assert all(validate_response(raw)[1] is not None for raw in invalid)
    feedback = {"accepted": True, "queue_vehicles": 1., "objective": "SECRET_PRIORITY", "duration_minutes": 90}
    prompt = build_prompt(directive="continue", time_minutes=10, state_summary={"queue_vehicles": 1.}, execution_feedback=feedback, memory=None)
    assert "SECRET_PRIORITY" not in prompt
    assert "2850" in prompt and "does not shorten" in prompt
    assert len(PARAPHRASES) * len(EXPECTED) * 2 == MAX_CALLS
    assert sum(item[2] for item in EXPECTED) == 3
    assert COST_ROWS["region_1"][0] == 4. and COST_ROWS["region_2"][1] == 4.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="queue_agent_resume_test_") as temporary:
        directory = Path(temporary)
        metadata = dict(variant="stateful", paraphrase=0, protocol_sha256="test-protocol",
                        endpoint=DEFAULT_ENDPOINT, model_record=dict(name=DEFAULT_MODEL,digest="test-digest"),
                        source_sha256="initial-source")
        write_new_json(directory / "session.json", metadata)
        original = (directory / "session.json").read_bytes()
        _resume_empty_session(directory, metadata, "Synthetic pre-call startup failure in self-test")
        assert (directory / "session.json").read_bytes() == original
        assert json.loads((directory / "startup_resume.json").read_bytes())["live_calls_retried"] == 0
        write_new_json(directory / "step_00_request.json", {"fixture": "saved request prevents resume"})
        try:
            _resume_empty_session(directory, metadata, "Test refusal after a request exists")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Empty-session recovery must refuse any existing step record")
    return {"status": "PASS", "tests": ["strict schema", "fixed cost rows", "hard-contract extra-key rejection", "memoryless no objective leak", "32-call cap", "three contextual decisions per trace", "empty-session recovery preserves metadata", "saved request forbids recovery"], "live_calls": 0}


def audit_records(output_root: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Recount raw calls and check their frozen protocol and feedback bindings."""
    root = Path(output_root)
    frozen = root / "frozen_protocol.json"
    assert frozen.read_bytes() == json_bytes(protocol_record()), "Protocol drift"
    digest = sha256_bytes(frozen.read_bytes())
    summaries = []
    total_attempts = total_responses = total_executions = 0
    for paraphrase in (0, 1):
        for variant in ("stateful", "memoryless"):
            directory = root / f"paraphrase_{paraphrase + 1}" / variant
            if not (directory / "session.json").exists():
                assert not require_complete, f"Missing session {directory}"
                continue
            session = json.loads((directory / "session.json").read_text(encoding="utf-8"))
            assert session["protocol_sha256"] == digest
            rows = []
            for step in range(8):
                request_path = directory / f"step_{step:02d}_request.json"
                response_path = directory / f"step_{step:02d}_response.json"
                execution_path = directory / f"step_{step:02d}_execution.json"
                if not request_path.exists():
                    assert not require_complete, f"Missing request {request_path}"
                    continue
                total_attempts += 1
                request = json.loads(request_path.read_text(encoding="utf-8"))
                assert request["protocol_sha256"] == digest
                assert request["operator_directive"] == PARAPHRASES[paraphrase][step]
                assert request["time_minutes"] == DECISION_MINUTES[step]
                assert request["execution_feedback_prompt"] == objective_free_feedback(request["execution_feedback_full"])
                expected_prompt = build_prompt(directive=request["operator_directive"],
                    time_minutes=request["time_minutes"], state_summary=request["state_summary"],
                    execution_feedback=request["execution_feedback_full"], memory=request["memory_before"])
                assert request["prompt"] == expected_prompt
                if variant == "memoryless":
                    assert request["memory_before"] is None
                if not response_path.exists():
                    assert not require_complete, f"Missing response {response_path}"
                    continue
                total_responses += 1
                response = json.loads(response_path.read_text(encoding="utf-8"))
                index = paraphrase * 16 + (0 if variant == "stateful" else 8) + step
                assert response["call_index"] == request["call_index"] == index
                assert response["request"]["options"]["seed"] == BASE_SEED + index
                assert response["request"]["prompt"] == request["prompt"]
                assert response["request"]["model"] == DEFAULT_MODEL
                assert response["prompt_sha256"] == sha256_bytes(request["prompt"].encode("utf-8"))
                parsed, validation_error = validate_response(response["raw_response"])
                expected_objective, expected_duration, contextual = EXPECTED[step]
                expected_tool = tool_request(parsed) if response["valid"] else None
                assert response["tool_request"] == expected_tool
                if response["valid"]:
                    assert validation_error is None and parsed == response["parsed"]
                correct = bool(expected_tool and expected_tool["objective"] == expected_objective
                               and expected_tool["duration_minutes"] == expected_duration)
                assert response["request_correct"] == correct
                assert response["contextual"] == contextual
                execution = None
                if execution_path.exists():
                    total_executions += 1
                    execution = json.loads(execution_path.read_text(encoding="utf-8"))["feedback"]
                else:
                    assert not require_complete, f"Missing execution {execution_path}"
                rows.append(dict(step=step, valid=response["valid"], correct=correct,
                    contextual=contextual, objective_correct=response["objective_correct"],
                    duration_correct=response["duration_correct"],
                    requested_objective=expected_tool["objective"] if expected_tool else None,
                    expected_objective=expected_objective, latency_seconds=response["latency_seconds"],
                    accepted=execution.get("accepted") if execution else None))
            summaries.append(dict(paraphrase=paraphrase, variant=variant,
                response_count=len(rows), valid=sum(row["valid"] for row in rows),
                correct=sum(row["correct"] for row in rows),
                contextual_count=sum(row["contextual"] for row in rows),
                contextual_correct=sum(row["correct"] and row["contextual"] for row in rows),
                explicit_count=sum(not row["contextual"] for row in rows),
                explicit_correct=sum(row["correct"] and not row["contextual"] for row in rows),
                rows=rows))
    assert total_attempts <= MAX_CALLS
    if require_complete:
        assert total_attempts == total_responses == total_executions == MAX_CALLS
    return dict(status="PASS", complete=total_responses == MAX_CALLS and total_executions == MAX_CALLS,
        protocol_sha256=digest, attempted_calls=total_attempts, response_records=total_responses,
        execution_records=total_executions, sessions=summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--freeze-protocol", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        print(json.dumps(installed_models(), indent=2))
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
    if args.freeze_protocol:
        path, digest = freeze_protocol(args.freeze_protocol)
        print(json.dumps({"path": str(path), "sha256": digest, "live_calls": 0}, indent=2))
    if args.audit:
        print(json.dumps(audit_records(args.audit, require_complete=not args.allow_incomplete), indent=2))
    if not (args.list_models or args.self_test or args.freeze_protocol or args.audit):
        parser.print_help()


if __name__ == "__main__":
    main()
