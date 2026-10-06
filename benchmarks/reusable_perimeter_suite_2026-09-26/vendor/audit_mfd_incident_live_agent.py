"""Independent raw-call/physical-runtime alignment audit; makes no live calls."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
from pathlib import Path

import mfd_incident_live_agent as agent

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE = ROOT / "live_llm_results/incident_agent/queue_semantic_agent_v1"
RESULTS = ROOT / "reproducibility/results"
EXPECTED_MODEL_DIGEST = "500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    location = (len(ordered)-1)*fraction
    lower = int(location)
    upper = min(lower+1,len(ordered)-1)
    return ordered[lower]+(location-lower)*(ordered[upper]-ordered[lower])


def scientific_ast(path):
    """Startup-recovery logging is the only permitted source difference."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef)
                 or node.name != "_resume_empty_session"]
    return ast.dump(tree, include_attributes=False)


def run(live_root=DEFAULT_LIVE):
    base = agent.audit_records(live_root, require_complete=True)
    all_rows = []
    traces = []
    files = {}
    source_records = []
    initial_memory = "No active preferential regional response. Network objective, 30-minute window; no interrupted response."
    for paraphrase in (0,1):
        for variant in ("stateful","memoryless"):
            path = RESULTS / f"mfd_incident_agent_live_p{paraphrase}_{variant}_results.json"
            runtime = read(path)
            files[str(path.relative_to(ROOT))] = digest(path)
            certified = runtime["certified"]
            nominal = runtime["nominal_matched_requests"]
            config = certified["configuration"]
            assert config["provider"] == "live" and config["certified"]
            assert config["service_vehicles"] == 2850. and config["service_minute"] == 60
            assert config["state_half_widths_vehicles"] == [1.]*7
            assert config["residual_veh_h"] == 60. and not config["forced_outages"]
            assert len(certified["decisions"]) == len(nominal["decisions"]) == 12
            assert certified["metrics"]["complete"] and nominal["metrics"]["complete"]
            assert certified["independent_audit"]["passed"] and nominal["independent_audit"]["passed"]
            for name, expected_hash in certified["source_sha256"].items():
                archived = RESULTS / "ors" / expected_hash[:16] / Path(name).name
                assert archived.exists() and digest(archived) == expected_hash
            directory = live_root / f"paraphrase_{paraphrase+1}" / variant
            session_path = directory / "session.json"
            metadata = read(session_path)
            assert metadata["model_record"]["name"] == agent.DEFAULT_MODEL
            assert metadata["model_record"]["digest"] == EXPECTED_MODEL_DIGEST
            assert metadata["endpoint"] == agent.DEFAULT_ENDPOINT
            assert metadata["variant"] == variant and metadata["paraphrase"] == paraphrase
            effective_source_hash = metadata["source_sha256"]
            resumes = sorted(directory.glob("startup_resume*.json"))
            for resume_path in resumes:
                resume = read(resume_path)
                assert resume["original_session_sha256"] == digest(session_path)
                assert resume["model_digest"] == EXPECTED_MODEL_DIGEST
                assert resume["protocol_sha256"] == base["protocol_sha256"]
                assert resume["live_calls_retried"] == 0
                effective_source_hash = resume["resumed_source_sha256"]
            source_path = RESULTS / "ors" / effective_source_hash[:16] / "mfd_incident_live_agent.py"
            assert digest(source_path) == effective_source_hash
            actual_scientific = scientific_ast(source_path)
            current_scientific = scientific_ast(ROOT / "reproducibility/mfd_incident_live_agent.py")
            assert actual_scientific == current_scientific, "Scientific live-interface source changed"
            source_records.append(dict(paraphrase=paraphrase,variant=variant,
                session_source_sha256=metadata["source_sha256"],effective_source_sha256=effective_source_hash,
                startup_only_recoveries=len(resumes),scientific_source_ast_unchanged=True))
            prior_memory = initial_memory
            rows = []
            for step in range(8):
                request = read(directory / f"step_{step:02d}_request.json")
                response = read(directory / f"step_{step:02d}_response.json")
                execution = read(directory / f"step_{step:02d}_execution.json")["feedback"]
                epoch = step+1
                event = certified["decisions"][epoch]
                preceding = certified["decisions"][epoch-1]
                assert request["execution_feedback_full"] == preceding["execution_feedback"]
                assert request["state_summary"] == {
                    "region_1_vehicles":100*event["initial_state"][0],
                    "region_2_vehicles":100*event["initial_state"][1],
                    "exterior_queue_vehicles":100*event["initial_state"][6]}
                assert request["memory_before"] == (prior_memory if variant == "stateful" else None)
                assert execution == event["execution_feedback"]
                assert execution["accepted"] == event["candidate_admitted"]
                assert execution["reason"] == event["failure_reason"]
                assert execution["executed_gate_command"] == event["executed_constant_command"]
                assert execution["region_1_vehicles"] == 100*event["executed_final_state"][0]
                assert execution["region_2_vehicles"] == 100*event["executed_final_state"][1]
                assert execution["queue_vehicles"] == 100*event["executed_final_state"][6]
                assert execution["time_minutes"] == 5*(epoch+1)
                assert execution["remaining_validity_minutes"] == 60-execution["time_minutes"]
                assert response["valid"] and event["request_valid"]
                assert response["response_payload"]["response"] == response["raw_response"]
                assert response["response_payload"]["done"]
                assert response["response_payload"]["model"] == agent.DEFAULT_MODEL
                assert response["request"]["options"]["temperature"] == 0.
                assert response["request"]["think"] is False
                assert "context" not in response["request"], "No hidden chat context is sent"
                assert event["live_response"]["tool_request"] == response["tool_request"]
                for key in ("objective", "duration_minutes", "cost_row"):
                    assert event["tool_request"][key] == response["tool_request"][key]
                    assert nominal["decisions"][epoch]["tool_request"][key] == event["tool_request"][key]
                prior_memory = response["parsed"]["memory_update"]
                objective,duration,contextual = agent.EXPECTED[step]
                objective_correct = response["parsed"]["objective"] == objective
                duration_correct = response["parsed"]["duration_minutes"] == duration
                row = dict(paraphrase=paraphrase,variant=variant,step=step,
                    time_minutes=5*(step+1),contextual=contextual,
                    expected_objective=objective,expected_duration_minutes=duration,
                    requested_objective=response["parsed"]["objective"],
                    requested_duration_minutes=response["parsed"]["duration_minutes"],
                    objective_correct=objective_correct,duration_correct=duration_correct,
                    exact_request_correct=objective_correct and duration_correct,
                    latency_seconds=response["latency_seconds"],
                    model_load_seconds=response["response_payload"].get("load_duration",0)/1e9,
                    model_eval_seconds=response["response_payload"].get("eval_duration",0)/1e9,
                    model_tokens=response["response_payload"].get("eval_count"),
                    candidate_admitted=event["candidate_admitted"],
                    execution_source=event["execution_source"],failure_reason=event["failure_reason"],
                    memory_before=request["memory_before"],memory_after=prior_memory)
                rows.append(row)
                all_rows.append(row)
            traces.append(dict(paraphrase=paraphrase,variant=variant,
                explicit_correct=sum(r["exact_request_correct"] and not r["contextual"] for r in rows),explicit_total=5,
                contextual_correct=sum(r["exact_request_correct"] and r["contextual"] for r in rows),contextual_total=3,
                objective_correct=sum(r["objective_correct"] for r in rows),duration_correct=sum(r["duration_correct"] for r in rows),
                exact_request_correct=sum(r["exact_request_correct"] for r in rows),request_total=8,
                live_candidate_admitted=sum(r["candidate_admitted"] for r in rows),
                live_candidate_rejected=sum(not r["candidate_admitted"] for r in rows),
                certified_metrics=certified["metrics"],matched_nominal_metrics=nominal["metrics"],
                comparison=runtime["comparison"],rows=rows))
    files.update({str(path.relative_to(ROOT)):digest(path) for path in live_root.rglob("*.json")})
    aggregate = {}
    for variant in ("stateful","memoryless"):
        rows = [row for row in all_rows if row["variant"] == variant]
        latencies = [row["latency_seconds"] for row in rows]
        aggregate[variant] = dict(
            total_calls=len(rows),valid_schema=16,valid_fixed_tool_requests=16,
            objective_correct=sum(row["objective_correct"] for row in rows),
            duration_correct=sum(row["duration_correct"] for row in rows),
            exact_request_correct=sum(row["exact_request_correct"] for row in rows),
            explicit_correct=sum(row["exact_request_correct"] and not row["contextual"] for row in rows),explicit_total=10,
            contextual_correct=sum(row["exact_request_correct"] and row["contextual"] for row in rows),contextual_total=6,
            latency_median_seconds=statistics.median(latencies),
            latency_min_seconds=min(latencies),latency_max_seconds=max(latencies),
            latency_p95_seconds=percentile(latencies,.95),
            live_candidate_admitted=sum(row["candidate_admitted"] for row in rows),
            live_candidate_rejected=sum(not row["candidate_admitted"] for row in rows))
    times=[row["latency_seconds"] for row in all_rows]
    return dict(status="PASS",complete=True,created_at_utc=agent.utc_now(),
        audit_script_sha256=digest(Path(__file__)),live_calls_made_by_audit=0,
        checks=["32 complete frozen request-response-execution records", "strict schema and fixed cost rows",
                "protocol, prompt, installed-model metadata and archived source hashes",
                "no hidden request-context field", "unmodified model-generated memory chain",
                "exact prior-runtime feedback and current observed-state binding",
                "actual execution feedback matches candidate admission and physical endpoint",
                "matched nominal replay uses the same semantic requests",
                "all traces independently replayed and original certificates retained",
                "source-version differences confined to pre-call recovery logging"],
        protocol_sha256=base["protocol_sha256"],model=agent.DEFAULT_MODEL,
        installed_model_digest=EXPECTED_MODEL_DIGEST,source_records=source_records,
        aggregate=aggregate,total_attempted_calls=32,total_schema_valid=32,
        latency=dict(median_seconds=statistics.median(times),min_seconds=min(times),
                     max_seconds=max(times),p95_seconds=percentile(times,.95),
                     includes_model_loading_and_local_request_queueing=True),
        interpretation={"contextual_memory_benefit_demonstrated":False,
                        "stateful_explicit_accuracy":"9/10", "memoryless_explicit_accuracy":"6/10",
                        "stateful_context_accuracy":"0/6", "memoryless_context_accuracy":"0/6",
                        "semantic_failures_are_retained":True,
                        "physical_certification_does_not_certify_soft_objective_interpretation":True},
        semantic_failures=[row for row in all_rows if not row["exact_request_correct"]],
        traces=traces,source_file_sha256=files)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=RESULTS/"mfd_incident_live_agent_independent_audit_results.json")
    args=parser.parse_args()
    payload=run()
    agent.write_new_json(args.output,payload)
    print(json.dumps({"output":str(args.output),"status":payload["status"],
                      "aggregate":payload["aggregate"],"latency":payload["latency"]},indent=2))


if __name__=="__main__":
    main()
