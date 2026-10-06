"""Independent v2 typed-intent, canonical-state and actual-feedback audit.

Zero live calls. The independent reference reducer is reused from the v1 audit,
not the production compiler. Raw typed validation and conversion are implemented
separately below. Physical proof validity has its own independent audit.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import random
import statistics

from audit_mfd_task_state_agent import (ReferenceTasks, gold_schedule, poisoned_view,
    read, sha, digest, encoding, relative, parse_direct, INITIAL_MEMORY, MODEL, MODEL_SHA)

ROOT=Path(__file__).resolve().parents[1]
LIVE=ROOT/"live_llm_results/task_state_primary_v2"
RUNTIME=ROOT/"reproducibility/results/task_state_runtime_v2"
LEGACY_LIVE=ROOT/"live_llm_results/task_state_primary_v1"
ASSIGN=("set","interrupt")
CONTEXT=("continue","resume","cancel","network")


def independent_intent(raw):
    value=json.loads(raw)
    if not isinstance(value,dict) or not isinstance(value.get("operation"),str): raise ValueError("typed object")
    op=value["operation"]
    if op in ASSIGN:
        if set(value)!={"operation","objective","minutes","label"}: raise ValueError("assignment fields")
        if value["objective"] not in ("region_1","region_2"): raise ValueError("region")
        if not isinstance(value["label"],str) or not value["label"].strip() or len(value["label"])>80:
            raise ValueError("label")
    elif op=="extend":
        if set(value)!={"operation","minutes"}: raise ValueError("extension fields")
    elif op in CONTEXT:
        if set(value)!={"operation"}: raise ValueError("context fields")
    else: raise ValueError("unknown operation")
    if op in ASSIGN or op=="extend":
        n=value["minutes"]
        if type(n) is not int or not 1<=n<=120: raise ValueError("positive integer minutes")
    canonical=dict(operation=op,objective=value["objective"] if op in ASSIGN else "unchanged",
        minutes=value["minutes"] if op in ASSIGN or op=="extend" else 0,
        label=value["label"] if op in ASSIGN else "")
    return value,canonical


def typed_gold(value):
    op=value["operation"]
    keys=("operation","objective","minutes","label") if op in ASSIGN else ("operation","minutes") if op=="extend" else ("operation",)
    return {k:value[k] for k in keys}


def reducer_tests():
    import mfd_task_state_agent_v2 as implementation
    rng=random.Random(202609112); checks=0
    for _ in range(30):
        reference=ReferenceTasks(); production=implementation.TaskCompiler(); now=0
        for step in range(40):
            now+=rng.randrange(0,8); op=rng.choice(ASSIGN+CONTEXT+("extend",))
            raw=dict(operation=op)
            if op in ASSIGN: raw.update(objective=rng.choice(("region_1","region_2")),minutes=rng.randrange(1,40),label=f"task{step}")
            elif op=="extend": raw["minutes"]=rng.randrange(1,40)
            validated,canonical=independent_intent(json.dumps(raw))
            assert implementation.validate_intent(raw)==validated
            assert implementation.canonical_intent(raw)==canonical
            reference.tick(now)
            try: expected=reference.reduce(canonical,"offline"); ref_fail=False
            except ValueError: ref_fail=True
            try: actual=production.apply(canonical,now,"offline"); prod_fail=False
            except ValueError: prod_fail=True
            assert ref_fail==prod_fail
            if not ref_fail: assert actual==expected
            for key,val in reference.fields().items(): assert production.state[key]==val
            checks+=1
    return dict(status="PASS",randomized_transitions=checks,seed=202609112,live_calls=0)


def audit(live_root=LIVE,allow_incomplete=False):
    frozen_path=live_root/"frozen_protocol.json"
    frozen=read(frozen_path); semantic=frozen["semantic"]; physical=frozen["physical"]
    assert semantic["version"]=="task-state-agent-v2-20260911"
    assert semantic["max_calls"]==40 and semantic["total_study_call_cap"]==48
    assert semantic["planned_development_calls"]==0 and semantic["preserved_version1_development_calls"]==8
    assert sha(ROOT/"reproducibility/mfd_task_state_agent_v2.py")==frozen["agent_source_sha256"]
    assert sha(ROOT/"TASK_STATE_AGENT_V2_PROTOCOL.md")==frozen["protocol_source_sha256"]
    for name,value in semantic["dependency_sha256"].items(): assert sha(ROOT/name)==value
    assert semantic["expected_model_digest"]==MODEL_SHA and physical["frozen_before_calls"]
    assert physical==read(live_root/"physical_protocol.json")
    hashes={relative(path):sha(path) for path in (frozen_path,live_root/"physical_protocol.json")}
    # Preserve the complete pre-change development audit and every payload it bound.
    prior_path=ROOT/"reproducibility/results/task_state_agent_development_partial_audit.json"
    prior=read(prior_path)
    assert prior["status"]=="PASS" and prior["attempted_calls"]==prior["response_records"]==prior["execution_records"]==8
    for name,value in prior["source_sha256"].items(): assert sha(ROOT/name)==value,("Development payload changed",name)
    assert len(list(LEGACY_LIVE.rglob("step_*_request.json")))==8
    hashes[relative(prior_path)]=sha(prior_path)
    costs=physical["objective_cost_rows"]
    gold=gold_schedule("heldout")
    assert [list(row["request"]) for row in gold]==semantic["scoring_requests"]["heldout"]
    assert [typed_gold(row["intent"]) for row in gold]==semantic["oracle_operations"]["heldout"]
    runs={}
    for path in sorted(RUNTIME.glob("*/result.json")):
        run=read(path); cfg=run["configuration"]
        if cfg["provider"] not in ("original","structured") or not cfg["certified"] or cfg["live_call_scope"]!="actual episode feedback": continue
        key=(cfg["split"],cfg["paraphrase"],"prose" if cfg["provider"]=="original" else "task_state")
        assert key not in runs,"Duplicate runtime for a frozen session"
        runs[key]=run; hashes[relative(path)]=sha(path)
    sessions=[]; rows=[]; attempted=responses=executions=0
    for directory in sorted(live_root.glob("*/*")):
        if not directory.is_dir() or not (directory/"session.json").is_file(): continue
        metadata=read(directory/"session.json")
        interface,phrase,attacked=metadata["interface"],metadata["paraphrase"],metadata["intervention"]
        assert metadata["split"]=="heldout" and interface in ("prose","task_state") and phrase in (0,1)
        assert metadata["model_record"]["name"]==MODEL and metadata["model_record"]["digest"]==MODEL_SHA
        assert metadata["agent_source_sha256"]==frozen["agent_source_sha256"]
        assert metadata["dependency_sha256"]==semantic["dependency_sha256"]
        assert metadata["protocol_sha256"]==sha(frozen_path) and metadata["endpoint"]=="http://127.0.0.1:11434"
        key=("poison" if attacked else "heldout",phrase,interface); run=runs.get(key)
        assert allow_incomplete or run is not None,("Missing runtime",key)
        if run is not None:
            cfg=run["configuration"]
            assert cfg["source_sha256"]["reproducibility/mfd_task_state_agent_v2.py"]==frozen["agent_source_sha256"]
            assert cfg["protocol_sha256"]==physical["physical_protocol_sha256"]
            for name,value in cfg["source_sha256"].items(): assert sha(ROOT/name)==value
            assert cfg["kernel_source_sha256"]==physical["scientific_kernel_sha256"]
        state=ReferenceTasks(); memory=INITIAL_MEMORY; previous_after=None; session_rows=[]
        for step,directive in enumerate(semantic["heldout"][phrase]):
            request_path=directory/f"step_{step:02}_request.json"
            response_path=directory/f"step_{step:02}_response.json"
            execution_path=directory/f"step_{step:02}_execution.json"
            if not request_path.exists():
                assert allow_incomplete,("Missing request",request_path)
                break
            attempted+=1; request=read(request_path); now=5*(step+1)
            index=40+step if attacked else 8+phrase*16+(0 if interface=="prose" else 8)+step
            assert request["call_index"]==index and request["step_index"]==step and request["time_minutes"]==now
            assert request["directive"]==directive and request["protocol_sha256"]==sha(frozen_path)
            state.tick(now)
            expected_memory=state.view() if interface=="task_state" else memory
            assert request["memory_before"]==expected_memory
            intervention=semantic["memory_view_intervention"]
            poisoned=attacked and step in intervention["step_indices"]
            view=poisoned_view(expected_memory,now,intervention) if poisoned else expected_memory
            assert request["memory_view"]==view and request["memory_view_intervened"]==poisoned
            prompt=request["prompt"]
            assert prompt.endswith("Current operator instruction:\n"+directive+"\nReturn only the requested JSON.")
            assert "scoring_only" not in prompt and "oracle_operations" not in prompt
            assert json.dumps(request["state_summary"],ensure_ascii=False) in prompt
            feedback={k:request["execution_feedback_full"][k] for k in semantic["feedback_allowlist"] if k in request["execution_feedback_full"]}
            assert json.dumps(feedback,ensure_ascii=False) in prompt
            assert (json.dumps(view,ensure_ascii=False) if interface=="task_state" else view) in prompt
            expected=gold[step]; expected_raw=typed_gold(expected["intent"])
            assert request["scoring_only_expected_intent"]==expected_raw
            assert (request["scoring_only_expected_request"]["objective"],request["scoring_only_expected_request"]["duration_minutes"])==expected["request"]
            if not response_path.exists():
                assert allow_incomplete,("Missing response",response_path)
                break
            responses+=1; response=read(response_path); body=response["request"]
            assert body["model"]==MODEL and body["prompt"]==prompt and body["stream"] is False and body["think"] is False
            assert body["options"]==dict(temperature=0.,seed=semantic["base_seed"]+index,num_predict=400)
            assert "context" not in body
            assert body["format"]==semantic["intent_schema" if interface=="task_state" else "prose_schema"]
            import hashlib
            assert response["prompt_sha256"]==hashlib.sha256(prompt.encode()).hexdigest()
            assert response["raw_response"]==response["response_payload"].get("response","")
            assert response["call_index"]==index and response["time_minutes"]==now and response["step_index"]==step
            assert dt.datetime.fromisoformat(request["created_at_utc"])<=dt.datetime.fromisoformat(response["created_at_utc"])
            if interface=="task_state":
                for field,value in state.fields().items(): assert response["compiler_state_before"][field]==value
            if previous_after is not None:
                history=previous_after["history"]
                assert response["compiler_state_before"]["history"][:len(history)]==history
            raw=None; canonical=None; pair=None; valid=False; backup=copy.deepcopy(state)
            if response["transport_error"] is None:
                try:
                    raw=json.loads(response["raw_response"])
                    if interface=="task_state":
                        raw,canonical=independent_intent(response["raw_response"])
                        pair=state.reduce(canonical,dict(call_index=index,request_sha256=sha(request_path)))
                    else:
                        raw=parse_direct(response["raw_response"]); pair=(raw["objective"],raw["duration_minutes"])
                        memory=raw["memory_update"]
                    valid=True
                except (ValueError,TypeError,KeyError): state=backup
            assert response["parsed"]==raw and response["canonical_compiler_intent"]==canonical
            assert response["valid"]==valid and (response["error"] is None)==valid
            if valid:
                assert response["response_payload"]["done"] is True
                obj,remaining=pair
                assert response["tool_request"]==dict(objective=obj,duration_minutes=remaining,priority_minutes=remaining,
                    cost_row=costs[obj],hard_contract_source="trusted external runtime")
            else: assert response["tool_request"] is None
            if interface=="task_state":
                for field,value in state.fields().items(): assert response["compiler_state_after"][field]==value
                if valid:
                    assert response["compiler_state_after"]["history"][-1]==dict(time_minutes=now,event="model_intent",intent=canonical,
                        provenance=dict(call_index=index,request_sha256=sha(request_path)),prior_state_sha256=digest(response["compiler_state_before"]))
                else: assert response["compiler_state_after"]==response["compiler_state_before"]
            else: assert response["prose_memory_after"]==memory
            previous_after=response["compiler_state_after"]
            exact=bool(valid and pair==expected["request"])
            op_correct=(bool(valid and all(raw.get(k)==v for k,v in expected_raw.items() if k!="label")) if interface=="task_state" else None)
            assert response["exact_request_correct"]==exact and response["operation_correct"]==op_correct
            assert response["contextual"]==expected["contextual"]
            actual_feedback=None; event=None
            if execution_path.exists():
                executions+=1; execution=read(execution_path); actual_feedback=execution["feedback"]
                assert dt.datetime.fromisoformat(response["created_at_utc"])<=dt.datetime.fromisoformat(execution["created_at_utc"])
                if run is not None:
                    event=run["decisions"][step+1]; preceding=run["decisions"][step]
                    assert actual_feedback==event["execution_feedback"] and request["execution_feedback_full"]==preceding["execution_feedback"]
                    assert actual_feedback["accepted"]==event["candidate_admitted"]
                    assert actual_feedback["executed_gate_command"]==event["executed_constant_command"]
                    assert request["state_summary"]==dict(region_1_vehicles=100*event["initial_state"][0],
                        region_2_vehicles=100*event["initial_state"][1],exterior_queue_vehicles=100*event["initial_state"][6])
                    assert event["request_valid"]==valid
                    if valid:
                        for field in ("objective","duration_minutes","cost_row"): assert event["tool_request"][field]==response["tool_request"][field]
                    else: assert event["tool_request"] is None
            else: assert allow_incomplete,("Missing execution",execution_path)
            row=dict(interface=interface,paraphrase=phrase,intervention=attacked,step=step,call_index=index,
                contextual=expected["contextual"],valid=valid,parsed_intent=raw,canonical_intent=canonical,
                expected_objective=expected["request"][0],expected_remaining_minutes=expected["request"][1],
                objective=pair[0] if valid else None,remaining_minutes=pair[1] if valid else None,
                objective_correct=bool(valid and pair[0]==expected["request"][0]),
                duration_correct=bool(valid and pair[1]==expected["request"][1]),
                exact_request_correct=exact,operation_correct=op_correct,latency_seconds=response["latency_seconds"],
                memory_view_poisoned=poisoned,candidate_admitted=actual_feedback.get("accepted") if actual_feedback else None,
                invalid_language_request=not valid,
                candidate_certificate_evaluated=None if event is None else event["certificate_reference"] is not None,
                physical_certificate_rejected=None if event is None else bool(valid and event["certificate_reference"] and not event["candidate_admitted"]),
                valid_request_without_certificate=None if event is None else bool(valid and event["certificate_reference"] is None),
                retained_original_suffix=None if event is None else event["execution_source"]=="retained_original_certified_suffix")
            rows.append(row); session_rows.append(row)
        sessions.append(dict(key=list(key),responses=len(session_rows),expected_responses=8,rows=session_rows,
            runtime_case=None if run is None else run["configuration"]["case_id"],runtime_metrics=None if run is None else run["metrics"]))
    if not allow_incomplete: assert len(sessions)==5 and attempted==responses==executions==40
    assert attempted<=40 and len({r["call_index"] for r in rows})==len(rows)
    assert len(list(live_root.rglob("step_*_request.json")))==attempted
    aggregates={}
    for label in ("heldout_prose","heldout_task_state","poison_task_state"):
        selected=[r for r in rows if r["interface"]==("prose" if label.endswith("prose") else "task_state") and r["intervention"]==label.startswith("poison")]
        if not selected: continue
        contextual=[r for r in selected if r["contextual"]]; explicit=[r for r in selected if not r["contextual"]]
        aggregates[label]=dict(total=len(selected),valid=sum(r["valid"] for r in selected),
            exact_request_correct=sum(r["exact_request_correct"] for r in selected),
            objective_correct=sum(r["objective_correct"] for r in selected),duration_correct=sum(r["duration_correct"] for r in selected),
            operation_correct=sum(bool(r["operation_correct"]) for r in selected) if label!="heldout_prose" else None,
            contextual_correct=sum(r["exact_request_correct"] for r in contextual),contextual_total=len(contextual),
            explicit_correct=sum(r["exact_request_correct"] for r in explicit),explicit_total=len(explicit),
            candidate_admitted=sum(r["candidate_admitted"] is True for r in selected),
            candidate_not_admitted=sum(r["candidate_admitted"] is False for r in selected),
            invalid_language_requests=sum(r["invalid_language_request"] for r in selected),
            runtime_bound_records=sum(r["candidate_certificate_evaluated"] is not None for r in selected),
            candidate_certificates_evaluated=sum(r["candidate_certificate_evaluated"] is True for r in selected),
            physical_certificate_rejections=sum(r["physical_certificate_rejected"] is True for r in selected),
            valid_requests_without_certificate=sum(r["valid_request_without_certificate"] is True for r in selected),
            retained_original_suffix_epochs=sum(r["retained_original_suffix"] is True for r in selected),
            latency_median_seconds=statistics.median(r["latency_seconds"] for r in selected),
            latency_range_seconds=[min(r["latency_seconds"] for r in selected),max(r["latency_seconds"] for r in selected)])
    for path in live_root.rglob("*.json"): hashes[relative(path)]=sha(path)
    return dict(status="PASS",complete=attempted==responses==executions==40,
        attempted_calls=attempted,response_records=responses,execution_records=executions,
        preserved_development_calls=8,total_observed_study_calls=8+attempted,new_live_calls=0,
        source_sha256=hashes,audit_source_sha256=sha(Path(__file__)),aggregate=aggregates,sessions=sessions,
        audit_dependency_sha256={"reproducibility/audit_mfd_task_state_agent.py":sha(ROOT/"reproducibility/audit_mfd_task_state_agent.py")},
        independent_reducer_tests=reducer_tests(),
        scope=["Independent typed parsing, canonical conversion, task-state recursion and exact gold schedule",
               "Raw requests/responses, source/model/protocol hashes, immutable v1 development and total48 accounting",
               "Actual physical observations and preceding/current execution feedback",
               "Poison changes model-visible memory only; subsequent parsed intents can change canonical tasks",
               "No model-immunity claim; physical certificate/replay validity is independently audited elsewhere"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root",type=Path,default=LIVE)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"reproducibility/results/task_state_agent_v2_independent_audit.json")
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(reducer_tests(),indent=2)); return
    if args.output.exists(): raise FileExistsError(args.output)
    result=audit(args.live_root,args.allow_incomplete)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as stream: stream.write(encoding(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ("sessions","source_sha256")},indent=2))


if __name__=="__main__": main()
