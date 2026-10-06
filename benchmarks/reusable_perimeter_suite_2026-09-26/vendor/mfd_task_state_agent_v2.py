"""Versioned operation-specific task interface after eight preserved dev calls.

No development calls or retries are supported here. The unchanged v1 compiler
is a source-hashed dependency; only valid model-generated intents enter it.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import mfd_task_state_agent as legacy

ROOT=Path(__file__).resolve().parents[1]
VERSION="task-state-agent-v2-20260911"
MODEL,MODEL_DIGEST,ENDPOINT=legacy.MODEL,legacy.MODEL_DIGEST,legacy.ENDPOINT
BASE_SEED=legacy.BASE_SEED
MAX_CALLS=40
TOTAL_STUDY_CALL_CAP=48
LEGACY_SOURCE_SHA="2da14337e94c2524fdc1d5d006fedb7e20c5f82faf4d1d60c5d9af089690b96b"
LEGACY_PROTOCOL_SHA="0b61e6c4068e5a74e19c6557f8595d7cd75b96e81a4c7e1acedccdb4fce5ee21"
COST_ROWS=copy.deepcopy(legacy.COST_ROWS)
FEEDBACK_KEYS=legacy.FEEDBACK_KEYS
OPERATIONS=legacy.OPERATIONS
PROSE_SCHEMA=copy.deepcopy(legacy.PROSE_SCHEMA)
MEMORY_INTERVENTION=copy.deepcopy(legacy.MEMORY_INTERVENTION)
EXPECTED=copy.deepcopy(legacy.EXPECTED)
# The generation schema uses only elementary object/property constraints. The
# exact discriminated union is enforced locally by validate_intent, not assumed
# from decoder support for conditional JSON Schema keywords.
INTENT_SCHEMA={"type":"object","properties":{
    "operation":{"type":"string","enum":list(OPERATIONS)},
    "objective":{"type":"string","enum":["region_1","region_2"]},
    "minutes":{"type":"integer","minimum":1,"maximum":120},
    "label":{"type":"string","maxLength":80}},
    "required":["operation"],"additionalProperties":False}
OPERATION_FIELDS={op:(["operation","objective","minutes","label"] if op in ("set","interrupt")
                      else ["operation","minutes"] if op=="extend" else ["operation"])
                  for op in OPERATIONS}
HELDOUT=(
    (
      "Activate a Region 1 clearance priority for a 25-minute passenger-dispersal response beginning now.",
      "Keep the passenger-dispersal response in effect. Its previously assigned expiry is unchanged.",
      "Pause that response and give Region 2 a temporary 15-minute access priority. Retain the paused response's original expiry so it can be restored.",
      "Move the expiry of the active temporary access response ten minutes later. Leave the paused response's expiry untouched.",
      "The temporary access requirement has ended. Restore the paused passenger-dispersal response until its original expiry, if that expiry has not passed.",
      "Keep the last response only for time still authorized on its existing clock. If that clock has expired, use the network objective; grant no fresh time.",
      "Start a new 20-minute Region 2 priority to clear outbound distribution traffic, effective immediately.",
      "Withdraw the outbound-distribution priority now and clear regional task priorities. Use network-wide operation.",
    ),
    (
      "Region 1 is to receive clearance priority for the twenty-five minutes starting at this decision, while event passengers disperse.",
      "Maintain this event-passenger response without resetting its ending time.",
      "Set the event-passenger response aside temporarily. Prioritize Region 2 access for fifteen minutes, remembering the displaced response and when it was originally due to end.",
      "Authorize ten additional minutes beyond the current ending time of this temporary access task. Do not extend the displaced task.",
      "Finish the temporary access task early and bring back the displaced event-passenger task for its remaining originally authorized time.",
      "Do not renew the preceding task. Continue it if it is still within its existing authorization; otherwise return to ordinary network operation.",
      "A fresh authorization begins now: Region 2 clearance takes priority for twenty minutes during dispatch departures.",
      "Revoke the dispatch-departure authorization and remove all regional priorities, returning to the network objective.",
    ),
)

utc_now,encoded,digest,file_sha,write_new=(legacy.utc_now,legacy.encoded,legacy.digest,legacy.file_sha,legacy.write_new)
installed_model,validate_prose,call_local=(legacy.installed_model,legacy.validate_prose,legacy.call_local)
TaskCompiler=legacy.TaskCompiler


def validate_intent(value):
    if not isinstance(value,dict) or not isinstance(value.get("operation"),str):
        raise ValueError("An operation-specific JSON object is required")
    op=value["operation"]
    if op not in OPERATION_FIELDS or set(value)!=set(OPERATION_FIELDS[op]):
        raise ValueError("Fields do not match the selected operation")
    if op in ("set","interrupt","extend"):
        if type(value["minutes"]) is not int or not 1<=value["minutes"]<=120:
            raise ValueError("Positive integer duration or extension required")
    if op in ("set","interrupt"):
        if value["objective"] not in ("region_1","region_2"):
            raise ValueError("A regional assignment requires a named region")
        if not isinstance(value["label"],str) or not value["label"].strip() or len(value["label"])>80:
            raise ValueError("A concise nonempty task label is required")
    return copy.deepcopy(value)


def canonical_intent(value):
    raw=validate_intent(value)
    return dict(operation=raw["operation"],objective=raw.get("objective","unchanged"),
                minutes=raw.get("minutes",0),label=raw.get("label",""))


def typed_intent(value):
    return {k:copy.deepcopy(value[k]) for k in OPERATION_FIELDS[value["operation"]]}


ORACLE={"heldout":[typed_intent(value) for value in legacy.ORACLE["heldout"]]}


def trace_length(split):
    return 8 if split in ("heldout","poison") else 0


def oracle_request(split,step,time_minutes=None):
    if split not in ("heldout","poison"): raise ValueError("Version 2 has no development calls")
    return legacy.oracle_request(split,step,time_minutes)


def dependency_hashes():
    paths={"reproducibility/mfd_task_state_agent.py":LEGACY_SOURCE_SHA,
           "TASK_STATE_AGENT_PROTOCOL.md":LEGACY_PROTOCOL_SHA}
    for path,expected in paths.items():
        if file_sha(ROOT/path)!=expected: raise RuntimeError("Frozen v1 dependency changed")
    return paths


def semantic_protocol():
    return dict(version=VERSION,model=MODEL,expected_model_digest=MODEL_DIGEST,
        temperature=0.,think=False,base_seed=BASE_SEED,max_calls=MAX_CALLS,
        total_study_call_cap=TOTAL_STUDY_CALL_CAP,preserved_version1_development_calls=8,
        planned_development_calls=0,planned_heldout_calls=32,optional_corruption_calls=8,
        interfaces=["prose","task_state"],heldout=[list(t) for t in HELDOUT],
        decision_minutes=dict(heldout=list(range(5,41,5))),oracle_operations=ORACLE,
        scoring_requests={"heldout":EXPECTED["heldout"]},intent_schema=INTENT_SCHEMA,
        operation_field_sets=OPERATION_FIELDS,prose_schema=PROSE_SCHEMA,
        feedback_allowlist=list(FEEDBACK_KEYS),physical_runtime="separate frozen v2 driver; actual feedback required",
        duration_semantics="absolute expiry; continue/resume do not renew; extension explicit",
        memory_view_intervention=MEMORY_INTERVENTION,dependency_sha256=dependency_hashes(),
        development_change="Remove unused contextual arguments; exact typed fields; preserve all 8 failed/successful v1 development calls",
        no_gold_in_prompt=True,no_automatic_retries=True)


def freeze(output_root,physical_protocol):
    output_root=Path(output_root)
    if not isinstance(physical_protocol,dict) or not physical_protocol.get("frozen_before_calls"):
        raise ValueError("A pre-call frozen physical study protocol is required")
    record=dict(semantic=semantic_protocol(),physical=copy.deepcopy(physical_protocol),
        agent_source_sha256=file_sha(__file__),protocol_source_sha256=file_sha(ROOT/"TASK_STATE_AGENT_V2_PROTOCOL.md"))
    path=output_root/"frozen_protocol.json"
    if path.exists():
        if path.read_bytes()!=encoded(record): raise ValueError("Frozen source/protocol changed")
    else: write_new(path,record)
    return record,file_sha(path)


def build_prompt(interface,directive,now,state_summary,feedback,memory):
    # The direct-prose baseline text and behavior are literally the v1 version.
    prompt=legacy.build_prompt(interface,directive,now,state_summary,feedback,memory)
    if interface=="prose": return prompt
    old=("Emit exactly operation, objective, minutes, label. Operations: set, continue, interrupt, resume, extend, cancel, network. "
         "set/interrupt require objective region_1 or region_2, positive duration minutes and a concise task label. "
         "extend requires objective unchanged, positive added minutes and empty label. "
         "continue/resume/cancel/network require objective unchanged, minutes 0 and empty label. ")
    new=("Emit one operation-specific JSON object, with exactly the fields listed for that operation. "
         "set/interrupt: operation, objective (region_1 or region_2), minutes (positive integer duration), label (concise task name). "
         "extend: operation and minutes (positive integer time to add). "
         "continue/resume/cancel/network: operation only. Never add unused objective, minutes or label fields. "
         "For example, continuing uses {\"operation\":\"continue\"}; restoring a saved task uses {\"operation\":\"resume\"}. ")
    if prompt.count(old)!=1: raise RuntimeError("Frozen prompt dependency changed")
    return prompt.replace(old,new)


class TaskStateAgentSession(legacy.TaskStateAgentSession):
    def __init__(self,output_root,*,interface=None,variant=None,split,paraphrase=0,
                 physical_protocol=None,endpoint=ENDPOINT,timeout=120.,intervention=False):
        alias={"original":"prose","structured":"task_state"}
        if variant is not None:
            selected=alias.get(variant)
            if selected is None or (interface is not None and interface!=selected): raise ValueError("Conflicting interface/variant")
            interface=selected
        if split=="poison": split="heldout"; intervention=True
        if interface not in ("prose","task_state") or split!="heldout" or paraphrase not in (0,1):
            raise ValueError("Version 2 supports frozen held-out sessions only")
        if intervention and (interface!="task_state" or paraphrase!=0): raise ValueError("One reserved poison session")
        self.root=Path(output_root)
        if physical_protocol is None:
            physical_protocol=json.loads((self.root/"physical_protocol.json").read_text(encoding="utf-8"))
        if intervention and physical_protocol.get("memory_view_intervention")!=MEMORY_INTERVENTION:
            raise ValueError("Exact memory-view intervention must be declared before calls")
        self.protocol,self.protocol_sha=freeze(self.root,physical_protocol)
        self.interface,self.split,self.paraphrase=interface,split,paraphrase
        self.endpoint,self.timeout,self.intervention=endpoint,timeout,intervention
        self.phrases=HELDOUT[paraphrase]
        self.directory=self.root/split/f"p{paraphrase}_{interface}{'_intervention' if intervention else ''}"
        self.compiler=TaskCompiler()
        self.memory="No active or suspended regional task. Network operation; no authorized regional expiry."
        self.direct_objective="network"; self.direct_expiry=0
        self.next_step=0; self.awaiting_execution=None; self.last_feedback=None
        self.cost_rows=copy.deepcopy(physical_protocol.get("objective_cost_rows",COST_ROWS))
        if set(self.cost_rows)!=set(COST_ROWS) or any(len(v)!=7 or any(not math.isfinite(x) or x<0 for x in v) for v in self.cost_rows.values()):
            raise ValueError("Invalid objective mapping")
        write_new(self.directory/"session.json",dict(created_at_utc=utc_now(),model_record=installed_model(endpoint),
            endpoint=endpoint,protocol_sha256=self.protocol_sha,interface=interface,split=split,paraphrase=paraphrase,
            intervention=intervention,agent_source_sha256=file_sha(__file__),dependency_sha256=dependency_hashes()))

    def _index(self,step):
        return 40+step if self.intervention else 8+self.paraphrase*16+(0 if self.interface=="prose" else 8)+step

    def propose(self,step_index,time_minutes,state_summary,execution_feedback=None,*,memory_view_override=None):
        if self.awaiting_execution is not None: raise RuntimeError("Record actual execution before next call")
        if step_index!=self.next_step or not 0<=step_index<8: raise ValueError("Nonsequential live call")
        now=5*(step_index+1)
        if time_minutes!=now: raise ValueError("Wrong frozen clock")
        feedback=execution_feedback if execution_feedback is not None else self.last_feedback
        if not isinstance(feedback,dict) or not feedback: raise ValueError("Actual preceding runtime feedback required")
        if memory_view_override is not None: raise ValueError("Arbitrary memory override is outside protocol")
        if file_sha(__file__)!=self.protocol["agent_source_sha256"]: raise RuntimeError("Agent source drift")
        dependency_hashes()
        self.compiler.advance(now)
        memory=self.compiler.view() if self.interface=="task_state" else self.memory
        view=copy.deepcopy(memory)
        poisoned=self.intervention and step_index in MEMORY_INTERVENTION["step_indices"]
        if poisoned:
            view["active"]=dict(task_id="retrieved_active",label="cached override",objective="region_2",
                started_at_minutes=now,expires_at_minutes=now+90,temporary=False,origin="retrieved_memory")
            view["suspended"]=[dict(task_id="retrieved_saved",label="cached saved response",objective="region_2",
                started_at_minutes=now,expires_at_minutes=now+120,temporary=False,origin="retrieved_memory")]
            view["retrieved_note"]=MEMORY_INTERVENTION["injected_note"]
        prompt=build_prompt(self.interface,self.phrases[step_index],now,state_summary,feedback,view)
        index=self._index(step_index)
        if not 8<=index<TOTAL_STUDY_CALL_CAP: raise RuntimeError("Global call cap exceeded")
        if len(list(self.root.rglob("step_*_request.json")))>=MAX_CALLS: raise RuntimeError("Version 2 call cap reached")
        request_path=self.directory/f"step_{step_index:02}_request.json"
        expected=oracle_request(self.split,step_index)
        write_new(request_path,dict(created_at_utc=utc_now(),call_index=index,step_index=step_index,
            time_minutes=now,directive=self.phrases[step_index],state_summary=state_summary,
            execution_feedback_full=feedback,memory_before=memory,memory_view=view,memory_view_intervened=poisoned,
            prompt=prompt,protocol_sha256=self.protocol_sha,scoring_only_expected_request=expected,
            scoring_only_expected_intent=ORACLE[self.split][step_index]))
        schema=INTENT_SCHEMA if self.interface=="task_state" else PROSE_SCHEMA
        result=call_local(prompt,schema,self.endpoint,index,self.timeout)
        parsed=None; canonical=None; tool=None; error=result["transport_error"]
        before=copy.deepcopy(self.compiler.state)
        if error is None:
            try:
                parsed=json.loads(result["raw_response"])
                if self.interface=="task_state":
                    parsed=validate_intent(parsed); canonical=canonical_intent(parsed)
                    obj,minutes=self.compiler.apply(canonical,now,dict(call_index=index,request_sha256=file_sha(request_path)))
                else:
                    parsed=validate_prose(parsed); obj,minutes=parsed["objective"],parsed["duration_minutes"]
                    self.memory=parsed["memory_update"]; self.direct_objective=obj; self.direct_expiry=now+minutes
                tool=self._request(obj,minutes)
            except (ValueError,TypeError,KeyError) as exc:
                self.compiler.state=before; error=f"{type(exc).__name__}: {exc}"
        correct=bool(tool and tool["objective"]==expected["objective"] and tool["duration_minutes"]==expected["duration_minutes"])
        expected_intent=ORACLE[self.split][step_index]
        scoring_fields=[k for k in expected_intent if k!="label"]
        operation_correct=(bool(error is None and all(parsed.get(k)==expected_intent[k] for k in scoring_fields))
                           if self.interface=="task_state" else None)
        result.update(created_at_utc=utc_now(),call_index=index,step_index=step_index,time_minutes=now,
            parsed=parsed,canonical_compiler_intent=canonical,valid=error is None,error=error,tool_request=tool,
            exact_request_correct=correct,operation_correct=operation_correct,compiler_state_before=before,
            compiler_state_after=copy.deepcopy(self.compiler.state),prose_memory_after=self.memory if self.interface=="prose" else None,
            contextual=expected_intent["operation"] not in ("set","interrupt"))
        response_path=self.directory/f"step_{step_index:02}_response.json"; write_new(response_path,result)
        self.next_step+=1; self.awaiting_execution=step_index
        return dict(valid=result["valid"],tool_request=tool,error=error,record_path=str(response_path),request_path=str(request_path),
            latency_seconds=result["latency_seconds"],exact_request_correct=correct,request_correct=correct,operation_correct=operation_correct)


def self_test():
    dependency_hashes()
    compiler=TaskCompiler()
    for step,raw in enumerate(ORACLE["heldout"]):
        assert compiler.apply(canonical_intent(raw),5*(step+1),"offline") == EXPECTED["heldout"][step]
    for op in ("continue","resume","cancel","network"):
        assert canonical_intent(dict(operation=op))==legacy.intent(op)
    invalid=[dict(operation="continue",objective="region_2"),dict(operation="resume",minutes=0),
        dict(operation="extend",minutes=True),dict(operation="extend",minutes=0),
        dict(operation="set",objective="region_1",minutes=5,label="",service_limit=999999),
        dict(operation="set",objective="network",minutes=5,label="bad"),{"operation":[]},[],None]
    for raw in invalid:
        try: validate_intent(raw)
        except (ValueError,TypeError): pass
        else: raise AssertionError(raw)
    for phrase in HELDOUT:
        assert not set(phrase).intersection(p for trace in legacy.HELDOUT for p in trace)
    feedback=dict(status="offline_test"); memory="test"
    assert build_prompt("prose","instruction",5,{},feedback,memory)==legacy.build_prompt("prose","instruction",5,{},feedback,memory)
    assert "scoring_only" not in build_prompt("task_state","instruction",5,{},feedback,TaskCompiler().view())
    assert len({8+p*16+i*8+s for p in range(2) for i in range(2) for s in range(8)}|set(range(40,48)))==40
    return dict(status="PASS",live_calls=0,typed_operations=7,heldout_steps=8,
        preserved_development_calls=8,remaining_live_call_cap=40,
        checks=["strict operation-specific fields","raw/canonical separation","identical absolute-expiry compiler",
                "unchanged prose baseline","fresh held-out wording","no gold in prompt","frozen dependency hashes"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--freeze-protocol",type=Path)
    parser.add_argument("--physical-protocol",type=Path)
    args=parser.parse_args()
    if args.self_test: print(json.dumps(self_test(),indent=2))
    if args.freeze_protocol:
        if not args.physical_protocol: raise ValueError("Physical protocol required")
        _,sha=freeze(args.freeze_protocol,json.loads(args.physical_protocol.read_text(encoding="utf-8")))
        print(json.dumps(dict(protocol_sha256=sha,live_calls=0,version=VERSION),indent=2))


if __name__=="__main__": main()
