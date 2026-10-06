"""Independent online execution-binding and class-balance replay audit.

No LLM calls, optimization, verifier edits, or changes to previous records.
Contract/history checks are separate from numerical implementation diagnostics.
An outer-hull membership check is never used to authorize a correlated restart.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from audit_mfd_incident_certified_runtime import exact_affine_contract, mode_fingerprint


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PERIOD = Fraction(1, 12)
END = Fraction(1)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frac(value):
    return Fraction.from_float(float(value))


def descriptor_hash(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def exact_gate_margins(command):
    u = list(map(frac, command))
    values = [margin for value in u for margin in (1-value, value)]
    values += [margin for a,b in ((1,0),(3,2))
               for margin in (Fraction(3,4)-u[a]+u[b], Fraction(3,4)+u[a]-u[b])]
    return values


def check_certificate(cert, *, now, center, widths, commands, residual_veh_h,
                      service_vehicles, expected_model):
    """Reconstruct stored contract without invoking any reachability kernel."""
    desc = cert["proof_descriptor"]
    assert cert["proof_id"] == descriptor_hash(desc)
    assert Fraction(desc["initial_time"]) == now
    durations = list(map(Fraction, desc["durations"]))
    expected = [d for d in (min(PERIOD, END-now), max(Fraction(0), END-now-PERIOD)) if d > 0]
    assert durations == expected and now + sum(durations) == END
    assert desc["initial_state"] == list(center)
    assert cert["initial_box_units"] == list(widths)
    assert frac(desc["residual_bound"]) == frac(residual_veh_h/100.)
    assert desc["service_checks"] == [dict(exact_time="1", coordinate=0, upper=service_vehicles/100.)]
    assert desc["model_contract"] == expected_model
    assert list(map(Fraction, desc["pulse"])) == [frac(.3), Fraction(1), Fraction(1,4), frac(1/12)]
    assert desc["step_seconds"] == 15.
    assert desc["terminal_regional_upper"] is None and desc["terminal_queue_upper"] is None
    assert cert["adapter_audit"]["commands"] == commands
    assert len(commands) == len(durations)
    for row, radius in zip(desc["initial_L"], desc["initial_radius"]):
        assert frac(radius) >= sum(abs(frac(a))*frac(w) for a,w in zip(row,widths))
    for index, command in enumerate(commands):
        record = dict(center=[0.]*7, feedforward=command, gain=np.zeros((4,7)).tolist(),
                      A=np.zeros((7,7)).tolist(), P=np.eye(7).tolist())
        fingerprint = mode_fingerprint(record)
        assert desc["mode_fingerprints"][index] == fingerprint
        assert desc["affine_constraint_contracts"][index]["mode_fingerprint"] == fingerprint
        exact_affine_contract(desc["affine_constraint_contracts"][index], record)
        assert min(exact_gate_margins(command)) >= 0
    if cert["accepted"]:
        assert cert["verified_continuous_physical"] and cert["verified_scheduled_service"]
        assert cert["verified_terminal_service"] and not cert["failures"]
        assert cert["certified_until_h"] == cert["enclosed_until_h"] == 1.
        assert len(cert["service_check_results"]) == 1
        service = cert["service_check_results"][0]
        assert service["passed"] and service["upper"] <= service_vehicles/100.
        margins = cert["continuous_constraint_margins_lower"]
        assert min(margins.values() if isinstance(margins,dict) else margins) >= 0.
    else:
        assert cert["failures"]
    checkpoints = cert["reachable_checkpoints"]
    for item in checkpoints:
        assert item["proof_id"] == cert["proof_id"]
        assert now <= Fraction(item["exact_time_fraction"]) <= END
        if "signed_generator_method" in desc:
            assert item["outer_hull_only"] and not item["restart_authorized"]
    if cert["accepted"]:
        times = {Fraction(item["exact_time_fraction"]) for item in checkpoints}
        assert all(k*PERIOD in times for k in range(13) if now <= k*PERIOD <= END)
    return dict(proof_id=cert["proof_id"], accepted=cert["accepted"], blocks=len(commands),
                affine_contract_rows=24*len(commands), checkpoint_count=len(checkpoints),
                complete_arithmetic_recomputation=False)


def independent_class_field(x, command, offered, model_contract):
    """Class-balance equations assembled independently of the runtime plant RHS."""
    n1,n2,n11,n22,n10,n20,z = x
    u12,u21,u02,u20 = command
    n12, n21 = n1-n11-n10, n2-n22-n20
    a,b,c = model_contract["MFD_coefficients"]
    speed1, speed2 = (a*n1+b)*n1+c, (a*n2+b)*n2+c
    q11,q12,q10 = speed1*np.array([n11,n12,n10])
    q22,q21,q20 = speed2*np.array([n22,n21,n20])
    d11,d12,d10,d22,d21,d20 = model_contract["internal_demand"]
    admission = min(model_contract["release_rate"]*z,
                    model_contract["admission_capacity"]*u02)
    split = model_contract["admission_split"]
    class1 = np.array([d11+u21*q21-q11, d12-u12*q12, d10-u12*q10])
    class2 = np.array([d22+u12*q12+split*admission-q22,
                       d21+(1-split)*admission-u21*q21, d20+u12*q10-u20*q20])
    field = np.array([sum(class1),sum(class2),class1[0],class2[0],class1[2],class2[2],offered-admission])
    return field, q11+q22+u20*q20


def independent_arrival(t, model_contract):
    start, duration, ramp, amplitude = .25, 1., 1/12, .3
    if t < start or t > start+duration:
        weight = 0.
    elif t < start+ramp:
        weight = (t-start)/ramp
    elif t > start+duration-ramp:
        weight = (start+duration-t)/ramp
    else:
        weight = 1.
    return model_contract["base_arrival"]*(1+amplitude*weight)


def independent_physical_margin(x, command, model_contract):
    n1,n2,n11,n22,n10,n20,z = x
    margins = list(x)+[n1-n11-n10,n2-n22-n20,
        model_contract["regional_jam_limits"][0]-n1,
        model_contract["regional_jam_limits"][1]-n2, model_contract["queue_limit"]-z]
    margins += [float(m) for m in exact_gate_margins(command)]
    return min(margins)


def replay_schedule(schedule, model_contract, initial, residual_intervals=()):
    """Numerical consistency only: independently integrate the actual commands.

    Each schedule item supplies exact left/right clocks and a constant command.
    No measured centers reset the independent physical trajectory.
    """
    y = np.r_[initial, np.zeros(4)]
    records, minimum, max_queue = [], np.inf, 100*initial[6]
    cursor = Fraction(0)
    for item in schedule:
        left, right = Fraction(item["left"]), Fraction(item["right"])
        assert left == cursor and right > left
        command = np.asarray(item["command"], dtype=float)
        cuts = sorted(set([left,right]+[frac(k) for k in (.25,.25+1/12)
                                      if float(left) < k < float(right)]+
                          [Fraction(item[key]) for item in residual_intervals for key in ("left","right")
                           if left < Fraction(item[key]) < right]))
        for a,b in zip(cuts[:-1],cuts[1:]):
            residual = sum(item["value_units_h"] for item in residual_intervals
                           if Fraction(item["left"]) <= (a+b)/2 < Fraction(item["right"]))
            def rhs(t, state):
                offered = independent_arrival(t, model_contract)+residual
                field, departures = independent_class_field(state[:7], command, offered, model_contract)
                return np.r_[field, 100*sum(state[[0,1,6]]),100*state[6],100*departures,
                             100*(sum(model_contract["internal_demand"])+offered)]
            solution = solve_ivp(rhs,(float(a),float(b)),y,method="DOP853",rtol=2e-12,
                                 atol=2e-13,max_step=1/720,dense_output=True)
            assert solution.success
            for point in solution.sol(np.linspace(float(a),float(b),31))[:7].T:
                minimum = min(minimum,independent_physical_margin(point,command,model_contract))
                max_queue = max(max_queue,100*point[6])
            y = solution.y[:,-1]
        records.append(dict(exact_time=str(right),state=y[:7].tolist(),TTS_veh_h=float(y[7])))
        cursor = right
    balance = 100*sum(np.asarray(initial)[[0,1,6]])+y[10]-100*sum(y[[0,1,6]])-y[9]
    assert abs(balance) < 1e-6
    return dict(exact_end=str(cursor),checkpoints=records,final_state=y[:7].tolist(),
                total_time_spent_veh_h=float(y[7]),queue_time_veh_h=float(y[8]),
                completed_trips_vehicles=float(y[9]),offered_vehicles=float(y[10]),
                maximum_sampled_queue_vehicles=float(max_queue),
                minimum_sampled_physical_margin=float(minimum) if schedule else None,
                conservation_residual_vehicles=float(balance),
                sampling_is_numerical_diagnostic=True, no_observation_center_resets=True)


def physical_box_minimum(center, widths, command):
    c,w = list(map(frac,center)),list(map(frac,widths))
    margins = [a-b for a,b in zip(c,w)]
    margins += [c[a]-c[b]-c[d]-w[a]-w[b]-w[d] for a,b,d in ((0,2,4),(1,3,5))]
    margins += [Fraction(limit)-c[j]-w[j] for j,limit in ((0,100),(1,100),(6,20))]
    margins += exact_gate_margins(command)
    return min(margins)


def audit_run(case, source, paths):
    spec,contract = case["specification"],case["current_contract"]
    certified = spec["method"] != "nominal"
    csha = descriptor_hash(contract)
    assert csha == case["current_contract_sha256"]
    assert case["source_sha256"] == contract["immutable_source_sha256"] == source["source_sha256"]
    assert case["protocol_sha256"] == source["protocol_sha256"] == contract["online_protocol_sha256"]
    assert case["request_mode"] == "ARCHIVED_SEMANTIC_REPLAY"
    widths = [spec["width"]/100.]*6+[min(spec["width"],10.)/100.]
    assert contract["physical_half_widths"] == widths
    assert contract["residual_bound"] == spec["residual"]/100.
    assert contract["service_checks"] == [dict(exact_time="1",coordinate=0,upper=spec["service"]/100.)]
    assert contract["mission_start_exact_h"] == "0" and contract["mission_expiry_exact_h"] == "1"
    assert contract["deadline_exact_h"] == "1"
    assert contract["pulse"] == [str(frac(.3)),"1","1/4",str(frac(1/12))]
    baseline=read(RESULTS/"mfd_incident_uncertainty_frontier_v1_r60_conservation_box_proof.json")
    assert contract["model"] == baseline["certificate"]["proof_descriptor"]["model_contract"]
    request_path=RESULTS/"mfd_incident_agent_live_p0_stateful_results.json"
    requests=read(request_path)["certified"]["decisions"]
    active=None
    accepted,executed,schedule,residuals,proofs,retained = [],[],[],[],[],[]
    chain=descriptor_hash(dict(genesis=csha))
    state=case["actual_initial_state"]
    assert state == case["trajectory"]["state"][0]
    center0=np.array([30.,30.,15.,15.,5.,5.,2.])
    signs=np.array([1.,-1.,1.,-1.,1.,-1.,1.])
    expected_initial=center0.copy()
    if spec["realization"] == "adverse": expected_initial += .5*np.asarray(widths)*signs
    assert state == expected_initial.tolist()
    counts=dict(decisions=0,candidate_attempts=0,no_candidate_epochs=0,accepted_candidates=0,
                rejected_full_certificates=0,failed_preconditions=0,retained_epochs=0,
                unresolved_initializations=0,first_command_changes=0)
    first_command=None
    for epoch,event in enumerate(case["decisions"]):
        counts["decisions"]+=1
        now,following=epoch*PERIOD,(epoch+1)*PERIOD
        assert event["epoch"] == epoch and Fraction(event["exact_time_h"]) == now
        assert Fraction(event["exact_next_decision_h"]) == following
        assert event["actual_initial_state"] == state == case["trajectory"]["state"][30*epoch]
        expected_request={k:requests[epoch][k] for k in ("epoch","request_valid","tool_request")}
        assert event["archived_request"] == expected_request
        assert event["source_request_sha256"] == sha(request_path)
        assert event["request_mode"] == "ARCHIVED_SEMANTIC_REPLAY"
        info=event["information_box"]
        assert Fraction(info["exact_time_h"]) == now and info["half_widths"] == widths
        assert info["sensor_contract"] == contract["sensor_contract"]
        identity={k:v for k,v in info.items() if k not in ("information_id","simulation_containment_audit")}
        assert descriptor_hash(identity) == info["information_id"]
        exact_errors=[abs(frac(x)-frac(c)) for x,c in zip(state,info["center"])]
        assert all(error <= frac(width) for error,width in zip(exact_errors,widths))
        assert info["simulation_containment_audit"]["passed"]
        assert info["simulation_containment_audit"]["actual_state"] == state
        expected_center=center0.copy() if epoch==0 else np.array(state)
        if spec["realization"]=="adverse" and epoch>0:
            expected_center += .5*np.asarray(widths)*signs*((-1)**epoch)
        assert info["center"] == expected_center.tolist()
        command_list=event["candidate_commands"]
        counts["candidate_attempts" if command_list is not None else "no_candidate_epochs"]+=1
        if epoch==0:
            commission=event["commissioning_source"]
            path=ROOT/commission["source"]
            assert sha(path)==commission["source_sha256"]
            prior=read(path)
            expected_commands=(next(c for c in prior["runs"] if c["amplitude"]==.3 and c["duration_h"]==1.)["decisions"][0]["controls"]
                               if spec["commissioning"]==2850 else
                               next(c for c in prior["rows"] if c["nominal_target_vehicles"]==spec["commissioning"])["commands"])
            assert command_list==expected_commands
        artifact=None
        if event["certificate_reference"] is not None:
            reference=event["certificate_reference"]
            assert certified and command_list is not None
            pair={k:reference[k] for k in ("path","sha256")}
            assert descriptor_hash(pair)==reference["registry_id"]
            assert case["trusted_proof_registry"][reference["registry_id"]]==pair
            path=ROOT/reference["path"]
            assert sha(path)==reference["sha256"]
            paths[reference["path"]]=reference["sha256"]
            artifact=read(path)
            assert artifact["origin"]=="trusted_local_verifier" and artifact["method"]==spec["method"]
            assert artifact["case_id"]==spec["case_id"] and artifact["epoch"]==epoch
            assert artifact["exact_installation_candidate_h"]==str(now)
            assert artifact["current_contract_sha256"]==csha and artifact["information_box"]==info
            assert artifact["commands"]==command_list
            assert artifact["exact_durations_h"]==event["candidate_exact_durations_h"]
            assert artifact["certification_source_sha256"]==source["source_sha256"]
            assert artifact["online_protocol_sha256"]==source["protocol_sha256"]
            cert=artifact["certificate"]
            if cert is not None:
                proof=check_certificate(cert,now=now,center=info["center"],widths=widths,
                    commands=command_list,residual_veh_h=spec["residual"],service_vehicles=spec["service"],
                    expected_model=contract["model"])
                proof.update(path=reference["path"],sha256=reference["sha256"],epoch=epoch)
                proofs.append(proof)
                if not cert["accepted"]: counts["rejected_full_certificates"]+=1
                assert event["candidate_admitted"]==cert["accepted"]
            else:
                assert not event["candidate_admitted"] and artifact["precondition_failure"] is not None
                counts["failed_preconditions"]+=1
                if artifact["precondition_failure"]["kind"]=="invalid_initial_physical_information_set":
                    assert physical_box_minimum(info["center"],widths,command_list[0])<0
        else:
            assert not certified or command_list is None
        if event["candidate_admitted"]:
            counts["accepted_candidates"]+=1
            if certified:
                assert artifact["certificate"]["accepted"]
                installed=event["installation"]
                identity={k:v for k,v in installed.items() if k!="installation_id"}
                assert descriptor_hash(identity)==installed["installation_id"]
                assert installed["method"]==spec["method"]
                assert installed["original_proof_reference"]==event["certificate_reference"]
                assert installed["original_proof_id"]==artifact["certificate"]["proof_id"]
                assert installed["original_commands"]==command_list
                assert installed["original_exact_durations_h"]==event["candidate_exact_durations_h"]
                assert installed["original_installation_exact_h"]==str(now) and installed["original_expiry_exact_h"]=="1"
                assert installed["original_epoch"]==epoch and installed["information_id"]==info["information_id"]
                assert installed["current_contract_sha256"]==csha and installed["executed_prefix_hash_at_installation"]==chain
                assert installed["accepted_history_prefix_hash"]==descriptor_hash(accepted)
                assert installed["predecessor_installation_id"]==(None if active is None else active["installation_id"])
                active=copy.deepcopy(installed)
                accepted.append(copy.deepcopy(installed))
            else:
                active=dict(original_commands=command_list,original_epoch=epoch)
        else:
            assert event["installation"] is None
        actual_command=event["executed_constant_command"]
        if actual_command is None:
            assert active is None and epoch==0 and certified
            assert event["execution_source"]=="unresolved" and not case["metrics"]["complete"]
            counts["unresolved_initializations"]+=1
            continue
        assert active is not None
        counts["first_command_changes"]+=first_command is None or first_command!=actual_command
        first_command=actual_command
        assert event["first_command_changed"]==(epoch==0 or case["decisions"][epoch-1].get("executed_constant_command")!=actual_command)
        if certified:
            cursor=Fraction(active["original_installation_exact_h"])
            selected=None
            for index,(command,duration) in enumerate(zip(active["original_commands"],active["original_exact_durations_h"])):
                end=cursor+Fraction(duration)
                if cursor<=now<end and following<=end: selected=(index,command)
                cursor=end
            assert selected is not None and cursor==END
            index,command=selected
            assert actual_command==command
            binding=dict(installation_id=active["installation_id"],proof_id=active["original_proof_id"],
                proof_reference=active["original_proof_reference"],original_command_index=index,
                original_epoch=active["original_epoch"],original_installation_exact_h=active["original_installation_exact_h"],
                original_expiry_exact_h="1",executed_prefix_hash=chain,exact_left_h=str(now),exact_right_h=str(following))
            assert event["execution_binding"]==binding
            record=dict(**binding,command=actual_command)
            record["execution_record_hash"]=descriptor_hash(record)
            assert event["committed_execution"]==record
            chain=record["execution_record_hash"]
            executed.append(record)
            expected_source="verified_replacement" if event["candidate_admitted"] else "retained_original_certified_suffix"
            assert event["execution_source"]==expected_source
            assert event["outer_hull_diagnostic"]["authority_used"] is False
            if not event["candidate_admitted"]:
                counts["retained_epochs"]+=1
                retained.append(dict(epoch=epoch,original_epoch=active["original_epoch"],
                    proof_id=active["original_proof_id"],original_expiry="1"))
        else:
            assert actual_command==active["original_commands"][0 if active["original_epoch"]==epoch else -1]
        assert Fraction(event["executed_end_time_h"])==following
        actual_commands=case["trajectory"]["command"][30*epoch:30*(epoch+1)]
        assert actual_commands==[actual_command]*30
        state=event["actual_final_state"]
        assert state==case["trajectory"]["state"][30*(epoch+1)]
        schedule.append(dict(left=str(now),right=str(following),command=actual_command))
        residual=event["actual_residual_units_per_hour"]
        expected_residual=0. if spec["realization"]=="nominal" else spec["residual"]/100.*(1 if epoch//3%2==0 else -1)
        assert residual==expected_residual and abs(frac(residual))<=frac(contract["residual_bound"])
        residuals.append(dict(left=str(now),right=str(following),value_units_h=residual))
    assert accepted==case["accepted_history"] and executed==case["executed_history"]
    metrics=case["metrics"]
    assert counts["accepted_candidates"]==metrics["candidate_acceptances"]
    assert counts["decisions"]-counts["accepted_candidates"]==metrics["rejected_candidates"]
    assert counts["retained_epochs"]==metrics["retained_epochs"]
    assert counts["first_command_changes"]==metrics["first_command_changes"]
    assert metrics["complete"]==(len(schedule)==12)
    if metrics["complete"] and certified:
        assert case["expiry_guard"]["passed"] and case["expiry_guard"]["rejected"]
        assert case["expiry_guard"]["tested_interval"]==["1","13/12"]
    replay=replay_schedule(schedule,contract["model"],case["actual_initial_state"],residuals)
    state_errors=[100*max(abs(np.array(item["state"])-np.array(event["actual_final_state"])))
                  for item,event in zip(replay["checkpoints"],[e for e in case["decisions"] if e["executed_constant_command"] is not None])]
    error=max(state_errors,default=0.)
    assert error<.005,(spec["case_id"],error)
    tts_difference=replay["total_time_spent_veh_h"]-metrics["tts_veh_h"]
    queue_difference=replay["queue_time_veh_h"]-metrics["queue_veh_h"]
    assert abs(tts_difference)<.02 and abs(queue_difference)<.02
    if metrics["complete"] and certified:
        assert 100*replay["final_state"][0]<=spec["service"]
        assert replay["minimum_sampled_physical_margin"]>=-1e-9
    return dict(case_id=spec["case_id"],specification=spec,counts=counts,proof_checks=proofs,
                retained_history=retained,independent_replay=replay,
                maximum_epoch_state_error_vehicles=error,TTS_difference_veh_h=tts_difference,
                queue_time_difference_veh_h=queue_difference,
                exact_sensor_containment_checks=len(case["decisions"]),
                fresh_installation_does_not_require_parent_set_containment=True,
                original_proof_retention_without_outer_hull_restart=True)


def mutation_tests(source, completed):
    """Exercise the production authority on copied, cached proof fixtures.

    The fixture registry is seeded explicitly from previously audited artifacts
    to avoid new proof calls. White-box artifact mutations are separately
    identified; ordinary file tampering is tested without updating its hash.
    No scientific artifact, kernel or runtime global is modified.
    """
    import inspect
    import mfd_structure_online_runtime as runtime

    required = ['base_original','base_positive_queue','base_affine_v3','binding1625_affine_v3']
    if not all(name in completed for name in required):
        return dict(completed=False, missing_fixture_cases=[n for n in required if n not in completed])
    rows = []
    with tempfile.TemporaryDirectory(prefix='mfd-authority-audit-') as temporary:
        directory=Path(temporary)
        number=0

        def fixture(case_id='base_affine_v3', epoch=0, mutate=None, authority_method=None):
            nonlocal number
            case=completed[case_id]
            event=case['decisions'][epoch]
            artifact=read(ROOT/event['certificate_reference']['path'])
            if mutate is not None:
                mutate(artifact)
            number+=1
            path=directory/f'fixture_{number:03}.json'
            path.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            pair=dict(path=str(path),sha256=sha(path))
            reference=dict(registry_id=runtime.digest(pair),**pair)
            contract=copy.deepcopy(case['current_contract'])
            store=runtime.TrustedProofStore('independent_audit_fixture',source['source_sha256'])
            store.registry[reference['registry_id']]=copy.deepcopy(pair)
            authority=runtime.RuntimeAuthority(contract,authority_method or case['specification']['method'],store)
            return authority,store,reference,copy.deepcopy(artifact['information_box']),contract,path

        def installed(case_id='base_affine_v3'):
            values=fixture(case_id)
            authority,store,reference,info,contract,path=values
            authority.install(reference,info,Fraction(0),contract)
            return values

        def reject(name, action, expected=None, scope='production_guard'):
            try:
                action()
            except runtime.AuthorityError as exc:
                if expected is not None:
                    assert expected in str(exc),(name,str(exc),expected)
                rows.append(dict(name=name,passed=True,rejected=True,reason=str(exc),scope=scope))
            else:
                raise AssertionError(('authority mutation incorrectly admitted',name))

        def positive(name, **details):
            rows.append(dict(name=name,passed=True,rejected=False,**details))

        def reseal_info(info):
            identity={k:v for k,v in info.items() if k not in ('information_id','simulation_containment_audit')}
            info['information_id']=runtime.digest(identity)

        def reseal_descriptor(artifact):
            certificate=artifact['certificate']
            certificate['proof_id']=runtime.digest(certificate['proof_descriptor'])

        # Positive-control cases establish that each real method can install,
        # commit, and retain its already certified second block without a reset.
        for method in ('original','positive_queue','affine_v3'):
            authority,store,ref,info,contract,path=installed('base_'+method)
            first,binding=authority.command_for(0,PERIOD,contract)
            before=copy.deepcopy(authority.active)
            authority.commit_execution(0,PERIOD,first,contract)
            tail,retained=authority.command_for(PERIOD,2*PERIOD,contract)
            assert tail==before['original_commands'][1]
            assert authority.active==before and retained['proof_id']==binding['proof_id']
            authority.commit_execution(PERIOD,2*PERIOD,tail,contract)
            positive('valid_install_and_retention_'+method,original_proof_unchanged=True)

        authority,store,ref,info,contract,path=fixture()
        reject('unresolved_initialization',lambda:authority.command_for(0,PERIOD,contract),'no accepted plan')
        reject('skipped_installation_clock',lambda:authority.install(ref,info,PERIOD,contract),'installation clock')

        for name,mutator in (
            ('task_version',lambda c:c.__setitem__('version','forged_task_version')),
            ('model_storage',lambda c:c['model'].__setitem__('queue_limit',21.)),
            ('forecast',lambda c:c['pulse'].__setitem__(0,'1/2')),
            ('residual',lambda c:c.__setitem__('residual_bound',.7)),
            ('service_limit',lambda c:c['service_checks'][0].__setitem__('upper',99.)),
            ('service_deadline',lambda c:c['service_checks'][0].__setitem__('exact_time','13/12')),
            ('missing_pending_service',lambda c:c.__setitem__('service_checks',[])),
            ('mission_expiry',lambda c:c.__setitem__('mission_expiry_exact_h','2')),
            ('sensor_contract',lambda c:c.__setitem__('sensor_contract','forged_sensor')),
            ('information_width',lambda c:c['physical_half_widths'].__setitem__(0,.02))):
            changed=copy.deepcopy(contract);mutator(changed)
            reject('changed_contract_'+name,lambda changed=changed:authority.install(ref,info,0,changed),'trusted contract changed')

        changed=copy.deepcopy(contract)
        key=next(iter(changed['immutable_source_sha256']))
        changed['immutable_source_sha256'][key]='0'*64
        bad=runtime.RuntimeAuthority(changed,'affine_v3',store)
        reject('forged_installed_source_manifest',lambda:bad.install(ref,info,0,changed),'trusted source changed')

        for name,mutator in (
            ('stale',lambda i:i.__setitem__('exact_time_h','-1/12')),
            ('future',lambda i:i.__setitem__('exact_time_h','1/12')),
            ('center_one_ulp',lambda i:i['center'].__setitem__(0,float(np.nextafter(i['center'][0],np.inf)))),
            ('width_one_ulp',lambda i:i['half_widths'].__setitem__(0,float(np.nextafter(i['half_widths'][0],np.inf)))),
            ('sensor_identity',lambda i:i.__setitem__('sensor_contract','forged_sensor'))):
            changed=copy.deepcopy(info);mutator(changed);reseal_info(changed)
            reject('changed_information_'+name,lambda changed=changed:authority.install(ref,changed,0,contract))
        changed=copy.deepcopy(info);changed['information_id']='0'*64
        reject('forged_information_id',lambda:authority.install(ref,changed,0,contract),'information identity mismatch')
        changed=copy.deepcopy(ref);changed['registry_id']='0'*64
        reject('unregistered_proof_reference',lambda:authority.install(changed,info,0,contract),'unregistered proof artifact')
        changed=copy.deepcopy(ref);changed['sha256']='0'*64
        reject('altered_proof_reference_hash',lambda:authority.install(changed,info,0,contract),'unregistered proof artifact')

        # Tampering with a copied artifact, while retaining its registered hash,
        # is rejected even when the descriptor and descriptor hash stay valid.
        tampered=read(path);tampered['certificate']['accepted']=False
        path.write_text(json.dumps(tampered,indent=2)+'\n',encoding='utf-8')
        reject('changed_evidence_flag_same_descriptor',lambda:authority.install(ref,info,0,contract),'byte hash mismatch')

        authority,store,ref,info,contract,path=fixture(authority_method='positive_queue')
        reject('wrong_certified_method',lambda:authority.install(ref,info,0,contract),'method or task contract mismatch')

        # Inner guards are also tested with explicitly white-box, resealed
        # fixtures. These are fault injections, not claims that an agent can
        # write the trusted registry or obtain a false verifier result.
        def mutate_command(artifact):
            artifact['commands'][0][0]=float(np.nextafter(artifact['commands'][0][0],np.inf))

        def mutate_expiry(artifact):
            artifact['exact_durations_h']=['1/12','5/6']
            artifact['certificate']['proof_descriptor']['durations']=['1/12','5/6']
            reseal_descriptor(artifact)

        def mutate_service(artifact):
            artifact['certificate']['proof_descriptor']['service_checks']=[]
            reseal_descriptor(artifact)

        def mutate_affine_source(artifact):
            descriptor=artifact['certificate']['proof_descriptor']
            descriptor['signed_generator_method']['source_sha256']['reproducibility/mfd_incident_affine_reachability.py']='0'*64
            reseal_descriptor(artifact)

        def mutate_affine_contraction(artifact):
            artifact['certificate']['proof_descriptor']['signed_generator_method']['generator_or_remainder_contraction']=True
            reseal_descriptor(artifact)

        for name,mutation,expected in (
            ('command_binding',mutate_command,'command binding'),
            ('shortened_expiry',mutate_expiry,'absolute expiry'),
            ('missing_service',mutate_service,'physical/time/service assumptions'),
            ('affine_source',mutate_affine_source,'kernel source or correlation contract'),
            ('affine_unverified_contraction',mutate_affine_contraction,'kernel source or correlation contract')):
            authority,store,ref,info,contract,path=fixture(mutate=mutation)
            reject('white_box_'+name,lambda:authority.install(ref,info,0,contract),expected,
                   scope='white_box_resealed_fixture_fault_injection')

        authority,store,ref,info,contract,path=installed()
        reject('skipped_execution_clock',lambda:authority.command_for(PERIOD,2*PERIOD,contract),'contiguous')
        reject('zero_execution_interval',lambda:authority.command_for(0,0,contract),'contiguous')
        reject('crossing_original_command_boundary',lambda:authority.command_for(0,2*PERIOD,contract),'one original command block')
        correct,_=authority.command_for(0,PERIOD,contract)
        changed=copy.deepcopy(correct);changed[0]=float(np.nextafter(changed[0],np.inf))
        reject('one_ulp_wrong_executed_command',lambda:authority.commit_execution(0,PERIOD,changed,contract),'differs from')
        assert authority.next_time==0 and not authority.execution_history
        authority.commit_execution(0,PERIOD,correct,contract)
        reject('repeated_execution_commit',lambda:authority.commit_execution(0,PERIOD,correct,contract),'contiguous')
        changed_info=copy.deepcopy(info);changed_info['exact_time_h']=str(PERIOD);reseal_info(changed_info)
        reject('old_proof_relabelled_as_fresh',lambda:authority.install(ref,changed_info,PERIOD,contract),'not generated for this information')
        for k in range(1,12):
            command,_=authority.command_for(k*PERIOD,(k+1)*PERIOD,contract)
            authority.commit_execution(k*PERIOD,(k+1)*PERIOD,command,contract)
        reject('execution_beyond_original_expiry',lambda:authority.command_for(END,END+PERIOD,contract),'expired')
        reject('installation_at_expired_mission',lambda:authority.install(ref,info,END,contract),'mission expired')

        for name,mutation in (
            ('schedule',lambda a:a.active['original_commands'][1].__setitem__(0,float(np.nextafter(a.active['original_commands'][1][0],np.inf)))),
            ('expiry',lambda a:a.active.__setitem__('original_expiry_exact_h','2')),
            ('history',lambda a:a.accepted_history[-1].__setitem__('original_epoch',999))):
            authority,store,ref,info,contract,path=installed();mutation(authority)
            reject('tampered_installed_'+name,lambda:authority.command_for(0,PERIOD,contract),'history or original schedule changed')

        # A genuine naturally rejected service replacement preserves the
        # already installed affine parent and its original second block.
        case=completed['binding1625_affine_v3']
        authority,store,ref,info,contract,path=installed('binding1625_affine_v3')
        first,_=authority.command_for(0,PERIOD,contract)
        authority.commit_execution(0,PERIOD,first,contract)
        saved_parent=copy.deepcopy(authority.active)
        event=case['decisions'][1]
        failed_ref=event['certificate_reference']
        assert not read(ROOT/failed_ref['path'])['certificate']['accepted']
        pair={k:failed_ref[k] for k in ('path','sha256')}
        store.registry[failed_ref['registry_id']]=pair
        reject('genuine_failed_candidate',lambda:authority.install(failed_ref,event['information_box'],PERIOD,contract),
               'no accepted complete finite-plan certificate')
        assert authority.active==saved_parent
        tail,binding=authority.command_for(PERIOD,2*PERIOD,contract)
        assert tail==saved_parent['original_commands'][1] and binding['proof_id']==saved_parent['original_proof_id']
        positive('failed_install_preserves_original_correlated_parent',original_expiry=saved_parent['original_expiry_exact_h'])

        # This static check complements the mathematical counterexample in
        # audit_online_structure_semantics.py: hull diagnostics never enter
        # authority admission, command selection or execution commitment.
        authority_source=inspect.getsource(runtime.RuntimeAuthority)
        assert 'contains_information_box' not in authority_source and 'outer_hull_diagnostic' not in authority_source
        positive('outer_hull_diagnostic_not_used_by_authority')
    return dict(completed=True,passed=True,tests=rows,negative_tests=sum(r['rejected'] for r in rows),
        positive_controls=sum(not r['rejected'] for r in rows),
        production_runtime_sha256=sha(Path(runtime.__file__)),
        fixture_scope='temporary copies of previously audited trusted artifacts; no new optimization or proof calls',
        source_artifacts_unchanged=True)


def audit_completed(output=None, previous=None):
    """Audit a stable snapshot of every case currently published by the runner."""
    import mfd_structure_online_runtime as runtime
    source_path=RESULTS/'mfd_structure_online_runtime_results.json'
    raw=source_path.read_bytes()
    source=json.loads(raw)
    assert source['source_sha256']==runtime.source_hashes()
    assert source['frozen_menu']==runtime.declared_menu()
    assert source['protocol_sha256']==sha(ROOT/source['protocol_path'])
    assert source['new_live_calls']==0 and source['scientific_kernel_changes'] is False
    manifest={source_path.relative_to(ROOT).as_posix():hashlib.sha256(raw).hexdigest()}
    for name,digest in source['source_sha256'].items():
        assert sha(ROOT/name)==digest,('source',name)
        snapshot=source['source_snapshots'][name]
        assert sha(ROOT/snapshot)==digest,('source_snapshot',snapshot)
        manifest[name]=digest;manifest[snapshot]=digest
    expected={c['case_id']:c for c in source['frozen_menu']}
    assert len(source['cases'])==len({c['case_id'] for c in source['cases']})
    prior={r['case_id']:r for r in (previous or {}).get('runs',[])}
    runs,completed=[],{}
    for entry in source['cases']:
        path=ROOT/entry['run_file']
        assert sha(path)==entry['run_sha256']
        case=read(path)
        assert case['specification']==entry['specification']==expected[entry['case_id']]
        assert case['metrics']==entry['metrics']
        manifest[entry['run_file']]=entry['run_sha256']
        completed[entry['case_id']]=case
        cached=prior.get(entry['case_id'])
        if cached is not None and cached.get('run_sha256')==entry['run_sha256']:
            record=copy.deepcopy(cached)
            for name,digest in record['proof_artifact_sha256'].items():
                assert sha(ROOT/name)==digest
                manifest[name]=digest
        else:
            proof_paths={}
            record=audit_run(case,source,proof_paths)
            record.update(run_file=entry['run_file'],run_sha256=entry['run_sha256'],proof_artifact_sha256=proof_paths)
            manifest.update(proof_paths)
            print(json.dumps(dict(audited_case=entry['case_id'],counts=record['counts'],
                maximum_state_error_vehicles=record['maximum_epoch_state_error_vehicles'],
                TTS_difference_veh_h=record['TTS_difference_veh_h'])),flush=True)
        runs.append(record)
    for comparison in source['comparisons']:
        left,right=completed[comparison['case_id']],completed[comparison['comparator']]
        assert left['specification']['realization']==right['specification']['realization']=='nominal'
        assert left['specification']['commissioning']==right['specification']['commissioning']
        assert left['specification']['service']==right['specification']['service']
        assert left['decisions'][0]['candidate_commands']==right['decisions'][0]['candidate_commands']
        both=left['metrics']['complete'] and right['metrics']['complete']
        assert comparison['matched_complete_horizon']==both
        if both:
            delta=left['metrics']['tts_veh_h']-right['metrics']['tts_veh_h']
            assert comparison['incremental_tts_veh_h']==delta
            assert comparison['incremental_tts_percent']==100*delta/right['metrics']['tts_veh_h']
        else:
            assert comparison['incremental_tts_veh_h'] is None and comparison['incremental_tts_percent'] is None
    mutations=(copy.deepcopy(previous['authority_mutations'])
               if previous and previous.get('authority_mutations',{}).get('completed') else mutation_tests(source,completed))
    if mutations.get('completed'):
        assert mutations['production_runtime_sha256']==sha(ROOT/'reproducibility/mfd_structure_online_runtime.py')
    complete=source['completed'] and len(runs)==len(source['frozen_menu']) and mutations.get('completed',False)
    assert source['completed']==(len(source['cases'])==len(source['frozen_menu']))
    result=dict(passed=True,complete=complete,source_result_completed=source['completed'],
        audited_source_result_sha256=hashlib.sha256(raw).hexdigest(),
        audit_source_sha256=sha(Path(__file__)),protocol_sha256=source['protocol_sha256'],
        expected_cases=len(source['frozen_menu']),audited_cases=len(runs),runs=runs,
        authority_mutations=mutations,artifact_sha256=manifest,
        totals=dict(proofs=sum(len(r['proof_checks']) for r in runs),
                    exact_sensor_boxes=sum(r['exact_sensor_containment_checks'] for r in runs),
                    retained_epochs=sum(r['counts']['retained_epochs'] for r in runs),
                    maximum_epoch_state_error_vehicles=max((r['maximum_epoch_state_error_vehicles'] for r in runs),default=0.),
                    maximum_abs_TTS_difference_veh_h=max((abs(r['TTS_difference_veh_h']) for r in runs),default=0.)),
        numerical_scope='independent class-balance replays and exact binding checks; no claim of full recomputation of all nonlinear interval arithmetic')
    if output is not None:
        output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--poll-seconds',type=float,default=20.)
    parser.add_argument('--output',type=Path,default=RESULTS/'mfd_structure_online_runtime_audit_results.json')
    args=parser.parse_args()
    if not 1<=args.poll_seconds<=60: raise ValueError('poll interval must be 1..60 seconds')
    previous=None
    if args.output.exists():
        previous=read(args.output)
        if previous['audit_source_sha256']!=sha(Path(__file__)):
            previous=None  # Re-audit after audit-code changes; scientific inputs remain unchanged.
    last_source=None
    while True:
        source_path=RESULTS/'mfd_structure_online_runtime_results.json'
        current=sha(source_path)
        if current!=last_source:
            try:
                previous=audit_completed(args.output,previous)
            except json.JSONDecodeError:
                time.sleep(.2)
                continue  # Runner may be publishing its next summary at this instant.
            last_source=previous['audited_source_result_sha256']
            print(json.dumps(dict(passed=previous['passed'],complete=previous['complete'],
                audited_cases=previous['audited_cases'],expected_cases=previous['expected_cases'],
                authority_negative_tests=previous['authority_mutations'].get('negative_tests',0),
                totals=previous['totals'])),flush=True)
        if not args.watch or previous['complete']: break
        time.sleep(args.poll_seconds)


if __name__=='__main__':
    main()
