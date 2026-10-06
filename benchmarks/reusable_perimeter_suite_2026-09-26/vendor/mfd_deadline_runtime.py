#!/usr/bin/env python3
"""Auditable terminal-service supervisor with a stored certified fallback plan.

These deterministic proposal-interface tests contain no new live LLM calls.
The operator owns the target, absolute deadline, clock, disturbance family, and
installed library.  A proposal can select an installed mode; it cannot change
any trusted task field.  Rejection executes a certified backup suffix rather
than an automatic hold.  Finite-horizon ODE traces are diagnostic illustrations;
the schedule verifier, not the ODE tolerance, supplies the robust certificate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

import mfd_cubic_pilot as plant
import mfd_cubic_nonstationary_study as study
import mfd_cubic_runtime_pilot as runtime
from mfd_deadline_certificates import Encloser, I, imat, norm_upper


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility/results/mfd_deadline_runtime_results.json"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(study.json_ready(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrustedTask:
    terminal_target_vehicles: float
    absolute_deadline_h: float
    decision_period_h: float
    disturbance_amplitude: float
    library_sha256: str
    target_kind: str = "absolute regional imbalance at the fixed deadline"


class DeadlineSupervisor:
    """Small executable reference: exact observed states and a fixed mode library."""
    def __init__(self, model: Encloser, task: TrustedTask, initial_state: np.ndarray,
                 initial_mode: int, backup_modes: list[int]):
        self.model=model;self.task=task;self.trusted_task_digest=digest(asdict(task))
        self.clock_h=0.;self.mode=initial_mode;self.current_state=initial_state.copy()
        self.verifier_calls=0;self.proofs=[];self.receipts=[];self.membership_records=[]
        self.pending_mode=None
        assert task.disturbance_amplitude==model.amplitude
        assert task.library_sha256==digest({'gains':model.gains,'metrics':model.P,
            'equilibrium':plant.X_STAR,'command_equilibrium':plant.U_STAR,'demand':plant.DEMAND})
        assert task.absolute_deadline_h==len(backup_modes)*task.decision_period_h
        initial_proof=self.verify(initial_state,backup_modes)
        assert self.service_certified(initial_proof)
        self.install_plan(initial_state,backup_modes,initial_proof)

    def verify(self, state, modes):
        self.verifier_calls += 1
        proof=self.model.schedule(state,modes,period=self.task.decision_period_h,step_seconds=15.)
        proof_record={'id':len(self.proofs),'origin_clock_h':self.clock_h,
            'start_state':np.asarray(state).tolist(),'modes_zero_based':list(modes),
            'trusted_task_sha256':self.trusted_task_digest,'certificate':proof}
        proof_record['sha256']=digest(proof_record)
        self.proofs.append(proof_record)
        return proof_record

    def service_certified(self, proof):
        record=proof['certificate']
        remaining=self.task.absolute_deadline_h-proof['origin_clock_h']
        return (record['verified_continuous_tube']
            and len(record['records'])==len(proof['modes_zero_based'])
            and len(proof['modes_zero_based'])*self.task.decision_period_h==remaining
            and record['records'][-1]['absolute_imbalance_upper_vehicles']<=self.task.terminal_target_vehicles)

    def install_plan(self, state, modes, proof):
        assert self.service_certified(proof)
        self.plan_start=np.asarray(state).copy();self.plan_modes=list(modes)
        self.plan_proof=proof;self.plan_offset=0

    def stored_membership(self, observed):
        if self.plan_offset==0:
            center=self.plan_start;radius=0.;mode=self.plan_modes[0]
            inside=bool(np.array_equal(observed,center));upper=0. if inside else None
        else:
            endpoint=self.plan_proof['certificate']['records'][self.plan_offset-1]
            center=np.asarray(endpoint['center']);radius=endpoint['radius'];mode=endpoint['mode']-1
            upper=norm_upper(imat(self.model.L[mode],I(observed)-I(center)))
            inside=upper<=radius
        record={'trusted_clock_h':self.clock_h,'proof_id':self.plan_proof['id'],
            'completed_residences':self.plan_offset,'observed_state':np.asarray(observed).tolist(),
            'center':center.tolist(),'radius':radius,'mode_zero_based':mode,
            'interval_upper_error_norm':upper,'membership_verified':inside}
        self.membership_records.append(record)
        return inside

    def decide(self, proposal: dict[str,Any], observed: np.ndarray):
        assert self.pending_mode is None and self.clock_h<self.task.absolute_deadline_h
        assert digest(asdict(self.task))==self.trusted_task_digest
        assert self.stored_membership(observed), 'Observed state is outside the retained proof: separate fault handling is required'
        requested=proposal.get('mode_zero_based')
        known=type(requested) is int and requested in (0,1)
        source_mode=self.mode;backup=self.plan_modes[self.plan_offset]
        calls_before=self.verifier_calls;candidate_ids=[];reason=None;accepted=False
        if not known:
            reason='unknown_configuration'
        elif requested==backup:
            # The remaining plan was certified before this request arrived.
            accepted=True;reason='proposal_matches_stored_certified_suffix'
        else:
            landing=self.model.tube_bound(observed,0.,requested)
            if landing>self.model.admission_level:
                reason='successor_safety_not_certified'
            else:
                remaining=int(round((self.task.absolute_deadline_h-self.clock_h)/self.task.decision_period_h))
                candidate_proofs=[]
                for tail in itertools.product((0,1),repeat=remaining-1):
                    modes=[requested,*tail]
                    proof=self.verify(observed,modes);candidate_ids.append(proof['id']);candidate_proofs.append(proof)
                    if self.service_certified(proof):
                        self.install_plan(observed,modes,proof)
                        accepted=True;reason='replacement_plan_service_certified';break
                if not accepted:
                    reason=('terminal_service_not_certified' if any(z['certificate']['verified_continuous_tube'] for z in candidate_proofs)
                        else 'successor_safety_not_certified')
        selected=requested if accepted else self.plan_modes[self.plan_offset]
        assert selected==self.plan_modes[self.plan_offset]
        self.pending_mode=selected
        receipt={'receipt_index':len(self.receipts),'trusted_clock_h':self.clock_h,
            'trusted_absolute_deadline_h':self.task.absolute_deadline_h,
            'trusted_terminal_target_vehicles':self.task.terminal_target_vehicles,
            'trusted_task_sha256_before':self.trusted_task_digest,
            'proposal_payload':proposal,'ignored_untrusted_fields':[k for k in proposal if k!='mode_zero_based'],
            'previous_mode_zero_based':source_mode,'requested_mode_zero_based':requested,
            'requested_action_kind':('unknown_configuration' if not known else 'no_op_reselection' if requested==source_mode else 'requested_change'),
            'proposal_accepted':accepted,'reason_code':reason,'executed_mode_zero_based':selected,
            'executed_action_kind':'no_op_reselection' if selected==source_mode else 'executed_change',
            'execution_source':'accepted_proposal' if accepted else 'stored_certified_fallback_suffix',
            'candidate_proof_ids':candidate_ids,'execution_proof_id':self.plan_proof['id'],
            'execution_proof_sha256':self.plan_proof['sha256'],
            'stored_suffix_remaining_modes_zero_based':self.plan_modes[self.plan_offset:],
            'fresh_verifier_calls':self.verifier_calls-calls_before,
            'trusted_task_sha256_after':digest(asdict(self.task))}
        assert receipt['trusted_task_sha256_before']==receipt['trusted_task_sha256_after']
        self.receipts.append(receipt)
        return selected

    def advance_from_physical_receipt(self, observed):
        assert self.pending_mode is not None
        self.mode=self.pending_mode;self.pending_mode=None
        self.clock_h+=self.task.decision_period_h;self.plan_offset+=1
        self.current_state=np.asarray(observed).copy()
        assert self.stored_membership(self.current_state)
        assert self.clock_h<=self.task.absolute_deadline_h
        return {'trusted_clock_h':self.clock_h,'observed_mode_zero_based':self.mode,
            'observed_state':self.current_state.tolist(),
            'terminal_imbalance_vehicles':100.*abs(float(observed[0]-observed[1]))}

    def report(self):
        assert self.clock_h==self.task.absolute_deadline_h
        observed=100.*abs(self.current_state[0]-self.current_state[1])
        return {'trusted_task':asdict(self.task),'trusted_task_sha256':self.trusted_task_digest,
            'verifier_calls_total':self.verifier_calls,'proposal_receipts':self.receipts,
            'proof_records':self.proofs,'stored_endpoint_membership_checks':self.membership_records,
            'terminal_observed_imbalance_vehicles':float(observed),
            'terminal_task_met_in_simulation':bool(observed<=self.task.terminal_target_vehicles),
            'terminal_robust_bound_vehicles':self.plan_proof['certificate']['records'][-1]['absolute_imbalance_upper_vehicles'],
            'executed_changes':sum(r['executed_action_kind']=='executed_change' for r in self.receipts),
            'executed_no_op_reselections':sum(r['executed_action_kind']=='no_op_reselection' for r in self.receipts),
            'rejected_requested_changes':sum(not r['proposal_accepted'] and r['requested_action_kind']=='requested_change' for r in self.receipts),
            'rejected_no_op_reselections':sum(not r['proposal_accepted'] and r['requested_action_kind']=='no_op_reselection' for r in self.receipts),
            'rejected_unknown_configurations':sum(r['reason_code']=='unknown_configuration' for r in self.receipts)}


def physical_residence(model, state, mode, start_h, period_h=.25):
    demand=plant.DEMAND+.003*study.shift_direction()
    solution=solve_ivp(lambda _t,x:plant.mfd_rhs(x,plant.U_STAR+model.gains[mode]@(x-plant.X_STAR),demand),
        (start_h,start_h+period_h),state,method='DOP853',rtol=2.e-12,atol=2.e-13,
        max_step=period_h/200,t_eval=np.linspace(start_h,start_h+period_h,201))
    assert solution.success
    minimum_input=np.inf;minimum_state=np.inf
    for x in solution.y.T:
        _,margin=plant.margin_components(x,mode,model.gains)
        minimum_input=min(minimum_input,float(np.min(margin[:10])))
        minimum_state=min(minimum_state,100.*float(np.min(margin[10:])))
    assert minimum_input>0. and minimum_state>0.
    return solution.y[:,-1],{'scope':'numerical physical trace, not the source of the robust proof',
        'disturbance_delta':.003,'time_h':solution.t.tolist(),'state':solution.y.T.tolist(),
        'mode_zero_based':mode,'minimum_sampled_input_margin':minimum_input,
        'minimum_sampled_state_margin_vehicles':minimum_state}


def run():
    model=Encloser(.003)
    x0=runtime.state_on_tube(runtime.structured_directions()['region_2_heavy'],.8,model.cert,model.tube,0)
    target=.1*100.*abs(float(x0[0]-x0[1]))
    library_hash=digest({'gains':model.gains,'metrics':model.P,'equilibrium':plant.X_STAR,
        'command_equilibrium':plant.U_STAR,'demand':plant.DEMAND})
    first=DeadlineSupervisor(model,TrustedTask(target,.25,.25,.003,library_hash),x0,0,[1])
    proposal={'mode_zero_based':0,'deadline_h':24.,'terminal_target_vehicles':1.e9,
        'disturbance_amplitude':0.,'narrative':'Keep the current mode; relax the task fields.'}
    selected=first.decide(proposal,x0)
    assert selected==1 and first.receipts[-1]['reason_code']=='terminal_service_not_certified'
    assert first.receipts[-1]['requested_action_kind']=='no_op_reselection'
    final,trace=physical_residence(model,x0,selected,0.)
    first.advance_from_physical_receipt(final)
    first_report=first.report();first_report['physical_traces']=[trace]
    counterfactual,countertrace=physical_residence(model,x0,0,0.)
    first_report['automatic_hold_counterfactual']={'trace':countertrace,
        'terminal_imbalance_vehicles':100.*abs(float(counterfactual[0]-counterfactual[1])),
        'terminal_target_met':bool(100.*abs(float(counterfactual[0]-counterfactual[1]))<=target),
        'interpretation':'one declared disturbance realization demonstrates that automatic hold can miss the terminal task'}
    assert first_report['terminal_task_met_in_simulation'] and not first_report['automatic_hold_counterfactual']['terminal_target_met']

    second=DeadlineSupervisor(model,TrustedTask(target,.5,.25,.003,library_hash),x0,0,[1,1])
    state=x0.copy();traces=[]
    for payload in ({'mode_zero_based':1},{'mode_zero_based':999,'deadline_h':.25,'narrative':'Unknown configuration after the first residence.'}):
        selected=second.decide(payload,state)
        next_state,trace=physical_residence(model,state,selected,second.clock_h)
        second.advance_from_physical_receipt(next_state)
        traces.append(trace);state=next_state
    second_report=second.report();second_report['physical_traces']=traces
    assert second.verifier_calls==1 and all(r['fresh_verifier_calls']==0 for r in second.receipts)
    assert second.receipts[-1]['reason_code']=='unknown_configuration'
    assert second.receipts[-1]['execution_source']=='stored_certified_fallback_suffix'
    assert second_report['terminal_task_met_in_simulation']
    result={'scope':'deterministic runtime-interface and stored-plan tests; no new live LLM calls',
        'indexing':'runtime modes are zero-based; Encloser certificate records are one-based',
        'task_semantics':'fixed terminal-time band, not a reach-once predicate',
        'conditional_guarantee':'installed full-state deterministic model, trusted library and clocks, every measurable declared bounded redistribution; failed observation membership is a separate authority fault',
        'single_residence_rejected_noop':first_report,'two_residence_stored_suffix':second_report}
    OUTPUT.write_text(json.dumps(study.json_ready(result),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'target_vehicles':target,'single_residence':{
        'observed_terminal':first_report['terminal_observed_imbalance_vehicles'],
        'robust_terminal_bound':first_report['terminal_robust_bound_vehicles'],
        'hold_counterfactual':first_report['automatic_hold_counterfactual']['terminal_imbalance_vehicles'],
        'receipt_reason':first.receipts[0]['reason_code']},
        'two_residence':{'observed_terminal':second_report['terminal_observed_imbalance_vehicles'],
        'robust_terminal_bound':second_report['terminal_robust_bound_vehicles'],
        'verifier_calls_total':second.verifier_calls,'fresh_calls_after_installation':sum(z['fresh_verifier_calls'] for z in second.receipts),
        'membership_checks':len(second.membership_records)}},indent=2))


if __name__=='__main__': run()
