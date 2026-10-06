"""Prospectively frozen operating-library comparison after discovery.

Three distinct target pairs selected from the recorded64candidate commissioning
sweep. The original high-target library remains independently archived. No model
parameter, physical limit or operator deadline is changed to make a run pass.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import mfd_incident_model as model
import mfd_incident_library_design as design
import mfd_incident_library_study as study

RECORD=model.ROOT/'reproducibility/results/mfd_incident_commissioned_library_v2.json'
CHOICES=((20.,22.,72.,3000.),(22.,24.,78.,3000.),(24.,26.,84.,3000.))
DEVELOPMENT_CASES=((.2,.5),(.2,1.),(.3,.5),(.3,1.),(.4,.5),(.4,1.))
HOLDOUT_CASES=((.25,.75),(.35,.75))


def installed_library(payload):
    """Load the commissioned binary coefficients without repeating LQR synthesis."""
    modes=[]
    for record in payload['controller_library']:
        arrays=[np.asarray(record[key],dtype=float) for key in ('center','feedforward','gain','A','P')]
        if [a.shape for a in arrays]!=[(7,),(4,),(4,7),(7,7),(7,7)] or not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError('invalid installed controller dimensions or coefficients')
        modes.append(model.Mode(record['name'],*arrays))
    return modes


def library():
    if RECORD.exists():return installed_library(json.loads(RECORD.read_text()))
    return [design.synthesize(*choice) for choice in CHOICES]


def commission():
    if RECORD.exists():
        payload=json.loads(RECORD.read_text())
        expected=dict(choices=[list(c) for c in CHOICES],
                      fixed_common_physical_demand=model.INTERNAL_DEMAND.tolist(),
                      operator_service=dict(time_h=study.SERVICE_TIME,n1_upper=study.SERVICE_UPPER),
                      development_cases=[list(c) for c in DEVELOPMENT_CASES],
                      holdout_cases=[list(c) for c in HOLDOUT_CASES],
                      uncertainty=dict(offered_arrival_residual_units_per_hour=.6,initial_state_box_units=.01))
        if any(payload.get(key)!=value for key,value in expected.items()):
            raise RuntimeError('commissioned protocol changed; create a new version explicitly')
        modes=installed_library(payload)
        if len(modes)!=len(CHOICES):raise ValueError('installed library count changed')
        text=json.dumps(payload,sort_keys=True,indent=2)+'\n'
        return modes,hashlib.sha256(text.encode()).hexdigest()
    modes=library()
    payload=dict(selection='three lowest-cost distinct target pairs among sampled-feasible primary commissioning candidates',
                 discovery_source='mfd_incident_library_design_results.json',choices=CHOICES,
                 fixed_common_physical_demand=model.INTERNAL_DEMAND.tolist(),
                 operator_service=dict(time_h=study.SERVICE_TIME,n1_upper=study.SERVICE_UPPER),
                 controller_library=[mode.record() for mode in modes],
                 development_cases=DEVELOPMENT_CASES,holdout_cases=HOLDOUT_CASES,
                 holdout_scope='validation combinations excluded from commissioning cost selection; already evaluated with superseded v1 library',
                 revision='v2 corrects a mistaken third-distinct-target ranking in v1; all v1 records retained',
                 uncertainty=dict(offered_arrival_residual_units_per_hour=.6,initial_state_box_units=.01),
                 provenance='reference arrivals are controller design parameters; actual arrival is common to all policies')
    text=json.dumps(payload,sort_keys=True,indent=2)+'\n'
    if RECORD.exists():
        if json.loads(RECORD.read_text())!=json.loads(text):raise RuntimeError('commissioned protocol changed; create a new version explicitly')
    else:RECORD.write_text(text)
    return modes,hashlib.sha256(text.encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--certified',action='store_true')
    parser.add_argument('--primary',action='store_true');parser.add_argument('--holdout',action='store_true')
    parser.add_argument('--verify-step-seconds',type=float,default=15.)
    args=parser.parse_args();modes,protocol_hash=commission()
    cases=((.3,1.),) if args.primary else HOLDOUT_CASES if args.holdout else DEVELOPMENT_CASES+HOLDOUT_CASES
    out=model.ROOT/f'reproducibility/results/mfd_incident_operational_v2_{"certified" if args.certified else "nominal"}_{"primary" if args.primary else "holdout" if args.holdout else "grid"}_results.json'
    data=dict(commissioned_protocol_sha256=protocol_hash,runs=[])
    for amplitude,duration in cases:
        started=time.perf_counter()
        run=study.simulate(amplitude,duration,certified=args.certified,library=modes,
                           verify_step_seconds=args.verify_step_seconds)
        run['held_out']=bool((amplitude,duration) in HOLDOUT_CASES)
        data['runs'].append(run)
        out.write_text(json.dumps(data,indent=2)+'\n')
        print({key:run[key] for key in ('amplitude','duration_h','total_time_spent_veh_h','service_n1_vehicles',
                                      'maximum_queue_vehicles','changes','fallbacks','full_horizon_completed',
                                      'all_decisions_certified','rejected_certificates')},'seconds',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
