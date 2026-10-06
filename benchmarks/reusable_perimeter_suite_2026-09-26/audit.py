"""Independent physical-model, gradient, replay and certificate audits."""
import json,itertools
import numpy as np
from common import *
from run_regulation import audit as native_audit

def initial_audit():
    rng=np.random.default_rng(92926); maximum=0.
    for _ in range(30):
        x=np.asarray(fixture()["center"])+rng.uniform(-.2,.2,7)
        u=rng.uniform(.1,.9,4); offer=rng.uniform(55,90)
        f,dep=batch_field(x[None,:],u,offer)
        maximum=max(maximum,float(max(abs(f[0]-plant.field(x,u,offer)))),abs(float(dep[0])-plant.departure_rate(x,u)))
    assert maximum<1e-10
    derivatives=mpc.derivative_audit(); native=native_audit()
    return dict(vectorized_model_max_error=maximum,seven_state_derivatives=derivatives,native_derivatives=native,
        exact_compilation_checks=[exact_command_constraints(row) for row in compile_commands(fixture()["backup_commands"])[0]])

def replay_audit():
    results=[]
    paths=sorted((HERE/"results/contracts").glob("*.json"))
    full_paths=sorted((HERE/"results/full_scenario").glob("*.json"))
    reserve_paths=sorted((HERE/"results/reserve_refresh").glob("*.json"))
    # Every eighth contract plus the primary inherited contract; all policies there.
    selected=paths[::8]+[p for p in paths if p.stem=="D027"]+full_paths[::8]+[p for p in full_paths if p.stem=="D027"]+reserve_paths
    for path in dict.fromkeys(selected):
        record=json.loads(path.read_text());c=record["contract"]
        for name,policy in record.get("policies",{}).items():
            assert policy["certificate"]["adapter_audit"]["commands"]==policy["commands"]
            assert policy["certificate"]["adapter_audit"]["proof_hash_recomputed"]
            for profile in ("constant_high","alternating_5min"):
                exact=adaptive_replay(np.asarray(policy["commands"]),c,profile=profile)
                sampled=next(r for r in policy["realizations"] if r["corner"]=="+++++++" and r["profile"]==profile)
                errors={key:abs(exact[key]-sampled[key]) for key in ("tts_veh_h","queue_veh_h","completed_trips",
                         "priority_cost","deadline_n1_vehicles","peak_queue_vehicles")}
                assert max(errors.values())<.08,(path.name,name,profile,errors)
                assert abs(exact["conservation_residual_vehicles"])<1e-6
                if policy["certificate"]["accepted"]:
                    assert exact["minimum_state_margin"]>=-1e-7
                    if c["deadline_vehicles"] is not None:
                        assert exact["deadline_n1_vehicles"]<=c["deadline_vehicles"]+1e-6
                results.append(dict(case=c["id"],policy=name,profile=profile,errors=errors,adaptive=exact))
    return results

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--replay",action="store_true");args=parser.parse_args()
    if args.replay:
        record=dict(passed=True,replays=replay_audit());save(HERE/"results/independent_replay_audit.json",record)
        print("Independent adaptive replays passed:",len(record["replays"]))
    else:
        record=initial_audit();save(HERE/"results/initial_audit.json",record);print(json.dumps(record,indent=2))
