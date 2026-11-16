"""Matched performance and figure data for the certified 600-veh/h contract."""
from pathlib import Path
import hashlib, json, sys
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'congested_recovery_2026-11-10'
sys.path.insert(0, str(OLD))
import evaluate

OUT = HERE / 'consolidated_recovery'
OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    jobs = json.loads((HERE / 'arrival_extension_protocol.json').read_text())['jobs']
    plans = {'proposal': json.loads((OLD / 'results/proposal_planning.json').read_text())['commands']}
    for name, weight in [('repair', .65), ('backup', 1.)]:
        plans[name] = next(j['commands'] for j in jobs if j['w'] == 600 and j['weight'] == weight)
    summaries = {}
    for name, commands in plans.items():
        en = evaluate.ensemble(commands, d=25, w=600, step_seconds=2)
        tr = evaluate.trace(commands, d=25, w=600)
        evaluate.save_trace(OUT / (name + '_trace.csv'), tr)
        adverse = next(r for r in en['realizations'] if r['corner'] == '+++++++' and r['profile'] == 'constant_high')
        errors = {key: float(tr[-1, col] - adverse[key]) for key, col in
                  [('tts', 8), ('queue_time', 9), ('departures', 10), ('priority_cost', 11)]}
        errors.update(final_n1=float(100*tr[-1, 1]-adverse['final_n1']),
                      final_n2=float(100*tr[-1, 2]-adverse['final_n2']),
                      final_queue=float(100*tr[-1, 7]-adverse['final_queue']))
        assert max(abs(v) for v in errors.values()) < .1, errors
        assert en['max_conservation_error'] < 1e-7
        if name != 'proposal':
            assert en['physical_passes'] == en['service_passes'] == 640
        en['independent_DOP853_minus_RK4'] = errors
        en['commands'] = commands
        (OUT / (name+'_ensemble.json')).write_text(json.dumps(en, indent=2)+'\n')
        summaries[name] = {k:v for k,v in en.items() if k != 'realizations'}
        print(json.dumps(dict(plan=name,physical_passes=en['physical_passes'],service_passes=en['service_passes'],
                              peak_queue=en['statistics']['peak_queue']['max'],
                              tts=en['statistics']['tts']['median'],jd=en['statistics']['priority_cost']['median'])), flush=True)
    def med(plan, measure): return summaries[plan]['statistics'][measure]['median']
    changes = dict(priority_cost_reduction_percent=100*(1-med('repair','priority_cost')/med('backup','priority_cost')),
                   tts_increase_percent=100*(med('repair','tts')/med('backup','tts')-1),
                   recovered_benefit_percent=100*(med('backup','priority_cost')-med('repair','priority_cost')) /
                       (med('backup','priority_cost')-med('proposal','priority_cost')))
    bounds = json.loads((HERE/'bound_summary.json').read_text())
    for name, weight in [('repair', .65), ('backup', 1.)]:
        b = bounds[f'arrival_w600_lambda{weight}']['bounds']
        s = summaries[name]['statistics']
        assert s['peak_queue']['max'] <= b['queue_upper'] + 1e-6
        summaries[name]['certified'] = b
        summaries[name]['regional_width_to_sampled_spread'] = [b['width'][i] / (s[k]['max'] - s[k]['min'])
                                                              for i,k in enumerate(['final_n1','final_n2'])]
    output = dict(protocol=dict(d=25,queue_error=10,arrival_error=600,repair_weight=.65,realizations=640,
                               integration_step_seconds=2,independent_integrator='DOP853',
                               costs='medians over the same 128 corners times five arrival profiles'),
                  plans=summaries, changes=changes,
                  source_hashes={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [OLD/'evaluate.py', OLD/'run_study.py', HERE/'arrival_extension_protocol.json']})
    (OUT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(changes),flush=True)
