"""Generate the manuscript's handover results from every completed case.

All prose templates are fixed here; only calculated counts and ranges change.
No outcomes are selected or removed. The script also validates contract binding.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import numpy as np
import csv

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
DEST=ROOT/'output/package/PERIMETER_BENCHMARK_REVISION_2026-09-26'

def walk(obj):
    if isinstance(obj,dict):
        if 'proof_id' in obj and 'proof_descriptor' in obj:
            yield obj
            return
        for v in obj.values():yield from walk(v)
    elif isinstance(obj,list):
        for v in obj:yield from walk(v)

def main():
    files=sorted((HERE/'results/delayed_handover').glob('D*.json'))
    assert len(files)==51,len(files)
    records=[json.loads(p.read_text()) for p in files]
    assert len(set(r['contract']['id'] for r in records))==17
    assert all(sum(r['delay_seconds']==d for r in records)==17 for d in (60,120,300))
    proofs=0; membership=0; source_cache={}; rows=[]
    for r in records:
        assert r['summary']['physical_passes']==640 and r['summary']['service_passes']==640
        assert r['independent_audit']['passed']
        assert r['adaptive_adverse_replay']['physical_pass'] and r['adaptive_adverse_replay']['service_pass']
        assert abs(r['adaptive_adverse_replay']['conservation_residual_vehicles'])<1e-6
        assert len(r['realizations'])==640
        membership+=len(r['realizations'])
        assert min(v['activation_box_membership_margin_units'] for v in r['realizations'])>=-1e-7
        now=Fraction(r['activation_time_fraction'])
        if r['replacement_installed']:
            c=r['installed_certificate']
            assert c['accepted'] and r['preparation_seconds']<=r['delay_seconds']
            assert c['adapter_audit']['commands']==r['installed_commands']
            descriptor=c['proof_descriptor']
            assert Fraction(descriptor['initial_time'])==now
            assert now+sum(Fraction(v) for v in descriptor['durations'])==Fraction(1)
            assert np.array_equal(descriptor['initial_state'],r['activation_center_units'])
            widths=np.asarray(r['activation_width_units']);generators=np.asarray(descriptor['signed_generator_method']['initial']['generators'])
            assert widths.shape==(7,) and generators.shape==(7,7)
        for c in walk(r):
            assert hashlib.sha256(json.dumps(c['proof_descriptor'],sort_keys=True).encode()).hexdigest()==c['proof_id']
            assert c['adapter_audit']['proof_hash_recomputed']
            assert all(v['passed'] for v in c['adapter_audit']['exact_actuator_checks'])
            for relative,expected in c['proof_descriptor']['signed_generator_method']['source_sha256'].items():
                source=(HERE/relative).resolve();assert source.is_relative_to(HERE)
                if relative not in source_cache:source_cache[relative]=hashlib.sha256(source.read_bytes()).hexdigest()
                assert source_cache[relative]==expected,relative
            proofs+=1
        reason=('replacement_ready' if r['replacement_installed'] else 'late_result' if r['late_result'] else
                'activation_box_not_admissible' if 'replacement_information_rejected' in r else 'fresh_proof_unavailable')
        rows.append(dict(contract=r['contract']['id'],delay_seconds=r['delay_seconds'],replacement=r['replacement_installed'],
            theta=r['selected_theta'],reason=reason,preparation_seconds=r['preparation_seconds'],
            maximum_activation_half_width_vehicles=max(r['box_audit']['width_vehicles']),
            physical_passes=640,service_passes=640,median_tts_veh_h=r['summary']['tts_veh_h']['median']))
    groups=[]
    for d in (60,120,300):
        g=[r for r in records if r['delay_seconds']==d]
        groups.append(dict(delay_seconds=d,cases=17,installed=sum(r['replacement_installed'] for r in g),
            retained=sum(not r['replacement_installed'] for r in g),late=sum(r['late_result'] for r in g),
            physically_admissible_activation_box_failed=sum('replacement_information_rejected' in r for r in g),
            maximum_preparation_seconds=max(r['preparation_seconds'] for r in g),
            physical_passes=sum(r['summary']['physical_passes'] for r in g),
            service_passes=sum(r['summary']['service_passes'] for r in g)))
    summary=dict(cases=51,contracts=17,realizations=32640,proof_records_checked=proofs,
        activation_membership_checks=membership,adaptive_replays=51,all_physical_and_service_pass=True,
        installed=sum(r['replacement_installed'] for r in records),groups=groups,rows=rows,
        scope='whole propagated information set, no new observation; host-specific planned-slot timing; original deadline unchanged')
    (HERE/'results/delayed_handover_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (HERE/'results/DELAYED_HANDOVER_CASES.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table='\n'.join(f"{g['delay_seconds']//60} & {g['installed']}/17 & {g['retained']}/17 & {g['late']}/17 & {g['physical_passes']}/10880\\\\" for g in groups)
    installed=summary['installed']; retained=51-installed
    tex=rf'''\begin{{table}}[t]
\centering\small
\caption{{Planned-time handover: seventeen contracts per slot, with 640 original-information realizations per contract. A late verdict leaves the incumbent in force. The last column requires both physical and service satisfaction.}}
\label{{tab:delayed-handover}}
\begin{{tabular}}{{@{{}}rrrrr@{{}}}}
\toprule
Slot delay & Replacement & Retention & Late result & Feasible runs\\
(min) & cases & cases & cases & physical/service\\
\midrule
{table}
\bottomrule
\end{{tabular}}
\end{{table}}

Across the 51 cases, {installed} replacements are certified and completed
within their activation budget; {retained} cases retain the original
authorized schedule. All 32,640 trajectories satisfy the physical and
minute-60 service rows. Each installed proof uses the propagated activation
box, future absolute command times and unchanged deadline. Independent
adaptive replays verify all 51 adverse trajectories and their membership
in the activation boxes.

Increasing the preparation slot gives additional compute time while also
changing the future information set and the remaining service horizon.
The wider propagated boxes can include physically inadmissible class
combinations, which prevents a new box-based proof. In those cases the
incumbent remains authorized by its original correlated certificate.
The test therefore evaluates scheduled replacement and certified retention
together; slot length, information-set representation and available
computation jointly determine which replacement can take effect.
'''
    (DEST/'delayed_handover_results.tex').write_text(tex,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
