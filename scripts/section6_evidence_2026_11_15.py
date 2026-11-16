from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'output/package/PERIMETER_PARTB_CONSOLIDATED_2026-11-12'
OUT = ROOT / 'output/review/section6_revision_2026_11_15'
OUT.mkdir(exist_ok=True)
ARCHIVE = ROOT / 'output/archive/PERIMETER_BEFORE_SECTION6_2026-11-15'
assert not ARCHIVE.exists(), 'Previous archive exists; inspect before rerunning.'
ARCHIVE.mkdir()
for p in [PROJECT/'main.tex', PROJECT/'main.aux', ROOT/'CURRENT_REDUCED_PAPER.txt',
          ROOT/'output/pdf/perimeter_paper_reduced_2026-11-15_sections4_5_revised.pdf']:
    shutil.copy2(p, ARCHIVE/p.name)
for name in ['main_with_section6_revised_2026_11_15.tex', 'section_6_numerical_revised_2026_11_15.tex',
             'section_6_numerical_revised_preview.pdf', 'codex_request_section6_2026_11_15.md']:
    shutil.copy2(Path('C:/Users/alien/Downloads')/name, ARCHIVE/name)

used = {}
def read(relative):
    p = ROOT / relative
    raw = p.read_bytes()
    used[relative] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

SUITE = 'benchmarks/reusable_perimeter_suite_2026-09-26/results'
with (ROOT/SUITE/'CASE_SUMMARY.csv').open() as f:
    rows = list(csv.DictReader(f))
used[SUITE+'/CASE_SUMMARY.csv'] = hashlib.sha256((ROOT/SUITE/'CASE_SUMMARY.csv').read_bytes()).hexdigest()
contracts = {r['id']: {k:r[k] for k in ['id','split','width_vehicles','residual_veh_h','deadline_vehicles','weight']} for r in rows if r['task']=='D'}
assert len(contracts) == 80
records=[]
for key,ct in sorted(contracts.items()):
    full=read(f'benchmarks/fixed_portfolio_2026-11-10/results_final/full_{key}.json')
    nominal=read(f'benchmarks/fixed_portfolio_2026-11-10/results_final/nominal_{key}.json')
    data=read(f'{SUITE}/contracts/{key}.json')
    assert float(ct['width_vehicles']) == data['contract']['width_vehicles']
    assert full['job']['width'] == [float(ct['width_vehicles'])/100]*7
    assert full['job']['w'] == float(ct['residual_veh_h'])
    assert full['job']['deadline'] == float(ct['deadline_vehicles'])
    assert full['job']['old_accepted'] == full['attempts'][0]['passed']
    records.append(dict(id=key,split=ct['split'],d=float(ct['width_vehicles']),w=float(ct['residual_veh_h']),
        eta=float(ct['deadline_vehicles']),p=float(ct['weight']),
        nominal_signed=nominal['job']['old_accepted'],nominal_sequential=nominal['summary']['accepted'],
        five_signed=data['policies']['scenario_mpc']['certificate']['accepted'],
        full_signed=full['job']['old_accepted'],full_sequential=full['summary']['accepted'],winner=full['summary']['winner']))
counts={}
for split in ['development','held_out','all']:
    rr=[r for r in records if split=='all' or r['split']==split]
    counts[split]={'total':len(rr),**{k:sum(r[k] for r in rr) for k in ['nominal_signed','nominal_sequential','five_signed','full_signed','full_sequential']}}
assert counts['all'] == dict(total=80,nominal_signed=36,nominal_sequential=36,five_signed=40,full_signed=44,full_sequential=50)
assert counts['held_out']['total']==8 and counts['held_out']['full_signed']==8
assert all(r['split']=='development' for r in records if r['full_sequential'] and not r['full_signed'])

old=read('benchmarks/section5_support_2026-11-11/summaries/congested_audit_and_summary.json')
refined=read('benchmarks/reviewer_strengthening_2026-11-11/four_box_bound_summary.json')
timings=read('benchmarks/reviewer_strengthening_2026-11-11/timing_summary.json')
widths=[]
for plan in ['backup','repair']:
    samples=[old['terminal_spreads'][plan][k]['sampled_width'] for k in ['n1','n2']]
    for name,variant in [('Standard','baseline'),('Refined','tight12')]:
        values=[old['terminal_spreads'][plan][k]['certified_outer_width'] for k in ['n1','n2']] if variant=='baseline' else refined[plan]['width'][:2]
        ts=[t['seconds'] for t in timings if t['id']==f'depth_{plan}_4500' and t['variant']==variant]
        assert len(ts)==2
        widths.append(dict(plan=plan,implementation=name,width=values,sampled=samples,
                           ratio=[a/b for a,b in zip(values,samples)],seconds=ts,
                           acceptance_boxes=4 if variant=='baseline' else 1))

repair=read(SUITE+'/contracts/D027.json')['policies']['grid_repair']['certificate']
snapshot=read(SUITE+'/supervisory_vehicle_flows_2026-11-08/validation.json')
queue_sim=snapshot['policies']['grid_repair']['peak_queue_vehicles']
queue_bound=(20-repair['continuous_constraint_margins_lower']['exterior_queue_upper'])*100
assert queue_sim<=queue_bound+1e-8
witnesses=read('benchmarks/reviewer_strengthening_2026-11-11/queue_witnesses_w600.json')['rows']
assert witnesses[0]['weight']==0 and all(r['lower_vehicles_rounded_down']>2000 for r in witnesses if r['weight']<.65)

ring=read('output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK/reproducibility/results/mfd_multiregion_transfer_results.json')
ring_cases=[]
for n,w in [(2,300),(3,60)]:
    rr=next(r for r in ring['cases'] if r['regions']==n and r['width_vehicles']==25 and r['residual_veh_h']==w and r['method']=='inventory')
    assert rr['accepted']
    ring_cases.append(dict(regions=n,d=25,w=w,upper=max(r['upper_vehicles'] for r in rr['terminal_service_rows'])))

delay=read(SUITE+'/delayed_handover_summary.json')
assert delay['realizations']==32640 and delay['all_physical_and_service_pass']
assert [r['installed'] for r in delay['groups']]==[8,11,11]
structure=read('benchmarks/manuscript_review_2026-11-08/results/coordinate_comparison.json')['rows']
assert max(r['d'] for r in structure if r['method']=='signed' and r['accepted'])==50
assert max(r['d'] for r in structure if r['method'] in ['physical_box','class_box','conservation_box'] and r['accepted'])==10
summary=read(SUITE+'/summary.json')
assert summary['deadline_all']['modified_proposals']==38
assert summary['deadline_all']['certified_repair_available']==74
congested=read('benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery/summary.json')
assert congested['protocol']['arrival_error']==600 and congested['protocol']['repair_weight']==.65
assert congested['plans']['proposal']['physical_passes']==320
assert congested['plans']['repair']['physical_passes']==640

result=dict(counts=counts,acceptance_map=records,widths_and_times=widths,
            repaired_queue=dict(simulated=queue_sim,certified=queue_bound),ring_cases=ring_cases,
            source_sha256=used,repository=dict(private=True,release='paper-2026-11-11',release_exists=True),
            simulations_rerun=False)
(OUT/'verified_evidence.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,4,figsize=(12,6.5),sharex=True,sharey=True)
panels=[(2750,2),(2850,2),(2950,2),(2800,3),(2750,4),(2850,4),(2950,4),(2900,3)]
for ax,(eta,p) in zip(axes.flat,panels):
    rr=[r for r in records if r['eta']==eta and r['p']==p]
    held=all(r['split']=='held_out' for r in rr)
    for r in rr:
        marker,color=('o','#007a80') if r['full_signed'] else (('^','#bb682b') if r['full_sequential'] else ('x','#768390'))
        ax.scatter(r['d'],r['w'],marker=marker,c=color,s=65,zorder=3,linewidths=1.5)
        if r['split']=='held_out':ax.scatter(r['d'],r['w'],marker='o',facecolors='none',edgecolors='#202c36',s=160,linewidths=.9,zorder=4)
    ax.set_title(rf'$\eta={eta},\ p={p}$'+('\nHeld-out settings' if held else ''),fontsize=10)
    ax.set(xlim=(-2,54),ylim=(35,325),xticks=[1,10,25,50],yticks=[60,150,300])
    if held:
        ax.set_facecolor('#f5f7f8')
        for d in [5,20]:ax.axvline(d,c='#a8b2b9',ls=':',lw=.65,zorder=0)
        for w in [100,225]:ax.axhline(w,c='#a8b2b9',ls=':',lw=.65,zorder=0)
    ax.grid(alpha=.18)
for ax in axes[1]:ax.set_xlabel(r'$d$ (vehicles)')
for ax in axes[:,0]:ax.set_ylabel(r'$\bar w$ (veh/h)')
handles=[Line2D([],[],marker='o',ls='',color='#007a80',label='Certified: signed conservation'),
         Line2D([],[],marker='^',ls='',color='#bb682b',label='Certified only by sequential procedure'),
         Line2D([],[],marker='x',ls='',color='#768390',label='Inconclusive'),
         Line2D([],[],marker='o',mfc='none',ls='',color='#202c36',ms=10,label='Held-out case')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2,frameon=False,fontsize=9)
fig.tight_layout(rect=(0,.12,1,1),h_pad=1.7,w_pad=1.5)
fig.savefig(PROJECT/'figures/acceptance_map.pdf',bbox_inches='tight',pad_inches=.08)
fig.savefig(OUT/'acceptance_map.png',dpi=150,bbox_inches='tight',pad_inches=.08)
plt.close(fig)
print(json.dumps({k:v for k,v in result.items() if k not in ['acceptance_map','source_sha256']},indent=2))
