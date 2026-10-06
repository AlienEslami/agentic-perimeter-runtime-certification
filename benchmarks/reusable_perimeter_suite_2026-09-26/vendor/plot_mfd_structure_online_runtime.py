"""Audited online service trajectories and matched nominal TTS increments.

All three certificates at commissioning target1600 have identical trajectories.
The affine1625 case is a different commissioning design for the SAME1700 task.
Both cumulative cost differences use their own matched semantic-request replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'reproducibility/results'
AUDIT=RESULTS/'mfd_structure_online_runtime_audit_results.json'
SOURCE=RESULTS/'mfd_structure_online_runtime_results.json'
PREFIX=RESULTS/'mfd_structure_online_runtime'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run():
    audit,source=load(AUDIT),load(SOURCE)
    assert audit['passed'] and audit['complete'] and source['completed']
    assert audit['audited_cases']==audit['expected_cases']==17
    assert audit['audited_source_result_sha256']==sha(SOURCE)
    records={r['case_id']:r for r in audit['runs']}
    requested=['binding1600_original','binding1600_positive_queue','binding1600_affine_v3',
               'binding1625_affine_v3','nominal_1700_c1600','nominal_1700_c1625']
    cases={}
    hashes={AUDIT.relative_to(ROOT).as_posix():sha(AUDIT),SOURCE.relative_to(ROOT).as_posix():sha(SOURCE)}
    for name in requested:
        record=records[name]
        path=ROOT/record['run_file']
        assert sha(path)==record['run_sha256']
        hashes[record['run_file']]=record['run_sha256']
        case=load(path)
        assert case['metrics']['complete'] and case['specification']['service']==1700.
        assert case['specification']['realization']=='nominal'
        assert case['request_mode']=='ARCHIVED_SEMANTIC_REPLAY'
        cases[name]=case
    reference=cases['binding1600_original']
    assert all(reference['trajectory']==cases[name]['trajectory'] for name in requested[1:3])
    nominal_difference=100*float(np.max(np.abs(np.asarray(cases['nominal_1700_c1600']['trajectory']['state'])-
                                             cases['nominal_1700_c1625']['trajectory']['state'])))
    assert nominal_difference<1e-7
    data=dict(scope='nominal-realization online semantic-request replay; independent adaptive checkpoint values',
        plot_source_sha256=sha(Path(__file__)),input_sha256=hashes,
        trusted_service_limit_vehicles=1700.,deadline_minutes=60.,
        initial_information_half_widths_vehicles=[1.]*7,residual_bound_vehicles_per_hour=60.,
        commissioning1600_three_certificate_trajectories_exactly_equal=True,
        matched_nominal_replay_maximum_state_difference_vehicles=nominal_difference,
        interpretation='The1625 plan is a different commissioning design, not a matched-command superiority comparison',
        curves={})
    for name in requested:
        rec=records[name]['independent_replay']
        case=cases[name]
        data['curves'][name]=dict(commissioning_target_vehicles=case['specification']['commissioning'],
            trajectory_minutes=(60*np.asarray(case['trajectory']['time_h'])).tolist(),
            trajectory_n1_vehicles=(100*np.asarray(case['trajectory']['state'])[:,0]).tolist(),
            checkpoint_minutes=[0.]+[60*float(__import__('fractions').Fraction(r['exact_time'])) for r in rec['checkpoints']],
            checkpoint_n1_vehicles=[100*case['actual_initial_state'][0]]+[100*r['state'][0] for r in rec['checkpoints']],
            cumulative_TTS_veh_h=[0.]+[r['TTS_veh_h'] for r in rec['checkpoints']],
            total_TTS_veh_h=rec['total_time_spent_veh_h'],deadline_n1_vehicles=100*rec['final_state'][0])
    comparisons={}
    for name,nominal in [('binding1600_original','nominal_1700_c1600'),('binding1625_affine_v3','nominal_1700_c1625')]:
        left,right=data['curves'][name],data['curves'][nominal]
        assert left['checkpoint_minutes']==right['checkpoint_minutes']
        delta=np.asarray(left['cumulative_TTS_veh_h'])-right['cumulative_TTS_veh_h']
        comparisons[name]=dict(nominal_comparator=nominal,minutes=left['checkpoint_minutes'],
            cumulative_incremental_TTS_veh_h=delta.tolist(),final_incremental_TTS_veh_h=float(delta[-1]),
            final_incremental_TTS_percent=100*float(delta[-1])/right['total_TTS_veh_h'])
    data['matched_comparisons']=comparisons
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.0,4.1))
    fig.subplots_adjust(left=.09,right=.98,bottom=.26,top=.73,wspace=.30)
    navy,teal,gray='#264b7f','#138574','#757575'
    labels={
        'binding1600_original':'Original / positive / affine: target 1600',
        'binding1625_affine_v3':'Affine: target 1625',
        'nominal_1700_c1600':'Matched nominal replays (overlap)'}
    lines=[]
    for name,color,style in [('binding1600_original',navy,'-'),('binding1625_affine_v3',teal,'--'),
                              ('nominal_1700_c1600',gray,':')]:
        row=data['curves'][name]
        line,=ax.plot(row['trajectory_minutes'],row['trajectory_n1_vehicles'],
            color=color,linestyle=style,linewidth=1.65,label=labels[name])
        lines.append(line)
        ax.plot(60,row['deadline_n1_vehicles'],'o',color=color,markersize=3)
    ax.hlines(1700,52,60,colors='black',linestyles='-.',linewidth=.85)
    ax.plot(60,1700,marker='D',color='black',markersize=3)
    ax.annotate('1700 at 60 min',xy=(59.6,1700),xytext=(37,1880),fontsize=8,
        arrowprops=dict(arrowstyle='-',lw=.65,color='black'),ha='left',va='bottom')
    ax.set(xlabel='Time (min)',ylabel='Region 1 accumulation (vehicles)',xlim=(0,61),ylim=(1500,3070))
    ax.set_title('(a) Regional service',loc='left',pad=8)
    ax.set_xticks([0,15,30,45,60]);ax.set_yticks([1500,2000,2500,3000])
    for name,color,style in [('binding1600_original',navy,'-'),('binding1625_affine_v3',teal,'--')]:
        row=comparisons[name]
        bx.plot(row['minutes'],row['cumulative_incremental_TTS_veh_h'],color=color,
            linestyle=style,linewidth=1.7,marker='o',markersize=2.5)
        final=row['final_incremental_TTS_veh_h']
        bx.annotate(f'+{final:.1f}',xy=(60,final),xytext=(-3,7),textcoords='offset points',
            color=color,fontsize=9,ha='right',va='bottom')
    bx.axhline(0,color=gray,linewidth=.6)
    bx.set(xlabel='Time (min)',ylabel='Cumulative additional TTS (veh-h)',xlim=(0,61),ylim=(-5,180))
    bx.set_title('(b) Cost versus matched nominal replay',loc='left',pad=8)
    bx.set_xticks([0,15,30,45,60]);bx.set_yticks([0,50,100,150])
    for axes in (ax,bx):
        axes.grid(axis='y',color='#dddddd',linewidth=.5,alpha=.8)
        axes.set_axisbelow(True)
    fig.suptitle(r'Online service task: $100n_1(1\,\mathrm{h})\leq1700$ vehicles',y=.98,fontsize=12)
    fig.legend(handles=lines,loc='upper center',bbox_to_anchor=(.52,.89),ncol=2,frameon=False,
               columnspacing=1.4,handlelength=2.8)
    fig.text(.09,.105,'Targets 1600 and 1625 refer to commissioning plans; the latter is a separate design. The three target-1600 curves coincide.',fontsize=7.6)
    fig.text(.09,.063,'Each TTS difference uses its own matched nominal request replay under the same physical and 1700-vehicle service contracts.',fontsize=7.6)
    fig.savefig(PREFIX.with_suffix('.pdf'),metadata={'Title':'Online structure-aware service certification',
        'Subject':'Matched nominal request replay; separate1600/1625 commissioning designs'})
    fig.savefig(PREFIX.with_suffix('.png'),dpi=220)
    plt.close(fig)
    data['figure_sha256']={p.name:sha(p) for p in [PREFIX.with_suffix('.pdf'),PREFIX.with_suffix('.png')]}
    output=PREFIX.with_name(PREFIX.name+'_figuredata.json')
    output.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(figure_data_sha256=sha(output),comparisons=comparisons),indent=2))


if __name__=='__main__':
    run()
