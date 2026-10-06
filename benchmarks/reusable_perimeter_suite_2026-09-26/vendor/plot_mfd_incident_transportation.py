"""Publication figure for the primary queue-aware transportation experiment.

No raw simulation record is modified. The queue-blind curve is analytic;
certified-library and MPC curves use their stored 10-second physical traces.
Cumulative TTS is reconstructed from exactly the inventory used in the table.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import cumulative_trapezoid


HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results'
CERTIFIED=RESULTS/'mfd_incident_operational_v2_certified_primary_results.json'
NOMINAL=RESULTS/'mfd_incident_operational_v2_nominal_grid_results.json'
MPC=RESULTS/'mfd_incident_mpc_grid_results.json'
OUTPUT=RESULTS/'mfd_incident_transportation'
AMPLITUDE=.3
DURATION=1.
START=.25
RAMP=1/12
HORIZON=2.
DEADLINE=1.
TARGET_VEHICLES=2850.
STORAGE_VEHICLES=2000.


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def primary(payload):
    return next(r for r in payload['runs'] if r['amplitude']==AMPLITUDE and r['duration_h']==DURATION)


def ramp_integrals(times):
    """Exact integral and double integral of the unit-height trapezoid."""
    local=np.asarray(times)-START
    first=np.zeros_like(local)
    second=np.zeros_like(local)
    for k,value in enumerate(local):
        if value<=0:continue
        if value<RAMP:
            first[k]=value*value/(2*RAMP)
            second[k]=value**3/(6*RAMP)
        elif value<=DURATION-RAMP:
            first[k]=value-RAMP/2
            second[k]=value*value/2-RAMP*value/2+RAMP*RAMP/6
        elif value<DURATION:
            tail=DURATION-value
            left=DURATION-RAMP
            left_integral=left*left/2-RAMP*left/2+RAMP*RAMP/6
            first[k]=DURATION-RAMP-tail*tail/(2*RAMP)
            second[k]=left_integral+(DURATION-RAMP)*(value-left)+(tail**3-RAMP**3)/(6*RAMP)
        else:
            first[k]=DURATION-RAMP
            second[k]=(DURATION-RAMP)*(value-DURATION/2)
    return first,second


def analytic_incumbent(times):
    first,second=ramp_integrals(times)
    q=100*(2+60*AMPLITUDE*first)
    cumulative_queue=100*(2*times+60*AMPLITUDE*second)
    return dict(time_h=times,region1_vehicles=np.full_like(times,3000.),
                queue_vehicles=q,cumulative_tts_veh_h=6000*times+cumulative_queue,
                cumulative_queue_veh_h=cumulative_queue,
                source='exact queue-blind incumbent: fixed original equilibrium commands',
                initial_command=[.5,.5,.5,.52])


def physical_curve(times,states,source):
    times=np.asarray(times,dtype=float)
    states=np.asarray(states,dtype=float)
    assert states.shape==(len(times),7)
    assert np.all(np.diff(times)>0) and abs(times[-1]-HORIZON)<1e-9
    return dict(time_h=times,region1_vehicles=100*states[:,0],queue_vehicles=100*states[:,6],
                cumulative_tts_veh_h=100*cumulative_trapezoid(states[:,0]+states[:,1]+states[:,6],times,initial=0),
                cumulative_queue_veh_h=100*cumulative_trapezoid(states[:,6],times,initial=0),source=source)


def endpoint_record(curve):
    times=curve['time_h']
    return dict(total_time_spent_veh_h=float(curve['cumulative_tts_veh_h'][-1]),
                queue_time_veh_h=float(curve['cumulative_queue_veh_h'][-1]),
                maximum_queue_vehicles=float(max(curve['queue_vehicles'])),
                final_queue_vehicles=float(curve['queue_vehicles'][-1]),
                deadline_n1_vehicles=float(np.interp(DEADLINE,times,curve['region1_vehicles'])))


def assert_endpoint_values(actual,expected,tolerance=2e-7):
    errors={key:actual[key]-value for key,value in expected.items()}
    assert max(abs(value) for value in errors.values())<tolerance,errors
    return errors


def shade_incident(ax):
    # Alpha encodes the known trapezoid, including its five-minute ramps.
    minutes=np.linspace(0,120,1201)
    local=minutes/60-START
    shape=np.maximum(0,np.minimum(1,np.minimum(local/RAMP,(DURATION-local)/RAMP)))
    rgba=np.ones((2,len(minutes),4))
    rgba[:,:,:3]=np.array([.65,.69,.73])
    rgba[:,:,3]=.16*shape
    bottom,top=ax.get_ylim()
    ax.imshow(rgba,extent=(0,120,bottom,top),aspect='auto',origin='lower',
              interpolation='bilinear',zorder=0)


def main():
    certified=primary(load(CERTIFIED))
    nominal=primary(load(NOMINAL))
    mpc=primary(load(MPC))
    assert certified['all_decisions_certified'] and certified['full_horizon_completed']
    assert [e['executed_mode'] for e in certified['events']]==[e['executed_mode'] for e in nominal['events']]
    assert np.array_equal(np.asarray(certified['states']),np.asarray(nominal['states']))
    assert certified['total_time_spent_veh_h']==nominal['total_time_spent_veh_h']
    curves={
        'incumbent':analytic_incumbent(np.linspace(0,HORIZON,721)),
        'certified_library':physical_curve(certified['times'],certified['states'],str(CERTIFIED.relative_to(HERE.parent))),
        'mpc':physical_curve(mpc['trajectory']['time_h'],mpc['trajectory']['state'],str(MPC.relative_to(HERE.parent))),
    }
    endpoints={key:endpoint_record(curve) for key,curve in curves.items()}
    tests={
        'incumbent':assert_endpoint_values(endpoints['incumbent'],dict(total_time_spent_veh_h=14462.5,
                     queue_time_veh_h=2462.5,maximum_queue_vehicles=1850.,final_queue_vehicles=1850.,deadline_n1_vehicles=3000.)),
        'certified_library':assert_endpoint_values(endpoints['certified_library'],dict(
                     total_time_spent_veh_h=certified['total_time_spent_veh_h'],queue_time_veh_h=certified['queue_time_veh_h'],
                     maximum_queue_vehicles=certified['maximum_queue_vehicles'],final_queue_vehicles=certified['final_queue_vehicles'],
                     deadline_n1_vehicles=certified['service_n1_vehicles'])),
        'mpc':assert_endpoint_values(endpoints['mpc'],dict(total_time_spent_veh_h=mpc['metrics']['total_time_spent_veh_h'],
                     queue_time_veh_h=mpc['metrics']['queue_time_veh_h'],maximum_queue_vehicles=mpc['metrics']['maximum_queue_vehicles'],
                     final_queue_vehicles=mpc['metrics']['final_queue_vehicles'],deadline_n1_vehicles=100*mpc['metrics']['deadline_n1'])),
    }
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':12,
                         'axes.titlesize':12,'xtick.labelsize':11,'ytick.labelsize':11,
                         'legend.fontsize':11,'axes.linewidth':.75,'pdf.fonttype':42,'ps.fonttype':42,
                         'lines.linewidth':2.,'savefig.facecolor':'white'})
    styles={
        'incumbent':dict(color='#555B63',linestyle=(0,(4,2)),label='Queue-blind incumbent (analytic)'),
        'certified_library':dict(color='#0072B2',linestyle='-',label='Certified library (= nominal)'),
        'mpc':dict(color='#D55E00',linestyle='-',label='Continuous-input MPC'),
    }
    # At a 5.3-inch manuscript width, the 12-point labels print at 8.15 points.
    # Give the cumulative cost panel a full row rather than compressing three
    # panels and their labels into a single line.
    fig=plt.figure(figsize=(7.8,6.0))
    grid=fig.add_gridspec(2,2,left=.115,right=.978,bottom=.09,top=.795,
                          hspace=.62,wspace=.43,height_ratios=[1,1.05])
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1]),fig.add_subplot(grid[1,:])]
    panels=[('region1_vehicles','Region 1 accumulation (veh)','(a) Regional service',(1400,3400)),
            ('queue_vehicles','Exterior queue (veh)','(b) Boundary storage',(0,2350)),
            ('cumulative_tts_veh_h','Cumulative TTS (veh-h)','(c) Total time spent',(0,16000))]
    for ax,(field,ylabel,title,limits) in zip(axes,panels):
        ax.set_xlim(0,120);ax.set_ylim(*limits)
        shade_incident(ax)
        for key,curve in curves.items():
            ax.plot(60*curve['time_h'],curve[field],**styles[key],zorder=3)
        ax.set_xlabel('Time (min)');ax.set_ylabel(ylabel);ax.set_title(title,loc='left',pad=8,fontweight='bold')
        ax.set_xticks([0,30,60,90,120])
        ax.grid(axis='y',color='#D5D9DD',linewidth=.55,alpha=.72,zorder=1)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['bottom','left']].set_color('#697078')
        ax.tick_params(width=.6,length=3,color='#697078')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0].set_yticks([1500,2000,2500,3000])
    axes[0].plot([60,60],[1400,TARGET_VEHICLES],color='#333333',ls=(0,(2,3)),lw=.8,zorder=2)
    axes[0].plot([0,60],[TARGET_VEHICLES,TARGET_VEHICLES],color='#333333',ls=(0,(2,3)),lw=.8,zorder=2)
    axes[0].plot(60,TARGET_VEHICLES,marker='D',markersize=4,color='#333333',zorder=5)
    axes[0].annotate('2,850 veh\nat 60 min',xy=(60,TARGET_VEHICLES),xytext=(89,3190),
                     fontsize=11,ha='center',va='center',color='#333333',
                     arrowprops=dict(arrowstyle='-',color='#555555',lw=.8))
    axes[1].set_yticks([0,500,1000,1500,2000])
    axes[1].axhline(STORAGE_VEHICLES,color='#333333',ls=(0,(3,3)),lw=.85,zorder=2)
    axes[1].text(119,2080,'Storage: 2,000 veh',ha='right',va='bottom',fontsize=11,color='#333333')
    axes[2].set_yticks([0,5000,10000,15000])
    for key in curves:
        endpoint=endpoints[key]['total_time_spent_veh_h']
        if key=='mpc':
            # Keep the wider two-decimal label away from both rising curves.
            axes[2].annotate(f'{endpoint:,.2f}',xy=(120,endpoint),xytext=(118,6500),
                             color=styles[key]['color'],fontsize=11,ha='right',va='top',
                             arrowprops=dict(arrowstyle='-',color=styles[key]['color'],lw=.8))
        else:
            axes[2].text(118,endpoint+310,f'{endpoint:,.2f}',color=styles[key]['color'],
                         fontsize=11,ha='right',va='bottom')
    handles=[Line2D([0],[0],**style) for style in styles.values()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.0),ncol=2,
               frameon=False,handlelength=2.6,columnspacing=1.8,labelspacing=.5)
    fig.text(.54,.876,'Shading: 30% arrival surge, 15-75 min; five-minute ramps',
             ha='center',va='center',fontsize=11,color='#545B63')
    fig.savefig(OUTPUT.with_suffix('.pdf'))
    fig.savefig(OUTPUT.with_suffix('.png'),dpi=240)
    plt.close(fig)
    provenance=dict(scope='primary transportation figure from preserved raw records plus analytic incumbent',
                    figure_size_inches=[7.8,6.0],
                    layout='two upper panels; cumulative TTS spans the full lower row',
                    base_font_points=12,minimum_font_points=11,
                    intended_manuscript_width_inches=5.3,
                    base_font_at_manuscript_width_points=12*5.3/7.8,
                    amplitude=AMPLITUDE,duration_h=DURATION,
                    incident_start_minutes=60*START,ramp_minutes=60*RAMP,
                    service_deadline_minutes=60*DEADLINE,service_upper_vehicles=TARGET_VEHICLES,
                    exterior_storage_limit_vehicles=STORAGE_VEHICLES,
                    source_sha256={str(path.relative_to(HERE.parent)):hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in (CERTIFIED,NOMINAL,MPC)},
                    certified_equals_nominal_physical_trace=True,
                    cumulative_tts_method='analytic incumbent; cumulative trapezoid of stored 10-second full-system inventory for library and MPC',
                    endpoints=endpoints,endpoint_difference_from_source=tests,
                    table_rounding_check=dict(incumbent_tts_2dp=round(endpoints['incumbent']['total_time_spent_veh_h'],2),
                                              library_tts_2dp=round(endpoints['certified_library']['total_time_spent_veh_h'],2),
                                              mpc_tts_2dp=round(endpoints['mpc']['total_time_spent_veh_h'],2)),
                    note='Incumbent is an exact mechanism diagnostic; MPC has numerical, not robust, constraints. No additional LLM simulation.')
    OUTPUT.with_name(OUTPUT.name+'_figuredata.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(provenance['endpoints'],indent=2),flush=True)


if __name__=='__main__':main()
