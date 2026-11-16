"""Plot saved results and reconstruct only the archived nominal predictor."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'output/package/PERIMETER_PARTB_PRE_SIMULATION_2026-11-10'
OUT = ROOT / 'output/review/writing_revision_2026_11_12'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
    'axes.spines.top':False, 'axes.spines.right':False,
    'pdf.fonttype':42, 'ps.fonttype':42})
BLUE, TEAL, RED, GRAY = '#265a83', '#007a80', '#bb482b', '#768390'
evidence = {}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save(fig, name):
    fig.savefig(PROJECT/'figures'/(name+'.pdf'), bbox_inches='tight', pad_inches=.08)
    fig.savefig(OUT/(name+'.png'), dpi=165, bbox_inches='tight', pad_inches=.08)
    plt.close(fig)


# The predictor is evaluated by the same interval arithmetic as the saved run.
# This does not rerun the certificate, optimize a plan, or simulate a trajectory.
sys.path.insert(0, str(ROOT/'benchmarks/reviewer_strengthening_2026-11-11'))
import probe
import mfd_incident_affine_reachability as af
from mfd_incident_mpc_certification import constant_command_mode
from mfd_incident_verifier import Pulse

path = ROOT/'benchmarks/reviewer_strengthening_2026-11-11/full_certificates/arrival_w600_lambda0.65.json'
saved = read(path)
c = saved['certificate']
assert c['accepted'] and c['verified_continuous_physical'] and c['verified_scheduled_service']
steps = c['interval_steps'][-3:]
start = max((r for r in c['reachable_checkpoints'] if r['time_h'] <= steps[0]['left_h']), key=lambda r:r['time_h'])
center = np.array(start['center'])
pulse = Pulse(*[Fraction(v) for v in c['proof_descriptor']['pulse']])
modes = [constant_command_mode(np.array(u),f'constant_block_{i}') for i,u in enumerate(saved['job']['commands'])]
L = np.array(start['metric_L'])
inverse = np.linalg.inv(L)
assert np.array_equal(L@inverse,np.eye(7))
draw = []
checkpoint_errors = []
for step in c['interval_steps']:
    if step['left_h'] < start['time_h'] - 1e-12:
        continue
    left = Fraction(step['left_h']).limit_denominator(2400000)
    right = Fraction(step['right_h']).limit_denominator(2400000)
    assert right > left
    mode = modes[step['segment']]
    h = af.interval_fraction(right-left)
    arrival0, arrival_slope = pulse.affine(left,right)
    branch,_ = af.branch_for_box(af.I(center),mode)
    assert branch == 'gate'
    values = af.closed_field([af.I(value) for value in center],mode,arrival0,branch)
    fbox = af.I(np.array([v.lo for v in values]),np.array([v.hi for v in values]))
    velocity = fbox.mid()
    acceleration_box = probe.vector_imat(af.jacobian(af.I(center),mode,branch),fbox)
    slope_row = np.zeros(7); slope_row[6] = 1.
    acceleration = ((acceleration_box+af.I(slope_row)*arrival_slope)*af.I(.5)).mid()
    endpoint = af.I(center)+af.I(velocity)*h+af.I(acceleration)*h*h
    next_center = endpoint.mid()
    if step['left_h'] >= steps[0]['left_h']-1e-12:
        local = np.linspace(0,float(right-left),121)
        predictor = 100*(center[0]+velocity[0]*local+acceleration[0]*local**2)
        trial = float(probe.vector_imat(af.I(np.abs(inverse)),af.I(step['apriori_radius'])).hi[0])*100
        draw.append({'minutes':60*(float(left)+local),'predictor':predictor,
                     'trial':trial,'left':60*float(left),'right':60*float(right)})
    center = next_center
    for cp in c['reachable_checkpoints']:
        if abs(cp['time_h']-float(right)) < 1e-12:
            error = np.max(np.abs(center-np.array(cp['center'])))*100
            checkpoint_errors.append(float(error))
            assert error < 1e-8, (right,error)
assert len(draw)==3
final = c['reachable_checkpoints'][-1]
endpoint_center = 100*final['center'][0]
endpoint_radius = 100*float(probe.vector_imat(af.I(np.abs(inverse)),af.I(final['radius'])).hi[0])
stored_lower,stored_upper = np.array(c['terminal_coordinate_intervals']['n1'])*100
assert abs(endpoint_center-endpoint_radius-stored_lower)<1e-8
assert abs(endpoint_center+endpoint_radius-stored_upper)<1e-8
assert endpoint_center+endpoint_radius<3300
for row in draw:
    assert min(row['predictor']-row['trial'])>0

fig,ax = plt.subplots(figsize=(10.0,3.8))
for row in draw:
    t,p,r=row['minutes'],row['predictor'],row['trial']
    ax.fill_between(t,p-r,p+r,color=BLUE,alpha=.18,zorder=1)
    ax.plot(t,p+r,color=BLUE,lw=.8,alpha=.8)
    ax.plot(t,p-r,color=BLUE,lw=.8,alpha=.8)
    ax.plot(t,p,color=BLUE,lw=1.8,zorder=3)
for x in [59.25,59.5,59.75,60]:
    ax.axvline(x,color='#a1adb7',ls=':',lw=.8,zorder=0)
ax.axhline(0,color='#35434f',ls='--',lw=1)
ax.text(59.27,115,r'Physical limit: $n_1(t)\geq0$',fontsize=10,color='#35434f')
ax.errorbar([60],[endpoint_center],yerr=[[endpoint_radius],[endpoint_radius]],fmt='o',
            color=TEAL,ecolor=TEAL,capsize=5,elinewidth=2,ms=4,zorder=5)
ax.plot([60],[3300],marker='D',ms=6,mfc='white',mec=TEAL,mew=1.5,zorder=6)
ax.annotate(r'Service limit $\eta_r=3300$ at $T_r$',xy=(60,3300),xytext=(59.63,3160),
            fontsize=10,color=TEAL,arrowprops=dict(arrowstyle='-',color=TEAL,lw=.9))
ax.annotate('Endpoint bound',xy=(60,stored_upper),xytext=(59.84,2660),ha='center',
            fontsize=9,color=TEAL,arrowprops=dict(arrowstyle='-',color=TEAL,lw=.8))
ax.set(xlim=(59.23,60.06),ylim=(-70,3500),ylabel='Region 1 accumulation (vehicles)',xlabel='Time (minutes)')
ax.set_xticks([59.25,59.5,59.75,60],[r'59.25'+'\n'+r'$t_l$',r'59.50'+'\n'+r'$t_{l+1}$',r'59.75'+'\n'+r'$t_{l+2}$',r'60.00'+'\n'+r'$T_r$'])
ax.set_yticks([0,1000,2000,3000])
ax.grid(axis='y',alpha=.12)
ax.legend(handles=[Line2D([],[],color=BLUE,lw=1.8,label=r'Predictor $p_{1l}(t)$'),
                   Patch(facecolor=BLUE,alpha=.18,label='Projected trial enclosure')],
          loc='upper left',fontsize=9,frameon=False)
fig.tight_layout()
save(fig,'verification_enclosure_2026_11_12')
evidence['verification_figure']={'source':str(path.relative_to(ROOT)), 'sha256':sha(path),
    'proof_id':c['proof_id'],'intervals_minutes':[[r['left'],r['right']] for r in draw],
    'predictor_checkpoint_max_error_vehicles':max(checkpoint_errors),
    'endpoint_center_vehicles':endpoint_center,'endpoint_radius_vehicles':endpoint_radius,
    'endpoint_lower_vehicles':stored_lower,'endpoint_upper_vehicles':stored_upper,
    'service_limit_vehicles':3300,'physical_limit':'n1 >= 0',
    'method':'Projection of stored trial and endpoint boxes; predictor reconstructed using the saved-run interval arithmetic. No new simulation or certificate.'}


# Rename comparison labels only. All plotted values and pass/fail flags are saved data.
path = ROOT/'benchmarks/manuscript_review_2026-11-08/results/coordinate_comparison.json'
rows=read(path)['rows']
methods=['physical_box','class_box','conservation_box','positive_queue','signed']
labels=['State intervals','Destination intervals','Conservation intervals','Nonnegativity/queue','Signed conservation']
colors=[GRAY,'#ab5d30',BLUE,'#7971a6',TEAL]
fig,axs=plt.subplots(1,2,figsize=(11.1,4.0),gridspec_kw={'width_ratios':[1.2,1]})
for m,label,color in zip(methods,labels,colors):
    rr=sorted([r for r in rows if r['method']==m and r['accepted']],key=lambda r:r['d'])
    if rr:
        axs[0].plot([r['d'] for r in rr],[r['deadline_bound_vehicles'] for r in rr],'-o',c=color,label=label,ms=4)
axs[0].axhline(2850,c='#29353e',ls='--',label='Service limit',lw=1)
axs[0].set(xlabel='Initial regional error bound (vehicles)',ylabel='Certified deadline bound (vehicles)',title='(a) Deadline bounds')
axs[0].legend(fontsize=8.2,frameon=True,facecolor='white',edgecolor='none',framealpha=.96,
              loc='upper left',bbox_to_anchor=(.01,.94))
axs[0].grid(alpha=.15)
for i,m in enumerate(methods):
    for j,d in enumerate([1,5,10,25,50,100]):
        row=next(r for r in rows if r['method']==m and r['d']==d)
        axs[1].scatter(j,i,marker='o' if row['accepted'] else 'x',s=65,c=TEAL if row['accepted'] else '#a3aab1')
axs[1].set(xticks=range(6),xticklabels=[1,5,10,25,50,100],yticks=range(5),yticklabels=labels,
           xlabel='Initial regional error bound (vehicles)',title='(b) Complete certificate')
axs[1].invert_yaxis()
axs[1].tick_params(labelsize=9)
fig.tight_layout(w_pad=2)
save(fig,'structure_certification_comparison')
evidence['structure_figure']={'source':str(path.relative_to(ROOT)),'sha256':sha(path),
    'method_labels':dict(zip(methods,labels)),'values_changed':False,'records':len(rows)}


# Make the existing small queue violation legible without rerunning trajectories.
paths={k:ROOT/f'output/review/numerical_section_2026-11-10/{k}.csv'
       for k in ['unchecked_tool_switching','certified_tool_switching']}
data={k:np.loadtxt(p,delimiter=',') for k,p in paths.items()}
assert all(a.shape[1]==8 for a in data.values())
overflow=float(np.max(data['unchecked_tool_switching'][:,7])*100)
assert abs(overflow-2005.59)<.02
fig,axs=plt.subplots(2,1,figsize=(10.1,5.0),sharex=True)
for name,color,label in [('unchecked_tool_switching',RED,'Unchecked replacements'),
                         ('certified_tool_switching',TEAL,'Supervised replacements')]:
    a=data[name]
    axs[0].plot(a[:,0]*60,a[:,7]*100,c=color,lw=1.7,label=label)
    axs[1].plot(a[:,0]*60,a[:,1]*100,c=color,lw=1.7)
axs[0].axhline(2000,c='#263741',ls='--',lw=1)
axs[0].text(40,2050,'Queue capacity: 2000',fontsize=9,color='#263741')
axs[0].set(ylabel='Queue (vehicles)',ylim=(0,2320))
axs[0].legend(loc='lower right',frameon=False,fontsize=9)
axs[1].axhline(2750,c='#263741',ls='--',lw=1)
axs[1].text(42,2800,'Minute-60 service limit: 2750',fontsize=9,color='#263741')
axs[1].set(ylabel='Region 1 (vehicles)',xlabel='Time (minutes)',ylim=(1500,2930),xlim=(20,60))
for ax in axs:
    for t in [25,30]:ax.axvline(t,c=GRAY,ls=':',lw=.9)
    ax.grid(alpha=.12)
for minute,label,textx,texty in [(20,'Initial repair',21,1030),(25,'Accept',26.5,800),(30,'Reject; keep current plan',32,1050)]:
    a=data['certified_tool_switching']
    idx=int(np.argmin(abs(a[:,0]*60-minute)))
    axs[0].annotate(label,xy=(minute,a[idx,7]*100),xytext=(textx,texty),
                    fontsize=9,color='#263741',ha='left' if minute!=25 else 'center',
                    arrowprops=dict(arrowstyle='-',lw=.8,color=GRAY))
inset=axs[0].inset_axes([.04,.61,.40,.30])
a=data['unchecked_tool_switching']
inset.plot(a[:,0]*60,a[:,7]*100,c=RED,lw=1.4)
inset.axhline(2000,c='#263741',ls='--',lw=.9)
inset.set(xlim=(59.65,60.02),ylim=(1986,2013),xticks=[59.7,60],yticks=[1990,2000,2010])
inset.tick_params(labelsize=7)
inset.set_facecolor('#fffdfb')
for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#bcc5cc')
inset.text(59.67,2009.5,f'Peak {overflow:.1f}',color=RED,fontsize=8)
inset.plot([60],[overflow],'o',ms=3,color=RED)
inset.set_title(f'Overflow: {overflow-2000:.1f} vehicles',fontsize=8,pad=3)
fig.tight_layout(h_pad=1.2)
save(fig,'successive_decisions_traffic')
evidence['replacement_figure']={'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths.values()},
    'peak_queue_vehicles':overflow,'overflow_vehicles':overflow-2000,
    'values_changed':False,'new_simulations':False}

(OUT/'figure_validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items()},indent=2))
