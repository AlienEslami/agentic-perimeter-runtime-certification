"""Redraw the recovery evidence in four panels from archived results only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'output/package/PERIMETER_PARTB_CONSOLIDATED_2026-11-12'
(PROJECT / 'figures').mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'output/review/simulation_figure_trim_2026_11_16'
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the previously validated certificate reconstruction and stored envelope.
# Stop before plotting, sampling, planning, or other figure generation.
wrapper = (ROOT / 'scripts/plot_results_refinement_2026_11_15.py').read_text()
wrapper = wrapper[:wrapper.index("exec(compile(code,__file__,'exec')")]
ns = {'__file__': __file__}
exec(compile(wrapper, __file__, 'exec'), ns)
code = ns['code']
a = code.index('# Demand is the actual stored plant forecast')
b = code.index('# Reconstruct the same polynomial predictor', a)
code = code[:a] + code[b:]
code = code[:code.index('fig,axs=plt.subplots(3,2')]
code = code.replace('output/review/results_refinement_2026_11_15',
                    'output/review/simulation_figure_trim_2026_11_16')
assert 'for k in range(1801)' not in code
data = {'__file__': __file__, '__name__': '__main__'}
exec(compile(code, __file__, 'exec'), data)
draw, endpoint, endrad, envelope, traces, checks = (
    data[k] for k in ['draw','endpoint','endrad','envelope','traces','checks'])

BLUE, TEAL, RED, GRAY = '#285e8c', '#007c83', '#b94c34', '#737f89'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
    'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,
    'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42})
colors = {'proposal':RED,'repair':TEAL,'backup':GRAY}
styles = {'proposal':'--','repair':'-','backup':':'}
labels = {'proposal':'Proposed plan','repair':'Repaired plan','backup':'Backup plan'}
crit = 3391.93080685
fig, axs = plt.subplots(2,2,figsize=(8.8,5.7))
for j, ax in enumerate(axs.flat[:3]):
    field = ['n1_vehicles','n2_vehicles','z_vehicles'][j]
    if j < 2:
        for d in draw:
            ss = np.linspace(0,d['right']-d['left'],9)
            pp = 100*(d['center'][j]+d['vel'][j]*ss+d['acc'][j]*ss**2)
            ax.fill_between(60*(d['left']+ss),pp-d['rad'][j],pp+d['rad'][j],
                            color=BLUE,alpha=.15,linewidth=0)
        ax.errorbar([60],[endpoint[j]],yerr=[endrad[j]],fmt='none',
                    ecolor=BLUE,capsize=3,lw=1)
        ax.set_ylim(1000,5200)
        ax.set_yticks([1000,2000,3000,4000,5000])
    else:
        tx, qh = [], []
        for d in draw:
            tx.extend([60*d['left'],60*d['right']])
            qh.extend([d['qhi']]*2)
        ax.plot(tx,qh,color=BLUE,lw=1.2,ls='-.',zorder=4)
        ax.set_ylim(400,2800)
    ax.fill_between(envelope[:,0]*60,envelope[:,1+j],envelope[:,4+j],
                    color=TEAL,alpha=.36,linewidth=0)
    for name, tr in traces.items():
        ax.plot(tr['time_h']*60,tr[field],color=colors[name],
                ls=styles[name],lw=1.5)
    ax.axhline(crit if j<2 else 2000,color='#333c43',ls='--',lw=.8)
    ax.axvline(5,color='#89969f',ls=':',lw=.8,zorder=0)
    ax.set(xlim=(0,61),xlabel='Time (minutes)',
           ylabel='Vehicles' if j<2 else 'Waiting vehicles')
    ax.set_title(['(a) Region 1','(b) Region 2','(c) Exterior queue'][j],loc='left')
    ax.grid(axis='y',alpha=.15)
    if j==0:
        ax.plot(60,3300,'D',ms=4,mfc='white',mec='#333c43',clip_on=False)
        ax.annotate('Deadline target: 3300',xy=(60,3300),xytext=(58,4500),
                    ha='right',fontsize=7.6,
                    arrowprops={'arrowstyle':'-','lw':.6})
    if j==2:
        ax.text(2,2070,'Storage limit: 2000',fontsize=7.7)

ax=axs[1,1]
n=np.linspace(0,10000,500)
G=lambda n:1.4877e-7*n**3-.0029815*n*n+15.0912*n
ax.plot(n,G(n)/1000,color='#37434b',lw=1.4)
ax.axvline(crit,color=GRAY,ls=':',lw=1)
ax.plot(4500,G(4500)/1000,'o',color=GRAY,ms=4)
ends=[traces['repair'][f][-1] for f in ['n1_vehicles','n2_vehicles']]
ax.scatter(ends,[G(v)/1000 for v in ends],color=TEAL,s=20)
ax.annotate('Congested start',xy=(4500,G(4500)/1000),xytext=(6000,24.8),
            fontsize=7.8,arrowprops={'arrowstyle':'-','lw':.6})
ax.annotate('After recovery\n(repaired plan, 60 min)',
            xy=(ends[0],G(ends[0])/1000),xytext=(4500,5),fontsize=7.8,
            arrowprops={'arrowstyle':'-','lw':.6})
ax.text(crit-180,25.4,'Critical',fontsize=7.6,ha='right')
ax.set(xlim=(0,10000),ylim=(0,27),xlabel='Accumulation (vehicles)',
       ylabel=r'Potential outflow ($10^3$ veh/h)',xticks=[0,5000,10000])
ax.set_title('(d) Recovery on the MFD',loc='left')
handles=[Line2D([],[],color=colors[n],ls=styles[n],lw=1.5,label=labels[n])
         for n in ['proposal','repair','backup']]
handles += [Patch(facecolor=BLUE,alpha=.15,label='Certified regional bounds'),
            Patch(facecolor=TEAL,alpha=.36,label='Simulated range (repaired plan)'),
            Line2D([],[],color=BLUE,ls='-.',label='Certified queue upper bound')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,
           fontsize=7.8,bbox_to_anchor=(.5,.005),columnspacing=1.4)
fig.tight_layout(rect=(0,.105,1,1),h_pad=1.5,w_pad=2)
name='congested_recovery_compact_2026_11_16'
fig.savefig(PROJECT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
fig.savefig(OUT/f'{name}.png',dpi=190,bbox_inches='tight',pad_inches=.06)
plt.close(fig)

sources=[data['certpath'],ROOT/'benchmarks/partb_traffic_presentation_2026-11-15/recovery_sampled_envelope.csv']
sources += [data['recdir']/f'{name}_trace.csv' for name in ['proposal','repair','backup']]
report={'new_simulations':False,'panels':4,'control_input_panels_removed':True,
        'same_saved_trajectories_and_certificate':True,
        'predictor_checkpoint_error_vehicles':max(checks),
        'sampled_envelope_within_archived_certificate':True,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(OUT/'figure_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
