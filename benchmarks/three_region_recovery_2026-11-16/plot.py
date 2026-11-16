"""Four-panel comparison from validated three-region results."""
from pathlib import Path
from fractions import Fraction
import gzip,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE=Path(__file__).resolve().parent
DATA=HERE/'results'
OUT=HERE.parents[1]/'output/review/three_region_recovery_2026_11_16'
OUT.mkdir(parents=True,exist_ok=True)
report=json.loads((DATA/'validated_results.json').read_text())
proof=json.loads(gzip.decompress((DATA/'monotone_repair045_class_d25_w300_proof.json.gz').read_bytes()))
traces={key:np.load(DATA/f'{key}_ensemble.npz') for key in ['proposal','repair','backup']}
assert proof['accepted']
BLUE,TEAL,RED,GRAY='#315d96','#087f80','#b84d35','#727780'
colors={'proposal':RED,'repair':TEAL,'backup':GRAY}
styles={'proposal':(0,(5,2.5)),'repair':'-','backup':(0,(1.8,2))}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(10.4,7.0))
critical=100*min(np.roots([3*.0014877,2*(-.29815),15.0912]))
for j,ax in enumerate(axs.flat[:3]):
    for rec in proof['steps']:
        left,right=[float(Fraction(rec[k])) for k in ['left','right']]
        local=np.linspace(0,right-left,4)
        components=slice(3*j,3*j+3)
        c=sum(rec['center'][components]);v=sum(rec['velocity'][components]);a=sum(rec['acceleration'][components])
        center=100*(c+v*local+a*local**2)
        radius=100*rec['row_error_upper'][9+j]
        ax.fill_between(60*(left+local),np.maximum(0,center-radius),center+radius,
                        fc=BLUE,alpha=.17,lw=0,zorder=0)
    for name,tr in traces.items():
        ax.plot(60*tr['time_h'],tr['regional_vehicles'][:,0,j],color=colors[name],ls=styles[name],lw=1.7)
    ax.axhline(critical,color='#48545d',ls=(0,(6,4)),lw=.8,zorder=-1)
    ax.plot(60,3300,'D',color='#27323b',ms=5,clip_on=False)
    ax.errorbar(60,100*np.array(proof['terminal']['center'])[components].sum(),
                yerr=report['plans']['repair']['certified_endpoint_upper'][j]-100*np.array(proof['terminal']['center'])[components].sum(),
                fmt='none',ecolor=BLUE,lw=1.3,capsize=3)
    ax.set(title=f'({"abc"[j]}) Region {j+1}',ylabel='Vehicles in region',ylim=(1250,4800),yticks=[1500,2500,3500,4500])
    ax.set_title(f'({"abc"[j]}) Region {j+1}',loc='left')
    # Remove the duplicate centered title created by set(title=...).
    ax.set_title('',loc='center')
    if j==0:
        ax.text(2,critical+80,'Critical accumulation',fontsize=8.8,color='#48545d')
        ax.annotate('Target at 60 min: 3300',xy=(60,3300),xytext=(30,3980),fontsize=9,
                    arrowprops=dict(arrowstyle='-',lw=.7,color='#27323b'))
ax=axs.flat[3]
bt,lo,hi=[],[],[]
for rec in proof['steps']:
    bt.extend([60*float(Fraction(rec[k])) for k in ['left','right']])
    lo.extend([100*rec['queues'][0]['continuous'][0]]*2)
    hi.extend([100*rec['queues'][0]['continuous'][1]]*2)
ax.fill_between(bt,lo,hi,color=BLUE,alpha=.17,lw=0,zorder=0)
for name,tr in traces.items():
    ax.plot(60*tr['time_h'],tr['state_vehicles'][:,0,9],color=colors[name],ls=styles[name],lw=1.7)
ax.axhline(2000,color='#27323b',lw=1,ls=(0,(6,4)))
ax.text(1,2040,'Queue capacity: 2000',fontsize=9,color='#27323b')
ax.annotate('Proposal: 2265',xy=(60,2265),xytext=(35,2420),fontsize=9,color=RED,
            arrowprops=dict(arrowstyle='-',lw=.7,color=RED))
ax.annotate('Repair: at most 1993',xy=(60,1992.75),xytext=(26,760),fontsize=9,color=TEAL,
            arrowprops=dict(arrowstyle='-',lw=.7,color=TEAL))
ax.text(2,370,'Queues 2 and 3: at most 1311 vehicles',fontsize=8.8,color='#48545d')
ax.set(ylabel='Vehicles waiting',ylim=(250,2580),yticks=[500,1000,1500,2000,2500])
ax.set_title('(d) Entry queue of Region 1',loc='left')
for ax in axs.flat:
    ax.axvline(5,color='#8b9298',ls=':',lw=.7,zorder=-2)
    ax.set(xlim=(0,61),xticks=[0,15,30,45,60],xlabel='Time (minutes)')
    ax.grid(axis='y',alpha=.15)
handles=[Line2D([],[],color=colors[k],ls=styles[k],lw=1.8,label=name) for k,name in [('proposal','Proposed plan'),('repair','Repaired plan'),('backup','Backup plan')]]
handles.append(Patch(fc=BLUE,alpha=.17,label='Guaranteed vehicle-count range for the repaired plan'))
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=2,frameon=False,
           handlelength=2.6,columnspacing=2.5,fontsize=9.5)
fig.subplots_adjust(left=.082,right=.975,top=.96,bottom=.16,wspace=.27,hspace=.36)
fig.savefig(OUT/'three_region_recovery_comparison.pdf',bbox_inches='tight')
fig.savefig(OUT/'three_region_recovery_comparison.png',dpi=190,bbox_inches='tight')
caption=(r'Three-region recovery with one exterior queue per region. '
 'All regions start with 4500 vehicles and all queues with 500 vehicles at the initial estimate. '
 'Panels (a)--(c) show regional counts; panel (d) shows the queue affected by the priority request. '
 'The proposed, repaired and backup curves use identical allowed conditions: each destination-specific initial count is '
 '25/3 vehicles above its estimate, each queue is 10 vehicles above its estimate, and the arrival error is +300 veh/h '
 'at every queue. The shaded ranges apply to the repaired plan for every allowed initial state and every measurable '
 'arrival-error signal bounded by 300 veh/h per queue; they are bounds on vehicle counts. '
 'The diamonds mark the 3300-vehicle targets at minute 60. The vertical dotted lines mark the scheduled input change '
 'at minute 5; all three plans start at minute 0. The proposal violates queue storage, while the repaired and backup '
 'plans are verified over the full hour. Queues 2 and 3, omitted from panel (d), remain below a verified upper bound '
 'of 1311 vehicles. Displayed guaranteed counts are rounded upward to whole vehicles.')
(OUT/'figure_caption.txt').write_text(caption,encoding='utf-8')
print(OUT)
