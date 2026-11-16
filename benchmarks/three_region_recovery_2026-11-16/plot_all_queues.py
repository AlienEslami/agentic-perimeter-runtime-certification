"""Pair each regional count with its queue, using the saved verified study."""
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
OUT=HERE.parents[1]/'output/review/three_region_paper_2026_11_16'
OUT.mkdir(parents=True,exist_ok=True)
report=json.loads((DATA/'validated_results.json').read_text())
proof=json.loads(gzip.decompress((DATA/'monotone_repair045_class_d25_w300_proof.json.gz').read_bytes()))
traces={k:np.load(DATA/f'{k}_ensemble.npz') for k in ['proposal','repair','backup']}
assert proof['accepted']
BLUE,TEAL,RED,GRAY='#315d96','#087f80','#b84d35','#727780'
colors={'proposal':RED,'repair':TEAL,'backup':GRAY}
styles={'proposal':(0,(5,2.5)),'repair':'-','backup':(0,(1.8,2))}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':11,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(3,2,figsize=(9.2,8.8))
critical=100*min(np.roots([3*.0014877,2*(-.29815),15.0912]))
validation={}
for j in range(3):
    an,aq=axs[j];components=slice(3*j,3*j+3)
    bt,lo,hi=[],[],[]
    for rec in proof['steps']:
        left,right=[float(Fraction(rec[k])) for k in ['left','right']]
        local=np.linspace(0,right-left,4)
        c=sum(rec['center'][components]);v=sum(rec['velocity'][components]);a=sum(rec['acceleration'][components])
        center=100*(c+v*local+a*local**2);radius=100*rec['row_error_upper'][9+j]
        an.fill_between(60*(left+local),np.maximum(0,center-radius),center+radius,fc=BLUE,alpha=.17,lw=0,zorder=0)
        bt.extend([60*left,60*right]);lo.extend([100*rec['queues'][j]['continuous'][0]]*2);hi.extend([100*rec['queues'][j]['continuous'][1]]*2)
        t=traces['repair']['time_h'];mask=(t>=left-1e-13)&(t<=right+1e-13)
        q=traces['repair']['state_vehicles'][mask,:,9+j]
        assert np.all(q>=lo[-1]-1e-5) and np.all(q<=hi[-1]+1e-5)
    aq.fill_between(bt,lo,hi,color=BLUE,alpha=.17,lw=0,zorder=0)
    for name in ['backup','proposal','repair']:
        tr=traces[name]
        an.plot(60*tr['time_h'],tr['regional_vehicles'][:,0,j],color=colors[name],ls=styles[name],lw=1.6)
        aq.plot(60*tr['time_h'],tr['state_vehicles'][:,0,9+j],color=colors[name],ls=styles[name],lw=1.6)
    an.axhline(critical,color='#48545d',ls=(0,(6,4)),lw=.8,zorder=-1)
    an.plot(60,3300,'D',color='#27323b',ms=4.5,clip_on=False)
    endpoint=100*np.array(proof['terminal']['center'])[components].sum()
    an.errorbar(60,endpoint,yerr=report['plans']['repair']['certified_endpoint_upper'][j]-endpoint,
                fmt='none',ecolor=BLUE,lw=1.2,capsize=3)
    an.set(ylabel='Vehicles in region',ylim=(1250,4800),yticks=[1500,2500,3500,4500])
    an.set_title(f'({"ace"[j]}) Region {j+1}',loc='left')
    aq.axhline(2000,color='#27323b',ls=(0,(6,4)),lw=.8)
    aq.set(ylabel='Vehicles waiting',ylim=(250,2580),yticks=[500,1000,1500,2000,2500])
    aq.set_title(f'({"bdf"[j]}) Entry queue of Region {j+1}',loc='left')
    if j==0:
        an.text(1,critical+100,'Critical accumulation',fontsize=8.8,color='#48545d')
        an.annotate('Target at 60 min: 3300',xy=(60,3300),xytext=(23,4300),fontsize=8.6,
                    arrowprops=dict(arrowstyle='-',lw=.7,color='#27323b'))
        aq.text(1,2060,'Queue capacity: 2000',fontsize=8.5,color='#27323b')
        aq.annotate('Proposal: 2265',xy=(60,2265),xytext=(30,2440),fontsize=8.5,color=RED,
                    arrowprops=dict(arrowstyle='-',lw=.7,color=RED))
        aq.annotate('Repair: at most 1993',xy=(60,1992.75),xytext=(23,620),fontsize=8.5,color=TEAL,
                    arrowprops=dict(arrowstyle='-',lw=.7,color=TEAL))
    else:
        aq.text(1,2070,'Queue capacity: 2000',fontsize=8.5,color='#27323b')
        aq.text(2,1720,'All three plan curves coincide',fontsize=8.5,color='#48545d')
        aq.text(30,590,'Verified maximum: 1311',fontsize=8.5,color=TEAL,ha='center')
    validation[f'queue_{j+1}']=dict(guaranteed_maximum=max(hi),rounded_up=int(np.ceil(max(hi))),
        sampled_maximum=float(traces['repair']['state_vehicles'][:,:,9+j].max()))
for j in [1,2]:
    for name in ['proposal','backup']:
        assert np.allclose(traces[name]['state_vehicles'][:,0,9+j],traces['repair']['state_vehicles'][:,0,9+j],rtol=0,atol=1e-7)
for row in axs:
    for ax in row:
        ax.axvline(5,color='#8b9298',ls=':',lw=.7,zorder=-2)
        ax.set(xlim=(0,61),xticks=[0,15,30,45,60],xlabel='Time (minutes)')
        ax.grid(axis='y',alpha=.15)
handles=[Line2D([],[],color=colors[k],ls=styles[k],lw=1.8,label=name) for k,name in [('proposal','Proposed plan'),('repair','Repaired plan'),('backup','Backup plan')]]
handles.append(Patch(fc=BLUE,alpha=.17,label='Guaranteed vehicle-count range for the repaired plan'))
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.008),ncol=2,frameon=False,
           handlelength=2.4,columnspacing=1.6,fontsize=9.2)
fig.subplots_adjust(left=.09,right=.98,top=.967,bottom=.12,wspace=.30,hspace=.49)
fig.savefig(OUT/'three_region_recovery_all_queues.pdf',bbox_inches='tight')
fig.savefig(OUT/'three_region_recovery_all_queues.png',dpi=190,bbox_inches='tight')
(OUT/'figure_data_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps(validation))
