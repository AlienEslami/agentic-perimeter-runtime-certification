from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,Circle,FancyArrowPatch
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1]
SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
sys.path.insert(0,str(SUITE))
import common
OUT=ROOT/'output/package/PERIMETER_PARTB_PRE_SIMULATION_2026-11-10/figures'
DATA=ROOT/'output/review/numerical_section_2026-11-10';DATA.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False})
TEAL='#007c83';RED='#b84732';BLUE='#285c87';GRAY='#74818e'
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.1)
    fig.savefig(DATA/(name+'.png'),dpi=130,bbox_inches='tight',pad_inches=.1)
    plt.close(fig)
def car(ax,x,y,color,w=.17):
    ax.add_patch(FancyBboxPatch((x,y),w,.075,boxstyle='round,pad=0.005,rounding_size=0.018',fc=color,ec='none'))
    ax.add_patch(Rectangle((x+w*.25,y+.075),w*.48,.035,fc=color,ec='none'))
    for xx in [x+w*.2,x+w*.8]:ax.add_patch(Circle((xx,y),.018,fc='#243447'))
def snapshot(ax,x,u,title,offered):
    ax.set(xlim=(-.12,4.65),ylim=(-.3,2.7));ax.axis('off');ax.set_title(title,fontsize=17,pad=8)
    for j,left in [(0,0),(1,1.68)]:
        congested=x[j]>3391.93080685
        ax.add_patch(FancyBboxPatch((left,.65),1.3,1.5,boxstyle='round,pad=.03,rounding_size=.08',fc='#fff0db' if congested else '#edf6f7',ec=BLUE,lw=1.5))
        ax.text(left+.65,1.96,f'Region {j+1}',ha='center',va='center',fontsize=14,weight='bold')
        ax.text(left+.65,1.68,f'{x[j]:,.0f}',ha='center',fontsize=18,weight='bold',color=BLUE)
        for k in range(round(x[j]/250)):
            car(ax,left+.12+(k%5)*.22,1.39-(k//5)*.17,BLUE)
    queuecolor=RED if x[6]>2000 else TEAL
    ax.add_patch(FancyBboxPatch((3.55,.65),.92,1.5,boxstyle='round,pad=.03,rounding_size=.08',fc='#ffebe5' if x[6]>2000 else '#f2f6f5',ec=queuecolor,lw=1.5))
    ax.text(4.01,1.96,'Queue',ha='center',fontsize=14,weight='bold')
    ax.text(4.01,1.68,f'{x[6]:,.0f}',ha='center',fontsize=16,weight='bold',color=queuecolor)
    for k in range(round(x[6]/250)):car(ax,3.67+(k%3)*.23,1.39-(k//3)*.2,queuecolor)
    ax.text(4.01,.43,'Limit 2,000',ha='center',fontsize=13,color=queuecolor)
    n1,n2,n11,n22,n10,n20,z=x/100
    q=common.plant.original.class_flow
    f12=100*u[0]*(q(n1-n11-n10,n1)+q(n10,n1));f21=100*u[1]*q(n2-n22-n20,n2)
    adm=min(60*x[6],12000*u[2]);dep=100*u[3]*q(n20,n2)
    def arrow(start,end,flow,color):
        if flow<1:return
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11+flow/1600,lw=.7+flow/5000,color=color))
    arrow((1.3,1.43),(1.68,1.43),f12,BLUE);arrow((1.68,1.03),(1.3,1.03),f21,BLUE)
    arrow((3.55,1.25),(2.98,1.25),adm,TEAL);arrow((4.01,2.58),(4.01,2.18),offered,GRAY)
    for j,left in [(0,0),(1,1.68)]:
        arrow((left+.65,.63),(left+.65,.15),100*q([n11,n22][j],[n1,n2][j])+(dep if j==1 else 0),BLUE)
        ax.text(left+.65,-.08,'Departures',ha='center',fontsize=12)
    return [float(f12),float(f21),float(adm),float(dep)]
def snapshots(congested=False):
    if congested:
        p=ROOT/'benchmarks/congested_recovery_2026-11-10/results';times=[0,30,60]
        arrays=[np.loadtxt(p/f'{n}_d1_w60_trace.csv',delimiter=',',skiprows=1) for n in ['proposal','repair']]
        get=lambda a,k:(a[k,1:8],a[k,12:16]);label='congested_snapshots'
    else:
        p=SUITE/'results/supervisory_vehicle_flows_2026-11-08';times=[20,40,60]
        arrays=[np.loadtxt(p/f'D027_{n}_trajectory.csv',delimiter=',',skiprows=1) for n in ['nominal_mpc','grid_repair']]
        get=lambda a,k:(100*a[k,1:8],a[k,-4:]);label='supervisory_snapshots'
    fig,axs=plt.subplots(2,3,figsize=(11.5,7.1))
    fig.subplots_adjust(left=.025,right=.99,bottom=.04,top=.92,hspace=.38,wspace=.12)
    evidence=[]
    for i,a in enumerate(arrays):
      for j,t in enumerate(times):
        k=np.argmin(abs(a[:,0]*60-t));x,u=get(a,k)
        offered=100*common.plant.arrival(t/60,amplitude=.3,duration=1.)+60 if congested else a[k,8]
        flow=snapshot(axs[i,j],x,u,f'{t} minutes',offered)
        evidence.append(dict(row=i,minute=t,state_vehicles=x.tolist(),input=u.tolist(),flows_veh_h=flow))
    fig.text(.03,.97,'(a) Proposed plan without supervision',fontsize=18,weight='bold',color=RED)
    fig.text(.03,.48,'(b) Certified repair',fontsize=18,weight='bold',color=TEAL)
    save(fig,label);(DATA/(label+'.json')).write_text(json.dumps(evidence,indent=2))
snapshots();snapshots(True)

# Congested trajectories, with independent continuous-time queue certificate.
p=ROOT/'benchmarks/congested_recovery_2026-11-10/results'
fig,axs=plt.subplots(1,3,figsize=(10.5,4.8))
for n,c,l in [('proposal',RED,'Proposed'),('repair',TEAL,'Repaired'),('backup',GRAY,'Backup')]:
    a=np.loadtxt(p/f'{n}_d1_w60_trace.csv',delimiter=',',skiprows=1)
    for ax,j,title in zip(axs,[1,2,7],['Region 1','Region 2','Exterior queue']):
        ax.plot(a[:,0]*60,a[:,j],c=c,lw=2,label=l);ax.set(title=title,xlabel='Time (minutes)',ylabel='Vehicles',xlim=(0,60));ax.grid(alpha=.15)
for ax in axs[:2]:ax.axhline(3391.93080685,color='#354557',ls='--',label='Critical accumulation')
axs[0].plot([60],[3300],marker='D',ms=6,c='black',label='Service limit',clip_on=False)
cert=read(p/'repair_02_certificate.json');t=[];bound=[]
for st in cert['interval_steps']:
    t.extend([st['left_h']*60,st['right_h']*60]);bound.extend([2000-100*st['continuous_constraint_margins_lower'][11]]*2)
axs[2].plot(t,bound,':',c='#003d42',lw=1.5,label='Certified upper bound');axs[2].axhline(2000,c='black',ls='--',label='Capacity')
handles,legend_labels=axs[0].get_legend_handles_labels(); h2,l2=axs[2].get_legend_handles_labels()
for h,l in zip(h2,l2):
    if l not in legend_labels:handles.append(h);legend_labels.append(l)
fig.legend(handles,legend_labels,loc='upper center',ncol=4,fontsize=11,frameon=False)
fig.tight_layout(rect=(0,0,1,.84));save(fig,'congested_counts_certificate')

# Same-plan certification comparison; no fabricated values at failed checks.
rows=read(ROOT/'benchmarks/manuscript_review_2026-11-08/results/coordinate_comparison.json')['rows']
methods=['physical_box','class_box','conservation_box','positive_queue','signed']
labels=['Raw accumulations','Destination-specific','Conservation','Positivity/queue','Signed generators']
fig,axs=plt.subplots(1,2,figsize=(10.5,4.2),gridspec_kw={'width_ratios':[1.2,1]})
for m,l,c in zip(methods,labels,[GRAY,'#ab5d30',BLUE,'#7971a6',TEAL]):
    rr=sorted([r for r in rows if r['method']==m and r['accepted']],key=lambda r:r['d'])
    if rr:axs[0].plot([r['d'] for r in rr],[r['deadline_bound_vehicles'] for r in rr],'-o',c=c,label=l)
axs[0].axhline(2850,c='black',ls='--',label='Service limit');axs[0].set(xlabel='Initial regional error bound (vehicles)',ylabel='Certified deadline bound (vehicles)',title='(a) Certified deadline bounds');axs[0].legend(fontsize=10);axs[0].grid(alpha=.15)
for i,m in enumerate(methods):
 for j,d in enumerate([1,5,10,25,50,100]):
    r=next(r for r in rows if r['method']==m and r['d']==d)
    axs[1].scatter(j,i,marker='o' if r['accepted'] else 'x',s=100,c=TEAL if r['accepted'] else '#a3aab1')
axs[1].set(xticks=range(6),xticklabels=[1,5,10,25,50,100],yticks=range(5),yticklabels=labels,xlabel='Initial regional error bound (vehicles)',title='(b) Complete certificate');axs[1].invert_yaxis();axs[1].tick_params(labelsize=11)
fig.tight_layout();save(fig,'structure_certification_comparison')

# Repair distributions, preserving the manuscript's matched denominators.
vals=[[],[],[]]
for f in sorted((SUITE/'results/contracts').glob('D*.json')):
    r=read(f)
    if (r.get('selected_theta') or 0)<=0:continue
    pol=r['policies']
    for a,b,n in zip(pol[r['certified_repair_source']]['realizations'],pol[r['certified_retention_source']]['realizations'],pol['nominal_mpc']['realizations']):
        assert (a['corner'],a['profile'])==(b['corner'],b['profile'])==(n['corner'],n['profile'])
        vals[0].append(100*(b['priority_cost']-a['priority_cost'])/b['priority_cost'])
        vals[1].append(100*(a['tts_veh_h']-b['tts_veh_h'])/b['tts_veh_h'])
        vals[2].append(100*(b['priority_cost']-a['priority_cost'])/(b['priority_cost']-n['priority_cost']))
assert all(len(v)==24320 for v in vals)
fig,axs=plt.subplots(1,3,figsize=(10.5,4))
for ax,v,title in zip(axs,vals,['Priority improvement\nover retention','TTS increase\nover retention','Requested benefit\nrecovered']):
    q=np.percentile(v,[5,25,50,75,95]);ax.bxp([dict(med=q[2],q1=q[1],q3=q[3],whislo=q[0],whishi=q[4],fliers=[])],positions=[1],widths=.4,patch_artist=True,boxprops=dict(facecolor='#d5eeeb'),medianprops=dict(color=TEAL,lw=2))
    ax.set(title=title,ylabel='Percent',xticks=[]);ax.grid(axis='y',alpha=.15);ax.text(1.25,q[2],f'{q[2]:.1f}%',color=TEAL,va='center')
fig.tight_layout();save(fig,'repair_cost_distribution')
(DATA/'repair_quantiles.json').write_text(json.dumps([np.percentile(v,[5,25,50,75,95]).tolist() for v in vals]))

# Replay saved sequential plans only; no new request selection or optimization.
p=SUITE/'results/sequential_authority_2026-10-01';record=read(p/'D025_D024.json');c=record['contract'];commands=read(p/'D025_D024_counterfactuals.json')['commands']
fig,axs=plt.subplots(3,1,figsize=(10,7),sharex=True,gridspec_kw={'height_ratios':[1,1,.5]})
for name,col,label in [('unchecked_tool_switching',RED,'Unchecked replacements'),('certified_tool_switching',TEAL,'Supervised replacements')]:
 x=np.asarray(common.fixture()['center'])+c['width_vehicles']/100;parts=[]
 for (left,right),u in zip([(1/3,5/12),(5/12,.5),(.5,1)],commands[name]):
    sol=solve_ivp(lambda t,y:common.plant.field(y,u,common.arrival(t,c)+c['residual_veh_h']/100),(left,right),x,rtol=1e-11,atol=1e-12,method='DOP853',max_step=1/720,dense_output=True)
    assert sol.success
    tt=np.linspace(left,right,round((right-left)*3600)+1);yy=sol.sol(tt).T;parts.append(np.column_stack([tt,yy]));x=sol.y[:,-1]
 a=np.concatenate(parts);np.savetxt(DATA/(name+'.csv'),a,delimiter=',')
 axs[0].plot(a[:,0]*60,a[:,7]*100,c=col,label=label);axs[1].plot(a[:,0]*60,a[:,1]*100,c=col)
 if name.startswith('unchecked'):assert abs(max(a[:,7])*100-2005.59)<.02
axs[0].axhline(2000,c='black',ls='--');axs[1].axhline(2750,c='black',ls='--')
axs[0].set(ylabel='Queue (vehicles)');axs[1].set(ylabel='Region 1 (vehicles)');axs[0].legend(fontsize=10)
for ax in axs[:2]:
    ax.axvline(25,c=GRAY,ls=':');ax.axvline(30,c=GRAY,ls=':');ax.grid(alpha=.15)
axs[2].set(ylim=(0,1),yticks=[],xlabel='Time (minutes)',xlim=(20,60));axs[2].hlines(.4,20,60,color=TEAL,lw=3)
for t,text in [(20,'Initial repair'),(25,'Accept'),(30,'Reject; retain'),(60,'Deadline')]:
 axs[2].plot(t,.4,'o',c=TEAL);axs[2].text(t,.12 if t==20 else .68,text,ha='left' if t==20 else 'right' if t==60 else 'center',fontsize=10)
fig.tight_layout();save(fig,'successive_decisions_traffic')

# Topology and result summaries of the existing connected-ring examples.
fig,axs=plt.subplots(1,2,figsize=(10,4.8))
for ax,N,w,bound in zip(axs,[2,3],[300,60],[1552,2060]):
 ax.set(xlim=(-1.8,1.8),ylim=(-2.0,1.6));ax.axis('off')
 pts=[np.array([np.cos(2*np.pi*j/N+np.pi/2),np.sin(2*np.pi*j/N+np.pi/2)])*.75 for j in range(N)]
 for j,xy in enumerate(pts):
    other=pts[(j+1)%N]
    ax.add_patch(FancyArrowPatch(xy,other,connectionstyle='arc3,rad=.25',arrowstyle='-|>',mutation_scale=15,color=BLUE,shrinkA=28,shrinkB=28,lw=2))
    ax.add_patch(Circle(xy,.25,fc='#e5f1f6',ec=BLUE,lw=1.5));ax.text(*xy,f'R{j+1}',ha='center',va='center',fontsize=14)
    qxy=xy*1.6;ax.add_patch(Rectangle(qxy-.10,.2,.2,fc='#f7e6c9',ec='#a17026'));ax.add_patch(FancyArrowPatch(qxy,xy,arrowstyle='-|>',shrinkA=8,shrinkB=20,color='#a17026',mutation_scale=12))
 ax.text(0,1.48,f'{N} regions / {N} entry queues',ha='center',fontsize=14,weight='bold')
 ax.text(0,-1.75,f'Arrival-error bound: {w} veh/h per queue\nLargest deadline bound: {bound:,} / 2,700 vehicles',ha='center',fontsize=11)
fig.tight_layout();save(fig,'network_extension_results')
print('Generated eight figure assets from stored simulation and certification records.')
