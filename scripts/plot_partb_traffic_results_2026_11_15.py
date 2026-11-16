"""Traffic-oriented displays from fixed plans and archived certificates."""
from pathlib import Path
from fractions import Fraction
import sys,json,itertools,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'output/package/PERIMETER_PARTB_CONSOLIDATED_2026-11-12'
OUT=ROOT/'output/review/traffic_results_2026_11_15'
DATA=ROOT/'benchmarks/partb_traffic_presentation_2026-11-15'
OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
sys.path.insert(0,str(SUITE))
import common
from scipy.integrate import solve_ivp
BLUE,TEAL,RED,GRAY='#285e8c','#007c83','#b94c34','#737f89'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
 'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
def read(p): return json.loads(Path(p).read_text())
def save(fig,name):
 fig.savefig(PROJECT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.06)
 fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight',pad_inches=.06)
 plt.close(fig)
validation={}

# Demand is the actual stored plant forecast, not an illustrative new event.
t=np.linspace(0,1,601); forecast=np.array([common.plant.arrival(v,amplitude=.3,duration=1)*100 for v in t])
fig,axs=plt.subplots(2,1,figsize=(8.3,3.8),sharex=True,gridspec_kw={'height_ratios':[3,1.15]})
ax=axs[0]
ax.fill_between(60*t,forecast-600,forecast+600,color=BLUE,alpha=.13,label='Congested case: arrival error bound 600 veh/h')
ax.fill_between(60*t,forecast-300,forecast+300,where=t>=1/3,color=TEAL,alpha=.25,label='Priority example: arrival error bound 300 veh/h')
ax.plot(60*t,forecast,color='#243c4b',lw=1.8,label='Arrival forecast')
ax.set(ylabel='Exterior arrivals (veh/h)',ylim=(5200,8700),yticks=[6000,7800])
ax.legend(loc='lower right',fontsize=7.6,frameon=False)
ax.grid(axis='y',alpha=.18)
ax=axs[1]; ax.set(ylim=(-.5,1.5),yticks=[0,1],yticklabels=['Replanning example','Congested recovery'],xlabel='Time (minutes)',xlim=(0,61))
ax.hlines([0,1],[20,0],[60,60],color=[TEAL,BLUE],lw=2)
for xx,yy,lab,off in [(0,1,'Start',.17),(5,1,'Input change',-.37),(60,1,'Deadline',.17),
 (20,0,'Initial plan',-.35),(25,0,'Accept',.22),(30,0,'Reject',-.35),(60,0,'Deadline',.22)]:
 ax.plot(xx,yy,'o',ms=3,color=BLUE if yy else TEAL)
 ax.text(xx,yy+off,lab,fontsize=7.4,ha='left' if xx==0 else 'right' if xx==60 else 'center')
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
fig.tight_layout(h_pad=.5);save(fig,'traffic_scenarios_2026_11_15')

# Reconstruct the same polynomial predictor used by the saved full-hour check.
sys.path.insert(0,str(ROOT/'benchmarks/reviewer_strengthening_2026-11-11'))
import probe
import mfd_incident_affine_reachability as af
from mfd_incident_mpc_certification import constant_command_mode
from mfd_incident_verifier import Pulse
certpath=ROOT/'benchmarks/reviewer_strengthening_2026-11-11/full_certificates/arrival_w600_lambda0.65.json'
saved=read(certpath);cert=saved['certificate'];assert cert['accepted']
cp0=cert['reachable_checkpoints'][0];center=np.array(cp0['center']);L=np.array(cp0['metric_L']);inv=np.linalg.inv(L)
pulse=Pulse(*[Fraction(v) for v in cert['proof_descriptor']['pulse']])
modes=[constant_command_mode(np.array(u),f'constant_block_{i}') for i,u in enumerate(saved['job']['commands'])]
draw=[];checks=[]
for step in cert['interval_steps']:
 left=Fraction(step['left_h']).limit_denominator(2400000);right=Fraction(step['right_h']).limit_denominator(2400000)
 if right==left:continue
 h=af.interval_fraction(right-left);mode=modes[step['segment']]
 ar,sl=pulse.affine(left,right);branch,_=af.branch_for_box(af.I(center),mode);assert branch=='gate'
 values=af.closed_field([af.I(v) for v in center],mode,ar,branch)
 fb=af.I(np.array([v.lo for v in values]),np.array([v.hi for v in values]));vel=fb.mid()
 sr=np.zeros(7);sr[6]=1
 acc=((probe.vector_imat(af.jacobian(af.I(center),mode,branch),fb)+af.I(sr)*sl)*af.I(.5)).mid()
 rad=100*probe.vector_imat(af.I(abs(inv)),af.I(step['apriori_radius'])).hi
 qhi=2000-100*step['continuous_constraint_margins_lower'][11]
 draw.append({'left':float(left),'right':float(right),'center':center.copy(),'vel':vel,'acc':acc,'rad':rad,'qhi':qhi})
 center=(af.I(center)+af.I(vel)*h+af.I(acc)*h*h).mid()
 for cp in cert['reachable_checkpoints']:
  if abs(cp['time_h']-float(right))<1e-12:
   error=max(abs(center-np.array(cp['center'])))*100;checks.append(float(error));assert error<1e-8
assert len(draw)==240
final=cert['reachable_checkpoints'][-1]
endpoint=100*np.array(final['center']);endrad=100*probe.vector_imat(af.I(abs(inv)),af.I(final['radius'])).hi

# Regenerate sampled envelopes only, with unchanged plans/model/sampling and 2 s integration.
recdir=ROOT/'benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery'
plans={n:read(recdir/f'{n}_ensemble.json') for n in ['proposal','repair','backup']}
corners=np.array(list(itertools.product((-1.,1.),repeat=7)))
x=np.tile(np.array(cp0['center']),(640,1))+np.repeat(corners,5,axis=0)*np.array([.25]*6+[.1])
commands=np.array(plans['repair']['commands']);h=2/3600
envelope=[]
for k in range(1801):
 envelope.append(np.r_[k*h,100*x[:,[0,1,6]].min(axis=0),100*x[:,[0,1,6]].max(axis=0)])
 if k==1800:break
 u=commands[0 if k*h<1/12-1e-10 else 1]
 def rhs(y,clock):
  alt=1 if int((k*h+h/2)*12)%2==0 else -1
  w=np.tile(np.array([0,1,-1,alt,np.sin(12*np.pi*clock)])*6,128)
  return common.batch_field(y,u,common.plant.arrival(clock,amplitude=.3,duration=1)+w)[0]
 t=k*h; f1=rhs(x,t);f2=rhs(x+h*f1/2,t+h/2);f3=rhs(x+h*f2/2,t+h/2);f4=rhs(x+h*f3,t+h)
 x=x+h/6*(f1+2*f2+2*f3+f4)
envelope=np.array(envelope)
for j,key in enumerate(['final_n1','final_n2','final_queue']):
 st=plans['repair']['statistics'][key]
 assert abs(envelope[-1,1+j]-st['min'])<1e-7
 assert abs(envelope[-1,4+j]-st['max'])<1e-7
np.savetxt(DATA/'recovery_sampled_envelope.csv',envelope,delimiter=',',header='time_h,n1_min,n2_min,z_min,n1_max,n2_max,z_max',comments='')
for row in envelope:
 j=min(int((row[0]+1e-10)*240),239);d=draw[j];s=row[0]-d['left'];p=100*(d['center']+d['vel']*s+d['acc']*s*s)
 assert all(row[1:3]>=p[:2]-d['rad'][:2]-1e-6)
 assert all(row[4:6]<=p[:2]+d['rad'][:2]+1e-6)
 assert row[6]<=d['qhi']+1e-5

traces={n:np.genfromtxt(recdir/f'{n}_trace.csv',delimiter=',',names=True) for n in plans}
colors={'proposal':RED,'repair':TEAL,'backup':GRAY};styles={'proposal':'--','repair':'-','backup':':'}
labels={'proposal':'Proposed','repair':'Modified','backup':'Backup'}
fig,axs=plt.subplots(3,2,figsize=(8.8,7.6));crit=3391.93080685
for j,ax in enumerate(axs.flat[:3]):
 field=['n1_vehicles','n2_vehicles','z_vehicles'][j]
 if j<2:
  for d in draw:
   ss=np.linspace(0,d['right']-d['left'],9);pp=100*(d['center'][j]+d['vel'][j]*ss+d['acc'][j]*ss**2)
   ax.fill_between(60*(d['left']+ss),pp-d['rad'][j],pp+d['rad'][j],color=BLUE,alpha=.15,linewidth=0)
  ax.errorbar([60],[endpoint[j]],yerr=[endrad[j]],fmt='none',ecolor=BLUE,capsize=3,lw=1)
 else:
  tx=[];qh=[]
  for d in draw:tx.extend([60*d['left'],60*d['right']]);qh.extend([d['qhi']]*2)
  ax.plot(tx,qh,color=BLUE,lw=1.2,ls='-.')
 ax.fill_between(envelope[:,0]*60,envelope[:,1+j],envelope[:,4+j],color=TEAL,alpha=.36,linewidth=0)
 for name,tr in traces.items():ax.plot(tr['time_h']*60,tr[field],color=colors[name],ls=styles[name],lw=1.4)
 ax.axhline(crit if j<2 else 2000,color='#333c43',ls='--',lw=.8)
 ax.set(xlim=(0,61),xlabel='Time (minutes)',ylabel='Vehicles')
 ax.set_title(['(a) Region 1','(b) Region 2','(c) Exterior queue'][j],loc='left')
 ax.grid(axis='y',alpha=.15)
 if j==0:
  ax.plot(60,3300,'D',ms=4,mfc='white',mec='#333c43',clip_on=False)
  ax.annotate('Target at 60 min: 3300',(60,3300),xytext=(58,3950),ha='right',fontsize=7.4,arrowprops={'arrowstyle':'-','lw':.6})
 if j==2:ax.text(2,2045,'Storage limit: 2000',fontsize=7.5)
ax=axs[1,1];n=np.linspace(0,10000,500);G=lambda n:1.4877e-7*n**3-.0029815*n*n+15.0912*n
ax.plot(n,G(n)/1000,color='#37434b',lw=1.3);ax.axvline(crit,color=GRAY,ls=':',lw=1)
ax.plot(4500,G(4500)/1000,'o',color=GRAY,ms=4)
ends=[traces['repair'][f][-1] for f in ['n1_vehicles','n2_vehicles']]
ax.scatter(ends,[G(v)/1000 for v in ends],color=TEAL,s=18)
ax.annotate('Initial accumulation',(4500,G(4500)/1000),(5800,25),fontsize=7.7,arrowprops={'arrowstyle':'-','lw':.6})
ax.annotate('Modified endpoints\nat 60 min',(ends[0],G(ends[0])/1000),(5600,10),fontsize=7.5,arrowprops={'arrowstyle':'-','lw':.6})
ax.text(crit-180,25.2,'Critical',fontsize=7.3,ha='right')
ax.set(xlim=(0,10000),ylim=(0,27),xlabel='Accumulation (vehicles)',ylabel=r'Outflow ($10^3$ veh/h)',xticks=[0,5000,10000]);ax.set_title('(d) MFD operating points',loc='left')
for j,ax in enumerate(axs[2]):
 comp=2 if j==0 else 0
 for name,plan in plans.items():
  u=np.array(plan['commands']);ax.step([0,5,60],[u[0,comp],u[1,comp],u[1,comp]],where='post',color=colors[name],ls=styles[name],lw=1.5)
 ax.set(xlim=(0,61),ylim=(-.03,1.06),xlabel='Time (minutes)',ylabel=['Entry input $u_{02}$','Transfer input $u_{12}$'][j],yticks=[0,.5,1]);ax.grid(axis='y',alpha=.15)
 ax.set_title(['(e) Admission control','(f) Transfer from Region 1 to 2'][j],loc='left')
handles=[Line2D([],[],color=colors[n],ls=styles[n],lw=1.5,label=labels[n]) for n in plans]
handles += [Patch(facecolor=BLUE,alpha=.15,label='Certified regional bounds (modified)'),Patch(facecolor=TEAL,alpha=.36,label='Sampled range (modified)'),Line2D([],[],color=BLUE,ls='-.',label='Certified queue upper bound')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=7.8,bbox_to_anchor=(.5,-.005))
fig.tight_layout(rect=(0,.07,1,1),h_pad=1.4,w_pad=2);save(fig,'congested_traffic_controls_2026_11_15')
validation['congestion']={'certificate_sha256':hashlib.sha256(certpath.read_bytes()).hexdigest(),
 'predictor_checkpoint_error_vehicles':max(checks),'envelope_matches_saved_endpoint_statistics':True,
 'sampled_envelope_within_archived_certificate':True,'samples':640,'step_seconds':2}

# Inputs are the same stored schedules used to generate the archived replacement trajectories.
seqpath=SUITE/'results/sequential_authority_2026-10-01/D025_D024_counterfactuals.json'
seq=read(seqpath)['commands'];fig,axs=plt.subplots(2,2,figsize=(8.8,5.9))
for name,color,label in [('unchecked_tool_switching',RED,'Unchecked replacements'),('certified_tool_switching',TEAL,'Supervised replacements')]:
 a=np.loadtxt(ROOT/f'output/review/numerical_section_2026-11-10/{name}.csv',delimiter=',')
 for ax,j in [(axs[0,0],7),(axs[0,1],1)]:ax.plot(a[:,0]*60,a[:,j]*100,color=color,lw=1.6,label=label)
 u=np.array(seq[name])
 for ax,j in [(axs[1,0],2),(axs[1,1],0)]:ax.step([20,25,30,60],list(u[:,j])+[u[-1,j]],where='post',color=color,lw=1.6,ls='--' if name.startswith('unchecked') else '-')
axs[0,0].axhline(2000,color='#333c43',ls='--',lw=.8)
axs[0,1].plot(60,2750,'D',ms=4,mfc='white',mec='#333c43',clip_on=False)
axs[0,1].text(58,2790,'Target: 2750',ha='right',fontsize=7.6)
axs[0,0].set(ylim=(0,2300),ylabel='Queue (vehicles)');axs[0,1].set(ylim=(1500,2940),ylabel='Region 1 (vehicles)')
axs[1,0].set(ylim=(.3,.72),ylabel='Entry input $u_{02}$');axs[1,1].set(ylim=(.3,1.08),ylabel='Transfer input $u_{12}$')
for ax,title in zip(axs.flat,['(a) Exterior waiting','(b) Regional deadline','(c) Admission control','(d) Interregional control']):
 ax.set_title(title,loc='left');ax.set(xlim=(20,61),xlabel='Time (minutes)');ax.grid(axis='y',alpha=.15)
 for tt in [25,30]:ax.axvline(tt,color=GRAY,ls=':',lw=.8)
axs[0,0].text(25,2240,'Accept',ha='center',fontsize=7.5);axs[0,0].text(33,2080,'Reject',ha='center',fontsize=7.5)
inset=axs[0,0].inset_axes([.63,.05,.34,.18]);a=np.loadtxt(ROOT/'output/review/numerical_section_2026-11-10/unchecked_tool_switching.csv',delimiter=',')
inset.plot(a[:,0]*60,a[:,7]*100,color=RED,lw=1);inset.axhline(2000,color='#333c43',ls='--',lw=.8)
inset.set(xlim=(59.7,60),ylim=(1992,2008),xticks=[59.7,60],yticks=[2000]);inset.tick_params(labelsize=6)
inset.set_title('Peak: 2005.6',fontsize=7);inset.set_facecolor('white')
for spine in inset.spines.values():spine.set_visible(True);spine.set_color('#aeb8be')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,fontsize=8.5,bbox_to_anchor=(.5,-.005))
fig.tight_layout(rect=(0,.045,1,1),w_pad=2,h_pad=1.2);save(fig,'replanning_controls_2026_11_15')

# Existing paired repair performance, with a stable denominator of 38 plans x 640 simulations.
rows=[]
for f in sorted((SUITE/'results/contracts').glob('D*.json')):
 d=read(f)
 if (d.get('selected_theta') or 0)<=0:continue
 pol=d['policies'];aa=pol[d['certified_repair_source']]['realizations'];bb=pol[d['certified_retention_source']]['realizations'];nn=pol['nominal_mpc']['realizations']
 for a,b,n in zip(aa,bb,nn):
  assert (a['corner'],a['profile'])==(b['corner'],b['profile'])==(n['corner'],n['profile'])
  rows.append([100*(b['priority_cost']-a['priority_cost'])/b['priority_cost'],100*(a['tts_veh_h']-b['tts_veh_h'])/b['tts_veh_h'],100*(b['priority_cost']-a['priority_cost'])/(b['priority_cost']-n['priority_cost'])])
assert len(rows)==24320
q=np.percentile(rows,[5,25,50,75,95],axis=0)
common.save(DATA/'repair_traffic_statistics.json',{'count':len(rows),'percentiles':[5,25,50,75,95],
 'columns':['priority_cost_reduction_percent','tts_increase_percent','requested_benefit_recovered_percent'],'values':q.tolist()})
fig,axs=plt.subplots(1,3,figsize=(8.5,2.5))
for j,(ax,title) in enumerate(zip(axs,['Priority-cost reduction','TTS increase','Requested benefit recovered'])):
 ax.bxp([dict(med=q[2,j],q1=q[1,j],q3=q[3,j],whislo=q[0,j],whishi=q[4,j],fliers=[])],positions=[1],widths=.35,patch_artist=True,boxprops={'facecolor':'#d4e9e8'},medianprops={'color':TEAL,'lw':2})
 ax.set(xticks=[],ylabel='Percent',title=title);ax.grid(axis='y',alpha=.15);ax.text(1.22,q[2,j],f'{q[2,j]:.1f}',fontsize=8,color=TEAL,va='center')
fig.tight_layout(w_pad=2);save(fig,'repair_traffic_tradeoffs_2026_11_15')

# Replace only the vocabulary in the existing acceptance-map asset via its source script.
common.save(DATA/'figure_validation.json',validation)
print('Generated scenario, congestion/input, replacement/input, and repair-cost figures.',flush=True)
