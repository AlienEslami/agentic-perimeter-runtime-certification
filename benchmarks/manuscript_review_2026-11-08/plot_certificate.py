from pathlib import Path
import csv
import hashlib
import json
import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
sys.path.insert(0,str(SUITE))
from common import arrival, residual, fixture, PROFILES

record=json.loads((SUITE/'results/contracts/D027.json').read_text())
certificate_file=HERE/'results/D027_repair_continuous_certificate.json'
cert=json.loads(certificate_file.read_text())['certificate']
assert cert['accepted'] and cert['certified_until_h']==1
steps=cert['interval_steps']; c=record['contract']
edges=np.array([s['left_h'] for s in steps]+[steps[-1]['right_h']])
lower=np.array([100*s['independent_physics']['autonomous_queue']['continuous'][0] for s in steps])
upper=np.array([100*s['independent_physics']['autonomous_queue']['continuous'][1] for s in steps])
assert max(upper)<=2000
replays=[]
for sign in (-1,1):
    for profile in PROFILES:
        start=1/3; z=fixture()['center'][6]+sign*c['width_vehicles']/100
        times=[]; values=[]
        for left,right in zip(np.arange(20,60,5)/60,np.arange(25,65,5)/60):
            command=record['policies']['grid_repair']['commands'][0 if left<25/60-1e-10 else 1][2]
            def rhs(t,y):
                clock=min(t,np.nextafter(right,left))
                return [arrival(t,c)+residual(clock,c['residual_veh_h']/100,profile)-min(60*y[0],120*command)]
            sol=solve_ivp(rhs,(left,right),[z],method='DOP853',rtol=1e-12,atol=1e-13,max_step=1/720,dense_output=True)
            assert sol.success
            ts=np.linspace(left,right,301); zs=100*sol.sol(ts)[0]
            times.extend(ts); values.extend(zs);z=sol.y[0,-1]
        times=np.array(times);values=np.array(values)
        ix=np.minimum(np.searchsorted(edges,times,side='right')-1,len(upper)-1)
        assert np.min(values-lower[ix])>=-1e-6
        assert np.max(values-upper[ix])<=1e-6
        replays.append((sign,profile,times,values))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(7.2,3.25),layout='constrained')
ax.stairs(upper,edges*60,baseline=lower,fill=True,color='#D8EBEE',label='Certified queue enclosure')
ax.stairs(upper,edges*60,baseline=None,color='#006C78',linewidth=2,label='Certified upper bound')
for i,(_,_,times,values) in enumerate(replays):
    ax.plot(times*60,values,color='#66727E',alpha=.55,lw=.8,label='Selected replays' if i==0 else None)
ax.axhline(2000,color='#B5472B',ls='--',lw=1.7,label='Storage limit: 2000 vehicles')
ax.set(xlim=(20,60),ylim=(0,2150),xlabel='Time (minutes)',ylabel='Exterior queue (vehicles)')
ax.set_xticks(np.arange(20,61,5));ax.grid(axis='y',alpha=.15)
ax.legend(loc='upper left',fontsize=8.5,frameon=False,bbox_to_anchor=(0,.88))
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
fig.savefig(FIG/'certified_queue_enclosure.pdf');fig.savefig(FIG/'certified_queue_enclosure.png',dpi=220)
with (HERE/'results/continuous_queue_bounds.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['left_h','right_h','lower_vehicles','upper_vehicles'])
    writer.writerows(zip(edges[:-1],edges[1:],lower,upper))
with (HERE/'results/queue_bound_replays.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['initial_queue_sign','profile','time_h','queue_vehicles'])
    for sign,profile,t,v in replays:writer.writerows((sign,profile,a,b) for a,b in zip(t,v))
summary=dict(certificate_sha256=hashlib.sha256(certificate_file.read_bytes()).hexdigest(),
             intervals=len(steps),replays=len(replays),maximum_continuous_upper_vehicles=float(max(upper)),
             minimum_storage_margin_vehicles=float(2000-max(upper)),sampled_containment='pass',
             scope='Bounds apply at every time in each interval and to all contract-admissible measurable residuals.')
(HERE/'results/figure_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
