"""Independent adaptive integration of representative/worst sampled CBF cases."""
from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[1];SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26'
sys.path.insert(0,str(SUITE));import common
OUT=ROOT/'benchmarks/partb_traffic_presentation_2026-11-15'
d=json.loads((SUITE/'results/contracts/D027.json').read_text());c=d['contract'];commands=np.array(d['policies']['nominal_mpc']['commands'])
saved=json.loads((OUT/'cbf_ensemble.json').read_text())['realizations']
selected={}
for r in [max(saved,key=lambda r:r['deadline_n1_vehicles']),min(saved,key=lambda r:r['deadline_n1_vehicles']),max(saved,key=lambda r:r['peak_queue_vehicles'])]+[next(r for r in saved if r['corner']=='+++++++' and r['profile']==p) for p in common.PROFILES]:
 selected[(r['corner'],r['profile'])]=r
checks=[]
for (corner,profile),record in selected.items():
 x=np.array(common.fixture()['center'])+np.array([1 if v=='+' else -1 for v in corner])*c['width_vehicles']/100
 y=np.r_[x,np.zeros(4)];peak=x[6];minimum=np.inf
 for k in range(240):
  left=1/3+k/360;right=left+1/360
  nominal=commands[0 if k<30 else 1];u=common.cbf(y[:7],nominal,common.arrival(left,c),c['residual_veh_h']/100);assert u is not None
  def fun(t,z):
   clock=min(t,np.nextafter(right,left));rr=common.residual((left+right)/2 if profile=='alternating_5min' else clock,c['residual_veh_h']/100,profile)
   f=common.plant.field(z[:7],u,common.arrival(clock,c)+rr)
   return np.r_[f,100*(z[0]+z[1]+z[6]),100*z[6],100*common.plant.departure_rate(z[:7],u),100*(z[0]+c['weight']*z[1]+z[6])]
  sol=solve_ivp(fun,(left,right),y,method='DOP853',rtol=1e-11,atol=1e-12,max_step=1/1440,dense_output=True);assert sol.success
  states=sol.sol(np.linspace(left,right,9))[:7].T
  peak=max(peak,states[:,6].max());minimum=min(minimum,(states@common.mpc.STATE_A.T+common.mpc.STATE_B).min());y=sol.y[:,-1]
 observed={'tts_veh_h':float(y[7]),'priority_cost':float(y[10]),'peak_queue_vehicles':float(100*peak),'deadline_n1_vehicles':float(100*y[0])}
 diff={key:observed[key]-record[key] for key in observed}
 assert max(abs(v) for v in diff.values())<.1,diff
 assert minimum>=-1e-7 and observed['deadline_n1_vehicles']>c['deadline_vehicles']
 checks.append({'corner':corner,'profile':profile,'adaptive_minus_rk4':diff})
common.save(OUT/'cbf_integration_audit.json',{'method':'DOP853, maximum 2.5-second step, same 10-second feedback updates','checks':checks})

# All columns are drawn from the same case and the same 640 (corner, profile) pairs.
full=json.loads((SUITE/'results/full_scenario/D027.json').read_text())
policies={'Nominal MPC':d['policies']['nominal_mpc'],'CBF-QP':json.loads((OUT/'cbf_ensemble.json').read_text()),
 'Current plan':d['policies']['continuation'],'Nominal-plan repair':d['policies']['grid_repair'],
 'Full-scenario MPC':full['policies']['full_scenario_mpc']}
table=[]
keys=None
for name,p in policies.items():
 rows=p['realizations'];pairs={(r['corner'],r['profile']) for r in rows}
 if keys is None:keys=pairs
 assert len(rows)==len(pairs)==640 and pairs==keys
 s=p['summary'];both=sum(r['physical_pass'] and r['service_pass'] for r in rows)
 table.append({'plan':name,'median_tts':s['tts_veh_h']['median'],'median_priority_cost':s['priority_cost']['median'],
 'median_queue_waiting':s['queue_veh_h']['median'],'maximum_queue':s['peak_queue_vehicles']['maximum'],
 'maximum_deadline_n1':s['deadline_n1_vehicles']['maximum'],'physical_passes':s['physical_passes'],'service_passes':s['service_passes'],'both_passes':both})
common.save(OUT/'matched_controller_table.json',{'case':'D027','contract':c,'samples_per_plan':640,'rows':table})
print(json.dumps({'audit_cases':len(checks),'rows':table},indent=2))
