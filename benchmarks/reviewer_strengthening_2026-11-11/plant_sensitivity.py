"""Plant mismatch sensitivity only: these trajectories are not certified."""
import itertools,json,math
import numpy as np
from probe import HERE,get_job,save,common

def simulate(plan,gscale,dscale,step=2.5,job=None):
    job=get_job(f'depth_{plan}_4500') if job is None else job
    commands=np.array(job['commands']);arrival_bound=job['w']/100
    signs=np.repeat(np.array(list(itertools.product((-1.,1.),repeat=7))),5,axis=0)
    profiles=np.tile(np.arange(5),128);width=np.array(job['width']);x=np.array(job['center'])+signs*width
    initial=x.copy();h=step/3600;steps=round(1/h)
    peak=x[:,6].copy();minimum=np.full(640,np.inf);balance=np.zeros(640)
    demand=common.plant.INTERNAL_DEMAND*dscale
    def field(y,t,u,k):
        n1,n2,n11,n22,n10,n20,z=y.T;u12,u21,u02,u20=u
        v1=gscale*(.0014877*n1*n1-.29815*n1+15.0912)
        v2=gscale*(.0014877*n2*n2-.29815*n2+15.0912)
        m11,m12,m10=n11*v1,(n1-n11-n10)*v1,n10*v1
        m22,m21,m20=n22*v2,(n2-n22-n20)*v2,n20*v2
        admission=np.minimum(60*z,120*u02)
        alternating=arrival_bound if int(((k+.5)*h+1e-10)*12)%2==0 else -arrival_bound
        errors=np.array([0,arrival_bound,-arrival_bound,alternating,arrival_bound*math.sin(12*math.pi*t)])[profiles]
        arrivals=common.plant.arrival(t,amplitude=.3,duration=1.)+errors
        d11,d12,d10,d22,d21,d20=demand
        f=np.stack((-m11-(m12+m10)*u12+m21*u21+d11+d12+d10,
            -m22+(m12+m10)*u12-m21*u21+admission-m20*u20+d22+d21+d20,
            -m11+m21*u21+d11,-m22+m12*u12+.5*admission+d22,
            -m10*u12+d10,-m20*u20+m10*u12+d20,arrivals-admission),axis=1)
        massrate=sum(demand)+arrivals-m11-m22-m20*u20
        return f,massrate
    for k in range(steps):
        t=k*h;u=commands[0 if (k+.5)*h<1/12 else 1]
        f1,m1=field(x,t,u,k);y2=x+h*f1/2
        f2,m2=field(y2,t+h/2,u,k);y3=x+h*f2/2
        f3,m3=field(y3,t+h/2,u,k);y4=x+h*f3
        f4,m4=field(y4,t+h,u,k);new=x+h/6*(f1+2*f2+2*f3+f4)
        for y in (x,y2,y3,y4,new):
            minimum=np.minimum(minimum,np.min(y@common.mpc.STATE_A.T+common.mpc.STATE_B,axis=1))
            peak=np.maximum(peak,y[:,6])
        balance+=h/6*(m1+2*m2+2*m3+m4);x=new
    residual=100*((x[:,0]+x[:,1]+x[:,6])-(initial[:,0]+initial[:,1]+initial[:,6])-balance)
    return dict(plan=plan,mfd_multiplier=gscale,demand_multiplier=dscale,step_seconds=step,
        simulated_runs=640,physical_passes=int(np.sum(minimum>=-1e-7)),service_passes=int(np.sum(100*x[:,0]<=3300+1e-6)),
        maximum_queue_vehicles=float(100*peak.max()),maximum_deadline_n1_vehicles=float(100*x[:,0].max()),
        maximum_final_n2_vehicles=float(100*x[:,1].max()),conservation_residual_vehicles=float(abs(residual).max()))

if __name__=='__main__':
    protocol=dict(scope='Out-of-certificate plant perturbations: numerical sensitivity, no formal guarantee or microscopic validation.',
        mfd_and_internal_demand_multipliers=[.95,1.,1.05],plans=['backup','repair'],
        initial_and_arrival_uncertainty='Same d=(25,25,25,25,25,25,10) vehicles, wbar=300 veh/h as congested experiment',
        corners=128,arrival_profiles=5,step_seconds=2.5)
    save(HERE/'plant_sensitivity_protocol.json',protocol)
    rows=[]
    for plan,g,d in itertools.product(['backup','repair'],[.95,1.,1.05],[.95,1.,1.05]):
        r=simulate(plan,g,d);rows.append(r);print(json.dumps(r),flush=True)
    # Independent step refinement for the lowest-production, highest-demand plant.
    refinement=[simulate(plan,.95,1.05,1.25) for plan in ('backup','repair')]
    for r in refinement:
        original=next(v for v in rows if v['plan']==r['plan'] and v['mfd_multiplier']==.95 and v['demand_multiplier']==1.05)
        assert r['physical_passes']==original['physical_passes'] and r['service_passes']==original['service_passes']
        assert abs(r['maximum_deadline_n1_vehicles']-original['maximum_deadline_n1_vehicles'])<.1
    save(HERE/'plant_sensitivity.json',dict(protocol=protocol,rows=rows,refinement=refinement))
