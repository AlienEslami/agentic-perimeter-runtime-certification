from pathlib import Path
from fractions import Fraction
import argparse, gzip, hashlib, json, sys, time
import numpy as np
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import mfd_multiregion_transfer as v
import refined
from mfd_deadline_certificates import I, imat, isum
OUT=HERE/'results'

def save(name,data):
    p=OUT/name
    if p.suffix=='.gz':p.write_bytes(gzip.compress(json.dumps(data,allow_nan=False).encode(),compresslevel=5))
    else:p.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

class Recovery(v.Ring):
    def __init__(self,base=6000,commands=None):
        super().__init__(3)
        self.initial[:self.nc]*=1.5
        self.initial[self.nc:]=5.
        self.base=base/100.
        self.deadline=33.
        self.controls=np.asarray(commands if commands is not None else [[1,1,1,0,0,0],[1,1,1,.6,.6,.6]],float)
        self.plan_times=[Fraction(0),Fraction(1,12),Fraction(1)]
        self.knots=sorted(set(self.plan_times+[Fraction(1,4),Fraction(3,4)]))
        self.refined=False
    def traffic_jacobian(self,box,command):
        return refined.traffic_jacobian(self,box,command) if self.refined else super().traffic_jacobian(box,command)
    def command(self,t):return self.controls[0 if t<Fraction(1,12) else 1].copy()
    def arrival(self,t):
        a=np.full(3,self.base)
        if Fraction(1,4)<=t<Fraction(3,4):a[0]*=1.3
        return a
    def descriptor(self):
        d=super().descriptor()
        d['installed_plan']=[dict(left=str(a),right=str(b),command=self.command(a).tolist()) for a,b in zip(self.plan_times[:-1],self.plan_times[1:])]
        d['service']['all_regional_upper_vehicles']=100*self.deadline
        d['base_arrival_veh_h']=100*self.base
        d['arrival_surge']=dict(queue=0,amplitude=.3,start='1/4',end='3/4')
        return d

def field(net,x,u,a):
    c=x[:9].reshape(3,3); totals=c.sum(axis=1)
    rates=v.MFD_A*totals**2+v.MFD_B*totals+v.MFD_C
    departure=c*rates[:,None]
    outgoing=departure*u[:3,None]; np.fill_diagonal(outgoing,0.)
    internal=net.demand.reshape(3,3)-outgoing+np.roll(outgoing,1,axis=0)
    internal[np.diag_indices(3)]-=np.diag(departure)
    adm=np.minimum(60*x[9:],120*u[3:])
    return np.r_[internal.ravel()+net.B@adm,a-adm]

def trajectory(net,d=0.,w=0.,adverse=False,profile=0,points=1201):
    widths=np.r_[np.full(9,d/300.),np.full(3,min(d,10)/100.)]
    x0=net.initial+widths if adverse else net.initial.copy()
    x=np.r_[x0,0.,0.]
    times=[]; states=[]
    knots=sorted(set(net.knots+[Fraction(1,2)]))
    for a,b in zip(knots[:-1],knots[1:]):
        u=net.command(a); forecast=net.arrival(a)
        def rhs(t,y):
            error=np.full(3,w/100.) if profile==0 else (w/100.)*np.where((np.arange(3)+int(t*12))%2,-1.,1.)
            dx=field(net,y[:12],u,forecast+error)
            totals=y[:9].reshape(3,3).sum(axis=1)
            return np.r_[dx,100*sum(y[:12]),100*(4*totals[0]+sum(totals[1:])+sum(y[9:12]))]
        tt=np.linspace(float(a),float(b),max(2,round((float(b-a))*(points-1))+1))
        sol=solve_ivp(rhs,(float(a),float(b)),x,method='DOP853',rtol=2e-10,atol=2e-12,max_step=1/720,t_eval=tt)
        if not sol.success:raise RuntimeError(sol.message)
        times.extend(sol.t[1:] if times else sol.t)
        states.extend(sol.y.T[1:] if states else sol.y.T)
        x=sol.y[:,-1]
    z=np.asarray(states); count=100*z[:,:12]; n=count[:,:9].reshape(-1,3,3).sum(2)
    return dict(time_h=list(map(float,times)),state_vehicles=count.tolist(),
                regional_vehicles=n.tolist(),tts=float(x[12]),priority_cost=float(x[13]),
                maximum_queues=count[:,9:].max(0).tolist(),final_regional=n[-1].tolist(),
                maximum_regional=n.max(0).tolist(),minimum_component=float(count.min()))

def explore():
    rows=[]
    for base in [4000,6000]:
        for admission in [.4,.5,.6,.65,.75]:
            net=Recovery(base,[[1,1,1,0,0,0],[1,1,1,admission,.65,.65]])
            for d,w in [(0,0),(75,300)]:
                tr=trajectory(net,d,w,adverse=d>0,points=241)
                row=dict(base_arrivals=base,entry_1=admission,d=d,w=w,**{k:tr[k] for k in ['maximum_queues','final_regional','maximum_regional','tts','priority_cost']})
                rows.append(row); print(json.dumps(row),flush=True)
    save('exploratory_traffic.json',rows)

def baseline():
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'vendor').glob('*.py')}
    proof=v.verify(v.Ring(3),'inventory',25,60,hashes)
    audit=v.audit_record(proof)
    save('baseline_repeated_proof.json.gz',proof)
    save('baseline_repeated_audit.json',audit)
    print(json.dumps(dict(accepted=proof['accepted'],seconds=proof['wall_seconds'],service=proof['service_rows'],audit=audit)),flush=True)

def candidate(weight):
    # A prescribed request reduces entry to Region 1, retaining all other inputs.
    proposal=np.array([[1,1,1,0,0,0],[1,1,1,.495,.5,.5]],float)
    backup=np.array([[1,1,1,0,0,0],[1,1,1,.55,.5,.5]],float)
    return Recovery(6000,(1-weight)*proposal+weight*backup)

def verify_new(net,method,d,w,step=Fraction(1,240)):
    started=time.perf_counter(); residual=w/100.
    widths=np.r_[np.full(9,d/300.),np.full(3,min(d,10)/100.)]
    L=net.transform if method=='inventory' else np.eye(net.n)
    T=net.inverse if method=='inventory' else np.eye(net.n)
    initial_G=imat(L,np.diag(widths)); G=initial_G.mid(); rho=v.rowsum(initial_G-I(G))
    center=net.initial.copy(); initial_box=I(center)+I(-widths,widths)
    qbox=initial_box[net.nc:]
    queues=[[float(qbox.lo[i]),float(qbox.hi[i])] for i in range(net.r)]
    initial=dict(center=center.tolist(),G=G.tolist(),rho=rho.tolist(),queues=queues)
    hashes={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'refined.py',HERE/'PROTOCOL.md',HERE/'PLAN_SELECTION.md',*(HERE/'vendor').glob('*.py')]}
    snapshot=HERE/'source_snapshots'/v.digest(hashes)[:12];snapshot.mkdir(parents=True,exist_ok=True)
    for name,digest in hashes.items():
        dest=snapshot/Path(name).name
        if not dest.exists():dest.write_bytes((HERE/name).read_bytes())
        assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    descriptor=dict(network=net.descriptor(),method=method,tight_quadratic_range=net.refined,regional_error_vehicles=d,
                    physical_half_widths=widths.tolist(),arrival_error_veh_h_per_queue=w,
                    transform=L.tolist(),inverse=T.tolist(),initial_G=G.tolist(),initial_rho=rho.tolist(),
                    nominal_step=str(step),minimum_step='1/1920',source_sha256=hashes)
    # Positivity and finite-time existence use the actual changed arrival rates.
    arrivals=np.array([net.arrival(t) for t in net.knots[:-1]])
    margin=I(4.)*I(v.MFD_A)*I(v.MFD_C)-I(v.MFD_B)*I(v.MFD_B)
    mass=isum(initial_box[k] for k in range(net.n))+isum(I(q) for q in net.demand)+isum(I(a)+I(residual) for a in arrivals.max(0))
    positive=dict(passed=bool(margin.lo>0 and v.MFD_A>0 and initial_box.lo.min()>=0 and net.demand.min()>=0 and arrivals.min()-residual>=0),
                  four_ac_minus_b_squared_lower=float(margin.lo),
                  minimum_arrival_veh_h=float(100*(arrivals.min()-residual)),
                  one_hour_total_mass_upper_vehicles=float((100*mass).hi))
    assert positive['passed']
    assert np.min((I(net.limits)-imat(net.rows,initial_box)).lo)>=0
    gate_checks=[]
    for now in net.plan_times[:-1]:
        exact=[Fraction.from_float(float(a)) for a in net.command(now)]
        assert all(0<=a<=1 for a in exact)
        gate_checks.append(dict(left=str(now),exact_command=list(map(str,exact)),passed=True))
    steps=[]; failed=[]; left=Fraction(0); failure=None
    while left<1:
        knot=min(k for k in net.knots if k>left); h=min(step,knot-left)
        while True:
            try:
                result=v.validated_step(net,method,center,G,rho,queues,left,left+h,residual)
                break
            except v.Failure as ex:
                failed.append(dict(reason=ex.reason,evidence=ex.evidence))
                if h<=Fraction(1,1920):
                    failure=dict(kind=ex.reason,left=str(left));break
                h/=2
        if failure:break
        center,G,rho,queues,record=result
        steps.append(record);left+=h
        if min(record['physical_margins'])<0:
            failure=dict(kind='physical_constraint_not_certified',right=str(left),
                         rows=[net.names[j] for j,m in enumerate(record['physical_margins']) if m<0]);break
    physical=failure is None and left==1; services=[]
    if physical:
        rows=imat(net.regional_rows,T)
        upper=imat(net.regional_rows,center)+I(v.rowsum(imat(rows,G)))+imat(I(rows.mag()),rho)
        services=[dict(region=i,upper_vehicles=float((upper[i]*I(100.)).hi),limit_vehicles=100*net.deadline,
                       passed=bool(upper[i].hi<=net.deadline)) for i in range(3)]
    service=physical and all(q['passed'] for q in services)
    if physical and not service:failure=dict(kind='service_not_certified')
    proof=dict(descriptor=descriptor,descriptor_sha256=v.digest(descriptor),initial=initial,steps=steps,
               failed_trials=failed,structural_positivity=positive,exact_gate_checks=gate_checks,
               continuous_physical_verified=physical,scheduled_service_verified=service,
               accepted=physical and service,service_rows=services,failure=failure,reached_time=str(left),
               terminal=dict(center=center.tolist(),G=G.tolist(),rho=rho.tolist(),queues=queues),
               wall_seconds=time.perf_counter()-started)
    previous=initial;clock=Fraction(0)
    for r in steps:
        assert Fraction(r['left'])==clock
        for k,kprev in [('center','center'),('initial_G','G'),('initial_rho','rho'),('queue_initial','queues')]:
            assert r[k]==previous[kprev]
        assert r['command']==net.command(clock).tolist() and r['arrival']==net.arrival(clock).tolist()
        assert not any(clock<k<Fraction(r['right']) for k in net.knots)
        assert np.all(np.array(r['attempts'][-1]['radius'])<np.array(r['attempts'][-1]['trial']))
        assert r['generator_or_remainder_contracted'] is False
        clock=Fraction(r['right']);previous=dict(center=r['next_center'],G=r['next_G'],rho=r['next_rho'],queues=[q['endpoint'] for q in r['queues']])
    proof['chain_audit']=dict(passed=True,steps=len(steps),all_model_and_plan_knots_split=True)
    return proof

def certify(method,d,w,weight):
    net=candidate(weight)
    name=f'candidate_l{weight:.2f}_{method}_d{d:g}_w{w:g}'
    proof=verify_new(net,method,d,w)
    save(name+'_proof.json.gz',proof)
    summary=dict(name=name,weight=weight,method=method,d=d,w=w,accepted=proof['accepted'],
                 reached_minutes=60*float(Fraction(proof['reached_time'])),failure=proof['failure'],
                 services=proof['service_rows'],seconds=proof['wall_seconds'],steps=len(proof['steps']))
    save(name+'_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['explore','baseline','certify'])
    p.add_argument('--method',default='class');p.add_argument('--d',type=float,default=75);p.add_argument('--w',type=float,default=300)
    p.add_argument('--weight',type=float,default=1);a=p.parse_args()
    if a.action=='certify':certify(a.method,a.d,a.w,a.weight)
    else:globals()[a.action]()
