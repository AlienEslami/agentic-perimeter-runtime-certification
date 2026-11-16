"""Matched simulations and checks of the three-region recovery certificates."""
from study import *
from mpmath import mp

N=64
def load(name):return json.loads(gzip.decompress((OUT/name).read_bytes()))

def ensemble(net,initial,errors):
    count=len(initial);y=np.c_[initial,np.zeros((count,2))].ravel()
    all_t=[];all_y=[]
    for k in range(12):
        left,right=k/12,(k+1)/12;u=net.command(Fraction(k,12));forecast=net.arrival(Fraction(k,12))
        def rhs(t,flat):
            state=flat.reshape(count,14);c=state[:,:9].reshape(count,3,3);totals=c.sum(2)
            rates=v.MFD_A*totals**2+v.MFD_B*totals+v.MFD_C
            departure=c*rates[:,:,None];outgoing=departure*u[:3][None,:,None]
            outgoing[:,np.arange(3),np.arange(3)]=0
            internal=net.demand.reshape(1,3,3)-outgoing+np.roll(outgoing,1,axis=1)
            internal[:,np.arange(3),np.arange(3)]-=departure[:,np.arange(3),np.arange(3)]
            adm=np.minimum(60*state[:,9:12],120*u[3:])
            regional=internal.reshape(count,9)+adm@net.B.T
            queues=forecast+errors[:,k,:]-adm
            tts=100*state[:,:12].sum(1)
            priority=tts+300*totals[:,0]
            return np.c_[regional,queues,tts,priority].ravel()
        times=np.linspace(left,right,51)
        sol=solve_ivp(rhs,(left,right),y,method='DOP853',rtol=2e-10,atol=2e-12,max_step=1/240,t_eval=times)
        assert sol.success
        points=sol.y.T.reshape(-1,count,14)
        all_t.extend(times if not all_t else times[1:]);all_y.extend(points if not all_y else points[1:])
        y=points[-1].ravel()
    return np.array(all_t),np.array(all_y)

def verify_saved(proof,net):
    assert proof['accepted'] and v.digest(proof['descriptor'])==proof['descriptor_sha256']
    # Recompute every saved interval enclosure from its predecessor; the
    # independent simulations below are additional diagnostics, not the proof.
    previous=proof['initial'];rho=np.array(previous['rho']);G=np.array(previous['G']);center=np.array(previous['center']);queues=previous['queues']
    for step in proof['steps']:
        center,G,rho,queues,again=v.validated_step(net,'class',center,G,rho,queues,Fraction(step['left']),Fraction(step['right']),3.)
        assert again==step
    assert np.all(np.array([r['upper_vehicles'] for r in proof['service_rows']])<=3300)

def check_interval_jacobian():
    rng=np.random.default_rng(20261116);net=candidate(.45);net.refined=True
    max_field_error=0.;max_fd_error=0.
    for k in range(160):
        center=np.r_[rng.uniform(.5,30,9),rng.uniform(4,20,3)]
        widths=np.r_[rng.uniform(.01,4,9),rng.uniform(.01,2,3)]
        box=I(center-widths,center+widths);u=rng.uniform(.05,1,6)
        Jbox=net.jacobian(box,u,'class')
        for _ in range(8):
            x=rng.uniform(box.lo,box.hi);point=net.jacobian(I(x),u,'class')
            assert np.all(point.lo>=Jbox.lo-1e-10) and np.all(point.hi<=Jbox.hi+1e-10)
        # Independently differentiate the numerical balances at points away
        # from the admission switch, and compare the two field implementations.
        xp=np.r_[center[:9],np.full(3,10.)];a=np.array([60.,78.,60.])
        vfield=np.array(net.field(xp,u,a),float)
        max_field_error=max(max_field_error,float(np.max(abs(vfield-field(net,xp,u,a)))))
        fd=np.column_stack([(field(net,xp+np.eye(12)[i]*1e-5,u,a)-field(net,xp-np.eye(12)[i]*1e-5,u,a))/(2e-5) for i in range(12)])
        max_fd_error=max(max_fd_error,float(np.max(abs(fd-net.jacobian(I(xp),u,'class').mid()))))
    assert max_field_error<1e-10 and max_fd_error<1e-6
    # High-precision direct derivatives at physical corners independently
    # check the tightened interval evaluation in the actual certified boxes.
    mp.dps=60;tested=0
    proof=load('monotone_repair045_class_d25_w300_proof.json.gz')
    for rec in proof['steps'][::8]:
        box=I(rec['corridor_lower'],rec['corridor_upper']);bound=net.traffic_jacobian(box,rec['command'])
        for bits in __import__('itertools').product([0,1],repeat=3):
            point=np.array(box.lo)
            for reg in range(3):
                for j in range(3):point[3*reg+j]=[box.lo,box.hi][bits[j]][3*reg+j]
            mat=[[mp.mpf(0) for _ in range(9)] for _ in range(9)]
            for i in range(3):
                S=sum(mp.mpf(float(point[i*3+j])) for j in range(3));a,b,c=map(mp.mpf,[v.MFD_A,v.MFD_B,v.MFD_C])
                for j in range(3):
                    row=i*3+j
                    for k in range(3):
                        deriv=mp.mpf(float(point[row]))*(2*a*S+b)+(a*S*S+b*S+c if j==k else 0)
                        deriv*=1 if i==j else mp.mpf(rec['command'][i])
                        mat[row][i*3+k]-=deriv
                        if i!=j:mat[((i+1)%3)*3+j][i*3+k]+=deriv
            for i in range(9):
                for j in range(9):assert mp.mpf(float(bound.lo[i,j]))<=mat[i][j]<=mp.mpf(float(bound.hi[i,j]))
            tested+=1
    return dict(sampled_interval_points=1280,high_precision_corner_matrices=tested,
                independent_field_max_error=max_field_error,finite_difference_max_error=max_fd_error)

def run():
    rng=np.random.default_rng(20261116);net=candidate(.45)
    widths=np.r_[np.full(9,25/300.),np.full(3,.1)]
    initial=net.initial+rng.uniform(-1,1,(N,12))*widths
    errors=rng.uniform(-3,3,(N,12,3))
    for i,(ix,wx) in enumerate([(1,1),(1,-1),(-1,1),(-1,-1)]):
        initial[i]=net.initial+ix*widths;errors[i]=wx*3
    initial[4]=net.initial;errors[4]=0
    np.savez_compressed(OUT/'matched_scenarios.npz',initial_scaled=initial,errors_scaled=errors)
    report=dict(scenarios=N,random_seed=20261116,regional_total_error_vehicles=25,
                destination_component_error_vehicles=25/3,queue_error_vehicles=10,
                arrival_error_veh_h_per_queue=300,scenario_error_block_minutes=5,plans={})
    for name,weight,proofname in [('proposal',0,None),('repair',.45,'monotone_repair045_class_d25_w300_proof.json.gz'),('backup',1,'monotone_backup_class_d25_w300_proof.json.gz')]:
        net=candidate(weight);net.refined=True;t,states=ensemble(net,initial,errors)
        physical=states[:,:,:12]*100;regional=physical[:,:,:9].reshape(-1,N,3,3).sum(3);q=physical[:,:,9:]
        np.savez_compressed(OUT/f'{name}_ensemble.npz',time_h=t,state_vehicles=physical,regional_vehicles=regional,tts=states[-1,:,12],priority_cost=states[-1,:,13])
        row=dict(weight=weight,inputs=net.controls.tolist(),maximum_queues=q.max((0,1)).tolist(),
                 final_region_maxima=regional[-1].max(0).tolist(),final_region_adverse=regional[-1,0].tolist(),
                 adverse_tts=float(states[-1,0,12]),adverse_priority=float(states[-1,0,13]),
                 tts_median=float(np.median(states[-1,:,12])),priority_median=float(np.median(states[-1,:,13])))
        # Cross-check one ensemble member with an independently arranged RHS.
        single=trajectory(net,25,300,True,points=1201)
        assert np.max(abs(np.array(single['final_regional'])-regional[-1,0]))<1e-5
        if proofname:
            proof=load(proofname);verify_saved(proof,net)
            for rec in proof['steps']:
                left,right=map(lambda k:float(Fraction(rec[k])),['left','right'])
                mask=(t>=left-1e-13)&(t<=right+1e-13)
                upper=np.nextafter(100*np.nextafter(np.array(rec['row_predictor_upper'])[9:12]+np.array(rec['row_error_upper'])[9:12],np.inf),np.inf)
                assert np.all(regional[mask]<=upper+1e-5)
                qlo=100*np.array([q['continuous'][0] for q in rec['queues']]);qhi=100*np.array([q['continuous'][1] for q in rec['queues']])
                assert np.all(q[mask]>=qlo-1e-5) and np.all(q[mask]<=qhi+1e-5)
            row['certified_endpoint_upper']=[r['upper_vehicles'] for r in proof['service_rows']]
            row['certified_queue_upper']=[max(100*rec['queues'][i]['continuous'][1] for rec in proof['steps']) for i in range(3)]
            row['observed_check_seconds']=proof['wall_seconds']
            row['recomputed_intervals']=len(proof['steps'])
        report['plans'][name]=row
        print(name,row,flush=True)
    # Exact rational queue balance for all rejected lower grid weights.
    witnesses=[]
    for j in range(10):
        weight=j/20;u=Fraction.from_float(float(candidate(weight).controls[1,3]))
        zfinal=Fraction(510)+Fraction(7200)-Fraction(11000)*u
        witnesses.append(dict(weight=weight,queue1_final_lower=float(zfinal),overflow_proven=zfinal>2000))
    assert all(r['overflow_proven'] for r in witnesses[:9]) and not witnesses[9]['overflow_proven']
    report['exact_queue_witnesses']=witnesses
    report['arithmetic_checks']=check_interval_jacobian()
    report['repair_grid_conclusion']='0.45 passes all constraints; all lower 0.05-grid weights have a queue-overflow witness.'
    a,b,c=[report['plans'][key] for key in ['proposal','repair','backup']]
    report['adverse_recovered_priority_benefit_percent']=100*(c['adverse_priority']-b['adverse_priority'])/(c['adverse_priority']-a['adverse_priority'])
    report['adverse_tts_increase_vs_backup_percent']=100*(b['adverse_tts']/c['adverse_tts']-1)
    save('validated_results.json',report)
    print('VALIDATION PASSED',flush=True)

if __name__=='__main__':run()
