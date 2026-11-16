from run_study import *
import itertools,csv
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROFILES=['zero','constant_high','constant_low','alternating_5min','sinusoidal_10min']
CRITICAL=3391.93080685

def ensemble(commands,d=25,w=300,step_seconds=2):
    corners=np.array(list(itertools.product([-1.,1.],repeat=7)))
    widths=np.array([d]*6+[min(d,10)])/100
    x=np.tile(X0,(640,1))+np.repeat(corners,5,axis=0)*widths
    initial=x.copy();cost=np.zeros((640,4));inputs=np.zeros(640)
    peak=x[:,[0,1,6]].copy();mini=np.full(640,np.inf)
    first_cross=np.full((640,2),np.nan)
    h=step_seconds/3600; rows=[]
    def rhs(y,t,k):
        alt=1 if int((k*h+h/2)*12)%2==0 else -1
        residual=np.tile(np.array([0,1,-1,alt,np.sin(12*np.pi*t)])*w/100,128)
        f,dep=common.batch_field(y,u,forecast(t)+residual)
        c=100*np.stack([y[:,0]+y[:,1]+y[:,6],y[:,6],dep,y[:,0]+4*y[:,1]+y[:,6]],axis=1)
        return f,c,100*(sum(plant.INTERNAL_DEMAND)+forecast(t)+residual)
    for k in range(round(1/h)):
        t=k*h;u=np.asarray(commands[0 if t<1/12-1e-10 else 1])
        f1,c1,a1=rhs(x,t,k);x2=x+h*f1/2
        f2,c2,a2=rhs(x2,t+h/2,k);x3=x+h*f2/2
        f3,c3,a3=rhs(x3,t+h/2,k);x4=x+h*f3
        f4,c4,a4=rhs(x4,t+h,k)
        new=x+h/6*(f1+2*f2+2*f3+f4)
        cost+=h/6*(c1+2*c2+2*c3+c4);inputs+=h/6*(a1+2*a2+2*a3+a4)
        for state in [x,x2,x3,x4,new]:
            mini=np.minimum(mini,(state@mpc.STATE_A.T+mpc.STATE_B).min(axis=1))
            peak=np.maximum(peak,state[:,[0,1,6]])
        x=new
        for j in range(2):
            hit=np.isnan(first_cross[:,j])&(100*x[:,j]<=CRITICAL)
            first_cross[hit,j]=(t+h)*60
    balance=100*(x[:,[0,1,6]].sum(axis=1)-initial[:,[0,1,6]].sum(axis=1))-inputs+cost[:,2]
    for i in range(640):
        rows.append(dict(corner=''.join('+' if q>0 else '-' for q in corners[i//5]),profile=PROFILES[i%5],
            physical_pass=bool(mini[i]>=-1e-7),service_pass=bool(100*x[i,0]<=LIMIT),
            peak_n1=float(100*peak[i,0]),peak_n2=float(100*peak[i,1]),peak_queue=float(100*peak[i,2]),
            final_n1=float(100*x[i,0]),final_n2=float(100*x[i,1]),final_queue=float(100*x[i,6]),
            tts=float(cost[i,0]),queue_time=float(cost[i,1]),departures=float(cost[i,2]),priority_cost=float(cost[i,3]),
            conservation_error=float(balance[i]),minimum_state_margin_vehicles=float(100*mini[i]),
            first_below_critical_minutes=[None if np.isnan(v) else float(v) for v in first_cross[i]]))
    stats={k:dict(min=min(r[k] for r in rows),max=max(r[k] for r in rows),median=float(np.median([r[k] for r in rows])))
           for k in ['peak_n1','peak_n2','peak_queue','final_n1','final_n2','final_queue','tts','queue_time','departures','priority_cost']}
    return dict(d=d,w=w,runs=640,physical_passes=sum(r['physical_pass'] for r in rows),
      service_passes=sum(r['service_pass'] for r in rows),statistics=stats,
      max_conservation_error=max(abs(r['conservation_error']) for r in rows),realizations=rows)

def trace(commands,d=25,w=300):
    widths=np.array([d]*6+[min(d,10)])/100
    y=np.r_[X0+widths,np.zeros(4)];parts=[]
    for left,right in [(0,1/12),(1/12,.25),(.25,1/3),(1/3,1.)]:
        u=np.asarray(commands[0 if left<1/12-1e-10 else 1])
        def f(t,y):
            fx,dep=common.batch_field(y[None,:7],u,forecast(t)+w/100)
            return np.r_[fx[0],100*(y[0]+y[1]+y[6]),100*y[6],100*dep[0],100*(y[0]+4*y[1]+y[6])]
        sol=solve_ivp(f,(left,right),y,method='DOP853',rtol=1e-11,atol=1e-12,max_step=1/720,dense_output=True)
        assert sol.success
        ts=np.linspace(left,right,round((right-left)*3600)+1)
        ys=sol.sol(ts).T
        parts.append(np.column_stack([ts,ys,np.tile(u,(len(ts),1))]));y=sol.y[:,-1]
    return np.concatenate(parts)

def main():
    all_results={}
    for name in ['proposal','backup']:
        commands=json.loads((OUT/(name+'_planning.json')).read_text())['commands']
        for d,w in [(25,300),(1,60)]:
            rec=ensemble(commands,d,w);save(OUT/f'{name}_d{d}_w{w}_ensemble.json',rec)
            tr=trace(commands,d,w)
            save_trace(OUT/f'{name}_d{d}_w{w}_trace.csv',tr)
            adverse=next(r for r in rec['realizations'] if r['corner']=='+++++++' and r['profile']=='constant_high')
            errors={key:float(tr[-1,col]-adverse[key]) for key,col in [('tts',8),('queue_time',9),('departures',10),('priority_cost',11)]}
            errors.update(final_n1=float(100*tr[-1,1]-adverse['final_n1']),final_n2=float(100*tr[-1,2]-adverse['final_n2']))
            assert max(abs(v) for v in errors.values())<.1,errors
            rec.pop('realizations');rec['dop853_vs_rk4_errors']=errors;all_results[f'{name}_d{d}_w{w}']=rec
            print(name,d,w,json.dumps(rec),flush=True)
    save(OUT/'ensemble_summary.json',all_results)

def save_trace(path,tr):
    values=tr.copy();values[:,1:8]*=100
    np.savetxt(path,values,delimiter=',',header='time_h,n1_vehicles,n2_vehicles,n11_vehicles,n22_vehicles,n10_vehicles,n20_vehicles,z_vehicles,tts_veh_h,queue_time_veh_h,departures_vehicles,priority_cost_veh_h,u12,u21,u02,u20',comments='')

if __name__=='__main__':main()
