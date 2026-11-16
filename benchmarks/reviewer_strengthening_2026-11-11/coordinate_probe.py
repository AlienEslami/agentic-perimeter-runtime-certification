"""Exploratory fixed orthogonal coordinates; every bound is recomputed outward."""
import argparse,hashlib,json,time
import numpy as np
from scipy.linalg import schur
import probe
from probe import HERE,core,affine,save
from mfd_incident_mpc_certification import constant_command_mode
from mfd_deadline_certificates import inverse_interval,norm_upper,imat,I

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='backup');p.add_argument('--width',type=float,default=50)
    p.add_argument('--center',type=float,default=4500);a=p.parse_args()
    job=probe.get_job(f'depth_{a.plan}_4500');job['width']=[a.width/100]*6+[.1]
    point=np.array(job['center']);point[:6]*=a.center/4500
    mode=constant_command_mode(job['commands'][1])
    J=core.jacobian(I(point),mode,'gate').mid()[:6,:6]
    _,Q=schur(J,output='real');L=np.eye(7);L[:6,:6]=Q.T
    base_factory=probe.verifier
    def factory(*args):
        v=base_factory(*args)
        original=v.geometry
        def geometry(mode):
            geo=original(mode)
            if not geo.get('schur_probe'):
                inverse=inverse_interval(L)
                geo.update(L=L,T=inverse,schur_probe=True,
                    metric_selection=dict(strategy='fixed_regional_schur',selection_state=point.tolist(),L=L.tolist()),
                    coordinate_support=np.array([norm_upper(inverse[k]) for k in range(7)]),
                    support=np.array([norm_upper(imat(geo['rows'][k:k+1],inverse)) for k in range(len(geo['names']))]),
                    constraint_box_map=imat(geo['rows'],inverse).mag(),disturbance_factor=norm_upper(I(L[:,6])),
                    jacobian_record=core.transformed_jacobian_record(mode,L,inverse))
            return geo
        v.geometry=geometry
        return v
    probe.verifier=factory
    r=probe.check(job,'tight12','class_box',1.01,False)
    r['coordinate_probe']=dict(L=L.tolist(),selection_state=point.tolist(),source_sha256=hashlib.sha256(__file__.encode()).hexdigest(),exploratory=True)
    save(HERE/'results'/f'schur_{a.plan}_d{a.width}_center{a.center}.json',r)
    print(json.dumps({k:r['summary'][k] for k in ('accepted','minutes','seconds','service')}),flush=True)
