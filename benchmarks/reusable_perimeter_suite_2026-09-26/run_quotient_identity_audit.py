"""Independent numerical audit of the algebraic five-state quotient.

The derivation in the report is algebraic; this detects implementation errors.
"""
from __future__ import annotations

import json
import numpy as np

from common import HERE, fixture, mpc, plant, save
from run_falsification import records
from run_reduced_adjoint_probe import reduced_jacobian
from mfd_incident_affine_reachability import I


def reduced_field(y, u, arrival):
    n1,n2,n11,b,z=y
    h1=.0014877*n1*n1-.29815*n1+15.0912
    h2=.0014877*n2*n2-.29815*n2+15.0912
    u12,u21,u02,u20=u
    assert u20==1.
    admission=min(60*z,120*u02)
    d11,d12,d10,d22,d21,d20=plant.INTERNAL_DEMAND
    p=n1-n11;q=n2-b
    return np.array([
        -(n11+u12*p)*h1+u21*q*h2+d11+d12+d10,
        -(b+u21*q)*h2+u12*p*h1+admission+d22+d21+d20,
        -n11*h1+u21*q*h2+d11,
        -b*h2+u12*p*h1+.5*admission+d22+d20,
        arrival-admission,
    ])


def audit():
    H=np.zeros((5,7))
    for r,k in enumerate((0,1,2,3,6)):H[r,k]=1
    H[3,5]=1
    rng=np.random.default_rng(20261001)
    commands=[np.asarray(r["policies"]["full_scenario_mpc"]["commands"]) for _,r in records()]
    max_field=max_jacobian=0.
    tested=0
    for command_pair in commands:
        for u in command_pair:
            for _ in range(20):
                x=np.asarray(fixture()["center"])+rng.uniform(-.4,.4,7)
                # Keep the branch away from its nondifferentiable tie.
                if abs(60*x[6]-120*u[2]) < .05:continue
                arrival=float(rng.uniform(75,81))
                seven=plant.field(x,u,arrival)
                five=reduced_field(H@x,u,arrival)
                max_field=max(max_field,float(np.max(abs(H@seven-five))))
                _,J7,_=mpc.field_jacobians(x,u,arrival)
                branch="supply" if 60*x[6]<120*u[2] else "gate"
                J5=reduced_jacobian(I(x),u,branch).mid()
                max_jacobian=max(max_jacobian,float(np.max(abs(H@J7-J5@H))))
                tested+=1
    assert tested>1000 and max_field<1e-11 and max_jacobian<1e-11
    result=dict(tested_state_command_pairs=tested,max_field_identity_error=max_field,
                max_jacobian_identity_error=max_jacobian,
                algebraic_identity="H f7(x,u)=f5(Hx,u) and H J7(x,u)=J5(Hx,u) H for u20=1")
    save(HERE/"results/directional_service_2026-10-01/quotient_identity_audit.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":audit()
