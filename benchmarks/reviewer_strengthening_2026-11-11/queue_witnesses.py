from fractions import Fraction
import json,math
import numpy as np
from probe import HERE,common,save

F=Fraction.from_float
old=HERE.parent/'congested_recovery_2026-11-10/results'
p=np.array(json.loads((old/'proposal_planning.json').read_text())['commands'])
b=np.array(json.loads((old/'backup_planning.json').read_text())['commands'])
forecast=60*(1+F(.3)*(1-F(.25)-F(1/12)/2))
rows=[]
for k in range(13):
    weight=k/20;u=common.compile_commands((1-weight)*p+weight*b)[0]
    lower=100*(F(float(5+.1))+forecast+6-120*sum(dt*F(float(v[2])) for dt,v in zip([Fraction(1,12),Fraction(11,12)],u)))
    assert lower>2000
    rows.append(dict(weight=weight,exact_lower=str(lower),lower_vehicles_rounded_down=math.floor(lower*100)/100))
save(HERE/'queue_witnesses_w600.json',dict(argument='a(z,u02)<=120 u02 in stored 100-vehicle units, regardless of the active branch',rows=rows))
print(rows[-1])
