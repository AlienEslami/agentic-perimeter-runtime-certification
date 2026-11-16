"""Independent numerical and record audits for the implementation refinement."""
import hashlib,itertools,json
from fractions import Fraction
import numpy as np
import mpmath as mp
from probe import HERE,affine,vector_imat,power_monomials,verifier,save

def validate_arithmetic():
    rng=np.random.default_rng(1111);count=0
    for m,k,n in [(7,7,7),(31,7,7),(49,18,1),(7,1,7)]:
        for trial in range(100):
            al=rng.normal(size=(m,k));ah=al+rng.random((m,k))
            bl=rng.normal(size=(k,n));bh=bl+rng.random((k,n))
            a=affine.I(al,ah);b=affine.I(bl[:,0],bh[:,0]) if n==1 else affine.I(bl,bh)
            old=affine.imat(a,b);new=vector_imat(a,b)
            assert np.array_equal(old.lo,new.lo) and np.array_equal(old.hi,new.hi)
            count+=1
    powers=0
    for _ in range(100):
        lo=rng.uniform(-10,10,7);hi=lo+rng.uniform(0,15,7)
        for exponent in (1,2,3):
            ex=np.eye(7,dtype=int)*exponent
            value=power_monomials(ex,affine.I(lo,hi))
            for k in range(7):
                points=[Fraction.from_float(float(lo[k])),Fraction.from_float(float(hi[k]))]
                if lo[k]<=0<=hi[k]:points.append(Fraction(0))
                exact=[x**exponent for x in points]
                assert Fraction.from_float(float(value.lo[k]))<=min(exact)
                assert Fraction.from_float(float(value.hi[k]))>=max(exact)
                powers+=1
    # High-precision matrix exponential provides a separate numerical audit of bounds.
    ns=verifier('tight12').step.__globals__;mp.mp.dps=70
    for _ in range(5):
        A=rng.normal(size=(7,7))*20;h=affine.I(1/240)
        E,Q,_=ns['exponential_and_integral'](A,h)
        B=mp.matrix(14)
        for i in range(7):
            for j in range(7):B[i,j]=mp.mpf(float(A[i,j]))*mp.mpf(float(h.lo))
            B[i,i+7]=mp.mpf(float(h.lo))
        exact=mp.expm(B)
        for i,j in itertools.product(range(7),repeat=2):
            assert mp.mpf(float(E.lo[i,j]))<=exact[i,j]<=mp.mpf(float(E.hi[i,j]))
            assert mp.mpf(float(Q.lo[i,j]))<=exact[i,j+7]<=mp.mpf(float(Q.hi[i,j]))
    return dict(bit_identical_matrix_products=count,exact_rational_power_checks=powers,matrix_function_cases=5)

def audit_records():
    rows=[];added=[];lost=[]
    for path in sorted((HERE/'comparison').glob('*.json')):
        r=json.loads(path.read_text());s=r['summary'];rows.append(s)
        assert s['accepted']==any(a['passed'] for a in r['attempts'])
        for a in r['attempts']:
            if a['passed']:assert len(a['children'])==a['divisions']**2
            for ch in a['children']:
                c=ch['record']['certificate']
                assert c['proof_id']==hashlib.sha256(json.dumps(c['proof_descriptor'],sort_keys=True).encode()).hexdigest()
                if c['accepted']:
                    assert c['verified_continuous_physical'] and c['verified_scheduled_service']
                    assert c['certified_until_h']==1.
                    assert all(v['passed'] for v in c['service_check_results'])
                    assert all(v['passed'] for v in c['adapter_audit']['exact_actuator_checks'])
        old=HERE.parent/'fixed_portfolio_2026-11-10/results_final'/path.name
        if old.exists():
            before=json.loads(old.read_text())['summary']['accepted']
            if s['accepted'] and not before:added.append(s['id'])
            if before and not s['accepted']:lost.append(s['id'])
    groups={g:dict(total=sum(r['group']==g for r in rows),accepted=sum(r['group']==g and r['accepted'] for r in rows)) for g in set(r['group'] for r in rows)}
    return dict(groups=groups,added=added,lost=lost,completed=len(rows))

if __name__=='__main__':
    out=dict(arithmetic=validate_arithmetic(),records=audit_records())
    save(HERE/'validation.json',out);print(json.dumps(out,indent=2))
