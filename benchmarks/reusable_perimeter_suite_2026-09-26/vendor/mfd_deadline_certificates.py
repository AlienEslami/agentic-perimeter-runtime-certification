#!/usr/bin/env python3
"""Deadline certificates for the unchanged installed MFD library.

Outward-rounded interval arithmetic bounds a quadratic predictor's exact
degree-eight residual and the Jacobian on a self-validating error tube.
Binary floating-point installed coefficients define the mathematical plant.
No numerical ODE solver tolerance is treated as a proof bound.
"""
from __future__ import annotations
import json
import math
import time
from pathlib import Path
import numpy as np
from mpmath import iv
from scipy.linalg import eigvalsh
import mfd_cubic_pilot as p
import mfd_cubic_nonstationary_study as s

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproducibility/results/mfd_deadline_certificate_results.json"
DN, UP = -np.inf, np.inf


class I:
    __array_priority__ = 10000
    def __init__(self, lo, hi=None):
        if isinstance(lo, I):
            self.lo, self.hi = lo.lo, lo.hi
        else:
            self.lo = np.asarray(lo, dtype=float)
            self.hi = np.asarray(lo if hi is None else hi, dtype=float)
    def __add__(self, other):
        b = I(other)
        return I(np.nextafter(self.lo+b.lo, DN), np.nextafter(self.hi+b.hi, UP))
    __radd__ = __add__
    def __neg__(self): return I(-self.hi, -self.lo)
    def __sub__(self, other): return self + (-I(other))
    def __rsub__(self, other): return I(other) + (-self)
    def __mul__(self, other):
        b = I(other)
        products = np.array([self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi])
        return I(np.nextafter(np.min(products, axis=0), DN), np.nextafter(np.max(products, axis=0), UP))
    __rmul__ = __mul__
    def __getitem__(self, key): return I(self.lo[key], self.hi[key])
    def mid(self): return self.lo/2 + self.hi/2
    def mag(self): return np.maximum(abs(self.lo), abs(self.hi))


def isum(items):
    z = I(0.)
    for a in items: z = z+a
    return z


def imat(a, b):
    a, b = I(a), I(b)
    if b.lo.ndim == 1:
        return isum(a[:, k]*b[k] for k in range(a.lo.shape[1]))
    return isum(I(a.lo[:, k:k+1], a.hi[:, k:k+1])*b[k:k+1, :] for k in range(a.lo.shape[1]))


def norm_upper(a):
    a = I(a).mag().ravel()
    v = isum(I(x)*I(x) for x in a)
    return float(np.nextafter(np.sqrt(v.hi), UP))


def operator_upper(a):
    a = I(a)
    gram = imat(I(a.lo.T, a.hi.T), a)
    rows = isum(I(gram.mag()[:, j]) for j in range(gram.lo.shape[1]))
    return float(np.nextafter(np.sqrt(np.max(rows.hi)), UP))


def symmetric_eigen_upper(a):
    """Generalized Gershgorin in a numerically chosen, interval-verified basis."""
    a=I(a);_,Q=np.linalg.eigh(a.mid())
    H=imat(Q.T,imat(a,Q));G=imat(Q.T,Q)
    top=max(float((I(H[i,i].hi)+isum(I(H[i,j].mag()) for j in range(6) if j!=i)).hi) for i in range(6))
    bottom=min(float((I(G[i,i].lo)-isum(I(G[i,j].mag()) for j in range(6) if j!=i)).lo) for i in range(6))
    gram_top=max(float((I(G[i,i].hi)+isum(I(G[i,j].mag()) for j in range(6) if j!=i)).hi) for i in range(6))
    assert bottom>0
    return float((iv.mpf(top)/iv.mpf(bottom if top>=0 else gram_top)).b)


def inverse_interval(a):
    a=I(a);n=a.lo.shape[0]
    t = np.linalg.inv(a.mid())
    residual = I(np.eye(n))-imat(a, t)
    eps = float(np.max(isum(I(residual.mag()[:, j]) for j in range(n)).hi))
    tnorm = float(np.max(isum(I(abs(t[:, j])) for j in range(n)).hi))
    assert eps < 1.
    # Neumann-series norm correction for T (I-E)^(-1).
    correction = float((iv.mpf(tnorm)*iv.mpf(eps)/(1-iv.mpf(eps))).b)
    return I(np.nextafter(t-correction, DN), np.nextafter(t+correction, UP))


class D:
    """First-order interval automatic differentiation."""
    def __init__(self, value, derivative=None):
        self.v = I(value)
        self.d = I(np.zeros(6) if derivative is None else derivative)
    @staticmethod
    def cast(b): return b if isinstance(b, D) else D(b)
    def __add__(self,b):
        b=self.cast(b); return D(self.v+b.v,self.d+b.d)
    __radd__=__add__
    def __neg__(self): return D(-self.v,-self.d)
    def __sub__(self,b): return self+(-self.cast(b))
    def __rsub__(self,b): return self.cast(b)+(-self)
    def __mul__(self,b):
        b=self.cast(b); return D(self.v*b.v,self.d*b.v+self.v*b.d)
    __rmul__=__mul__


class J:
    """Outward interval polynomial in local time; exact up to degree eight."""
    def __init__(self, coefficients):
        self.c = [I(z) for z in coefficients]
    @staticmethod
    def cast(b): return b if isinstance(b,J) else J([b])
    def __add__(self,b):
        b=self.cast(b); n=max(len(self.c),len(b.c))
        return J([(self.c[k] if k<len(self.c) else I(0))+(b.c[k] if k<len(b.c) else I(0)) for k in range(n)])
    __radd__=__add__
    def __neg__(self): return J([-z for z in self.c])
    def __sub__(self,b): return self+(-self.cast(b))
    def __rsub__(self,b): return self.cast(b)+(-self)
    def __mul__(self,b):
        b=self.cast(b)
        return J([isum(self.c[j]*b.c[k-j] for j in range(len(self.c)) if 0<=k-j<len(b.c)) for k in range(len(self.c)+len(b.c)-1)])
    __rmul__=__mul__


class Poly:
    """Sparse exact-degree multivariate polynomial with interval coefficients."""
    zero=(0,)*6
    def __init__(self,terms): self.terms=terms
    @staticmethod
    def cast(v): return v if isinstance(v,Poly) else Poly({Poly.zero:I(v)})
    def __add__(self,b):
        b=self.cast(b);out=self.terms.copy()
        for k,v in b.terms.items(): out[k]=out.get(k,I(0))+v
        return Poly(out)
    __radd__=__add__
    def __neg__(self): return Poly({k:-v for k,v in self.terms.items()})
    def __sub__(self,b): return self+(-self.cast(b))
    def __rsub__(self,b): return self.cast(b)+(-self)
    def __mul__(self,b):
        b=self.cast(b);out={}
        for a,u in self.terms.items():
            for c,v in b.terms.items():
                k=tuple(a[i]+c[i] for i in range(6))
                out[k]=out.get(k,I(0))+u*v
        return Poly(out)
    __rmul__=__mul__


def transformed_jacobian_polynomial(L,T,gain):
    variables=[]
    for j in range(6):
        exponent=[0]*6;exponent[j]=1
        variables.append(Poly({tuple(exponent):I(1.)}))
    physical=[Poly.cast(p.X_STAR[i])+sum(variables[j]*T[i,j] for j in range(6)) for i in range(6)]
    physical_field=field(physical,gain)
    transformed=[sum(physical_field[j]*L[i,j] for j in range(6)) for i in range(6)]
    derivatives=[];keys=set()
    for i in range(6):
        for j in range(6):
            terms={}
            for k,v in transformed[i].terms.items():
                if k[j]:
                    exponent=list(k);exponent[j]-=1;exponent=tuple(exponent)
                    terms[exponent]=v*I(k[j]);keys.add(exponent)
            derivatives.append(terms)
    keys=sorted(keys)
    lo=np.array([[d.get(k,I(0)).lo for k in keys] for d in derivatives])
    hi=np.array([[d.get(k,I(0)).hi for k in keys] for d in derivatives])
    return np.array(keys,dtype=int),I(lo,hi)


def evaluate_jacobian_polynomial(record,box):
    powers,coeff=record
    terms=I(np.ones(len(powers)))
    for j in range(6):
        for k in range(1,4):
            mask=powers[:,j]>=k
            factor=I(np.where(mask,box.lo[j],1.),np.where(mask,box.hi[j],1.))
            terms=terms*factor
    result=imat(coeff,terms)
    return I(result.lo.reshape(6,6),result.hi.reshape(6,6))


def exponential_interval(A,h,order=24):
    """Matrix exponential power series with a verified induced-infinity tail."""
    A=I(A);term=I(np.eye(6));result=term
    for k in range(1,order+1):
        term=imat(term,A)*I(h)*I(float((iv.mpf(1)/k).a),float((iv.mpf(1)/k).b))
        result=result+term
    norm=float(np.max(isum(I(A.mag()[:,j]) for j in range(6)).hi))
    z=iv.mpf(norm)*iv.mpf(h)
    tail=float((iv.exp(z)*z**(order+1)/math.factorial(order+1)).b)
    return result+I(-tail,tail)


def adjoint_over_step(row,A,h,order=16):
    """Enclose row*exp(A*t) for all 0<=t<=h, including series tail."""
    row=I(row);A=I(A);term=row;result=row
    for k in range(1,order+1):
        term=imat(term,A)*I(float((iv.mpf(1)/k).a),float((iv.mpf(1)/k).b))
        # Powers of one nonnegative time variable; coefficient cancellations
        # occur before the interval time power is applied.
        hp=iv.mpf(h)**k
        result=result+term*I(0.,float(hp.b))
    norm=float(np.max(isum(I(A.mag()[:,j]) for j in range(6)).hi))
    row_one=float(isum(I(row.mag().ravel()[j]) for j in range(6)).hi)
    z=iv.mpf(norm)*iv.mpf(h)
    tail=float((iv.mpf(row_one)*iv.exp(z)*z**(order+1)/math.factorial(order+1)).b)
    return result+I(-tail,tail)


def field(x, gain):
    u=[p.U_STAR[i]+sum((x[j]-p.X_STAR[j])*gain[i,j] for j in range(6)) for i in range(4)]
    n1,n2,n11,n22,n10,n20=x
    u12,u21,u02,u20=u
    v1=p.MFD_A*n1*n1+p.MFD_B*n1+p.MFD_C
    v2=p.MFD_A*n2*n2+p.MFD_B*n2+p.MFD_C
    q11,q12,q10=n11*v1,(n1-n11-n10)*v1,n10*v1
    q22,q21,q20=n22*v2,(n2-n22-n20)*v2,n20*v2
    d11,d12,d10,d22,d21,d20=map(float,p.DEMAND)
    return [-q11-(q12+q10)*u12+q21*u21+d11+d12+d10,
            -q22+(q12+q10)*u12-q21*u21+(p.Q01+p.Q02)*u02-q20*u20+d22+d21+d20,
            -q11+q21*u21+d11,-q22+q12*u12+p.Q02*u02+d22,
            -q10*u12+d10,-q20*u20+q10*u12+d20]


def jacobian(box, gain):
    vals=field([D(box[i],np.eye(6)[i]) for i in range(6)],gain)
    return I(np.array([v.d.lo for v in vals]),np.array([v.d.hi for v in vals]))


def iv_exp_flow(mu, h, radius, forcing):
    if mu == 0.: return float((iv.mpf(radius)+iv.mpf(h)*iv.mpf(forcing)).b)
    a=iv.exp(iv.mpf(mu)*iv.mpf(h))
    return float((a*iv.mpf(radius)+(a-1)/iv.mpf(mu)*iv.mpf(forcing)).b)


class Encloser:
    def __init__(self, amplitude=.003,nominal_delta=0.):
        if not math.isfinite(amplitude) or amplitude<0 or not math.isfinite(nominal_delta):
            raise ValueError('finite nominal demand and nonnegative finite disturbance amplitude required')
        self.gains,self.cert,self.tube=s.load_installed()
        self.admission_level=float(np.nextafter(self.tube['common_level']*(1.-1e-8),DN))
        self.P=self.cert['matrices']
        self.L=[]; self.T=[]; self.row=[]; self.tube_factor=[]
        self.jac_polynomials=[]
        self.linear=[]
        self.geometric_constraint_audits=[]
        for i,P in enumerate(self.P):
            chol=np.linalg.cholesky(P).T
            M=chol@(p.A+p.B@self.gains[i])@np.linalg.inv(chol)
            _,Q=np.linalg.eigh((M+M.T)/2)
            L=Q.T@chol
            T=inverse_interval(L)
            self.L.append(L); self.T.append(T)
            self.row.append(np.array([norm_upper(T[k]) for k in range(6)]))
            # Direct quadratic bound: ||P^(1/2) T||_2^2 <= row-sum norm(T^T P T).
            product=imat(I(T.lo.T,T.hi.T),imat(P,T))
            product=(product+I(product.lo.T,product.hi.T))*I(.5)
            self.tube_factor.append(float(np.nextafter(np.sqrt(np.max(isum(I(product.mag()[:,k]) for k in range(6)).hi)),UP)))
            psym=(I(P)+I(P.T))*I(.5)
            assert -symmetric_eigen_upper(-psym)>0
            inverseP=inverse_interval(psym)
            state_rows=np.vstack((-np.eye(6),np.eye(6)[:2],[-1,0,1,0,1,0],[0,-1,0,1,0,1]))
            state_bounds=np.array([0.]*6+[100.,100.,0.,0.])
            constraints=[(I(row),I(bound)-imat(row.reshape(1,6),p.X_STAR)[0]) for row,bound in zip(state_rows,state_bounds)]
            for row,offset in zip(self.gains[i],p.U_STAR):
                constraints.extend([(I(row),I(1.)-I(offset)),(-I(row),I(offset))])
            for hi,lo in ((1,0),(3,2)):
                row=I(self.gains[i][hi])-I(self.gains[i][lo]);offset=I(p.U_STAR[hi])-I(p.U_STAR[lo])
                constraints.extend([(row,I(.75)-offset),(-row,I(.75)+offset)])
            slacks=[]
            for row,margin in constraints:
                quad=isum(row[k]*imat(inverseP,row)[k] for k in range(6))
                support=float(iv.sqrt(iv.mpf(self.admission_level)*iv.mpf(float(quad.hi))).b)
                slacks.append(float((margin-I(support)).lo))
            assert min(slacks)>0,slacks
            self.geometric_constraint_audits.append({'constraint_count':len(slacks),'minimum_affine_constraint_slack':min(slacks)})
            self.jac_polynomials.append(transformed_jacobian_polynomial(L,T,self.gains[i]))
            self.linear.append(evaluate_jacobian_polynomial(self.jac_polynomials[-1],I(np.zeros(6))).mid())
        channel=np.array([[1,1,1,0,0,0],[0,0,0,1,1,1],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1]],dtype=float)
        base=imat(channel,I(p.DEMAND)*I([1,1,1,-1,-1,-1]))
        self.disturbance=base*I(amplitude)
        self.nominal=base*I(nominal_delta)
        self.eta=[norm_upper(imat(L,self.disturbance)) for L in self.L]
        self.amplitude=amplitude
        self.nominal_delta=nominal_delta

    def tube_bound(self,x,r,mode):
        e=I(x)-p.X_STAR
        v=isum(e[k]*imat(self.P[mode],e)[k] for k in range(6))
        upper=(iv.sqrt(iv.mpf(max(0,float(v.hi))))+iv.mpf(r)*iv.mpf(self.tube_factor[mode]))**2
        return float(np.nextafter(float(upper.b),UP))

    def switch_radius(self,r,source,destination):
        return r if source==destination else float(np.nextafter(r*operator_upper(imat(self.L[destination],self.T[source])),UP))

    def step(self,x,r,mode,h):
        L,T=self.L[mode],self.T[mode]
        fc=[z+self.nominal[k] for k,z in enumerate(field([I(z) for z in x],self.gains[mode]))]
        f=I(np.array([z.lo for z in fc]),np.array([z.hi for z in fc]))
        v=f.mid(); acc=imat(jacobian(I(x),self.gains[mode]),f).mid()/2
        curve=[J([x[k],v[k],acc[k]]) for k in range(6)]
        polynomials=[z+self.nominal[k] for k,z in enumerate(field(curve,self.gains[mode]))]
        residuals=[polynomials[k]-J([v[k],2*acc[k]]) for k in range(6)]
        assert max(len(z.c) for z in residuals)<=9
        residual=I(np.zeros(6))
        power=I(1.)
        for k in range(9):
            coeff=I(np.array([z.c[k].lo if k<len(z.c) else 0 for z in residuals]),np.array([z.c[k].hi if k<len(z.c) else 0 for z in residuals]))
            transformed=imat(L,coeff)
            residual=residual+I(-transformed.mag(),transformed.mag())*power
            power=power*I(h)
        defect=norm_upper(residual)
        time_box=I(0.,h)
        curve_box=I(x)+I(v)*time_box+I(acc)*time_box*time_box
        end_box=I(x)+I(v)*I(h)+I(acc)*I(h)*I(h)
        end=end_box.mid()
        rounding=norm_upper(imat(L,end_box-I(end)))
        R=max(1e-9,1.1*(r+h*(self.eta[mode]+defect)))
        for attempt in range(16):
            metric_curve=imat(L,I(x)-p.X_STAR)+imat(L,v)*time_box+imat(L,acc)*time_box*time_box
            metric_box=metric_curve+I(-R,R)
            mat=evaluate_jacobian_polynomial(self.jac_polynomials[mode],metric_box)
            sym=(mat+I(mat.lo.T,mat.hi.T))*I(.5)
            mu=symmetric_eigen_upper(sym)
            rend=iv_exp_flow(mu,h,r,float((I(self.eta[mode])+I(defect)).hi))
            if max(r,rend)<R:
                final_r=float((I(rend)+I(rounding)).hi)
                # Entire curve/error set must stay within the active SOS tube.
                ec=curve_box-I(p.X_STAR)
                vc=isum(ec[k]*imat(self.P[mode],ec)[k] for k in range(6))
                # Box quadratic above loses correlation; center-path P-norm is instead bounded by triangle.
                vx=self.tube_bound(x,0,mode)
                path_motion=norm_upper(imat(L,v)*time_box+imat(L,acc)*time_box*time_box)
                vmax=float(np.nextafter(float(((iv.sqrt(iv.mpf(vx))+iv.mpf(self.tube_factor[mode])*(iv.mpf(path_motion)+iv.mpf(R)))**2).b),UP))
                return end,final_r,dict(mu=mu,defect=defect,radius=final_r,tube_upper=vmax,attempts=attempt+1,apriori_radius=R,jacobian=mat,rounding=rounding)
            R=max(R*1.5,1.1*rend)
            if R>10: raise ValueError('error enclosure diverged')
        raise ValueError('failed a-priori tube inclusion')

    def schedule(self,x0,modes,period=.25,step_seconds=2.):
        if not modes or period<=0 or step_seconds<=0: raise ValueError('nonempty schedule and positive clocks required')
        x=np.asarray(x0,dtype=float);r=0.;current=modes[0]
        if x.shape!=(6,) or not np.all(np.isfinite(x)) or any(m not in (0,1) for m in modes):
            raise ValueError('finite six-state snapshot and installed modes required')
        records=[]; tube_max=0.; maxmu=-float('inf'); failures=[]; started=time.perf_counter()
        pieces=2**math.ceil(math.log2(max(1.,period*3600/step_seconds)));h=period/pieces
        for k,mode in enumerate(modes):
            r=self.switch_radius(r,current,mode); current=mode
            landing=self.tube_bound(x,r,mode)
            if landing>self.admission_level:
                failures.append(dict(kind='landing',epoch=k,bound=landing)); break
            pieces=2**math.ceil(math.log2(max(1.,period*3600/step_seconds)));h=period/pieces
            segment_radius=r;segment_info=[]
            for _ in range(pieces):
                try: x,r,info=self.step(x,r,mode,h)
                except ValueError as exc:
                    failures.append(dict(kind='enclosure',epoch=k,message=str(exc)));break
                tube_max=max(tube_max,info['tube_upper']);maxmu=max(maxmu,info['mu'])
                segment_info.append(info)
                if info['tube_upper']>self.admission_level:
                    failures.append(dict(kind='continuous_tube',epoch=k,bound=info['tube_upper']));break
            if failures: break
            a=np.array([1.,-1,0,0,0,0])
            support=norm_upper(imat(I(a.reshape(1,6)),self.T[mode]))
            margin=float((I(support)*I(r)*I(100.)).hi)
            # Output-sensitive Duhamel bound retains the rank-one demand
            # channel instead of projecting a six-dimensional error ball.
            row=imat(a.reshape(1,6),self.T[mode]);A=self.linear[mode]
            transition=exponential_interval(A,h)
            directional=I(0.)
            b=imat(self.L[mode],self.disturbance)
            for info in reversed(segment_info):
                q=adjoint_over_step(row,A,h)
                demand=float(imat(q,b).mag()[0])
                nonlinear=I(norm_upper(imat(q,info['jacobian']-A)))*I(info['apriori_radius'])
                residual=I(norm_upper(q))*I(info['defect'])
                directional=directional+(I(demand)+nonlinear+residual)*I(h)+I(norm_upper(row))*I(info['rounding'])
                row=imat(row,transition)
            directional=directional+I(norm_upper(row))*I(segment_radius)
            directional_margin=float((directional*I(100.)).hi)
            center_output=imat(a.reshape(1,6),I(x))*I(100.)
            best_margin=min(margin,directional_margin)
            output_interval=center_output+I(-best_margin,best_margin)
            lower=0. if output_interval.lo[0]<=0<=output_interval.hi[0] else float(min(abs(output_interval.lo[0]),abs(output_interval.hi[0])))
            records.append(dict(epoch=k+1,time_h=(k+1)*period,mode=mode+1,center=x.tolist(),radius=r,imbalance_center_vehicles=100*(x[0]-x[1]),imbalance_error_vehicles=margin,directional_error_vehicles=directional_margin,ball_absolute_imbalance_upper_vehicles=float((I(center_output.mag()[0])+I(margin)).hi),absolute_imbalance_upper_vehicles=float(output_interval.mag()[0]),absolute_imbalance_lower_vehicles=lower))
        return dict(modes=[m+1 for m in modes],amplitude=self.amplitude,nominal_delta=self.nominal_delta,step_seconds=step_seconds,actual_substep_seconds=h*3600,records=records,verified_continuous_tube=not failures,safety_basis='continuous polynomial-predictor/error enclosure inside independently verified affine-admissible ellipsoid',admission_level=self.admission_level,geometric_constraint_audits=self.geometric_constraint_audits,failures=failures,maximum_predictor_outer_tube_bound=tube_max,maximum_log_norm_upper=maxmu if math.isfinite(maxmu) else None,wall_seconds=time.perf_counter()-started)


def scalar_budgets(target_vehicles,amplitude,steps,period=.25):
    """Exploratory floating-point DP; not used to accept certified actions."""
    _,c,t=s.load_installed();P=c['matrices'];a=np.array([1.,-1,0,0,0,0]);ell=t['common_level']
    sigma=np.array([a@np.linalg.solve(pm,a) for pm in P])
    robust=s.robust_record(amplitude,15,c,t)
    rate=c['rate'] if amplitude==0 else robust['practical_flow_rate_per_hour']
    floors=np.zeros(2) if amplitude==0 else np.array(robust['modewise_ultimate_levels'])
    nu=np.array([[float(max(eigvalsh(P[j],P[m]))) if j!=m else 1 for j in range(2)] for m in range(2)])
    budget=np.full((steps+1,2),-np.inf);threshold=np.full_like(budget,-np.inf);choice=np.zeros_like(budget,dtype=int)
    budget[0]=np.minimum(ell,(target_vehicles/100)**2/sigma)
    for k in range(1,steps+1):
        threshold[k]=np.minimum(ell,floors+np.exp(rate*period)*(budget[k-1]-floors))
        for m in range(2):
            candidates=[min(ell,threshold[k,j]/nu[m,j]) if threshold[k,j]>=0 else -np.inf for j in range(2)]
            choice[k,m]=int(np.argmax(candidates));budget[k,m]=max(candidates)
    return dict(budgets=budget.tolist(),thresholds=threshold.tolist(),choice=choice.tolist(),sigma=sigma.tolist(),nu=nu.tolist(),rate=rate,floors=floors.tolist())


def scalar_terminal_bounds(model,x,mode,h=.25):
    """Outward forward bounds: original Young comparison and sharper radius."""
    P=model.P[mode];e=I(x)-p.X_STAR
    value=isum(e[k]*imat(P,e)[k] for k in range(6))
    a=np.array([1.,-1,0,0,0,0]);psym=(I(P)+I(P.T))*I(.5)
    sigma=imat(a.reshape(1,6),imat(inverse_interval(psym),a))
    gamma2=isum(model.disturbance[k]*imat(P,model.disturbance)[k] for k in range(6))
    bridge_path=ROOT/'reproducibility/results/mfd_deadline_sos_bridge_results.json'
    if not bridge_path.exists(): raise RuntimeError('Run mfd_deadline_sos_bridge.py before the scalar comparison study')
    bridge=json.loads(bridge_path.read_text(encoding='utf-8'))['modes'][mode]
    lam=iv.mpf(bridge['effective_rate_lower_per_hour']);alpha=iv.exp(-lam*iv.mpf(h)/2)
    gamma=iv.sqrt(iv.mpf(max(0,float(gamma2.hi))))+iv.mpf(bridge['certificate_radius_roundoff_forcing_upper_per_hour'])
    rho=2*gamma/lam
    v=iv.mpf(float(value.hi));support=iv.sqrt(iv.mpf(float(sigma.hi[0])))*100
    young=support*iv.sqrt(alpha*v+(1-alpha)*rho**2)
    radius=support*(alpha*iv.sqrt(v)+(1-alpha)*rho)
    return {'young_support_upper_vehicles':float(young.b),'radius_support_upper_vehicles':float(radius.b)}


def run_study():
    import mfd_cubic_runtime_pilot as rt
    models={amp:Encloser(amp) for amp in (0.,.003,.006)}
    results={'scope':'robust terminal-service certification; discovery-selected mechanism, full grid retained','target_fractions':[.5,.25,.1],'deadline_minutes':15,'demand_family':'every measurable symmetric rank-one redistribution signal with |delta(t)|<=amplitude','cases':[]}
    directions={'region1_heavy':np.array([1.,-1,.5,-.5,.25,-.25]),'region2_heavy':np.array([-1.,1,-.5,.5,-.25,.25])}
    for initial_mode in (0,1):
        for label,direction in list(directions.items())+[('positive_support',np.linalg.solve(models[0.].P[initial_mode],np.array([1.,-1,0,0,0,0]))),('negative_support',-np.linalg.solve(models[0.].P[initial_mode],np.array([1.,-1,0,0,0,0])))]:
            x=rt.state_on_tube(direction,.8,models[0.].cert,models[0.].tube,mode=initial_mode)
            initial=abs(100*(x[0]-x[1]))
            for amplitude,model in models.items():
                case=dict(name=label,initial_mode=initial_mode+1,amplitude=amplitude,initial_state=x.tolist(),initial_imbalance_vehicles=initial,targets_vehicles=[initial*f for f in (.5,.25,.1)],candidates=[])
                for destination in (0,1):
                    certificate=model.schedule(x,[destination],step_seconds=15.)
                    attempts=[]
                    for finer in (7.5,3.75):
                        if certificate['failures'] and certificate['failures'][0]['kind']=='continuous_tube':
                            attempts.append(certificate)
                            certificate=model.schedule(x,[destination],step_seconds=finer)
                    certificate['coarser_failed_enclosures']=attempts
                    certificate['scalar_bounds']=scalar_terminal_bounds(model,x,destination)
                    if certificate['verified_continuous_tube']:
                        bound=certificate['records'][-1]['absolute_imbalance_upper_vehicles']
                        certificate['passes_targets']=[bool(bound<=threshold) for threshold in case['targets_vehicles']]
                    else: certificate['passes_targets']=[False]*3
                    case['candidates'].append(certificate)
                results['cases'].append(case)
                print(label,initial_mode+1,amplitude,[c['passes_targets'] for c in case['candidates']],flush=True)
    primary=next(c for c in results['cases'] if c['name']=='region2_heavy' and c['initial_mode']==1 and c['amplitude']==.003)
    results['primary_case']=primary
    fixed=Encloser(0.,nominal_delta=.003)
    results['fixed_adverse_hold_certificate']=fixed.schedule(primary['initial_state'],[0],step_seconds=15.)
    results['primary_refinement_checks']=[models[.003].schedule(primary['initial_state'],[1],step_seconds=h) for h in (30.,7.5)]
    results['two_interval_plan_checks']=[models[.003].schedule(primary['initial_state'],modes,step_seconds=15.) for modes in ([0,0],[0,1],[1,0],[1,1])]
    results['summary']={'cases':len(results['cases']),'candidate_schedules':sum(len(c['candidates']) for c in results['cases']),'verified_safe_candidate_schedules':sum(z['verified_continuous_tube'] for c in results['cases'] for z in c['candidates']),'case_successes_by_target':[sum(any(z['passes_targets'][k] for z in c['candidates']) for c in results['cases']) for k in range(3)]}
    OUT.write_text(json.dumps(s.json_ready(results),indent=2,allow_nan=False),encoding='utf-8')
    print(results['summary'],flush=True)


if __name__=='__main__': run_study()
