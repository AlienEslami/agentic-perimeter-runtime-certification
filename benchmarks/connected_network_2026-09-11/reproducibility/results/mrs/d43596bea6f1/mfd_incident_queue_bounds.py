"""Outward scalar queue enclosures for constant gates and affine arrival.

For z'=a(t)+w(t)-min(alpha*z,q), scalar comparison makes the exact reachable
endpoint interval [z_-(t),z_+(t)]: use the lower/upper initial endpoint and
w(t)=-eta/+eta throughout. These extremal residuals are admissible, so this
characterization is sharp. The implementation encloses those scalar flows;
it does not use a numerical ODE tolerance as a certificate.

On a step, constant lower/upper arrivals have explicit flows, including their
single possible threshold crossing. Their monotone trajectories bound the
whole affine-arrival step. If that bound establishes one common admission
branch, the affine-input formula tightens the endpoint enclosure. Otherwise
the constant-arrival comparison remains valid and can be refined by exact
rational substeps. Every operation uses outward interval arithmetic.

Nonnegative initial queues, gates, and the entire lower arrival envelope are
required. Then z>=0 is forward invariant: at zero, z'=a+w>=0. Intersecting the
queue enclosure with [0,infinity) therefore uses a proved physical property.
This module applies to constant commands, not state-feedback gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from mpmath import iv

from mfd_deadline_certificates import I


def frac(value):
    if isinstance(value,Fraction):
        return value
    return Fraction.from_float(float(value))


def _iv(value):
    if isinstance(value,I):
        return iv.mpf([float(value.lo),float(value.hi)])
    if isinstance(value,Fraction):
        return iv.mpf(value.numerator)/iv.mpf(value.denominator)
    return iv.mpf(float(value))


def _out(value):
    return I(np.nextafter(float(value.a),-np.inf),np.nextafter(float(value.b),np.inf))


def _pair(value):
    value=I(value)
    return [float(value.lo),float(value.hi)]


def _nonnegative(value):
    value=I(value)
    if value.hi < 0:
        raise ArithmeticError("An allegedly positive queue enclosure has negative upper endpoint")
    return I(max(0.,float(value.lo)),max(0.,float(value.hi)))


def _supply_flow(z,arrival,alpha,time_h):
    decay=iv.exp(-_iv(alpha)*_iv(time_h))
    return _iv(z)*decay+_iv(arrival)*(1-decay)/_iv(alpha)


def constant_arrival_flow(z0,arrival,gate_capacity,duration_h,release_rate=60.):
    """Outward endpoint of the exact scalar flow with constant nonnegative input.

    Inputs here are scalar binary coefficients; an exact Fraction specifies
    time. Queue intervals are propagated by monotonicity in queue and input,
    and antitonicity in gate capacity, in queue_step_bounds below.
    """
    z0,arrival,q,alpha=map(float,(z0,arrival,gate_capacity,release_rate))
    duration=frac(duration_h)
    if not all(math.isfinite(v) for v in (z0,arrival,q,alpha)):
        raise ValueError("Finite scalar queue coefficients are required")
    if min(z0,arrival,q)<0 or alpha<=0 or duration<0:
        raise ValueError("Nonnegative queue, arrival, gate and time; positive release rate required")
    if duration==0:
        return I(z0)
    z,a,qf,af=map(frac,(z0,arrival,q,alpha))
    threshold=qf/af
    if af*z>=qf:
        if a>=qf:
            result=_iv(z)+( _iv(a)-_iv(qf))*_iv(duration)
        else:
            hitting=(z-threshold)/(qf-a)
            if duration<=hitting:
                result=_iv(z)+(_iv(a)-_iv(qf))*_iv(duration)
            else:
                result=_supply_flow(threshold,a,af,duration-hitting)
    else:
        if a<=qf:
            result=_supply_flow(z,a,af,duration)
        else:
            # The upward crossing time is logarithmic. An unresolved interval
            # comparison retains both branch formulas instead of guessing.
            hitting=iv.log((_iv(a)-_iv(af)*_iv(z))/(_iv(a)-_iv(qf)))/_iv(af)
            hitting_box=_out(hitting)
            now=_iv(duration)
            supply=_supply_flow(z,a,af,duration)
            gate=_iv(threshold)+(_iv(a)-_iv(qf))*(now-hitting)
            if duration<=frac(float(hitting_box.lo)):
                result=supply
            elif duration>=frac(float(hitting_box.hi)):
                result=gate
            else:
                left,right=_out(supply),_out(gate)
                return _nonnegative(I(min(float(left.lo),float(right.lo)),max(float(left.hi),float(right.hi))))
    return _nonnegative(_out(result))


def queue_step_bounds(z_lo,z_hi,duration_h,arrival0,arrival_slope=0.,
                      residual_bound=0.,command_u02=1.,capacity=120.,release_rate=60.,
                      substeps=1):
    """Return endpoint/continuous/admission intervals as JSON-ready pairs.

    All queue quantities are in the manuscript's normalized 100-vehicle units;
    time is hours. arrival0 and arrival_slope may be existing I intervals and
    describe one affine nominal-arrival piece, relative to this step's left
    endpoint. The caller must split at any arrival/gate discontinuity or knot.
    No queue-storage limit is imposed here; the caller checks its upper bound.
    """
    initial=I(float(z_lo),float(z_hi))
    duration=frac(duration_h)
    a0,slope=I(arrival0),I(arrival_slope)
    eta=float(residual_bound)
    u,C,alpha=map(float,(command_u02,capacity,release_rate))
    if not all(np.isfinite([initial.lo,initial.hi,a0.lo,a0.hi,slope.lo,slope.hi,eta,u,C,alpha])):
        raise ValueError("Finite scalar intervals are required")
    if initial.lo<0 or initial.lo>initial.hi or eta<0 or u<0 or C<0 or alpha<=0 or duration<0:
        raise ValueError("Nonnegative ordered queue/gate/residual/time and positive release rate required")
    if a0.lo>a0.hi or slope.lo>slope.hi or type(substeps) is not int or substeps<1:
        raise ValueError("Ordered affine intervals and a positive integer substep count required")
    # This installed real product is enclosed, not silently replaced by its
    # nearest binary floating-point approximation.
    q=_nonnegative(_out(_iv(C)*_iv(u)))
    h=duration/substeps
    current=initial
    total_lo,total_hi=float(initial.lo),float(initial.hi)
    pieces=[]
    for k in range(substeps):
        offset=h*k
        left_a=_out(_iv(a0)+_iv(slope)*_iv(offset))
        right_a=_out(_iv(a0)+_iv(slope)*_iv(offset+h))
        nominal_lo=min(float(left_a.lo),float(right_a.lo))
        nominal_hi=max(float(left_a.hi),float(right_a.hi))
        forcing=_out(iv.mpf([nominal_lo,nominal_hi])+iv.mpf([-eta,eta]))
        if forcing.lo<0:
            # Exact zero is admissible; directed rounding can create one tiny
            # negative ulp. Decide the physical hypothesis with exact rational
            # endpoint arithmetic, independently of that rounded enclosure.
            exact_min=min(frac(float(a0.lo))+frac(float(slope.lo))*offset,
                          frac(float(a0.lo))+frac(float(slope.lo))*(offset+h))
            if exact_min-frac(eta)<0:
                raise ValueError("Lower arrival envelope can be negative; queue positivity hypothesis fails")
            forcing=I(0.,forcing.hi)
        lower=constant_arrival_flow(float(current.lo),float(forcing.lo),float(q.hi),h,alpha)
        upper=constant_arrival_flow(float(current.hi),float(forcing.hi),float(q.lo),h,alpha)
        comparison=I(lower.lo,upper.hi)
        continuous=I(min(float(current.lo),float(lower.lo)),max(float(current.hi),float(upper.hi)))
        branch="crossing_comparison"
        endpoint=comparison
        # A branch assertion is certified over the whole step, for every
        # reachable queue and all allowed coefficients/residual functions.
        if frac(alpha)*frac(float(continuous.hi))<=frac(float(q.lo)):
            branch="supply_affine_exact_extremals"
            decay=iv.exp(-_iv(alpha)*_iv(h))
            phi1=(1-decay)/_iv(alpha)
            phi2=(_iv(h)-phi1)/_iv(alpha)
            phi1_bounds,phi2_bounds=_out(phi1),_out(phi2)
            phi1=iv.mpf([max(0.,float(phi1_bounds.lo)),float(phi1_bounds.hi)])
            phi2=iv.mpf([max(0.,float(phi2_bounds.lo)),float(phi2_bounds.hi)])
            tightened=_out(_iv(current)*decay+(_iv(left_a)+iv.mpf([-eta,eta]))*phi1+_iv(slope)*phi2)
            endpoint=I(max(float(comparison.lo),float(tightened.lo)),min(float(comparison.hi),float(tightened.hi)))
        elif frac(alpha)*frac(float(continuous.lo))>=frac(float(q.hi)):
            branch="gate_affine_exact_extremals"
            tightened=_out(_iv(current)+(_iv(left_a)+iv.mpf([-eta,eta])-_iv(q))*_iv(h)+_iv(slope)*_iv(h)*_iv(h)/2)
            endpoint=I(max(float(comparison.lo),float(tightened.lo)),min(float(comparison.hi),float(tightened.hi)))
        endpoint=_nonnegative(endpoint)
        if endpoint.lo>endpoint.hi:
            raise ArithmeticError("Independent endpoint bounds have empty intersection")
        total_lo=min(total_lo,float(continuous.lo))
        total_hi=max(total_hi,float(continuous.hi))
        pieces.append(dict(exact_left_offset_h=str(offset),exact_right_offset_h=str(offset+h),
            arrival_envelope=_pair(forcing),branch=branch,endpoint=_pair(endpoint),continuous=_pair(continuous)))
        current=endpoint
    whole=_nonnegative(I(total_lo,total_hi))
    release=_out(_iv(alpha)*_iv(whole))
    admission=I(max(0.,min(float(release.lo),float(q.lo))),max(0.,min(float(release.hi),float(q.hi))))
    return dict(endpoint=_pair(current),continuous=_pair(whole),admission=_pair(admission),
        gate_capacity=_pair(q),positivity_proved=True,exact_duration_h=str(duration),
        method="scalar monotonicity + outward exact constant-input flow; affine branch tightening",
        exact_reachable_set_characterization="extremal initial queues and constant residual signs",
        computational_scope="outward enclosure, affine arrival comparisons refined by rational substeps",
        pieces=pieces)


def self_tests():
    """Analytic checks and independent numerical diagnostics, not proof by sampling."""
    from scipy.integrate import solve_ivp
    from mpmath import mp
    analytic_cases=[(2.,60.,120.,Fraction(1,12)),(2.,90.,30.,Fraction(1,3)),
                    (.1,90.,60.,Fraction(1,3)),(3.,30.,60.,Fraction(1,3)),
                    (0.,0.,0.,Fraction(1)),(2.,60.,60.,Fraction(1)),
                    (.5,60.,60.,Fraction(1)),(1.,0.,60.,Fraction(1))]
    maximum_error=0.
    for z,a,q,h in analytic_cases:
        bound=constant_arrival_flow(z,a,q,h)
        numeric=solve_ivp(lambda t,y:[a-min(60*y[0],q)],(0,float(h)),[z],
                          rtol=1e-12,atol=1e-13,max_step=max(float(h)/200,1e-8)).y[0,-1]
        maximum_error=max(maximum_error,max(float(bound.lo)-numeric,numeric-float(bound.hi),0.))
        assert float(bound.lo)-2e-9<=numeric<=float(bound.hi)+2e-9
    # High-precision analytic reference checks avoid ODE event/tolerance error.
    with mp.workdps(90):
        for z,a,q,h in analytic_cases:
            Z,A,Q,H=map(mp.mpf,(z,a,q,float(h)))
            H=mp.mpf(h.numerator)/h.denominator
            threshold=Q/60
            supply=lambda start,t:A/60+(start-A/60)*mp.exp(-60*t)
            if Z>=threshold:
                hit=mp.inf if A>=Q else (Z-threshold)/(Q-A)
                exact=Z+(A-Q)*H if H<=hit else supply(threshold,H-hit)
            elif A<=Q:
                exact=supply(Z,H)
            else:
                hit=mp.log((A-60*Z)/(A-Q))/60
                exact=supply(Z,H) if H<=hit else threshold+(A-Q)*(H-hit)
            bound=constant_arrival_flow(z,a,q,h)
            assert mp.mpf(float(bound.lo))<=exact<=mp.mpf(float(bound.hi))
    rng=np.random.default_rng(20260911)
    maximum_enclosure_width=0.
    for index in range(36):
        z=rng.uniform(.01,3.); width=min(z/2,rng.uniform(0,.3))
        a=rng.uniform(30,100); slope=rng.uniform(-100,100); eta=rng.uniform(0,12)
        q=rng.uniform(5,120); h=Fraction(1,48)
        record=queue_step_bounds(z-width,z+width,h,a,slope,eta,q/120,substeps=4)
        maximum_enclosure_width=max(maximum_enclosure_width,record["endpoint"][1]-record["endpoint"][0])
        for initial,sign in ((z-width,-1),(z+width,1),(z,0)):
            sol=solve_ivp(lambda t,y:[a+slope*t+sign*eta-min(60*y[0],120*(q/120))],
                (0,float(h)),[initial],rtol=1e-11,atol=1e-12,max_step=float(h)/100,dense_output=True)
            assert sol.success
            value=float(sol.y[0,-1])
            assert record["endpoint"][0]-2e-8<=value<=record["endpoint"][1]+2e-8
            sample=sol.sol(np.linspace(0,float(h),101))[0]
            assert min(sample)>=record["continuous"][0]-2e-8
            assert max(sample)<=record["continuous"][1]+2e-8
    # Positive invariance must survive a zero lower arrival and zero queue.
    zero=queue_step_bounds(0.,0.,Fraction(1,60),12.,0.,12.,.5)
    assert zero["endpoint"][0]==0. and zero["continuous"][0]==0.
    try:
        queue_step_bounds(0.,1.,Fraction(1,60),10.,0.,12.,.5)
    except ValueError:
        pass
    else:
        raise AssertionError("A negative lower arrival must not receive a positivity certificate")
    return dict(status="PASS",analytic_cases=len(analytic_cases),
        high_precision_closed_form_containment=True,random_affine_cases=36,
        numerical_trajectories=108,diagnostic_constant_flow_maximum_ode_discrepancy=maximum_error,
        maximum_random_endpoint_width=maximum_enclosure_width,
        zero_arrival_boundary_proved=True,negative_arrival_hypothesis_rejected=True,
        proof_basis="scalar comparison, positive invariance, exact analytic flows, outward interval arithmetic",
        numerical_diagnostics_are_not_certificates=True)


def archived_plan_demo():
    """Queue-only certificates for the frozen plan; not full-plant certificates."""
    from mfd_incident_verifier import Pulse
    root=Path(__file__).resolve().parents[1]
    source=root/"reproducibility/results/mfd_incident_uncertainty_frontier_results.json"
    previous=json.loads(source.read_text(encoding="utf-8"))
    commands=previous["commands"]
    pulse=Pulse(amplitude=.3,duration=1.)
    cuts=sorted(set([Fraction(k,240) for k in range(241)]+
                    [cut for cut in pulse.knots if 0<cut<1]))
    cases=[]
    for width in (1.,10.):
        for residual in (60.,120.,300.,600.,1200.):
            # Fraction arithmetic installs the declared initial physical box;
            # converting it to floating endpoints is directed outward.
            initial=_out(iv.mpf(2)+iv.mpf([-width/100,width/100]))
            zlo,zhi=float(initial.lo),float(initial.hi)
            whole_lo,whole_hi=zlo,zhi
            branches={}
            endpoint_records=[]
            for left,right in zip(cuts[:-1],cuts[1:]):
                u=commands[0 if left<Fraction(1,12) else 1][2]
                arrival,slope=pulse.affine(left,right)
                record=queue_step_bounds(zlo,zhi,right-left,arrival,slope,residual/100,u)
                zlo,zhi=record["endpoint"]
                whole_lo=min(whole_lo,record["continuous"][0])
                whole_hi=max(whole_hi,record["continuous"][1])
                branch=record["pieces"][0]["branch"]
                branches[branch]=branches.get(branch,0)+1
                endpoint_records.append(dict(exact_time_h=str(right),endpoint=record["endpoint"],
                    continuous=record["continuous"],admission=record["admission"],branch=branch))
            cases.append(dict(initial_queue_half_width_vehicles=width,residual_vehicles_per_hour=residual,
                endpoint_queue_vehicles=[100*zlo,100*zhi],
                continuous_queue_vehicles=[100*whole_lo,100*whole_hi],
                nonnegative_and_within_2000_storage=whole_lo>=0 and whole_hi<=20,
                branch_counts=branches,steps=endpoint_records))
    return dict(scope="queue-only outward certificates; regional classes and service not checked here",
        source_plan_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        archived_commands_unchanged=True,nominal_arrival="same primary 30% one-hour trapezoid",
        clock="exact 15-second steps, additionally split at binary-installed pulse knots",
        cases=cases)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    parser.add_argument("--archived-plan-demo",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.self_test:
        print(json.dumps(self_tests(),indent=2))
    if args.archived_plan_demo:
        payload=archived_plan_demo()
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open("x",encoding="utf-8") as handle:
                json.dump(payload,handle,indent=2,allow_nan=False)
                handle.write("\n")
        print(json.dumps({"scope":payload["scope"],"cases":[{k:v for k,v in row.items() if k!="steps"} for row in payload["cases"]]},indent=2))
    if not (args.self_test or args.archived_plan_demo):
        parser.print_help()


if __name__=="__main__":
    main()
