"""Reproducible frozen-plan probes; archived implementation is never edited."""
from pathlib import Path
import argparse, cProfile, hashlib, inspect, json, pstats, sys, textwrap, time
from fractions import Fraction
from functools import lru_cache
import numpy as np

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent / 'reusable_perimeter_suite_2026-09-26'
sys.path.insert(0, str(SUITE))
import common
import mfd_incident_affine_reachability as affine
import mfd_incident_verifier as core
from common import save, verify_command_plan

def vector_imat(a,b):
    """Same products, summation order and outward rounding as the archived imat."""
    I=affine.I; a,b=I(a),I(b)
    vector=b.lo.ndim==1
    bl=b.lo[:,None] if vector else b.lo
    bh=b.hi[:,None] if vector else b.hi
    al,ah=a.lo[:,:,None],a.hi[:,:,None]
    products=np.array([al*bl[None,:,:],al*bh[None,:,:],ah*bl[None,:,:],ah*bh[None,:,:]])
    lo=np.nextafter(np.min(products,axis=0),-np.inf)
    hi=np.nextafter(np.max(products,axis=0),np.inf)
    sl=np.zeros((a.lo.shape[0],bl.shape[1]));sh=sl.copy()
    for k in range(a.lo.shape[1]):
        sl=np.nextafter(sl+lo[:,k,:],-np.inf)
        sh=np.nextafter(sh+hi[:,k,:],np.inf)
    return I(sl[:,0],sh[:,0]) if vector else I(sl,sh)

def power_monomials(powers,box):
    """Outward integer powers; unlike repeated interval products, squares stay nonnegative."""
    I=affine.I;terms=I(np.ones(len(powers)))
    for index in range(7):
        for power in (1,2,3):
            mask=powers[:,index]==power
            if not np.any(mask):continue
            x=I(box.lo[index],box.hi[index])
            if power==2:
                lo=0. if x.lo<=0<=x.hi else min(float(x.lo*x.lo),float(x.hi*x.hi))
                hi=max(float(x.lo*x.lo),float(x.hi*x.hi))
                value=I(max(0.,np.nextafter(lo,-np.inf)),np.nextafter(hi,np.inf))
            elif power==3:
                # Point products each rounded outwards, using monotonicity of the cube.
                left=I(x.lo)*I(x.lo)*I(x.lo);right=I(x.hi)*I(x.hi)*I(x.hi)
                value=I(left.lo,right.hi)
            else:value=x
            terms=terms*I(np.where(mask,value.lo,1.),np.where(mask,value.hi,1.))
    return terms

def verifier(variant='baseline', padding=1.1):
    if variant == 'baseline':
        return affine.AffineGeneratorVerifier()
    namespace = dict(vars(affine))
    namespace['reciprocal'] = lru_cache(None)(affine.reciprocal)
    namespace['imat']=vector_imat
    core_namespace=dict(vars(core));core_namespace['imat']=vector_imat
    if variant.startswith('tight'):core_namespace['monomials']=power_monomials
    exec(inspect.getsource(core.evaluate_transformed_jacobian),core_namespace)
    namespace['evaluate_transformed_jacobian']=core_namespace['evaluate_transformed_jacobian']
    for fun in ('exponential_and_integral', 'signed_flow_range'):
        source = inspect.getsource(getattr(affine, fun))
        # The remainder bounds are still computed for the actual truncation order.
        if variant in ('order12', 'tight12'):
            source = source.replace('order=ORDER', 'order=12')
        exec(source, namespace)
    source = textwrap.dedent(inspect.getsource(affine.AffineGeneratorVerifier.step))
    if variant.startswith('tight'):
        source = source.replace('I(1.1)', 'I(self.trial_padding)').replace('range(24)', 'range(64)')
    exec(source, namespace)
    class ProbeVerifier(affine.AffineGeneratorVerifier):
        step = namespace['step']
        trial_padding = padding
    return ProbeVerifier()

def get_job(name):
    protocol = json.loads((HERE.parent/'fixed_portfolio_2026-11-10/protocol_final.json').read_text())
    return next(j for j in protocol['jobs'] if j['id']==name).copy()

def check(job, variant='baseline', metric='class_box', padding=1.01, retain=True, step=15):
    v = verifier(variant, padding); v.metric_strategy = metric
    start = time.perf_counter()
    c = verify_command_plan(np.array(job['center']), np.array(job['commands']),
        [Fraction(s) for s in job['durations']], initial_time_h=Fraction(job['start']),
        amplitude=job['amplitude'], duration=job['event_duration'], initial_box=np.array(job['width']),
        residual_bound=job['w']/100, step_seconds=step, verifier=v, retain_steps=retain,
        service_checks=[dict(time_h=Fraction(1),coordinate=0,upper=job['deadline']/100)])
    elapsed = time.perf_counter()-start
    # Bind the additional implementation/configuration to a separate result digest.
    meta = dict(variant=variant, metric=metric, padding=padding, step=step,
        source_hashes=common.portable_source_hashes(), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if variant != 'baseline':
        c['proof_descriptor']['implementation_refinement']=meta
        c['proof_descriptor']['signed_generator_method']['series_order']=12 if variant in ('order12','tight12') else 18
        c['proof_id']=hashlib.sha256(json.dumps(c['proof_descriptor'],sort_keys=True).encode()).hexdigest()
        for checkpoint in c['reachable_checkpoints']:checkpoint['proof_id']=c['proof_id']
        c['adapter_audit']['proof_hash_recomputed']=hashlib.sha256(json.dumps(c['proof_descriptor'],sort_keys=True).encode()).hexdigest()==c['proof_id']
    result = dict(job=job,configuration=meta,certificate=c,
        summary=dict(accepted=c['accepted'],minutes=60*c['certified_until_h'],seconds=elapsed,
            service=c['service_check_results'],failures=c['failures'],
            terminal=c['terminal_coordinate_intervals']))
    result['result_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,default=str).encode()).hexdigest()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',default='depth_backup_4500')
    p.add_argument('--variant',default='baseline');p.add_argument('--metric',default='class_box')
    p.add_argument('--padding',type=float,default=1.01);p.add_argument('--width',type=float)
    p.add_argument('--profile',action='store_true');a=p.parse_args()
    job=get_job(a.job)
    if a.width is not None:job['width']=[a.width/100]*6+[min(a.width,10)/100]
    out=HERE/'results';out.mkdir(exist_ok=True)
    stem=f'{a.job}_{a.variant}_{a.metric}_p{a.padding}_d{a.width}'
    if a.profile:
        profiler=cProfile.Profile();profiler.enable()
    result=check(job,a.variant,a.metric,a.padding)
    if a.profile:
        profiler.disable();profiler.dump_stats(str(out/(stem+'.prof')))
        with (out/(stem+'_profile.txt')).open('w') as f:pstats.Stats(profiler,stream=f).sort_stats('cumtime').print_stats(35)
    save(out/(stem+'.json'),result)
    s=result['summary']
    print(json.dumps({k:s[k] for k in ('accepted','minutes','seconds','service')},default=str),flush=True)

if __name__=='__main__':main()
