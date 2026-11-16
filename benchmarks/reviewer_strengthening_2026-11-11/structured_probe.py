"""Exploratory generator-plus-remainder trial tube, with a first-exit remainder check.

The trial tube is p(s)+L^-1 Gamma(s)[-1,1]+L^-1[-U,U]. Its convexity
and inclusion of p(s) cover the complete mean-value segment. Bounds on
J are intersected only when both evaluations cover this entire tube.
Acceptance requires the continuous remainder bound to lie strictly in U.
"""
import argparse,hashlib,inspect,json,textwrap
import numpy as np
import probe
from probe import affine,core,HERE,save

def make_verifier(padding=1.1):
    v=probe.verifier('tight12',padding)
    ns=dict(v.step.__globals__)
    record_source=inspect.getsource(core.transformed_jacobian_record)
    old='states = [Polynomial.cast(mode.center[i]) + sum(variables[j] * inverse[i, j] for j in range(DIM))\n              for i in range(DIM)]'
    assert old in record_source
    record_source=record_source.replace(old,'states = [Polynomial.cast(mode.center[i]) + variables[i] for i in range(DIM)]')
    needle='    # Switching to supply-limited admission changes only the constant Jacobian.'
    replacement='''    # Derivative above is L Df in physical polynomial variables; transform columns exactly before range evaluation.
    transformed_coefficients = [imat(I(coefficients.lo[:,k].reshape(DIM,DIM),coefficients.hi[:,k].reshape(DIM,DIM)),inverse) for k in range(len(keys))]
    coefficients = I(np.stack([c.lo.ravel() for c in transformed_coefficients],axis=1),np.stack([c.hi.ravel() for c in transformed_coefficients],axis=1))
'''+needle
    assert needle in record_source;record_source=record_source.replace(needle,replacement)
    rec_ns=dict(vars(core));rec_ns['imat']=probe.vector_imat;exec(record_source,rec_ns)
    original_geometry=v.geometry
    def geometry(mode):
        g=original_geometry(mode)
        if 'physical_coordinate_record' not in g:
            g['physical_coordinate_record']=rec_ns['transformed_jacobian_record'](mode,g['L'],g['T'])
        return g
    v.geometry=geometry
    source=textwrap.dedent(inspect.getsource(affine.AffineGeneratorVerifier.step))
    source=source.replace('I(1.1)','I(self.trial_padding)').replace('range(24)','range(64)')
    old='trial = (I(self.trial_padding)*(I(np.maximum(r0, flow_support[:DIM]))+I(h.hi)*I(base_forcing)+I(1e-12))).hi'
    new='trial = (I(self.trial_padding)*(I(rho)+I(h.hi)*I(base_forcing)+I(1e-10))).hi'
    assert old in source;source=source.replace(old,new)
    source=source.replace('physical_support = imat(I(inverse.mag()), I(trial)).hi',
        'physical_support = (I(flow_support[DIM:DIM+DIM])+imat(I(inverse.mag()), I(trial))).hi')
    source=source.replace('metric_box = metric_curve+I(-trial, trial)',
        'total_trial = (I(flow_support[:DIM])+I(trial)).hi\n        metric_box = metric_curve+I(-total_trial, total_trial)')
    needle='        M = np.maximum.reduce([matrix.mag() for matrix in matrices])'
    addition='''        direct = [evaluate_transformed_jacobian(geo['physical_coordinate_record'], physical_box-I(mode.center), smooth, centered="exact")
                  for smooth in (("gate", "supply") if branch == "both" else (branch,))]
        matrices = [I(np.maximum(a.lo,b.lo),np.minimum(a.hi,b.hi)) for a,b in zip(matrices,direct)]
        assert all(np.all(a.lo<=a.hi) for a in matrices)
'''
    assert needle in source;source=source.replace(needle,addition+needle)
    source=source.replace('if np.all(continuous_radius < trial):','if np.all(continuous_remainder < trial):')
    source=source.replace('I(continuous_radius)*I(self.trial_padding)','I(continuous_remainder)*I(self.trial_padding)')
    source=source.replace('apriori_radius=trial.tolist()', 'apriori_radius=total_trial.tolist()')
    source=source.replace('attempts.append(dict(trial_radius=trial.tolist(),',
        'attempts.append(dict(trial_radius=total_trial.tolist(), trial_remainder=trial.tolist(), continuous_remainder=continuous_remainder.tolist(),')
    exec(source,ns)
    v.step=ns['step'].__get__(v,type(v));return v

def run(job,metric='class_box',retain=False):
    original=probe.verifier
    v=make_verifier()
    probe.verifier=lambda *args:v
    try:r=probe.check(job,'tight12',metric,1.1,retain)
    finally:probe.verifier=original
    c=r['certificate'];c['proof_descriptor']['structured_trial']=dict(
        definition='p+Linv Gamma xi+Linv eta; |xi|<=1, |eta|<U',
        strict_check='continuous_remainder < trial_remainder',
        jacobian_intersection='transformed-polynomial and transformed physical derivative bounds on the same structured tube',
        source_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    c['proof_id']=hashlib.sha256(json.dumps(c['proof_descriptor'],sort_keys=True).encode()).hexdigest()
    for cp in c['reachable_checkpoints']:cp['proof_id']=c['proof_id']
    return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',default='backup');p.add_argument('--width',type=float,default=50)
    p.add_argument('--metric',default='class_box');a=p.parse_args()
    job=probe.get_job(f'depth_{a.plan}_4500');job['width']=[a.width/100]*6+[.1]
    r=run(job,a.metric,True);save(HERE/'results'/f'structured_{a.plan}_d{a.width}_{a.metric}.json',r)
    print(json.dumps({k:r['summary'][k] for k in ('accepted','minutes','seconds','service')}),flush=True)
