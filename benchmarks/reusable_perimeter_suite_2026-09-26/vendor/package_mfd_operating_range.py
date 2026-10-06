"""Archive the operating-range experiment with byte-checked data provenance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME='FOUNDATION_MFD_OPERATING_RANGE_AGENT_EXPERIMENT_2026-09-10'


def native(path):
    value=str(path.resolve())
    return '\\\\?\\'+value if os.name=='nt' and not value.startswith('\\\\?\\') else value


def data(path):
    with open(native(path),'rb') as handle: return handle.read()


def sha(value): return hashlib.sha256(value).hexdigest()


def sources():
    files=list((ROOT/'reproducibility').glob('*.py'))
    files+=list((ROOT/'reproducibility/results').glob('*.json'))
    files+=list((ROOT/'results').glob('*.json'))
    files += [p for p in (ROOT/'live_llm_results').rglob('*') if p.is_file()]
    for name in ('ors','operating_range_source_snapshots'):
        files += [p for p in (ROOT/'reproducibility/results'/name).rglob('*') if p.is_file()]
    files += list(ROOT.glob('*PROTOCOL.md'))
    files += [ROOT/p for p in ['OPERATING_RANGE_AGENT_RESULTS.md','requirements.txt',
        'requirements-synthesis.txt','MPC_CERTIFICATION_IMPLEMENTATION.md','MPC_CERTIFICATION_RESULTS.md',
        'INCIDENT_MODEL_AUDIT.md','Archived/mfd_incident_verifier_pre_mpc.py']]
    for name in ('mfd_incident_service_retention','mfd_operating_frontiers'):
        files += [ROOT/'reproducibility/results'/(name+suffix) for suffix in ('.png','.pdf')]
    return sorted(set(files))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/(NAME+'.zip')
    if path.exists(): raise FileExistsError(path)
    mandatory=['mfd_operating_range_aggregate_audit_results.json',
        'mfd_incident_live_agent_independent_audit_results.json',
        'mfd_incident_service_frontier_audit_results.json',
        'mfd_incident_uncertainty_frontiers_audit_results.json']
    for name in mandatory:
        audit=json.loads(data(ROOT/'reproducibility/results'/name))
        assert audit.get('passed',audit.get('status')=='PASS'),name
    files=sources()
    manifest=dict(kind='Experimental supplement; manuscript release and model weights excluded',
        report='OPERATING_RANGE_AGENT_RESULTS.md',prior_live_files_retained=387,new_live_calls=32,
        selection='All current reproducibility Python modules, top-level JSON results, historical and new raw live files; new protocols, plots, proofs and source snapshots.',
        sha256={p.relative_to(ROOT).as_posix():sha(data(p)) for p in files})
    environment=dict(python=platform.python_version(),platform=platform.platform(),
        installed_packages={name:importlib.metadata.version(name) for name in
        ('numpy','scipy','matplotlib','mpmath','cvxpy','clarabel')},
        model='qwen3:8b; exact local model digest and inference options are stored in raw session records')
    environment_text=json.dumps(environment,indent=2)+'\n'
    manifest['sha256']['EXPERIMENT_ENVIRONMENT.json']=sha(environment_text.encode())
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source in files:
            name=source.relative_to(ROOT).as_posix()
            payload=data(source)
            assert sha(payload)==manifest['sha256'][name]
            z.writestr(name,payload)
        z.writestr('EXPERIMENT_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('EXPERIMENT_ENVIRONMENT.json',environment_text)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for name,expected in manifest['sha256'].items(): assert sha(z.read(name))==expected,name
        assert len(z.namelist())==len(files)+2
    for source,target in [(ROOT/'OPERATING_RANGE_AGENT_RESULTS.md',args.output_dir/'OPERATING_RANGE_AGENT_RESULTS_2026-09-10.md')]+[
        (ROOT/'reproducibility/results'/(name+suffix),args.output_dir/'reproducibility/results'/(name+suffix))
        for name in ('mfd_incident_service_retention','mfd_operating_frontiers') for suffix in ('.png','.pdf')]:
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists(): assert data(target)==data(source),target
        else: shutil.copy2(native(source),native(target))
    print(json.dumps(dict(path=str(path),files=len(files)+2,megabytes=path.stat().st_size/1e6,
        sha256=sha(data(path)),all_archive_payload_hashes_verified=True),indent=2))


if __name__=='__main__': main()
