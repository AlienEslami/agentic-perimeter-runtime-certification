"""Create a new, byte-verified experiment release without altering old releases."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'FOUNDATION_MFD_STRUCTURE_PRESERVING_REACHABILITY_2026-09-10'
REPORT = 'STRUCTURE_PRESERVING_REACHABILITY_RESULTS.md'


def native(path):
    value = str(path.resolve())
    return '\\\\?\\' + value if os.name == 'nt' and not value.startswith('\\\\?\\') else value


def data(path):
    with open(native(path), 'rb') as handle:
        return handle.read()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def sources():
    paths = list((ROOT/'reproducibility').glob('*.py'))
    paths += list((ROOT/'reproducibility/results').glob('*.json'))
    paths += list((ROOT/'reproducibility/results').glob('*executed_source.py'))
    paths += list((ROOT/'results').glob('*.json'))
    for folder in ('ors', 'pr', 'ars', 'ar', 'operating_range_source_snapshots'):
        paths += [p for p in (ROOT/'reproducibility/results'/folder).rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'live_llm_results').rglob('*') if p.is_file()]
    paths += list(ROOT.glob('*PROTOCOL.md'))
    paths += list(ROOT.glob('*REACHABILITY*REVIEW.md'))
    paths += [ROOT/p for p in (REPORT, 'DESTINATION_STRIP_ADDENDUM.md',
        'OPERATING_RANGE_AGENT_RESULTS.md', 'requirements.txt', 'requirements-synthesis.txt',
        'MPC_CERTIFICATION_IMPLEMENTATION.md', 'MPC_CERTIFICATION_RESULTS.md',
        'INCIDENT_MODEL_AUDIT.md', 'Archived/mfd_incident_verifier_pre_mpc.py')]
    for stem in ('mfd_incident_service_retention', 'mfd_operating_frontiers',
                 'mfd_structure_preserving_frontiers', 'mfd_reachability_service_margin'):
        for suffix in ('.png', '.pdf'):
            path = ROOT/'reproducibility/results'/(stem+suffix)
            if path.exists():
                paths.append(path)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--audit', action='append', default=[])
    args = parser.parse_args()
    mandatory = ['mfd_structure_preserving_aggregate_audit_results.json'] + args.audit
    for name in mandatory:
        audit = json.loads(data(ROOT/'reproducibility/results'/name))
        assert audit.get('passed', audit.get('status') == 'PASS'), name
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir/(NAME+'.zip')
    if target.exists():
        raise FileExistsError(target)
    files = sources()
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        installed_packages={name: importlib.metadata.version(name) for name in
                            ('numpy', 'scipy', 'matplotlib', 'mpmath', 'cvxpy', 'clarabel')},
        new_live_calls=0, manuscript_changed=False, commit_or_push=False)
    environment_bytes = (json.dumps(environment, indent=2)+'\n').encode()
    manifest = dict(kind='Structure-preserving finite-plan reachability experiment', report=REPORT,
        selection='Current Python modules and top-level JSON results; executed-version source archives; protocols and proof reviews; prior raw live records retained for provenance. Manuscript and model weights excluded.',
        audits=mandatory, sha256={p.relative_to(ROOT).as_posix(): sha(data(p)) for p in files})
    manifest['sha256']['EXPERIMENT_ENVIRONMENT.json'] = sha(environment_bytes)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            payload = data(path)
            assert sha(payload) == manifest['sha256'][name], ('Source drift', name)
            archive.writestr(name, payload)
        archive.writestr('EXPERIMENT_ENVIRONMENT.json', environment_bytes)
        archive.writestr('EXPERIMENT_MANIFEST.json', json.dumps(manifest, indent=2)+'\n')
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(files)+2
        for name, expected in manifest['sha256'].items():
            assert sha(archive.read(name)) == expected, name
    copies = [(ROOT/REPORT, args.output_dir/'STRUCTURE_PRESERVING_REACHABILITY_RESULTS_2026-09-10.md')]
    for suffix in ('.png', '.pdf'):
        relative = Path('reproducibility/results')/('mfd_structure_preserving_frontiers'+suffix)
        copies.append((ROOT/relative, args.output_dir/relative))
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            assert data(source) == data(destination)
        else:
            shutil.copy2(native(source), native(destination))
    print(json.dumps(dict(path=str(target), files=len(files)+2,
        megabytes=target.stat().st_size/1e6, sha256=sha(data(target)),
        archive_crc_and_all_payload_sha256_passed=True), indent=2))


if __name__ == '__main__':
    main()
