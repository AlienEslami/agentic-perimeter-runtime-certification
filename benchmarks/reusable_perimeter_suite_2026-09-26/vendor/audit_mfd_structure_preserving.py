"""Matched-contract/provenance audit; arithmetic audits are separate artifacts."""
from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reproducibility/results'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bound_hash(descriptor):
    return hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()


def resolve_snapshot(name, expected):
    current = ROOT / name.replace('\\', '/')
    if current.is_file() and sha(current) == expected:
        return current
    for folder in ('pr', 'ars', 'ors', 'ar'):
        path = RESULTS / folder / expected[:16] / current.name
        if path.is_file() and sha(path) == expected:
            return path
    for version in ('v1', 'v2'):
        path = RESULTS / f'mfd_incident_affine_reachability_{version}_executed_source.py'
        if path.is_file() and sha(path) == expected:
            return path
    raise AssertionError(('No matching executed source', name, expected))


def inspect_certificate(payload, reference, width, residual, version):
    cert = payload['certificate']
    desc = cert['proof_descriptor']
    assert cert['proof_id'] == bound_hash(desc)
    assert desc['durations'] == reference['exact_durations_h']
    assert Fraction(desc['initial_time']) == 0
    assert desc['initial_state'] == reference['initial_state']
    assert desc['model_contract'] == reference['_model_contract']
    assert desc['pulse'] == reference['_pulse']
    assert desc['affine_constraint_contracts'] == reference['_affine_constraint_contracts']
    assert desc['service_checks'] == [{'exact_time': '1', 'coordinate': 0, 'upper': 28.5}]
    assert Fraction(desc['residual_bound']) == Fraction.from_float(residual / 100.)
    assert desc['step_seconds'] == 15.
    assert np.array_equal(cert['adapter_audit']['commands'], reference['commands'])
    widths = [width / 100.] * 6 + [min(width, 10.) / 100.]
    assert np.array_equal(cert['initial_box_units'], widths)
    assert all(a['passed'] for a in cert['adapter_audit']['exact_actuator_checks'])
    for checkpoint in cert['reachable_checkpoints']:
        assert checkpoint['proof_id'] == cert['proof_id']
        if version == 'affine_v3':
            assert checkpoint['outer_hull_only'] and not checkpoint['restart_authorized']
    if cert['accepted']:
        assert cert['verified_continuous_physical']
        assert cert['verified_scheduled_service'] and cert['verified_terminal_service']
        assert cert['certified_until_h'] == 1.
        assert all(item['passed'] for item in cert['service_check_results'])
        margins = cert['continuous_constraint_margins_lower']
        assert min(margins.values() if isinstance(margins, dict) else margins) >= 0.
    if version == 'affine_v3':
        method = desc['signed_generator_method']
        assert method['source_sha256'] == payload['source_sha256']
        assert not method['generator_or_remainder_contraction']
        assert not method['generator_reduction']
    for name, expected in payload.get('source_sha256', {}).items():
        resolve_snapshot(name, expected)
    service = cert['service_check_results']
    return dict(accepted=cert['accepted'], proof_id=cert['proof_id'],
                service_upper_vehicles=100.*service[0]['upper'] if service else None,
                certified_until_h=cert['certified_until_h'],
                first_failure=({key: cert['failures'][0][key] for key in
                    ('kind', 'constraint', 'time_h', 'left_h', 'right_h')
                    if key in cert['failures'][0]} if cert['failures'] else None),
                completed_steps=cert['completed_interval_steps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-archive', type=Path)
    parser.add_argument('--prior-manuscript', type=Path)
    args = parser.parse_args()
    reference_path = RESULTS / 'mfd_incident_uncertainty_frontier_results.json'
    reference = read(reference_path)
    baseline_proof = read(ROOT/reference['rows'][0]['proof_file'])
    baseline = baseline_proof.get('certificate', baseline_proof)['proof_descriptor']
    for key in ('model_contract', 'pulse', 'affine_constraint_contracts'):
        reference['_'+key] = baseline[key]
    records, input_hashes = [], {str(reference_path.relative_to(ROOT)): sha(reference_path)}
    for path in sorted(RESULTS.glob('mfd_positive_*.json')):
        payload = read(path)
        if 'certificate' not in payload:
            continue
        width, residual = payload['state_half_width_vehicles'], payload['residual_vehicles_h']
        record = inspect_certificate(payload, reference, width, residual, 'positive')
        record.update(method=payload['variant'], version=payload['tag'],
            state_half_width_vehicles=width, residual_vehicles_per_hour=residual,
            file=path.relative_to(ROOT).as_posix())
        records.append(record)
        input_hashes[record['file']] = sha(path)
    for version in ('v1', 'v2', 'v3'):
        name = ('mfd_incident_affine_reachability_results.json' if version == 'v1'
                else f'mfd_incident_affine_reachability_{version}_results.json')
        path = RESULTS / name
        result = read(path)
        input_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
        for name, expected in result['source_sha256'].items():
            resolve_snapshot(name, expected)
        for row in result['rows']:
            proof_path = ROOT / row['proof_file']
            assert sha(proof_path) == row['proof_sha256']
            width, residual = row['state_half_width_vehicles'], row['residual_vehicles_per_hour']
            record = inspect_certificate(read(proof_path), reference, width, residual, 'affine_'+version)
            assert record['accepted'] == row['accepted']
            record.update(method='signed_affine', version=version,
                state_half_width_vehicles=width, residual_vehicles_per_hour=residual,
                file=proof_path.relative_to(ROOT).as_posix())
            records.append(record)
            input_hashes[record['file']] = sha(proof_path)
        if version in ('v2', 'v3'):
            observed = {(r['state_half_width_vehicles'], r['residual_vehicles_per_hour']) for r in result['rows']}
            protocol = result['protocol']
            expected = set(map(tuple, protocol['full_grid'] if version == 'v2' else
                               protocol['first_tests'] + protocol['followup_grid']))
            assert observed == expected, ('Incomplete declared grid', version, expected-observed)
    preserved = []
    if args.prior_archive:
        with zipfile.ZipFile(args.prior_archive) as archive:
            manifest = json.loads(archive.read('EXPERIMENT_MANIFEST.json'))
            for name, expected in manifest['sha256'].items():
                if name == 'EXPERIMENT_ENVIRONMENT.json':
                    continue
                path = ROOT / name
                assert sha(path) == expected, ('Prior payload changed', name)
                preserved.append(name)
    paper = []
    if args.prior_manuscript:
        names = ['FOUNDATION_MFD_PARTB_ELSEVIER.tex', 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY.tex',
                 'FOUNDATION_REFERENCES.bib']
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/'manuscript_streamlined').glob('*.tex')]
        for name in names:
            assert sha(ROOT/name) == sha(args.prior_manuscript/name)
            paper.append(name)
    output = dict(passed=True, scope='Independent matched-contract and source-preservation audit; consult separate arithmetic audits.',
        fixed_plan=reference['commands'], fixed_horizon_h=1., fixed_service_vehicles=2850.,
        unchanged_prior_archive_files=len(preserved), unchanged_manuscript_sources=paper,
        proof_count=len(records), rows=records, input_sha256=input_hashes,
        audit_source_sha256=sha(Path(__file__)),
        provenance_notes=['v1/v2 affine method sources are externally bound by saved payload hashes; v3 binds method and immutable protocol inside its proof descriptor.',
                          'Failed and less effective prototypes remain in the record. No new model calls or manuscript edits.'])
    target = RESULTS / 'mfd_structure_preserving_aggregate_audit_results.json'
    target.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k not in ('rows', 'input_sha256', 'fixed_plan')}, indent=2))


if __name__ == '__main__':
    main()
