"""Check the incident manuscript against the frozen numerical/audit records.

This checks reported numbers, source bindings and rendered deliverables. It is
separate from the interval-flow proof and independent adaptive model audits.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reproducibility/results'


def read(name, folder=RESULTS):
    return json.loads((folder / name).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source_path = ROOT / 'FOUNDATION_MFD_PARTB_ELSEVIER.tex'
    source = source_path.read_text(encoding='utf-8')
    grid_name = 'mfd_incident_operational_v2_certified_grid_results.json'
    nominal_name = 'mfd_incident_operational_v2_nominal_grid_results.json'
    grid, nominal = read(grid_name), read(nominal_name)
    audit = read('mfd_incident_certified_grid_audit_results.json')
    nominal_audit = read('mfd_incident_operational_v2_nominal_audit_results.json')
    assert audit['status'] == 'pass' and audit['complete_grid']
    assert audit['available_cases'] == audit['expected_cases'] == 8
    assert audit['source_sha256'] == digest(RESULTS / grid_name)
    assert nominal_audit['status'] == 'pass'
    assert nominal_audit['source_sha256'] == digest(RESULTS / nominal_name)
    by_case = {(r['amplitude'], r['duration_h']): r for r in nominal['runs']}
    import mfd_incident_operational_study as operating
    modes, protocol_hash = operating.commission()
    assert protocol_hash == grid['commissioned_protocol_sha256']
    assert [mode.record() for mode in modes] == grid['runs'][0]['controller_library']
    accepted = []
    for run in grid['runs']:
        other = by_case[(run['amplitude'], run['duration_h'])]
        assert run['full_horizon_completed'] and run['all_decisions_certified']
        assert run['service_pass'] and not run['fallbacks'] and not run['rejected_certificates']
        for key in ('times', 'states', 'commands', 'total_time_spent_veh_h',
                    'queue_time_veh_h', 'departures', 'changes'):
            assert run[key] == other[key], key
        for event in run['events']:
            assert event['retained_plan_certificate_verified']
            candidates = [a for a in event['verification_attempts'] if a['certificate']['accepted']]
            assert len(candidates) == 1
            accepted.append(candidates[0]['certificate'])
    assert len(accepted) == 192

    comparison = read('mfd_incident_results_comparison.json')
    assert comparison['audit_binds_current_nominal_source']
    assert comparison['summary']['cases'] == 8
    assert all(c['differences']['library_minus_mpc_tts_veh_h'] > 0
               for c in comparison['cases'])
    primary = next(c for c in comparison['cases'] if c['amplitude'] == .3 and c['duration_h'] == 1.)
    table = source.split('\\label{tab:incident-primary}', 1)[1].split('\\end{table}', 1)[0]
    expected_rows = {
        'Queue-blind incumbent': (14462.5, 2462.5, 1850., 3000.),
        'Nominal library': tuple(primary['library'][k] for k in
                                 ('tts_veh_h', 'queue_veh_h', 'max_queue_vehicles', 'deadline_n1_vehicles')),
        'Certified library': tuple(primary['library'][k] for k in
                                   ('tts_veh_h', 'queue_veh_h', 'max_queue_vehicles', 'deadline_n1_vehicles')),
        'Continuous-input MPC': tuple(primary['mpc'][k] for k in
                                      ('tts_veh_h', 'queue_veh_h', 'max_queue_vehicles', 'deadline_n1_vehicles')),
    }
    for label, expected in expected_rows.items():
        row = next(line for line in table.splitlines() if line.startswith(label + ' &'))
        shown = tuple(float(part.strip().strip('\\')) for part in row.split('&')[1:])
        assert np.max(np.abs(np.array(shown)-expected)) <= .00501, (label, shown, expected)

    pilot = read('mfd_incident_verifier_pilot_results.json', ROOT / 'results')
    ablations = {a['strategy']: a for a in pilot['coordinate_ablation']}
    assert not ablations['physical_box']['accepted']
    assert ablations['class_box']['accepted'] and ablations['conservation_box']['accepted']
    for strategy, printed in (('class_box', 2219.66), ('conservation_box', 2168.57)):
        actual = 100 * ablations[strategy]['service'][0]['upper']
        assert actual <= printed < actual + .01
        assert str(printed) in source
    adverse = read('mfd_incident_adverse_fixed_plan_audit_results.json')
    assert len(adverse['cases']) == 48 and adverse['status'] == 'no_falsification_found'
    assert adverse['all_sampled_points_inside_retained_checkpoints']
    assert (RESULTS / 'mfd_incident_retained_suffix_audit_results.json').is_file()
    assert (ROOT / 'results/mfd_incident_counterproposal_results.json').is_file()
    assert 'intermediate predictor or' in source and 'expiry' in source
    assert 'Author preprint: arXiv:2508.19526v1' in (ROOT / 'FOUNDATION_REFERENCES.bib').read_text()

    source_labels = re.findall(r'\\label\{([^}]+)\}', source)
    assert len(source_labels) == len(set(source_labels)), 'duplicate source labels'
    referenced = {key for key in re.findall(r'\\(?:ref|eqref|appref)\{([^}]+)\}', source)
                  if not key.startswith('#')}
    missing = referenced-set(source_labels)
    assert not missing, missing
    report = dict(passed=False, source_sha256=digest(source_path), cases=8,
                  accepted_plan_certificates=len(accepted), matched_nominal_paths=True,
                  primary_table_basis='independent adaptive nominal replays',
                  maximum_queue_vehicles=max(r['maximum_queue_vehicles'] for r in grid['runs']),
                  proof_scope='componentwise source proof plus independent contract/numerical audits',
                  pdfs={})
    for stem in ('FOUNDATION_MFD_PARTB_ELSEVIER', 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY'):
        log = (ROOT / (stem + '.log')).read_text(encoding='utf-8', errors='replace')
        for forbidden in ('Overfull \\hbox', 'Overfull \\vbox', 'undefined references',
                          'There were undefined citations', 'Rerun to get cross-references right'):
            assert forbidden not in log, (stem, forbidden)
        assert (ROOT / (stem + '.bbl')).read_text().count('\\bibitem') == 57
        pdf = ROOT / (stem + '.pdf')
        reader = PdfReader(pdf)
        extracted = ' '.join(' '.join((p.extract_text() or '') for p in reader.pages).split())
        for fragment in ('Queue-Aware', 'Conservation', '9654.68', '7852.00', '2168.57', '5.01'):
            assert fragment.lower() in extracted.lower(), (stem, fragment)
        report['pdfs'][stem] = dict(pages=len(reader.pages), sha256=digest(pdf))
    report['passed'] = True
    (RESULTS / 'mfd_incident_manuscript_audit_results.json').write_text(
        json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
