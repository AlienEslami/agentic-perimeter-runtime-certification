"""Portable saved-data and preservation audit for the online manuscript release.

This reads existing evidence; it makes no LLM call and changes no simulation.
Use --prior-archive once to import the previous experiment's payload manifest.
Subsequent audits, including extracted-package checks, use that portable record.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'reproducibility/results'
PRIOR = RESULTS / 'online_structure_prior_payload_manifest.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-archive', type=Path)
    parser.add_argument('--output', type=Path, default=RESULTS/'online_revision_audit_results.json')
    args = parser.parse_args()
    if args.prior_archive:
        with zipfile.ZipFile(args.prior_archive) as archive:
            manifest = json.loads(archive.read('EXPERIMENT_MANIFEST.json'))
        generated = {'EXPERIMENT_ENVIRONMENT.json'}
        prior = dict(archive= args.prior_archive.name, archive_sha256=sha(args.prior_archive),
                     excluded_generated_payloads=sorted(generated),
                     sha256={k:v for k,v in manifest['sha256'].items() if k not in generated})
        if PRIOR.exists():
            assert read(PRIOR) == prior
        else:
            PRIOR.write_text(json.dumps(prior, indent=2)+'\n', encoding='utf-8')
    prior = read(PRIOR)
    for name, digest in prior['sha256'].items():
        assert sha(ROOT/name) == digest, ('changed prior scientific payload', name)

    runtime = read(RESULTS/'mfd_structure_online_runtime_results.json')
    audit = read(RESULTS/'mfd_structure_online_runtime_audit_results.json')
    assert runtime['completed'] and audit['passed'] and audit['complete']
    assert runtime['new_live_calls'] == 0 and runtime['scientific_kernel_changes'] is False
    assert len(runtime['frozen_menu']) == len(runtime['cases']) == audit['audited_cases'] == 17
    assert audit['audited_source_result_sha256'] == sha(RESULTS/'mfd_structure_online_runtime_results.json')
    assert audit['audit_source_sha256'] == sha(ROOT/'reproducibility/audit_mfd_structure_online_runtime.py')
    for name, digest in audit['artifact_sha256'].items():
        assert sha(ROOT/name) == digest, ('online artifact drift', name)
    runs = {r['case_id']: r for r in audit['runs']}
    assert sum(r['counts']['unresolved_initializations'] for r in runs.values()) == 4
    assert sum(r['counts']['no_candidate_epochs'] for r in runs.values()) == 0
    assert sum(r['counts']['failed_preconditions'] for r in runs.values()) == 0
    assert audit['totals']['proofs'] == 124
    assert audit['totals']['exact_sensor_boxes'] == 160
    assert audit['totals']['retained_epochs'] == 46
    for name in ('wide50_original', 'wide50_positive_queue', 'wide25_original', 'wide25_positive_queue'):
        assert runs[name]['counts']['unresolved_initializations'] == 1
        assert runs[name]['counts']['decisions'] == 1
    for name in ('wide50_affine_v3', 'wide25_affine_v3', 'adverse25_affine_v3'):
        assert runs[name]['counts']['accepted_candidates'] == 11
        assert runs[name]['counts']['rejected_full_certificates'] == 1
        assert 100*runs[name]['independent_replay']['final_state'][0] < 2850
    for name in ('binding1600_original', 'binding1600_positive_queue', 'binding1600_affine_v3', 'binding1625_affine_v3'):
        assert runs[name]['counts']['accepted_candidates'] == 2
        assert runs[name]['counts']['retained_epochs'] == 10
        assert 100*runs[name]['independent_replay']['final_state'][0] < 1700
    common = runs['binding1600_original']['independent_replay']['total_time_spent_veh_h']
    signed = runs['binding1625_affine_v3']['independent_replay']['total_time_spent_veh_h']
    for name in ('binding1600_positive_queue','binding1600_affine_v3'):
        assert runs[name]['independent_replay']['total_time_spent_veh_h'] == common
    nominal = runs['nominal_1700_c1600']['independent_replay']['total_time_spent_veh_h']
    savings = dict(common_tts=common, signed_tts=signed, nominal_tts=nominal,
                   difference_veh_h=common-signed, reduction_percent=100*(common-signed)/common,
                   common_premium_percent=100*(common-nominal)/nominal,
                   signed_premium_percent=100*(signed-nominal)/nominal)
    for key, expected in [('difference_veh_h','41.20'),('reduction_percent','0.90'),
                          ('common_premium_percent','3.52'),('signed_premium_percent','2.59')]:
        assert f'{savings[key]:.2f}' == expected
    held = read(RESULTS/'mfd_reachability_service_margin_results.json')['best_certified_grid_points']
    assert held['original']['nominal_target_vehicles'] == 1600
    assert held['positive_queue']['nominal_target_vehicles'] == 1600
    assert held['affine_v3']['nominal_target_vehicles'] == 1625
    assert f"{held['affine_v3']['TTS_saving_vs_original_matched_grid_veh_h']:.2f}" == '41.33'
    semantics = read(RESULTS/'mfd_online_structure_semantics_audit_results.json')
    assert semantics['passed'] and semantics['initial_shape_restart_rejected']
    for name, digest in semantics['source_sha256'].items():
        assert sha(ROOT/name) == digest, ('semantic audit provenance', name)
    live = [p for p in (ROOT/'live_llm_results').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    result = dict(passed=True, audit_source_sha256=sha(Path(__file__)),
                  prior_preserved_payloads=len(prior['sha256']), retained_live_files=len(live),
                  online_cases=17, unresolved_initializations=4,
                  online_audit_sha256=sha(RESULTS/'mfd_structure_online_runtime_audit_results.json'),
                  online_metrics=savings, held_plan_comparison_separate=True,
                  science_kernels_and_prior_results_unchanged=True, new_live_calls=0)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
