"""Read-only checks of the current manuscript's saved evidence; no new runs."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
read = lambda p: json.loads((ROOT / p).read_text(encoding='utf-8'))
manifest = read('PAPER_RELEASE_MANIFEST_2026-11-15.json')
for name, expected in manifest['copied_research_files'].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
paper = ROOT / 'paper/reduced_2026-11-15/main.tex'
assert hashlib.sha256(paper.read_bytes()).hexdigest() == manifest['source_sha256']
SUITE = 'benchmarks/reusable_perimeter_suite_2026-09-26/results/'
comparison = read('benchmarks/partb_traffic_presentation_2026-11-15/matched_controller_table.json')
case = read(SUITE + 'contracts/D027.json')
for label, key in [('Nominal MPC', 'nominal_mpc'), ('Current plan', 'continuation'), ('Nominal-plan repair', 'grid_repair')]:
    samples = case['policies'][key]['realizations']
    row = next(r for r in comparison['rows'] if r['plan'] == label)
    assert len(samples) == 640
    assert abs(np.median([r['tts_veh_h'] for r in samples])-row['median_tts']) < 1e-8
    assert abs(max(r['peak_queue_vehicles'] for r in samples)-row['maximum_queue']) < 1e-8
    assert sum(r['physical_pass'] and r['service_pass'] for r in samples) == row['both_passes']
cbf = read('benchmarks/partb_traffic_presentation_2026-11-15/cbf_ensemble.json')['realizations']
assert len(cbf) == 640 and sum(r['service_pass'] for r in cbf) == 0
evidence = read('output/review/results_refinement_2026_11_15/evidence_and_source_checks.json')
for row in evidence['larger_uncertainty_comparison']:
    record = read(SUITE + 'contracts/' + row['case'] + '.json')['policies']['grid_repair']
    assert record['certificate']['accepted']
    assert abs(max(r['peak_queue_vehicles'] for r in record['realizations'])-row['simulated_max_queue']) < 1e-8
    assert sum(r['physical_pass'] and r['service_pass'] for r in record['realizations']) == 640
plant = read('benchmarks/reviewer_strengthening_2026-11-11/plant_sensitivity.json')
repair = [r for r in plant['rows'] if r['plan'] == 'repair']
assert len(repair) == 9
assert sum(r['service_passes'] for r in repair) == 5480
failures = [r for r in repair if r['service_passes'] < 640]
assert len(failures) == 1
assert failures[0]['mfd_multiplier'] == .95 and failures[0]['demand_multiplier'] == 1.05
assert abs(failures[0]['maximum_deadline_n1_vehicles']-4174.047202516222) < 1e-7
mapping = read('output/review/section6_revision_2026_11_15/verified_evidence.json')['acceptance_map']
assert len(mapping) == 80 and sum(r['full_sequential'] for r in mapping) == 50
assert sum(r['full_signed'] for r in mapping) == 44
print(json.dumps({'passed': True, 'release_files_hash_verified': len(manifest['copied_research_files']),
                  'matched_samples_per_plan': 640, 'larger_error_cases': 2,
                  'model_error_plants_per_plan': 9, 'acceptance_map_cases': 80,
                  'new_experiments': False}, indent=2))
