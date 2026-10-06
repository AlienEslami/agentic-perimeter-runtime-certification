"""Matched six-point commissioning-margin study; original sources unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

import mfd_incident_model as model
from mfd_incident_service_frontier import solve_service, replay_plan, independent_fixed_replay
from mfd_incident_mpc_certification import ConstantCommandVerifier, verify_command_plan
from mfd_incident_positive_reachability import PositiveVerifier
import mfd_incident_affine_reachability as affine

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'reproducibility/results'
PROTOCOL = ROOT/'REACHABILITY_SERVICE_MARGIN_PROTOCOL.md'
OUTPUT = RESULTS/'mfd_reachability_service_margin_results.json'
AUDIT = RESULTS/'mfd_reachability_service_margin_audit.json'
TARGETS = [1700, 1690, 1675, 1650, 1625, 1600]
METHODS = ['original', 'positive_queue', 'affine_v3']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8')


def hashes():
    result = affine.source_hashes()
    for path in [Path(__file__), PROTOCOL, ROOT/'reproducibility/mfd_incident_service_frontier.py',
                 ROOT/'reproducibility/mfd_incident_mpc.py',
                 ROOT/'reproducibility/audit_mfd_incident_model.py',
                 ROOT/'reproducibility/audit_mfd_incident_library_study.py']:
        result[path.relative_to(ROOT).as_posix()] = digest(path)
    return result


def summary(payload):
    best = {}
    for method in METHODS:
        eligible = [r for r in payload['rows'] if r.get('proofs', {}).get(method, {}).get('accepted')]
        if eligible:
            row = min(eligible, key=lambda r: r['replay']['total_time_spent_veh_h'])
            best[method] = dict(nominal_target_vehicles=row['nominal_target_vehicles'],
                TTS_veh_h=row['replay']['total_time_spent_veh_h'],
                service_upper_vehicles=row['proofs'][method]['service_upper_vehicles'],
                proof_id=row['proofs'][method]['proof_id'])
        else:
            best[method] = None
    if best['original']:
        reference = best['original']['TTS_veh_h']
        for row in best.values():
            if row is not None:
                row['TTS_saving_vs_original_matched_grid_veh_h'] = reference-row['TTS_veh_h']
                row['TTS_saving_vs_original_matched_grid_percent'] = 100*(reference-row['TTS_veh_h'])/reference
    return best


def run():
    source_hashes = hashes()
    if OUTPUT.exists():
        payload = json.loads(OUTPUT.read_text())
        assert payload['source_sha256'] == source_hashes, 'Source drift; preserve previous results'
    else:
        snapshots = {}
        for name, sha in source_hashes.items():
            destination = RESULTS/'ars'/sha[:16]/Path(name).name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copyfile(ROOT/name, destination)
            assert digest(destination) == sha
            snapshots[name] = destination.relative_to(ROOT).as_posix()
        payload = dict(scope='matched discrete commissioning-margin grid; first fixed plan only',
            source_sha256=source_hashes, source_snapshots=snapshots,
            study_protocol_path=PROTOCOL.relative_to(ROOT).as_posix(), study_protocol_sha256=digest(PROTOCOL),
            targets_vehicles=TARGETS, methods=METHODS,
            trusted_contract=dict(service_limit_vehicles=1700, deadline_exact_h='1',
                physical_initial_half_widths_vehicles=[1.]*7, residual_vehicles_per_hour=60.,
                initial_state=model.BASE_STATE.tolist(), amplitude=.3, pulse_duration_h=1.,
                command_durations_exact_h=['1/12', '11/12'], proof_step_seconds=15.), rows=[])
        save(OUTPUT, payload)
    for target in TARGETS:
        existing = next((r for r in payload['rows'] if r['nominal_target_vehicles'] == target), None)
        if existing is None:
            started = time.perf_counter()
            selected, optimizer = solve_service(model.BASE_STATE,
                service_target_vehicles=target, deadline_h=1., horizon_h=1.)
            row = dict(nominal_target_vehicles=target, optimizer=optimizer,
                       optimizer_wall_seconds=time.perf_counter()-started, proofs={})
            if selected is None:
                row['classification'] = 'optimizer_failed_to_find_feasible_plan'
            else:
                commands = selected[1].reshape(2, 4)
                row['commands'] = commands.tolist()
                row['exact_command_fractions'] = [[str(Fraction.from_float(float(v))) for v in block] for block in commands]
                row['replay'] = replay_plan(model.BASE_STATE, commands, service_target_vehicles=1700.)
                row['independent_replay'] = independent_fixed_replay(model.BASE_STATE, commands)
                row['prediction_replay_deadline_difference_vehicles'] = row['replay']['deadline_n1_vehicles']-100*optimizer['predicted_target_n1']
                row['classification'] = 'nominal_plan_selected'
            payload['rows'].append(row)
            save(OUTPUT, payload)
        else:
            row = existing
        if 'commands' not in row:
            continue
        for method in METHODS:
            if method in row['proofs']:
                continue
            verifier = (ConstantCommandVerifier() if method == 'original' else
                        PositiveVerifier('positive_queue') if method == 'positive_queue' else affine.AffineGeneratorVerifier())
            certificate = verify_command_plan(model.BASE_STATE, row['commands'], [Fraction(1,12), Fraction(11,12)],
                initial_box=.01, residual_bound=.6, step_seconds=15., verifier=verifier,
                service_checks=[dict(time_h=Fraction(1), coordinate=0, upper=17.)], retain_steps=True)
            assert hashes() == source_hashes, 'Source drift during proof'
            proof_path = RESULTS/f'mfd_reachability_service_margin_n{target}_{method}_proof.json'
            proof_payload = dict(study_protocol_sha256=digest(PROTOCOL), source_sha256=source_hashes,
                nominal_target_vehicles=target, method=method, commands=row['commands'], certificate=certificate)
            save(proof_path, proof_payload)
            row['proofs'][method] = dict(accepted=certificate['accepted'],
                continuous_physical_certified=certificate['verified_continuous_physical'],
                scheduled_service_certified=certificate['verified_scheduled_service'],
                service_upper_vehicles=100*certificate['service_check_results'][0]['upper'] if certificate['service_check_results'] else None,
                certified_until_h=certificate['certified_until_h'], failures=[{k:v for k,v in f.items() if k not in ('attempts','failed_step')} for f in certificate['failures']],
                wall_seconds=certificate['wall_seconds'], proof_id=certificate['proof_id'],
                proof_file=proof_path.relative_to(ROOT).as_posix(), proof_sha256=digest(proof_path))
            payload['best_certified_grid_points'] = summary(payload)
            save(OUTPUT, payload)
            print(json.dumps(dict(target=target, method=method, nominal_TTS=row['replay']['total_time_spent_veh_h'],
                accepted=certificate['accepted'], service_upper_vehicles=row['proofs'][method]['service_upper_vehicles'],
                wall_seconds=certificate['wall_seconds'])), flush=True)
    payload['completed'] = len(payload['rows']) == len(TARGETS) and all('commands' not in r or len(r['proofs']) == 3 for r in payload['rows'])
    payload['best_certified_grid_points'] = summary(payload)
    save(OUTPUT, payload)
    audit()
    print(json.dumps(payload['best_certified_grid_points'], indent=2), flush=True)
    return payload


def audit():
    payload = json.loads(OUTPUT.read_text())
    assert payload['source_sha256'] == hashes()
    assert payload['study_protocol_sha256'] == digest(PROTOCOL)
    assert payload['targets_vehicles'] == TARGETS and payload['methods'] == METHODS
    assert [r['nominal_target_vehicles'] for r in payload['rows']] == TARGETS
    for name, snapshot in payload['source_snapshots'].items():
        assert digest(ROOT/snapshot) == payload['source_sha256'][name]
    errors, proof_count = [], 0
    for row in payload['rows']:
        if 'commands' not in row:
            assert not row['optimizer']['feasible_solution_found']
            continue
        assert np.array_equal(row['commands'], row['optimizer']['controls'])
        independent = independent_fixed_replay(model.BASE_STATE, row['commands'])
        replay = row['replay']
        error = dict(target=row['nominal_target_vehicles'],
            final_state_vehicles=100*float(np.max(abs(np.asarray(independent['final_state'])-replay['final_state']))),
            TTS_veh_h=abs(independent['total_time_spent_veh_h']-replay['total_time_spent_veh_h']),
            deadline_vehicles=abs(independent['deadline_n1_vehicles']-replay['deadline_n1_vehicles']))
        assert max(v for k,v in error.items() if k != 'target') < 1e-4, error
        assert abs(replay['conservation_residual_vehicles']) < 1e-6
        errors.append(error)
        descriptors = []
        for method in METHODS:
            item = row['proofs'][method]
            path = ROOT/item['proof_file']
            assert digest(path) == item['proof_sha256']
            data = json.loads(path.read_text())
            assert data['source_sha256'] == payload['source_sha256']
            assert data['study_protocol_sha256'] == payload['study_protocol_sha256']
            assert data['commands'] == row['commands']
            cert, desc = data['certificate'], data['certificate']['proof_descriptor']
            assert hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest() == cert['proof_id'] == item['proof_id']
            assert cert['accepted'] == item['accepted']
            assert cert['initial_box_units'] == [.01]*7
            assert desc['residual_bound'] == .6
            assert cert['adapter_audit']['commands'] == row['commands']
            assert cert['adapter_audit']['zero_feedback_gain']
            assert all(check['passed'] for check in cert['adapter_audit']['exact_actuator_checks'])
            assert Fraction(desc['initial_time']) == 0 and list(map(Fraction, desc['durations'])) == [Fraction(1,12),Fraction(11,12)]
            assert len(cert['service_check_results']) == 1
            assert cert['service_check_results'][0]['limit'] == 17.
            assert cert['service_check_results'][0]['time_h'] == 1.
            descriptors.append(desc)
            proof_count += 1
        for key in ['model_contract', 'pulse', 'service_checks', 'mode_fingerprints', 'durations', 'initial_time']:
            assert all(desc[key] == descriptors[0][key] for desc in descriptors), key
    assert payload['best_certified_grid_points'] == summary(payload)
    result = dict(passed=True, source_sha256=digest(Path(__file__)),
        results_sha256=digest(OUTPUT), protocol_sha256=digest(PROTOCOL),
        exact_proof_descriptors_checked=proof_count, all_targets_and_failures_retained=True,
        identical_commands_and_trusted_contract_across_methods=True,
        independent_replay_errors=errors, best_certified_grid_points=summary(payload))
    save(AUDIT, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(), indent=2)) if args.audit_only else run()
