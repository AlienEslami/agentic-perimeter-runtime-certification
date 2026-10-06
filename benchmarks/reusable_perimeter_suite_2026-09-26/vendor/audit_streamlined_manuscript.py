"""Read-only source/data/build audit of the streamlined manuscript.

Writes only the explicitly named audit report. No simulation, synthesis,
live call, or existing evidence record is rerun or modified. --source-only
defers generated IEEE/PDF/log checks while the document is being built.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
MAIN = 'FOUNDATION_MFD_PARTB_ELSEVIER.tex'
IEEE = 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY.tex'
SUMMARY = 'reproducibility/results/mfd_incident_mpc_certification_summary_results.json'
COMPARISON = 'reproducibility/results/mfd_incident_results_comparison.json'
LIVE = 'live_llm_results/cubic/sequential_semantic_agent_v3'
OLD_RELEASE = Path('C:/Users/alien/Documents/Mathematical foundation/output/package/'
                   'FOUNDATION_MFD_PARTB_REVIEWED_2026-09-10_INCIDENT')
EXTERNAL_INPUTS = {'glyphtounicode'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def compact(text):
    return re.sub(r'\s+', '', text)


def expand(path, root, files, active=()):
    path = path.resolve()
    assert path not in active, f'Recursive input cycle: {path}'
    assert path.is_file(), f'Missing manuscript input: {path}'
    files[path.relative_to(root).as_posix()] = digest(path)
    text = uncomment(path.read_text(encoding='utf-8-sig'))

    def include(match):
        name = match[1]
        if name in EXTERNAL_INPUTS:
            return match[0]
        relative = Path(name if Path(name).suffix else name + '.tex')
        candidates = [root / relative, path.parent / relative]
        target = next((p for p in candidates if p.is_file()), None)
        assert target is not None, f'Unresolved input {name!r} in {path.name}'
        assert target.resolve().is_relative_to(root), f'Input outside project: {target}'
        return expand(target, root, files, (*active, path))

    return re.sub(r'\\(?:input|include)\s*\{([^}]+)\}', include, text)


def table_block(text, label):
    blocks = re.findall(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', text, re.S)
    selected = [b for b in blocks if '\\label{' + label + '}' in b]
    assert len(selected) == 1, f'Missing or duplicate table: {label}'
    return selected[0]


def numeric_rows(block):
    rows = []
    for raw in block.split('\\\\'):
        if '&' not in raw:
            continue
        cells = raw.strip().split('&')
        numeric = []
        for cell in cells[1:]:
            clean = cell.strip().replace(r'\%', '').replace('$', '')
            if re.fullmatch(r'-?\d+(?:\.\d+)?', clean):
                numeric.append(float(clean))
            else:
                break
        if len(numeric) == len(cells) - 1:
            first = re.sub(r'\\(?:midrule|toprule|bottomrule)', '', cells[0]).strip()
            rows.append((first, numeric))
    return rows


def rounded_equal(actual, expected, places=2):
    return f'{actual:.{places}f}' == f'{expected:.{places}f}'


def check_source(root, source, report):
    files = {}
    text = expand(root / source, root, files)
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    duplicate = [k for k, n in Counter(labels).items() if n > 1]
    assert not duplicate, f'Duplicate labels: {duplicate}'
    refs = []
    for value in re.findall(r'\\(?:ref|eqref|pageref|appref|autoref|[cC]ref)\*?\{([^}]+)\}', text):
        refs.extend(k.strip() for k in value.split(',') if '#' not in k)
    missing = sorted(set(refs) - set(labels))
    assert not missing, f'Unresolved references: {missing}'
    cites = set()
    for value in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', text):
        cites.update(k.strip() for k in value.split(',') if k.strip() != '*')
    bibliographies = re.findall(r'\\bibliography\{([^}]+)\}', text)
    bibkeys = set()
    bibfiles = {}
    for names in bibliographies:
        for name in names.split(','):
            path = root / (name.strip() if name.strip().endswith('.bib') else name.strip() + '.bib')
            assert path.is_file(), f'Missing bibliography: {path}'
            bibfiles[path.relative_to(root).as_posix()] = digest(path)
            bibkeys.update(re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', path.read_text(encoding='utf-8-sig')))
    assert not (cites - bibkeys), f'Unresolved citations: {sorted(cites - bibkeys)}'
    graphics = {}
    for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
        candidates = [root / name, root / 'figures' / name]
        if not Path(name).suffix:
            candidates = [p.with_suffix(suffix) for p in candidates for suffix in ('.pdf', '.png', '.jpg')]
        path = next((p for p in candidates if p.is_file()), None)
        assert path is not None, f'Missing graphic: {name}'
        graphics[path.relative_to(root).as_posix()] = digest(path)
    assert labels and cites and graphics, 'Empty source audit target'
    sharp = re.search(r'\\begin\{align\*?\}(?:(?!\\end\{align\*?\}).)*?'
                      r'\\label\{eq:new-sharp-robust-radius\}.*?\\end\{align\*?\}', text, re.S)
    assert sharp, 'Sharp robust-radius display is absent'
    sharp_text = compact(sharp[0])
    assert r'\bar\kappa+\bar\nu(1-\beta)c' in sharp_text, 'Sharp numerator changed'
    assert not re.search(r'\\(?:bar\s*)?\\?nu\s*\^\s*\{?2\}?\s*\(1-\\beta\)c', sharp_text), \
        'Spurious second nu in robust numerator'
    assert not re.search(r'(?<!100)n_1\(1\\,\\mathrm\s*h\)\\le(?:2850|1779)', text), \
        'Physical deadline missing 100-vehicle normalization'
    report['source'] = dict(files=files, unique_labels=len(labels), reference_uses=len(refs),
                            bibliography_sources=bibfiles, cited_keys=sorted(cites),
                            bibliography_count=len(cites), graphics=graphics,
                            sharp_robust_numerator=r'\bar\kappa+\bar\nu(1-\beta)c',
                            resolved=True)
    return text, cites, labels


def check_mpc(root, text, report):
    summary = read_json(root / SUMMARY)
    old = read_json(root / COMPARISON)
    hashes = {SUMMARY: digest(root / SUMMARY)}
    for name, expected in summary['source_sha256'].items():
        path = root / 'reproducibility/results' / name
        assert digest(path) == expected, f'Changed MPC summary source: {name}'
        hashes[path.relative_to(root).as_posix()] = expected
    grid_path = root / 'reproducibility/results/mfd_incident_mpc_certified_grid_results.json'
    grid = read_json(grid_path)
    for name, expected in grid['source_sha256'].items():
        path = root / name.replace('\\', '/')
        assert digest(path) == expected, f'Changed MPC generating source: {name}'
        hashes[path.relative_to(root).as_posix()] = expected
    assert summary['complete_cases'] == len(summary['cases']) == len(grid['runs']) == 8
    assert summary['certificate_attempts'] == summary['accepted_certificates'] == 192
    assert summary['rejected_certificates'] == 0 and summary['all_nominal_commands_and_states_identical']
    changes = sum(c['command_changes_including_initial'] for c in summary['cases'])
    noops = sum(c['admitted_unchanged_first_commands'] for c in summary['cases'])
    assert (changes, noops) == (42, 150)
    protocol = grid['protocol']
    required = dict(proposal_minutes=5.0, horizon_minutes=60.0, fixed_deadline_h=1.0,
                    target_n1=28.5, queue_limit=20.0, initial_box_units=0.01,
                    residual_bound_units_per_hour=0.6, verifier_step_seconds=15.0,
                    maxiter=100, ftol=1e-8, feasibility_tolerance=1e-7, no_new_llm_calls=True)
    assert all(protocol[k] == v for k, v in required.items()), 'Frozen MPC protocol changed'
    primary = next(c for c in old['cases'] if c['amplitude'] == 0.3 and c['duration_h'] == 1.0)
    rows = dict(numeric_rows(table_block(text, 'tab:new-incident-primary')))
    # Analytic queue-blind incumbent: 16.5 normalized excess arrivals centered
    # at hour .75; initial queue 2; duration2h; regional inventory fixed at60.
    blind_queue_time = 100 * (2 * 2 + 16.5 * (2 - 0.75))
    expected_rows = {'Queue-blind incumbent': [12000 + blind_queue_time, blind_queue_time, 1850., 3000.]}
    for label in ('Nominal library', 'Certified library'):
        v = primary['library']
        expected_rows[label] = [v['tts_veh_h'], v['queue_veh_h'], v['max_queue_vehicles'], v['deadline_n1_vehicles']]
    for label in ('Nominal MPC', 'Certified MPC'):
        v = primary['mpc']
        expected_rows[label] = [v['tts_veh_h'], v['queue_veh_h'], v['max_queue_vehicles'], v['deadline_n1_vehicles']]
    assert set(rows) == set(expected_rows), f'Primary table rows: {list(rows)}'
    for label, values in expected_rows.items():
        assert len(rows[label]) == len(values) and all(rounded_equal(a, b) for a, b in zip(rows[label], values)), \
            f'Primary table differs from adaptive/analytic record: {label}'
    appendix_rows = numeric_rows(table_block(text, 'tab:new-mpc-grid'))
    assert len(appendix_rows) == 8
    for (amplitude, numbers), case, old_case in zip(appendix_rows, summary['cases'], old['cases']):
        expected = [60 * case['duration_h'], case['certified_library_tts_veh_h'],
                    case['certified_mpc_tts_veh_h'], case['tts_reduction_percent_of_library']]
        assert float(amplitude) == 100 * case['amplitude']
        assert all(rounded_equal(a, b) for a, b in zip(numbers, expected))
        assert case['nominal_mpc_tts_veh_h'] == case['certified_mpc_tts_veh_h'] == old_case['mpc']['tts_veh_h']
        assert case['certified_library_tts_veh_h'] == old_case['library']['tts_veh_h']
        assert case['successful_certificates'] == 24 and case['failed_certificates'] == 0
    robust = summary['primary_initial_plan_deadline_upper_vehicles']
    upward = Decimal(str(robust)).quantize(Decimal('0.01'), rounding=ROUND_CEILING)
    assert str(upward) == '1779.42' and '1779.42' in text
    for pattern in (r'192', r'42 exact first-command changes', r'150 plans',
                    r'100n_1\(1\\,\\mathrm h\)\\le2850', r'60 vehicles/h',
                    r'five minutes', r'15-minute', r'30-minute'):
        assert re.search(pattern, re.sub(r'\s+', ' ', text)), f'Missing count/unit/clock: {pattern}'
    for field in ('median', 'maximum'):
        value = f"{summary['total_decision_seconds'][field]:.2f}"
        assert value in text, f'Missing total runtime: {value}'
    report['mpc'] = dict(source_hashes=hashes, cases=8, certificates=192, rejected=0,
                         first_command_changes_including_initial=changes, unchanged_first_commands=noops,
                         primary_rows=expected_rows, all_grid_rows_match_adaptive_records=True,
                         robust_initial_deadline_upper=robust, robust_display_rounded_up=str(upward),
                         unchanged_protocol=required)


def check_live(root, text, report):
    directory = root / LIVE
    aggregate = read_json(directory / 'mfd_sequential_semantic_agent_aggregate.json')
    expected = dict(total_live_calls=240, context_requests_per_variant=45,
                    stateful_context_requests_correct=33, memoryless_context_requests_correct=1,
                    proposals_per_variant=120, stateful_true_objective_proposals=115,
                    memoryless_true_objective_proposals=99)
    assert all(aggregate['inclusion_audit'][k] == v for k, v in expected.items())
    stats = {v: Counter() for v in ('stateful', 'memoryless')}
    raw_hashes = {}
    count = 0
    for path in sorted(directory.glob('*.json')):
        if path.name == 'mfd_sequential_semantic_agent_aggregate.json':
            continue
        data = read_json(path)
        if not all(k in data for k in ('model', 'variant', 'decisions')):
            continue
        assert len(data['decisions']) == 8
        count += 1
        raw_hashes[path.relative_to(root).as_posix()] = digest(path)
        current = stats[data['variant']]
        for d in data['decisions']:
            call = d['tool_call']
            assert call['valid'] and call['request']['options']['temperature'] == 0.0
            parsed = json.loads(call['raw_response'])
            correct = parsed['objective'] == d['true_objective'] and parsed['horizon_minutes'] == d['true_horizon_minutes']
            assert correct == d['tool_request_correct']
            oracle = d['proposed_mode'] == d['true_objective_rollout']['lower_primary_component_mode']
            assert oracle == d['true_oracle_proposal']
            current['calls'] += 1
            current['oracle_matches'] += int(oracle)
            kind = 'context' if d['context_dependent'] else 'explicit'
            current[kind + '_calls'] += 1
            current[kind + '_correct'] += int(correct)
            current[d['supervisor_outcome']] += 1
    assert count == 30
    assert sum(s['calls'] for s in stats.values()) == 240
    assert all(s['calls'] == 120 and s['context_calls'] == 45 and s['explicit_correct'] == 75 for s in stats.values())
    assert (stats['stateful']['context_correct'], stats['memoryless']['context_correct']) == (33, 1)
    assert (stats['stateful']['oracle_matches'], stats['memoryless']['oracle_matches']) == (115, 99)
    assert [stats['stateful'][k] for k in ('executed', 'rejected', 'no_op_reselection')] == [45, 2, 73]
    for token in ('240', '33/45', '1/45', '115/120', '99/120'):
        assert token in text, f'Missing live-study denominator: {token}'
    report['live'] = dict(aggregate_sha256=digest(directory / 'mfd_sequential_semantic_agent_aggregate.json'),
                          raw_trace_files=count, raw_trace_hashes=raw_hashes,
                          independently_recounted={k: dict(v) for k, v in stats.items()})


def check_preservation(root, previous, report):
    manifest_path = previous / 'PACKAGE_MANIFEST.json'
    capture_path = root / 'reproducibility/results/streamlined_prior_live_manifest.json'
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        live = {k.replace('\\', '/'): v for k, v in manifest['sha256'].items()
                if k.replace('\\', '/').startswith('live_llm_results/')}
        assert len(live) == manifest['retained_live_record_files'] == 387
        captured = dict(source_release=str(previous), source_manifest_sha256=digest(manifest_path),
                        retained_live_record_files=387, sha256=live)
        # Capture the existing immutable release's hashes once so an extracted
        # package can repeat this audit without the author's Windows directory.
        if capture_path.is_file():
            assert read_json(capture_path) == captured, 'Captured prior-release manifest differs'
        else:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            capture_path.write_text(json.dumps(captured, indent=2) + '\n', encoding='utf-8')
        for relative, expected in live.items():
            assert digest(previous / relative) == expected, f'Prior release modified: {relative}'
        mode = 'original immutable release and captured portable manifest'
    else:
        assert capture_path.is_file(), f'Prior release and captured manifest unavailable: {manifest_path}'
        captured = read_json(capture_path)
        live = captured['sha256']
        assert len(live) == captured['retained_live_record_files'] == 387
        mode = 'captured prior-release manifest; original release directory unavailable'
    for relative, expected in live.items():
        assert digest(root / relative) == expected, f'Live archive file changed: {relative}'
    report['preserved_live_archive'] = dict(prior_release=captured['source_release'],
                                           prior_manifest_sha256=captured['source_manifest_sha256'],
                                           portable_capture_sha256=digest(capture_path),
                                           comparison_mode=mode,
                                           unchanged_files=len(live), json_files=sum(k.endswith('.json') for k in live),
                                           sha256=live)


def math_displays(text):
    pattern = re.compile(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}'
                         r'(.*?)\\end\{\1\}|\\\[(.*?)\\\]', re.S)
    displays = []
    for m in pattern.finditer(text):
        body = m[2] if m[1] else m[3]
        body = re.sub(r'\\(?:scriptsize|tiny|small|footnotesize)', '', body)
        body = re.sub(r'\\setlength\{\\arraycolsep\}\{[^}]+\}', '', body)
        displays.append(compact(body))
    return displays


def check_builds(root, text, cites, labels, report):
    ieee_files = {}
    ieee = expand(root / IEEE, root, ieee_files)
    assert r'\input{manuscript_streamlined/' not in ieee, 'IEEE modules have not been flattened'
    displays, ieee_displays = math_displays(text), math_displays(ieee)
    assert displays == ieee_displays, 'IEEE and main displayed mathematics differ; regenerate reading source'
    assert re.findall(r'\\label\{([^}]+)\}', ieee) == labels, 'IEEE and main label sequence differ'
    main_abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, re.S)[1]
    ieee_abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', ieee, re.S)[1]
    assert compact(main_abstract) == compact(ieee_abstract), 'IEEE abstract differs'
    from pypdf import PdfReader
    layouts = {}
    for name in (MAIN, IEEE):
        stem = (root / name).with_suffix('')
        log_path, pdf_path, bbl_path = (stem.with_suffix(suffix) for suffix in ('.log', '.pdf', '.bbl'))
        assert all(p.is_file() for p in (log_path, pdf_path, bbl_path)), f'Incomplete build: {name}'
        log = log_path.read_text(encoding='utf-8', errors='replace')
        prohibited = {
            'overfull': r'Overfull \\[hv]box',
            'undefined': r'(?:Reference|Citation)[^\n]*undefined|There were undefined references|Undefined control sequence',
            'duplicate_label': r'Label[^\n]*multiply defined|multiply-defined labels',
            'rerun': r'Label\(s\) may have changed|Rerun to get cross-references right',
            'latex_error': r'^! ',
        }
        findings = {key: re.findall(pattern, log, re.M) for key, pattern in prohibited.items()}
        assert not any(findings.values()), f'Build diagnostics in {name}: {findings}'
        bbl = bbl_path.read_text(encoding='utf-8', errors='replace')
        used = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', bbl)
        assert len(used) == len(set(used)) and set(used) == cites, f'Bibliography differs from actual citations: {name}'
        reader = PdfReader(pdf_path)
        assert reader.pages, f'Empty PDF: {name}'
        extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
        assert '??' not in extracted, f'Unresolved printed reference marker: {name}'
        assert not re.search(r'Appendix\s+Appendix', extracted), f'Duplicated appendix prefix: {name}'
        assert '1779.42' in extracted.replace(',', '') or '1779.42' in compact(extracted), \
            f'Initial proof bound absent in PDF: {name}'
        layouts[name] = dict(pages=len(reader.pages), pdf_sha256=digest(pdf_path),
                              log_sha256=digest(log_path), bibliography_count=len(used),
                              bibliography_sha256=digest(bbl_path), clean_diagnostics=True,
                              extracted_text_has_no_unresolved_marker=True)
    report['generated_builds'] = dict(displayed_mathematics_count=len(displays),
                                      ieee_math_and_labels_equal=True,
                                      ieee_source_sha256=ieee_files[IEEE], layouts=layouts,
                                      visual_qa_scope='PDF text/build checks only; page-image review is a separate required audit')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source', default=MAIN)
    parser.add_argument('--source-only', action='store_true', help='Defer generated IEEE/PDF/log checks during rebuilds')
    parser.add_argument('--previous-release', type=Path, default=OLD_RELEASE)
    parser.add_argument('--output', type=Path,
                        default=Path('reproducibility/results/streamlined_manuscript_audit_results.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    # Protect existing scientific evidence even if an accidental output argument
    # targets a similarly named file.
    assert 'streamlined' in output.name and 'audit' in output.name and output.suffix == '.json', \
        'Output must be a separately named streamlined audit JSON'
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), root=str(root),
                  source_only=args.source_only, no_simulations_or_live_calls=True,
                  audit_script_sha256=digest(Path(__file__)))
    checks = []
    try:
        text, cites, labels = check_source(root, args.source, report)
        checks.append('source')
        check_mpc(root, text, report)
        checks.append('mpc_tables_counts_and_provenance')
        check_live(root, text, report)
        checks.append('independent_live_call_recount')
        check_preservation(root, args.previous_release, report)
        checks.append('prior_live_archive_preserved')
        if args.source_only:
            report['generated_builds'] = dict(status='deferred by --source-only')
        else:
            check_builds(root, text, cites, labels, report)
            checks.append('ieee_mathematics_pdf_logs_and_bibliographies')
        report['pass'] = True
    except Exception as exc:
        report['pass'] = False
        report['failure'] = f'{type(exc).__name__}: {exc}'
    report['completed_checks'] = checks
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(passed=report['pass'], checks=checks, output=str(output),
                          failure=report.get('failure')), indent=2))
    if not report['pass']:
        sys.exit(1)


if __name__ == '__main__':
    main()
