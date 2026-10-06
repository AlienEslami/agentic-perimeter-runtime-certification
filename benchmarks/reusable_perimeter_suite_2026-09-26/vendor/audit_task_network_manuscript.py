"""Read-only integrity audit for the task-state/network manuscript revision.

Only this program's named JSON output is written. No scientific computation,
LaTeX build, bibliography regeneration, or old audit is run. --source-only
checks the authoritative source while deferring the generated reading copy,
compiled files, and unfinished experimental records. A successful partial
audit has status SOURCE_ONLY_PASS and complete=false, never final PASS.

PDF text checks complement, and do not replace, rendered-page visual review.
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

from audit_streamlined_manuscript import digest, read_json, math_displays, table_block


ROOT = Path(__file__).resolve().parents[1]
MAIN = 'FOUNDATION_MFD_PARTB_ELSEVIER.tex'
IEEE = 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY.tex'
EXTERNAL_INPUTS = {'glyphtounicode'}
NETWORK = 'reproducibility/results/mfd_multiregion_transfer_results.json'
AUDITS = {
    'network': 'reproducibility/results/mfd_multiregion_transfer_independent_audit_results.json',
    'task_runtime': 'reproducibility/results/mfd_task_state_runtime_v2_independent_audit_results.json',
    'task_semantics': 'reproducibility/results/task_state_agent_v2_independent_audit.json',
}


def uncomment(text):
    # An odd run of backslashes escapes %, while an even run leaves a comment.
    return re.sub(r'(?<!\\)((?:\\\\)*)%[^\n]*', lambda m: m[1], text)


def compact(text):
    return re.sub(r'\s+', '', text)


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def expand(path, root, files, active=(), import_base=None):
    """Expand local input/include/import/subimport, detecting cycles and drift."""
    path = path.resolve()
    ensure(path.is_relative_to(root), f'Outside-project input: {path}')
    ensure(path not in active, f'Cyclic input: {path}')
    ensure(path.is_file(), f'Missing source input: {path}')
    key = path.relative_to(root).as_posix()
    before = digest(path)
    raw = path.read_text(encoding='utf-8-sig')
    ensure(digest(path) == before, f'Source changed while reading: {key}')
    ensure(key not in files or files[key] == before, f'Source changed during expansion: {key}')
    files[key] = before
    text = uncomment(raw)
    pattern = r'\\(?:(input|include)\s*\{([^}]+)\}|(import|subimport)\*?\s*\{([^}]+)\}\s*\{([^}]+)\})'

    def load(match):
        if match[1]:
            name = match[2]
            if name in EXTERNAL_INPUTS:
                return match[0]
            bases = [import_base, root, path.parent]
            next_import = import_base
        else:
            name = match[5]
            base = (import_base or path.parent) if match[3] == 'subimport' else root
            next_import = (base / match[4]).resolve()
            bases = [next_import]
        rel = Path(name if Path(name).suffix else name + '.tex')
        target = next((base / rel for base in bases if base and (base / rel).is_file()), None)
        ensure(target is not None, f'Unresolved input {name!r} in {key}')
        return expand(target, root, files, (*active, path), next_import)

    return re.sub(pattern, load, text)


def resolve_asset(root, name, directories, extensions):
    ensure('#' not in name and '\\' not in name, f'Dynamic asset path needs explicit audit support: {name}')
    rel = Path(name)
    attempts = [root / rel] + [root / d / rel for d in directories]
    expanded = [p if p.suffix else p.with_suffix(ext) for p in attempts
                for ext in ([''] if p.suffix else extensions)]
    target = next((p.resolve() for p in expanded if p.is_file()), None)
    ensure(target is not None, f'Unresolved asset: {name}')
    ensure(target.is_relative_to(root), f'Asset outside project: {target}')
    return target


def cite_sequence(text):
    return [tuple(k.strip() for k in m.split(',') if '#' not in k)
            for m in re.findall(r'\\(?:cite\w*|nocite)\*?(?:\[[^]]*\])*\{([^}]+)\}', text)]


def source_check(root, source):
    files = {}
    text = expand(root / source, root, files)
    labels = re.findall(r'\\label\s*\{([^}]+)\}', text)
    duplicates = [k for k, v in Counter(labels).items() if v > 1]
    ensure(not duplicates, f'Duplicate labels: {duplicates}')
    refs = [k.strip() for item in re.findall(
        r'\\(?:ref|eqref|pageref|appref|autoref|[cC]ref|nameref)\*?\s*\{([^}]+)\}', text)
        for k in item.split(',') if '#' not in k]
    ensure(not set(refs) - set(labels), f'Unresolved references: {sorted(set(refs)-set(labels))}')
    directories = ['', 'figures']
    for block in re.findall(r'\\graphicspath\s*\{((?:\{[^}]*\})+)\}', text):
        directories.extend(re.findall(r'\{([^}]*)\}', block))
    graphics = {}
    for name in re.findall(r'\\includegraphics\*?(?:\[[^]]*\])?\s*\{([^}]+)\}', text):
        asset = resolve_asset(root, name, directories, ['.pdf', '.png', '.jpg', '.jpeg', '.eps'])
        graphics[asset.relative_to(root).as_posix()] = digest(asset)
    bibfiles, keys = {}, []
    for names in re.findall(r'\\bibliography\s*\{([^}]+)\}|\\addbibresource(?:\[[^]]*\])?\s*\{([^}]+)\}', text):
        for name in (names[0] or names[1]).split(','):
            asset = resolve_asset(root, name.strip(), [''], ['.bib'])
            bibfiles[asset.relative_to(root).as_posix()] = digest(asset)
            keys.extend(re.findall(r'@(?!(?:comment|string|preamble)\b)\w+\s*\{\s*([^,\s]+)\s*,',
                                   uncomment(asset.read_text(encoding='utf-8-sig')), flags=re.I))
    ensure(len(keys) == len(set(keys)), 'Duplicate BibTeX keys')
    citations = {key for group in cite_sequence(text) for key in group if key != '*'}
    ensure(not citations - set(keys), f'Undefined citations: {sorted(citations-set(keys))}')
    body = text.split(r'\begin{document}', 1)[1]
    body = re.split(r'\\appendi(?:x|ces)\b', body, maxsplit=1)
    counts = {name: len(re.findall(r'\\begin\{' + name + r'\*?\}', text))
              for name in ['theorem', 'proposition', 'corollary', 'assumption', 'definition',
                           'remark', 'figure', 'table']}
    counts.update(labels=len(labels), reference_uses=len(refs), distinct_citations=len(citations),
                  bibliography_entries=len(keys), displayed_equations=len(math_displays(text)),
                  main_sections=len(re.findall(r'\\section\*?\{', body[0])),
                  appendix_sections=len(re.findall(r'\\section\*?\{', body[1])) if len(body) > 1 else 0)
    return text, dict(source=source, source_files=files, graphic_files=graphics,
                      bibliography_files=bibfiles, installed_inputs=sorted(EXTERNAL_INPUTS),
                      labels=labels, citation_keys=sorted(citations), counts=counts,
                      expanded_source_sha256=sha_text(text))


def body_content(text):
    if r'\end{frontmatter}' in text:
        text = text.split(r'\end{frontmatter}', 1)[1]
    else:
        text = text.split(r'\end{IEEEkeywords}', 1)[1]
    text = text.split(r'\end{document}', 1)[0]
    text = re.sub(r'\\(?:FloatBarrier|clearpage)\b', '', text)
    text = text.replace(r'\appendices', r'\appendix')
    text = text.replace(r'\numberwithin{equation}{section}', '')
    text = re.sub(r'\\bibliographystyle\{(?:elsarticle-num|IEEEtranN)\}', r'\\bibliographystyle{SAME}', text)
    text = re.sub(r'\\(begin|end)\{(table|figure)\*?\}(?:\[[^]]*\])?', r'\\\1{\2}', text)
    text = text.replace(r'\tiny\setlength{\arraycolsep}{1.7pt}',
                        r'\scriptsize\setlength{\arraycolsep}{2.5pt}')
    return compact(text)


def source_parity(main_text, ieee_text):
    a, b = body_content(main_text), body_content(ieee_text)
    if a != b:
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        raise AssertionError(f'Reading-copy content mismatch at canonical character {first}: '
                             f'Elsevier={a[max(0,first-100):first+220]!r}; '
                             f'IEEE={b[max(0,first-100):first+220]!r}')
    ensure(math_displays(main_text) == math_displays(ieee_text), 'Display equations differ between layouts')
    ensure(re.findall(r'\\label\{([^}]+)\}', main_text) == re.findall(r'\\label\{([^}]+)\}', ieee_text),
           'Label sequences differ between layouts')
    ensure(cite_sequence(main_text) == cite_sequence(ieee_text), 'Citation sequences differ between layouts')
    for field in ['title', 'abstract']:
        pattern = r'\\title\{([^}]+)\}' if field == 'title' else r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
        x, y = re.search(pattern, main_text, re.S), re.search(pattern, ieee_text, re.S)
        ensure(x and y and compact(x[1]) == compact(y[1]), f'{field} differs between layouts')
    return dict(passed=True, canonical_body_sha256=sha_text(a), canonical_characters=len(a),
                allowed_layout_changes=['document/front matter and title-author formatting',
                                        'float stars/placement and FloatBarrier/clearpage',
                                        'appendix command and IEEE equation numbering',
                                        'bibliography style',
                                        'specified 2.5pt/scriptsize to 1.7pt/tiny array formatting'],
                exact_display_math=True, exact_label_order=True, exact_citation_order=True,
                exact_title_and_abstract=True, full_canonical_body_equal=True)


def table_rows(block):
    rows = []
    for row in block.split('\\\\'):
        if '&' not in row:
            continue
        cells = [re.sub(r'\\(?:midrule|toprule|bottomrule)\b', '', x).strip() for x in row.split('&')]
        rows.append(cells)
    return rows


def ready_audit(root, name, relative, partial, deferred):
    path = root / relative
    if not path.is_file():
        ensure(partial, f'Missing independent {name} audit: {relative}')
        deferred.append(f'{name}: independent audit not yet present')
        return None
    data = read_json(path)
    ensure(data.get('passed') is True or data.get('pass') is True or data.get('status') == 'PASS',
           f'Independent audit is not passing: {relative}')
    if data.get('complete') is False:
        ensure(partial, f'Independent audit incomplete: {relative}')
        deferred.append(f'{name}: independent audit incomplete')
        return None
    return data


def network_table(root, text, audit):
    data = read_json(root / NETWORK)
    ensure(data.get('complete') is True, 'Network experiment incomplete')
    ensure(digest(root/NETWORK) == audit['summary_sha256'], 'Network summary differs from independent audit')
    block = table_block(text, 'tab:network-transfer')
    rows = [row for row in table_rows(block) if re.fullmatch(r'\d+/\d+', row[0])]
    index = {(c['regions'], c['method'], c['width_vehicles'], c['residual_veh_h']): c for c in data['cases']}
    ensure(len(index) == len(data['cases']) == 2*len(rows), 'Network table does not cover exactly all matched cases')
    checks = []
    for row in rows:
        ensure(len(row) == 5, f'Unexpected network table shape: {row}')
        regions, n = map(int, row[0].split('/'))
        ensure(n == regions*regions + regions, f'Wrong state dimension in network row: {row}')
        d, w = int(row[1]), int(row[2])
        for method, printed in zip(['class', 'inventory'], row[3:]):
            case = index[(regions, method, d, w)]
            ensure(digest(root / case['proof_file']) == case['proof_sha256'], f'Network proof hash changed: {case["case_id"]}')
            if case['accepted']:
                value = max(x['upper_vehicles'] for x in case['terminal_service_rows'])
                expected = str(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_CEILING))
                ensure(printed == expected, f'Network bound {case["case_id"]}: printed {printed}, upward rounded {expected}')
            else:
                expected = '---'
                ensure(printed == expected, f'Unresolved network plan shown as accepted: {case["case_id"]}')
            checks.append(dict(case_id=case['case_id'], printed=printed, accepted=case['accepted']))
    ensure(len(index) == audit['proof_count'], 'Network proof count differs from independent audit')
    ensure(sum(c['steps'] for c in data['cases']) == audit['interval_step_count'],
           'Network saved interval count differs from independent audit')
    return dict(passed=True, results_sha256=digest(root / NETWORK), table_label='tab:network-transfer',
                checked_cells=checks, attempts=len(index), accepted=sum(c['accepted'] for c in data['cases']),
                recorded_steps=sum(c['steps'] for c in data['cases']),
                twelve_state_success_seconds=[c['wall_seconds'] for c in data['cases'] if c['regions']==3 and c['accepted']])


def evidence_check(root, text, partial, deferred):
    audits, metadata = {}, {}
    for name, relative in AUDITS.items():
        audits[name] = ready_audit(root, name, relative, partial, deferred)
        if audits[name] is not None:
            metadata[name] = dict(path=relative, sha256=digest(root / relative), passed=True)
    report = dict(independent_audits=metadata)
    if audits['network']:
        report['network_table'] = network_table(root, text, audits['network'])
    if audits['task_runtime']:
        cases = audits['task_runtime']['cases']
        bindings = []
        for case in cases:
            path = root / case['result_path']
            ensure(digest(path) == case['result_sha256'], f'Task result hash differs from independent audit: {path}')
            bindings.append(dict(case_id=case['case_id'], path=case['result_path'], sha256=case['result_sha256']))
        report['task_runtime_bindings'] = bindings
    task_source = root / 'manuscript_streamlined/task_state_evidence.tex'
    if not task_source.is_file() or not audits['task_semantics'] or not audits['task_runtime']:
        ensure(partial, 'Task-state table/evidence is not complete')
        deferred.append('task-state numerical table: awaiting final source and both independent audits')
    else:
        report['task_table'] = task_table(root, text, audits['task_runtime'], audits['task_semantics'])
    return report


def task_table(root, text, runtime, semantics):
    block = table_block(text, 'tab:task-state-live')
    rows = {row[0]: row[1:] for row in table_rows(block) if len(row) == 5 and row[1].isdigit()}
    names = {'Prose memory, held-out': 'heldout_prose',
             'Typed task state, held-out': 'heldout_task_state',
             'Typed state, corrupted view': 'poison_task_state'}
    ensure(set(rows) == set(names), f'Unexpected live table rows: {list(rows)}')
    checked = []
    for label, key in names.items():
        record = semantics['aggregate'][key]
        expected = [str(record['total']), f'{record["valid"]}/{record["total"]}',
                    f'{record["exact_request_correct"]}/{record["total"]}',
                    f'{record["contextual_correct"]}/{record["contextual_total"]}']
        ensure(rows[label] == expected, f'Wrong live fractions for {label}: {rows[label]} vs {expected}')
        checked.append(dict(row=label, values=expected, semantic_audit_group=key))
    source = uncomment((root/'manuscript_streamlined/task_state_evidence.tex').read_text(encoding='utf-8'))
    cases = {case['case_id']: case for case in runtime['cases']}
    # Main-text performance values come from the tighter independent DOP853
    # replay, not the coarser online trapezoid metrics. They are diagnostics.
    values = []
    for name in ['heldout_p0_original', 'heldout_p1_original', 'heldout_p0_structured',
                 'heldout_p1_structured', 'network', 'oracle']:
        ensure(name in cases, f'Missing audited task runtime case: {name}')
        value = cases[name]['independent_replay']['tts_veh_h']
        printed = f'{value:.2f}'
        ensure(re.search(r'(?<![\d.])' + re.escape(printed) + r'(?!\d)', source),
               f'Independent TTS {printed} for {name} is not reflected in task-state evidence')
        values.append(dict(case_id=name, metric='independently_replayed_tts_veh_h', value=value, printed=printed))
    overflow = cases['command_pressure_nominal']['independent_replay']['first_storage_overflow_minutes']
    ensure(overflow is not None and f'{overflow:.2f}' in source, 'Storage overflow time does not match independent replay')
    values.append(dict(case_id='command_pressure_nominal', metric='first_storage_overflow_minutes',
                       value=overflow, printed=f'{overflow:.2f}'))
    typed = [cases[f'heldout_p{p}_structured']['independent_replay'] for p in [0, 1]]
    prose = [cases[f'heldout_p{p}_original']['independent_replay'] for p in [0, 1]]
    ensure(all(t['oracle_task_cost_veh_h'] > p['oracle_task_cost_veh_h'] for t,p in zip(typed,prose)),
           'The stated worse typed-task weighted costs do not match the matched records')
    total = semantics['attempted_calls'] + semantics['preserved_development_calls']
    ensure(f'{total}-call study' in source, f'Incorrect total study call count; expected {total}')
    for version, menu in runtime['protocol_menu'].items():
        ensure(menu['complete'] and all(c['count'] == c['expected'] for c in menu['categories']),
               f'Incomplete runtime menu version {version}')
    traffic = traffic_table(text, runtime, cases)
    return dict(passed=True, table_label='tab:task-state-live', rows=checked,
                live_calls=semantics['attempted_calls'], preserved_development_calls=semantics['preserved_development_calls'],
                total_study_calls=total, runtime_cases=len(cases), diagnostic_values=values,
                matched_traffic_table=traffic,
                paired_typed_weighted_cost_worse=True,
                metrics_scope='Independent numerical replay, not a new reachability proof or an optimality comparison')


def traffic_table(text, runtime, cases):
    block = table_block(text, 'tab:task-state-traffic')
    rows = {row[0]: [x.replace('$', '') for x in row[1:]] for row in table_rows(block)
            if len(row) == 6 and re.fullmatch(r'\d+\.\d+', row[1])}
    names = {'Prose, phrase 1': 'heldout_p0_original',
             'Typed, phrase 1': 'heldout_p0_structured',
             'Prose, phrase 2': 'heldout_p1_original',
             'Typed, phrase 2': 'heldout_p1_structured',
             'Typed, corrupted view': 'poison_structured'}
    ensure(set(rows) == set(names), f'Unexpected matched traffic rows: {list(rows)}')
    pairs = {p['certified_case']: p for p in runtime['matched_nominal_comparisons']}
    ensure(set(pairs) == set(names.values()), 'Matched traffic table omits or adds a saved comparison')
    checks = []
    for label, key in names.items():
        pair = pairs[key]
        ensure(pair['exact_semantic_request_match'], f'Unmatched request comparator for {key}')
        cert, nominal = cases[key], cases[pair['nominal_case']]
        a, b = cert['independent_replay'], nominal['independent_replay']
        percentage = 100*(a['tts_veh_h']/b['tts_veh_h']-1)
        cost_difference = a['oracle_task_cost_veh_h']-b['oracle_task_cost_veh_h']
        ensure(abs(percentage-pair['tts_premium_percent']) < 1e-10, f'TTS percent audit discrepancy: {key}')
        ensure(abs(cost_difference-pair['oracle_task_cost_difference_veh_h']) < 1e-8,
               f'Common intended-task cost discrepancy: {key}')
        expected = [f'{a["tts_veh_h"]:.2f}', f'{b["tts_veh_h"]:.2f}', f'{percentage:.2f}',
                    f'{a["maximum_queue_vehicles"]:.2f}', str(cert['retained_count'])]
        ensure(rows[label] == expected, f'Matched traffic row {label}: {rows[label]} vs {expected}')
        ensure(percentage < 0 and cost_difference < 0, f'Stated paired reductions do not hold: {key}')
        checks.append(dict(row=label, certified_case=key, nominal_case=pair['nominal_case'],
                           printed=expected, percentage_change=percentage,
                           intended_task_cost_difference_veh_h=cost_difference))
    return dict(passed=True, table_label='tab:task-state-traffic', rows=checks,
                reduction_range_percent=[min(-p['tts_premium_percent'] for p in pairs.values()),
                                         max(-p['tts_premium_percent'] for p in pairs.values())],
                all_common_intended_task_cost_differences_negative=True)


def aux_labels(text):
    # IEEE subsection numbers contain nested groups such as \\mbox {II-C}.
    # Parse balanced TeX groups rather than discarding those existing labels.
    def group(start):
        ensure(text[start] == '{', 'Expected an aux group')
        level=1; end=start+1
        while end<len(text) and level:
            if text[end]=='{': level+=1
            elif text[end]=='}': level-=1
            end+=1
        ensure(level==0, 'Unclosed aux label group')
        return text[start+1:end-1],end
    labels={}
    for match in re.finditer(r'\\newlabel\{',text):
        key,pos=group(match.end()-1)
        ensure(text[pos:pos+2]=='{{', 'Malformed aux label fields')
        number,pos=group(pos+1)
        page,pos=group(pos)
        ensure(page.isdigit(), 'Nonnumeric aux page')
        labels[key]={'number':number,'page':int(page)}
    return labels


def build_check(root, source, text, source_report, maximum_overfull):
    from pypdf import PdfReader
    base = root / Path(source).stem
    paths = {ext: base.with_suffix('.'+ext) for ext in ['pdf', 'log', 'aux', 'bbl']}
    for ext, path in paths.items():
        ensure(path.is_file(), f'Missing final build {ext}: {path}')
    logs = paths['log'].read_text(encoding='utf-8', errors='replace')
    bad = re.findall(r'(?m)^!.*|[^\n]*(?:(?:Reference|Citation)[^\n]*undefined|There were undefined references|'
                     r'Undefined control sequence|multiply.defined labels|Label[^\n]*multiply defined|'
                     r'Label\(s\) may have changed|Rerun to get cross-references right)[^\n]*', logs)
    ensure(not bad, f'Unresolved/fatal/rerun build messages in {source}: {bad}')
    overfull = [dict(kind=kind, points=float(value), message=message) for message, kind, value in re.findall(
        r'(Overfull \\([hv])box \(([0-9.]+)pt [^\n]*)', logs)]
    ensure(not any(row['points'] > maximum_overfull for row in overfull),
           f'Overfull boxes exceed {maximum_overfull}pt in {source}: {overfull}')
    deps = {**source_report['source_files'], **source_report['bibliography_files'], **source_report['graphic_files']}
    stale = [name for name in deps if (root/name).stat().st_mtime > paths['pdf'].stat().st_mtime + 2]
    ensure(not stale, f'PDF predates current inputs: {stale}')
    aux = paths['aux'].read_text(encoding='utf-8', errors='replace')
    labels = aux_labels(aux)
    missing_labels = set(source_report['labels']) - set(labels)
    ensure(not missing_labels, f'Labels missing in final aux: {sorted(missing_labels)}')
    bib = paths['bbl'].read_text(encoding='utf-8', errors='replace')
    bibkeys = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', bib)
    ensure(len(bibkeys) == len(set(bibkeys)), f'Duplicate compiled bibliography entries in {source}')
    ensure(set(bibkeys) == set(source_report['citation_keys']), f'Compiled bibliography differs from cited keys in {source}')
    reader = PdfReader(paths['pdf'])
    page_text = [page.extract_text() or '' for page in reader.pages]
    bad_pages = [i+1 for i, page in enumerate(page_text) if re.search(r'\?\?|Appendix\s+Appendix', page)]
    ensure(not bad_pages, f'Unresolved PDF placeholders on pages: {bad_pages}')
    references_pages = [i+1 for i, page in enumerate(page_text)
                        if re.search(r'(?im)^\s*references\s*$', page)]
    ensure(len(references_pages) == 1, f'Expected one references heading, found pages {references_pages}')
    body, appendices = re.split(r'\\appendi(?:x|ces)\b', text, maxsplit=1)
    appendix_keys = re.findall(r'\\section\*?\{[^}]+\}\s*\\label\{([^}]+)\}', appendices)
    appendix_pages = [labels[key]['page'] for key in appendix_keys]
    ensure(appendix_pages, 'Appendix headings/labels were not located')
    first_appendix = min(appendix_pages)
    refs_page = references_pages[0]
    ensure(1 <= first_appendix <= refs_page <= len(reader.pages), 'Invalid appendix/reference page ordering')
    dimensions = Counter((round(float(p.mediabox.width), 3), round(float(p.mediabox.height), 3)) for p in reader.pages)
    return dict(passed=True, files={ext: {'path': p.relative_to(root).as_posix(), 'sha256': digest(p)} for ext,p in paths.items()},
                total_pages=len(reader.pages), body_pages_before_appendices=first_appendix-1,
                first_appendix_page=first_appendix, appendix_count=len(appendix_keys),
                appendix_heading_pages=dict(zip(appendix_keys, appendix_pages)),
                appendix_page_span_including_shared_reference_page=refs_page-first_appendix+1,
                references_first_page=refs_page, references_page_span=len(reader.pages)-refs_page+1,
                bibliography_entry_count=len(bibkeys), source_counts=source_report['counts'],
                page_sizes_points=[{'width': wh[0], 'height': wh[1], 'pages': n} for wh,n in dimensions.items()],
                overfull_boxes=overfull, overfull_threshold_points=maximum_overfull,
                undefined_messages=[], source_freshness_checked=True,
                visual_layout_review='Not performed by this text audit; requires rendered-page review')


def run(root, source_only=False, maximum_overfull=0.5):
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_only=source_only,
                  audit_source_sha256=digest(Path(__file__)), helper_source_sha256=digest(Path(__file__).with_name('audit_streamlined_manuscript.py')),
                  passed=False, complete=False, status='FAIL', errors=[], deferred=[],
                  scope='Source/build integrity and saved evidence cross-checks; no scientific reruns or visual-layout proof')
    try:
        main_text, main = source_check(root, MAIN)
        report['authoritative_source'] = main
        report['evidence'] = evidence_check(root, main_text, source_only, report['deferred'])
        if source_only:
            report['deferred'].append('generated IEEE source parity and final PDF/log/aux/bbl checks')
        else:
            ieee_text, ieee = source_check(root, IEEE)
            report['reading_copy_source'] = ieee
            report['layout_parity'] = source_parity(main_text, ieee_text)
            report['builds'] = {MAIN: build_check(root, MAIN, main_text, main, maximum_overfull),
                                IEEE: build_check(root, IEEE, ieee_text, ieee, maximum_overfull)}
        # Detect concurrent edits rather than silently mixing manuscript versions.
        for section in ['authoritative_source', 'reading_copy_source']:
            for category in ['source_files', 'bibliography_files', 'graphic_files']:
                for path, expected in report.get(section, {}).get(category, {}).items():
                    ensure(digest(root/path) == expected, f'Manuscript input changed during audit: {path}')
        for audit in report['evidence']['independent_audits'].values():
            ensure(digest(root/audit['path']) == audit['sha256'],
                   f'Independent evidence audit changed during manuscript audit: {audit["path"]}')
        report.update(passed=True, complete=not source_only,
                      status='SOURCE_ONLY_PASS' if source_only else 'PASS')
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--max-overfull-pt', type=float, default=.5)
    parser.add_argument('--output', type=Path, default=Path('reproducibility/results/task_network_manuscript_final_audit_results.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root/args.output
    ensure(output.resolve().is_relative_to(root), 'Audit report must remain inside the project')
    ensure(output.name.startswith('task_network_manuscript_') and output.suffix == '.json',
           'Only a distinct task_network_manuscript_*.json audit report may be written')
    report = run(root, args.source_only, args.max_overfull_pt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'complete': report['complete'],
                      'errors': report['errors'], 'deferred': report['deferred'], 'output': str(output)}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
