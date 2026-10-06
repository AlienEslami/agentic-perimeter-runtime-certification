#!/usr/bin/env python3
"""Cross-check the revised manuscript, both layouts, and deadline records."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'reproducibility/results'


def read(name): return json.loads((RESULTS/name).read_text(encoding='utf-8'))


def labeled_equations(text):
    found={}
    for match in re.finditer(r'\\begin\{(equation|align|gather)\}(.*?)\\end\{\1\}',text,re.S):
        body=match.group(0)
        for label in re.findall(r'\\label\{([^}]+)\}',body): found[label]=re.sub(r'\s+','',body)
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-source',type=Path)
    args=parser.parse_args()
    source=(ROOT/'FOUNDATION_MFD_PARTB_ELSEVIER.tex').read_text(encoding='utf-8')
    report={'passed':False,'checks':[]}
    study=read('mfd_deadline_certificate_results.json')
    assert study['summary']=={'cases':24,'candidate_schedules':48,'verified_safe_candidate_schedules':30,'case_successes_by_target':[24,24,24]}
    for case in study['cases']:
        for candidate in case['candidates']:
            if candidate['verified_continuous_tube']:
                assert not candidate['failures']
                assert candidate['maximum_predictor_outer_tube_bound']<=candidate['admission_level']
                assert all(x['constraint_count']==22 and x['minimum_affine_constraint_slack']>0 for x in candidate['geometric_constraint_audits'])
                terminal=candidate['records'][-1]['absolute_imbalance_upper_vehicles']
                assert candidate['passes_targets']==[terminal<=target for target in case['targets_vehicles']]
    primary=study['primary_case'];target=primary['targets_vehicles'][2]
    hold,switch=primary['candidates']
    assert hold['verified_continuous_tube'] and switch['verified_continuous_tube']
    assert hold['records'][-1]['absolute_imbalance_upper_vehicles']>target
    assert switch['records'][-1]['absolute_imbalance_upper_vehicles']<target
    fixed=study['fixed_adverse_hold_certificate']
    assert fixed['verified_continuous_tube']
    assert fixed['records'][-1]['absolute_imbalance_lower_vehicles']>target
    assert fixed['records'][-1]['absolute_imbalance_lower_vehicles']>29.64
    keys=['young_support_upper_vehicles','radius_support_upper_vehicles','ball_absolute_imbalance_upper_vehicles','absolute_imbalance_upper_vehicles']
    rows=['Quadratic flow comparison','Direct radius comparison','Nonlinear error-ball projection','Output-sensitive nonlinear enclosure']
    for k,label in zip(keys,rows):
        match=re.search(re.escape(label)+r'\s*&\s*([\d.]+)\s*&\s*([\d.]+)',source)
        assert match,label
        for displayed,candidate in zip(map(float,match.groups()),(hold,switch)):
            actual=candidate['scalar_bounds'][k] if 'support' in k else candidate['records'][-1][k]
            assert actual<=displayed<actual+.011,(label,actual,displayed)
    report['checks'].append('24 cases, all 48 candidate dispositions, geometric safety, terminal targets, fixed-disturbance miss, upward table rounding')
    bridge=read('mfd_deadline_sos_bridge_results.json')
    assert bridge['all_fresh_sos_witnesses_interval_verified']
    assert all(m['effective_rate_lower_per_hour']>1.55 and m['deadline_level_inside_verified_coordinate_domain'] for m in bridge['modes'])
    report['checks'].append('fresh SOS witnesses and exact-binary coefficient bridge')
    current=labeled_equations(source)
    report['equation_label_count']=len(current)
    if args.previous_source:
        previous=labeled_equations(args.previous_source.read_text(encoding='utf-8'))
        assert all(key in current and current[key]==value for key,value in previous.items())
        report['unchanged_previous_equation_labels']=len(previous)
        report['new_equation_labels']=sorted(set(current)-set(previous))
    abstract=source.split('\\begin{abstract}',1)[1].split('\\end{abstract}',1)[0]
    assert len(abstract.split())<=250
    report['abstract_word_count']=len(abstract.split())
    names=['FOUNDATION_MFD_PARTB_ELSEVIER','FOUNDATION_MFD_PARTB_IEEE_READING_COPY']
    labels=[]
    report['pdf_pages']={}
    for stem in names:
        log=(ROOT/(stem+'.log')).read_text(encoding='utf-8',errors='replace')
        for forbidden in ('Overfull \\hbox','Overfull \\vbox','undefined references','There were undefined citations','Rerun to get cross-references right'):
            assert forbidden not in log,(stem,forbidden)
        bbl=(ROOT/(stem+'.bbl')).read_text(encoding='utf-8')
        assert bbl.count('\\bibitem')==57
        reader=PdfReader(ROOT/(stem+'.pdf'))
        report['pdf_pages'][stem]=len(reader.pages)
        text=' '.join(' '.join(page.extract_text() or '' for page in reader.pages).split())
        for phrase in ('Deadline-preserving','Output-sensitive','28.23','10.44','0.5874'):
            assert phrase in text,(stem,phrase)
        aux=(ROOT/(stem+'.aux')).read_text(encoding='utf-8')
        labels.append(dict(re.findall(r'\\newlabel\{(eq:[^}]+)\}\{\{([^}]+)\}',aux)))
    assert labels[0]==labels[1]
    report['checks'].append('both PDFs, complete references, equation numbering and rendered-text presence')
    report['passed']=True
    report['source_sha256']=hashlib.sha256(source.encode()).hexdigest()
    (RESULTS/'mfd_deadline_manuscript_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
