"""Stage and hash-verify the complete task-state/network paper release.

Existing releases are never overwritten. Prepare, build/audit the staged
sources, then archive. The two publication copies come from staged PDFs.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK'
STEMS = ('FOUNDATION_MFD_PARTB_ELSEVIER', 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY')


def native(path):
    p = str(path.resolve())
    return '\\\\?\\' + p if os.name == 'nt' else p


def data(path):
    with open(native(path), 'rb') as stream:
        return stream.read()


def sha(path):
    return hashlib.sha256(data(path)).hexdigest()


def eligible(path):
    return path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.aux','.out','.spl','.blg')


def sources():
    files = [ROOT/(stem+ext) for stem in STEMS for ext in ('.tex','.pdf','.bbl','.log')]
    files += [ROOT/name for name in ('FOUNDATION_REFERENCES.bib','IEEEtran.cls','IEEEtranN.bst',
                                    'requirements.txt','requirements-synthesis.txt','mfd_architecture_error.png')]
    files += list(ROOT.glob('*.md'))
    files += list((ROOT/'manuscript_streamlined').glob('*.tex'))
    files += list((ROOT/'Archived').glob('*.tex'))
    for folder in ('reproducibility','live_llm_results','results','figures'):
        files += [p for p in (ROOT/folder).rglob('*') if eligible(p)]
    prior = json.loads((ROOT/'reproducibility/results/online_structure_prior_payload_manifest.json').read_text())
    files += [ROOT/name for name in prior['sha256']]
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare','archive'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    package = out/'package'/NAME
    if args.stage == 'prepare':
        package.mkdir(parents=True, exist_ok=False)
        files = sources()
        for source in files:
            assert source.is_file(), source
            target = package/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(native(source),native(target))
            assert sha(source) == sha(target)
        print(json.dumps(dict(package=str(package), files=len(files)), indent=2))
        return
    assert package.is_dir()
    required = ('task_network_prior_preservation_audit_results.json',
                'mfd_task_state_runtime_v1_independent_audit_results.json',
                'mfd_task_state_runtime_v2_independent_audit_results.json',
                'task_state_agent_v2_independent_audit.json',
                'mfd_multiregion_transfer_independent_audit_results.json',
                'task_network_manuscript_final_audit_results.json')
    for filename in required:
        audit = json.loads(data(package/'reproducibility/results'/filename))
        assert audit.get('passed', audit.get('pass')) is True or audit.get('status') == 'PASS', filename
        if 'complete' in audit:
            assert audit['complete'] is True, filename
    for path in sources():
        # Every final page has a visually reviewed counterpart from the root build.
        # Staged LaTeX rebuilds may change PDF IDs/log paths; source and science cannot drift.
        relative = path.relative_to(ROOT)
        generated = (path.parent == ROOT and path.stem in STEMS and path.suffix in ('.pdf','.log','.bbl')) or path.name == 'task_network_manuscript_final_audit_results.json'
        if not generated:
            assert sha(path) == sha(package/relative), ('staged source/evidence drift', relative)
    visual = json.loads(data(package/'reproducibility/results/task_network_visual_qa_results.json'))
    assert visual['manual_visual_review_completed']
    for stem in STEMS:
        assert visual['formats'][stem]['identical_page_content_streams']
        assert visual['formats'][stem]['packaged_pdf_sha256'] == sha(package/(stem+'.pdf'))
    files = sorted(p for p in package.rglob('*') if eligible(p) and p.name != 'PACKAGE_MANIFEST.json')
    manifest = dict(revision=NAME, new_live_calls=48, development_calls=8,
                    heldout_calls=32, memory_intervention_calls=8,
                    paid_external_calls=0, commit_or_push=False,
                    sha256={p.relative_to(package).as_posix():sha(p) for p in files})
    manifest_path = package/'PACKAGE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    archive = out/(NAME+'_PACKAGE.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files+[manifest_path]:
            z.writestr(path.relative_to(package).as_posix(), data(path))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)+1
        for name, digest in manifest['sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest, name
    (out/'pdf').mkdir(exist_ok=True)
    delivered=[]
    names=(NAME, 'FOUNDATION_MFD_PARTB_IEEE_REVIEWED_2026-09-11_TASK-NETWORK')
    for stem, name in zip(STEMS,names):
        target=out/'pdf'/(name+'.pdf')
        assert not target.exists(), target
        shutil.copy2(native(package/(stem+'.pdf')),native(target))
        assert sha(target)==sha(package/(stem+'.pdf'))
        delivered.append(str(target))
    verification=dict(archive=str(archive), sha256=sha(archive), files=len(files)+1,
                      megabytes=archive.stat().st_size/1e6, all_payload_hashes_pass=True, pdfs=delivered)
    (out/(NAME+'.verification.json')).write_text(json.dumps(verification,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    main()
