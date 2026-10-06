#!/usr/bin/env python3
"""Stage and archive the reviewed manuscript with all retained result records.

Run ``prepare``, build and audit both layouts inside the printed directory,
then run ``archive``. Existing release directories/archives are not replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STEMS = ('FOUNDATION_MFD_PARTB_ELSEVIER', 'FOUNDATION_MFD_PARTB_IEEE_READING_COPY')


def eligible(path: Path, boundary: Path = ROOT) -> bool:
    return (path.is_file() and '__pycache__' not in path.parts
            and path.suffix not in {'.pyc', '.aux', '.out', '.blg', '.spl'}
            and not any(part.startswith('.') for part in path.relative_to(boundary).parts))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_file(source: Path, target: Path) -> None:
    # CopyFile2 otherwise rejects long Windows paths in nested live archives.
    if os.name == 'nt':
        shutil.copy2('\\\\?\\' + str(source.resolve()), '\\\\?\\' + str(target.resolve()))
    else:
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'archive'))
    parser.add_argument('--date', default='2026-09-05')
    parser.add_argument('--revision', default='', help='Optional short release suffix.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output',
                        help='Use a short path on Windows for deep live-record directories.')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', args.date):
        parser.error('Use a YYYY-MM-DD release date.')
    if args.revision and not re.fullmatch(r'[A-Za-z0-9-]+', args.revision):
        parser.error('Use letters, digits, and hyphens in the revision suffix.')
    suffix = '_' + args.revision.upper() if args.revision else ''
    name = f'FOUNDATION_MFD_PARTB_REVIEWED_{args.date}' + suffix
    output_root = args.output_dir.resolve()
    package = output_root / 'package' / name
    base_files = [f'{stem}{ext}' for stem in STEMS
                  for ext in ('.tex', '.pdf', '.bbl', '.log')]
    base_files += [
        'FOUNDATION_REFERENCES.bib', 'IEEEtran.cls', 'IEEEtranN.bst',
        'SUBMISSION_PACKAGE_README.md',
        f'FOUNDATION_MFD_PARTB_REVIEW_{args.date}.md',
        'mfd_architecture_error.png', 'requirements.txt',
        'requirements-synthesis.txt', 'MFD_AVERAGE_DWELL_PROTOCOL.md',
        'MFD_CALIBRATION_UNCERTAINTY_PROTOCOL.md',
        'MFD_NONSTATIONARY_STUDY_PROTOCOL.md', 'MFD_SEQUENTIAL_LLM_PROTOCOL.md',
        'DEADLINE_STUDY_PROTOCOL.md', 'DEADLINE_PRIOR_ART.md',
        'DEADLINE_THEORY_AUDIT.md', 'DEADLINE_OUTPUT_PROOF.md',
    ]
    if args.revision.upper() in ('INCIDENT', 'STREAMLINED'):
        base_files += [
            f'FOUNDATION_MFD_PARTB_REVIEW_{args.date}_INCIDENT.md',
            'INCIDENT_STUDY_PROTOCOL.md', 'INCIDENT_PRIOR_ART.md',
            'INCIDENT_MODEL_AUDIT.md', 'INCIDENT_RESULTS_COMPARISON.md',
        ]
        base_files += [p.relative_to(ROOT).as_posix()
                       for p in sorted((ROOT / 'results').glob('mfd_incident*.json'))]
    if args.revision.upper() == 'STREAMLINED':
        base_files += [
            'MPC_CERTIFICATION_PROTOCOL.md', 'MPC_CERTIFICATION_IMPLEMENTATION.md',
            'MPC_CERTIFICATION_RESULTS.md', 'STREAMLINED_REVISION_REPORT.md',
            'Archived/FOUNDATION_MFD_PARTB_PRE_MPC_STREAMLINE_2026-09-10.tex',
            'Archived/mfd_incident_verifier_pre_mpc.py',
        ]
        base_files += [p.relative_to(ROOT).as_posix()
                       for p in sorted((ROOT / 'manuscript_streamlined').glob('*.tex'))]
    if args.stage == 'prepare':
        package.mkdir(parents=True, exist_ok=False)
        sources = [ROOT / item for item in base_files]
        sources += [path for folder in ('reproducibility', 'live_llm_results')
                    for path in (ROOT / folder).rglob('*') if eligible(path)]
        for source in sources:
            target = package / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            copy_file(source, target)
        print(f'Staged {len(sources)} files: {package}')
        return

    if not package.is_dir():
        parser.error('Run prepare before archive.')
    # Keep the exact source and every raw result used by the reviewed manuscript.
    for item in base_files:
        assert (package / item).is_file(), item
    for stem in STEMS:
        assert digest(package / (stem + '.tex')) == digest(ROOT / (stem + '.tex'))
    raw_files = [p for p in (ROOT / 'live_llm_results').rglob('*') if eligible(p)]
    for source in raw_files:
        assert digest(source) == digest(package / source.relative_to(ROOT)), source
    source_text = (package / (STEMS[0] + '.tex')).read_text(encoding='utf-8')
    for module in (package / 'manuscript_streamlined').glob('*.tex'):
        source_text += '\n' + module.read_text(encoding='utf-8')
    for filename in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', source_text):
        assert (package / filename).is_file(), filename

    files = sorted(p for p in package.rglob('*') if eligible(p, package)
                   and p.name != 'PACKAGE_MANIFEST.json')
    manifest = {
        'release': args.date,
        'revision': args.revision,
        'retained_live_record_files': len(raw_files),
        'sha256': {p.relative_to(package).as_posix(): digest(p) for p in files},
    }
    manifest_path = package / 'PACKAGE_MANIFEST.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    files.append(manifest_path)
    archive = output_root / f'{name}_PACKAGE.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in files:
            z.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist()) == len(files)
    output = output_root / 'pdf'
    output.mkdir(parents=True, exist_ok=True)
    for stem, final_name in zip(STEMS, (name, f'FOUNDATION_MFD_PARTB_IEEE_REVIEWED_{args.date}' + suffix)):
        target = output / (final_name + '.pdf')
        if target.exists():
            raise FileExistsError(target)
        copy_file(package / (stem + '.pdf'), target)
        assert digest(target) == digest(package / (stem + '.pdf'))
        print(target)
    print(f'Verified archive: {archive} ({archive.stat().st_size / 1e6:.2f} MB)')
    print(f'{len(files)} entries; {len(raw_files)} unchanged live-record files.')


if __name__ == '__main__':
    main()
