"""Package the verified MPC experiment separately from unchanged paper PDFs."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME='FOUNDATION_MFD_MPC_CERTIFICATION_EXPERIMENT_2026-09-10'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    files=list((ROOT/'reproducibility').glob('*.py'))
    files+=list((ROOT/'reproducibility/results').glob('*.json'))
    files+=list((ROOT/'results').glob('mfd_incident_mpc*.json'))
    files += [ROOT/name for name in ('MPC_CERTIFICATION_PROTOCOL.md','MPC_CERTIFICATION_IMPLEMENTATION.md',
              'MPC_CERTIFICATION_RESULTS.md','requirements.txt','requirements-synthesis.txt',
              'Archived/mfd_incident_verifier_pre_mpc.py')]
    return sorted(set(files))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','archive'))
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    package=args.output_dir/'package'/NAME
    if args.stage=='prepare':
        package.mkdir(parents=True,exist_ok=False)
        for source in sources():
            target=package/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
        print(package)
        return
    report=json.loads((package/'reproducibility/results/mfd_incident_mpc_certification_summary_results.json').read_text())
    assert report['complete_cases']==8 and report['certificate_attempts']==192
    for source in sources():
        target=package/source.relative_to(ROOT)
        assert digest(source)==digest(target),source
    files=sorted(p for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    manifest=dict(kind='experiment supplement; manuscript PDFs unchanged',
                  sha256={str(p.relative_to(package)):digest(p) for p in files})
    mpath=package/'EXPERIMENT_MANIFEST.json'
    mpath.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    files.append(mpath)
    output=args.output_dir/(NAME+'.zip')
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in files:
            archive.write(path,path.relative_to(package))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None and len(archive.namelist())==len(files)
    report_output=args.output_dir/'MPC_CERTIFICATION_RESULTS_2026-09-10.md'
    shutil.copy2(package/'MPC_CERTIFICATION_RESULTS.md',report_output)
    print(f'Verified {len(files)} files, {output.stat().st_size/1e6:.2f} MB: {output}')
    print(report_output)


if __name__=='__main__':
    main()
