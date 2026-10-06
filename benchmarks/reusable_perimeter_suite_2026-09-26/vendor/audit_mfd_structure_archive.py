"""Validate a release and exercise audit entry points from a fresh extraction."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


def native(path):
    value = str(path.resolve())
    return '\\\\?\\'+value if os.name == 'nt' else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination-parent', type=Path, required=True)
    args = parser.parse_args()
    folder = Path(tempfile.mkdtemp(prefix='srqa_', dir=args.destination_parent))
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read('EXPERIMENT_MANIFEST.json'))
        for name in archive.namelist():
            parts = PurePosixPath(name)
            assert not parts.is_absolute() and '..' not in parts.parts and '\\' not in name
        assert len(archive.namelist()) == len(manifest['sha256'])+1
        for name, expected in manifest['sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        archive.extractall(native(folder))
    for name, expected in manifest['sha256'].items():
        with open(native(folder/name), 'rb') as handle:
            assert hashlib.sha256(handle.read()).hexdigest() == expected, name
    checks = []
    for script, extra in [
        ('audit_mfd_structure_preserving.py', []),
        ('audit_affine_matrix_functions_review.py', []),
        ('mfd_incident_queue_bounds.py', ['--self-test'])]:
        result = subprocess.run([sys.executable, str(folder/'reproducibility'/script), *extra],
            cwd=folder, text=True, encoding='utf-8', capture_output=True, check=True)
        checks.append(dict(script=script, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result = dict(passed=True, archive=str(args.archive), extracted_directory=str(folder),
        payloads_verified=len(manifest['sha256']), fresh_extraction_audits=checks,
        note='The extracted aggregate audit regenerates its local summary after the payload hash check; release archive is unchanged.')
    target = args.archive.with_suffix('.verification.json')
    with target.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(passed=True, extraction=str(folder), verification_record=str(target),
                         payloads_verified=len(manifest['sha256']), independent_entry_points=len(checks)), indent=2))


if __name__ == '__main__':
    main()
