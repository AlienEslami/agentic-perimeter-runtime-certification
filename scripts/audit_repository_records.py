"""Read-only copy-integrity and certificate-binding audit; no network calls."""
from pathlib import Path
from fractions import Fraction
import ast
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "benchmarks/reusable_perimeter_suite_2026-09-26"


def certificates(obj):
    if isinstance(obj, dict):
        if "proof_id" in obj and "proof_descriptor" in obj:
            yield obj
            return
        for value in obj.values():
            yield from certificates(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from certificates(value)


def main():
    manifest = json.loads((ROOT / "REPRODUCTION_MANIFEST.json").read_text(encoding="utf-8"))
    for relative, record in manifest["files"].items():
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT), relative
        stored = path.read_bytes()
        assert hashlib.sha256(stored).hexdigest() == record["stored_sha256"], relative
        original = gzip.decompress(stored) if record["compression"] == "gzip" else stored
        assert hashlib.sha256(original).hexdigest() == record["source_sha256"], relative
    sources = 0
    for path in ROOT.rglob("*.py"):
        if ".git" not in path.parts and "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            sources += 1
    proof_count = 0
    source_hashes = {}
    folders = ["contracts", "full_scenario", "reserve_refresh", "end_to_end_api", "delayed_handover", "sequential_authority_2026-10-01"]
    for folder in folders:
        for path in sorted((SUITE / "results" / folder).rglob("*.json")):
            obj = json.loads(path.read_text(encoding="utf-8"))
            for certificate in certificates(obj):
                descriptor = certificate["proof_descriptor"]
                expected = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
                assert expected == certificate["proof_id"], path.relative_to(ROOT)
                adapter = certificate.get("adapter_audit", {})
                if adapter:
                    assert adapter["proof_hash_recomputed"], path.relative_to(ROOT)
                    for row in adapter["commands"]:
                        command = [Fraction.from_float(float(v)) for v in row]
                        assert len(command) == 4 and all(0 <= v <= 1 for v in command)
                        assert abs(command[1] - command[0]) <= Fraction(3, 4)
                        assert abs(command[3] - command[2]) <= Fraction(3, 4)
                    assert all(check["passed"] for check in adapter["exact_actuator_checks"])
                for relative, expected_hash in descriptor.get("signed_generator_method", {}).get("source_sha256", {}).items():
                    source = (SUITE / relative).resolve()
                    assert source.is_relative_to(SUITE), relative
                    if relative not in source_hashes:
                        source_hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
                    assert source_hashes[relative] == expected_hash, relative
                proof_count += 1
    print(json.dumps({"passed": True, "copied_files_verified": len(manifest["files"]), "python_sources_parsed": sources, "certificate_records_verified": proof_count, "certificate_sources_verified": len(source_hashes)}, indent=2))


if __name__ == "__main__":
    main()
