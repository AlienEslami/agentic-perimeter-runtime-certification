"""Restore compressed connected-ring proofs for offline audits."""
from pathlib import Path
import gzip
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "REPRODUCTION_MANIFEST.json").read_text(encoding="utf-8"))
    unpacked = reused = 0
    for relative, record in manifest["files"].items():
        if not relative.startswith("benchmarks/connected_network_2026-09-11/") or record["compression"] != "gzip":
            continue
        source = ROOT / relative
        payload = gzip.decompress(source.read_bytes())
        assert hashlib.sha256(payload).hexdigest() == record["source_sha256"], relative
        destination = source.with_suffix("")
        assert destination.resolve().is_relative_to(ROOT)
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != record["source_sha256"]:
                raise RuntimeError(f"Existing record differs: {destination.relative_to(ROOT)}")
            reused += 1
            continue
        with destination.open("xb") as handle:
            handle.write(payload)
        unpacked += 1
    print(f"Verified connected-network proofs: {unpacked} unpacked, {reused} already present.")


if __name__ == "__main__":
    main()
