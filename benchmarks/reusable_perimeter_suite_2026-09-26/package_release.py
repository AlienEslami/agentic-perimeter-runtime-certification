"""Validate the recorded experiment and make a portable, private handoff ZIP.

This is a record-integrity check, additional to the numerical audits. It does
not replace the interval proofs or establish any new mathematical guarantee.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
from fractions import Fraction
import zipfile

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
RELEASE = "REUSABLE_PERIMETER_BENCHMARK_2026-09-26"
REPORT = WORKSPACE / "output/pdf/PERIMETER_BENCHMARK_SWEEP_2026-09-26.pdf"
ARCHIVE = WORKSPACE / "archive/pre_reusable_benchmark_2026-09-26"
PRESERVED = {
    "FOUNDATION_MFD_PARTB_CONSISTENCY_REVISION_2026-09-25.zip":
        "f439a78c706a45aa0d962716c011803e15c719930d08db9881b6db e21663087e".replace(" ", ""),
    "OVERLEAF_FULL_ELSEVIER_2026-09-25.zip":
        "d2677f21d35c97e83357436ea8ead6e317f6d75e429ba131b39e41d5e8848f94",
}
SECRET_PATTERN = re.compile(rb"(?<![A-Za-z0-9_])sk-[A-Za-z0-9_-]{20,}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_commands(commands):
    for row in commands:
        u = [Fraction.from_float(float(v)) for v in row]
        assert len(u) == 4
        assert all(Fraction(0) <= v <= Fraction(1) for v in u)
        assert abs(u[1] - u[0]) <= Fraction(3, 4)
        assert abs(u[3] - u[2]) <= Fraction(3, 4)


def walk_certificates(obj):
    if isinstance(obj, dict):
        if "proof_id" in obj and "proof_descriptor" in obj:
            yield obj
            return
        for value in obj.values():
            yield from walk_certificates(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from walk_certificates(value)


def main():
    summary = json.loads((HERE / "results/summary.json").read_text())
    counts = summary["counts"]
    assert counts == dict(regulation_cases=24, queue_contracts=12,
                         deadline_development=72, deadline_held_out=8,
                         all_constant_plan_realizations=275200)
    assert len(list((HERE / "results/contracts").glob("*.json"))) == 92
    assert len(list((HERE / "results/full_scenario").glob("*.json"))) == 80
    assert len(list((HERE / "results/reserve_refresh").glob("*.json"))) == 12
    assert len(list((HERE / "results/regulation").glob("*.json"))) == 24
    assert len(list((HERE / "results/end_to_end_api").glob("operational_*.json"))) == 8

    checked = accepted = policy_count = realizations = 0
    hashes = {}
    proof_lookup = {}
    roots = ["contracts", "full_scenario", "reserve_refresh", "end_to_end_api"]
    for root in roots:
        for path in sorted((HERE / "results" / root).glob("*.json")):
            if root == "end_to_end_api" and not path.name.startswith("operational_"):
                continue
            record = json.loads(path.read_text())
            for certificate in walk_certificates(record):
                descriptor = certificate["proof_descriptor"]
                calculated = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
                assert calculated == certificate["proof_id"], path.name
                adapter = certificate["adapter_audit"]
                assert adapter["proof_hash_recomputed"]
                assert adapter["commands_bound_by_mode_fingerprints"]
                exact_commands(adapter["commands"])
                assert all(a["passed"] for a in adapter["exact_actuator_checks"])
                sources = descriptor["signed_generator_method"]["source_sha256"]
                for relative, expected in sources.items():
                    if relative not in hashes:
                        source = (HERE / relative).resolve()
                        assert source.is_relative_to(HERE)
                        hashes[relative] = digest(source)
                    assert hashes[relative] == expected, relative
                checked += 1
                accepted += int(certificate["accepted"])
                proof_lookup[certificate["proof_id"]] = certificate
            for name, policy in record.get("policies", {}).items():
                cert = policy["certificate"]
                assert cert["adapter_audit"]["commands"] == policy["commands"], (path.name, name)
                assert len(policy["realizations"]) == 640
                realizations += len(policy["realizations"])
                policy_count += 1
                if cert["accepted"]:
                    assert policy["summary"]["physical_passes"] == 640
                    if record["contract"]["deadline_vehicles"] is not None:
                        assert policy["summary"]["service_passes"] == 640
                    else:
                        assert policy["summary"]["service_passes"] is None
            if root == "end_to_end_api":
                matching = [c for c in walk_certificates(record)
                            if c["accepted"] and c["adapter_audit"]["commands"] == record["executed_commands"]]
                assert matching, path.name
                assert record["summary"]["physical_passes"] == 640
                assert record["summary"]["service_passes"] == 640
    assert realizations == counts["all_constant_plan_realizations"]

    portfolio = json.loads((HERE / "results/certified_tool_selection.json").read_text())
    assert len(portfolio["rows"]) == 80 and portfolio["available"] == 80
    for row in portfolio["rows"]:
        certificate = proof_lookup[row["proof_id"]]
        assert certificate["accepted"] and certificate["adapter_audit"]["commands"] == row["commands"]
        assert row["available"]
    replay = json.loads((HERE / "results/independent_replay_audit.json").read_text())
    assert replay["passed"] and len(replay["replays"]) == 138
    ledger = json.loads((HERE / "results/api/budget_ledger.json").read_text())
    assert ledger["budget_usd"] == 10 and len(ledger["calls"]) == 117
    assert all(c["status"] == "completed" for c in ledger["calls"])
    cost = sum(c["charged_usd"] for c in ledger["calls"])
    assert cost <= 10 and abs(cost - summary["api_cost"]["known_token_cost_usd"]) < 1e-12
    assert REPORT.exists()
    for name, expected in PRESERVED.items():
        assert digest(ARCHIVE / name) == expected, name

    validation = dict(passed=True, proof_records_checked=checked,
        accepted_proof_records_checked=accepted, recorded_constant_plans=policy_count,
        constant_plan_realizations=realizations, adaptive_replays=138,
        certified_portfolio_contracts=80, live_tool_traces=8, api_attempts=117,
        api_estimated_token_cost_usd=cost, api_budget_usd=10,
        original_manuscripts_unchanged=True, certificate_source_hashes_checked=hashes,
        scope="record integrity plus the supplied numerical audits; no new unconditional availability theorem")
    validation_path = HERE / "results/RELEASE_VALIDATION.json"
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")

    entries = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".tmp"}:
            continue
        data = path.read_bytes()
        if path.suffix.lower() in {".py", ".json", ".md", ".txt", ".csv", ".tex", ".bib"}:
            assert not SECRET_PATTERN.search(data), "Credential-like content found; release stopped"
        entries.append((path, "benchmark/" + path.relative_to(HERE).as_posix()))
    entries.append((REPORT, "report/" + REPORT.name))
    for path in sorted(ARCHIVE.iterdir()):
        if path.is_file():
            entries.append((path, "preserved_manuscript/" + path.name))
    manifest = "\n".join(digest(path) + "  " + name for path, name in entries) + "\n"
    target = WORKSPACE / "output/package" / (RELEASE + ".zip")
    assert target.resolve().is_relative_to(WORKSPACE)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for path, name in entries:
            package.write(path, RELEASE + "/" + name)
        package.writestr(RELEASE + "/SHA256SUMS.txt", manifest)
        package.writestr(RELEASE + "/README.md",
            "# Perimeter benchmark handoff\n\n"
            "Read report/PERIMETER_BENCHMARK_SWEEP_2026-09-26.pdf first.\n\n"
            "benchmark/ contains the definitions, scripts, all final outcomes, raw API responses and reproducibility guide.\n\n"
            "preserved_manuscript/ contains the unchanged 25 September packages.\n\n"
            "This private handoff preserves development records, explicitly excluded from final summaries.\n")
    with zipfile.ZipFile(target) as package:
        assert package.testzip() is None
        assert package.read(RELEASE + "/benchmark/results/summary.json") == (HERE / "results/summary.json").read_bytes()
    print(json.dumps(dict(validation=validation, package=str(target), files=len(entries) + 2,
                          zip_bytes=target.stat().st_size, zip_sha256=digest(target)), indent=2))


if __name__ == "__main__":
    main()
