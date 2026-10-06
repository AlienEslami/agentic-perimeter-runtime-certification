"""Full saved-step audit of the six independent service-margin affine proofs.

Reuses the record checker, not the verifier's plan/step routine. This adds no
optimization, ODE replay, or scientific proof. The 18-method matched study's
separate audit covers its other two enclosure kernels and numerical replays.
"""
from pathlib import Path
import json
import time

from audit_mfd_incident_affine_reachability import audit_proof, target, sha, ROOT, RESULTS


OUTPUT = RESULTS / "mfd_affine_service_margin_audit_results.json"


def run():
    started = time.perf_counter()
    source = RESULTS / "mfd_reachability_service_margin_results.json"
    study = json.loads(source.read_text())
    assert study["completed"] and len(study["rows"]) == 6
    assert sha(ROOT/study["study_protocol_path"]) == study["study_protocol_sha256"]
    kernel = json.loads((RESULTS/"mfd_incident_affine_reachability_v3_results.json").read_text())
    summary = dict(source_sha256=study["source_sha256"], protocol_sha256=kernel["protocol_sha256"],
                   method_source_sha256=kernel["source_sha256"])
    result = dict(status="RUNNING", passed=False, study_file=source.relative_to(ROOT).as_posix(),
        study_sha256=sha(source), audit_source_sha256=sha(Path(__file__)),
        checker_source_sha256=sha(Path(__file__).with_name("audit_mfd_incident_affine_reachability.py")),
        scope="All exact clocks, G/rho links, signed supports, self-inclusion, physical margins and scalar queue links; selected full corridor/matrix/forcing and90-digit diagnostics; no scientific reruns.",
        source_audits=[], proofs=[])
    def save():
        OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for name, expected in study["source_sha256"].items():
        assert sha(ROOT/name) == expected
        assert sha(ROOT/study["source_snapshots"][name]) == expected
        result["source_audits"].append(dict(source=name, sha256=expected, passed=True))
    module, _ = target(3)
    # Most consequential successful row first; no selection affects the full set.
    rows = sorted(study["rows"], key=lambda r: (r["nominal_target_vehicles"] != 1625, -r["nominal_target_vehicles"]))
    save()
    for study_row in rows:
        nominal = study_row["nominal_target_vehicles"]
        record = study_row["proofs"]["affine_v3"]
        proof = ROOT/record["proof_file"]
        assert sha(proof) == record["proof_sha256"]
        wrapper = json.loads(proof.read_text())
        cert = wrapper["certificate"]
        assert wrapper["study_protocol_sha256"] == study["study_protocol_sha256"]
        assert wrapper["nominal_target_vehicles"] == nominal
        assert wrapper["commands"] == study_row["commands"] == cert["adapter_audit"]["commands"]
        assert cert["proof_descriptor"]["service_checks"] == [dict(exact_time="1", coordinate=0, upper=17.)]
        row = dict(row_id=f"service_n{nominal}", proof_file=record["proof_file"], proof_sha256=record["proof_sha256"],
            accepted=cert["accepted"], certified_until_h=cert["certified_until_h"], wall_seconds=cert["wall_seconds"],
            maximum_box_hull_radius=cert["maximum_metric_radius"],
            service_upper_vehicles=100.*cert["service_check_results"][0]["upper"] if cert["service_check_results"] else None)
        checked = audit_proof(module, 3, summary, row)
        checked["nominal_target_vehicles"] = nominal
        result["proofs"].append(checked)
        save()
        print(json.dumps({k:checked[k] for k in ("row_id", "passed", "interval_steps", "accepted")}), flush=True)
    result.update(status="PASS", passed=True, proof_count=len(result["proofs"]),
        interval_step_count=sum(p["interval_steps"] for p in result["proofs"]),
        selected_corridor_step_count=sum(len(p["selected_full_arithmetic"]) for p in result["proofs"]),
        saved_matrix_90digit_entry_checks=sum(p["reference_90digit_entry_checks"] for p in result["proofs"]),
        wall_seconds=time.perf_counter()-started)
    save()
    print(json.dumps({k:result[k] for k in ("status", "proof_count", "interval_step_count", "wall_seconds")}), flush=True)
    return result


if __name__ == "__main__":
    run()
