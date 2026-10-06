"""Extract the existing minute-20 benchmark state without changing it."""
from common import HERE, save
import hashlib,json
from pathlib import Path
root=HERE.parents[1]
source=root/"output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK/reproducibility/results/sor/adverse25_affine_v3/run.json"
run=json.loads(source.read_text(encoding="utf-8")); event=run["decisions"][4]
proof=root/"output/package/FOUNDATION_MFD_PARTB_REVIEWED_2026-09-11_TASK-NETWORK"/event["execution_binding"]["proof_reference"]["path"]
record=json.loads(proof.read_text(encoding="utf-8"))
save(HERE/"anchor.json",dict(center=event["information_box"]["center"],
    original_half_widths=event["information_box"]["half_widths"],candidate_commands=event["candidate_commands"],
    backup_commands=[record["commands"][1]]*2,source_epoch=4,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    continuation_source_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),source_certificate_accepted=record.get("accepted",record.get("certificate",{}).get("accepted"))))
print("Portable minute-20 anchor prepared.")
