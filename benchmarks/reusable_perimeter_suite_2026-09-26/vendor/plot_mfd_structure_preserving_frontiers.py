"""Source-bound comparison of tested reachability certificates, not a threshold.

This script performs no controller optimization, reachability run, or live call.
It checks archived proof/summary agreement and the common physical contract,
then exports a categorical figure and its complete source-hashed data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
WIDTHS = [1, 5, 10, 25, 50, 100]
RESIDUALS = [60, 120, 300, 600, 1200]
COMMON_KEYS = ("initial_state", "initial_time", "durations", "mode_fingerprints",
               "pulse", "service_checks", "model_contract", "affine_constraint_contracts",
               "terminal_regional_upper", "terminal_queue_upper", "step_seconds")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def build_data():
    hashes, panels, descriptors = {}, {}, []

    def read(path):
        hashes[relative(path)] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def row_from_proof(path, width, residual, expected=None):
        record = read(path)
        cert = record["certificate"]
        descriptor = cert["proof_descriptor"]
        descriptors.append({k: descriptor[k] for k in COMMON_KEYS})
        widths = [width / 100.] * 6 + [min(width, 10.) / 100.]
        assert cert["initial_box_units"] == widths, (path, "different initial physical box")
        assert cert["residual_bound_units_per_hour"] == residual / 100.
        accepted = cert["accepted"]
        upper = None
        if accepted:
            assert cert["verified_continuous_physical"] and cert["verified_scheduled_service"]
            assert cert["certified_until_h"] == 1.
            service = [r for r in cert["service_check_results"] if r["time_h"] == 1.
                       and r["coordinate"] == 0]
            assert len(service) == 1 and service[0]["passed"] and service[0]["limit"] == 28.5
            upper = 100. * service[0]["upper"]
            assert upper <= 2850.
        failure = cert["failures"][0] if cert["failures"] else None
        if not accepted:
            assert failure is not None
        if expected is not None:
            assert digest(path) == expected["proof_sha256"], (path, "proof hash mismatch")
            assert accepted == expected["accepted"]
            assert upper == expected["service_upper_vehicles"]
            assert cert["certified_until_h"] == expected["certified_until_h"]
        return dict(state_half_width_vehicles=width, queue_half_width_vehicles=min(width, 10),
                    residual_half_width_vehicles_per_hour=residual, accepted=accepted,
                    service_upper_vehicles=upper,
                    displayed_service_upper_vehicles=math.ceil(upper) if accepted else None,
                    certified_until_h=cert["certified_until_h"],
                    displayed_certified_minutes=math.floor(60. * cert["certified_until_h"] * 10.) / 10.,
                    first_failure={k: failure[k] for k in ("kind", "time_h", "constraint")
                                   if k in failure} if failure else None,
                    proof_file=relative(path), proof_sha256=digest(path), proof_id=cert["proof_id"],
                    executed_source_sha256=record.get("source_sha256", {}))

    old = read(RESULTS / "mfd_incident_uncertainty_frontier_results.json")
    panels["original_conservation_box"] = []
    for row in old["rows"]:
        if row["metric"] != "conservation_box" or row["initial_argument"] != "seven_entry_vector":
            continue
        panels["original_conservation_box"].append(row_from_proof(
            ROOT / row["proof_file"].replace("\\", "/"), row["initial_half_width_vehicles"],
            row["residual_vehicles_per_hour"], row))

    panels["positive_queue_v4_v5"] = []
    positive_paths = sorted(RESULTS.glob("mfd_positive_positive_queue_*_axis_v4.json"))
    positive_paths += sorted(RESULTS.glob("mfd_positive_positive_queue_*_joint_v5.json"))
    assert len(positive_paths) == 14
    for path in positive_paths:
        source = read(path)
        assert source["variant"] == "positive_queue"
        panels["positive_queue_v4_v5"].append(row_from_proof(
            path, source["state_half_width_vehicles"], source["residual_vehicles_h"]))

    affine = read(RESULTS / "mfd_incident_affine_reachability_v3_results.json")
    assert len(affine["rows"]) == 17
    assert affine["source_sha256"]["reproducibility/mfd_incident_affine_reachability.py"] == (
        "87baeff5f3a2bb2cd1f3a2059c7c342e01a93d4f2afa2ad2b93b397ab3800e8e")
    panels["signed_generator_queue_v3"] = [row_from_proof(
        ROOT / row["proof_file"], row["state_half_width_vehicles"],
        row["residual_vehicles_per_hour"], row) for row in affine["rows"]]
    for key, rows in panels.items():
        cells = [(r["state_half_width_vehicles"], r["residual_half_width_vehicles_per_hour"]) for r in rows]
        assert len(cells) == len(set(cells)), (key, "duplicate tested cell")
        assert all(w in WIDTHS and r in RESIDUALS for w, r in cells)
        rows.sort(key=lambda r: (r["residual_half_width_vehicles_per_hour"], r["state_half_width_vehicles"]))
    assert all(d == descriptors[0] for d in descriptors), "Model/commands/forecast/contracts differ"
    hashes[relative(Path(__file__))] = digest(Path(__file__))
    return dict(
        scope="One fixed archived MPC plan; complete one-hour physical + regional-service certification.",
        comparison_checks="Exact stored physical boxes, commands (mode fingerprints), forecast, model and service contracts agree.",
        state_order=descriptors[0]["model_contract"]["state_order"],
        initial_half_width_rule="First six physical-state coordinates: +/- row width vehicles; queue: +/- min(width,10) vehicles.",
        residual_rule="Measurable additive arrival residual, absolute value at most column value (vehicles/hour).",
        axes=dict(state_half_width_vehicles=WIDTHS, residual_half_width_vehicles_per_hour=RESIDUALS),
        common_contract=descriptors[0], source_sha256=hashes, panels=panels,
        counts={k: dict(tested=len(rows), passed=sum(r["accepted"] for r in rows),
                        failed=sum(not r["accepted"] for r in rows), untested=30-len(rows))
                for k, rows in panels.items()},
        display_rules=dict(success="One-hour regional upper support, rounded upward to whole vehicles.",
                           failure="Last fully certified time, rounded downward to 0.1 minute; failed proof is not an unsafe trajectory.",
                           untested="Grey, not inferred from adjacent cells.",
                           grid="Categorical tested grid: no linear/log interpolation or global necessary threshold.",
                           comparison="A method's pass count is not directly compared as a rate because tested grids differ."))


def plot(data, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    ink, green, amber, grey = "#20313C", "#C5E5D8", "#F8DCC1", "#EEF0F2"
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 6.3), sharey=True)
    fig.subplots_adjust(left=.082, right=.985, bottom=.29, top=.765, wspace=.16)
    names = [("original_conservation_box", "(a) Original conservation box"),
             ("positive_queue_v4_v5", "(b) Positivity + scalar queue"),
             ("signed_generator_queue_v3", "(c) Signed generators + queue")]
    for ax, (key, title) in zip(axes, names):
        lookup = {(r["state_half_width_vehicles"], r["residual_half_width_vehicles_per_hour"]): r
                  for r in data["panels"][key]}
        for y, residual in enumerate(RESIDUALS):
            for x, width in enumerate(WIDTHS):
                row = lookup.get((width, residual))
                color = grey if row is None else green if row["accepted"] else amber
                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor=color, edgecolor="white", linewidth=2))
                if row is None:
                    ax.text(x, y, "-", ha="center", va="center", color="#8A949A", fontsize=10)
                elif row["accepted"]:
                    ax.text(x, y, str(row["displayed_service_upper_vehicles"]), ha="center", va="center",
                            color="#174C3A", fontsize=9.3, fontweight="bold")
                else:
                    ax.text(x, y+.14, "x", ha="center", va="center", color="#8E471C", fontsize=10,
                            fontweight="bold")
                    ax.text(x, y-.15, f'{row["displayed_certified_minutes"]:.1f} m', ha="center", va="center",
                            color="#8E471C", fontsize=7.6)
        ax.set_xlim(-.5, 5.5)
        ax.set_ylim(-.5, 4.5)
        ax.set_xticks(range(6), [str(x) for x in WIDTHS])
        ax.set_yticks(range(5), [str(x) for x in RESIDUALS])
        ax.set_title(title, loc="left", pad=14, fontsize=10.6, color=ink, fontweight="bold")
        ax.tick_params(length=0, pad=7, labelsize=9)
        ax.set_xlabel("Initial-state half-width (vehicles)", labelpad=10, fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Arrival residual half-width (veh/h)", labelpad=11, fontsize=10)
    fig.suptitle("Transportation structure enlarges the tested certification range", y=.973,
                 fontsize=15, color=ink, fontweight="bold")
    fig.text(.5, .915, "Same MPC commands, one-hour horizon, 2,850-vehicle regional deadline and 2,000-vehicle queue limit",
             ha="center", fontsize=10, color=ink)
    fig.legend(handles=[Patch(facecolor=green, label="Certified: regional bound (veh)"),
                        Patch(facecolor=amber, label="Not certified: last verified time"),
                        Patch(facecolor=grey, label="Untested")],
               loc="upper center", bbox_to_anchor=(.535, .875), ncol=3, frameon=False,
               fontsize=9, handlelength=1.2, columnspacing=2.5)
    fig.text(.082, .185, "Complementary bounds", fontweight="bold", fontsize=10.4, color=ink)
    fig.text(.082, .142,
             "At 1 veh / 300 veh/h, positivity gives the tighter bound (2,009 vs 2,256 veh).\n"
             "At 10 veh / 300 veh/h, signed generators give the tighter bound (2,303 vs 2,700 veh).",
             fontsize=9.2, color=ink, linespacing=1.5, va="top")
    fig.text(.082, .044,
             "Queue half-width = min(initial-state half-width, 10 veh). Successful bounds are rounded up; times are rounded down.\n"
             "Each cell is a tested sufficient certificate. A failed enclosure does not establish an unsafe trajectory; grey cells are not inferred.",
             fontsize=8.2, color="#4C5B65", linespacing=1.5, va="center")
    fig.savefig(output.with_suffix(".png"), dpi=230, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white", metadata={
        "Title": "Structure-preserving MFD reachability: tested fixed-plan certification grid",
        "Subject": "Source-bound categorical grid; failed proofs and untested cells retained.",
        "Creator": relative(Path(__file__)), "CreationDate": None, "ModDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "mfd_structure_preserving_frontiers")
    parser.add_argument("--replace-figure", action="store_true", help="Replace this figure's generated artifacts only.")
    args = parser.parse_args()
    output = args.output.resolve()
    data_path = output.with_name(output.name + "_figuredata.json")
    manifest_path = output.with_name(output.name + "_manifest.json")
    targets = [output.with_suffix(".png"), output.with_suffix(".pdf"), data_path, manifest_path]
    if not args.replace_figure and any(p.exists() for p in targets):
        raise FileExistsError("Figure artifact exists; use a new output stem or explicit --replace-figure.")
    data = build_data()
    output.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    plot(data, output)
    manifest_path.write_text(json.dumps(dict(
        generated_sha256={relative(p): digest(p) for p in targets[:-1]},
        source_sha256=data["source_sha256"], counts=data["counts"]), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", counts=data["counts"], manifest=relative(manifest_path)), indent=2))


if __name__ == "__main__":
    main()
