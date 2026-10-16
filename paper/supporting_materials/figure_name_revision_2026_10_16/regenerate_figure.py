"""Redraw an existing scientific figure using frozen arrays and revised names.

Reads only archived figure-data JSON. It performs no simulation, numerical
integration, optimization, network call, or change to scientific arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "frozen_figuredata.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Frozen figure-data JSON; defaults to the file beside this script.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Explicit local folder for the regenerated PDF, PNG and verification record.")
    args = parser.parse_args()
    source = args.data.resolve()
    output = args.output_dir.resolve()
    original_hash = digest(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    original_arrays = json.dumps(
        {"curves": data["curves"], "matched_comparisons": data["matched_comparisons"]},
        sort_keys=True, allow_nan=False, separators=(",", ":"),
    )
    comparisons = data["matched_comparisons"]
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / "mfd_structure_online_runtime"

    # Plot settings and every numerical curve match the archived generator.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,bx)=plt.subplots(1,2,figsize=(8.0,4.1))
    fig.subplots_adjust(left=.09,right=.98,bottom=.26,top=.73,wspace=.30)
    navy,teal,gray='#264b7f','#138574','#757575'
    labels={
        'binding1600_original':'All three enclosures: target 1600',
        'binding1625_affine_v3':'Signed: target 1625',
        'nominal_1700_c1600':'Matched nominal replays (overlap)'}
    lines=[]
    for name,color,style in [('binding1600_original',navy,'-'),('binding1625_affine_v3',teal,'--'),
                              ('nominal_1700_c1600',gray,':')]:
        row=data['curves'][name]
        line,=ax.plot(row['trajectory_minutes'],row['trajectory_n1_vehicles'],
            color=color,linestyle=style,linewidth=1.65,label=labels[name])
        lines.append(line)
        ax.plot(60,row['deadline_n1_vehicles'],'o',color=color,markersize=3)
    ax.hlines(1700,52,60,colors='black',linestyles='-.',linewidth=.85)
    ax.plot(60,1700,marker='D',color='black',markersize=3)
    ax.annotate('1700 at 60 min',xy=(59.6,1700),xytext=(37,1880),fontsize=8,
        arrowprops=dict(arrowstyle='-',lw=.65,color='black'),ha='left',va='bottom')
    ax.set(xlabel='Time (min)',ylabel='Region 1 accumulation (vehicles)',xlim=(0,61),ylim=(1500,3070))
    ax.set_title('(a) Regional service',loc='left',pad=8)
    ax.set_xticks([0,15,30,45,60]);ax.set_yticks([1500,2000,2500,3000])
    for name,color,style in [('binding1600_original',navy,'-'),('binding1625_affine_v3',teal,'--')]:
        row=comparisons[name]
        bx.plot(row['minutes'],row['cumulative_incremental_TTS_veh_h'],color=color,
            linestyle=style,linewidth=1.7,marker='o',markersize=2.5)
        final=row['final_incremental_TTS_veh_h']
        bx.annotate(f'+{final:.1f}',xy=(60,final),xytext=(-3,7),textcoords='offset points',
            color=color,fontsize=9,ha='right',va='bottom')
    bx.axhline(0,color=gray,linewidth=.6)
    bx.set(xlabel='Time (min)',ylabel='Cumulative additional TTS (veh-h)',xlim=(0,61),ylim=(-5,180))
    bx.set_title('(b) Cost versus matched nominal replay',loc='left',pad=8)
    bx.set_xticks([0,15,30,45,60]);bx.set_yticks([0,50,100,150])
    for axes in (ax,bx):
        axes.grid(axis='y',color='#dddddd',linewidth=.5,alpha=.8)
        axes.set_axisbelow(True)
    fig.suptitle(r'Online service task: $100n_1(1\,\mathrm{h})\leq1700$ vehicles',y=.98,fontsize=12)
    fig.legend(handles=lines,loc='upper center',bbox_to_anchor=(.52,.89),ncol=2,frameon=False,
               columnspacing=1.4,handlelength=2.8)
    fig.text(.09,.105,'Targets 1600 and 1625 refer to initial plans; the latter is a separate design. The three target-1600 curves coincide.',fontsize=7.6)
    fig.text(.09,.063,'Each TTS difference uses its own matched nominal request replay under the same physical and 1700-vehicle service contracts.',fontsize=7.6)
    fig.savefig(prefix.with_suffix('.pdf'),metadata={'Title':'Online structure-aware service certification',
        'Subject':'Matched nominal request replay; separate 1600/1625 initial-plan designs'})
    fig.savefig(prefix.with_suffix('.png'),dpi=220)
    plt.close(fig)

    unchanged_arrays = json.dumps(
        {"curves": data["curves"], "matched_comparisons": data["matched_comparisons"]},
        sort_keys=True, allow_nan=False, separators=(",", ":"),
    )
    assert unchanged_arrays == original_arrays
    assert digest(source) == original_hash
    record = {
        "operation": "Display-name revision from frozen figure data only",
        "input": source.name,
        "input_sha256": original_hash,
        "scientific_arrays_sha256": hashlib.sha256(original_arrays.encode()).hexdigest(),
        "scientific_arrays_unchanged": True,
        "axes_limits_ticks_styles_unchanged": True,
        "simulation_or_API_calls": False,
        "display_labels": labels,
        "outputs": {p.name: digest(p) for p in [prefix.with_suffix('.pdf'), prefix.with_suffix('.png')]},
    }
    (output / "FIGURE_NAME_REVISION_QA.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
