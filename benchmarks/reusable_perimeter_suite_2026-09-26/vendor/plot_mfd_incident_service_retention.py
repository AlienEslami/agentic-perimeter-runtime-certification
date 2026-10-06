"""Static paired service-retention figure from source-hashed simulation records.

No simulation is rerun, no curve is rescaled or smoothed, and the rejected
replacement is explicitly counterfactual. Whole-hour costs share the same
executed five-minute prefix. This is one replacement attempt, not rolling MPC.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(__file__).resolve().parent/'results'
SOURCE = RESULTS/'mfd_incident_service_retention_results.json'
AUDIT = RESULTS/'mfd_incident_service_frontier_audit_results.json'
OUTPUT = RESULTS/'mfd_incident_service_retention'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def upward_hundredth(value):
    scaled = 100*Fraction.from_float(float(value))
    return (-(-scaled.numerator//scaled.denominator))/100.


def main():
    record, audit = load(SOURCE), load(AUDIT)
    audit_hashes = {key.replace('\\', '/'): value for key, value in audit['result_sha256'].items()}
    assert audit['passed'] and audit_hashes[SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)
    old = record['commissioning']['certificate']
    new = record['replacement_certificate']
    assert old['accepted'] and not new['accepted'] and record['no_forced_outage']
    assert record['trusted_contract']['same_contract_before_and_after_rejection']
    assert record['original_expiry_preserved']
    assert old['proof_descriptor']['service_checks'] == new['proof_descriptor']['service_checks']
    assert record['trusted_contract']['service_limit_vehicles'] == 1700.
    assert record['commissioning']['nominal_target_vehicles'] == 1600.
    prefix = record['prefix']['trajectory']
    times0 = np.asarray(prefix['time_h'])
    states0 = np.asarray(prefix['state'])
    curves = {}
    for key, row_name in [('retained', 'retained_suffix_replay'),
                          ('replacement', 'replacement_replay_to_common_deadline')]:
        continuation = record[row_name]['trajectory']
        times = np.r_[times0, continuation['time_h'][1:]]
        values = np.vstack((states0, continuation['state'][1:]))
        assert np.array_equal(states0[-1], np.asarray(continuation['state'][0]))
        assert abs(times[-1]-1.) < 1e-12 and abs(times0[-1]-1/12) < 1e-12
        curves[key] = dict(time_minutes=(60*times).tolist(),
                           region1_vehicles=(100*values[:, 0]).tolist())
    trusted = record['trusted_contract']['service_limit_vehicles']
    upper_old = 100*old['service_check_results'][0]['upper']
    upper_new = 100*new['service_check_results'][0]['upper']
    assert upper_old <= trusted < upper_new
    costs = [record['nominal_replacement_total_TTS_veh_h'], record['retained_total_TTS_veh_h']]
    delta = costs[1]-costs[0]
    premium = 100*delta/costs[0]
    assert abs(premium-record['retention_TTS_premium_percent']) < 1e-12
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.labelsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 12,
        'ytick.labelsize': 12, 'legend.fontsize': 12,
        'axes.linewidth': .75, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    blue, orange, gray = '#0072B2', '#D55E00', '#4B535B'
    fig = plt.figure(figsize=(8.2, 5.25))
    grid = fig.add_gridspec(1, 2, left=.10, right=.98, top=.79, bottom=.255,
                           width_ratios=[1.45, 1.], wspace=.47)
    left, right = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    for key, color, style in [('retained', blue, '-'), ('replacement', orange, (0, (4, 2)))]:
        curve = curves[key]
        left.plot(curve['time_minutes'], curve['region1_vehicles'], color=color,
                  linestyle=style, linewidth=2.1, zorder=3)
    left.plot(60*times0, 100*states0[:, 0], color=gray, linewidth=2.4, zorder=4)
    left.axvline(5, color=gray, linewidth=.9, linestyle=(0, (2, 3)), zorder=1)
    left.axvline(60, color=gray, linewidth=.8, linestyle=(0, (2, 3)), zorder=1)
    left.plot([24, 60], [trusted, trusted], color=gray, linewidth=.95, linestyle=(0, (3, 2)), zorder=1)
    left.plot(60, trusted, marker='D', markersize=5, color=gray, zorder=5)
    for key, color in [('retained', blue), ('replacement', orange)]:
        left.plot(60, curves[key]['region1_vehicles'][-1], marker='o', markersize=5, color=color, zorder=4)
    left.annotate('Replacement rejected\nat 5 min', xy=(5, 2880), xytext=(18, 3015),
                  fontsize=12, ha='left', va='top',
                  arrowprops=dict(arrowstyle='-', color=gray, linewidth=.8))
    left.annotate('Trusted limit\n1700 veh at 60 min', xy=(60, trusted), xytext=(30, 2290),
                  fontsize=12, ha='left', va='center', color=gray,
                  arrowprops=dict(arrowstyle='-', color=gray, linewidth=.8))
    left.set(xlim=(0, 63), ylim=(1500, 3120), xticks=[0, 15, 30, 45, 60],
             yticks=[1600, 2000, 2400, 2800], xlabel='Time (min)',
             ylabel='Region 1 accumulation (veh)')
    left.set_title('(a) Regional service', loc='left', pad=10, fontweight='bold')
    bars = right.bar([0, 1], costs, width=.62, color=[orange, blue], zorder=3)
    bars[0].set_hatch('//')
    bars[0].set_edgecolor('white')
    for index, cost in enumerate(costs):
        right.text(index, cost+95, f'{cost:,.2f}', ha='center', va='bottom', fontsize=12)
    right.set(ylim=(0, 5600), yticks=[0, 2000, 4000], xticks=[0, 1],
              xticklabels=['Replacement\n(counterfactual)', 'Retained\nsuffix'],
              ylabel='Whole-hour TTS (veh-h)')
    right.text(.5, 5420, f'+{delta:,.2f} veh-h (+{premium:.2f}%)', ha='center', va='top', fontsize=12)
    right.set_title('(b) Whole-hour traffic cost', loc='left', pad=10, fontweight='bold')
    for axis in (left, right):
        axis.grid(axis='y', color='#D5D9DD', linewidth=.55, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
        axis.tick_params(width=.6, length=3)
    fig.legend(handles=[Line2D([0], [0], color=blue, linewidth=2.1, label='Executed retained suffix'),
                        Line2D([0], [0], color=orange, linewidth=2.1, linestyle=(0, (4, 2)),
                               label='Rejected replacement (counterfactual)')],
               loc='upper center', bbox_to_anchor=(.54, .995), ncol=1, frameon=False,
               handlelength=3., labelspacing=.4)
    fig.text(.50, .145, 'Backup planned for 1600 veh; trusted service limit: 1700 veh at 60 min.',
             ha='center', va='center', fontsize=12)
    fig.text(.50, .098, f'Deadline bounds: retained {upward_hundredth(upper_old):.2f} veh; replacement {upward_hundredth(upper_new):.2f} veh (fails).',
             ha='center', va='center', fontsize=12)
    fig.text(.50, .05, 'Shared first 5 min and arrival; no further proposals; original expiry: 60 min.',
             ha='center', va='center', fontsize=12, color=gray)
    fig.savefig(OUTPUT.with_suffix('.pdf'))
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=220)
    plt.close(fig)
    provenance = dict(scope='Nominal realization of single-replacement retention study; rejected plan is a counterfactual',
        source_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in [SOURCE, AUDIT]},
        plot_script_sha256=sha(Path(__file__)),
        minimum_font_points=12, size_inches=[8.2, 5.25],
        commissioned_nominal_target_vehicles=1600., trusted_service_limit_vehicles=trusted,
        deadline_minutes=60., replacement_attempt_minutes=5., further_proposals=0,
        original_expiry_minutes=60., forced_outage=False,
        retained_certified_deadline_upper_vehicles=upper_old,
        replacement_uncertified_deadline_upper_vehicles=upper_new,
        whole_hour_TTS_veh_h=dict(replacement=costs[0], retained=costs[1]),
        retention_cost_veh_h=delta, retention_cost_percent=premium,
        cost_scope='Cost of this commissioned backup versus this particular replacement; not optimal price of certification',
        prefix_shared_exactly=True, curves=curves,
        annotation_rounding='Certificate upper bounds rounded upward to0.01vehicles; descriptive TTS values rounded nearest0.01veh-h',
        physical_and_service_scope='Nominal adaptive trajectories; both branches use identical initial prefix and arrival realization')
    OUTPUT.with_name(OUTPUT.name+'_figuredata.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    assert audit_hashes[SOURCE.relative_to(ROOT).as_posix()] == sha(SOURCE)
    print(json.dumps(dict(pdf=str(OUTPUT.with_suffix('.pdf')), png=str(OUTPUT.with_suffix('.png')),
                          source_preserved=True, retention_TTS_percent=premium)), flush=True)


if __name__ == '__main__':
    main()
