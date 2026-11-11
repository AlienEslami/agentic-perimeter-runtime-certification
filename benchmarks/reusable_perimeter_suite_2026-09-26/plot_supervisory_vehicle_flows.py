"""Replay the existing D027 comparison and plot its physical flow balances.

Run from any directory. No planning, scenario selection, or stored results are
changed. Saved constant commands are integrated independently with DOP853;
the original 10-second RK4 results and vehicle conservation are cross-checked.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from common import HERE, START, END, arrival, fixture, plant

OUT = HERE / 'results/supervisory_vehicle_flows_2026-11-08'
FIG = HERE / 'figures'
POLICIES = ('nominal_mpc', 'grid_repair', 'continuation')
COLORS = {'nominal_mpc': '#B5472B', 'grid_repair': '#007E87', 'continuation': '#737C85'}
LABELS = {'nominal_mpc': 'Without supervision', 'grid_repair': 'With certified repair',
          'continuation': 'Retained plan'}


def rates(x, u, offered):
    n1, n2, n11, n22, n10, n20, z = x
    q11, q12, q10 = [plant.original.class_flow(v, n1) for v in (n11, n1-n11-n10, n10)]
    q22, q21, q20 = [plant.original.class_flow(v, n2) for v in (n22, n2-n22-n20, n20)]
    a = min(60*z, 120*u[2])
    f12 = u[0]*(q12+q10)
    f21 = u[1]*q21
    exit2 = u[3]*q20
    d1, d2 = sum(plant.INTERNAL_DEMAND[:3]), sum(plant.INTERNAL_DEMAND[3:])
    return 100*np.array([offered, a, d1+f21, q11+f12,
                         d2+f12+a, q22+f21+exit2, f12, f21, q11, q22, exit2, d1, d2])


def replay(record, name):
    c = record['contract']
    commands = np.asarray(record['policies'][name]['commands'])
    x0 = np.asarray(fixture()['center']) + c['width_vehicles']/100
    # Augment with integrated regional/queue balances and the two paper costs.
    y = np.r_[x0, np.zeros(5)]
    chunks = []
    for block, (left, right) in enumerate(((float(START), 5/12), (5/12, float(END)))):
        u = commands[block]
        def rhs(t, y):
            offered = arrival(t, c) + c['residual_veh_h']/100
            flow = rates(y[:7], u, offered)
            f = plant.field(y[:7], u, offered)
            balance = flow[[2, 4, 0]] - flow[[3, 5, 1]]
            assert np.max(abs(balance-100*f[[0, 1, 6]])) < 1e-8
            costs = 100*np.array([y[0]+y[1]+y[6], y[0]+c['weight']*y[1]+y[6]])
            return np.r_[f, balance, costs]
        sol = solve_ivp(rhs, (left, right), y, method='DOP853',
                        rtol=1e-12, atol=1e-13, max_step=1/720, dense_output=True)
        assert sol.success, sol.message
        t = np.linspace(left, right, round((right-left)*3600)+1)
        values = sol.sol(t).T
        flow = np.array([rates(v[:7], u, arrival(ti, c)+c['residual_veh_h']/100)
                         for ti, v in zip(t, values)])
        chunks.append((t, values, flow, np.tile(u, (len(t), 1))))
        y = sol.y[:, -1]
    # Keep both sides at the switching instant: states are continuous and rates jump.
    t, values, flow, u = [np.concatenate([part[k] for part in chunks]) for k in range(4)]
    x = values[:, :7]
    conservation = float(np.max(abs(100*(x[:, [0, 1, 6]]-x0[[0, 1, 6]])-values[:, 7:10])))
    old = next(r for r in record['policies'][name]['realizations']
               if r['corner']=='+++++++' and r['profile']=='constant_high')
    result = dict(peak_queue_vehicles=float(100*x[:, 6].max()),
                  deadline_n1_vehicles=float(100*x[-1, 0]),
                  tts_veh_h=float(y[10]), priority_cost=float(y[11]),
                  conservation_error_vehicles=conservation,
                  minimum_physical_margin=float(min(np.min(plant.margins(a, b)) for a, b in zip(x, u))),
                  certified=record['policies'][name]['certificate']['accepted'],
                  proof_id=record['policies'][name]['certificate']['proof_id'])
    result['differences_from_archived_RK4'] = {k: result[k]-old[k] for k in
        ('peak_queue_vehicles', 'deadline_n1_vehicles', 'tts_veh_h', 'priority_cost')}
    assert max(abs(v) for v in result['differences_from_archived_RK4'].values()) < .02
    assert conservation < 1e-6
    if name != 'nominal_mpc':
        assert result['minimum_physical_margin'] >= -1e-9
        assert result['deadline_n1_vehicles'] <= c['deadline_vehicles']
        assert result['certified']
    header = ['time_h', 'n1_units', 'n2_units', 'n11_units', 'n22_units', 'n10_units', 'n20_units', 'z_units',
              'arrival_veh_h', 'admission_veh_h', 'region1_additions_veh_h', 'region1_removals_veh_h',
              'region2_additions_veh_h', 'region2_removals_veh_h', 'transfer12_veh_h', 'transfer21_veh_h',
              'completion1_veh_h', 'completion2_veh_h', 'exterior_exit_veh_h', 'internal1_veh_h', 'internal2_veh_h',
              'u12', 'u21', 'u02', 'u20']
    with (OUT/f'D027_{name}_trajectory.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(np.column_stack([t, x, flow, u]))
    return dict(t=t*60, x=x*100, flow=flow, u=u, result=result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    source = HERE/'results/contracts/D027.json'
    record = json.loads(source.read_text())
    runs = {name: replay(record, name) for name in POLICIES}
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.titlesize':10, 'axes.labelsize':9, 'legend.fontsize':8,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.0), sharex=True,
                             sharey='row', layout='constrained')
    titles = ('Exterior queue', 'Region 1', 'Region 2')
    for col, name in enumerate(POLICIES[:2]):
        r = runs[name]
        for row in range(3):
            ax = axes[row, col]
            incoming, outgoing = r['flow'][:, 2*row:2*row+2].T
            ax.plot(r['t'], incoming, color='#2B455C', lw=1.8, label='Incoming / generated')
            ax.plot(r['t'], outgoing, color='#007E87', lw=1.8, ls='--', label='Outgoing / completed')
            ax.fill_between(r['t'], incoming, outgoing, where=incoming>outgoing,
                            interpolate=True, color='#D78A58', alpha=.24)
            ax.fill_between(r['t'], incoming, outgoing, where=incoming<outgoing,
                            interpolate=True, color='#69B6B1', alpha=.22)
            ax.set_title(f'({chr(97+2*row+col)}) {titles[row]}', loc='left')
            ax.axvline(25, color='#9AA3AA', lw=.8, ls=':')
            ax.grid(axis='y', color='#E3E7EA', lw=.6)
            ax.set_xlim(20, 60)
            ax.set_xticks([20, 30, 40, 50, 60])
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            if col == 0: ax.set_ylabel('Flow (veh/h)')
            if row == 2: ax.set_xlabel('Time (min)')
        axes[0, col].text(.5, 1.30, LABELS[name], transform=axes[0, col].transAxes,
                          ha='center', fontweight='bold', color=COLORS[name], fontsize=11)
    axes[0, 0].legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none')
    fig.savefig(FIG/'supervisory_flow_balances.pdf', bbox_inches='tight')
    fig.savefig(FIG/'supervisory_flow_balances.png', dpi=220, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 3.0), layout='constrained')
    for ax, index, title in zip(axes, (6, 0, 1), ('(a) Exterior queue', '(b) Region 1', '(c) Region 2')):
        for name in POLICIES:
            r = runs[name]
            ax.plot(r['t'], r['x'][:, index], color=COLORS[name],
                    ls={'nominal_mpc':'--','grid_repair':'-','continuation':':'}[name],
                    lw=1.9 if name!='continuation' else 1.3, label=LABELS[name])
        ax.set_title(title, loc='left')
        ax.set_xlim(20, 62)
        ax.set_xticks([20, 40, 60])
        ax.set_xlabel('Time (min)')
        ax.grid(axis='y', color='#E3E7EA', lw=.6)
    axes[0].set_ylabel('Vehicles')
    axes[0].axhline(2000, color='#50565C', ls='-.', lw=1)
    axes[0].text(21, 2035, 'Storage limit: 2000', fontsize=7)
    axes[0].annotate('2180', (60, runs['nominal_mpc']['x'][-1, 6]), xytext=(44, 2400),
                     fontsize=8, color=COLORS['nominal_mpc'], arrowprops={'arrowstyle':'-', 'color':COLORS['nominal_mpc']})
    axes[0].annotate('1683', (60, runs['grid_repair']['x'][-1, 6]), xytext=(44, 1250),
                     fontsize=8, color=COLORS['grid_repair'], arrowprops={'arrowstyle':'-', 'color':COLORS['grid_repair']})
    axes[0].set_ylim(0, 2700)
    # This is a deadline-only target, so mark the endpoint rather than a continuous line.
    axes[1].scatter([60], [2850], marker='_', s=220, color='#222222', zorder=8)
    axes[1].annotate('2850 at 60 min', (60, 2850), xytext=(22, 3170), fontsize=7,
                     arrowprops={'arrowstyle':'-', 'color':'#555555'})
    axes[1].text(23, 2920, 'Unchecked endpoint: 2852.2', color=COLORS['nominal_mpc'], fontsize=6.8)
    axes[1].set_ylim(1500, 3300)
    axes[2].set_ylim(1150, 2600)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3,
               frameon=False, fontsize=8)
    fig.savefig(FIG/'supervisory_vehicle_counts.pdf', bbox_inches='tight')
    fig.savefig(FIG/'supervisory_vehicle_counts.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
    summary = dict(case='D027', contract=record['contract'], selected_repair_weight=record['selected_theta'],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        initial_state_units=(np.asarray(fixture()['center'])+.01).tolist(),
        residual_profile='constant_high',
        integration='DOP853; rtol=1e-12, atol=1e-13; maximum step 5 seconds; output every second',
        scope='Finite-plan matched replay at minutes 20--60; certificates are separately stored in D027.json.',
        policies={name:r['result'] for name,r in runs.items()})
    (OUT/'validation.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
