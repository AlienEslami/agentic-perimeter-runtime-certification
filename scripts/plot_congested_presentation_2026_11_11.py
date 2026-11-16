"""Present stored trajectories with the literature MFD; no simulation or fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'output/package/PERIMETER_PARTB_PRE_SIMULATION_2026-11-10'
OUT = ROOT / 'tmp/pdfs/section5_presentation_2026_11_11'
OUT.mkdir(parents=True, exist_ok=True)
PATHS = {
    'proposal': ROOT / 'benchmarks/congested_recovery_2026-11-10/results/proposal_d25_w300_trace.csv',
    'repair': ROOT / 'benchmarks/congested_diagnostics_2026-11-10/results/wide_repair_trace.csv',
    'backup': ROOT / 'benchmarks/congested_recovery_2026-11-10/results/backup_d25_w300_trace.csv',
}
hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in PATHS.items()}
data = {k: np.genfromtxt(p, delimiter=',', names=True) for k, p in PATHS.items()}
COLORS = {'proposal': '#B5472B', 'repair': '#007E87', 'backup': '#737C85'}
STYLES = {'proposal': '--', 'repair': '-', 'backup': ':'}
LABELS = {'proposal': 'Proposed plan (unchecked)', 'repair': 'Repaired plan', 'backup': 'Backup plan'}
critical = float(min(np.roots([3 * 1.4877e-7, -2 * .0029815, 15.0912])))
def outflow(n):
    return 1.4877e-7 * n**3 - .0029815 * n**2 + 15.0912 * n

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.5))
for ax, field, title in zip(axs.flat, ['n1_vehicles', 'n2_vehicles', 'z_vehicles'],
        ['(a) Region 1', '(b) Region 2', '(c) Exterior queue']):
    for key, trajectory in data.items():
        ax.plot(trajectory['time_h'] * 60, trajectory[field], color=COLORS[key],
                ls=STYLES[key], lw=1.8)
    bound = 2000 if field == 'z_vehicles' else critical
    ax.axhline(bound, color='#343A40', ls=(0, (5, 3)), lw=1)
    ax.text(2, bound + 65, 'Storage limit: 2000' if field == 'z_vehicles' else
            'Critical accumulation', fontsize=8, color='#343A40')
    ax.set(xlim=(0, 61), xlabel='Time (min)', ylabel='Vehicles', title=title)
    ax.set_title(title, loc='left', pad=7)
    ax.set_title('')
    ax.set_xticks([0, 15, 30, 45, 60])
    ax.grid(axis='y', alpha=.22, lw=.5)
    if field == 'n1_vehicles':
        ax.plot(60, 3300, marker='D', ms=5, mfc='white', mec='#343A40', clip_on=False)
        ax.annotate('Deadline: 3300', (60, 3300), xytext=(29, 2800), fontsize=7.7,
                    arrowprops={'arrowstyle': '-', 'color': '#343A40', 'lw': .7})
    elif field == 'z_vehicles':
        ax.set_ylim(350, 2750)
        ax.annotate('2310', (60, data['proposal'][field][-1]), xytext=(44, 2460),
                    fontsize=8, color=COLORS['proposal'],
                    arrowprops={'arrowstyle': '-', 'color': COLORS['proposal'], 'lw': .7})

ax = axs[1, 1]
n = np.linspace(0, 10000, 600)
ax.plot(n, outflow(n) / 1000, color='#343A40', lw=1.5)
ax.axvline(critical, color='#74818e', ls=':', lw=.9)
ax.text(critical + 180, 3, 'Critical', color='#586574', fontsize=7.5)
initial = float(data['repair']['n1_vehicles'][0])
assert initial == float(data['repair']['n2_vehicles'][0])
ax.plot(initial, outflow(initial) / 1000, 'o', color='#343A40', ms=4)
ax.annotate('Initial state', (initial, outflow(initial) / 1000), xytext=(6100, 24),
            fontsize=8, arrowprops={'arrowstyle': '-', 'color': '#343A40', 'lw': .7})
endpoints = [float(data['repair'][key][-1]) for key in ['n1_vehicles', 'n2_vehicles']]
ax.scatter(endpoints, [outflow(x) / 1000 for x in endpoints], color=COLORS['repair'], s=19, zorder=5)
for end in endpoints:
    ax.annotate('', (end, outflow(end) / 1000), xytext=(1450, 12.2),
                arrowprops={'arrowstyle': '-', 'color': COLORS['repair'], 'lw': .6})
ax.text(600, 11.6, 'Repair at minute 60\n(both regions)', color=COLORS['repair'],
        fontsize=7.5, va='top')
ax.annotate('', (3000, outflow(3000) / 1000), xytext=(4200, outflow(4200) / 1000),
            arrowprops={'arrowstyle': '-|>', 'color': COLORS['repair'], 'lw': 1.5,
                        'connectionstyle': 'arc3,rad=.13'})
ax.set(xlim=(0, 10000), ylim=(0, 26), xlabel='Accumulation (vehicles)',
       ylabel=r'Outflow ($10^3$ veh/h)', xticks=[0, 5000, 10000], yticks=[0, 10, 20])
ax.set_title('(d) MFD and regional recovery', loc='left', pad=7)
ax.grid(axis='y', alpha=.22, lw=.5)
handles = [Line2D([0], [0], color=COLORS[k], ls=STYLES[k], lw=1.8, label=LABELS[k]) for k in PATHS]
ax.get_xticklabels()[-1].set_horizontalalignment('right')
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015),
           ncol=3, frameon=False, fontsize=8.2, columnspacing=1.2)
fig.subplots_adjust(left=.10, right=.98, bottom=.15, top=.95, wspace=.29, hspace=.47)
fig.savefig(PROJECT / 'figures/wide_uncertainty_recovery.pdf')
fig.savefig(OUT / 'wide_uncertainty_recovery.png', dpi=180)
plt.close(fig)
assert hashes == {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in PATHS.items()}
(OUT / 'figure_validation.json').write_text(json.dumps({
    'inputs_unchanged': hashes, 'critical_accumulation_vehicles': critical,
    'initial_accumulation_vehicles': initial, 'repair_regional_endpoints_vehicles': endpoints,
    'simulation_rerun': False,
}, indent=2), encoding='utf-8')
print('Updated congested-recovery figure from unchanged saved trajectories.')
