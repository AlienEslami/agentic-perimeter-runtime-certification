"""Replot archived results and bounds; no new simulations or certificates."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/review/terminology_figures_2026_11_12'
OUT.mkdir(parents=True, exist_ok=True)

# Reuse the validated predictor reconstruction, now for the complete hour.
source = (ROOT/'scripts/plot_writing_revision_2026_11_12.py').read_text()
source = source.replace('writing_revision_2026_11_12', 'terminology_figures_2026_11_12')
source = source.replace("steps = c['interval_steps'][-3:]", "steps = c['interval_steps']")
source = source.replace('assert right > left', 'if right == left:\n        continue\n    assert right > left')
source = source.replace('assert len(draw)==3', "assert len(draw)==240")
start = source.index("fig,ax = plt.subplots(figsize=(10.0,3.8))")
end = source.index("evidence['verification_figure']", start)
source = source[:start] + r'''
fig,ax = plt.subplots(figsize=(9.6,3.7))
for row in draw:
    t,p,r=row['minutes'],row['predictor'],row['trial']
    ax.fill_between(t,p-r,p+r,color=BLUE,alpha=.20,zorder=1,linewidth=0,edgecolor='none')
    ax.plot(t,p+r,color=BLUE,lw=.7,alpha=.8)
    ax.plot(t,p-r,color=BLUE,lw=.7,alpha=.8)
    ax.plot(t,p,color=BLUE,lw=1.7,zorder=3)
ax.axhline(3300,color=TEAL,ls='--',lw=1.1)
ax.errorbar([60],[endpoint_center],yerr=[[endpoint_radius],[endpoint_radius]],
            fmt='o',color=TEAL,ecolor=TEAL,capsize=4,elinewidth=1.7,ms=3,zorder=5)
ax.plot([60],[3300],marker='D',ms=6,mfc='white',mec=TEAL,mew=1.5,zorder=6)
ax.annotate(f'Endpoint upper bound: {stored_upper:,.1f}',xy=(60,stored_upper),
            xytext=(59,3700),ha='right',fontsize=9,color=TEAL,
            arrowprops=dict(arrowstyle='-',color=TEAL,lw=.8))
ax.set(xlim=(0,61.5),ylim=(0,5500),ylabel='Region 1 accumulation (vehicles)',
       xlabel='Time (minutes)',xticks=[0,10,20,30,40,50,60],yticks=[0,1000,2000,3000,4000,5000])
ax.grid(axis='y',alpha=.14)
ax.legend(handles=[Line2D([],[],color=BLUE,lw=1.7,label=r'Predictor $p_{1l}(t)$'),
                   Patch(facecolor=BLUE,alpha=.20,label='Certified state bounds'),
                   Line2D([],[],color=TEAL,ls='--',lw=1.1,label='Target at 60 min: 3300 vehicles')],
          loc='upper right',fontsize=9,frameon=False)
fig.tight_layout()
save(fig,'verification_enclosure_2026_11_12')

''' + source[end:]
source = source.replace("'Nonnegativity/queue'", "'Nonnegativity and queue bounds'")
source = source.replace("figsize=(11.1,4.0)", "figsize=(12.1,4.1)")
source = source.replace("'method_labels':dict(zip(methods,labels))", "'method_labels':dict(zip(methods,labels))")
source = source.replace("axs[0].legend(loc='lower right',frameon=False,fontsize=9)",
    "axs[0].legend(loc='upper left',bbox_to_anchor=(.02,.99),frameon=True,facecolor='white',edgecolor='none',framealpha=.95,fontsize=9)")
source = source.replace("inset_axes([.04,.61,.40,.30])", "inset_axes([.70,.04,.26,.22])")
source = source.replace("inset.text(59.67,2009.5", "inset.text(59.67,2008.5")
source = source.replace("f'Peak {overflow:.1f}'", "f'Peak {overflow:.1f} (+{overflow-2000:.1f})'")
source = source.replace("inset.set_title(f'Overflow: {overflow-2000:.1f} vehicles',fontsize=8,pad=3)", '')
source = source.replace("print(json.dumps({k:v for k,v in evidence.items()},indent=2))", "print('Stored-certificate plot and comparison plots regenerated.')")
(OUT/'reconstructed_plot_source.py').write_text(source,encoding='utf-8')
exec(compile(source, str(OUT/'reconstructed_plot_source.py'), 'exec'), {'__file__':__file__,'__name__':'__main__'})

# Keep the original trajectory values; identify reference lines in legends.
source = (ROOT/'scripts/plot_congested_presentation_2026_11_11.py').read_text()
source = source.replace("ROOT / 'tmp/pdfs/section5_presentation_2026_11_11'", "ROOT / 'output/review/terminology_figures_2026_11_12'")
source = source.replace("OUT / 'figure_validation.json'", "OUT / 'congested_figure_validation.json'")
for old,new in [
 ('benchmarks/congested_recovery_2026-11-10/results/proposal_d25_w300_trace.csv','benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery/proposal_trace.csv'),
 ('benchmarks/congested_diagnostics_2026-11-10/results/wide_repair_trace.csv','benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery/repair_trace.csv'),
 ('benchmarks/congested_recovery_2026-11-10/results/backup_d25_w300_trace.csv','benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery/backup_trace.csv')]:
    assert old in source
    source = source.replace(old,new)
source = source.replace("ax.annotate('2310'", "ax.annotate('2610'")
source = source.replace('xytext=(44, 2460)', 'xytext=(42, 2660)')
source = source.replace('ax.set_ylim(350, 2750)', 'ax.set_ylim(350, 3000)')
source = source.replace('wide_uncertainty_recovery', 'congested_recovery_w600')
old = """    ax.text(2, bound + 65, 'Storage limit: 2000' if field == 'z_vehicles' else
            'Critical accumulation', fontsize=8, color='#343A40')"""
new = """    if field == 'z_vehicles':
        ax.text(2, bound + 65, 'Storage limit: 2000', fontsize=8, color='#343A40')
    else:
        references = [Line2D([], [], color='#343A40', ls='--', lw=1,
                              label='Critical accumulation')]
        if field == 'n1_vehicles':
            references.append(Line2D([], [], color='#343A40', marker='D',
                mfc='white', ls='', ms=4, label='Target at 60 min: 3300'))
        ax.legend(handles=references, loc='upper right', fontsize=7.2,
                  frameon=False, handlelength=1.8)"""
assert old in source
source = source.replace(old,new)
old = """        ax.annotate('Deadline: 3300', (60, 3300), xytext=(29, 2800), fontsize=7.7,
                    arrowprops={'arrowstyle': '-', 'color': '#343A40', 'lw': .7})"""
assert old in source
source = source.replace(old,'')
exec(compile(source, __file__, 'exec'), {'__file__':__file__,'__name__':'__main__'})

# Redraw only the existing six snapshots. Counts, inputs and flows are unchanged.
source = (ROOT/'scripts/numerical_section_figures_2026_11_10.py').read_text()
source = source.split('snapshots();snapshots(True)')[0] + '\nsnapshots()\n'
source = source.replace("DATA=ROOT/'output/review/numerical_section_2026-11-10';DATA.mkdir(exist_ok=True)",
    "DATA=ROOT/'output/review/terminology_figures_2026_11_12';DATA.mkdir(exist_ok=True)")
source = source.replace("1.96,f'Region {j+1}'", "2.00,f'Region {j+1}'")
source = source.replace("fontsize=14,weight='bold'", "fontsize=11.5,weight='bold'")
source = source.replace("1.68,f'{x[j]:,.0f}',ha='center',fontsize=18", "1.68,f'{x[j]:,.0f}',ha='center',va='center',fontsize=15")
source = source.replace("1.96,'Queue',ha='center'", "2.00,'Queue',ha='center',va='center'")
source = source.replace("1.68,f'{x[6]:,.0f}',ha='center',fontsize=16", "1.68,f'{x[6]:,.0f}',ha='center',va='center',fontsize=14")
exec(compile(source, __file__, 'exec'), {'__file__':__file__,'__name__':'__main__'})
old = json.loads((ROOT/'output/review/numerical_section_2026-11-10/supervisory_snapshots.json').read_text())
new = json.loads((OUT/'supervisory_snapshots.json').read_text())
assert old == new, 'Snapshot values changed'
print('All plots use unchanged saved data; snapshot values match exactly.')
