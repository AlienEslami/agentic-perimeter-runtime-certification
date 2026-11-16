"""Redraw saved results and labels; no new simulations or planning."""
from pathlib import Path
import json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'output/package/PERIMETER_PARTB_CONSOLIDATED_2026-11-12'
(PROJECT/'figures').mkdir(parents=True,exist_ok=True)
OUT=ROOT/'output/review/results_refinement_2026_11_15'
OUT.mkdir(parents=True,exist_ok=True)

# Reconstruct the same stored certificate, reading the previously validated
# sampled envelope rather than integrating the traffic dynamics again.
code=(ROOT/'scripts/plot_partb_traffic_results_2026_11_15.py').read_text()
code=code[:code.index('# Existing paired repair performance')]
code=code.replace("OUT=ROOT/'output/review/traffic_results_2026_11_15'", "OUT=ROOT/'output/review/results_refinement_2026_11_15'")
code=code.replace("DATA=ROOT/'benchmarks/partb_traffic_presentation_2026-11-15'", "DATA=OUT/'plot_checks'")
a=code.index('# Regenerate sampled envelopes only'); b=code.index('traces={n:',a)
code=code[:a]+r'''
recdir=ROOT/'benchmarks/reviewer_strengthening_2026-11-11/consolidated_recovery'
plans={n:read(recdir/f'{n}_ensemble.json') for n in ['proposal','repair','backup']}
envelope=np.loadtxt(ROOT/'benchmarks/partb_traffic_presentation_2026-11-15/recovery_sampled_envelope.csv',delimiter=',',skiprows=1)
for j,key in enumerate(['final_n1','final_n2','final_queue']):
 st=plans['repair']['statistics'][key]
 assert abs(envelope[-1,1+j]-st['min'])<1e-7
 assert abs(envelope[-1,4+j]-st['max'])<1e-7
for row in envelope:
 j=min(int((row[0]+1e-10)*240),239);d=draw[j];s=row[0]-d['left'];p=100*(d['center']+d['vel']*s+d['acc']*s*s)
 assert all(row[1:3]>=p[:2]-d['rad'][:2]-1e-6)
 assert all(row[4:6]<=p[:2]+d['rad'][:2]+1e-6)
 assert row[6]<=d['qhi']+1e-5

'''+code[b:]
code=code.replace("'repair':'Modified'","'repair':'Repaired plan'")
code=code.replace('Modified endpoints','Repaired plan')
code=code.replace('(5600,10),fontsize=7.5','(5000,5),fontsize=7.5')
code=code.replace('(modified)','(repaired plan)')
code=code.replace("inset.set_title('Peak: 2005.6',fontsize=7);inset.set_facecolor('white')",
 "inset.set_facecolor('white')\ninset.tick_params(axis='x',labelbottom=False,length=0)\ninset.text(.04,.96,'Peak: 2005.6',transform=inset.transAxes,va='top',fontsize=6)\ninset.text(.97,.06,'59.7-60 min',transform=inset.transAxes,ha='right',va='bottom',fontsize=5.8)")
code+='\ncommon.save(DATA/"figure_validation.json",validation)\n'
exec(compile(code,__file__,'exec'),{'__file__':__file__,'__name__':'__main__'})

# Compact, one-panel evidence for the coordinate contribution.
path=ROOT/'benchmarks/manuscript_review_2026-11-08/results/coordinate_comparison.json'
rows=json.loads(path.read_text())['rows']
methods=['physical_box','class_box','conservation_box','positive_queue','signed']
labels=['State intervals','Destination intervals','Conservation intervals',
        'Nonnegativity and queue bounds','Signed conservation']
errors=[1,5,10,25,50,100]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(8.1,2.5))
for i,method in enumerate(methods):
 for j,d in enumerate(errors):
  row=next(r for r in rows if r['method']==method and r['d']==d)
  ax.scatter(j,i,marker='o' if row['accepted'] else 'x',s=52,
             c='#007a80' if row['accepted'] else '#8f9ca8',linewidths=1.3)
ax.set(xlim=(-.4,5.4),ylim=(4.5,-.5),xticks=range(6),xticklabels=errors,
       yticks=range(5),yticklabels=labels,xlabel=r'Initial regional error bound $d$ (vehicles)')
ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=8)
ax.grid(axis='x',alpha=.13)
handles=[Line2D([],[],marker='o',ls='',color='#007a80',label='Certified'),
         Line2D([],[],marker='x',ls='',color='#8f9ca8',label='Inconclusive')]
fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=8.5)
fig.tight_layout(rect=(0,.10,1,1))
name='coordinate_acceptance_compact_2026_11_15'
fig.savefig(PROJECT/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.07)
fig.savefig(OUT/f'{name}.png',dpi=180,bbox_inches='tight',pad_inches=.07)
plt.close(fig)
(OUT/'figure_changes.json').write_text(json.dumps({'new_simulations':False,
 'coordinate_records':len(rows),
 'coordinate_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'recovery_envelope_source':'benchmarks/partb_traffic_presentation_2026-11-15/recovery_sampled_envelope.csv'},indent=2))
print('Updated repaired-plan labels and added the compact coordinate figure from saved data.')
