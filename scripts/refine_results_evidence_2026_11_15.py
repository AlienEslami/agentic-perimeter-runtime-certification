"""Summarize archived experiments and clarify results without changing theory."""
from pathlib import Path
from collections import defaultdict
import json, re, hashlib, math
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT/'output/package/PERIMETER_PARTB_CONSOLIDATED_2026-11-12'
ARCHIVE=ROOT/'output/archive/PERIMETER_BEFORE_RESULTS_REFINEMENT_2026-11-15'
OUT=ROOT/'output/review/results_refinement_2026_11_15'
OUT.mkdir(parents=True,exist_ok=True)
SUITE=ROOT/'benchmarks/reusable_perimeter_suite_2026-09-26/results/contracts'
read=lambda p: json.loads(p.read_text(encoding='utf-8'))
old=(ARCHIVE/'main.tex').read_text(encoding='utf-8');s=old
evidence={'sources':{},'new_simulations':False}
def replace(a,b):
    global s
    assert s.count(a)==1,(a[:100],s.count(a))
    s=s.replace(a,b)
def getfloat(label,kind='table'):
    a=s.index('\\label{'+label+'}')
    begin=s.rfind('\\begin{'+kind+'}',0,a)
    end=s.index('\\end{'+kind+'}',a)+len('\\end{'+kind+'}')
    return s[begin:end]
def tracked(path):
    evidence['sources'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return read(path)

# The same traffic request at two available state-error bounds.
errorrows=[]
for case in ['D027','D063']:
    d=tracked(SUITE/(case+'.json'));pol=d['policies']['grid_repair']
    c=d['contract'];cert=pol['certificate'];samples=pol['realizations']
    assert c['weight']==4 and c['residual_veh_h']==300 and c['deadline_vehicles']==2850
    assert cert['accepted'] and len(samples)==640
    assert all(v['physical_pass'] and v['service_pass'] for v in samples)
    upper=100*cert['service_check_results'][0]['upper']
    assert upper<2850
    errorrows.append(dict(case=case,d=c['width_vehicles'],weight=d['selected_theta'],
        certified_deadline_upper=upper,simulated_max_queue=max(v['peak_queue_vehicles'] for v in samples),
        both_met=640,commands=pol['commands']))
for field in ['nominal_mpc','continuation']:
    one=read(SUITE/'D027.json')['policies'][field]['commands']
    other=read(SUITE/'D063.json')['policies'][field]['commands']
    assert np.max(np.abs(np.array(one)-other))<1e-12
evidence['larger_uncertainty_comparison']=errorrows

replace('second block ending at minute~60. The uncertainty bounds and service',
r'''second block ending at minute~60. This common two-block parameterization
keeps the proposed, repaired and backup schedules directly comparable.
The theory permits any finite block partition, but these experiments do
not evaluate schedules updated every few minutes. An accepted replacement
starts a new schedule. The uncertainty bounds and service''')
replace('nominal input using state feedback to enforce physical constraints.\nRepair tests',
r'''nominal input using state feedback to enforce physical constraints.
The regional service deadline is not included in this QP and is evaluated
separately. Repair tests''')
replace('between proposal and certified backup. The network-efficiency objective',
r'''between proposal and certified backup. A modified proposal accepted by
this procedure is called the \emph{repaired plan}. The network-efficiency objective''')
replace('The modified plan increases the entry input\nand the initial transfer from Region~1 to Region~2. More arriving\nvehicles enter the network, while both regions recover to accumulations\nbelow the critical value.',
r'''The repaired plan raises the entry input after minute~5 from 0.525 to
approximately 0.583. Capacity-limited admission therefore increases by
about 694~veh/h, reducing the queue at minute~60 by about 637 vehicles
under the same arrivals. It also raises the initial transfer input
$u_{12}$ from approximately 0.466 to 0.813. More arriving vehicles enter
the network, while both regions recover to accumulations below the
critical value.''')
replace('but violates requirements in part of the set. CBF-QP keeps the physical\nconstraints satisfied in every simulation, yet misses the regional\ndeadline in all of them.',
r'''but violates requirements in part of the set. CBF-QP keeps the physical
constraints satisfied in every simulation. It misses the regional
deadline in all of them; that requirement is outside its formulation.
This comparison shows the additional service
requirement addressed by the supervisor.''')
replace('program enforcing physical constraints.}',
r'''program enforcing physical constraints. Its deadline is an evaluation
criterion and is not included in the QP.}''')
replace('CBF-QP & 3238 & 6293 & 1844 & 2918 & 0/640 & Violation shown',
        r'CBF-QP & 3238 & 6293 & 1844 & 2918 & 0/640 & Physical limits met; deadline missed')

error_table=r'''\paragraph{Larger state-estimation errors}
The same nominal proposal and backup also admit a certified repair with
$d_x=25\mathbf1_7$ vehicles, retaining $p=4$, $\bar w=300$~veh/h and
the 2850-vehicle deadline. Table~\ref{tab:repair-error-comparison}
compares the two cases. The selected repair places greater weight on
backup for the larger error bound, while its deadline bound remains below the
target. These are tested error bounds and require calibration for a
particular network.
\begin{table}[pos=!htb]
\centering\small
\caption{Repaired plans for the same regional-priority request under
different componentwise state-error bounds $d_x=d\mathbf1_7$. The
proposal, backup, arrival-error bound of 300~veh/h, priority weight
$p=4$ and minute-60 deadline of 2850 vehicles are unchanged.
The certified deadline upper bound is rounded upward; maximum queue
is taken over 640 simulations and rounded to the nearest vehicle.
Both plans satisfy physical and service requirements in all 640
simulations and have full-interval certificates.}
\label{tab:repair-error-comparison}
\begin{tabularx}{\linewidth}{@{}Xrrr@{}}
\toprule
\shortstack{Error bound $d$\\(vehicles)} &
\shortstack{Repair weight\\$\lambda_{\rm rep}$} &
\shortstack{Certified deadline\\upper bound (vehicles)} &
\shortstack{Maximum simulated\\queue (vehicles)}\\
\midrule
'''
for r in errorrows:
    error_table+=f"{r['d']:.0f} & {r['weight']:.2f} & {math.ceil(r['certified_deadline_upper'])} & {r['simulated_max_queue']:.0f}"+r'\\'+'\n'
error_table+=r'''\bottomrule
\end{tabularx}
\end{table}

'''
replace(r'\paragraph{Traffic costs across regional-priority scenarios}',error_table+r'\paragraph{Traffic costs across regional-priority scenarios}')

# Compact traffic-cost statistics replace the three single-box panels.
repair=tracked(ROOT/'benchmarks/partb_traffic_presentation_2026-11-15/repair_traffic_statistics.json')
q=np.array(repair['values']);assert repair['count']==24320
trade=r'''\begin{table}[pos=!htb]
\centering\small
\caption{Traffic effects of 38 repaired plans over 24,320 matched
simulations. Changes are relative to the corresponding backup. Benefit
retained is $100B_\omega$, relative to the proposed priority improvement.
All entries are percentages. Percentiles describe the prescribed
simulation set, not a probability model for traffic uncertainty.}
\label{tab:repair-traffic-tradeoffs}
\begin{tabularx}{\linewidth}{@{}Xrr@{}}
\toprule
Traffic measure & Median & 5th--95th percentiles\\
\midrule
'''
for j,name in enumerate(['Priority-cost reduction','TTS increase','Requested priority benefit retained']):
    trade+=f'{name} & {q[2,j]:.1f} & {q[0,j]:.1f}--{q[4,j]:.1f}'+r'\\'+'\n'
trade+=r'''\bottomrule
\end{tabularx}
\end{table}'''
replace(getfloat('fig:repair-traffic-tradeoffs','figure'),trade)
s=s.replace(r'Figure~\ref{fig:repair-traffic-tradeoffs}',r'Table~\ref{tab:repair-traffic-tradeoffs}')

# Pair each CLF simulation with the same state corner/profile under MPC.
arrival_rows=defaultdict(list)
for path in sorted(SUITE.glob('Q*.json')):
    d=tracked(path);c=d['contract'];nom=d['policies']['nominal_mpc']
    assert nom['certificate']['accepted']
    ix={(v['corner'],v['profile']):v for v in nom['realizations']}
    for v in d['clf_extension']['realizations']:
        n=ix[(v['corner'],v['profile'])]
        assert v['completed'] and n['physical_pass']
        arrival_rows[c['amplitude']].append([n['tts_veh_h'],v['tts_veh_h'],c['duration_h']])
eff=r'''\begin{table}[pos=!htb]
\centering\small
\caption{Network-efficiency comparison grouped by exterior-arrival
increase. Each row pools three durations (30, 45 and 60 minutes),
sixteen initial-state corners and five arrival-error profiles, giving
240 matched simulations. TTS is evaluated over minutes 20--60 and
reported as a median in veh-h. The last column is the median percentage
reduction computed separately for each matched pair. CLF denotes the
locally extended controller described in Appendix~\ref{app:experimental-settings}.
All MPC plans are certified without modification.}
\label{tab:arrival-efficiency}
\begin{tabularx}{\linewidth}{@{}Xrrr@{}}
\toprule
Arrival increase & MPC TTS & CLF TTS & Paired reduction (\%)\\
\midrule
'''
eff_evidence=[]
for amp,vals in sorted(arrival_rows.items()):
    ar=np.array(vals);assert len(ar)==240 and len(set(ar[:,2]))==3
    med=np.median(ar[:,:2],axis=0);gain=np.median(100*(ar[:,1]-ar[:,0])/ar[:,1])
    eff+=f'{100*amp:.0f}\\% & {med[0]:.0f} & {med[1]:.0f} & {gain:.1f}'+r'\\'+'\n'
    eff_evidence.append(dict(arrival_increase=amp,samples=240,median_mpc_tts=med[0],median_clf_tts=med[1],median_paired_reduction=gain))
eff+=r'''\bottomrule
\end{tabularx}
\end{table}'''
evidence['network_efficiency_by_arrival_increase']=eff_evidence
replace('All MPC proposals\npass unchanged. Over 960 matched simulations, median paired TTS is\n24\\% lower than with the locally implemented extension of Zhong\net al.\'s CLF controller~\\cite{zhong2018robust}. This gain is produced\nby planning and preserved by verification.',
r'''All MPC proposals pass unchanged. Table~\ref{tab:arrival-efficiency}
shows median paired TTS reductions of 23.5--24.5\% across the four arrival
increases, relative to the locally implemented extension of Zhong
et al.'s CLF controller~\cite{zhong2018robust}. Across all 960 matched
simulations, the median reduction is 24\%. This gain is produced by
planning and preserved by verification.''')
needle='given in \\appref{app:experimental-settings}.\n\n\\subsection{Changing requests and delayed implementation}'
replace(needle,'given in \\appref{app:experimental-settings}.\n\n'+eff+'\n\n\\subsection{Changing requests and delayed implementation}')

replace(r'\paragraph{Interpretation of operator priorities}',r'\subsection{Interpretation of operator requests}')

# A short outcome panel shows the coordinate contribution without duplicating numerical detail.
replace('methods (Figure~\\ref{fig:structure-comparison}). The plan starts at',
        'methods (Figure~\\ref{fig:coordinate-acceptance}). The plan starts at')
compact=r'''\begin{figure}[pos=!htb]
\centering
\includegraphics[width=0.86\linewidth]{figures/coordinate_acceptance_compact_2026_11_15.pdf}
\caption{Effect of the bounding method for the same fixed
network-efficiency plan. Circles mark complete certificates and crosses
mark inconclusive checks at the tested regional initial-error bounds
$d$. Queue error is bounded by $\min(d,10)$ vehicles, arrival error by
60~veh/h, and Region~1 must contain at most 2850 vehicles at minute 60.
Signed conservation certifies $d=50$ vehicles, whereas the passing
interval methods reach $d=10$. An inconclusive check does not establish
a traffic violation. Figure~\ref{fig:structure-comparison} gives the
corresponding numerical upper bounds.}
\label{fig:coordinate-acceptance}
\end{figure}

'''
replace('is capacity-limited.\n\nIn the congested case,', 'is capacity-limited.\n\n'+compact+'In the congested case,')

# Keep experiment protocols in B and move the four outcome tables to G.
labels=['tab:benchmark-deadlines','tab:repair-outcomes','tab:activation-delay-results','tab:benchmark-api-execution']
moved={label:getfloat(label) for label in labels}
for block in moved.values():replace(block,'')
replace('This appendix gives the settings and detailed counts supporting',
        'This appendix gives the settings supporting')
replace('report the unchanged-proposal and repair counts. Validation scenarios',
        'in \\appref{app:additional-numerics} report the unchanged-proposal and\nrepair counts. Validation scenarios')
replace('\\appref{app:agent-evaluation} gives the protocol and\nTable~\\ref{tab:benchmark-api-execution}, while',
        '\\appref{app:agent-evaluation} gives the protocol,\nTable~\\ref{tab:benchmark-api-execution} reports the outcomes, and')
appendix=r'''\paragraph{Detailed outcomes of the numerical studies}
Tables~\ref{tab:benchmark-deadlines}--\ref{tab:benchmark-api-execution}
give the proposal, repair, delayed-implementation and agent outcomes
supporting Section~\ref{sec:new-evidence}. The corresponding protocols
and settings are in \appref{app:experimental-settings}.

'''+ '\n\n'.join(moved.values())+'\n\n'
replace('\\label{app:additional-numerics}\n\n','\\label{app:additional-numerics}\n\n'+appendix)

# Result labels refer to the accepted output as a repaired plan.
start=s.index(r'\section{Numerical Experiments}')
tail=s[start:]
for a,b in [('proposed, modified and','proposed, repaired and'),("modified plan's","repaired plan's"),('modified plan','repaired plan'),(' & Modified &',' & Repaired &'),('Nominal-plan repair &','Repaired plan &')]:tail=tail.replace(a,b)
tail=tail.replace('Simulation passes require all 640\nrealizations','Simulation passes require all 640\nsimulations')
s=s[:start]+tail

# Unchanged mathematical content and front matter.
assert s[:start]==old[:old.index(r'\section{Numerical Experiments}')]
for env in ['theorem','lemma','proposition','corollary','proof']:
    parse=lambda t:re.findall(r'\\begin\{'+env+r'\}.*?\\end\{'+env+r'\}',t,re.S)
    assert parse(s)==parse(old),env
assert s[s.index(r'\section{Conclusions}'):s.index(r'\section*{Author contributions')]==old[old.index(r'\section{Conclusions}'):old.index(r'\section*{Author contributions')]
assert r'\FloatBarrier' not in s
assert '\x0c' not in s and '\x08' not in s and '\x0b' not in s and '\r' not in s
(PROJECT/'main.tex').write_text(s,encoding='utf-8')
evidence.update(source_sha256=hashlib.sha256((PROJECT/'main.tex').read_bytes()).hexdigest(),all_formal_results_and_proofs_unchanged=True,
                front_matter_sections2_to5_unchanged=True,conclusion_future_work_unchanged=True)
(OUT/'evidence_and_source_checks.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print('Updated numerical text and tables from archived data. Formal mathematics unchanged.')
