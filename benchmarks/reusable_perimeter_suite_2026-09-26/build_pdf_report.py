"""Render the reviewed benchmark summary using bundled PDF dependencies."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/"results/summary.json").read_text())
ROOT=HERE.parents[1]
OUTPUT=ROOT/"output/pdf/PERIMETER_BENCHMARK_SWEEP_2026-09-26.pdf"
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name="Text",fontName="Helvetica",fontSize=10.3,leading=15,spaceAfter=9,textColor=colors.HexColor("#243746")))
styles.add(ParagraphStyle(name="SmallText",parent=styles["Text"],fontSize=9,leading=12,spaceAfter=5))
styles.add(ParagraphStyle(name="TitleText",fontName="Helvetica-Bold",fontSize=23,leading=28,spaceAfter=15,textColor=colors.HexColor("#123B55")))
styles.add(ParagraphStyle(name="HeadingText",fontName="Helvetica-Bold",fontSize=15,leading=19,spaceAfter=12,textColor=colors.HexColor("#123B55")))
styles.add(ParagraphStyle(name="Cell",fontName="Helvetica",fontSize=8.5,leading=11))
styles.add(ParagraphStyle(name="HeaderCell",fontName="Helvetica-Bold",fontSize=8.5,leading=11,textColor=colors.white))
story=[]

def p(text,style="Text"):
    story.append(Paragraph(text,styles[style]))
def h(text):p(text,"HeadingText")
def page():story.append(PageBreak())
def table(headers,rows,widths):
    values=[[Paragraph(escape(str(v)),styles["HeaderCell"]) for v in headers]]
    values.extend([[Paragraph(escape(str(v)),styles["Cell"]) for v in row] for row in rows])
    obj=Table(values,colWidths=widths,repeatRows=1,hAlign="LEFT")
    obj.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#123B55")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#F1F5F8"),colors.white]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),
        ("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),
        ("BOTTOMPADDING",(0,0),(-1,-1),7),("LINEBELOW",(0,0),(-1,0),.5,colors.HexColor("#123B55")),
        ("LINEBELOW",(0,-1),(-1,-1),.5,colors.HexColor("#CAD5DE"))]))
    story.extend([obj,Spacer(1,12)])
def percent_range(s):return f"{s['minimum']:.1f}-{s['maximum']:.1f}% (median {s['median']:.1f}%)"

d=DATA["deadline_all"];q=DATA["queue"];r=DATA["regulation"];fs=d["full_scenario"]
p("Reusable perimeter-control<br/>benchmark: complete sweep","TitleText")
p("Results and manuscript recommendations | 26 September 2026","SmallText")
p("The current manuscript is preserved. This report evaluates a reusable family of transportation problems and keeps the original paper unchanged while the new evidence is reviewed.")
h("What the study establishes")
p(f"The checker admitted {q['nominal_certified']}/12 queue-only MPC plans without changing their commands. Under deadline and uncertainty contracts, {d['nominal_certified']}/80 nominal proposals were admitted directly; certified replacement or repair was available in {d['certified_repair_available']}/80. Each selected certified plan met the physical and service tests across its 640 corner/profile realizations.")
p("The principal value is a controlled way to replace a plan while preserving storage and a scheduled regional obligation. Repair can recover regional-priority performance from a rejected request. Queue-inclusive waiting remains an explicit cost of that operational choice.")
p(f"For contracts where the archived continuation proof did not close, a separately planned reserve was tested. This extension provided certified authority in {DATA['reserve_refresh']['all_contracts_certified_with_refresh']}/80 deadline contracts, with the original availability outcomes preserved.")
table(["Study component","Completed scope"],[["Native regional regulation","24 cases; CLF and tracking MPC"],
    ["Queue-only plan replacement","12 contracts; 20-50% arrival surges"],
    ["Deadline replanning","72 development + 8 held-out contracts"],
    ["Stronger predictive comparator","All 128 corners x five profiles + nominal center"],
    ["OpenAI evidence","96 routing calls + eight live planning traces"]],[180,328])
p(f"The constant-plan study contains {DATA['counts']['all_constant_plan_realizations']:,} realized trajectories. Feedback and API denominators are reported separately. The conservative API accounting is USD {DATA['api_cost']['accounted_upper_usd']:.3f}, within the authorized USD 10 ceiling.","SmallText")
page()
h("1. Three transportation benchmark tasks")
table(["Task","Problem and comparison"],[["R: regional regulation","Four destination classes, two regional totals and two transfer gates. Regulate both totals to 3000 vehicles. Compare reproduced almost-smooth CLF with matched tracking MPC."],
    ["Q: economic plan replacement","Seven-state nonlinear MFD, destination classes and exterior queue. At minute 20, replace a two-block plan through minute 60 to minimize queue-inclusive total time spent."],
    ["D: deadline-constrained replanning","Same model and interval; add a Region-1 terminal limit and Region-2 priority. Compare direct MPC, physical CBF, verified continuation, certified repair and scenario planning."]],[140,368])
p("The seven-state model uses the literature cubic MFD and the manuscript's fixed synthetic internal-demand construction. Regional storage is 10,000 vehicles; exterior storage is 2,000 vehicles. Gates lie in [0,1], with the stated 0.75 directional limits. Offered arrivals are retained in the queue when admission is restricted.")
p("The replacement task starts from one inherited minute-20 information state. Its box half-width and arrival-residual bound are varied; its verified continuation is independently checked under each new contract. All plans share two command blocks: five minutes and the remaining 35 minutes.")
table(["Deadline grid","Development","Held-out combinations"],[["Sensing half-width","1, 10, 25, 50 vehicles","5, 20 vehicles"],
    ["Arrival residual","60, 150, 300 veh/h","100, 225 veh/h"],
    ["Region-1 deadline","2750, 2850, 2950 vehicles","2800, 2900 vehicles"],
    ["Region-2 weight","2, 4","3"]],[158,175,175])
p("Every constant plan is replayed from 128 initial-box corners under zero, constant-high, constant-low, five-minute alternating and ten-minute sinusoidal residuals. The CLF queue extension uses 16 predetermined corners and the same five profiles. The CBF diagnostic uses one matched all-high corner and constant-high residual per deadline contract.","SmallText")
p("Mathematical definitions are supplied in benchmark_formulation.tex. The full protocol identifies solver settings, information choices and certificate validity intervals.","SmallText")
page()
h("2. Regulation and queue-inclusive results")
p(f"Both regulation controllers completed all 24 cases within the tested constraints. Tracking MPC reduced integrated regional regulation error by {percent_range(r['tracking_mpc_regulation_error_reduction_percent'])}. This is a controller comparison, distinct from the effect of the execution checker.")
p(f"All 12 queue-only nominal plans were certified unchanged, and all 640 realizations per plan met the tested physical rows. On matched feasible trajectories, MPC reduced TTS relative to the explicitly extended CLF by {percent_range(q['mpc_tts_reduction_vs_clf_extension_percent'])}. The comparison contains {q['matched_feasible_runs']} matched feasible feedback/plan realizations.")
h("Deadline results: complete fixed grids")
rows=[]
for label,key in (("Development","deadline_development"),("Held-out","deadline_held_out"),("Combined","deadline_all")):
    rr=DATA[key];n=rr["contracts"]
    rows.append([label,n,f"{rr['nominal_certified']}/{n}",f"{rr['nominal_all_sampled_feasible']}/{n}",f"{rr['certified_repair_available']}/{n}"])
table(["Split","Contracts","Nominal family certificate","All nominal samples feasible","Certified output with repair"],rows,[83,60,122,122,121])
p(f"Nominal MPC exhibited a sampled physical violation in {d['nominal_contracts_with_physical_violation']}/80 contracts and a sampled service violation in {d['nominal_contracts_with_service_violation']}/80. The independently checked continuation remained available in {d['backup_available']}/80 contracts.")
p(f"A fresh reserve certified in {DATA['reserve_refresh']['reserves_certified']}/{DATA['reserve_refresh']['contracts']} additional contracts. Together with the existing admitted plans and repairs, the reserve-refresh extension supplied {DATA['reserve_refresh']['all_contracts_certified_with_refresh']}/80 certified outputs.","SmallText")
p(f"The physical CBF diagnostic completed {d['cbf_completed']}/80 adverse trajectories, met the physical rows in {d['cbf_physical_pass']}/80, and met the regional deadline in {d['cbf_service_pass']}/80. Its instantaneous physical rows and the plan checker's scheduled-service rows answer different questions.")
p("These counts separate finite-scenario behavior from uncertainty-family proof acceptance. A rejected enclosure records unavailable proof for the candidate; actual violations are established by the separate trajectory results.","SmallText")
page()
h("3. A matched deadline-conflict case")
p("All rows below use the same initial all-high corner, 300 veh/h high arrival residual, 2850-vehicle Region-1 deadline and 2000-vehicle queue limit. Region 2 has weight four. The interval is minute 20 through minute 60.")
labels={"nominal_mpc":"Nominal economic MPC","continuation":"Verified continuation","grid_repair":"Certified MPC repair",
    "scenario_mpc":"Five-scenario MPC","physical_CBF":"Physical CBF-QP","full_scenario_mpc":"All-scenario MPC","full_scenario_repair":"Certified all-scenario repair"}
rows=[]
for row in DATA["primary_case"]["rows"]:
    rows.append([labels[row["policy"]],f"{row['tts_veh_h']:.1f}",f"{row['peak_queue_vehicles']:.1f}",f"{row['deadline_n1_vehicles']:.1f}",
        "Pass" if row["physical_pass"] else "Fail","Pass" if row["service_pass"] else "Fail"])
table(["Policy","TTS (veh-h)","Peak queue (veh)","Region 1 at 60 min (veh)","Physical","Service"],rows,[164,64,77,91,56,56])
p("Nominal economic MPC pursues the new priority while exceeding exterior storage and the regional deadline in this realization. The CBF correction maintains physical rows but misses the deadline. The verified continuation and the certified repairs satisfy both requirements.")
p("The all-scenario planner covers a much larger finite ensemble. Its independent checker decides whether the plan also has authority over the full bounded-uncertainty family. Repair is applied to that tool through the same interface.")
p("TTS of an infeasible row is descriptive. Feasible policies are compared on regional-priority cost and queue-inclusive waiting together. The continuation can have lower TTS while a repair better serves the updated regional objective.","SmallText")
p("The five-scenario queue excess is 0.007 vehicle, rounded in the table; adaptive replay confirms it. The nominal economic MPC excess is 180 vehicles. The full-scenario planner adds one vehicle of prediction clearance.","SmallText")
page()
h("4. Uncertainty coverage and repair")
p(f"The all-scenario planner includes 641 prediction trajectories and one vehicle of fixed planning clearance. It passed every sampled physical/service test in {fs['all_sampled_feasible']}/80 contracts; the interval checker admitted {fs['certified_directly']}/80 directly. With the same repair procedure, certified outputs were available in {fs['certified_with_repair']}/80.")
story.append(Image(str(HERE/"figures/repair_uncertainty_grid.png"),width=508,height=205))
story.append(Spacer(1,12))
p("The figure fixes the deadline at 2850 vehicles and Region-2 weight at four. Cells show the first certified point on the ordered 0.05 interpolation grid toward the verified continuation. Every preceding point is checked. The search makes no global monotonicity assumption.","SmallText")
p("Cells marked n/a indicate that this archived continuation was unavailable under the widened contract. The separately reported reserve-refresh experiment plans and checks an alternative reserve before granting authority.","SmallText")
p(f"Across repaired proposals and matched realizations, median priority-cost improvement relative to retention was {d['repair_priority_improvement_vs_retention_percent']['median']:.2f}%. Median TTS change was {d['repair_tts_change_vs_retention_percent']['median']:+.2f}%. The released results include the complete ranges and every contract.")
strong=d["stronger_tool_priority_improvement_vs_original_certified_tool_percent"]
p(f"Using the stronger scenario tool inside the same architecture changed priority cost relative to the original certified tool by a median improvement of {strong['median']:.2f}% (range {strong['minimum']:.2f}% to {strong['maximum']:.2f}%). Planning quality and certification authority remain separately measurable.")
p("The five-scenario comparator is preserved alongside the stronger study. The extra study was introduced after early cases exposed the limited coverage of the small scenario set. Its settings were then fixed across the remaining development and held-out contracts.","SmallText")
page()
h("5. Live OpenAI agent evidence")
p("The interpretation/routing pilot uses active and interrupted priorities, absolute expiry, a lower-trust poisoned traffic-tool cache, and independent checker verdicts. It evaluates 96 calls to gpt-5.4-mini. The traffic plans for this pilot are precomputed benchmark tool results.")
rows=[]
for rr in DATA["api_pilot"]["rows"]:
    rows.append(["Explicit task record" if rr["interface"]=="structured_task_record" else "History only",
        "Poisoned" if rr["attack"] else "Clean",f"{rr['objective_correct']}/{rr['calls']}",f"{rr['tool_correct']}/{rr['calls']}",rr["physical_violations"]+rr["deadline_violations"]])
table(["Interface","Memory","Correct objective","Correct tool","Authorized-route violations"],rows,[145,67,95,95,106])
h("Actual agent-to-tool execution")
p("Eight additional live traces send the model's objective request to the numerical MPC, run the unchanged interval checker, and request repair or retention after rejection. The objective remains active through the original minute-60 expiry. These traces use one common deadline contract and poisoned lower-trust cache.")
rows=[]
for rr in DATA["live_tool_traces"]["rows"]:
    rows.append(["Explicit task record" if rr["interface"]=="structured_task_record" else "History only",
        f"{rr['objectives_correct']}/{rr['traces']}",rr["certified_repairs"],"Pass" if rr["all_physical_and_service_pass"] else "Fail"])
table(["Interface","Correct objective","Certified repairs","All 640 physical/service tests"],rows,[158,103,106,141])
p("The live traces separate contextual interpretation from enforcement of encoded physical and service requirements. A semantically mistaken priority can still generate an admissible plan; the independent deadline and storage rows remain in force. Complete requests, responses, model identifiers, plans and proof records are supplied.","SmallText")
p(f"Final live traces used {DATA['live_tool_traces']['api_calls']} API calls. All study attempts, including retained development attempts, cost USD {DATA['api_cost']['known_token_cost_usd']:.4f} from known token usage; conservative accounting is USD {DATA['api_cost']['accounted_upper_usd']:.4f}. API calls stop before the USD 10 budget can be exceeded.","SmallText")
page()
h("6. Validation, paper changes and release")
p("Validation covers the six existing CLF reproduction tests, algebraic equivalence of the vectorized seven-state model, analytic-gradient checks for both predictive implementations, exact compiled actuator rows, and adaptive DOP853 replay of selected original and stronger-comparator trajectories. Replays also check vehicle conservation and queue-inclusive cost integration.")
p(f"Individual proof checks took a median {DATA['timing']['proof_seconds']['median']:.1f} seconds, with a maximum {DATA['timing']['proof_seconds']['maximum']:.1f}, under concurrent desktop load. The simulations use precomputed activation-time commands; processor/API wall time is logged separately from traffic time. A delayed handover requires propagating the retained plan to its scheduled activation and checking that inherited state set.","SmallText")
p("Formal conclusions apply to the specified initial boxes and bounded arrival residuals through the original absolute expiry. MFD parameters and internal demand remain fixed in this synthetic study. Regulation spans three initial total pairs; the plan-replacement benchmark uses one inherited state. Every solver, certificate and sampled constraint result is retained.")
h("Recommended manuscript use")
p("1. Formulate the three benchmark tasks before presenting the architecture. Connect regional regulation, queue-inclusive planning and deadline-aware replacement to the perimeter-control literature.")
p("2. Keep one matched deadline table, the unchanged-admission result, and a compact uncertainty/repair summary in the main paper. Release full grids and detailed trajectories as a versioned benchmark.")
p("3. Present nominal MPC, scenario planning and predictive repair as tools operating under the same plan-admission rule. Lead with physical/service feasibility and the operator's conditional priority-versus-waiting tradeoff.")
p("4. Give contextual task-state accuracy its own denominator and retain the eight end-to-end traces as an interface demonstration. The encoded obligations define what the checker enforces.")
h("Reproducibility and literature anchors")
p("The package contains benchmark_formulation.tex, the portable state/continuation fixture, protocols, unchanged model/checker sources, all commands and certificates, raw API responses, CSV results and reproduction scripts. The previous full Elsevier and four-layout packages are preserved separately. The current manuscript is unchanged; this study performs no GitHub push.","SmallText")
p(f"An implementable tool portfolio selects the lowest nominal center-forecast cost among independently certified same-contract candidates. It supplied {DATA['certified_tool_portfolio']['available']}/80 outputs, with realized outcomes reserved for evaluation.","SmallText")
p('Zhong et al. (2018), Transportation Research Part B 117, 687-707. DOI: <link href="https://doi.org/10.1016/j.trb.2017.09.008">10.1016/j.trb.2017.09.008</link>. Reproduced native CLF equations and a separately identified queue/exit extension.',"SmallText")
p('Geroliminis, Haddad and Ramezani (2013), IEEE T-ITS 14(1), 348-359. DOI: <link href="https://doi.org/10.1109/TITS.2012.2216877">10.1109/TITS.2012.2216877</link>. Predictive perimeter-control lineage; this benchmark supplies its own local solver/blocking implementation.',"SmallText")
p("Haddad (2017), Optimal perimeter control synthesis for two urban regions with aggregate boundary queue dynamics, Transportation Research Part B. Exterior waiting motivates the benchmark's queue-inclusive accounting.","SmallText")

def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor("#CAD5DE"));canvas.line(52,40,560,40)
    canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#526777"))
    canvas.drawString(52,27,"Urban perimeter-control benchmark | 26 September 2026")
    canvas.drawRightString(560,27,str(doc.page));canvas.restoreState()

doc=SimpleDocTemplate(str(OUTPUT),pagesize=letter,rightMargin=52,leftMargin=52,topMargin=48,bottomMargin=54,
    title="Reusable perimeter-control benchmark: complete sweep",author="Research study")
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(OUTPUT)
