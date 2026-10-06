"""Run the frozen finite evaluation menu once; a failed episode stops the run."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--group',choices=('deterministic','live'),required=True)
args=p.parse_args()
if args.group=='deterministic':
    menu=[['--case','oracle','--provider','oracle'],
          ['--case','network','--provider','network'],
          ['--case','command_pressure_certified','--provider','command_pressure'],
          ['--case','command_pressure_nominal','--provider','command_pressure','--nominal']]
else:
    menu=[]
    for paraphrase in (0,1):
        for variant in ('original','structured'):
            name=f'heldout_p{paraphrase}_{variant}'
            menu.append(['--case',name,'--provider',variant,'--paraphrase',str(paraphrase)])
    menu.append(['--case','poison_structured','--provider','structured','--split','poison'])
    for name in ['heldout_p0_original','heldout_p0_structured','heldout_p1_original',
                 'heldout_p1_structured','poison_structured']:
        menu.append(['--case',name+'_nominal','--nominal','--replay-case',name])
for call in menu:
    subprocess.run([sys.executable,str(ROOT/'reproducibility/mfd_task_state_runtime_v2.py'),
                    *call],cwd=ROOT,check=True)
