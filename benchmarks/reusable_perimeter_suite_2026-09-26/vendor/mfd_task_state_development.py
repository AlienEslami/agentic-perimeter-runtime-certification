"""Frozen initial-condition development grid; no live model calls.

All old scientific kernels and results are read-only inputs. This grid chooses
an informative physical regime, not a test-set-optimal language prompt.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

import mfd_incident_model as model
import mfd_incident_mpc as mpc
from mfd_incident_service_frontier import solve_service, replay_plan

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproducibility/results/task_state_development'
PAIRS = [(3000, 3000), (3500, 2500), (2500, 3500), (4000, 4000)]
QUEUES = [200, 1000]
ROWS = {'network': [1., 1., 0., 0., 0., 0., 1.],
        'region_1': [4., 1., 0., 0., 0., 0., 1.],
        'region_2': [1., 4., 0., 0., 0., 0., 1.]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initial_state(pair, queue):
    x = model.BASE_STATE.copy()
    x[[0, 2, 4]] *= pair[0] / (100 * model.BASE_STATE[0])
    x[[1, 3, 5]] *= pair[1] / (100 * model.BASE_STATE[1])
    x[6] = queue / 100
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(scope='DEVELOPMENT ONLY: initial-condition sensitivity',
                    pairs=PAIRS, queues=QUEUES, rows=ROWS,
                    priority_minutes=30, service_vehicles=2850, horizon_h=1,
                    source_sha256=sha(__file__), worklog_sha256=sha(ROOT/'SPECIAL_ISSUE_REVISION_WORKLOG.md'))
    frozen = OUT/'protocol.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == json.loads(json.dumps(protocol))
    else:
        with frozen.open('x', encoding='utf-8') as f:
            json.dump(protocol, f, indent=2)
    library = model.default_library()
    for pair in PAIRS:
        for queue in QUEUES:
            x = initial_state(pair, queue)
            for objective, row in ROWS.items():
                path = OUT/f'n{pair[0]}_{pair[1]}_q{queue}_{objective}.json'
                if path.exists():
                    continue
                started = time.perf_counter()
                selected, report = solve_service(x, library=library, service_target_vehicles=2850.,
                    deadline_h=1., horizon_h=1., cost_row=row, priority_minutes=30)
                rec = dict(protocol_sha256=sha(frozen), pair_vehicles=pair, queue_vehicles=queue,
                           objective=objective, initial_state=x.tolist(), optimizer=report)
                if selected is not None:
                    commands = selected[1].reshape(2, 4)
                    sol = solve_ivp(lambda t, z: model.field(z, commands[0], mpc.arrival(t)),
                                    (0, 1/12), x, method='DOP853', rtol=1e-11, atol=1e-12,
                                    max_step=1/360)
                    assert sol.success
                    rec.update(commands=commands.tolist(), first_epoch_state=sol.y[:, -1].tolist(),
                               held_plan=replay_plan(x, commands, service_target_vehicles=2850.))
                rec['wall_seconds'] = time.perf_counter()-started
                with path.open('x', encoding='utf-8') as f:
                    json.dump(rec, f, indent=2, allow_nan=False)
                print(json.dumps(dict(case=path.stem, feasible=selected is not None,
                                      first_state=rec.get('first_epoch_state'),
                                      seconds=rec['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
