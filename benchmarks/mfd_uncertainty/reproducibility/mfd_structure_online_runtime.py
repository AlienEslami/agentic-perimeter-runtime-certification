"""Trusted fresh-box online integration of unchanged finite-plan certificates.

Protocol: STRUCTURE_ONLINE_RUNTIME_PROTOCOL.md. No scientific kernel is edited.
Semantic requests are archived REPLAY, not new live calls. Full accepted proof
artifacts and original correlated/auxiliary sets remain immutable on retention.
"""
from __future__ import annotations

import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

import mfd_incident_model as model
import mfd_incident_mpc as base
from mfd_incident_service_frontier import solve_service
from mfd_incident_mpc_certification import ConstantCommandVerifier, constant_command_mode, verify_command_plan
from mfd_incident_positive_reachability import PositiveVerifier
from mfd_incident_affine_reachability import AffineGeneratorVerifier, source_hashes as affine_sources
from mfd_incident_uncertainty_frontier import initial_supports
from mfd_incident_verifier import contains_information_box

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reproducibility/results"
PROTOCOL = ROOT / "STRUCTURE_ONLINE_RUNTIME_PROTOCOL.md"
OUTPUT = RESULTS / "mfd_structure_online_runtime_results.json"
REQUEST_ARCHIVE = RESULTS / "mfd_incident_agent_live_p0_stateful_results.json"
PERIOD, END = Fraction(1, 12), Fraction(1)
ROWS = {"network": [1., 1., 0., 0., 0., 0., 1.],
        "region_1": [4., 1., 0., 0., 0., 0., 1.],
        "region_2": [1., 4., 0., 0., 0., 0., 1.]}
SIGN = np.array([1., -1., 1., -1., 1., -1., 1.])


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def source_hashes():
    hashes = affine_sources()
    for name in ("reproducibility/mfd_structure_online_runtime.py", "STRUCTURE_ONLINE_RUNTIME_PROTOCOL.md",
                 "reproducibility/mfd_incident_service_frontier.py", "reproducibility/mfd_incident_mpc.py",
                 "reproducibility/mfd_incident_agent_runtime.py", "reproducibility/audit_mfd_incident_model.py",
                 "reproducibility/results/mfd_incident_agent_live_p0_stateful_results.json",
                 "reproducibility/results/mfd_reachability_service_margin_results.json"):
        hashes[name] = file_sha(ROOT/name)
    return hashes


def snapshot_sources(hashes):
    snapshots = {}
    for name, sha in hashes.items():
        dest = RESULTS / "sors" / sha[:16] / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(ROOT/name, dest)
        assert file_sha(dest) == sha
        snapshots[name] = dest.relative_to(ROOT).as_posix()
    return snapshots


def model_contract():
    return dict(equations="seven_state_class_conservation_with_min_admission_v1", state_order=list(model.STATE_ORDER),
        MFD_coefficients=[model.original.MFD_A, model.original.MFD_B, model.original.MFD_C],
        internal_demand=model.INTERNAL_DEMAND.tolist(), admission_capacity=model.CAPACITY,
        release_rate=model.RELEASE_RATE, base_arrival=model.BASE_ARRIVAL, queue_limit=model.QUEUE_LIMIT,
        regional_jam_limits=[100., 100.], command_box=[0., 1.], directional_limit=.75, admission_split=.5)


def make_contract(width_vehicles, residual_veh_h, service_vehicles, hashes):
    widths = [width_vehicles/100.]*6+[min(width_vehicles, 10.)/100.]
    return dict(version="structure_online_finite_mission_v1", model=model_contract(),
        pulse=[str(Fraction.from_float(.3)), "1", "1/4", str(Fraction.from_float(1/12))],
        residual_bound=residual_veh_h/100., physical_half_widths=widths,
        sensor_contract="trusted_current_physical_box_contains_actual_state_v1",
        deadline_exact_h="1", service_checks=[dict(exact_time="1", coordinate=0, upper=service_vehicles/100.)],
        mission_start_exact_h="0", mission_expiry_exact_h="1", online_protocol_sha256=file_sha(PROTOCOL),
        immutable_source_sha256=hashes)


def information_box(center, now, contract, *, actual_state=None):
    record = dict(exact_time_h=str(now), center=np.asarray(center).tolist(),
        half_widths=contract["physical_half_widths"], sensor_contract=contract["sensor_contract"],
        soundness_assumption="actual state belongs to this current physical box",
        actual_state_access="simulation audit only; authority sees center/width/time/contract")
    record["information_id"] = digest(record)
    if actual_state is not None:
        error = np.abs(np.asarray(actual_state)-np.asarray(center))
        record["simulation_containment_audit"] = dict(passed=bool(np.all(error <= np.asarray(record["half_widths"]))),
            absolute_error=error.tolist(), actual_state=np.asarray(actual_state).tolist())
        assert record["simulation_containment_audit"]["passed"]
    return record


class AuthorityError(ValueError):
    pass


class TrustedProofStore:
    """Private trusted-verifier artifact registry; semantic proposer has no API here.

    Registering a proof requires producing it through certify() below. Full byte
    hashes, not descriptor IDs/boolean claims, bind the immutable artifacts.
    """
    def __init__(self, case_id, hashes):
        self.case_id, self.hashes, self.registry = case_id, copy.deepcopy(hashes), {}

    def certify(self, method, info, commands, durations, now, contract, epoch):
        if source_hashes() != self.hashes:
            raise AuthorityError("source drift before trusted verification")
        if method not in ("original", "positive_queue", "affine_v3"):
            raise AuthorityError("unrecognized trusted verifier")
        validators = {"original": ConstantCommandVerifier, "positive_queue": lambda: PositiveVerifier("positive_queue"),
                      "affine_v3": AffineGeneratorVerifier}
        support = initial_supports(info["center"], commands[0], info["half_widths"])
        started = time.perf_counter()
        certificate, precondition = None, None
        if not support["valid"]:
            precondition = dict(kind="invalid_initial_physical_information_set", support=support)
        else:
            try:
                certificate = verify_command_plan(info["center"], commands, durations,
                    initial_time_h=now, amplitude=.3, duration=1., initial_box=info["half_widths"],
                    residual_bound=contract["residual_bound"], step_seconds=15., verifier=validators[method](),
                    service_checks=[dict(time_h=END, coordinate=0, upper=contract["service_checks"][0]["upper"])],
                    retain_steps=True)
            except ValueError as exc:
                precondition = dict(kind="verifier_precondition_failed", detail=str(exc))
        if source_hashes() != self.hashes:
            raise AuthorityError("source drift during trusted verification")
        artifact = dict(origin="trusted_local_verifier", method=method, case_id=self.case_id, epoch=epoch,
            exact_installation_candidate_h=str(now), online_protocol_sha256=file_sha(PROTOCOL),
            current_contract_sha256=digest(contract), information_box=copy.deepcopy(info),
            commands=copy.deepcopy(commands), exact_durations_h=list(map(str, durations)),
            certification_source_sha256=self.hashes, certificate=certificate, precondition_failure=precondition,
            proof_wall_seconds=time.perf_counter()-started)
        path = RESULTS / "sor" / self.case_id / f"p{epoch:02}.json"
        save_new(path, artifact)
        reference = dict(path=path.relative_to(ROOT).as_posix(), sha256=file_sha(path))
        key = digest(reference)
        self.registry[key] = copy.deepcopy(reference)
        return dict(registry_id=key, **reference), artifact

    def load_registered(self, reference):
        key = reference.get("registry_id")
        expected = self.registry.get(key)
        if expected != {k: reference.get(k) for k in ("path", "sha256")}:
            raise AuthorityError("unregistered proof artifact")
        path = ROOT/expected["path"]
        if file_sha(path) != expected["sha256"]:
            raise AuthorityError("proof artifact byte hash mismatch")
        artifact = json.loads(path.read_text())
        if artifact["origin"] != "trusted_local_verifier" or artifact["certification_source_sha256"] != self.hashes:
            raise AuthorityError("untrusted proof provenance")
        return artifact


class RuntimeAuthority:
    """Fresh installation and exact, contiguous retention of immutable plans.

    No checkpoint shape or numerical membership test enters these guards.
    Callers must preserve the trusted sensor/plant assumptions. No claims are
    made against modification of the local trusted interpreter or verifier.
    """
    def __init__(self, contract, method, store):
        self.contract = copy.deepcopy(contract)
        self.contract_sha256 = digest(contract)
        self.method, self.store = method, store
        self.active = None
        self.next_time = Fraction(contract["mission_start_exact_h"])
        self.execution_chain = digest(dict(genesis=self.contract_sha256))
        self.accepted_history = []
        self.execution_history = []

    def _contract(self, contract):
        if digest(contract) != self.contract_sha256:
            raise AuthorityError("trusted contract changed")
        if source_hashes() != self.contract["immutable_source_sha256"]:
            raise AuthorityError("trusted source changed")

    def install(self, reference, information, now, contract):
        self._contract(contract)
        now = Fraction(now)
        if now != self.next_time or now >= Fraction(contract["mission_expiry_exact_h"]):
            raise AuthorityError("installation clock not current or mission expired")
        if Fraction(information["exact_time_h"]) != now:
            raise AuthorityError("stale information timestamp")
        if information["sensor_contract"] != contract["sensor_contract"] or information["half_widths"] != contract["physical_half_widths"]:
            raise AuthorityError("information contract mismatch")
        info_for_id = {k:v for k,v in information.items() if k not in ("information_id", "simulation_containment_audit")}
        if digest(info_for_id) != information["information_id"]:
            raise AuthorityError("information identity mismatch")
        artifact = self.store.load_registered(reference)
        if artifact["method"] != self.method or artifact["current_contract_sha256"] != self.contract_sha256:
            raise AuthorityError("proof method or task contract mismatch")
        if artifact["exact_installation_candidate_h"] != str(now) or artifact["information_box"] != information:
            raise AuthorityError("proof was not generated for this information/installation")
        cert = artifact["certificate"]
        if cert is None or not all(cert[key] for key in ("accepted", "verified_continuous_physical", "verified_scheduled_service", "verified_terminal_service")):
            raise AuthorityError("candidate has no accepted complete finite-plan certificate")
        desc = cert["proof_descriptor"]
        if digest(desc) != cert["proof_id"]:
            raise AuthorityError("proof descriptor hash mismatch")
        if desc["initial_state"] != information["center"] or cert["initial_box_units"] != information["half_widths"]:
            raise AuthorityError("proof initial information set mismatch")
        if (Fraction(desc["initial_time"]) != now or desc["model_contract"] != contract["model"]
                or desc["pulse"] != contract["pulse"] or desc["residual_bound"] != contract["residual_bound"]
                or desc["service_checks"] != contract["service_checks"]):
            raise AuthorityError("proof physical/time/service assumptions mismatch")
        if artifact["exact_durations_h"] != desc["durations"] or artifact["commands"] != cert["adapter_audit"]["commands"]:
            raise AuthorityError("proof executed command binding mismatch")
        durations = list(map(Fraction, desc["durations"]))
        expiry = now+sum(durations)
        if expiry != Fraction(contract["mission_expiry_exact_h"]) or cert["certified_until_h"] != float(expiry):
            raise AuthorityError("proof absolute expiry mismatch")
        if min(cert["continuous_constraint_margins_lower"].values()) < 0:
            raise AuthorityError("negative certified physical support")
        if self.method == "affine_v3":
            method = desc.get("signed_generator_method", {})
            if method.get("rule") != "signed_initial_generator_plus_robust_variational_remainder_positive_queue_v3":
                raise AuthorityError("affine method descriptor mismatch")
            if method.get("source_sha256") != affine_sources() or method.get("generator_or_remainder_contraction") is not False:
                raise AuthorityError("affine kernel source or correlation contract mismatch")
        elif self.method == "positive_queue":
            method = desc.get("structural_positive_extension", {})
            if method.get("variant") != "positive_queue" or method.get("source_sha256") != contract["immutable_source_sha256"]["reproducibility/mfd_incident_positive_reachability.py"]:
                raise AuthorityError("positive-queue method mismatch")
        elif "signed_generator_method" in desc or "structural_positive_extension" in desc:
            raise AuthorityError("original-method contract mismatch")
        installation = dict(method=self.method, original_proof_reference=copy.deepcopy(reference),
            original_proof_id=cert["proof_id"], original_commands=copy.deepcopy(artifact["commands"]),
            original_exact_durations_h=desc["durations"], original_installation_exact_h=str(now), original_expiry_exact_h=str(expiry),
            original_epoch=artifact["epoch"], information_id=information["information_id"],
            current_contract_sha256=self.contract_sha256, executed_prefix_hash_at_installation=self.execution_chain,
            accepted_history_prefix_hash=digest(self.accepted_history),
            predecessor_installation_id=None if self.active is None else self.active["installation_id"],
            retained_evidence="immutable complete artifact, including correlated and independent auxiliary sets")
        installation["installation_id"] = digest(installation)
        self.active = copy.deepcopy(installation)
        self.accepted_history.append(copy.deepcopy(installation))
        return copy.deepcopy(installation)

    def command_for(self, now, following, contract):
        self._contract(contract)
        now, following = Fraction(now), Fraction(following)
        if self.active is None:
            raise AuthorityError("unresolved initialization: no accepted plan")
        active_payload = {k:v for k,v in self.active.items() if k != "installation_id"}
        if digest(active_payload) != self.active["installation_id"] or self.accepted_history[-1] != self.active:
            raise AuthorityError("accepted installation history or original schedule changed")
        if now != self.next_time or following <= now:
            raise AuthorityError("execution must be contiguous with the authorized prefix")
        if following > Fraction(self.active["original_expiry_exact_h"]):
            raise AuthorityError("original accepted plan expired")
        # Byte-hash check protects the full correlated + auxiliary evidence.
        artifact = self.store.load_registered(self.active["original_proof_reference"])
        if (artifact["commands"] != self.active["original_commands"]
                or artifact["exact_durations_h"] != self.active["original_exact_durations_h"]
                or artifact["certificate"]["proof_id"] != self.active["original_proof_id"]):
            raise AuthorityError("retained installation does not match its original full proof")
        cursor = Fraction(self.active["original_installation_exact_h"])
        for index, (command, duration) in enumerate(zip(self.active["original_commands"], self.active["original_exact_durations_h"])):
            endpoint = cursor+Fraction(duration)
            if cursor <= now < endpoint and following <= endpoint:
                return copy.deepcopy(command), dict(installation_id=self.active["installation_id"],
                    proof_id=self.active["original_proof_id"], proof_reference=self.active["original_proof_reference"],
                    original_command_index=index, original_epoch=self.active["original_epoch"],
                    original_installation_exact_h=self.active["original_installation_exact_h"],
                    original_expiry_exact_h=self.active["original_expiry_exact_h"],
                    executed_prefix_hash=self.execution_chain, exact_left_h=str(now), exact_right_h=str(following))
            cursor = endpoint
        raise AuthorityError("requested interval is not inside one original command block")

    def commit_execution(self, now, following, command, contract):
        expected, binding = self.command_for(now, following, contract)
        if expected != list(command):
            raise AuthorityError("executed command differs from the accepted original schedule")
        record = dict(**binding, command=list(command))
        record["execution_record_hash"] = digest(record)
        self.execution_chain = record["execution_record_hash"]
        self.execution_history.append(record)
        self.next_time = Fraction(following)
        return copy.deepcopy(record)


def declared_menu():
    cases = []
    for tag, width, residual in (("base", 1., 60.), ("wide50", 50., 120.), ("wide25", 25., 300.)):
        for method in ("original", "positive_queue", "affine_v3"):
            cases.append(dict(case_id=f"{tag}_{method}", method=method, width=width, residual=residual,
                service=2850., commissioning=2850, realization="nominal", nominal_comparator="nominal_2850"))
    for method in ("original", "positive_queue", "affine_v3"):
        cases.append(dict(case_id=f"binding1600_{method}", method=method, width=1., residual=60.,
            service=1700., commissioning=1600, realization="nominal", nominal_comparator="nominal_1700_c1600"))
    cases.append(dict(case_id="binding1625_affine_v3", method="affine_v3", width=1., residual=60.,
        service=1700., commissioning=1625, realization="nominal", nominal_comparator="nominal_1700_c1625"))
    cases.append(dict(case_id="adverse25_affine_v3", method="affine_v3", width=25., residual=300.,
        service=2850., commissioning=2850, realization="adverse", nominal_comparator=None))
    for case_id, service, commissioning in (("nominal_2850", 2850., 2850), ("nominal_1700_c1600", 1700., 1600), ("nominal_1700_c1625", 1700., 1625)):
        cases.append(dict(case_id=case_id, method="nominal", width=1., residual=60., service=service,
            commissioning=commissioning, realization="nominal", nominal_comparator=None))
    order = ["base_original", "base_positive_queue", "base_affine_v3", "binding1625_affine_v3"]
    return sorted(cases, key=lambda c: order.index(c["case_id"]) if c["case_id"] in order else len(order)+cases.index(c))


def commissioning_commands(target):
    if target == 2850:
        source = RESULTS/"mfd_incident_mpc_grid_results.json"
        archive = json.loads(source.read_text())
        chosen = next(r for r in archive["runs"] if r["amplitude"] == .3 and r["duration_h"] == 1.)["decisions"][0]
        return chosen["controls"], dict(source=source.relative_to(ROOT).as_posix(), source_sha256=file_sha(source), selection="first primary archived MPC plan")
    source = RESULTS/"mfd_reachability_service_margin_results.json"
    archive = json.loads(source.read_text())
    chosen = next(r for r in archive["rows"] if r["nominal_target_vehicles"] == target)
    return chosen["commands"], dict(source=source.relative_to(ROOT).as_posix(), source_sha256=file_sha(source), selection=f"frozen nominal target {target} vehicles")


def request_sequence():
    archive = json.loads(REQUEST_ARCHIVE.read_text())
    return [dict(epoch=e["epoch"], request_valid=e["request_valid"], tool_request=e["tool_request"])
            for e in archive["certified"]["decisions"]]


def realized_residual(t, spec):
    if spec["realization"] != "adverse":
        return 0.
    quarter = min(3, int(float(t)*4))
    return spec["residual"]/100.*(1 if quarter % 2 == 0 else -1)


def run_case(spec, hashes):
    case_id, method = spec["case_id"], spec["method"]
    certified = method != "nominal"
    contract = make_contract(spec["width"], spec["residual"], spec["service"], hashes)
    widths = np.asarray(contract["physical_half_widths"])
    actual = model.BASE_STATE.copy()
    if spec["realization"] == "adverse":
        actual += .5*widths*SIGN
    initial_actual = actual.copy()
    initial_inventory = 100.*sum(actual[[0, 1, 6]])
    store = TrustedProofStore(case_id, hashes)
    authority = RuntimeAuthority(contract, method, store) if certified else None
    initial_commands, commissioning_source = commissioning_commands(spec["commissioning"])
    requests = request_sequence()
    library = model.default_library()
    times, states, actual_commands, decisions = [0.], [actual.tolist()], [], []
    totals = np.zeros(4)  # TTS, exterior-queue delay, departures, all offered inflow
    minimum_margin = np.inf
    nominal_plan, last_command = None, None
    stopped = None
    began_case = time.perf_counter()
    for epoch in range(12):
        now, following = epoch*PERIOD, (epoch+1)*PERIOD
        started = time.perf_counter()
        request = requests[epoch]
        if request["epoch"] != epoch:
            raise ValueError("archived request clock mismatch")
        observation = (model.BASE_STATE.copy() if epoch == 0 else actual.copy())
        if spec["realization"] == "adverse" and epoch > 0:
            observation += .5*widths*SIGN*((-1)**epoch)
        info = information_box(observation, now, contract, actual_state=actual)
        req = copy.deepcopy(request["tool_request"])
        valid = bool(request["request_valid"] and req is not None)
        if valid:
            objective = "network" if req["objective"] == "balanced" else req["objective"]
            valid = objective in ROWS and req["cost_row"] == ROWS[objective] and int(req["duration_minutes"]) > 0
        event = dict(epoch=epoch, exact_time_h=str(now), exact_next_decision_h=str(following),
            actual_initial_state=actual.tolist(), information_box=info, request_mode="ARCHIVED_SEMANTIC_REPLAY",
            archived_request=request, source_request_sha256=file_sha(REQUEST_ARCHIVE),
            candidate_admitted=False, certificate_reference=None, candidate_commands=None,
            optimizer_report=None, failure_reason=None, proof_wall_seconds=0., installation=None)
        candidate = None
        if epoch == 0:
            candidate = np.asarray(initial_commands)
            event["commissioning_source"] = commissioning_source
        elif valid:
            offered_nominal = lambda t: model.arrival(t, amplitude=.3, duration=1.)
            warm = None
            if certified and authority.active is not None:
                warm = np.asarray(authority.active["original_commands"][-1])
            elif nominal_plan is not None:
                warm = np.asarray(nominal_plan["commands"][-1])
            selected, report = solve_service(observation, float(now), library, warm, offered_nominal,
                service_target_vehicles=spec["service"], deadline_h=1., horizon_h=float(END-now),
                cost_row=ROWS[objective], priority_minutes=int(req["duration_minutes"]))
            event["optimizer_report"] = report
            if selected is not None:
                candidate = selected[1].reshape(2, 4).copy()
        lengths = [min(PERIOD, END-now), max(Fraction(0), END-now-PERIOD)]
        if candidate is not None:
            pairs = [(u.tolist(), duration) for u, duration in zip(candidate, lengths) if duration > 0]
            commands, durations = [u for u, d in pairs], [d for u, d in pairs]
            event["candidate_commands"], event["candidate_exact_durations_h"] = commands, list(map(str, durations))
            if certified:
                ref, artifact = store.certify(method, info, commands, durations, now, contract, epoch)
                event["certificate_reference"] = ref
                event["proof_wall_seconds"] = artifact["proof_wall_seconds"]
                cert = artifact["certificate"]
                if cert is not None and cert["accepted"]:
                    event["installation"] = authority.install(ref, info, now, contract)
                    event["candidate_admitted"] = True
                else:
                    event["failure_reason"] = artifact["precondition_failure"] if cert is None else [f["kind"] for f in cert["failures"]]
            else:
                event["candidate_admitted"] = True
                nominal_plan = dict(commands=commands, installation=str(now), durations=list(map(str, durations)), epoch=epoch)
        else:
            event["failure_reason"] = "invalid_archived_request" if not valid else "optimizer_no_feasible_solution"
        if certified:
            try:
                command, binding = authority.command_for(now, following, contract)
            except AuthorityError as exc:
                stopped = dict(exact_time_h=str(now), reason=str(exc))
                event.update(execution_source="unresolved", executed_constant_command=None, stopped=stopped)
                decisions.append(event)
                break
            event["execution_source"] = "verified_replacement" if event["candidate_admitted"] else "retained_original_certified_suffix"
            event["execution_binding"] = binding
        else:
            if nominal_plan is None:
                stopped = dict(exact_time_h=str(now), reason="no nominal commissioning plan")
                event.update(execution_source="unresolved", executed_constant_command=None, stopped=stopped)
                decisions.append(event)
                break
            command = nominal_plan["commands"][0 if nominal_plan["epoch"] == epoch else -1]
            event["execution_source"] = "nominal_replacement" if event["candidate_admitted"] else "retained_nominal_suffix"
        event["executed_constant_command"] = command
        event["first_command_changed"] = last_command is None or last_command != command
        last_command = list(command)
        u = np.asarray(command)
        # The selected adverse residual is constant on each five-minute epoch;
        # freeze its one-sided value through RK4 endpoint evaluation.
        residual = realized_residual(float((now+following)/2), spec)
        event["actual_residual_units_per_hour"] = residual
        offered = lambda t: model.arrival(t, amplitude=.3, duration=1.)+residual
        local_margin = np.inf
        for step in range(30):
            t = float(now)+step*base.EXECUTION_STEP
            new = model.rk4(actual, lambda _x: u, t, base.EXECUTION_STEP, offered)
            local_margin = min(local_margin, float(min(model.margins(actual, u))), float(min(model.margins(new, u))))
            h = base.EXECUTION_STEP
            totals[0] += h/2*100*(base.COST_ROW@(actual+new))
            totals[1] += h/2*100*(actual[6]+new[6])
            totals[2] += h/2*100*(model.departure_rate(actual, u)+model.departure_rate(new, u))
            totals[3] += h/2*100*(2*sum(model.INTERNAL_DEMAND)+offered(t)+offered(t+h))
            actual = new
            times.append(t+h); states.append(actual.tolist()); actual_commands.append(command)
        minimum_margin = min(minimum_margin, local_margin)
        event.update(actual_final_state=actual.tolist(), sampled_physical_margin=local_margin,
            executed_end_time_h=str(following), cumulative_tts_veh_h=float(totals[0]))
        if certified:
            event["committed_execution"] = authority.commit_execution(now, following, command, contract)
            original = store.load_registered(authority.active["original_proof_reference"])["certificate"]
            checkpoints = {Fraction(c["exact_time_fraction"]): c for c in original["reachable_checkpoints"]}
            hull = contains_information_box(checkpoints[following], actual, 0.)
            event["outer_hull_diagnostic"] = dict(**hull, authority_used=False,
                explanation="sampled endpoint versus original proof outer hull; not correlated restart or retention authority")
        event["wall_seconds"] = time.perf_counter()-started
        decisions.append(event)
        print(json.dumps(dict(case=case_id, epoch=epoch, accepted=event["candidate_admitted"],
            source=event["execution_source"], reason=event["failure_reason"], n1_vehicles=100.*actual[0],
            proof_seconds=event["proof_wall_seconds"])), flush=True)
    complete = len(actual_commands) == 360 and stopped is None
    expiry_guard = None
    if certified and complete:
        try:
            authority.command_for(END, END+PERIOD, contract)
        except AuthorityError as exc:
            expiry_guard = dict(passed=True, rejected=True, reason=str(exc), tested_interval=["1", "13/12"])
        else:
            raise AssertionError("execution authorized after absolute mission expiry")
    metrics = dict(complete=complete, completed_horizon_h=times[-1], stopped=stopped,
        tts_veh_h=float(totals[0]), queue_veh_h=float(totals[1]), departures_vehicles=float(totals[2]),
        offered_inventory_vehicles=float(totals[3]), max_queue_vehicles=100.*max(s[6] for s in states),
        deadline_vehicles=100.*actual[0] if complete else None,
        realized_service_pass=bool(complete and 100.*actual[0] <= spec["service"]),
        minimum_sampled_physical_margin=float(minimum_margin) if np.isfinite(minimum_margin) else None,
        conservation_residual_vehicles=initial_inventory+float(totals[3])-100.*sum(actual[[0, 1, 6]])-float(totals[2]),
        candidate_acceptances=sum(d["candidate_admitted"] for d in decisions),
        rejected_candidates=sum(not d["candidate_admitted"] for d in decisions),
        retained_epochs=sum(d["execution_source"] == "retained_original_certified_suffix" for d in decisions),
        first_command_changes=sum(d.get("first_command_changed", False) for d in decisions),
        proof_seconds=sum(d["proof_wall_seconds"] for d in decisions), wall_seconds=time.perf_counter()-began_case)
    if source_hashes() != hashes:
        raise RuntimeError("source drift during online run")
    return dict(specification=spec, current_contract=contract, current_contract_sha256=digest(contract),
        protocol_sha256=file_sha(PROTOCOL), source_sha256=hashes, request_mode="ARCHIVED_SEMANTIC_REPLAY",
        measurement_scope="trusted fresh physical boxes; simulation checks actual inclusion; no statistical calibration claim",
        actual_initial_state=initial_actual.tolist(), nominal_initial_information_center=model.BASE_STATE.tolist(),
        decisions=decisions, metrics=metrics, expiry_guard=expiry_guard, trusted_proof_registry=store.registry,
        accepted_history=[] if authority is None else authority.accepted_history,
        executed_history=[] if authority is None else authority.execution_history,
        trajectory=dict(time_h=times, state=states, command=actual_commands))


def run(limit=None, only=None):
    hashes = source_hashes()
    snapshots = snapshot_sources(hashes)
    menu = declared_menu()
    if OUTPUT.exists():
        result = json.loads(OUTPUT.read_text())
        if result["source_sha256"] != hashes or result["frozen_menu"] != menu:
            raise RuntimeError("Source/menu changed; preserve outputs and use a separately frozen version")
    else:
        result = dict(status="RUNNING", completed=False, protocol_path=PROTOCOL.relative_to(ROOT).as_posix(),
            protocol_sha256=file_sha(PROTOCOL), source_sha256=hashes, source_snapshots=snapshots,
            frozen_menu=menu, cases=[], comparisons=[], scientific_kernel_changes=False,
            new_live_calls=0, request_scope="archived p0-stateful semantic REPLAY on each current state")
        save_new(OUTPUT, result)
    count = 0
    for spec in menu:
        if any(c["case_id"] == spec["case_id"] for c in result["cases"]):
            continue
        if only is not None and spec["case_id"] not in only:
            continue
        if limit is not None and count >= limit:
            break
        case = run_case(spec, hashes)
        file = RESULTS / "sor" / spec["case_id"] / "run.json"
        save_new(file, case)
        result["cases"].append(dict(case_id=spec["case_id"], specification=spec, metrics=case["metrics"],
            run_file=file.relative_to(ROOT).as_posix(), run_sha256=file_sha(file)))
        result["completed"] = len(result["cases"]) == len(menu)
        result["status"] = "COMPLETE" if result["completed"] else "RUNNING"
        result["comparisons"] = []
        cases = {c["case_id"]:c for c in result["cases"]}
        for entry in result["cases"]:
            comparator = entry["specification"]["nominal_comparator"]
            if comparator in cases:
                other = cases[comparator]
                both = entry["metrics"]["complete"] and other["metrics"]["complete"]
                delta = entry["metrics"]["tts_veh_h"]-other["metrics"]["tts_veh_h"] if both else None
                result["comparisons"].append(dict(case_id=entry["case_id"], comparator=comparator,
                    matched_complete_horizon=both, incremental_tts_veh_h=delta,
                    incremental_tts_percent=None if delta is None else 100.*delta/other["metrics"]["tts_veh_h"],
                    scope="same nominal realized plant/semantic requests/commissioning commands; uncertainty contracts differ explicitly"))
        OUTPUT.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        print(json.dumps(dict(case=spec["case_id"], completed=case["metrics"]["complete"],
            accepted=case["metrics"]["candidate_acceptances"], retained=case["metrics"]["retained_epochs"],
            deadline=case["metrics"]["deadline_vehicles"], tts=case["metrics"]["tts_veh_h"])), flush=True)
        count += 1
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    run(args.limit, args.only)
