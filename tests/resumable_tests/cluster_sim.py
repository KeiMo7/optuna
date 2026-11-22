"""Eight-machine cluster simulation on local processes + shared storage.

Simulates a cluster of ``N_WORKERS`` executor processes backed by a shared
SQLite database (trial state + lease ledger in the same DB) and a shared
checkpoint directory, and exercises three failure shapes:

1. **suspend**: some trials suspend mid-training and are resumed by whatever
   machine picks them up,
2. **crash**: one machine is killed with ``os._exit`` mid-trial (no cleanup);
   its lease expires and another machine takes the trial over,
3. **duplicate takeover**: two machines race to take over the same expired
   lease; exactly one may win.

Every machine appends events to a shared JSONL log. The parent process then
verifies the cluster-wide invariants and returns a report dict.
"""

from __future__ import annotations

import fcntl
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import optuna
from optuna.resumable import FileCheckpointStore
from optuna.resumable import RDBLeaseStore
from optuna.resumable import ResumableExecutor
from optuna.resumable._lease import BaseLeaseStore
from optuna.trial import TrialState


N_WORKERS = 8
N_TRIALS = 16
N_STEPS = 20
LEASE_TTL = 10.0
STEP_SLEEP = 0.02
CRASH_WORKER = 3
CRASH_STEP = 10
SUSPEND_STEP = 7
SUSPEND_STEP_2 = 15

optuna.logging.set_verbosity(optuna.logging.WARNING)


def expected_value(x: float, n_steps: int = N_STEPS) -> float:
    """Closed form of the deterministic objective's final value."""

    return n_steps * (x - 2) ** 2 + (n_steps - 1) * n_steps / 2


def expected_intermediate(x: float, step: int) -> float:
    return (step + 1) * (x - 2) ** 2 + step * (step + 1) / 2


def log_event(workdir: str, record: dict[str, Any]) -> None:
    record = dict(record)
    record["ts"] = time.time()
    with open(os.path.join(workdir, "events.jsonl"), "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(record) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


class LoggingLeaseStore(BaseLeaseStore):
    """Lease store wrapper that records ownership events for later auditing."""

    def __init__(self, inner: RDBLeaseStore, workdir: str, executor_id: str) -> None:
        self._inner = inner
        self._workdir = workdir
        self._executor_id = executor_id

    def acquire(self, trial_id, study_id, executor_id, ttl, max_concurrent=None):
        record = self._inner.acquire(trial_id, study_id, executor_id, ttl, max_concurrent)
        log_event(
            self._workdir,
            {
                "event": "lease_acquired",
                "trial_id": trial_id,
                "executor": executor_id,
                "token": record.fencing_token,
                "expires_at": record.expires_at,
            },
        )
        return record

    def renew(self, record, ttl):
        return self._inner.renew(record, ttl)

    def release(self, record):
        self._inner.release(record)
        log_event(
            self._workdir,
            {
                "event": "lease_released",
                "trial_id": record.trial_id,
                "executor": record.executor_id,
                "token": record.fencing_token,
            },
        )

    def takeover(self, trial_id, executor_id, ttl):
        record = self._inner.takeover(trial_id, executor_id, ttl)
        log_event(
            self._workdir,
            {
                "event": "lease_takeover",
                "trial_id": trial_id,
                "executor": executor_id,
                "token": record.fencing_token,
                "expires_at": record.expires_at,
            },
        )
        return record

    def get(self, trial_id):
        return self._inner.get(trial_id)

    def check(self, record):
        return self._inner.check(record)

    def count_active(self, study_id):
        return self._inner.count_active(study_id)

    def list_active(self, study_id=None):
        return self._inner.list_active(study_id)


def make_objective(workdir: str, worker_index: int):
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        payload = trial.checkpoint_payload or {}
        suspends = payload.get("suspends", 0)
        for step in range(trial.start_step, N_STEPS):
            log_event(
                workdir,
                {
                    "event": "exec_step",
                    "trial_id": trial.trial_id,
                    "trial_number": trial.number,
                    "step": step,
                    "executor": f"machine-{worker_index}",
                },
            )
            value = expected_intermediate(x, step)
            trial.report(value, step)
            log_event(
                workdir,
                {
                    "event": "reported",
                    "trial_id": trial.trial_id,
                    "step": step,
                    "executor": f"machine-{worker_index}",
                },
            )
            time.sleep(STEP_SLEEP)
            # Fault injection 1: cooperative suspend (machine being reclaimed).
            if trial.number % 5 == 1 and not trial.resumed and step == SUSPEND_STEP:
                log_event(
                    workdir,
                    {
                        "event": "suspended",
                        "trial_id": trial.trial_id,
                        "step": step,
                        "executor": f"machine-{worker_index}",
                    },
                )
                trial.suspend({"suspends": suspends + 1})
            if (
                trial.number % 5 == 1
                and trial.resumed
                and suspends == 1
                and step == SUSPEND_STEP_2
            ):
                log_event(
                    workdir,
                    {
                        "event": "suspended",
                        "trial_id": trial.trial_id,
                        "step": step,
                        "executor": f"machine-{worker_index}",
                    },
                )
                trial.suspend({"suspends": suspends + 1})
            # Fault injection 2: hard crash with no cleanup at all.
            if worker_index == CRASH_WORKER and step == CRASH_STEP:
                log_event(
                    workdir,
                    {
                        "event": "crashing",
                        "trial_id": trial.trial_id,
                        "step": step,
                        "executor": f"machine-{worker_index}",
                    },
                )
                with open(os.path.join(workdir, "crash_trial"), "w") as f:
                    f.write(str(trial.trial_id))
                os._exit(1)
        return expected_value(x)

    return objective


def worker(workdir: str, worker_index: int, n_trials_target: int) -> None:
    url = f"sqlite:///{workdir}/optuna.db"
    storage = optuna.storages.RDBStorage(url=url, engine_kwargs={"connect_args": {"timeout": 30}})
    study = optuna.load_study(study_name="sim", storage=storage)
    executor_id = f"machine-{worker_index}"
    lease_store = LoggingLeaseStore(RDBLeaseStore(url), workdir, executor_id)
    executor = ResumableExecutor(
        study,
        executor_id=executor_id,
        lease_store=lease_store,
        checkpoint_store=FileCheckpointStore(os.path.join(workdir, "ckpts")),
        max_concurrent=N_WORKERS,
        lease_ttl=LEASE_TTL,
        heartbeat_interval=LEASE_TTL / 5,
        poll_interval=0.2,
    )
    objective = make_objective(workdir, worker_index)
    while True:
        trials = study.get_trials(deepcopy=False)
        states = [t.state for t in trials]
        finished = sum(1 for s in states if s.is_finished())
        waiting = any(s == TrialState.WAITING for s in states)
        pending = any(s in (TrialState.RUNNING, TrialState.WAITING) for s in states)
        if finished >= n_trials_target and not pending:
            break
        if len(states) >= n_trials_target and not waiting:
            # Enough trials exist and none needs a worker right now: idle-wait
            # for stragglers, but keep reclaiming dead machines' trials —
            # otherwise a crashed RUNNING trial would wait here forever.
            executor.reap_expired()
            time.sleep(0.2)
            continue
        executor.run(objective, n_trials=1, timeout=60)


def takeover_racer(workdir: str, trial_id: int, name: str, queue) -> None:
    url = f"sqlite:///{workdir}/optuna.db"
    store = RDBLeaseStore(url)
    try:
        record = store.takeover(trial_id, name, ttl=30.0)
        queue.put((name, "ok", record.fencing_token))
    except Exception as e:
        queue.put((name, type(e).__name__, None))


def read_events(workdir: str) -> list[dict[str, Any]]:
    path = os.path.join(workdir, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_simulation(workdir: str, n_trials: int = N_TRIALS) -> dict[str, Any]:
    """Run the full 8-machine simulation and return a verification report."""

    Path(workdir).mkdir(parents=True, exist_ok=True)
    os.makedirs(os.path.join(workdir, "ckpts"), exist_ok=True)
    url = f"sqlite:///{workdir}/optuna.db"
    storage = optuna.storages.RDBStorage(url=url, engine_kwargs={"connect_args": {"timeout": 30}})
    optuna.create_study(
        study_name="sim", storage=storage, sampler=optuna.samplers.RandomSampler(seed=0)
    )

    ctx = mp.get_context("fork")
    workers = [
        ctx.Process(target=worker, args=(workdir, i, n_trials), name=f"machine-{i}")
        for i in range(N_WORKERS)
    ]
    started = time.time()
    for w in workers:
        w.start()

    crash_time: float | None = None
    reaped_time: float | None = None
    max_active_leases = 0
    lease_store = RDBLeaseStore(url)
    study_id = storage.get_study_id_from_name("sim")
    while any(w.is_alive() for w in workers):
        for w in workers:
            w.join(timeout=0.2)
        max_active_leases = max(max_active_leases, lease_store.count_active(study_id))
        if crash_time is None and not workers[CRASH_WORKER].is_alive():
            if workers[CRASH_WORKER].exitcode not in (None, 0):
                crash_time = time.time()
        if crash_time is not None and reaped_time is None:
            crash_file = os.path.join(workdir, "crash_trial")
            if os.path.exists(crash_file):
                crash_trial_id = int(open(crash_file).read())
                state = storage.get_trial(crash_trial_id).state
                if state != TrialState.RUNNING:
                    reaped_time = time.time()
    for w in workers:
        w.join()

    duration = time.time() - started

    # ------------------------------------------------------------------
    # Scenario 3: duplicate takeover of one expired lease.
    # ------------------------------------------------------------------
    study = optuna.load_study(study_name="sim", storage=storage)
    orphan = study.ask()  # RUNNING trial abandoned by a "dead" machine
    orphan_id = orphan._trial_id
    dead_lease = lease_store.acquire(orphan_id, study_id, "dead-executor", ttl=0.3)
    time.sleep(0.4)
    queue = ctx.Queue()
    racers = [
        ctx.Process(target=takeover_racer, args=(workdir, orphan_id, f"reaper-{i}", queue))
        for i in range(2)
    ]
    for r in racers:
        r.start()
    results = sorted(queue.get(timeout=30) for _ in racers)
    for r in racers:
        r.join(timeout=30)
    takeover_outcomes = [(name, outcome) for name, outcome, _ in results]
    takeover_ok = [r for r in results if r[1] == "ok"]
    # The retired executor must be fenced out even if it revives.
    from optuna.resumable import LeaseLostError

    fenced = False
    try:
        lease_store.check(dead_lease)
    except LeaseLostError:
        fenced = True

    # ------------------------------------------------------------------
    # Verify invariants from the event log and the final study state.
    # ------------------------------------------------------------------
    events = read_events(workdir)

    # (a) At most one executor owns a trial at any time.
    intervals: dict[int, list[tuple[float, float, str]]] = {}
    open_lease: dict[int, list[Any]] = {}
    for e in events:
        if e["event"] in ("lease_acquired", "lease_takeover"):
            open_lease[e["trial_id"]] = [e["ts"], e.get("expires_at"), e["executor"]]
        elif e["event"] == "lease_released":
            opened = open_lease.pop(e["trial_id"], None)
            if opened is not None:
                intervals.setdefault(e["trial_id"], []).append((opened[0], e["ts"], opened[2]))
    for trial_id, (start, expires_at, executor) in open_lease.items():
        intervals.setdefault(trial_id, []).append(
            (start, expires_at if expires_at is not None else float("inf"), executor)
        )
    overlaps = []
    for trial_id, spans in intervals.items():
        spans.sort()
        for (s1, e1, x1), (s2, e2, x2) in zip(spans, spans[1:]):
            if s2 < e1 - 1e-6 and x1 != x2:
                overlaps.append((trial_id, x1, x2))

    # (b) No *completed* training step was ever executed twice. A step counts
    # as completed once its report (and with it the checkpoint) is durable;
    # re-executing a step that was still in flight when its executor lost the
    # lease is inherent to failover and is tracked separately.
    reported_steps: set[tuple[int, int]] = set()
    duplicate_completed_executions = 0
    inflight_reexecutions = 0
    for e in events:
        if e["event"] == "reported":
            reported_steps.add((e["trial_id"], e["step"]))
        elif e["event"] == "exec_step":
            key = (e["trial_id"], e["step"])
            if key in reported_steps:
                duplicate_completed_executions += 1
    step_counts: dict[tuple[int, int], int] = {}
    for e in events:
        if e["event"] == "exec_step":
            key = (e["trial_id"], e["step"])
            step_counts[key] = step_counts.get(key, 0) + 1
    total_reexecutions = sum(c - 1 for c in step_counts.values())
    inflight_reexecutions = total_reexecutions - duplicate_completed_executions

    # (c) Final values and histories match the uninterrupted computation.
    trials = study.get_trials(deepcopy=False)
    complete = [t for t in trials if t.state == TrialState.COMPLETE]
    value_mismatches = []
    history_mismatches = []
    for t in complete:
        x = t.params["x"]
        if not math.isclose(t.value, expected_value(x), rel_tol=1e-9, abs_tol=1e-9):
            value_mismatches.append(t.number)
        for step, v in t.intermediate_values.items():
            if not math.isclose(v, expected_intermediate(x, step), rel_tol=1e-9, abs_tol=1e-9):
                history_mismatches.append((t.number, step))
        if sorted(t.intermediate_values) != list(range(N_STEPS)):
            history_mismatches.append((t.number, "incomplete"))

    suspended_events = [e for e in events if e["event"] == "suspended"]
    crash_events = [e for e in events if e["event"] == "crashing"]
    crash_trial_id = None
    crash_trial_complete = None
    if crash_events:
        crash_trial_id = crash_events[0]["trial_id"]
        crash_trial = storage.get_trial(crash_trial_id)
        crash_trial_complete = crash_trial.state == TrialState.COMPLETE

    report = {
        "workers": N_WORKERS,
        "n_trials_target": n_trials,
        "duration_seconds": round(duration, 2),
        "trials_total": len(trials),
        "trials_complete": len(complete),
        "suspends": len(suspended_events),
        "crashes": len(crash_events),
        "crash_trial_id": crash_trial_id,
        "crash_trial_complete": crash_trial_complete,
        "reap_latency_seconds": (
            round(reaped_time - crash_time, 2)
            if crash_time is not None and reaped_time is not None
            else None
        ),
        "max_active_leases": max_active_leases,
        "lease_overlaps": overlaps,
        "duplicate_step_executions": duplicate_completed_executions,
        "inflight_step_reexecutions": inflight_reexecutions,
        "value_mismatches": value_mismatches,
        "history_mismatches": history_mismatches,
        "takeover_race": takeover_outcomes,
        "takeover_exactly_one_winner": len(takeover_ok) == 1,
        "stale_executor_fenced": fenced,
    }
    report["ok"] = (
        not overlaps
        and duplicate_completed_executions == 0
        and not value_mismatches
        and not history_mismatches
        and len(complete) >= n_trials
        and crash_trial_complete is True
        and report["takeover_exactly_one_winner"]
        and fenced
        and report["reap_latency_seconds"] is not None
        and report["reap_latency_seconds"] < 3 * LEASE_TTL + 10
    )
    return report


def run_baseline_comparison(workdir: str, n_trials: int = N_TRIALS) -> list[dict[str, Any]]:
    """Re-run every completed simulation trial uninterrupted with identical params.

    Each parameter set produced by the cluster is enqueued into a fresh study
    and optimized without any suspension; the final objective values must be
    identical to the ones the suspend/resume cluster produced.
    """

    url = f"sqlite:///{workdir}/optuna.db"
    storage = optuna.storages.RDBStorage(url=url, engine_kwargs={"connect_args": {"timeout": 30}})
    sim = optuna.load_study(study_name="sim", storage=storage)
    sim_trials = sorted(
        (t for t in sim.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE),
        key=lambda t: t.number,
    )
    baseline = optuna.create_study(
        study_name="baseline", storage=storage, sampler=optuna.samplers.RandomSampler(seed=0)
    )
    for t in sim_trials:
        baseline.enqueue_trial({"x": t.params["x"]})

    def plain_objective(trial):
        x = trial.suggest_float("x", -10, 10)
        for step in range(N_STEPS):
            trial.report(expected_intermediate(x, step), step)
        return expected_value(x)

    baseline.optimize(plain_objective, n_trials=len(sim_trials))

    baseline_trials = sorted(
        (t for t in baseline.get_trials(deepcopy=False) if t.state == TrialState.COMPLETE),
        key=lambda t: t.number,
    )
    rows = []
    for st, bt in zip(sim_trials, baseline_trials):
        rows.append(
            {
                "x": st.params["x"],
                "baseline_value": bt.value,
                "sim_value": st.value,
                "equal": math.isclose(st.value, bt.value, rel_tol=1e-9, abs_tol=1e-9),
            }
        )
    return rows
