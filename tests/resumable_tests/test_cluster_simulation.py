from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

from tests.resumable_tests.cluster_sim import N_WORKERS
from tests.resumable_tests.cluster_sim import run_baseline_comparison
from tests.resumable_tests.cluster_sim import run_simulation


def _workdir() -> str:
    # /tmp may be a small tmpfs; keep the shared SQLite DB on a real filesystem.
    return tempfile.mkdtemp(prefix="clustersim-test-", dir=str(Path.home()))


def test_eight_machine_cluster_simulation():
    workdir = _workdir()
    try:
        report = run_simulation(workdir)
        assert report["workers"] == N_WORKERS == 8
        # At most one executor holds a trial at any moment.
        assert report["lease_overlaps"] == []
        # No completed (reported + checkpointed) step was ever executed twice.
        assert report["duplicate_step_executions"] == 0
        # Suspended and crashed trials finished with the uninterrupted values.
        assert report["value_mismatches"] == []
        assert report["history_mismatches"] == []
        assert report["trials_complete"] >= report["n_trials_target"]
        # The crashed machine's trial was reclaimed in bounded time and finished.
        assert report["crashes"] == 1
        assert report["crash_trial_complete"] is True
        assert report["reap_latency_seconds"] is not None
        # Cooperative suspends happened and were resumed.
        assert report["suspends"] >= 1
        # Duplicate takeover: exactly one winner, the loser was rejected.
        assert report["takeover_exactly_one_winner"] is True
        outcomes = sorted(outcome for _, outcome in report["takeover_race"])
        assert outcomes == ["TakeoverRejectedError", "ok"]
        # The retired executor is fenced out even if it revives.
        assert report["stale_executor_fenced"] is True
        assert report["ok"] is True

        rows = run_baseline_comparison(workdir)
        assert rows and all(row["equal"] for row in rows)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
