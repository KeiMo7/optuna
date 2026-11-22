"""Acceptance runner: executes the 8-machine simulation and writes output/ artifacts.

Usage:  python tests/resumable_tests/acceptance.py [output_dir]
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.resumable_tests.cluster_sim import run_baseline_comparison
from tests.resumable_tests.cluster_sim import run_simulation


def main() -> int:
    output_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="clustersim-acceptance-", dir=str(Path.home()))

    print(f"[acceptance] running 8-machine simulation in {workdir} ...")
    report = run_simulation(workdir)
    print(f"[acceptance] simulation finished: ok={report['ok']}")

    print("[acceptance] running uninterrupted baseline comparison ...")
    rows = run_baseline_comparison(workdir)

    (output_dir / "simulation_report.json").write_text(json.dumps(report, indent=1) + "\n")

    lines = [
        "Objective value comparison: uninterrupted baseline vs suspend/resume cluster",
        "(trials matched by identical sampled parameter x)",
        "",
        f"{'x':>22}  {'baseline':>18}  {'cluster (suspended/resumed)':>27}  equal",
    ]
    for row in rows:
        sim_value = f"{row['sim_value']:.10f}" if row["sim_value"] is not None else "missing"
        lines.append(
            f"{row['x']:>22.15f}  {row['baseline_value']:>18.10f}  {sim_value:>27}  {row['equal']}"
        )
    matched = sum(1 for r in rows if r["sim_value"] is not None)
    lines += [
        "",
        f"matched trials: {matched}/{len(rows)}; all equal: {all(r['equal'] for r in rows)}",
    ]
    (output_dir / "objective_comparison.txt").write_text("\n".join(lines) + "\n")

    def pf(ok: bool, fail_detail: object = "") -> str:
        return "PASS" if ok else f"FAIL: {fail_detail}"

    trial_counts = (
        f"{report['n_trials_target']} / {report['trials_total']} / {report['trials_complete']}"
    )
    crash_info = f"{report['crash_trial_id']} / {report['crash_trial_complete']}"
    r_lease = pf(not report["lease_overlaps"], report["lease_overlaps"])
    r_values = pf(not report["value_mismatches"], report["value_mismatches"])
    r_history = pf(not report["history_mismatches"], report["history_mismatches"])
    r_takeover = pf(report["takeover_exactly_one_winner"]) + f" {report['takeover_race']}"
    r_fenced = pf(report["stale_executor_fenced"])
    n_dup_steps = report["duplicate_step_executions"]
    n_inflight = report["inflight_step_reexecutions"]
    md = f"""# Resumable execution — recovery report

## Cluster simulation (8 machines, local processes + shared SQLite + shared checkpoint dir)

| metric | value |
| --- | --- |
| worker processes ("machines") | {report["workers"]} |
| trials targeted / created / complete | {trial_counts} |
| wall-clock duration | {report["duration_seconds"]} s |
| cooperative suspends (mid-training) | {report["suspends"]} |
| hard crashes (os._exit, no cleanup) | {report["crashes"]} |
| crashed trial id / completed after takeover | {crash_info} |
| crash reap latency (bounded) | {report["reap_latency_seconds"]} s |
| max concurrently active leases (slots) | {report["max_active_leases"]} |

## Invariant assertions

| invariant | result |
| --- | --- |
| at most one executor holds a trial at any time (lease intervals never overlap) | {r_lease} |
| duplicate executions of completed steps (reported + checkpointed) | {n_dup_steps} (must be 0) |
| in-flight steps re-executed after genuine failovers (informational) | {n_inflight} |
| final objective values equal uninterrupted computation | {r_values} |
| intermediate-value histories complete and unaltered | {r_history} |
| duplicate takeover race: exactly one winner | {r_takeover} |
| revived stale executor fenced out (writes rejected) | {r_fenced} |

Overall: **{"PASS" if report["ok"] else "FAIL"}**

See `simulation_report.json` for the raw report and `objective_comparison.txt`
for the per-trial baseline comparison.
"""
    (output_dir / "recovery_report.md").write_text(md)
    print(f"[acceptance] artifacts written to {output_dir}/")
    shutil.rmtree(workdir, ignore_errors=True)
    return 0 if report["ok"] and all(r["equal"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
