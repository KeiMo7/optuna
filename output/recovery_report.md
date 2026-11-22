# Resumable execution — recovery report

## Cluster simulation (8 machines, local processes + shared SQLite + shared checkpoint dir)

| metric | value |
| --- | --- |
| worker processes ("machines") | 8 |
| trials targeted / created / complete | 16 / 29 / 28 |
| wall-clock duration | 11.7 s |
| cooperative suspends (mid-training) | 12 |
| hard crashes (os._exit, no cleanup) | 1 |
| crashed trial id / completed after takeover | 2 / True |
| crash reap latency (bounded) | 1.41 s |
| max concurrently active leases (slots) | 6 |

## Invariant assertions

| invariant | result |
| --- | --- |
| at most one executor holds a trial at any time (lease intervals never overlap) | PASS |
| duplicate executions of completed steps (reported + checkpointed) | 0 (must be 0) |
| in-flight steps re-executed after genuine failovers (informational) | 0 |
| final objective values equal uninterrupted computation | PASS |
| intermediate-value histories complete and unaltered | PASS |
| duplicate takeover race: exactly one winner | PASS [('reaper-0', 'TakeoverRejectedError'), ('reaper-1', 'ok')] |
| revived stale executor fenced out (writes rejected) | PASS |

Overall: **PASS**

See `simulation_report.json` for the raw report and `objective_comparison.txt`
for the per-trial baseline comparison.
