from __future__ import annotations

import threading
import time

import pytest

from optuna.resumable import InMemoryLeaseStore
from optuna.resumable import LeaseConflictError
from optuna.resumable import LeaseLostError
from optuna.resumable import RDBLeaseStore
from optuna.resumable import SlotExhaustedError
from optuna.resumable import TakeoverRejectedError


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_stores(tmp_path):
    clock = FakeClock()
    yield "memory", InMemoryLeaseStore(now=clock), clock
    url = f"sqlite:///{tmp_path}/leases.db"
    yield "rdb", RDBLeaseStore(url, now=clock), clock


@pytest.fixture(params=["memory", "rdb"])
def store_and_clock(request, tmp_path):
    clock = FakeClock()
    if request.param == "memory":
        return InMemoryLeaseStore(now=clock), clock
    return RDBLeaseStore(f"sqlite:///{tmp_path}/leases.db", now=clock), clock


def test_acquire_and_release(store_and_clock):
    store, _ = store_and_clock
    rec = store.acquire(1, 10, "exec-a", ttl=30.0)
    assert rec.fencing_token == 1
    assert store.count_active(10) == 1
    store.release(rec)
    assert store.get(1) is None
    assert store.count_active(10) == 0


def test_conflict_while_lease_active(store_and_clock):
    store, _ = store_and_clock
    store.acquire(1, 10, "exec-a", ttl=30.0)
    with pytest.raises(LeaseConflictError) as excinfo:
        store.acquire(1, 10, "exec-b", ttl=30.0)
    message = str(excinfo.value)
    assert "exec-a" in message and "exec-b" in message and "1" in message


def test_slot_limit_enforced(store_and_clock):
    store, _ = store_and_clock
    store.acquire(1, 10, "exec-a", ttl=30.0, max_concurrent=1)
    with pytest.raises(SlotExhaustedError):
        store.acquire(2, 10, "exec-b", ttl=30.0, max_concurrent=1)
    # A different study is not affected.
    store.acquire(2, 11, "exec-b", ttl=30.0, max_concurrent=1)


def test_expired_lease_frees_slot_and_is_reaped(store_and_clock):
    store, clock = store_and_clock
    store.acquire(1, 10, "exec-a", ttl=10.0, max_concurrent=1)
    clock.advance(11.0)
    assert store.count_active(10) == 0
    rec = store.acquire(1, 10, "exec-b", ttl=10.0, max_concurrent=1)
    assert rec.executor_id == "exec-b"
    assert rec.fencing_token == 2  # token moved forward, old holder stays fenced out


def test_renew_extends_lease(store_and_clock):
    store, clock = store_and_clock
    rec = store.acquire(1, 10, "exec-a", ttl=10.0)
    clock.advance(8.0)
    rec = store.renew(rec, ttl=10.0)
    clock.advance(8.0)
    store.check(rec)  # still valid because of the renewal
    clock.advance(3.0)
    with pytest.raises(LeaseLostError):
        store.check(rec)


def test_takeover_only_after_expiry_and_only_once(store_and_clock):
    store, clock = store_and_clock
    store.acquire(1, 10, "exec-a", ttl=10.0)
    with pytest.raises(TakeoverRejectedError):
        store.takeover(1, "exec-b", ttl=10.0)
    clock.advance(11.0)
    taken = store.takeover(1, "exec-b", ttl=10.0)
    assert taken.fencing_token == 2
    # A second takeover of the fresh lease is rejected.
    with pytest.raises(TakeoverRejectedError) as excinfo:
        store.takeover(1, "exec-c", ttl=10.0)
    assert "exec-c" in str(excinfo.value)


def test_concurrent_takeover_exactly_one_winner(store_and_clock):
    store, clock = store_and_clock
    store.acquire(1, 10, "exec-a", ttl=10.0)
    clock.advance(11.0)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def attempt(name: str) -> None:
        barrier.wait(timeout=10)
        try:
            store.takeover(1, name, ttl=10.0)
            outcomes.append(f"{name}:ok")
        except TakeoverRejectedError:
            outcomes.append(f"{name}:rejected")

    threads = [threading.Thread(target=attempt, args=(f"exec-{c}",)) for c in "bc"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["exec-b:ok", "exec-c:rejected"] or sorted(outcomes) == [
        "exec-b:rejected",
        "exec-c:ok",
    ]


def test_stale_token_cannot_write_after_takeover(store_and_clock):
    store, clock = store_and_clock
    old = store.acquire(1, 10, "exec-a", ttl=10.0)
    clock.advance(11.0)
    store.takeover(1, "exec-b", ttl=10.0)
    # The old executor comes back to life and tries to write / renew.
    with pytest.raises(LeaseLostError) as excinfo:
        store.check(old)
    assert "exec-a" in str(excinfo.value)
    with pytest.raises(LeaseLostError):
        store.renew(old, ttl=10.0)
    # Releasing with a stale record must not delete the new lease.
    store.release(old)
    current = store.get(1)
    assert current is not None and current.executor_id == "exec-b"


def test_release_frees_slot(store_and_clock):
    store, _ = store_and_clock
    rec = store.acquire(1, 10, "exec-a", ttl=30.0, max_concurrent=1)
    store.release(rec)
    store.acquire(2, 10, "exec-b", ttl=30.0, max_concurrent=1)


def test_lease_without_ttl_never_expires(store_and_clock):
    store, clock = store_and_clock
    rec = store.acquire(1, 10, "exec-a", ttl=None)
    clock.advance(10**6)
    store.check(rec)
    with pytest.raises(TakeoverRejectedError):
        store.takeover(1, "exec-b", ttl=10.0)


def test_rdb_lease_store_shared_across_connections(tmp_path):
    # Two independent store objects on the same database file behave as one ledger.
    url = f"sqlite:///{tmp_path}/shared.db"
    store_a = RDBLeaseStore(url)
    store_b = RDBLeaseStore(url)
    rec = store_a.acquire(1, 10, "exec-a", ttl=30.0)
    with pytest.raises(LeaseConflictError):
        store_b.acquire(1, 10, "exec-b", ttl=30.0)
    assert store_b.count_active(10) == 1
    store_a.release(rec)
    assert store_b.count_active(10) == 0


def test_rdb_lease_store_real_expiry(tmp_path):
    url = f"sqlite:///{tmp_path}/expiry.db"
    store = RDBLeaseStore(url)
    store.acquire(1, 10, "exec-a", ttl=0.2)
    time.sleep(0.3)
    taken = store.takeover(1, "exec-b", ttl=30.0)
    assert taken.executor_id == "exec-b"
