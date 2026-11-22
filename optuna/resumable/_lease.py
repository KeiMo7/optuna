"""Lease ledger: the single source of truth for trial ownership and concurrency slots.

A *lease* row ``(trial_id, study_id, executor_id, fencing_token, expires_at)`` means
"trial ``trial_id`` is currently executed by ``executor_id``". Because exactly one
unexpired lease can exist per trial, and because the concurrency slot of a study is
defined as "one active lease", trial ownership and concurrency accounting can never
diverge: they are the same rows.

Two implementations are provided:

- :class:`InMemoryLeaseStore` for single-process use (no extra files, no heartbeat
  traffic beyond a thread).
- :class:`RDBLeaseStore` which keeps the ledger in the same relational database as
  the trial data (typically the same URL as :class:`~optuna.storages.RDBStorage`),
  so multi-process / multi-machine clusters share one consistent ledger.

Fencing tokens are monotonically increased on every ownership change. Any write
that presents a stale token is rejected with :class:`LeaseLostError`, which gives
the at-most-one-writer guarantee even if a retired executor briefly comes back.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
import threading
import time
from typing import Any
from typing import NamedTuple

from optuna.resumable._errors import LeaseConflictError
from optuna.resumable._errors import LeaseLostError
from optuna.resumable._errors import SlotExhaustedError
from optuna.resumable._errors import TakeoverRejectedError


class LeaseRecord(NamedTuple):
    """A snapshot of one lease row."""

    trial_id: int
    study_id: int
    executor_id: str
    fencing_token: int
    expires_at: float | None  # Epoch seconds; ``None`` means "never expires".

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at is None:
            return False
        return (time.time() if now is None else now) >= self.expires_at


class BaseLeaseStore(metaclass=abc.ABCMeta):
    """Interface of the lease ledger used by :class:`~optuna.resumable.ResumableExecutor`."""

    @abc.abstractmethod
    def acquire(
        self,
        trial_id: int,
        study_id: int,
        executor_id: str,
        ttl: float | None,
        max_concurrent: int | None = None,
    ) -> LeaseRecord:
        """Atomically take ownership of ``trial_id``, consuming one concurrency slot.

        Raises:
            LeaseConflictError: If an unexpired lease for the trial already exists.
            SlotExhaustedError: If ``max_concurrent`` active leases already exist for
                the study.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def renew(self, record: LeaseRecord, ttl: float | None) -> LeaseRecord:
        """Extend the lease. Raises :class:`LeaseLostError` if it is no longer ours."""
        raise NotImplementedError

    @abc.abstractmethod
    def release(self, record: LeaseRecord) -> None:
        """Give the lease (and its concurrency slot) back. Best effort, never raises."""
        raise NotImplementedError

    @abc.abstractmethod
    def takeover(self, trial_id: int, executor_id: str, ttl: float | None) -> LeaseRecord:
        """Atomically claim an *expired* lease, bumping its fencing token.

        Exactly one of any set of concurrent takeover attempts succeeds; the others
        get :class:`TakeoverRejectedError`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, trial_id: int) -> LeaseRecord | None:
        """Return the current lease row of the trial, or ``None``."""
        raise NotImplementedError

    @abc.abstractmethod
    def check(self, record: LeaseRecord) -> None:
        """Verify that ``record`` is still the valid lease; raise :class:`LeaseLostError`."""
        raise NotImplementedError

    @abc.abstractmethod
    def count_active(self, study_id: int) -> int:
        """Return the number of unexpired leases of the study (i.e. occupied slots)."""
        raise NotImplementedError

    @abc.abstractmethod
    def list_active(self, study_id: int | None = None) -> list[LeaseRecord]:
        """Return all unexpired leases, optionally restricted to one study."""
        raise NotImplementedError


def _check_record(current: LeaseRecord | None, record: LeaseRecord, now: float) -> None:
    if current is None:
        raise LeaseLostError(
            record.trial_id, record.executor_id, record.fencing_token, "the lease was released"
        )
    if current.executor_id != record.executor_id or current.fencing_token != record.fencing_token:
        raise LeaseLostError(
            record.trial_id,
            record.executor_id,
            record.fencing_token,
            "the trial was taken over",
            current_holder=current.executor_id,
        )
    if current.is_expired(now):
        raise LeaseLostError(
            record.trial_id,
            record.executor_id,
            record.fencing_token,
            "the lease expired",
            current_holder=current.executor_id,
        )


class InMemoryLeaseStore(BaseLeaseStore):
    """Process-local lease ledger. Suitable for single-process executors only."""

    def __init__(self, *, now: Callable[[], float] | None = None) -> None:
        self._records: dict[int, LeaseRecord] = {}
        self._lock = threading.Lock()
        self._now = now or time.time

    def acquire(
        self,
        trial_id: int,
        study_id: int,
        executor_id: str,
        ttl: float | None,
        max_concurrent: int | None = None,
    ) -> LeaseRecord:
        with self._lock:
            now = self._now()
            current = self._records.get(trial_id)
            if current is not None and current.is_expired(now):
                # Reap a stale row left by a crashed executor in the same step.
                return self._takeover_locked(trial_id, study_id, executor_id, ttl, now)
            if current is not None:
                raise LeaseConflictError(trial_id, current.executor_id, executor_id)
            if max_concurrent is not None and self._count_active_locked(study_id, now) >= (
                max_concurrent
            ):
                raise SlotExhaustedError(study_id, max_concurrent, executor_id)
            record = LeaseRecord(
                trial_id=trial_id,
                study_id=study_id,
                executor_id=executor_id,
                fencing_token=1,
                expires_at=None if ttl is None else now + ttl,
            )
            self._records[trial_id] = record
            return record

    def _takeover_locked(
        self, trial_id: int, study_id: int, executor_id: str, ttl: float | None, now: float
    ) -> LeaseRecord:
        current = self._records.get(trial_id)
        if current is None or not current.is_expired(now):
            raise TakeoverRejectedError(
                trial_id, executor_id, current.executor_id if current else None
            )
        record = LeaseRecord(
            trial_id=trial_id,
            study_id=study_id,
            executor_id=executor_id,
            fencing_token=current.fencing_token + 1,
            expires_at=None if ttl is None else now + ttl,
        )
        self._records[trial_id] = record
        return record

    def renew(self, record: LeaseRecord, ttl: float | None) -> LeaseRecord:
        with self._lock:
            now = self._now()
            _check_record(self._records.get(record.trial_id), record, now)
            renewed = record._replace(expires_at=None if ttl is None else now + ttl)
            self._records[record.trial_id] = renewed
            return renewed

    def release(self, record: LeaseRecord) -> None:
        with self._lock:
            current = self._records.get(record.trial_id)
            if (
                current is not None
                and current.executor_id == record.executor_id
                and current.fencing_token == record.fencing_token
            ):
                del self._records[record.trial_id]

    def takeover(self, trial_id: int, executor_id: str, ttl: float | None) -> LeaseRecord:
        with self._lock:
            current = self._records.get(trial_id)
            return self._takeover_locked(
                trial_id, current.study_id if current else -1, executor_id, ttl, self._now()
            )

    def get(self, trial_id: int) -> LeaseRecord | None:
        with self._lock:
            return self._records.get(trial_id)

    def check(self, record: LeaseRecord) -> None:
        with self._lock:
            _check_record(self._records.get(record.trial_id), record, self._now())

    def _count_active_locked(self, study_id: int, now: float) -> int:
        return sum(
            1 for r in self._records.values() if r.study_id == study_id and not r.is_expired(now)
        )

    def count_active(self, study_id: int) -> int:
        with self._lock:
            return self._count_active_locked(study_id, self._now())

    def list_active(self, study_id: int | None = None) -> list[LeaseRecord]:
        with self._lock:
            now = self._now()
            return [
                r
                for r in self._records.values()
                if not r.is_expired(now) and (study_id is None or r.study_id == study_id)
            ]


_TABLE_NAME = "resumable_leases"


class RDBLeaseStore(BaseLeaseStore):
    """Lease ledger stored in a relational database, shared across processes.

    Point it at the same database URL as the :class:`~optuna.storages.RDBStorage`
    of the study so that trial state, leases and concurrency slots live in one
    ledger. All mutating operations are single atomic SQL statements, so the
    guarantees hold across machines connected to the same database.
    """

    def __init__(
        self,
        url: str,
        *,
        table_name: str = _TABLE_NAME,
        engine_kwargs: dict[str, Any] | None = None,
        now: Callable[[], float] | None = None,
    ) -> None:
        import sqlalchemy

        self._table_name = table_name
        self._now = now or time.time
        engine_kwargs = dict(engine_kwargs or {})
        if url.startswith("sqlite"):
            engine_kwargs.setdefault("connect_args", {"timeout": 30})
            if url.endswith(":memory:"):
                engine_kwargs.setdefault("poolclass", sqlalchemy.pool.StaticPool)
        self._engine = sqlalchemy.create_engine(url, **engine_kwargs)
        self._metadata = sqlalchemy.MetaData()
        self._table = sqlalchemy.Table(
            table_name,
            self._metadata,
            sqlalchemy.Column("trial_id", sqlalchemy.Integer, primary_key=True),
            sqlalchemy.Column("study_id", sqlalchemy.Integer, index=True, nullable=False),
            sqlalchemy.Column("executor_id", sqlalchemy.String(255), nullable=False),
            sqlalchemy.Column("fencing_token", sqlalchemy.Integer, nullable=False),
            sqlalchemy.Column("expires_at", sqlalchemy.Float, nullable=True),
        )
        self._metadata.create_all(self._engine)
        if url.startswith("sqlite") and not url.endswith(":memory:"):
            with self._engine.connect() as conn:
                conn.exec_driver_sql("PRAGMA journal_mode=WAL")

    @staticmethod
    def _row_to_record(row: Any) -> LeaseRecord:
        return LeaseRecord(
            trial_id=row[0],
            study_id=row[1],
            executor_id=row[2],
            fencing_token=row[3],
            expires_at=row[4],
        )

    def _select(self, conn: Any, trial_id: int) -> LeaseRecord | None:
        from sqlalchemy import text

        row = conn.execute(
            text(
                f"SELECT trial_id, study_id, executor_id, fencing_token, expires_at "
                f"FROM {self._table_name} WHERE trial_id = :tid"
            ),
            {"tid": trial_id},
        ).first()
        return None if row is None else self._row_to_record(row)

    def acquire(
        self,
        trial_id: int,
        study_id: int,
        executor_id: str,
        ttl: float | None,
        max_concurrent: int | None = None,
    ) -> LeaseRecord:
        from sqlalchemy import text

        now = self._now()
        expires_at = None if ttl is None else now + ttl
        with self._engine.begin() as conn:
            # First, atomically revive an expired row if one exists (token + 1 keeps
            # fencing tokens monotone, so the crashed holder stays fenced out).
            revived = conn.execute(
                text(
                    f"UPDATE {self._table_name} "
                    f"SET executor_id = :eid, fencing_token = fencing_token + 1, "
                    f"    expires_at = :exp, study_id = :sid "
                    f"WHERE trial_id = :tid AND expires_at IS NOT NULL AND expires_at <= :now"
                ),
                {
                    "eid": executor_id,
                    "exp": expires_at,
                    "sid": study_id,
                    "tid": trial_id,
                    "now": now,
                },
            )
            if revived.rowcount == 0:
                limit = -1 if max_concurrent is None else int(max_concurrent)
                inserted = conn.execute(
                    text(
                        f"INSERT INTO {self._table_name} "
                        f"(trial_id, study_id, executor_id, fencing_token, expires_at) "
                        f"SELECT :tid, :sid, :eid, 1, :exp "
                        f"WHERE NOT EXISTS "
                        f"(SELECT 1 FROM {self._table_name} WHERE trial_id = :tid) "
                        f"AND (:lim < 0 OR (SELECT COUNT(*) FROM {self._table_name} "
                        f"     WHERE study_id = :sid "
                        f"       AND (expires_at IS NULL OR expires_at > :now)) < :lim)"
                    ),
                    {
                        "tid": trial_id,
                        "sid": study_id,
                        "eid": executor_id,
                        "exp": expires_at,
                        "lim": limit,
                        "now": now,
                    },
                )
                if inserted.rowcount != 1:
                    current = self._select(conn, trial_id)
                    if current is not None:
                        raise LeaseConflictError(trial_id, current.executor_id, executor_id)
                    raise SlotExhaustedError(
                        study_id, -1 if max_concurrent is None else max_concurrent, executor_id
                    )
            record = self._select(conn, trial_id)
            assert record is not None
            return record

    def renew(self, record: LeaseRecord, ttl: float | None) -> LeaseRecord:
        from sqlalchemy import text

        now = self._now()
        expires_at = None if ttl is None else now + ttl
        with self._engine.begin() as conn:
            updated = conn.execute(
                text(
                    f"UPDATE {self._table_name} SET expires_at = :exp "
                    f"WHERE trial_id = :tid AND executor_id = :eid "
                    f"AND fencing_token = :tok "
                    f"AND (expires_at IS NULL OR expires_at > :now)"
                ),
                {
                    "exp": expires_at,
                    "tid": record.trial_id,
                    "eid": record.executor_id,
                    "tok": record.fencing_token,
                    "now": now,
                },
            )
            if updated.rowcount != 1:
                _check_record(self._select(conn, record.trial_id), record, now)
                raise LeaseLostError(  # pragma: no cover - _check_record always raises here
                    record.trial_id, record.executor_id, record.fencing_token, "renew failed"
                )
            return record._replace(expires_at=expires_at)

    def release(self, record: LeaseRecord) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(
                    f"DELETE FROM {self._table_name} "
                    f"WHERE trial_id = :tid AND executor_id = :eid AND fencing_token = :tok"
                ),
                {"tid": record.trial_id, "eid": record.executor_id, "tok": record.fencing_token},
            )

    def takeover(self, trial_id: int, executor_id: str, ttl: float | None) -> LeaseRecord:
        from sqlalchemy import text

        now = self._now()
        expires_at = None if ttl is None else now + ttl
        with self._engine.begin() as conn:
            updated = conn.execute(
                text(
                    f"UPDATE {self._table_name} "
                    f"SET executor_id = :eid, fencing_token = fencing_token + 1, "
                    f"    expires_at = :exp "
                    f"WHERE trial_id = :tid AND expires_at IS NOT NULL AND expires_at <= :now"
                ),
                {"eid": executor_id, "exp": expires_at, "tid": trial_id, "now": now},
            )
            if updated.rowcount != 1:
                current = self._select(conn, trial_id)
                raise TakeoverRejectedError(
                    trial_id, executor_id, current.executor_id if current else None
                )
            record = self._select(conn, trial_id)
            assert record is not None
            return record

    def get(self, trial_id: int) -> LeaseRecord | None:
        with self._engine.begin() as conn:
            return self._select(conn, trial_id)

    def check(self, record: LeaseRecord) -> None:
        with self._engine.begin() as conn:
            _check_record(self._select(conn, record.trial_id), record, self._now())

    def count_active(self, study_id: int) -> int:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            return int(
                conn.execute(
                    text(
                        f"SELECT COUNT(*) FROM {self._table_name} "
                        f"WHERE study_id = :sid AND (expires_at IS NULL OR expires_at > :now)"
                    ),
                    {"sid": study_id, "now": self._now()},
                ).scalar_one()
            )

    def list_active(self, study_id: int | None = None) -> list[LeaseRecord]:
        from sqlalchemy import text

        sql = (
            f"SELECT trial_id, study_id, executor_id, fencing_token, expires_at "
            f"FROM {self._table_name} WHERE (expires_at IS NULL OR expires_at > :now)"
        )
        params: dict[str, Any] = {"now": self._now()}
        if study_id is not None:
            sql += " AND study_id = :sid"
            params["sid"] = study_id
        with self._engine.begin() as conn:
            return [self._row_to_record(row) for row in conn.execute(text(sql), params)]
