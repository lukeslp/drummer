"""Transactional, conservative project spending reservations; not a provider cap."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

MICROS = 1_000_000
TRANCHES = {"smoke": 20, "training": 100, "evaluation": 40, "interop": 25, "reserve": 65}


class BudgetError(ValueError):
    pass


def micro_usd(amount: str | float | Decimal) -> int:
    value = Decimal(str(amount))
    if not value.is_finite() or value < 0:
        raise BudgetError("Cost must be finite and non-negative")
    return int((value * MICROS).to_integral_value(rounding=ROUND_CEILING))


class BudgetLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY, tranche TEXT NOT NULL, kind TEXT NOT NULL,
                reserved INTEGER NOT NULL, charged INTEGER,
                status TEXT NOT NULL, metadata TEXT NOT NULL,
                created TEXT DEFAULT CURRENT_TIMESTAMP)""")

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def reserve(self, tranche: str, maximum: str | float | Decimal, *, kind="job",
                metadata: dict | None = None, entry_id: str | None = None) -> str:
        if tranche not in TRANCHES or kind not in {"job", "expense"}:
            raise BudgetError("Unknown tranche or reservation kind")
        amount = micro_usd(maximum)
        if amount == 0:
            raise BudgetError("Reservations must be positive")
        entry_id = entry_id or str(uuid.uuid4())
        with self._connection() as db:
            if kind == "job" and db.execute(
                "SELECT COUNT(*) FROM entries WHERE kind='job' AND status IN ('reserved','submitted','uncertain')"
            ).fetchone()[0]:
                raise BudgetError("An unresolved job already holds the single-job slot")
            total, partial = db.execute(
                "SELECT COALESCE(SUM(COALESCE(charged,reserved)),0), "
                "COALESCE(SUM(CASE WHEN tranche=? THEN COALESCE(charged,reserved) ELSE 0 END),0) FROM entries",
                (tranche,),
            ).fetchone()
            if total + amount > 250 * MICROS or partial + amount > TRANCHES[tranche] * MICROS:
                raise BudgetError("Reservation would exceed project or tranche ceiling")
            db.execute("INSERT INTO entries(id,tranche,kind,reserved,status,metadata) VALUES(?,?,?,?,?,?)",
                       (entry_id, tranche, kind, amount, "reserved", json.dumps(metadata or {}, sort_keys=True)))
        return entry_id

    def submitted(self, entry_id: str, job_id: str, url: str = "") -> None:
        self._update(entry_id, "submitted", {"job_id": job_id, "url": url})

    def uncertain(self, entry_id: str, reason: str) -> None:
        self._update(entry_id, "uncertain", {"uncertainty": reason})

    def _update(self, entry_id: str, status: str, metadata: dict) -> None:
        with self._connection() as db:
            row = db.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
            if row is None or row["status"] not in {"reserved", "submitted", "uncertain"}:
                raise BudgetError("Reservation is missing or already settled")
            data = json.loads(row["metadata"])
            data.update(metadata)
            db.execute("UPDATE entries SET status=?,metadata=? WHERE id=?",
                       (status, json.dumps(data, sort_keys=True), entry_id))

    def settle(self, entry_id: str, *, charged: str | float | Decimal | None = None,
               evidence: str) -> None:
        """None conservatively books the full reservation, never invents a receipt."""
        if not evidence.strip():
            raise BudgetError("Settlement requires evidence")
        with self._connection() as db:
            row = db.execute("SELECT * FROM entries WHERE id=?", (entry_id,)).fetchone()
            if row is None or row["charged"] is not None:
                raise BudgetError("Reservation is missing or already settled")
            cost = row["reserved"] if charged is None else micro_usd(charged)
            metadata = json.loads(row["metadata"])
            metadata.update({"settlement_evidence": evidence,
                             "cost_basis": "reserved_upper_bound" if charged is None else "reported_actual"})
            db.execute("UPDATE entries SET status='settled',charged=?,metadata=? WHERE id=?",
                       (cost, json.dumps(metadata, sort_keys=True), entry_id))

    def snapshot(self) -> dict:
        with self._connection() as db:
            rows = [dict(row) for row in db.execute("SELECT * FROM entries ORDER BY created,id")]
        for row in rows:
            row["metadata"] = json.loads(row["metadata"])
        committed = sum(row["charged"] if row["charged"] is not None else row["reserved"] for row in rows)
        return {"currency": "USD", "ceiling_micro_usd": 250 * MICROS,
                "committed_micro_usd": committed, "remaining_micro_usd": max(0, 250 * MICROS - committed),
                "entries": rows}
