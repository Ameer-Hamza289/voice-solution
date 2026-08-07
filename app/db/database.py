"""SQLite persistence for appointment slots, bookings, and emergency dispatches.

In production this layer would be replaced by a ServiceTitan / Housecall Pro
API adapter — the tool signatures in app/agent/tools.py would stay the same.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta

from app.config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS slots (
    id TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,      -- ISO datetime
    window_label TEXT NOT NULL,    -- e.g. 'Tue Jul 9, 8-10 AM'
    trade TEXT NOT NULL,           -- hvac | plumbing | electrical
    booked INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS bookings (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    slot_id TEXT NOT NULL REFERENCES slots(id),
    customer_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    address TEXT NOT NULL,
    issue TEXT NOT NULL,
    trade TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dispatches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    phone TEXT NOT NULL,
    address TEXT NOT NULL,
    issue TEXT NOT NULL,
    trade TEXT NOT NULL,
    severity TEXT NOT NULL,
    eta_minutes INTEGER NOT NULL
);
"""

_TRADES = ("hvac", "plumbing", "electrical")
_WINDOWS = ((8, "8-10 AM"), (10, "10 AM-12 PM"), (13, "1-3 PM"), (15, "3-5 PM"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0] == 0:
            _seed_slots(conn)


def _seed_slots(conn: sqlite3.Connection) -> None:
    """Create appointment windows for the next 7 days, skipping Sundays."""
    today = datetime.now().date()
    rows = []
    for offset in range(1, 8):
        day = today + timedelta(days=offset)
        if day.weekday() == 6:  # Sunday
            continue
        label_day = day.strftime("%a %b %d").replace(" 0", " ")
        for trade in _TRADES:
            for hour, window in _WINDOWS:
                start = datetime.combine(day, datetime.min.time()) + timedelta(hours=hour)
                rows.append((
                    uuid.uuid4().hex[:8],
                    start.isoformat(),
                    f"{label_day}, {window}",
                    trade,
                ))
    conn.executemany(
        "INSERT INTO slots (id, start_time, window_label, trade) VALUES (?, ?, ?, ?)",
        rows,
    )


def available_slots(trade: str, limit: int = 6) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, window_label, start_time FROM slots
               WHERE trade = ? AND booked = 0 AND start_time > ?
               ORDER BY start_time LIMIT ?""",
            (trade, datetime.now().isoformat(), limit),
        ).fetchall()
    return [dict(r) for r in rows]


def book_slot(slot_id: str, name: str, phone: str, address: str,
              issue: str, trade: str) -> dict | None:
    with _connect() as conn:
        slot = conn.execute(
            "SELECT id, window_label FROM slots WHERE id = ? AND booked = 0",
            (slot_id,),
        ).fetchone()
        if slot is None:
            return None
        booking_id = uuid.uuid4().hex[:8].upper()
        conn.execute("UPDATE slots SET booked = 1 WHERE id = ?", (slot_id,))
        conn.execute(
            """INSERT INTO bookings
               (id, created_at, slot_id, customer_name, phone, address, issue, trade)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (booking_id, datetime.now().isoformat(), slot_id,
             name, phone, address, issue, trade),
        )
    return {"booking_id": booking_id, "window": slot["window_label"]}


def create_dispatch(name: str, phone: str, address: str, issue: str,
                    trade: str, severity: str) -> dict:
    dispatch_id = "EMG-" + uuid.uuid4().hex[:6].upper()
    eta = 75  # minutes; middle of the promised 60-90 minute window
    with _connect() as conn:
        conn.execute(
            """INSERT INTO dispatches
               (id, created_at, customer_name, phone, address, issue, trade,
                severity, eta_minutes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (dispatch_id, datetime.now().isoformat(), name, phone, address,
             issue, trade, severity, eta),
        )
    return {"dispatch_id": dispatch_id, "eta_minutes": eta}
