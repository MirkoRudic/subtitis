"""SQLite: poslovi, red zadataka, događaji za SSE.

Baza je JEDINI kanal između web procesa i worker procesa. Web nikad ne obrađuje
— samo upisuje zadatak u red; worker ga uzima. Zato je WAL uključen: dva procesa
pišu u istu bazu.

Sadržaj poslova (`words.json`, `sentences.json`, SRT) ostaje na disku u
`jobs/<id>/`. Baza drži samo status i red — da bi brisanje baze značilo gubitak
istorije, a ne gubitak posla.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .. import config as cfg

#: Tok je EKSPLICITAN: posle `transcribed` ništa se ne dešava samo od sebe.
#: Prevod pokreće čovek, tek pošto pregleda original.
STATUSES = (
    "uploaded",
    "transcribing",
    "transcribed",   # <- PAUZA ZA PREGLED; original SRT je već za preuzimanje
    "translating",
    "done",
    "error",
)

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    source_name  TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    language     TEXT NOT NULL,
    status       TEXT NOT NULL,
    duration     REAL,
    error        TEXT,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    kind        TEXT NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',
    state       TEXT NOT NULL DEFAULT 'queued',
    error       TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_tasks_state ON tasks(state, id);

CREATE TABLE IF NOT EXISTS events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id   TEXT NOT NULL,
    ts       TEXT NOT NULL,
    kind     TEXT NOT NULL,
    message  TEXT NOT NULL DEFAULT '',
    progress REAL
);
CREATE INDEX IF NOT EXISTS ix_events_job ON events(job_id, id);

-- Jedan red, uvek id=1. Worker ga kuca dok radi; web po njemu zna da li obrada
-- uopšte postoji. Bez ovoga upload izgleda isto i kad worker radi i kad ne radi.
CREATE TABLE IF NOT EXISTS worker_beat (
    id   INTEGER PRIMARY KEY CHECK (id = 1),
    ts   TEXT NOT NULL,
    pid  INTEGER,
    mode TEXT NOT NULL DEFAULT 'proces'
);
"""

#: Koliko sekundi otkucaj važi. Worker kuca na `HEARTBEAT_SECONDS`; prag je
#: višestruko veći da kratak zastoj (GC, spor disk) ne obori indikator.
HEARTBEAT_STALE_SECONDS = 20.0


def db_path() -> Path:
    return cfg.JOBS_DIR / "subtitis.db"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


# --- poslovi ---------------------------------------------------------------


def create_job(job_id: str, source_name: str, source_path: str, language: str) -> None:
    ts = now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, source_name, source_path, language, status,"
            " created_at, updated_at) VALUES (?,?,?,?,'uploaded',?,?)",
            (job_id, source_name, source_path, language, ts, ts),
        )


def update_job(job_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = now()
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*fields.values(), job_id))


def get_job(job_id: str) -> sqlite3.Row | None:
    with connect() as conn:
        return conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()


def list_jobs() -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()


def delete_job(job_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM events WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM tasks WHERE job_id=?", (job_id,))
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))


# --- red zadataka ----------------------------------------------------------


def enqueue(job_id: str, kind: str, params: dict[str, Any] | None = None) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tasks (job_id, kind, params, created_at) VALUES (?,?,?,?)",
            (job_id, kind, json.dumps(params or {}), now()),
        )
        return int(cur.lastrowid)


def migrate_queued_profiles(job_id: str) -> int:
    """Zadaci u redu nastali pre migracije posla nose stara imena profila
    (`migrate.py`). Poziva se jednom, kad je posao upravo migriran."""
    from ..migrate import RENAMES

    changed = 0
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, params FROM tasks WHERE job_id=? AND state='queued'", (job_id,)
        ).fetchall()
        for row in rows:
            params = json.loads(row["params"] or "{}")
            old = params.get("profile")
            if old in RENAMES and RENAMES[old] != old:
                params["profile"] = RENAMES[old]
                conn.execute(
                    "UPDATE tasks SET params=? WHERE id=?", (json.dumps(params), row["id"])
                )
                changed += 1
    return changed


def pending_task(job_id: str) -> sqlite3.Row | None:
    """Zadatak koji čeka ili se izvršava — da UI ne nudi dugme dvaput."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND state IN ('queued','running')"
            " ORDER BY id LIMIT 1",
            (job_id,),
        ).fetchone()


def claim_next_task() -> sqlite3.Row | None:
    """Uzmi najstariji zadatak iz reda i označi ga kao `running`.

    Atomično preko `UPDATE ... WHERE id = (SELECT ...)`, pa dva workera ne mogu
    uzeti isti zadatak.
    """
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM tasks WHERE state='queued' ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE tasks SET state='running', started_at=? WHERE id=?",
            (now(), row["id"]),
        )
        conn.execute("COMMIT")
        return conn.execute("SELECT * FROM tasks WHERE id=?", (row["id"],)).fetchone()


def last_task(job_id: str, kind: str) -> sqlite3.Row | None:
    """Poslednji zadatak date vrste — da stranica kaže kako je radnja prošla."""
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND kind=? ORDER BY id DESC LIMIT 1",
            (job_id, kind),
        ).fetchone()


def finish_task(task_id: int, state: str, error: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE tasks SET state=?, error=?, finished_at=? WHERE id=?",
            (state, error, now(), task_id),
        )


def skip_queued_tasks(job_id: str, reason: str) -> int:
    """Označi sve zadatke posla koji još čekaju kao preskočene.

    Posle pada nekog koraka ostatak reda nema nad čim da radi. Bez ovoga se
    posle neuspele transkripcije pusti prelamanje i prijavi „gotovo".
    """
    with connect() as conn:
        cur = conn.execute(
            "UPDATE tasks SET state='skipped', error=?, finished_at=?"
            " WHERE job_id=? AND state='queued'",
            (reason, now(), job_id),
        )
        return cur.rowcount or 0


def queued_count() -> int:
    """Koliko zadataka čeka ili se izvršava — za indikator u zaglavlju."""
    with connect() as conn:
        return int(conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE state IN ('queued','running')"
        ).fetchone()["n"])


# --- otkucaj workera -------------------------------------------------------


def beat(mode: str = "proces", pid: int | None = None) -> None:
    """Zabeleži da worker živi. Zove se i dok traje dugačak zadatak."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO worker_beat (id, ts, pid, mode) VALUES (1,?,?,?)"
            " ON CONFLICT(id) DO UPDATE SET ts=excluded.ts, pid=excluded.pid,"
            " mode=excluded.mode",
            (now(), pid, mode),
        )


def clear_beat() -> None:
    """Worker se uredno gasi — indikator sme odmah da pocrveni."""
    with connect() as conn:
        conn.execute("DELETE FROM worker_beat")


def worker_status() -> dict[str, Any]:
    """Da li obrada radi, i šta je u redu.

    `alive` je jedina stvar koju UI treba da zna. Ostalo je za dijagnostiku.
    """
    with connect() as conn:
        row = conn.execute("SELECT * FROM worker_beat WHERE id=1").fetchone()
    queued = queued_count()
    if row is None:
        return {"alive": False, "age": None, "queued": queued, "mode": None}
    try:
        ts = datetime.fromisoformat(row["ts"])
    except ValueError:  # pragma: no cover — zapis iz budućnosti/pokvaren
        return {"alive": False, "age": None, "queued": queued, "mode": row["mode"]}
    age = (datetime.now(timezone.utc) - ts).total_seconds()
    return {
        "alive": age <= HEARTBEAT_STALE_SECONDS,
        "age": round(age, 1),
        "queued": queued,
        "mode": row["mode"],
    }


def job_task_summary(job_id: str) -> dict[str, int]:
    """Broj zadataka po stanju — dashboard prikazuje preskočene."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT state, COUNT(*) AS n FROM tasks WHERE job_id=? GROUP BY state",
            (job_id,),
        ).fetchall()
    return {r["state"]: int(r["n"]) for r in rows}


def failed_tasks(job_id: str, limit: int = 5) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND state IN ('error','skipped')"
            " ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()


def retryable_tasks(job_id: str) -> list[sqlite3.Row]:
    """Korak koji je pao i sve što je zbog njega preskočeno, redom.

    Posle pada se ostatak reda preskače (`skip_queued_tasks`); „Pokušaj
    ponovo" mora da pusti i njih, inače bi posle uspelog ponovnog koraka
    zavisni zadaci ostali neodrađeni.
    """
    with connect() as conn:
        failed = conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND state='error' ORDER BY id DESC LIMIT 1",
            (job_id,),
        ).fetchone()
        if failed is None:
            return []
        skipped = conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND state='skipped' AND id>? ORDER BY id",
            (job_id, failed["id"]),
        ).fetchall()
    return [failed, *skipped]


def reset_running_tasks() -> int:
    """Zadaci zatečeni kao `running` posle pada workera — vrati ih u red."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE tasks SET state='queued', started_at=NULL WHERE state='running'"
        )
        return cur.rowcount or 0


# --- događaji --------------------------------------------------------------


def add_event(job_id: str, kind: str, message: str = "", progress: float | None = None) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO events (job_id, ts, kind, message, progress) VALUES (?,?,?,?,?)",
            (job_id, now(), kind, message[:2000], progress),
        )


def events_since(job_id: str, last_id: int, limit: int = 200) -> list[sqlite3.Row]:
    with connect() as conn:
        return conn.execute(
            "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT ?",
            (job_id, last_id, limit),
        ).fetchall()


def last_event_id(job_id: str) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS n FROM events WHERE job_id=?", (job_id,)
        ).fetchone()
        return int(row["n"])


def recent_events(job_id: str, limit: int = 40) -> list[sqlite3.Row]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE job_id=? ORDER BY id DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
        return list(reversed(rows))


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]
