"""VAJB applog 1.0 - daily log files, readable text, never crashes the app.

Rules: RULES.md L1-L6.
    from applog import setup, get
    setup(APP_DIR / "data" / "logs", "myapp")   # once, at startup
    log = get("area")                            # anywhere
    log.info("import done: %s rows", n)
    log.exception("import failed")               # inside except: writes traceback

File: data/logs/myapp_2026-11-18.log, one line per event:
    2026-11-18 13:40:02  INFO   area     import done: 120 rows
No automatic deletion (L1). size_mb() and delete_older_than() are for the Settings page.
"""
import atexit
import faulthandler
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = "app"
_dir: Path | None = None
_prefix = "app"


class DailyFileHandler(logging.Handler):
    """Writes to <prefix>_YYYY-MM-DD.log; the name changes with the date, nothing is renamed
    (TimedRotatingFileHandler renames files and fails on Windows - L2)."""

    def __init__(self, directory: Path, prefix: str):
        super().__init__()
        self.directory, self.prefix = Path(directory), prefix
        self._lock = threading.Lock()

    def path(self, day: datetime | None = None) -> Path:
        return self.directory / f"{self.prefix}_{(day or datetime.now()):%Y-%m-%d}.log"

    def emit(self, record):
        try:
            line = self.format(record)
            with self._lock:
                with open(self.path(), "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            pass  # logging must never break the work it logs (L3)


def setup(directory, prefix: str, level=logging.INFO, console=False) -> logging.Logger:
    global _dir, _prefix
    _dir, _prefix = Path(directory), prefix
    _dir.mkdir(parents=True, exist_ok=True)
    logging.raiseExceptions = False
    root = logging.getLogger(_ROOT)
    if root.handlers:
        return root
    root.setLevel(level)
    root.propagate = False
    fmt = logging.Formatter("%(asctime)s  %(levelname)-6s %(name)-14s %(message)s", "%Y-%m-%d %H:%M:%S")
    h = DailyFileHandler(_dir, prefix)
    h.setFormatter(fmt)
    root.addHandler(h)
    if console and sys.stderr:
        s = logging.StreamHandler()
        s.setFormatter(fmt)
        root.addHandler(s)
    _install_crash_hooks(root)
    root.info("start pid=%s python=%s", os.getpid(), sys.version.split()[0])
    return root


def get(area: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT}.{area}")


def _install_crash_hooks(root: logging.Logger):
    """L5: a crash, a dying thread and every shutdown leave a trace."""
    try:
        crash = open(_dir / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(crash)
    except Exception:
        pass
    threading.excepthook = lambda a: root.error(
        "thread %s died", a.thread.name if a.thread else "?",
        exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    sys.excepthook = lambda t, v, tb: root.critical("unhandled error", exc_info=(t, v, tb))
    atexit.register(lambda: root.info("shutdown pid=%s", os.getpid()))


class timed:
    """with timed("render", "job 12"): ...  -> one line with duration; on error the traceback."""

    def __init__(self, area: str, what: str):
        self.log, self.what = get(area), what

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, et, e, tb):
        ms = int((time.time() - self.t0) * 1000)
        if e is None:
            self.log.info("%s (%d ms)", self.what, ms)
        else:
            self.log.error("%s failed after %d ms", self.what, ms, exc_info=(et, e, tb))
        return False


# --- for the Settings > Log page (UI12) ---

def files() -> list[Path]:
    return sorted(_dir.glob(f"{_prefix}_*.log")) if _dir else []


def size_mb() -> float:
    return round(sum(p.stat().st_size for p in files()) / 1e6, 1)


def read(days=2, problems_only=False, query=None, limit=500) -> list[str]:
    """Last lines (newest last). Traceback lines stay attached to their entry."""
    out = []
    for i in range(days - 1, -1, -1):
        p = _dir / f"{_prefix}_{(datetime.now() - timedelta(days=i)):%Y-%m-%d}.log"
        if p.exists():
            out += p.read_text(encoding="utf-8", errors="replace").splitlines()
    if problems_only:
        out = [l for l in out if "  INFO " not in l[:30]]
    if query:
        out = [l for l in out if query.lower() in l.lower()]
    return out[-limit:]


def delete_older_than(days: int) -> int:
    cut = time.time() - days * 86400
    n = 0
    for p in files():
        try:
            if p.stat().st_mtime < cut:
                p.unlink()
                n += 1
        except OSError:
            pass
    return n


def auto_trim(max_mb=500):
    """L1: deletes the oldest days only when the folder grows past max_mb."""
    fs = files()
    while fs and size_mb() > max_mb:
        try:
            fs.pop(0).unlink()
        except OSError:
            break
