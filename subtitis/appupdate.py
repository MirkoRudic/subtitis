"""Ažuriranje iz aplikacije (VAJB pravila V2, V3, V5), za kopiju koja je git klon.

- `check()` — `git fetch`, koliko je commitova iza i koja je verzija na GitHub-u.
- `pull()` — `git pull --ff-only`; `pip install` samo ako se promenio `requirements.txt`.
- `restart()` — nov proces (`subtitis desktop --no-open`) sačeka da stari pusti port, pa preuzme;
  otvoren prozor se sam osveži kad se promeni verzija na `/health`.
- `auto_loop()` — provera na startu i na sat; nova verzija se instalira sama samo kad nijedan prozor
  Subtitisa nije otvoren i ništa se ne obrađuje (V5). Otvoren prozor se nikad ne gasi zbog ažuriranja.

Menja samo kod; `jobs/`, `data/`, `tmp/` i `subtitis.toml` git ne dira.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from . import applog
from .config import ROOT_DIR

NOWIN = 0x08000000 if sys.platform == "win32" else 0
DETACH = 0x00000008 | 0x00000200 if sys.platform == "win32" else 0
WAIT_PORT_ENV = "SUBTITIS_WAIT_PORT"

log = applog.get("update")
#: `seen` — poslednji put kad se javio otvoren prozor (zaglavlje pita /worker/status na 3 s).
STATE: dict[str, Any] = {"seen": 0.0, "at": None, "ok": True, "behind": 0, "version": "", "port": None}


def is_git() -> bool:
    return (ROOT_DIR / ".git").exists()


def git_exe() -> str | None:
    for c in (shutil.which("git"), r"C:\Program Files\Git\cmd\git.exe", r"C:\Program Files (x86)\Git\cmd\git.exe"):
        if c and Path(c).exists():
            return c
    return None


def git(*args: str, timeout: int = 120) -> tuple[int, str]:
    exe = git_exe()
    if not exe:
        raise RuntimeError("Git nije pronađen na ovom računaru.")
    r = subprocess.run([exe, *args], cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, creationflags=NOWIN)
    return r.returncode, (r.stdout + r.stderr).strip()


def check() -> dict[str, Any]:
    try:
        code, out = git("fetch", "--quiet")
        if code:
            raise RuntimeError(out[-300:])
        _, n = git("rev-list", "--count", "HEAD..@{u}")
        _, remote = git("show", "@{u}:subtitis/__init__.py")
        m = re.search(r'__version__\s*=\s*"([^"]+)"', remote or "")
        STATE.update(at=time.time(), ok=True, behind=int(n) if n.strip().isdigit() else 0,
                     version=m.group(1) if m else "")
    except Exception as exc:
        log.warning("provera nove verzije: %s", exc)
        STATE.update(at=time.time(), ok=False)
    return {k: STATE[k] for k in ("at", "ok", "behind", "version")}


def pull() -> dict[str, Any]:
    _, before = git("rev-parse", "HEAD")
    code, out = git("pull", "--ff-only", timeout=300)
    if code:
        log.error("git pull: %s", out[-500:])
        return {"ok": False, "out": out}
    _, after = git("rev-parse", "HEAD")
    res: dict[str, Any] = {"ok": True, "out": out, "changed": before != after}
    if before != after:
        _, files = git("diff", "--name-only", before, after)
        if "requirements.txt" in files.splitlines():
            py = Path(sys.executable)
            if py.name.lower() == "pythonw.exe":
                py = py.with_name("python.exe")
            r = subprocess.run([str(py), "-m", "pip", "install", "-q", "-r", str(ROOT_DIR / "requirements.txt")],
                               cwd=ROOT_DIR, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=900, creationflags=NOWIN)
            res["pip_error"] = bool(r.returncode)
            if r.returncode:
                log.error("pip posle ažuriranja: %s", (r.stdout + r.stderr)[-800:])
        log.info("ažurirano %s -> %s", before[:7], after[:7])
        STATE["behind"] = 0
    return res


def restart() -> None:
    """Nov proces čeka port starog, pa preuzme; stari se gasi istim putem kao „Izađi“."""
    from .web import api

    env = dict(os.environ, **{WAIT_PORT_ENV: str(STATE.get("port") or "")})
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    subprocess.Popen([str(w if w.exists() else exe), "-m", "subtitis", "desktop", "--no-open"], cwd=ROOT_DIR,
                     env=env, creationflags=NOWIN | DETACH, close_fds=True)
    log.info("restart posle ažuriranja")
    if api.SHUTDOWN:
        api.SHUTDOWN()


def window_open() -> bool:
    return time.time() - STATE["seen"] < 300


def busy() -> bool:
    from .web import db

    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('transcribing', 'translating')").fetchone()[0]
    return bool(n) or db.queued_count() > 0


def auto_loop() -> None:
    time.sleep(90)
    last = 0.0
    while True:
        try:
            if time.time() - last > 3600:
                last = time.time()
                check()
            if STATE["ok"] and STATE["behind"] and not window_open() and not busy():
                log.info("tiho ažuriranje na %s", STATE["version"])
                r = pull()
                if r.get("ok") and r.get("changed"):
                    restart()
                    return
        except Exception:
            log.exception("tiho ažuriranje")
        time.sleep(60)
