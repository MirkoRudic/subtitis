"""`start()` mora da preživi gašenje onoga ko ga je pozvao (krug 28).

Pravi proces, pravi Job objekat sa „ubij sve pri zatvaranju" — baš slučaj
„Mirko zatvori Claude usred posla". Pomoćni proces pozove `start()` iz Job-a,
Job se zatvori (pomoćni umire), a server mora i dalje da odgovara.

Dve varijante: Job koji dozvoljava izlazak (put `CREATE_BREAKAWAY_FROM_JOB`)
i Job koji ga ne dozvoljava (rezerva preko WMI-ja). Server se diže na
slobodnom portu, nad privremenim poslovima, i gasi se na kraju.
"""

from __future__ import annotations

import ctypes
import json
import os
import socket
import subprocess
import sys
import time
from ctypes import wintypes

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Job objekti postoje samo na Windows-u")

KILL_ON_JOB_CLOSE = 0x2000
BREAKAWAY_OK = 0x0800
JobObjectExtendedLimitInformation = 9


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _Basic(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _Ext(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _Basic), ("IoInfo", _IO),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _job(breakaway: bool):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    h = k32.CreateJobObjectW(None, None)
    info = _Ext()
    info.BasicLimitInformation.LimitFlags = KILL_ON_JOB_CLOSE | (BREAKAWAY_OK if breakaway else 0)
    assert k32.SetInformationJobObject(wintypes.HANDLE(h), JobObjectExtendedLimitInformation,
                                       ctypes.byref(info), ctypes.sizeof(info))
    return k32, h


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


HELPER = (
    "import sys, json; sys.stdin.readline();"
    "from subtitis import mcp_server as m; r = m.start();"
    "print(json.dumps(r, ensure_ascii=False), flush=True); sys.stdin.readline()"
)


def _identity(port: int):
    import httpx

    try:
        return httpx.get(f"http://127.0.0.1:{port}/api/identity", timeout=1).json()
    except Exception:  # noqa: BLE001
        return None


@pytest.mark.parametrize("breakaway, put", [(True, "odvojen proces"), (False, "WMI")])
def test_server_prezivi_zatvaranje_onoga_ko_ga_je_pokrenuo(tmp_path, monkeypatch, breakaway, put):
    from subtitis import config as cfg

    port = _free_port()
    import site

    # Pomoćni proces ide na OSNOVNI interpreter, ne na `.venv` pokretač: pokretač
    # odmah pravi pravi interpreter kao dete, pre nego što ga stavimo u Job, pa
    # bi `start()` zvao proces koji u Job-u nije (prvi pokušaj ovog testa).
    paths = [str(cfg.ROOT_DIR), *site.getsitepackages()]
    env = dict(os.environ, SUBTITIS_PORT=str(port), SUBTITIS_JOBS_DIR=str(tmp_path / "jobs"),
               SUBTITIS_TMP_DIR=str(tmp_path / "tmp"), PYTHONPATH=os.pathsep.join(paths),
               PYTHONIOENCODING="utf-8")
    k32, job = _job(breakaway)
    helper = subprocess.Popen([sys._base_executable, "-c", HELPER], cwd=cfg.ROOT_DIR, env=env,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
    pid = None
    try:
        assert k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(int(helper._handle)))
        inside = wintypes.BOOL()
        k32.IsProcessInJob(wintypes.HANDLE(int(helper._handle)), wintypes.HANDLE(job), ctypes.byref(inside))
        assert inside.value, "pomoćni proces mora biti u Job-u"
        helper.stdin.write("go\n")
        helper.stdin.flush()
        line = helper.stdout.readline()
        r = json.loads(line)
        assert r.get("pokrenut") is True, r
        assert put in r["nacin"]
        if not r.get("radi"):
            # krug 29: posle 20 s `start()` vrati „pokrenut, pitaj status()" —
            # pod opterećenjem (cela zbirka testova) hladno pokretanje ume da traje
            deadline = time.monotonic() + 60
            while _identity(port) is None and time.monotonic() < deadline:
                time.sleep(0.5)
        ident = _identity(port)
        assert ident is not None, "server nije proradio ni za 80 s"
        pid = r.get("pid") or ident["pid"]

        # „Claude se zatvara": Job se zatvara, pomoćni proces umire.
        k32.CloseHandle(wintypes.HANDLE(job))
        job = None
        helper.wait(timeout=10)
        time.sleep(1.0)

        ident = _identity(port)
        assert ident is not None, "server je umro sa onim ko ga je pokrenuo"
        assert ident["pid"] == pid and os.path.normcase(ident["jobs_dir"]) == os.path.normcase(str(tmp_path / "jobs"))

        if breakaway:
            # Krug 29: rezerva gašenja nad PRAVIM serverom — `server.pid` mora da
            # nosi baš vlasnika porta (osnovni interpreter, ne `.venv` pokretač).
            from subtitis import mcp_server as m

            monkeypatch.setattr(cfg, "TMP_DIR", tmp_path / "tmp")
            r = m._fallback(port)
            assert r.get("pid") == pid and "iz server.pid" in r["poruka"], r
            assert r["ugasen"] is True or _identity(port) is None
    finally:
        if job is not None:
            k32.CloseHandle(wintypes.HANDLE(job))
        if helper.poll() is None:
            helper.kill()
        # gašenje servera iz testa: uredno, pa (samo ako zatreba) tačno taj PID
        try:
            import httpx

            httpx.post(f"http://127.0.0.1:{port}/api/shutdown", json={"force": True}, timeout=5)
        except Exception:  # noqa: BLE001
            pass
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and _identity(port) is not None:
            time.sleep(0.3)
        if _identity(port) is not None and pid:
            subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    assert _identity(port) is None, "server se nije ugasio preko /api/shutdown"
