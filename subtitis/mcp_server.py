"""MCP server: Claude desktop pokreće Subtitis i prati poslove (krug 28).

    subtitis mcp              # stdio; pokreće ga Claude desktop aplikacija
    subtitis mcp-install      # upiše unos u konfiguraciju Claude aplikacije
    subtitis mcp-uninstall

Tanak omotač, bez nove logike: alati zovu HTTP API dashboarda (`web/api.py`).
Ako nešto nedostaje, dodaje se tamo. Svaki alat vraća rečnik — kad dashboard
ne radi ili posao ne postoji, rečnik sa `greska`, nikad izuzetak.

Upravljanje procesom (krug 28, odobreno u `krugovi/28-odgovor.md`):

* `status()` — samo port: `GET /api/identity` na podešenom portu i narednih
  `PORT_SCAN`. Nikakva pretraga procesa po imenu ni komandnoj liniji (sinoć
  nije našla server koji je radio).
* `start()` — `.venv\\Scripts\\pythonw.exe -m subtitis desktop --no-open`,
  ODVOJEN od MCP procesa: `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
  CREATE_BREAKAWAY_FROM_JOB`. Ako Job objekat Claude aplikacije ne dozvoljava
  izlazak („Access denied"), rezerva je WMI (`Win32_Process.Create`): proces
  pravi Windows, pa nije ni u Job-u ni u stablu MCP-a. Zatvaranje Claude
  aplikacije tako ne prekida obradu usred posla.
* `stop()` — `POST /api/shutdown` (kao „Izađi" u traci); dok se obrađuje
  zadatak odbija, osim uz `force`. Rezerva kad server ne odgovara: gasi se
  TAČNO jedan proces, vlasnik porta (tabela TCP veza, ne ime procesa), i to
  samo ako mu je izvršni fajl unutar foldera aplikacije — inače se ne gasi
  ništa, nego se javlja šta je nađeno. Nikad roditelj ni stablo procesa.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import config as cfg

ROOT = cfg.ROOT_DIR

#: Koliko portova iznad podešenog se pita (isto kao `desktop.choose_port`).
PORT_SCAN = 20
#: Veza kojom Claude dolazi do ovog procesa ističe posle ~60 s (krug 29: poziv
#: `tests()` je pukao baš tu). Nijedan alat ne sme da čeka blizu toga: dugo se
#: radi u pozadini i prati se (`job`, `zadatak`). Ove granice drže svaki alat
#: ispod pola minuta i u najgorem slučaju.
START_TIMEOUT = 20.0
STOP_TIMEOUT = 10.0
SHUTDOWN_TIMEOUT = 5.0
HTTP_TIMEOUT = 20.0
REBUILD_TIMEOUT = 20.0  # prelamanje filma od 320 cue-ova: najviše 3,1 s (krug 29)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000

NOT_RUNNING = "Subtitis ne radi — pozovi start()."

#: `script()` pokreće samo ove skripte iz `scripts/` — merne, bez upisa u
#: poslove i bez modela. Nema alata za proizvoljnu komandu: ovo radi na
#: Mirkovoj mašini i ne sme da bude ljuska.
SCRIPTS = (
    "krug5_nulte_reci", "krug6_digital", "krug7_ravnomerno", "krug19_pauze",
    "krug20_veznici", "krug21_granica", "krug22_pauze", "krug23_lazna_pauza",
    "krug24_vreme", "krug24_zamenica", "krug25",
)
_ARG_RE = re.compile(r"^[\w.,:=/\\-]{1,200}$")


class ToolError(Exception):
    """Očekivan neuspeh alata — ide korisniku kao `greska`, ne kao izuzetak."""


# ==========================================================================
# HTTP
# ==========================================================================


def _identity(port: int, timeout: float = 0.5) -> dict[str, Any] | None:
    import httpx

    try:
        r = httpx.get(f"http://127.0.0.1:{port}/api/identity", timeout=timeout)
        data = r.json()
    except Exception:  # noqa: BLE001 — bilo šta osim Subtitisa na portu
        return None
    if not isinstance(data, dict) or data.get("app") != "subtitis":
        return None
    # Druga instanca nad drugim poslovima (test, druga kopija) nije „naša".
    if os.path.normcase(data.get("jobs_dir", "")) != os.path.normcase(str(cfg.JOBS_DIR)):
        return None
    return data


def find_server() -> tuple[int, dict[str, Any]] | None:
    """Port i `/api/identity` Subtitisa nad ovim poslovima, ili None."""
    base = cfg.port()
    for port in range(base, base + PORT_SCAN + 1):
        ident = _identity(port)
        if ident is not None:
            return port, ident
    return None


def _http(port: int):
    import httpx

    return httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=HTTP_TIMEOUT)


def _detail(r: Any) -> str:
    try:
        d = r.json().get("detail")
    except Exception:  # noqa: BLE001
        return r.text[:300]
    if isinstance(d, dict):
        return d.get("poruka") or json.dumps(d, ensure_ascii=False)
    if isinstance(d, list):  # pydantic greška u zahtevu
        return "; ".join(f"{'.'.join(map(str, e.get('loc', [])))}: {e.get('msg')}" for e in d)
    return str(d)


def _call(method: str, path: str, **kw: Any) -> Any:
    srv = find_server()
    if srv is None:
        raise ToolError(NOT_RUNNING)
    with _http(srv[0]) as c:
        r = c.request(method, path, **kw)
    if r.status_code == 404 and _detail(r) == "Not Found":
        # ruta ne postoji (ne posao): dashboard je stariji od ovog MCP-a
        raise ToolError(f"Dashboard ne zna za {path.split('?')[0]} — radi starija verzija "
                        f"({srv[1].get('signature') or srv[1].get('version')}). Ponovo ga pokreni.")
    if r.status_code >= 400:
        raise ToolError(_detail(r))
    return r.json()


def _safe(fn):
    """Alat ne baca: očekivan neuspeh i neočekivana greška idu kao `greska`."""
    import functools

    @functools.wraps(fn)
    def wrap(*a: Any, **kw: Any) -> dict[str, Any]:
        try:
            return fn(*a, **kw)
        except ToolError as exc:
            return {"greska": str(exc)}
        except Exception as exc:  # noqa: BLE001
            return {"greska": f"{type(exc).__name__}: {exc}"}

    return wrap


# ==========================================================================
# Upravljanje
# ==========================================================================


@_safe
def status() -> dict[str, Any]:
    """Radi li dashboard, port, verzija i krug, broj zadataka u redu."""
    srv = find_server()
    if srv is None:
        return {"radi": False, "port": cfg.port(), "poruka": "Dashboard ne radi; start() ga pokreće."}
    port, ident = srv
    return {
        "radi": True, "port": port, "url": f"http://127.0.0.1:{port}",
        "verzija": ident.get("version"), "krug": ident.get("round"), "potpis": ident.get("signature"),
        "u_redu": ident.get("queue"), "radi_na": ident.get("running"), "pid": ident.get("pid"),
        "poslovi": ident.get("jobs_dir"),
    }


def launcher() -> Path:
    """`pythonw.exe` iz `.venv` — pokretač, isto što i prečica. `sys.executable`
    u venv-u na Windows-u je osnovni interpreter, ne venv."""
    for p in (Path(sys.prefix) / "Scripts" / "pythonw.exe", ROOT / ".venv" / "Scripts" / "pythonw.exe"):
        if p.exists():
            return p
    return Path(sys.executable)


def server_cmd() -> list[str]:
    return [str(launcher()), "-m", "subtitis", "desktop", "--no-open"]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    # Paket nije instaliran u venv; `-m subtitis` radi iz korena aplikacije.
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def spawn_detached() -> str:
    """Pokreni server odvojeno od ovog procesa. Vrati kojim putem."""
    cmd, env = server_cmd(), _child_env()
    try:
        subprocess.Popen(
            cmd, cwd=ROOT, env=env, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB,
        )
        return "odvojen proces (izlazak iz Job objekta)"
    except OSError as exc:
        if getattr(exc, "winerror", None) != 5:  # 5 = Access denied: Job ne pušta
            raise
    spawn_wmi(cmd, env)
    return "WMI (Job objekat ne dozvoljava izlazak)"


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def spawn_wmi(cmd: list[str], env: dict[str, str]) -> None:
    """`Win32_Process.Create`: proces pravi WMI servis, ne ovaj proces. Okruženje
    se prenosi celo (inače nema ni `SystemRoot`, a sa njim ni mreže)."""
    line = subprocess.list2cmdline(cmd)
    env_list = ",".join(_ps_quote(f"{k}={v}") for k, v in env.items() if "\n" not in v)
    script = (
        "$st = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly "
        f"-Property @{{EnvironmentVariables=[string[]]@({env_list}); ShowWindow=[uint16]0}};"
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
        f"@{{CommandLine={_ps_quote(line)}; CurrentDirectory={_ps_quote(str(ROOT))}; ProcessStartupInformation=$st}};"
        "exit $r.ReturnValue"
    )
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", enc],
                       capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise ToolError(f"WMI pokretanje nije uspelo (kod {r.returncode}): {r.stderr.strip()[:300]}")


@_safe
def start() -> dict[str, Any]:
    """Pokreće dashboard ako ne radi — odvojeno od MCP procesa."""
    if find_server() is not None:
        return {**status(), "pokrenut": False, "poruka": "Već radi."}
    how = spawn_detached()
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if find_server() is not None:
            return {**status(), "pokrenut": True, "nacin": how}
        time.sleep(0.5)
    # Ne greška: hladno pokretanje ume da traje. Proces radi, odgovoriće.
    return {"radi": False, "pokrenut": True, "nacin": how,
            "poruka": f"Pokrenut, ali za {START_TIMEOUT:.0f} s još ne odgovara — pitaj status() "
                      "za nekoliko sekundi. Ako ni tada ne odgovara, vidi jobs/console.log."}


def _port_owner(port: int) -> dict[str, Any] | None:
    """Vlasnik porta iz tabele TCP veza (ne po imenu procesa): PID i izvršni fajl."""
    script = (
        f"$c = Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;"
        "if ($c) { $p = Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\";"
        " @{pid=$c.OwningProcess; exe=$p.ExecutablePath} | ConvertTo-Json -Compress }"
    )
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, timeout=30)
    out = r.stdout.strip()
    return json.loads(out) if out else None


def _kill(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(int(pid)), "/F"], capture_output=True, timeout=30)


def _port_open(port: int) -> bool:
    from .desktop import port_in_use

    return port_in_use("127.0.0.1", port)


def _read_server_pid() -> dict[str, Any] | None:
    try:
        return json.loads(cfg.server_pid_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _fallback(port: int) -> dict[str, Any]:
    """Server ne odgovara: gasi se TAČNO jedan proces — vlasnik porta — i samo
    ako je to baš server koji je zapisao `tmp/server.pid` (isti PID i isto
    vreme pokretanja; zaostao fajl pale instance ne sme da ugasi tuđ proces koji
    je u međuvremenu dobio isti PID). Nikad roditelj ni stablo procesa."""
    from .proc import same_process

    owner = _port_owner(port)
    if owner is None:
        return {"ugasen": True, "poruka": f"Na portu {port} više niko ne sluša."}
    rec = _read_server_pid()
    base = {"ugasen": False, "port": port, "vlasnik_porta": owner.get("pid"), "exe": owner.get("exe")}
    if rec is None:
        return {**base, "poruka": f"Port {port} drži PID {owner.get('pid')}, a {cfg.server_pid_path()} "
                                  "ne postoji — ne znam da li je to Subtitis. Ne gasim ništa."}
    base["server_pid"] = rec.get("pid")
    if rec.get("pid") != owner.get("pid"):
        return {**base, "poruka": f"Port {port} drži PID {owner.get('pid')}, a server.pid kaže "
                                  f"{rec.get('pid')}. Brojevi se ne poklapaju — ne gasim ništa."}
    if not same_process(rec["pid"], rec.get("started")):
        return {**base, "poruka": f"PID {rec['pid']} postoji, ali nije pokrenut kad i server koji je "
                                  "zapisao server.pid (zaostao fajl). Ne gasim ništa."}
    _kill(rec["pid"])
    return {"ugasen": not _port_open(port), "port": port, "pid": rec["pid"],
            "poruka": f"Server nije odgovarao; ugašen PID {rec['pid']} (vlasnik porta, iz server.pid)."}


@_safe
def stop(force: bool = False) -> dict[str, Any]:
    """Gasi dashboard. Dok se obrađuje zadatak odbija, osim uz `force=True`."""
    import httpx

    srv = find_server()
    if srv is None:
        port = cfg.port()
        if _port_open(port):
            return _fallback(port)
        return {"ugasen": True, "poruka": "Subtitis već ne radi."}
    port = srv[0]
    try:
        with _http(port) as c:
            r = c.post("/api/shutdown", json={"force": force}, timeout=SHUTDOWN_TIMEOUT)
    except httpx.HTTPError:
        return _fallback(port)
    if r.status_code == 409:
        d = r.json().get("detail", {})
        return {"ugasen": False, "poruka": d.get("poruka") if isinstance(d, dict) else str(d),
                "radi_na": d.get("running") if isinstance(d, dict) else None}
    if r.status_code >= 400:
        raise ToolError(_detail(r))
    deadline = time.monotonic() + STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not _port_open(port):
            return {"ugasen": True, "port": port, "prekinut": r.json().get("interrupted")}
        time.sleep(0.3)
    return _fallback(port)


@_safe
def jobs(limit: int = 20, stanje: str | None = None) -> dict[str, Any]:
    """Spisak poslova: id, naziv, stanje, jezici, kad je nastao."""
    params: dict[str, Any] = {"limit": limit}
    if stanje:
        params["state"] = stanje
    return _call("GET", "/api/jobs", params=params)


@_safe
def job(id: str) -> dict[str, Any]:  # noqa: A002 — ime parametra je iz zadatka
    """Jedan posao: stanje, napredak, jezici, broj upozorenja po vrsti."""
    return _call("GET", f"/api/jobs/{id}")


# ==========================================================================
# Rad
# ==========================================================================


@_safe
def transcribe(putanja: str, jezik: str = "sr", profil: str | None = None) -> dict[str, Any]:
    """Nov posao iz fajla na disku; vraća odmah, sa id-jem posla."""
    return _call("POST", "/api/jobs", json={"path": putanja, "language": jezik, "profile": profil})


@_safe
def translate(id: str, jezici: list[str]) -> dict[str, Any]:  # noqa: A002
    """Prevod na jedan ili više jezika; vraća odmah."""
    return _call("POST", f"/api/jobs/{id}/translate", json={"languages": list(jezici)})


@_safe
def rebuild(id: str, profil: str | None = None, pismo: str = "Latn", slova: str = "normal",  # noqa: A002
            interpunkcija: str = "keep", jezik: str | None = None) -> dict[str, Any]:
    """Prelomi odmah (sinhrono, kao editor). Ako potraje duže od 20 s, server
    nastavlja sam, a alat vraća da se prati preko job(id)."""
    import httpx

    try:
        return _call("POST", f"/api/jobs/{id}/rebuild", timeout=REBUILD_TIMEOUT, json={
            "language": jezik, "profile": profil, "script": pismo, "case": slova, "punct": interpunkcija})
    except httpx.TimeoutException:
        return {"gotovo": False, "poruka": f"Prelamanje traje duže od {REBUILD_TIMEOUT:.0f} s; server ga "
                                           "završava sam — prati job(id), pa export(id)."}


@_safe
def export(id: str, jezik: str | None = None) -> dict[str, Any]:  # noqa: A002
    """Putanja do gotovog SRT-a (ne sadržaj)."""
    params = {"lang": jezik} if jezik else {}
    return _call("GET", f"/api/jobs/{id}/export", params=params)


# ==========================================================================
# Sadržaj
# ==========================================================================


@_safe
def text(id: str, jezik: str | None = None, od: int = 1, do: int | None = None) -> dict[str, Any]:  # noqa: A002
    """Rečenice od..do (id-jevi), najviše 50 po pozivu; kaže koliko ih još ima."""
    params: dict[str, Any] = {"start": od}
    if jezik:
        params["lang"] = jezik
    if do is not None:
        params["end"] = do
    return _call("GET", f"/api/jobs/{id}/text", params=params)


@_safe
def flags(id: str, vrsta: str | None = None) -> dict[str, Any]:  # noqa: A002
    """Upozorenja sa tekstom, najviše 20; kaže koliko ih još ima."""
    return _call("GET", f"/api/jobs/{id}/flags", params={"kind": vrsta} if vrsta else {})


# ==========================================================================
# Krugovi: testovi i merne skripte (bez dashboarda)
# ==========================================================================


def _python() -> str:
    """`python.exe` iz `.venv` (sa konzolom — treba nam izlaz)."""
    p = ROOT / ".venv" / "Scripts" / "python.exe"
    return str(p) if p.exists() else sys.executable


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    env = _child_env() | {"PYTHONIOENCODING": "utf-8"}
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _tests_args(obrazac: str | None) -> list[str]:
    """Izbor testova za pytest: putanja UNUTAR `tests/` ili `-k` izraz.
    Putanja van `tests/` (`tests/../x.py`) bi pustila pytest da izvrši bilo
    koji .py na disku kao test (bezbednosni pregled kruga 28)."""
    if not obrazac:
        return []
    if not _ARG_RE.match(obrazac.replace(" ", "_")):
        raise ToolError("Obrazac sme da ima samo slova, brojeve i . , : = / \\ - _ i razmak.")
    if not obrazac.startswith("tests"):
        return ["-k", obrazac]
    target = (ROOT / obrazac.split("::")[0]).resolve()
    if not target.is_relative_to((ROOT / "tests").resolve()):
        raise ToolError(f"Putanja {obrazac!r} nije unutar tests/.")
    return [obrazac]


def _tests_work(sel: list[str], work: Path) -> dict[str, Any]:
    """Pravi posao `tests()` — u pokretaču, ne u alatu."""
    data, junit = work / ".coverage", work / "junit.xml"
    t0 = time.monotonic()
    _run([_python(), "-m", "coverage", "run", f"--data-file={data}", "-m", "pytest", "-q",
          "-p", "no:cacheprovider", f"--junitxml={junit}", *sel], timeout=3600)
    rep = _run([_python(), "-m", "coverage", "report", f"--data-file={data}"], timeout=600)
    if not junit.exists():
        raise ToolError("pytest nije napravio izveštaj — obrazac ne odgovara nijednom testu?")
    # junit.xml je naš, iz `tmp/` — ne tuđ ulaz (XXE nije pretnja)
    suite = ET.parse(junit).getroot()
    suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
    failures = []
    for tc in suite.iter("testcase"):
        bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
        if bad is not None and len(failures) < 20:
            msg = (bad.get("message") or bad.text or "").strip().splitlines()
            failures.append({"test": f"{tc.get('classname')}::{tc.get('name')}",
                             "poruka": "\n".join(msg[:6])[:800]})
    total = re.search(r"^TOTAL.*?(\d+)%\s*$", rep.stdout, re.M)
    n = int(suite.get("tests", 0))
    failed, errors, skipped = (int(suite.get(k, 0)) for k in ("failures", "errors", "skipped"))
    for f in (data, junit):
        f.unlink(missing_ok=True)  # privremeno se briše za sobom; ostaje samo rezultat
    return {"proslo": n - failed - errors - skipped, "palo": failed, "greske": errors,
            "preskoceno": skipped, "pokrivenost": f"{total.group(1)} %" if total else None,
            "trajanje_s": round(time.monotonic() - t0, 1), "palo_detalji": failures}


def _script_args(ime: str, argumenti: list[str] | None) -> list[str]:
    name = ime[:-3] if ime.endswith(".py") else ime
    if name not in SCRIPTS:
        raise ToolError(f"Skripta {ime!r} nije na spisku dozvoljenih: {', '.join(SCRIPTS)}.")
    args = list(argumenti or [])
    for a in args:
        if not _ARG_RE.match(a):
            raise ToolError(f"Argument {a!r} nije dozvoljen (samo slova, brojevi i . , : = / \\ - _).")
    return [str(ROOT / "scripts" / f"{name}.py"), *args]


def _script_work(args: list[str], work: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    r = _run([_python(), *args], timeout=3600)
    out = r.stdout
    return {"kod": r.returncode, "izlaz": out[-20000:], "skraceno": len(out) > 20000,
            "greske": r.stderr[-3000:] if r.returncode else "", "trajanje_s": round(time.monotonic() - t0, 1)}


# --- pokretanja u pozadini (krug 29) -----------------------------------------
#
# Veza do ovog procesa ističe posle ~60 s, a testovi traju minutima. Zato
# `tests` i `script` samo pokrenu pokretač (`python -m subtitis.mcp_server run
# <oznaka>`) i odmah vrate oznaku; `zadatak(oznaka)` kaže da li još radi i
# koliko traje, ili vraća rezultat. Sve je u `tmp/runs/<oznaka>/`:
# `meta.json` (vrsta, argumenti, PID i vreme pokretanja pokretača) i, na kraju,
# `result.json`. `tmp/` se prazni pri pokretanju servera — pokretanje koje
# radi ostaje (`config.empty_tmp`).

_RUN_ID_RE = re.compile(r"^(tests|script)-\d{8}-\d{6}-[0-9a-f]{6}$")
WORKERS = {"tests": _tests_work, "script": _script_work}


def runs_dir() -> Path:
    return cfg.TMP_DIR / cfg.RUNS_DIR_NAME


def _launch(kind: str, args: list[str]) -> dict[str, Any]:
    from .proc import start_time

    run_id = f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{os.urandom(3).hex()}"
    d = runs_dir() / run_id
    d.mkdir(parents=True)
    meta = {"kind": kind, "args": args, "started_at": time.time()}
    cfg.write_json(d / "meta.json", meta)
    proc = subprocess.Popen(
        [_python(), "-m", "subtitis.mcp_server", "run", run_id], cwd=ROOT,
        env=_child_env() | {"PYTHONIOENCODING": "utf-8", "SUBTITIS_TMP_DIR": str(cfg.TMP_DIR)},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    meta.update(pid=proc.pid, pid_started=start_time(proc.pid))
    cfg.write_json(d / "meta.json", meta)
    return {"oznaka": run_id, "radi": True,
            "poruka": f"Pokrenuto u pozadini — prati sa zadatak({run_id!r})."}


def run_main(run_id: str) -> int:
    """Pokretač: uradi posao i upiši `result.json` (i kad padne)."""
    d = runs_dir() / run_id
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    try:
        result = WORKERS[meta["kind"]](meta["args"], d)
    except Exception as exc:  # noqa: BLE001 — rezultat mora da postoji
        result = {"greska": str(exc) if isinstance(exc, ToolError) else f"{type(exc).__name__}: {exc}"}
    tmp = d / "result.json.part"
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(d / "result.json")  # atomično: `zadatak` nikad ne vidi pola fajla
    return 0


@_safe
def tests(obrazac: str | None = None) -> dict[str, Any]:
    """pytest uz pokrivenost, U POZADINI: vraća odmah oznaku; rezultat (prošlo,
    palo, pokrivenost, poruke samo za palo) daje zadatak(oznaka).
    `obrazac` je putanja unutar tests/ (`tests/test_srt.py`) ili `-k` izraz."""
    return _launch("tests", _tests_args(obrazac))


@_safe
def script(ime: str, argumenti: list[str] | None = None) -> dict[str, Any]:
    """Merna skripta iz `scripts/` (zatvoren spisak), U POZADINI: vraća odmah
    oznaku; izlaz daje zadatak(oznaka)."""
    return _launch("script", _script_args(ime, argumenti))


@_safe
def zadatak(oznaka: str) -> dict[str, Any]:
    """Tok pokretanja iz tests()/script(): dok radi — koliko traje; kad završi —
    rezultat."""
    from .proc import same_process

    if not _RUN_ID_RE.match(oznaka or ""):
        raise ToolError(f"Nepoznata oznaka {oznaka!r}.")
    d = runs_dir() / oznaka
    try:
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ToolError(f"Nema pokretanja {oznaka!r} (tmp/ se prazni pri pokretanju servera).") from None
    took = round(time.time() - meta["started_at"], 1)
    res = d / "result.json"
    if res.exists():
        return {"gotovo": True, "oznaka": oznaka, "vrsta": meta["kind"],
                **json.loads(res.read_text(encoding="utf-8"))}
    if "pid" in meta and not same_process(meta["pid"], meta.get("pid_started")):
        raise ToolError(f"Pokretanje {oznaka!r} je prekinuto posle {took} s — proces više ne radi, "
                        "a rezultata nema.")
    return {"gotovo": False, "oznaka": oznaka, "vrsta": meta["kind"], "traje_s": took}


TOOLS = (status, start, stop, jobs, job, transcribe, translate, rebuild, export, text, flags,
         tests, script, zadatak)


def build_server():
    from mcp.server.fastmcp import FastMCP

    srv = FastMCP("subtitis", instructions=(
        "Subtitis pravi i prevodi SRT titlove na Mirkovoj mašini. status() kaže radi li "
        "dashboard; start() ga pokreće. Dugi poslovi (transcribe, translate) vraćaju id "
        "odmah — prati ih preko job(id). text i flags vraćaju najviše 50 / 20 stavki."))
    for fn in TOOLS:
        srv.add_tool(fn)
    return srv


def main() -> None:
    import logging

    # stdout je kanal protokola; httpx bi svaki poziv beležio na stderr
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_server().run("stdio")


# ==========================================================================
# Instalacija u Claude desktop aplikaciju
# ==========================================================================

SERVER_KEY = "subtitis"


def claude_config_path() -> Path:
    """Konfiguracija Claude desktop aplikacije.

    Aplikacija iz Microsoft Store-a (MSIX) danas drži podatke u
    `%LOCALAPPDATA%\Claude-Data\<paket>\`; u starom preusmerenom folderu paketa
    (`Packages\Claude_*\LocalCache\Roaming\Claude`) ostaje samo oznaka
    `.user-data-left.*` (izmereno 2026-10-20, krug 28). Redom: `Claude-Data`,
    pa folder paketa ako podaci nisu iseljeni, pa obična instalacija
    (`%APPDATA%\Claude`)."""
    local = os.environ.get("LOCALAPPDATA")
    if local:
        for d in sorted(Path(local, "Claude-Data").glob("claude_*")):
            if d.is_dir():
                return d / "claude_desktop_config.json"
        for pkg in sorted(Path(local, "Packages").glob("Claude_*")):
            d = pkg / "LocalCache" / "Roaming" / "Claude"
            if d.is_dir() and not any(d.glob(".user-data-left*")):
                return d / "claude_desktop_config.json"
    return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"), "Claude",
                "claude_desktop_config.json")


def server_entry() -> dict[str, Any]:
    return {
        "command": str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "args": ["-m", "subtitis", "mcp"],
        "env": {"PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8"},
    }


def _load(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, data: dict[str, Any]) -> Path | None:
    bak = None
    if path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return bak


def install(path: Path | None = None) -> dict[str, Any]:
    """Upiše samo `mcpServers.subtitis`; ostalo ne dira. Drugi put ne menja ništa."""
    path = path or claude_config_path()
    data = _load(path)
    entry = server_entry()
    servers = data.setdefault("mcpServers", {})
    if servers.get(SERVER_KEY) == entry:
        return {"promenjeno": False, "fajl": str(path), "unos": entry}
    servers[SERVER_KEY] = entry
    bak = _save(path, data)
    return {"promenjeno": True, "fajl": str(path), "rezerva": str(bak) if bak else None, "unos": entry}


def uninstall(path: Path | None = None) -> dict[str, Any]:
    path = path or claude_config_path()
    data = _load(path)
    if SERVER_KEY not in data.get("mcpServers", {}):
        return {"promenjeno": False, "fajl": str(path)}
    del data["mcpServers"][SERVER_KEY]
    bak = _save(path, data)
    return {"promenjeno": True, "fajl": str(path), "rezerva": str(bak) if bak else None}


if __name__ == "__main__":  # pokretač pozadinskih pokretanja (`_launch`)
    if sys.argv[1:2] == ["run"] and len(sys.argv) == 3 and _RUN_ID_RE.match(sys.argv[2]):
        raise SystemExit(run_main(sys.argv[2]))
    raise SystemExit("upotreba: python -m subtitis.mcp_server run <oznaka>")
