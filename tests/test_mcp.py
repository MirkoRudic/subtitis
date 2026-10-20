"""MCP server (krug 28): alati, i dva slučaja koja će se stvarno dešavati —
dashboard ne radi, posao ne postoji. Alati idu kroz pravi HTTP API
(`TestClient`), bez pokretanja servera; proces (`start`/`stop` uz Job
objekat) je u `test_mcp_proces.py`."""

from __future__ import annotations

import json

import pytest
from test_web import make_job

from subtitis import mcp_server as m
from subtitis.models import Sentence, SentencesFile


class _Same:
    def __init__(self, client):
        self.c = client

    def __enter__(self):
        return self.c

    def __exit__(self, *a):
        return False


@pytest.fixture
def api(tmp_path, monkeypatch):
    """Izolovan Subtitis iza alata: `find_server` i `_http` vode u TestClient."""
    from fastapi.testclient import TestClient

    from subtitis import config as cfg
    from subtitis.web import app as app_mod
    from subtitis.web import db

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    db.init_db()
    with TestClient(app_mod.app) as client:
        ident = client.get("/api/identity").json()
        monkeypatch.setattr(m, "find_server", lambda: (8010, ident))
        monkeypatch.setattr(m, "_http", lambda port: _Same(client))
        yield client, db, cfg


def _not_running(monkeypatch):
    monkeypatch.setattr(m, "find_server", lambda: None)


# --------------------------------------------------------------------------
# dashboard ne radi / posao ne postoji
# --------------------------------------------------------------------------

POSLOVNI = [
    ("job", ("x",)), ("translate", ("x", ["en"])), ("rebuild", ("x",)), ("export", ("x",)),
    ("text", ("x",)), ("flags", ("x",)),
]


def test_dashboard_ne_radi_svaki_alat_kaze_jasno(monkeypatch):
    _not_running(monkeypatch)
    monkeypatch.setattr(m, "_port_open", lambda port: False)
    for name, args in POSLOVNI + [("jobs", ()), ("transcribe", ("C:/nema.wav",))]:
        r = getattr(m, name)(*args)
        assert r == {"greska": m.NOT_RUNNING}, name
    s = m.status()
    assert s["radi"] is False and "start()" in s["poruka"]
    assert m.stop() == {"ugasen": True, "poruka": "Subtitis već ne radi."}


def test_posao_ne_postoji_svaki_alat_kaze_jasno(api):
    for name, args in POSLOVNI:
        r = getattr(m, name)(*args)
        assert set(r) == {"greska"} and "Nema projekta x" in r["greska"], (name, r)


# --------------------------------------------------------------------------
# upravljanje
# --------------------------------------------------------------------------


def test_status_iz_identiteta(api):
    from subtitis import ROUND

    s = m.status()
    assert s["radi"] and s["port"] == 8010 and s["krug"] == ROUND
    assert s["potpis"].startswith("Subtitis ") and s["u_redu"] == 0 and s["radi_na"] is None


def test_identitet_nosi_red_i_ono_sto_se_radi(api):
    client, db, cfg = api
    jid = make_job(cfg, db)
    tid = db.enqueue(jid, "translate", {"to": "en"})
    with db.connect() as conn:
        conn.execute("UPDATE tasks SET state='running' WHERE id=?", (tid,))
    db.add_event(jid, "progress", "translate", 0.4)
    ident = client.get("/api/identity").json()
    assert ident["queue"] == 1 and ident["running"]["job"] == jid
    assert ident["running"]["kind"] == "translate" and ident["running"]["progress"] == 0.4


def test_gasenje_odbija_dok_radi_osim_uz_force(api, monkeypatch):
    from subtitis.web import api as api_mod

    client, db, cfg = api
    assert "ne podržava gašenje" in m.stop()["greska"]  # TestClient nema SHUTDOWN
    called = []
    monkeypatch.setattr(api_mod, "SHUTDOWN", lambda: called.append(1))
    monkeypatch.setattr("threading.Timer", lambda t, fn: type("T", (), {"start": lambda s: fn()})())
    jid = make_job(cfg, db)
    tid = db.enqueue(jid, "transcribe")
    with db.connect() as conn:
        conn.execute("UPDATE tasks SET state='running' WHERE id=?", (tid,))
    r = m.stop()
    assert r["ugasen"] is False and "force" in r["poruka"] and r["radi_na"]["job"] == jid
    assert not called
    monkeypatch.setattr(m, "_port_open", lambda port: False)
    r = m.stop(force=True)
    assert r["ugasen"] is True and r["prekinut"]["job"] == jid and called == [1]


def _rezerva(monkeypatch, owner_pid, rec):
    """Server ne odgovara, port drži `owner_pid`, `tmp/server.pid` je `rec`."""
    from subtitis import config as cfg

    _not_running(monkeypatch)
    monkeypatch.setattr(m, "_port_open", lambda port: True)
    monkeypatch.setattr(m, "_port_owner", lambda port: {"pid": owner_pid, "exe": r"C:\Python311\pythonw.exe"})
    killed = []
    monkeypatch.setattr(m, "_kill", lambda pid: killed.append(pid))
    cfg.server_pid_path().unlink(missing_ok=True)
    if rec is not None:
        cfg.write_json(cfg.server_pid_path(), rec)
    return killed


def test_rezerva_bez_pid_fajla_ne_gasi_nista(monkeypatch):
    killed = _rezerva(monkeypatch, 111, None)
    r = m.stop()
    assert r["ugasen"] is False and "ne postoji" in r["poruka"] and killed == []


def test_rezerva_kad_se_brojevi_ne_poklope_javi_oba_i_stane(monkeypatch):
    import os

    from subtitis.proc import start_time

    killed = _rezerva(monkeypatch, 111, {"pid": os.getpid(), "started": start_time(os.getpid())})
    r = m.stop()
    assert r["ugasen"] is False and killed == []
    assert r["vlasnik_porta"] == 111 and r["server_pid"] == os.getpid()
    assert "111" in r["poruka"] and str(os.getpid()) in r["poruka"]


def test_rezerva_zaostao_fajl_sa_tudjim_pidom_ne_gasi(monkeypatch):
    """Isti PID, ali drugo vreme pokretanja — PID je u međuvremenu dobio drugi proces."""
    import os

    from subtitis.proc import start_time

    killed = _rezerva(monkeypatch, os.getpid(), {"pid": os.getpid(), "started": start_time(os.getpid()) - 1})
    r = m.stop()
    assert r["ugasen"] is False and "zaostao fajl" in r["poruka"] and killed == []


def test_rezerva_gasi_tacno_server_iz_pid_fajla(monkeypatch):
    import os

    from subtitis.proc import start_time

    killed = _rezerva(monkeypatch, os.getpid(), {"pid": os.getpid(), "started": start_time(os.getpid())})
    r = m.stop()
    assert killed == [os.getpid()], "tačno jedan proces, nikad roditelj ni stablo"
    assert "iz server.pid" in r["poruka"]


def test_server_pise_pid_i_brise_ga_pri_urednom_gasenju(tmp_path, monkeypatch):
    import os

    from fastapi.testclient import TestClient

    from subtitis import config as cfg
    from subtitis.proc import start_time
    from subtitis.web import app as app_mod

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    with TestClient(app_mod.app):
        rec = json.loads(cfg.server_pid_path().read_text(encoding="utf-8"))
        assert rec["pid"] == os.getpid() and rec["started"] == start_time(os.getpid())
    assert not cfg.server_pid_path().exists()


def test_start_ne_dize_drugi_kad_vec_radi(api, monkeypatch):
    monkeypatch.setattr(m, "spawn_detached", lambda: pytest.fail("ne sme drugi"))
    r = m.start()
    assert r["radi"] and r["pokrenut"] is False


def test_start_odvojen_sa_wmi_rezervom(monkeypatch):
    seen = {}

    def fake_popen(cmd, **kw):
        seen["flags"] = kw["creationflags"]
        seen["cmd"] = cmd
        err = OSError("Access is denied")
        err.winerror = 5
        raise err

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(m, "spawn_wmi", lambda cmd, env: seen.update(wmi=cmd, env=env))
    how = m.spawn_detached()
    assert seen["flags"] & m.CREATE_BREAKAWAY_FROM_JOB and seen["flags"] & m.DETACHED_PROCESS
    assert seen["cmd"][1:] == ["-m", "subtitis", "desktop", "--no-open"]
    assert seen["cmd"][0].lower().endswith("pythonw.exe")
    assert "WMI" in how and seen["wmi"] == seen["cmd"]
    assert str(m.ROOT) in seen["env"]["PYTHONPATH"]


# --------------------------------------------------------------------------
# poslovi i rad
# --------------------------------------------------------------------------


def test_jobs_i_job(api):
    client, db, cfg = api
    jid = make_job(cfg, db)
    r = m.jobs()
    assert r["total"] == 1 and r["jobs"][0]["id"] == jid and r["jobs"][0]["languages"] == ["sr"]
    assert m.jobs(stanje="error")["total"] == 0
    j = m.job(jid)
    assert j["status"] == "transcribed" and j["languages"] == ["sr"] and j["pending"] is None
    assert j["warnings"] == {"nesigurno": 1}  # `make_job`: rečenica sa nesigurnom reči


def test_transcribe_sa_diska(api, tmp_path):
    client, db, cfg = api
    src = tmp_path / "snimak.wav"
    src.write_bytes(b"RIFF")
    r = m.transcribe(str(src), "sr", "digital")
    jid = r["id"]
    assert (cfg.job_dir(jid) / "upload" / "snimak.wav").read_bytes() == b"RIFF"
    assert src.exists(), "kopira se, original ostaje"
    t = db.pending_task(jid)
    assert t["kind"] == "transcribe" and json.loads(t["params"])["profile"] == "digital"
    assert "Nema fajla" in m.transcribe(str(tmp_path / "nema.wav"))["greska"]
    assert "Nepoznat profil 'nepostojeci'" in m.transcribe(str(src), "sr", "nepostojeci")["greska"]


def test_translate_vise_jezika(api):
    client, db, cfg = api
    jid = make_job(cfg, db)
    r = m.translate(jid, ["en", "fr"])
    assert r["queued"] == ["translate:en", "translate:fr"]
    with db.connect() as conn:
        tos = [json.loads(x["params"])["to"] for x in conn.execute("SELECT params FROM tasks").fetchall()]
    assert tos == ["en", "fr"]
    assert "već postoji zadatak" in m.translate(jid, ["de"])["greska"]
    assert "Izvorni jezik" in m.translate(make_job(cfg, db, job_id="drugi"), ["sr"])["greska"]


def test_rebuild_i_export(api):
    client, db, cfg = api
    jid = make_job(cfg, db)
    r = m.rebuild(jid, "tv")
    assert r["profile"] == "tv" and r["cues"] >= 1 and r["path"].endswith(".sr-Latn.srt")
    e = m.export(jid)
    assert e["path"] == r["path"]
    assert "nema SRT za 'fr'" in m.export(jid, "fr")["greska"]


def _many(cfg, jid, n, flag=None):
    jdir = cfg.job_dir(jid)
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[Sentence(id=k, s=k, e=k + 0.5, text=f"Rečenica broj {k}.", words=(0, 0),
                            flags=[flag] if flag else []) for k in range(1, n + 1)],
    ))


def test_text_najvise_50_i_kaze_koliko_jos(api):
    client, db, cfg = api
    jid = make_job(cfg, db)
    _many(cfg, jid, 60)
    r = m.text(jid)
    assert len(r["sentences"]) == 50 and r["more"] == 10 and r["next"] == 51
    assert r["sentences"][0] == {"id": 1, "at": "00:01,00", "text": "Rečenica broj 1.", "flags": []}
    r = m.text(jid, od=51)
    assert len(r["sentences"]) == 10 and r["more"] == 0 and r["next"] is None
    assert len(m.text(jid, od=5, do=7)["sentences"]) == 3
    assert "nema tekst na jeziku 'en'" in m.text(jid, "en")["greska"]


def test_flags_najvise_20_sa_tekstom(api):
    client, db, cfg = api
    jid = make_job(cfg, db)
    _many(cfg, jid, 25, flag="low_confidence")
    r = m.flags(jid, "nesigurno")
    assert len(r["flags"]) == 20 and r["total"] == 25 and r["more"] == 5
    assert r["flags"][0]["text"] == "Rečenica broj 1." and r["flags"][0]["kind"] == "nesigurno"
    assert m.job(jid)["warnings"]["nesigurno"] == 25
    assert "Nepoznata vrsta" in m.flags(jid, "nesto")["greska"]


# --------------------------------------------------------------------------
# krugovi: testovi i skripte
# --------------------------------------------------------------------------


def test_script_samo_sa_spiska(monkeypatch):
    assert "nije na spisku" in m.script("make_icon")["greska"]
    assert "nije dozvoljen" in m.script("krug25", ["a; rm -rf"])["greska"]
    seen = {}
    monkeypatch.setattr(m, "_launch", lambda kind, args: seen.update(kind=kind, args=args) or {"oznaka": "x"})
    assert m.script("krug25.py", ["udaljenost"]) == {"oznaka": "x"}
    assert seen["kind"] == "script" and seen["args"][0].endswith("krug25.py") and seen["args"][1:] == ["udaljenost"]


def test_tests_samo_unutar_foldera_tests():
    """Bezbednosni pregled kruga 28: `tests/../x.py` bi pytestu dao bilo koji .py."""
    assert "nije unutar tests/" in m.tests("tests/../subtitis/cli.py")["greska"]
    assert "nije unutar tests/" in m.tests("tests\\..\\scripts\\krug25.py")["greska"]
    assert m._tests_args("tests/test_srt.py::test_x") == ["tests/test_srt.py::test_x"]
    assert m._tests_args("sudar and not proces") == ["-k", "sudar and not proces"]


def _cekaj(oznaka, sekundi=180):
    import time

    deadline = time.monotonic() + sekundi
    while time.monotonic() < deadline:
        r = m.zadatak(oznaka)
        if r.get("gotovo") or "greska" in r:
            return r
        assert r["traje_s"] >= 0 and r["vrsta"] in ("tests", "script")
        time.sleep(0.5)
    raise AssertionError("nije završeno")


def test_tests_vraca_odmah_pa_zadatak_daje_rezultat():
    """Krug 29: veza do Claude-a ističe posle ~60 s — `tests` ne sme da čeka."""
    import time

    t0 = time.monotonic()
    r = m.tests("tests/test_css_sudari.py")
    assert time.monotonic() - t0 < 5, "mora da vrati odmah"
    assert r["radi"] is True and r["oznaka"].startswith("tests-")
    prvi = m.zadatak(r["oznaka"])
    assert prvi.get("gotovo") is False and "traje_s" in prvi
    res = _cekaj(r["oznaka"])
    assert res["gotovo"] and res["palo"] == 0 and res["proslo"] >= 5 and res["pokrivenost"].endswith("%")
    d = m.runs_dir() / r["oznaka"]
    assert sorted(x.name for x in d.iterdir()) == ["meta.json", "result.json"], "privremeno obrisano"


def test_script_u_pozadini_pa_rezultat(monkeypatch):
    monkeypatch.setitem(m.WORKERS, "script", lambda args, work: {"kod": 0, "izlaz": "ok"})
    r = m._launch("script", ["x"])
    assert m.run_main(r["oznaka"]) == 0  # pokretač u istom procesu, bez čekanja
    res = m.zadatak(r["oznaka"])
    assert res == {"gotovo": True, "oznaka": r["oznaka"], "vrsta": "script", "kod": 0, "izlaz": "ok"}


def test_zadatak_prekinut_i_nepoznat():
    from subtitis import config as cfg

    assert "Nepoznata oznaka" in m.zadatak("../../etc")["greska"]
    assert "Nema pokretanja" in m.zadatak("tests-20261020-120000-abcdef")["greska"]
    d = m.runs_dir() / "tests-20261020-120000-000000"
    d.mkdir(parents=True)
    cfg.write_json(d / "meta.json", {"kind": "tests", "args": [], "started_at": 0, "pid": 999999, "pid_started": 1})
    assert "prekinuto" in m.zadatak("tests-20261020-120000-000000")["greska"]


def test_ciscenje_tmp_ne_brise_pokretanje_koje_radi():
    import os

    from subtitis import config as cfg
    from subtitis.proc import start_time

    radi = m.runs_dir() / "tests-20261020-120000-aaaaaa"
    gotov = m.runs_dir() / "tests-20261020-120000-bbbbbb"
    for d in (radi, gotov):
        d.mkdir(parents=True)
        cfg.write_json(d / "meta.json", {"kind": "tests", "args": [], "started_at": 0,
                                         "pid": os.getpid(), "pid_started": start_time(os.getpid())})
    (gotov / "result.json").write_text("{}", encoding="utf-8")
    cfg.empty_tmp()
    assert radi.exists() and not gotov.exists()


def test_rebuild_dug_vraca_da_se_prati(api, monkeypatch):
    import httpx

    client, db, cfg = api
    jid = make_job(cfg, db)

    def spor(*a, **kw):
        assert kw.get("timeout") == m.REBUILD_TIMEOUT
        raise httpx.ReadTimeout("spor")

    monkeypatch.setattr(m, "_call", spor)
    r = m.rebuild(jid, "tv")
    assert r["gotovo"] is False and "job(id)" in r["poruka"]


def test_start_koji_ne_stigne_ne_vraca_gresku(monkeypatch):
    monkeypatch.setattr(m, "find_server", lambda: None)
    monkeypatch.setattr(m, "spawn_detached", lambda: "odvojen proces")
    monkeypatch.setattr(m, "START_TIMEOUT", 0.6)
    r = m.start()
    assert r["pokrenut"] is True and r["radi"] is False and "status()" in r["poruka"]


def test_nijedan_alat_ne_ceka_blizu_granice_veze():
    """Krug 29: veza ističe posle ~60 s; svaka granica čekanja ostaje ispod pola minuta."""
    assert m.START_TIMEOUT <= 20 and m.HTTP_TIMEOUT <= 20 and m.REBUILD_TIMEOUT <= 20
    assert m.STOP_TIMEOUT + m.SHUTDOWN_TIMEOUT <= 20
    assert m.PORT_SCAN * 0.5 <= 15  # `find_server` u najgorem slučaju


def test_tudji_host_se_odbija(api):
    """DNS rebinding: zahtev sa tuđim `Host` ne sme da stigne do API-ja."""
    client, db, cfg = api
    assert client.get("/api/identity", headers={"host": "napadac.example:8010"}).status_code == 400
    assert client.get("/api/identity", headers={"host": "127.0.0.1:8010"}).status_code == 200
    assert client.get("/api/identity", headers={"host": "localhost:8010"}).status_code == 200


# --------------------------------------------------------------------------
# instalacija
# --------------------------------------------------------------------------


def test_install_upise_samo_svoj_kljuc_i_drugi_put_nista(tmp_path):
    cfgf = tmp_path / "claude_desktop_config.json"
    cfgf.write_text(json.dumps({"mcpServers": {"drugi": {"command": "x"}}, "theme": "dark"}), encoding="utf-8")
    r = m.install(cfgf)
    assert r["promenjeno"] and r["rezerva"].endswith(".json.bak")
    data = json.loads(cfgf.read_text(encoding="utf-8"))
    assert data["theme"] == "dark" and data["mcpServers"]["drugi"] == {"command": "x"}
    entry = data["mcpServers"]["subtitis"]
    assert entry["args"] == ["-m", "subtitis", "mcp"] and entry["command"].endswith("python.exe")
    assert entry["env"]["PYTHONPATH"] == str(m.ROOT)
    before = cfgf.read_text(encoding="utf-8")
    assert m.install(cfgf)["promenjeno"] is False and cfgf.read_text(encoding="utf-8") == before
    assert m.uninstall(cfgf)["promenjeno"] is True
    assert "subtitis" not in json.loads(cfgf.read_text(encoding="utf-8"))["mcpServers"]
    assert m.uninstall(cfgf)["promenjeno"] is False


def test_install_bez_fajla_ga_napravi(tmp_path):
    cfgf = tmp_path / "Claude" / "claude_desktop_config.json"
    assert m.install(cfgf)["rezerva"] is None
    assert "subtitis" in json.loads(cfgf.read_text(encoding="utf-8"))["mcpServers"]


def test_konfiguracija_iz_microsoft_store_paketa(tmp_path, monkeypatch):
    """Redom: `Claude-Data` (današnja MSIX aplikacija), folder paketa ako
    podaci nisu iseljeni, pa `%APPDATA%\Claude`."""
    local = tmp_path / "local"
    msix = local / "Packages" / "Claude_abc" / "LocalCache" / "Roaming" / "Claude"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert m.claude_config_path() == tmp_path / "roaming" / "Claude" / "claude_desktop_config.json"
    msix.mkdir(parents=True)
    assert m.claude_config_path() == msix / "claude_desktop_config.json"
    (msix / ".user-data-left.claude_abc").write_text("")
    assert m.claude_config_path() == tmp_path / "roaming" / "Claude" / "claude_desktop_config.json"
    data = local / "Claude-Data" / "claude_abc"
    data.mkdir(parents=True)
    assert m.claude_config_path() == data / "claude_desktop_config.json"


def test_mcp_server_ima_sve_alate():
    import asyncio

    names = {t.name for t in asyncio.run(m.build_server().list_tools())}
    assert names == {"status", "start", "stop", "jobs", "job", "transcribe", "translate",
                     "rebuild", "export", "text", "flags", "tests", "script", "zadatak"}


def test_stari_dashboard_bez_rute_kaze_da_je_star(api, monkeypatch):
    client, db, cfg = api
    monkeypatch.setattr(m, "find_server", lambda: (8010, {"version": "1.5.0"}))
    r = m._safe(lambda: m._call("GET", "/api/nepostojeca-ruta"))()
    assert "starija verzija (1.5.0)" in r["greska"]


def test_jezik_ne_sme_da_izadje_iz_posla(api):
    """Bezbednosni pregled kruga 28: jezik ide u ime fajla."""
    client, db, cfg = api
    jid = make_job(cfg, db)
    for bad in ("../../x", "sr/../..", "sr.json", "SR-latn-x"):
        assert "Neispravna oznaka jezika" in m.text(jid, bad)["greska"], bad
        assert "Neispravna oznaka jezika" in m.export(jid, bad)["greska"], bad
    assert "Neispravna oznaka jezika" in m.translate(jid, ["../en"])["greska"]
    assert "Neispravna oznaka jezika" in m.rebuild(jid, "tv", jezik="..")["greska"]
    assert m.text(jid, "sr")["lang"] == "sr"
