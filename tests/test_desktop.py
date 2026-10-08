"""Testovi desktop pokretanja.

Ne diže se pravi server ni traka — proverava se logika oko njih: prepoznavanje
da instanca već radi, adresa koja se otvara, i prečica.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

from subtitis import desktop
from subtitis.config import SubtitisError


@pytest.fixture()
def slusalac():
    """Otvori pravi soket na slobodnom portu i vrati (host, port)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()
    yield host, port
    srv.close()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- prepoznavanje da instanca već radi ------------------------------------


def test_zauzet_port_se_prepoznaje(slusalac):
    host, port = slusalac
    assert desktop.port_in_use(host, port) is True


def test_slobodan_port_se_prepoznaje():
    assert desktop.port_in_use("127.0.0.1", free_port()) is False


import json as _json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def http_server(body: dict | str, port: int = 0):
    """HTTP server (podrazumevano na slobodnom portu) koji na sve odgovara istim telom."""

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = (_json.dumps(body) if isinstance(body, dict) else body).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


@pytest.fixture()
def subtitis_radi():
    """Lažni Subtitis: odgovara na /api/identity kao pravi."""
    from subtitis.web.app import IDENTITY

    srv = http_server({"app": IDENTITY, "version": "test"})
    yield srv.server_address
    srv.shutdown()


@pytest.fixture()
def tudja_aplikacija():
    """Tuđa web aplikacija na portu — odgovara, ali nije Subtitis."""
    srv = http_server("<html>neka druga aplikacija</html>")
    yield srv.server_address
    srv.shutdown()


@pytest.fixture(autouse=True)
def kratko_cekanje(monkeypatch):
    # tuđ soket koji ne odgovara ne sme da uspori testove
    monkeypatch.setattr(desktop, "IDENTITY_TIMEOUT_SECONDS", 0.3)


def test_stariji_subtitis_bez_identiteta_nije_tudj():
    """Instanca pre /api/identity: 404 na identitet, ali naslov strane je Subtitis.
    Proglasiti je tuđom značilo bi dići drugu instancu uz nju."""

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/":
                data = b"<html><head><title>Subtitis</title></head></html>"
                self.send_response(200)
            else:
                data = b'{"detail":"Not Found"}'
                self.send_response(404)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert desktop.is_subtitis(*srv.server_address) is True
    finally:
        srv.shutdown()


def test_tudja_aplikacija_sa_404_nije_subtitis():
    """404 na identitet i naslov bez „Subtitis" — tuđa aplikacija."""

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = b"<title>Neka druga aplikacija</title>"
            self.send_response(404 if self.path != "/" else 200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert desktop.is_subtitis(*srv.server_address) is False
    finally:
        srv.shutdown()


def test_identitet_razlikuje_subtitis(subtitis_radi, tudja_aplikacija, slusalac):
    assert desktop.is_subtitis(*subtitis_radi) is True
    assert desktop.is_subtitis(*tudja_aplikacija) is False
    assert desktop.is_subtitis(*slusalac) is False       # sluša, ne odgovara
    assert desktop.is_subtitis("127.0.0.1", free_port()) is False


def test_pravi_server_odgovara_identitetom():
    from fastapi.testclient import TestClient

    from subtitis.web.app import IDENTITY, app

    with TestClient(app) as c:
        assert c.get("/api/identity").json()["app"] == IDENTITY


def test_druga_instanca_se_ne_dize(subtitis_radi, monkeypatch):
    """Dva procesa nad istom bazom = dva workera i dva modela u VRAM-u."""
    host, port = subtitis_radi
    otvoreno = []
    monkeypatch.setattr(desktop.webbrowser, "open", otvoreno.append)

    def ne_sme(*a, **k):  # pragma: no cover — poziv bi bio greška
        raise AssertionError("server ne sme da se digne kad instanca već radi")

    monkeypatch.setattr("uvicorn.Server", ne_sme)
    assert desktop.run(host, port, say=lambda m: None) == 0
    assert otvoreno == [desktop.url_for(host, port)]


def test_druga_instanca_bez_pregledaca(subtitis_radi, monkeypatch):
    host, port = subtitis_radi
    otvoreno = []
    monkeypatch.setattr(desktop.webbrowser, "open", otvoreno.append)
    assert desktop.run(host, port, open_browser=False, say=lambda m: None) == 0
    assert otvoreno == []


@pytest.mark.parametrize("tudj", ["tudja_aplikacija", "slusalac"])
def test_tudj_port_nije_subtitis_nego_sledeci_slobodan(tudj, request):
    """Kvar: tuđa aplikacija na portu je ispadala „Subtitis već radi"."""
    host, port = request.getfixturevalue(tudj)
    choice = desktop.choose_port(host, port)
    assert choice.running is False
    assert choice.port > port and choice.foreign[0] == port
    assert not desktop.port_in_use(host, choice.port)
    line = choice.line(host)
    assert f"port {port} drži druga aplikacija" in line
    assert f":{choice.port}" in line


def test_nalazi_subtitis_iznad_tudjeg_porta(tudja_aplikacija):
    """Prošli put se digao na sledećem portu — sad ga mora naći tamo, ne dići drugi."""
    from subtitis.web.app import IDENTITY

    host, port = tudja_aplikacija
    try:
        srv = http_server({"app": IDENTITY}, port=port + 1)
    except OSError:
        pytest.skip("susedni port zauzet")
    try:
        choice = desktop.choose_port(host, port)
        assert (choice.port, choice.running, choice.foreign) == (port + 1, True, [port])
    finally:
        srv.shutdown()


def test_run_se_dize_na_sledecem_i_kaze_na_kom(tudja_aplikacija, monkeypatch):
    from subtitis.web.app import WORKER_ENV

    # `run` pali workera kroz okruženje; bez vraćanja bi svaki kasniji
    # TestClient digao pravog workera koji troši zadatke drugih testova.
    monkeypatch.setenv(WORKER_ENV, "0")
    host, port = tudja_aplikacija
    rečeno: list[str] = []
    dignut: list[int] = []

    class Stani(Exception):
        pass

    def config(app, host, port, **kw):
        dignut.append(port)
        raise Stani

    monkeypatch.setattr("uvicorn.Config", config)
    with pytest.raises(Stani):
        desktop.run(host, port, say=rečeno.append)
    assert dignut and dignut[0] > port
    assert f"Subtitis se diže na {desktop.url_for(host, dignut[0])}" in rečeno[0]
    assert f"port {port} drži druga aplikacija" in rečeno[0]


def test_svi_portovi_tudji(slusalac):
    host, port = slusalac
    with pytest.raises(SubtitisError, match="svi zauzeti"):
        desktop.choose_port(host, port, scan=0)


def test_podrazumevani_port_iz_podesavanja(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.delenv("SUBTITIS_PORT", raising=False)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    assert cfg.port() == 8010
    (tmp_path / "subtitis.toml").write_text("port = 8123" + chr(10), encoding="utf-8")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    assert cfg.port() == 8123
    # ekran podešavanja prepisuje fajl — port mora da preživi
    cfg.save_settings({"llm_model": "x"})
    assert cfg.port() == 8123
    monkeypatch.setenv("SUBTITIS_PORT", "8200")
    assert cfg.port() == 8200


# --- adresa ----------------------------------------------------------------


def test_adresa_za_slusanje_nije_adresa_za_pregledac():
    """`0.0.0.0` je adresa za slušanje; pregledač je ne otvara."""
    assert desktop.url_for("0.0.0.0", 8010) == "http://127.0.0.1:8010"
    assert desktop.url_for("::", 8010) == "http://127.0.0.1:8010"
    assert desktop.url_for("192.168.1.5", 9000) == "http://192.168.1.5:9000"


def test_cekanje_odustaje_kad_niko_ne_slusa():
    assert desktop.wait_until_listening("127.0.0.1", free_port(), 0.3) is False


def test_cekanje_odmah_uspeva_kad_neko_slusa(slusalac):
    host, port = slusalac
    assert desktop.wait_until_listening(host, port, 2.0) is True


# --- ikonica ---------------------------------------------------------------


def test_ikonica_postoji_i_ima_vise_velicina():
    assert desktop.ICON_PATH.exists(), "pokreni scripts/make_icon.py"
    from PIL import Image

    with Image.open(desktop.ICON_PATH) as im:
        sizes = sorted(im.info.get("sizes", []))
    assert (16, 16) in sizes, "traka koristi 16 px"
    assert (256, 256) in sizes, "Explorer koristi krupnu"


# --- prečica ---------------------------------------------------------------


def test_precica_cilja_interpreter_bez_konzole():
    from subtitis.shortcut import python_launcher

    exe, style = python_launcher()
    if exe.name == "pythonw.exe":
        assert style == 1
    else:  # pragma: no cover — venv bez pythonw
        assert style == 7, "bez pythonw prozor bar ide u minimizovan"


@pytest.mark.skipif(sys.platform != "win32", reason="prečice su Windows stvar")
def test_precica_se_pravi(tmp_path):
    from subtitis.shortcut import create_shortcut

    link = create_shortcut(name="Subtitis-test", target_dir=tmp_path)
    assert link.exists()
    assert link.suffix == ".lnk"
    assert link.stat().st_size > 0


@pytest.mark.skipif(sys.platform != "win32", reason="Windows API")
def test_desktop_se_nalazi_i_kad_je_lokalizovan():
    """Na srpskom Windows-u folder se zove `Radna površina`, ne `Desktop`.

    Uz OneDrive nije ni pod `%USERPROFILE%`. Traženje po imenu je pogađanje;
    jedini tačan izvor je `SHGetKnownFolderPath`.
    """
    from subtitis.shortcut import _desktop_from_windows, desktop_dir

    api = _desktop_from_windows()
    assert api is not None and api.exists()
    assert desktop_dir() == api


def test_desktop_pada_nazad_na_lokalizovano_ime(tmp_path, monkeypatch):
    from subtitis import shortcut

    (tmp_path / "Radna površina").mkdir()
    monkeypatch.setattr(shortcut, "_desktop_from_windows", lambda: None)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OneDrive", raising=False)
    assert shortcut.desktop_dir() == tmp_path / "Radna površina"


def test_desktop_javlja_jasno_kad_ga_nema(tmp_path, monkeypatch):
    from subtitis import shortcut

    monkeypatch.setattr(shortcut, "_desktop_from_windows", lambda: None)
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "nema"))
    monkeypatch.delenv("OneDrive", raising=False)
    monkeypatch.delenv("HOME", raising=False)
    with pytest.raises(SubtitisError, match="--dir"):
        shortcut.desktop_dir()


def test_precica_van_windowsa_javlja_jasno(monkeypatch):
    from subtitis import shortcut

    monkeypatch.setattr(shortcut.sys, "platform", "linux")
    with pytest.raises(SubtitisError, match="Windows"):
        shortcut.create_shortcut()
