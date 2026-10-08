"""Pokretanje bez standardnog izlaza — režim `pythonw.exe`.

Prečica na desktopu pokreće `pythonw.exe`, gde `sys.stdout` i `sys.stderr`
**ne postoje** (`None`), pa svaki ispis baca `AttributeError`. To je drugačije od
ranijeg problema sa kodnim rasporedom, gde je izlaz postojao ali nije umeo da
zapiše kvačice.

Testovi pod običnim interpreterom imaju uredan `stdout`, pa se on ovde gasi
namerno — u istom procesu (`monkeypatch`) i u zasebnom, gde se prolazi cela
ulazna tačka `python -m subtitis`.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from subtitis import bootstrap

ROOT = Path(__file__).resolve().parents[1]

STREAM_NAMES = ("stdout", "stderr", "__stdout__", "__stderr__")


@pytest.fixture()
def bez_izlaza(tmp_path, monkeypatch):
    """Vrati funkciju koja gasi izlaz; ona vraća `jobs/` direktorijum.

    Gašenje NE sme da bude u samoj fiksturi: pytest hvata izlaz i na početku
    faze poziva ponovo postavi `sys.stdout`, pa bi sve podmetnuto u pripremi
    bilo pregaženo. Zato se zove iz tela testa.
    """
    monkeypatch.setenv("SUBTITIS_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(bootstrap, "_crash_log_installed", False)
    # Kuke se kače na globalno stanje procesa; vrati ih posle testa.
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)

    def ugasi() -> Path:
        for name in STREAM_NAMES:
            monkeypatch.setattr(sys, name, None)
        return tmp_path / "jobs"

    return ugasi


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_entry(args: list[str], jobs: Path, code: str = "") -> subprocess.CompletedProcess:
    """Pokreni `python -m subtitis <args>` sa ugašenim stdout/stderr.

    `sitecustomize` nije opcija (uvozi se pre `-c`), pa se ulazna tačka poziva
    preko `runpy` iz koda koji je prethodno ugasio struje.
    """
    script = textwrap.dedent(
        f"""
        import runpy, sys
        sys.argv = ["subtitis"] + {args!r}
        {code}
        sys.stdout = sys.stderr = sys.__stdout__ = sys.__stderr__ = None
        runpy.run_module("subtitis", run_name="__main__", alter_sys=True)
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
        env={**os.environ, "SUBTITIS_JOBS_DIR": str(jobs)},
    )


# --- podmetanje struja -----------------------------------------------------


def test_struje_se_podmecu_kad_ih_nema(bez_izlaza):
    jobs = bez_izlaza()
    assert bootstrap.ensure_streams() is True
    for name in STREAM_NAMES:
        assert getattr(sys, name) is not None
    # `uvicorn` pita baš ovo pri pravljenju formatera loga.
    assert sys.stdout.isatty() is False
    sys.stdout.write("proba\n")
    sys.stdout.flush()
    assert "proba" in (jobs / bootstrap.CONSOLE_LOG_NAME).read_text("utf-8")


def test_postojece_struje_se_ne_diraju():
    pre = sys.stdout
    assert bootstrap.ensure_streams() is False
    assert sys.stdout is pre


def test_ispis_prezivljava_i_kad_jobs_nije_upisiv(tmp_path, monkeypatch):
    """Bez prava pisanja ostaje prazna kanta — ali ispis i dalje ne obara rad."""
    monkeypatch.setenv("SUBTITIS_JOBS_DIR", str(tmp_path / "jobs"))
    monkeypatch.setattr(bootstrap, "_open_console_log", lambda: bootstrap._NullStream())
    for name in STREAM_NAMES:
        monkeypatch.setattr(sys, name, None)
    assert bootstrap.ensure_streams() is True
    print("ništa ovo ne obara")
    assert sys.stdout.isatty() is False


def test_cfg_echo_prezivljava_odsustvo_izlaza(bez_izlaza):
    """`echo` sme da izgubi kvačicu, ali ne sme da baci — ni kad izlaza nema."""
    from subtitis import config as cfg

    jobs = bez_izlaza()
    cfg.echo("vraćeno je 5 rečenica")  # sa ugašenim stdout, bez bootstrap-a
    bootstrap.ensure_streams()
    cfg.echo("vraćeno je 5 rečenica")
    assert "rečenica" in (jobs / bootstrap.CONSOLE_LOG_NAME).read_text("utf-8")


def test_uvicorn_formater_prezivljava_odsustvo_izlaza(bez_izlaza):
    """Regresija: baš ovo je obaralo prečicu.

    `uvicorn.logging.DefaultFormatter.__init__` zove `sys.stdout.isatty()`, pa je
    pravljenje `uvicorn.Config` pucalo pre nego što se ijedan red ispiše.
    """
    import uvicorn

    bez_izlaza()
    with pytest.raises(ValueError, match="formatter"):
        uvicorn.Config("subtitis.web.app:app", host="127.0.0.1", log_level="warning")

    bootstrap.bootstrap()
    config = uvicorn.Config("subtitis.web.app:app", host="127.0.0.1", log_level="warning")
    assert config.host == "127.0.0.1"


def test_serve_ne_pada_bez_izlaza(bez_izlaza, monkeypatch):
    """Ne samo `desktop` — i `serve` diže `uvicorn` i ispisuje pre njega."""
    import uvicorn

    from subtitis import cli
    from subtitis.web.app import WORKER_ENV

    monkeypatch.setenv(WORKER_ENV, "0")  # da se izmena vrati posle testa
    pokrenuto: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: pokrenuto.update(k))

    bez_izlaza()
    bootstrap.bootstrap()
    cli.serve(host="127.0.0.1", port=free_port(), worker=False, open_browser=False)
    assert pokrenuto["host"] == "127.0.0.1"


# --- zapis o padu ----------------------------------------------------------


def test_pad_se_upisuje_u_datoteku(bez_izlaza):
    bez_izlaza()
    bootstrap.bootstrap()
    try:
        raise RuntimeError("namerno")
    except RuntimeError as exc:
        path = bootstrap.write_crash(type(exc), exc, exc.__traceback__)
    assert path is not None and path.name == bootstrap.CRASH_FILE_NAME
    text = path.read_text("utf-8")
    assert "RuntimeError: namerno" in text
    assert "test_pad_se_upisuje_u_datoteku" in text, "traži se pun traceback"


def test_kuka_se_ne_kaci_dvaput(bez_izlaza):
    bez_izlaza()
    bootstrap.install_crash_log()
    prva = sys.excepthook
    bootstrap.install_crash_log()
    assert sys.excepthook is prva


# --- cela ulazna tačka, u zasebnom procesu ---------------------------------


def test_ulazna_tacka_radi_bez_stdout_i_stderr(tmp_path):
    """`python -m subtitis desktop` sa `sys.stdout = sys.stderr = None`.

    Na portu odgovara lažni Subtitis, pa `desktop` prepozna da instanca već
    radi i uredno izađe umesto da digne server i blokira. Ispis na tom putu
    (`typer.echo`) je isti onaj koji je pod `pythonw` pucao.

    Goli soket više ne glumi instancu: otvoren port bez `/api/identity` je
    tuđa aplikacija, i `desktop` bi digao pravi server na sledećem portu.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from subtitis.web.app import IDENTITY

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            data = json.dumps({"app": IDENTITY}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    jobs = tmp_path / "jobs"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        proc = run_entry(["desktop", "--no-open", "--port", str(port)], jobs)
    finally:
        srv.shutdown()

    assert proc.returncode == 0, proc.stderr
    assert not (jobs / bootstrap.CRASH_FILE_NAME).exists()
    assert "Subtitis već radi" in (jobs / bootstrap.CONSOLE_LOG_NAME).read_text("utf-8")


def test_ulazna_tacka_bez_stdout_ne_pada_na_helpu(tmp_path):
    """`--help` je čist ispis kroz `click`, bez ijednog našeg reda."""
    jobs = tmp_path / "jobs"
    proc = run_entry(["--help"], jobs)
    assert proc.returncode == 0, proc.stderr
    text = (jobs / bootstrap.CONSOLE_LOG_NAME).read_text("utf-8")
    assert "desktop" in text and "serve" in text


def test_pad_pri_pokretanju_ostavlja_trag(tmp_path):
    """Pod `pythonw` pad je nevidljiv — mora da ostane datoteka sa traceback-om."""
    jobs = tmp_path / "jobs"
    proc = run_entry(
        ["desktop"], jobs,
        code="import subtitis.desktop as d; d.port_in_use = lambda *a, **k: 1 / 0",
    )
    assert proc.returncode != 0
    text = (jobs / bootstrap.CRASH_FILE_NAME).read_text("utf-8")
    assert "ZeroDivisionError" in text
    assert "port_in_use" in text, "traži se pun traceback, ne samo poruka"
