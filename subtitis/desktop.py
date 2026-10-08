"""Pokretanje kao obična desktop aplikacija.

    subtitis desktop

Bez terminala: prečica pokreće `pythonw.exe`, pa prozora nema. Umesto njega je
ikonica u sistemskoj traci sa „Otvori" i „Izađi".

**Jedna instanca.** Ako server već radi, ovo NE diže drugi — samo otvori
pregledač i izađe. Dva procesa nad istom SQLite bazom bi značila dva workera i
dva modela u VRAM-u istovremeno, što je upravo ono što ceo projekat izbegava.

Provera je pitanje upućeno portu (`/api/identity`), ne samo pokušaj konekcije:
otvoren port znači samo da NEŠTO sluša. Ranije je tuđa aplikacija na 8000
ispadala „Subtitis već radi" i pregledač se otvarao na tuđoj stranici. PID fajl
nije alternativa — preživi pad procesa i laže.

Kad port drži tuđa aplikacija, bira se sledeći slobodan (`choose_port`).

`pystray` i `Pillow` su NEOBAVEZNI. Bez njih server radi isto, samo bez ikonice
u traci — tada se gasi zatvaranjem prozora ili preko Task Managera.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as cfg

HERE = Path(__file__).resolve().parent
ICON_PATH = HERE / "web" / "static" / "subtitis.ico"

DEFAULT_HOST = "127.0.0.1"

#: Koliko portova iznad podešenog se proba kad je podešeni zauzet.
PORT_SCAN = 20

#: Koliko se čeka odgovor na `/api/identity`. Subtitis odgovara za par ms;
#: tuđa aplikacija koja ne odgovara ne sme da uspori pokretanje.
IDENTITY_TIMEOUT_SECONDS = 1.5

#: Koliko se čeka da server počne da sluša pre nego što se otvori pregledač.
STARTUP_TIMEOUT_SECONDS = 30.0


def port_in_use(host: str, port: int, timeout: float = 0.4) -> bool:
    """Da li neko već sluša na tom portu."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


_TITLE_RE = re.compile(rb"<title>[^<]*Subtitis[^<]*</title>", re.IGNORECASE)


def is_subtitis(host: str, port: int, timeout: float | None = None) -> bool:
    """Da li na tom portu odgovara Subtitis — ne bilo šta."""
    from .web.app import IDENTITY

    base = url_for(host, port)
    wait = timeout or IDENTITY_TIMEOUT_SECONDS
    try:
        with urllib.request.urlopen(f"{base}/api/identity", timeout=wait) as resp:
            data = json.loads(resp.read(4096))
        return isinstance(data, dict) and data.get("app") == IDENTITY
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            return False
    except (urllib.error.URLError, OSError, ValueError):
        return False
    # 404: nešto HTTP odgovara, ali bez `/api/identity`. To je i Subtitis
    # starije verzije (pre 2026-09-28) — njega NE SME proglasiti tuđim, inače
    # se digne druga instanca uz njega. Prepoznaje se po naslovu početne strane.
    try:
        with urllib.request.urlopen(f"{base}/", timeout=wait) as resp:
            return bool(_TITLE_RE.search(resp.read(16384)))
    except (urllib.error.URLError, OSError, ValueError):
        return False


@dataclass
class PortChoice:
    port: int
    #: Subtitis već sluša na `port` — ne dizati drugu instancu.
    running: bool
    #: Portovi preskočeni jer ih drži tuđa aplikacija.
    foreign: list[int] = field(default_factory=list)

    def line(self, host: str) -> str:
        url = url_for(host, self.port)
        if self.running:
            head = f"Subtitis već radi na {url}"
        else:
            head = f"Subtitis se diže na {url}"
        if self.foreign:
            taken = ", ".join(map(str, self.foreign))
            head += (
                f" — port {taken} drži druga aplikacija, pa je izabran "
                f"{self.port}. Za stalno: `port = …` u subtitis.toml."
            )
        return head


def choose_port(host: str, port: int, scan: int = PORT_SCAN) -> PortChoice:
    """Od podešenog porta naviše: prvi na kome odgovara Subtitis, ili prvi slobodan.

    Redosled je bitan. Ako je Subtitis prošli put zbog tuđe aplikacije sišao na
    8011, sledeće pokretanje mora da ga NAĐE tamo, a ne da digne drugi na 8012.
    """
    foreign: list[int] = []
    for p in range(port, port + scan + 1):
        if not port_in_use(host, p):
            return PortChoice(p, running=False, foreign=foreign)
        if is_subtitis(host, p):
            return PortChoice(p, running=True, foreign=foreign)
        foreign.append(p)
    from .config import SubtitisError

    raise SubtitisError(
        f"Portovi {port}–{port + scan} su svi zauzeti tuđim aplikacijama. "
        "Zadaj drugi: `port = …` u subtitis.toml ili --port."
    )


def url_for(host: str, port: int) -> str:
    # `0.0.0.0` je adresa za slušanje, ne za otvaranje u pregledaču.
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{shown}:{port}"


def wait_until_listening(host: str, port: int, timeout: float) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if port_in_use(host, port):
            return True
        time.sleep(0.2)
    return False


def _tray_image() -> Any:
    from PIL import Image

    return Image.open(ICON_PATH)


def run(
    host: str = DEFAULT_HOST,
    port: int | None = None,
    *,
    worker: bool = True,
    open_browser: bool = True,
    say: Any = print,
) -> int:
    """Pokreni (ili samo otvori) Subtitis. Vrati izlazni kod."""
    choice = choose_port(host, port if port is not None else cfg.port())
    port = choice.port
    say(choice.line(host))
    if choice.running:
        url = url_for(host, port)
        if open_browser:
            webbrowser.open(url)
        return 0

    import uvicorn

    from .web.app import WORKER_ENV

    import os

    os.environ[WORKER_ENV] = "1" if worker else "0"

    config = uvicorn.Config(
        "subtitis.web.app:app", host=host, port=port, log_level="warning"
    )
    server = uvicorn.Server(config)

    # Server u niti, traka u glavnoj: `pystray` na Windows-u traži glavnu nit
    # za petlju poruka.
    thread = threading.Thread(target=server.run, name="subtitis-server", daemon=True)
    thread.start()

    if not wait_until_listening(host, port, STARTUP_TIMEOUT_SECONDS):
        say("Server nije počeo da sluša — proveri log.")
        server.should_exit = True
        return 1

    url = url_for(host, port)
    say(f"Subtitis radi: {url}")
    if open_browser:
        webbrowser.open(url)

    try:
        import pystray
    except ImportError:
        say(
            "pystray nije instaliran — nema ikonice u traci.\n"
            "Server radi dok ovaj proces živi."
        )
        thread.join()
        return 0

    def on_open(icon: Any, item: Any) -> None:
        webbrowser.open(url)

    def on_quit(icon: Any, item: Any) -> None:
        server.should_exit = True
        icon.stop()

    icon = pystray.Icon(
        "subtitis",
        _tray_image(),
        f"Subtitis — {url}",
        menu=pystray.Menu(
            pystray.MenuItem("Otvori", on_open, default=True),
            pystray.MenuItem("Izađi", on_quit),
        ),
    )
    icon.run()          # blokira do „Izađi"
    thread.join(timeout=5.0)
    return 0
