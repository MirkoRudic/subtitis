"""Preživljavanje bez konzole i trag o padu pri pokretanju.

Prečica na desktopu pokreće `pythonw.exe`, a taj interpreter **nema standardni
izlaz uopšte**: `sys.stdout` i `sys.stderr` su `None`, ne samo pogrešno kodirani.
Svaki `print`, svaki `typer.echo` i — što je ovde stvarno oborilo pokretanje —
`uvicorn`-ov formater loga (`sys.stdout.isatty()`) tada baci `AttributeError`.
Kako izlaz dira i tuđi kod u putanji pokretanja, popravka ne može da bude po
pozivnom mestu; struje se podmeću jednom, na ulaznoj tački.

Zamena nije prazna kanta nego datoteka `jobs/console.log`: pod `pythonw` je to
jedini način da se vidi šta je server ispisao.

Drugi deo je isti problem u najgorem obliku: kad proces padne pod `pythonw`,
nema ni traceback-a ni prozora — ništa. Zato svaki neuhvaćen izuzetak, i iz
glavne niti i iz radnih niti, ide u `jobs/startup-error.txt` sa punim ispisom.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Iterator, TextIO

ROOT_DIR = Path(__file__).resolve().parent.parent

CRASH_FILE_NAME = "startup-error.txt"
CONSOLE_LOG_NAME = "console.log"

#: Preko ovoga se `console.log` seče na početku sesije — inače raste zauvek.
CONSOLE_LOG_MAX_BYTES = 1_000_000


def jobs_dir() -> Path:
    """Isto kao `config.JOBS_DIR`, ali bez uvoza `config`.

    Ovaj modul mora da radi i kad uvoz ostatka paketa pukne — upravo tada je
    zapis o grešci i najpotrebniji.
    """
    return Path(os.environ.get("SUBTITIS_JOBS_DIR", ROOT_DIR / "jobs"))


class _NullStream(io.TextIOBase):
    """Rezerva kad ni datoteka ne može da se otvori. Guta sve, ne baca ništa."""

    encoding = "utf-8"
    errors = "replace"

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        return len(s)

    def isatty(self) -> bool:
        return False


def _open_console_log() -> TextIO:
    try:
        d = jobs_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / CONSOLE_LOG_NAME
        # `errors="replace"`: kvačica sme da se izgubi, ispis ne sme da obori rad.
        mode = "a"
        if path.exists() and path.stat().st_size > CONSOLE_LOG_MAX_BYTES:
            mode = "w"
        fh = open(path, mode, encoding="utf-8", errors="replace", buffering=1)
        fh.write(
            f"\n=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"{' '.join(sys.argv)} ===\n"
        )
        return fh
    except OSError:  # pragma: no cover — nema prava pisanja
        return _NullStream()


def ensure_streams() -> bool:
    """Podmetni `sys.stdout`/`sys.stderr` ako ih nema. Vrati da li je menjano.

    Menjaju se i `sys.__stdout__`/`sys.__stderr__`: `logging` i `click` umeju da
    posegnu za njima, a pod `pythonw` su `None` isto kao i obični.
    """
    missing = [n for n in ("stdout", "stderr", "__stdout__", "__stderr__")
               if getattr(sys, n, None) is None]
    if not missing:
        return False
    stream = _open_console_log()
    for name in missing:
        setattr(sys, name, stream)
    return True


def write_crash(
    exc_type: type[BaseException],
    exc: BaseException,
    tb: TracebackType | None,
    *,
    context: str = "",
) -> Path | None:
    """Upiši pun traceback u `jobs/startup-error.txt`. Nikad ne baca.

    Isti izuzetak se ne upisuje dvaput: kroz njega prolaze i `crash_guard` u
    `cli.main` i onaj u `__main__`.
    """
    if getattr(exc, "_subtitis_zapisan", False):
        return None
    try:
        exc._subtitis_zapisan = True  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — retki izuzeci bez __dict__
        pass
    try:
        d = jobs_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / CRASH_FILE_NAME
        head = (
            f"=== {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            f"komanda    : {' '.join(sys.argv)}\n"
            f"interpreter: {sys.executable}\n"
            f"radni dir  : {os.getcwd()}\n"
            + (f"gde        : {context}\n" if context else "")
        )
        body = "".join(traceback.format_exception(exc_type, exc, tb))
        with path.open("a", encoding="utf-8", errors="replace", newline="\n") as fh:
            fh.write(head + body + "\n")
        return path
    except Exception:  # noqa: BLE001 — zapis o padu ne sme da izazove novi pad
        return None


_crash_log_installed = False


def install_crash_log() -> None:
    """Preusmeri neuhvaćene izuzetke u datoteku, pa pusti dalje kako je i bilo.

    Idempotentno: ulaznih tačaka ima dve (`python -m subtitis` i konzolna skripta
    `subtitis`), a dvostruko okačena kuka bi svaki pad upisala dvaput.
    """
    global _crash_log_installed
    if _crash_log_installed:
        return
    _crash_log_installed = True

    previous_hook = sys.excepthook

    def hook(
        exc_type: type[BaseException],
        exc: BaseException,
        tb: TracebackType | None,
    ) -> None:
        # `SystemExit` je uredan izlaz (`typer.Exit`), ne pad.
        if not issubclass(exc_type, SystemExit):
            write_crash(exc_type, exc, tb)
        previous_hook(exc_type, exc, tb)

    sys.excepthook = hook

    previous_thread_hook = threading.excepthook

    def thread_hook(args: Any) -> None:
        if args.exc_type is not None and not issubclass(args.exc_type, SystemExit):
            write_crash(
                args.exc_type, args.exc_value, args.exc_traceback,
                context=f"nit {getattr(args.thread, 'name', '?')}",
            )
        previous_thread_hook(args)

    threading.excepthook = thread_hook


@contextmanager
def crash_guard(context: str = "") -> Iterator[None]:
    """Zapiši svaki pad iz obuhvaćenog koda, pa ga pusti dalje nepromenjenog.

    `sys.excepthook` nije dovoljan: `typer` uhvati izuzetak, ispiše svoj šareni
    traceback i tek onda ga baci dalje — a do kuke se u tom slučaju ne stiže
    ovako kako očekujemo. Zato se ulazna tačka obavija i ovim.
    """
    try:
        yield
    except SystemExit:  # uredan izlaz (`typer.Exit`), ne pad
        raise
    except BaseException as exc:
        write_crash(type(exc), exc, exc.__traceback__, context=context)
        raise


def bootstrap() -> None:
    """Sve što mora da se odradi pre prve tuđe linije koda."""
    ensure_streams()
    install_crash_log()
