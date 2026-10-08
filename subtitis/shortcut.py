"""Prečica na desktopu koja pokreće Subtitis bez vidljivog prozora.

Prečica cilja `pythonw.exe` iz `.venv` — to je isti interpreter kao `python`,
samo bez konzole. Nije potreban ni VBS omotač ni `cmd`.

Ako `pythonw.exe` ne postoji (retko, npr. venv napravljen sa `--without-pip` na
neuobičajenoj instalaciji), pada nazad na `python.exe` uz `WindowStyle = 7`
(minimizovan), pa prozor bar ne smeta.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import ROOT_DIR, SubtitisError

ICON = ROOT_DIR / "subtitis" / "web" / "static" / "subtitis.ico"


def python_launcher() -> tuple[Path, int]:
    """(interpreter, WindowStyle). `pythonw` ne otvara konzolu."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    if pythonw.exists():
        return pythonw, 1
    return exe, 7  # minimizovan — bolje nego pun prozor


#: `FOLDERID_Desktop` iz `KnownFolders.h`.
_FOLDERID_DESKTOP = "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}"


def _desktop_from_windows() -> Path | None:
    """Pravi Desktop preko Windows API-ja.

    Ime foldera NIJE „Desktop" na lokalizovanom Windows-u — na srpskom je
    „Radna površina" — a uz OneDrive nije ni pod `%USERPROFILE%`. Jedini tačan
    izvor je `SHGetKnownFolderPath`; sve ostalo je pogađanje.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_byte * 8),
        ]

    guid = GUID()
    ole32 = ctypes.OleDLL("ole32")
    shell32 = ctypes.OleDLL("shell32")
    try:
        ole32.CLSIDFromString(_FOLDERID_DESKTOP, ctypes.byref(guid))
        out = ctypes.c_wchar_p()
        shell32.SHGetKnownFolderPath(
            ctypes.byref(guid), 0, None, ctypes.byref(out)
        )
        path = out.value
        ole32.CoTaskMemFree(out)
    except OSError:  # pragma: no cover — API nedostupan
        return None
    return Path(path) if path else None


def desktop_dir() -> Path:
    import os

    found = _desktop_from_windows()
    if found and found.exists():
        return found

    # Rezerve, redom po verovatnoći. `Radna površina` je tu jer je ovo alatka
    # za srpski Windows; OneDrive jer Windows 11 tamo preusmerava Desktop.
    candidates: list[Path] = []
    for root in (os.environ.get("OneDrive"), os.environ.get("USERPROFILE"),
                 os.environ.get("HOME")):
        if not root:
            continue
        candidates += [Path(root) / name for name in ("Desktop", "Radna površina")]
    for d in candidates:
        if d.exists():
            return d

    raise SubtitisError(
        "Ne mogu da nađem Desktop.\n"
        "Zadaj putanju: subtitis shortcut --dir <putanja>"
    )


def create_shortcut(name: str = "Subtitis", target_dir: Path | None = None) -> Path:
    """Napravi `.lnk` i vrati putanju. Samo Windows."""
    if sys.platform != "win32":
        raise SubtitisError("Prečice na desktopu prave se samo na Windows-u.")

    folder = Path(target_dir) if target_dir else desktop_dir()
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / f"{name}.lnk"

    interpreter, window_style = python_launcher()
    if not ICON.exists():
        raise SubtitisError(
            f"Nema ikonice {ICON}. Napravi je: python scripts/make_icon.py"
        )

    # `win32com` nije zavisnost projekta, pa se `.lnk` pravi preko PowerShell-a
    # i WScript.Shell-a — dostupno je na svakom Windows-u.
    import subprocess

    script = f"""
$s = (New-Object -ComObject WScript.Shell).CreateShortcut('{link}')
$s.TargetPath = '{interpreter}'
$s.Arguments = '-m subtitis desktop'
$s.WorkingDirectory = '{ROOT_DIR}'
$s.IconLocation = '{ICON}'
$s.WindowStyle = {window_style}
$s.Description = 'Subtitis — SRT titlovi iz audija'
$s.Save()
"""
    proc = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not link.exists():
        raise SubtitisError(
            "Prečica nije napravljena.\n"
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return link
