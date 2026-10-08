"""Normalizacija audija preko ffmpeg-a: WAV 16 kHz mono PCM s16le."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from . import config as cfg
from .config import SubtitisError

SAMPLE_RATE = 16_000
CHANNELS = 1
CODEC = "pcm_s16le"

#: Mesta na kojima se ffmpeg traži kad ga nema u PATH-u.
#:
#: Ovo nije udobnost nego nužnost: VS Code (i svaki dugo otvoren program) drži
#: kopiju PATH-a od trenutka pokretanja, pa terminal otvoren iz njega ne vidi
#: ffmpeg instaliran u međuvremenu. Poruka „nije u PATH-u" je tu tačna a
#: beskorisna — program je instaliran.
KNOWN_FFMPEG_GLOBS: tuple[str, ...] = (
    r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\bin\ffmpeg.exe",
    r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe",
    r"%LOCALAPPDATA%\Microsoft\WindowsApps\ffmpeg.exe",
    r"%ProgramFiles%\ffmpeg\bin\ffmpeg.exe",
    r"%ProgramData%\chocolatey\bin\ffmpeg.exe",
    r"%USERPROFILE%\scoop\shims\ffmpeg.exe",
    r"C:\ffmpeg\bin\ffmpeg.exe",
)

#: Odakle je ffmpeg došao — `doctor` to prijavljuje.
ORIGIN_CONFIG = "konfiguracija"
ORIGIN_PATH = "PATH"
ORIGIN_KNOWN = "poznata lokacija"

_cache: tuple[str, str] | None = None


def _from_known_locations() -> str | None:
    if sys.platform != "win32":
        return None
    for pattern in KNOWN_FFMPEG_GLOBS:
        expanded = os.path.expandvars(pattern)
        if "%" in expanded:
            continue  # promenljiva ne postoji na ovoj mašini
        for hit in sorted(glob.glob(expanded, recursive=True), reverse=True):
            if os.path.isfile(hit):
                return hit
    return None


def find_ffmpeg(use_cache: bool = True) -> tuple[str | None, str]:
    """Vrati (putanja, poreklo). Redosled: konfiguracija -> PATH -> poznate lokacije."""
    global _cache
    if use_cache and _cache is not None:
        return _cache

    configured = cfg.setting("ffmpeg_path", env="SUBTITIS_FFMPEG")
    if configured:
        p = Path(str(configured)).expanduser()
        if p.is_file():
            _cache = (str(p), ORIGIN_CONFIG)
            return _cache
        raise SubtitisError(
            f"Podešeni ffmpeg ne postoji: {p}\n"
            f"  Ispravi `ffmpeg_path` u {cfg.CONFIG_FILE} ili SUBTITIS_FFMPEG."
        )

    exe = shutil.which("ffmpeg")
    if exe:
        _cache = (exe, ORIGIN_PATH)
        return _cache

    found = _from_known_locations()
    if found:
        _cache = (found, ORIGIN_KNOWN)
        return _cache
    return None, ""


def ffmpeg_path() -> str:
    """Vrati putanju do ffmpeg-a ili baci jasnu grešku."""
    exe, _ = find_ffmpeg()
    if exe:
        return exe
    raise SubtitisError(
        "ffmpeg nije pronađen — ni u konfiguraciji, ni u PATH-u, ni na poznatim mestima.\n"
        "  Instaliraj:  winget install Gyan.FFmpeg\n"
        "  Ako JESTE instaliran, a ovo i dalje vidiš: program iz kog si pokrenuo\n"
        "  terminal (npr. VS Code) drži staru kopiju PATH-a. Umesto restarta upiši\n"
        f"  putanju u {cfg.CONFIG_FILE}:\n"
        '      ffmpeg_path = "C:\\\\putanja\\\\do\\\\ffmpeg.exe"\n'
        "  ili postavi SUBTITIS_FFMPEG."
    )


def reset_ffmpeg_cache() -> None:
    global _cache
    _cache = None


def check_environment() -> None:
    """Provera na startu — zovi je pre svakog posla koji dira audio."""
    ffmpeg_path()


def duration_of_wav(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())


def normalize(src_path: str | Path, dst_path: str | Path) -> float:
    """Konvertuj bilo koji audio/video ulaz u WAV 16 kHz mono s16le.

    Vraća trajanje u sekundama.
    """
    src, dst = Path(src_path), Path(dst_path)
    if not src.exists():
        raise SubtitisError(f"Ulazna datoteka ne postoji: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(src),
        "-vn",
        "-ac", str(CHANNELS),
        "-ar", str(SAMPLE_RATE),
        "-acodec", CODEC,
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SubtitisError(
            f"ffmpeg nije uspeo (izlazni kod {proc.returncode}) na '{src.name}'.\n"
            f"Komanda: {' '.join(cmd)}\n"
            f"stderr:\n{(proc.stderr or '').strip()}"
        )
    if not dst.exists() or dst.stat().st_size == 0:
        raise SubtitisError(f"ffmpeg je prošao ali izlaz je prazan: {dst}")

    return duration_of_wav(dst)
