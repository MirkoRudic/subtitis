"""Serijalizacija u SRT.

UTF-8 **bez BOM-a**, CRLF završeci — Windows plejeri su izbirljivi.
Imenovanje po BCP-47: `intervju.sr-Latn.srt`, `intervju.sr-Cyrl.srt`, `intervju.en.srt`.
"""

from __future__ import annotations

import codecs
import re
from pathlib import Path

from .config import SubtitisError
from .models import Cue, SourceCue

CRLF = "\r\n"


def timestamp(seconds: float) -> str:
    """`HH:MM:SS,mmm`."""
    if seconds < 0:
        seconds = 0.0
    ms_total = int(round(seconds * 1000))
    ms = ms_total % 1000
    s = (ms_total // 1000) % 60
    m = (ms_total // 60_000) % 60
    h = ms_total // 3_600_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def render(cues: list[Cue]) -> str:
    blocks: list[str] = []
    for n, cue in enumerate(cues, start=1):
        body = CRLF.join(cue.lines)
        blocks.append(
            f"{n}{CRLF}{timestamp(cue.s)} --> {timestamp(cue.e)}{CRLF}{body}{CRLF}"
        )
    return CRLF.join(blocks)


def write_srt(path: Path, cues: list[Cue]) -> Path:
    from .config import write_text_atomic

    # newline="" => ne diramo CRLF koje smo sami upisali; encoding utf-8 (bez BOM-a).
    # Preko `.tmp` i zamene: SRT se prepisuje pri svakom prelamanju, a neko ga
    # baš tada ume da preuzima (krug 11).
    write_text_atomic(path, render(cues), newline="")
    return path


def bcp47(language: str, script: str | None = None) -> str:
    """`sr` + `Cyrl` -> `sr-Cyrl`. Pismo se navodi samo za srpski."""
    if language == "sr" and script:
        return f"sr-{script}"
    return language


def srt_name(stem: str, language: str, script: str | None = None) -> str:
    return f"{stem}.{bcp47(language, script)}.srt"


# ==========================================================================
# Čitanje postojećeg SRT-a
# ==========================================================================

#: `ime.sr.srt`, `ime.sr-Latn.srt`, `ime.fr.srt` — jezička oznaka pred `.srt`.
_LANG_TAG_RE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]{2,4})?$")

_TIME = r"(\d+):(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
#: Iza vremena sme da stoji bilo šta (`X1:… Y1:…` koordinate) — ignoriše se.
_TIMING_RE = re.compile(rf"^\s*{_TIME}\s*-->\s*{_TIME}")

#: Kad fajl nije UTF-8: u jednom od ovih je ručno pravljen srpski titl najčešće
#: sačuvan na Windows-u. Oba dekodiraju skoro svaki bajt, pa se ne bira po
#: tome koje „prođe" nego po tome koje daje slova srpskog pisma.
LEGACY_ENCODINGS = ("cp1250", "cp1251")

_SERBIAN_LETTERS = set(
    "abcdefghijklmnopqrstuvwxyzčćžšđ"
    "абвгдђежзијклљмнњопрстћуфхцчџш"
)


def srt_stem(filename: str) -> str:
    """`film.sr.srt` -> `film`; `film.srt` -> `film`. Da izlaz bude `film.fr.srt`."""
    stem = Path(filename).stem
    base, dot, tag = stem.rpartition(".")
    if dot and base and _LANG_TAG_RE.match(tag):
        return base
    return stem


def decode(data: bytes) -> tuple[str, str]:
    """Bajtovi -> (tekst, kodiranje). BOM se poštuje i odbacuje."""
    for bom, enc in (
        (codecs.BOM_UTF8, "utf-8-sig"),
        (codecs.BOM_UTF16_LE, "utf-16"),
        (codecs.BOM_UTF16_BE, "utf-16"),
    ):
        if data.startswith(bom):
            return data.decode(enc), enc
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass

    def score(text: str) -> float:
        letters = [ch.lower() for ch in text if ch.isalpha()]
        return sum(ch in _SERBIAN_LETTERS for ch in letters) / (len(letters) or 1)

    candidates = [
        (data.decode(enc, errors="replace"), enc) for enc in LEGACY_ENCODINGS
    ]
    return max(candidates, key=lambda c: score(c[0]))


def _ms(h: str, m: str, s: str, frac: str) -> int:
    # "1,5" je pola sekunde, ne 5 ms
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + int(frac.ljust(3, "0"))


def parse(text: str) -> list[SourceCue]:
    """Tekst SRT-a -> cue-ovi, redom kojim su u fajlu.

    Tolerantno: BOM (ako ga `decode` nije već skinuo), CRLF/LF/CR, cue sa
    proizvoljno redova, nedostajući prazan red na kraju ili između cue-ova,
    nedostajući redni broj. Oznake (<i>) ostaju u redovima netaknute.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    timing_at = [k for k, line in enumerate(lines) if _TIMING_RE.match(line)]
    if not timing_at:
        raise SubtitisError(
            "Fajl nema nijedan red sa vremenom (`00:00:01,000 --> 00:00:02,000`) "
            "— ovo nije SRT."
        )

    cues: list[SourceCue] = []
    for n, at in enumerate(timing_at):
        m = _TIMING_RE.match(lines[at])
        assert m is not None
        g = m.groups()
        end_of_block = timing_at[n + 1] if n + 1 < len(timing_at) else len(lines)
        body = [line.rstrip() for line in lines[at + 1 : end_of_block]]
        while body and not body[-1].strip():
            body.pop()
        # Redni broj sledećeg cue-a; bez praznog reda ispred njega završio bi
        # kao poslednji red teksta ovog cue-a.
        if n + 1 < len(timing_at) and body and body[-1].strip().isdigit():
            body.pop()
            while body and not body[-1].strip():
                body.pop()
        cues.append(
            SourceCue(
                start_ms=_ms(*g[:4]),
                end_ms=_ms(*g[4:]),
                lines=[line for line in body if line.strip()],
            )
        )
    return cues


def timing(cues: list[SourceCue] | list[Cue]) -> list[tuple[int, int]]:
    """[(početak, kraj)] u milisekundama, kako bi ih upisao `render`."""
    out: list[tuple[int, int]] = []
    for c in cues:
        if isinstance(c, SourceCue):
            out.append((c.start_ms, c.end_ms))
        else:
            out.append((int(round(max(c.s, 0.0) * 1000)), int(round(max(c.e, 0.0) * 1000))))
    return out
