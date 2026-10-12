"""Pauze u govoru, iz zvuka — signal za prelom natpisa (krug 19).

Razmak između reči u `words.json` za to ne služi: Whisper bez VAD-a ne
ostavlja tišinu između reči nego je upija u trajanje susedne reči, pa je na
granici bez interpunkcije razmak tačno 0 u 825 od 830 slučajeva
(`merenja/dnevnik.md`, 2026-10-12). Pauze se zato čitaju iz samog zvuka,
Silero VAD-om — istim koji `asr.speech_measure` koristi za probu.

Tišine se računaju JEDNOM po poslu i čuvaju u `jobs/<id>/pauses.json`:

* fajl je mali i preživljava brisanje `source.wav` po roku
  (`config.purge_old_sources` briše samo zvuk);
* pravi se pri prepoznavanju govora, a za stariji posao lenjo, pri prvom
  prelamanju dok zvuk još postoji — bez migracije koja bi ponovo čitala zvuk
  svih poslova;
* posao bez fajla i bez zvuka se prelama kao ranije, bez pauza, i to piše u
  log posla — ne pada;
* fajl nosi čime je napravljen (`made_with`): bez toga se ne bi znalo da li
  se stare tišine smeju porediti sa novim.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from . import config as cfg
from .models import Word

PAUSES_FILE = "pauses.json"
SCHEMA_VERSION = 1

#: Parametri Silero VAD-a za tišine. `min_silence_duration_ms` je namerno
#: nizak: upisuje se sve od 0,10 s naviše, a prag koji tišinu čini pauzom
#: (`linebreak.PAUSE_MIN`) bira se pri prelamanju — promena praga ne traži
#: ponovno čitanje zvuka. `speech_pad_ms` 0: tišina se ne skraćuje.
VAD_THRESHOLD = 0.5
VAD_MIN_SILENCE_MS = 100
VAD_SPEECH_PAD_MS = 0
SAMPLE_RATE = 16000

#: Tišina se pripisuje granici dve reči koja leži u njoj ili joj je najbliža,
#: ali ne dalje od ovoliko sekundi. Whisper ume da razvuče reč preko tišine
#: (`najvažnija` od 1,1 s sa tišinom u sredini): tada granice blizu nema i
#: tišina se ne pripisuje nikome, umesto da se pripiše pogrešnoj granici.
SNAP_MAX = 0.25


class PausesFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    made_with: dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0
    #: tišine [početak, kraj] u sekundama, rastuće
    silences: list[tuple[float, float]] = Field(default_factory=list)


def _model_id() -> dict[str, Any]:
    import faster_whisper
    from faster_whisper.utils import get_assets_path

    name = "silero_vad_v6.onnx"
    path = Path(get_assets_path()) / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None
    return {"faster_whisper": faster_whisper.__version__, "model": name, "model_sha256_16": digest}


def detect(wav: Path) -> PausesFile:
    """Tišine u `wav` (16 kHz mono, kao `source.wav`)."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = dict(
        threshold=VAD_THRESHOLD, min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
    )
    audio = decode_audio(str(wav), sampling_rate=SAMPLE_RATE)
    spans = get_speech_timestamps(audio, VadOptions(**opts))
    silences = []
    for a, b in zip(spans, spans[1:]):
        s, e = a["end"] / SAMPLE_RATE, b["start"] / SAMPLE_RATE
        if e > s:
            silences.append((round(s, 3), round(e, 3)))
    return PausesFile(
        made_with={"vad": "silero", **_model_id(), **opts, "sample_rate": SAMPLE_RATE},
        duration=round(len(audio) / SAMPLE_RATE, 3),
        silences=silences,
    )


def path(job_id: str) -> Path:
    return cfg.job_dir(job_id) / PAUSES_FILE


def load(job_id: str) -> PausesFile | None:
    p = path(job_id)
    if not p.exists():
        return None
    return PausesFile.model_validate(cfg.read_json(p))


def ensure(job_id: str, log: Callable[[str], None] | None = None) -> PausesFile | None:
    """Tišine posla: iz `pauses.json`, ili iz `source.wav` (pa se upišu).

    Bez oba vraća None i to javlja kroz `log`. Nikad ne obara posao: greška
    pri čitanju zvuka znači prelamanje bez pauza, kao do sada.
    """
    say = log or (lambda m: None)
    try:
        have = load(job_id)
    except Exception as e:  # noqa: BLE001 — oštećen fajl nije razlog za pad
        say(f"pauze: {PAUSES_FILE} se ne može pročitati ({e}) — prelamanje bez pauza.")
        return None
    if have is not None:
        return have
    wav = cfg.job_dir(job_id) / "source.wav"
    if not wav.exists():
        say(f"pauze: nema ni {PAUSES_FILE} ni source.wav — prelamanje bez pauza, kao pre kruga 19.")
        return None
    try:
        made = detect(wav)
    except Exception as e:  # noqa: BLE001
        say(f"pauze: tišine se ne mogu izmeriti ({e}) — prelamanje bez pauza.")
        return None
    cfg.write_json(path(job_id), made.model_dump())
    from .untranslated import plural

    n = len(made.silences)
    say(f"pauze: {n} {plural(n, 'tišina', 'tišine', 'tišina')} iz source.wav u {PAUSES_FILE}.")
    return made


def boundary_pauses(words: list[Word], silences: list[tuple[float, float]]) -> list[float]:
    """Pauza posle tokena k (sekunde, 0 ako je nema), za svaki token.

    Granica je između dve uzastopne reči koje nisu interpunkcija; pauza se
    upisuje na poslednji token pre sledeće reči (zarez, ako ga ima — rez
    ispred interpunkcije ionako nije dozvoljen). Ivice rečenice učestvuju u
    izboru najbliže granice, ali se tamo ništa ne upisuje: tišina pre ili
    posle rečenice ne sme da se prilepi za njenu prvu ili poslednju granicu.
    """
    from .linebreak import _is_punct

    out = [0.0] * len(words)
    real = [k for k, w in enumerate(words) if not _is_punct(w.w)]
    if len(real) < 2 or not silences:
        return out
    # (vreme granice, indeks tokena za upis ili None za ivicu rečenice)
    bounds: list[tuple[float, int | None]] = [(words[real[0]].s, None)]
    bounds += [(words[b].s, b - 1) for b in real[1:]]
    bounds.append((words[real[-1]].e, None))
    lo, hi = bounds[0][0] - SNAP_MAX, bounds[-1][0] + SNAP_MAX
    for s, e in silences:
        if e < lo or s > hi:
            continue
        mid = (s + e) / 2

        def dist(t: float) -> tuple[float, float]:
            d = 0.0 if s <= t <= e else min(abs(t - s), abs(t - e))
            return d, abs(t - mid)

        t, k = min(bounds, key=lambda b: dist(b[0]))
        if k is None or dist(t)[0] > SNAP_MAX:
            continue
        out[k] = max(out[k], round(e - s, 3))
    return out
