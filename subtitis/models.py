"""Pydantic sheme za sve artefakte na disku.

`words.<lang>.json` je izvor istine. Sve ostalo je izvedeno i regenerabilno.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

#: `job.json` ima sopstvenu verziju. 2 = imena profila posle kruga 8
#: (`web` -> `tv`, `digital` -> `web`); prelaz radi `migrate.py`.
JOB_SCHEMA_VERSION = 2

Script = Literal["Latn", "Cyrl"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# words.<lang>.json
# --------------------------------------------------------------------------


class Word(BaseModel):
    i: int  # stabilan indeks; nikad se ne oslanjaj na poziciju u nizu
    s: float  # start, sekunde
    e: float  # end, sekunde
    w: str
    p: float  # verovatnoća ASR-a; < 0.6 => sumnjiva reč

    @property
    def dur(self) -> float:
        return max(0.0, self.e - self.s)


class AsrMeta(BaseModel):
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    detected_language: str | None = None
    detected_probability: float | None = None
    #: Brojači razloga odbacivanja/fallback-a pokupljeni iz DEBUG loga
    #: faster-whisper-a. NEPOTPUNI — vidi `asr.RejectionCapture`.
    rejections: dict[str, int] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)


class WordsFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    job_id: str
    source_file: str
    duration: float
    language: str
    script: Script = "Latn"
    asr: AsrMeta
    #: Naknadna obrada ijekavica -> ekavica: {applied: bool, changes: int}
    ekavica: dict[str, Any] = Field(default_factory=dict)
    words: list[Word] = Field(default_factory=list)


# --------------------------------------------------------------------------
# sentences.<lang>.json
# --------------------------------------------------------------------------


class CuePart(BaseModel):
    """Deo rečenice koji je u ulaznom SRT-u stajao u jednom cue-u.

    Postoji samo za poslove nastale iz SRT-a. Prevod ga nasleđuje od izvorne
    rečenice (`model_copy`), pa se prevedeni tekst raspoređuje nazad u ISTE
    cue-ove, srazmerno `share`.
    """

    cue: int  # pozicija cue-a u cues.json, od 0
    chars: int  # broj znakova ovog dela u originalu (bez oznaka)
    share: float  # udeo u rečenici, 0..1
    #: "" = bez kurziva, "all" = ceo deo u <i>, "partial" = samo deo — takav se
    #: posle prevoda ne može vratiti na isto mesto i prijavljuje se.
    italic: Literal["", "all", "partial"] = ""
    #: U originalu iza ovog dela ide novi red, a cue se nastavlja sledećom
    #: rečenicom (dijalog "- Da.\n- Ne."). Taj prelom se zadržava.
    line_break: bool = False


class Sentence(BaseModel):
    id: int
    s: float
    e: float
    text: str
    #: INKLUZIVNI raspon indeksa u words.json. Posao iz SRT-a nema reči, pa je
    #: tu (0, -1) — prazan raspon.
    words: tuple[int, int]
    speaker: str | None = None
    flags: list[str] = Field(default_factory=list)
    cue_parts: list[CuePart] = Field(default_factory=list)


class SentencesFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    job_id: str
    language: str
    script: Script = "Latn"
    derived_from: str
    # True => pajplajn ovaj fajl NIKAD ne sme automatski da prepiše
    edited_by_human: bool = False
    sentences: list[Sentence] = Field(default_factory=list)


# --------------------------------------------------------------------------
# cues.json — ulazni SRT (samo za posao koji prevodi postojeći titl)
# --------------------------------------------------------------------------


class SourceCue(BaseModel):
    #: Vremena u milisekundama, tačno kako su pročitana. Izlaz ih prepisuje
    #: bez ikakve obrade — to je jedina tvrda garancija ovog režima.
    start_ms: int
    end_ms: int
    #: Redovi kako su bili u fajlu, sa oznakama (<i>), bez vodećeg `{\\an8}`.
    lines: list[str] = Field(default_factory=list)
    #: Vodeći ASS blok (`{\\an8}`) — pozicija važi za ceo cue i vraća se ispred.
    prefix: str = ""


class CuesFile(BaseModel):
    schema_version: int = SCHEMA_VERSION
    job_id: str
    source_file: str
    language: str
    #: Pismo u kom je SRT stigao; rečenice se svejedno drže u latinici.
    script: str = "Latn"
    encoding: str = "utf-8"
    cues: list[SourceCue] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Profil prelamanja
# --------------------------------------------------------------------------


class Profile(BaseModel):
    name: str
    #: Čemu profil služi, jednom rečenicom. Stoji uz brojeve u dashboardu —
    #: `42 znaka` korisniku ne govori kad da ga izabere, a ovo govori.
    note: str = ""
    max_chars_per_line: int = 42
    #: Meka donja granica dužine reda (`linebreak.PENALTY_BELOW_MIN_CHARS`).
    #: 0 = bez nje; uvedena za `digital`, gde red od tri znaka trepne.
    min_chars_per_line: int = 0
    max_lines: int = 2
    max_cps: float = 20.0
    min_duration: float = 0.833
    max_duration: float = 7.0
    min_gap: float = 0.083
    #: Podrazumevani izlaz koji izbor profila POSTAVLJA (krug 8) — korisnik ih
    #: menja ručno; profil ih ne zaključava. Primenjuju se pri pisanju SRT-a.
    case: str = "normal"
    drop_punctuation: bool = False


# --------------------------------------------------------------------------
# Cue (rezultat prelamanja; ne perzistira se, ide pravo u SRT)
# --------------------------------------------------------------------------


class Cue(BaseModel):
    index: int = 0
    s: float
    e: float
    lines: list[str]
    sentence_id: int | None = None
    flags: list[str] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def cps(self) -> float:
        d = self.e - self.s
        if d <= 0:
            return float("inf")
        return len(self.text.replace("\n", " ")) / d


# --------------------------------------------------------------------------
# job.json
# --------------------------------------------------------------------------


class PhaseRecord(BaseModel):
    phase: str
    status: Literal["ok", "error"]
    started_at: str
    finished_at: str
    duration_s: float
    detail: dict[str, Any] = Field(default_factory=dict)


class Job(BaseModel):
    schema_version: int = JOB_SCHEMA_VERSION
    job_id: str
    source_file: str
    status: str = "created"
    language: str | None = None
    duration: float | None = None
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    history: list[PhaseRecord] = Field(default_factory=list)
    #: Verzija, krug i commit koda koji je napravio posao (krug 25).
    made_with: dict[str, Any] = Field(default_factory=dict)
