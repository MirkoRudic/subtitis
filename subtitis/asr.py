"""ASR preko faster-whisper.

Za srpski `language` se UVEK prosleđuje eksplicitno. Auto-detekcija ga meša sa
hrvatskim i vraća ijekavicu.

Whisper za srpski vraća čas ćirilicu, čas latinicu, ponekad pomešano u istom
fajlu — sve se normalizuje u latinicu pre upisa. Interno radimo samo latinicom.

Model se posle transkripcije obavezno oslobađa: ASR i LLM nikad ne stoje u
VRAM-u istovremeno (large-v3 ~5 GB, 14B Q4 ~11 GB — ne staju u 16 GB).
"""

from __future__ import annotations

import gc
import logging
import os
import site
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import SAMPLE_RATE
from .config import SubtitisError
from .models import AsrMeta, Word, WordsFile
from .segment import split_token
from .translit import detect_script, to_latin

# --- Konstante za štelovanje ----------------------------------------------

DEFAULT_MODEL = "large-v3"
DEFAULT_BEAM_SIZE = 5

#: Naziv kojim se traži auto-detekcija jezika umesto zadatog koda.
#: Za srpski i dalje treba zadati `sr`: detekcija ga meša sa hrvatskim i vraća
#: ijekavicu. Zato je ovo izbor, ne podrazumevano ponašanje.
AUTO_LANGUAGE = "auto"
DEFAULT_COMPUTE_TYPE = "float16"
DEFAULT_DEVICE = "cuda"

# --- Zaštita od petlji ponavljanja i halucinacija na tišini -----------------
#
# large-v3 na dužim snimcima ume da uđe u petlju i da halucinira tekst preko
# tišine. Glavni krivac je `condition_on_previous_text=True` (Whisperov
# podrazumevani izbor): model svoj izlaz koristi kao kontekst, pa kad jednom
# sklizne, ostaje zaglavljen do kraja fajla.

#: Ne hrani model sopstvenim izlazom. Ključno za duge fajlove.
CONDITION_ON_PREVIOUS_TEXT = False

#: Iznad ovog odnosa kompresije tekst je skoro sigurno ponavljanje -> ponovi
#: segment sa višom temperaturom.
COMPRESSION_RATIO_THRESHOLD = 2.4

#: Ispod ove prosečne log-verovatnoće segment se smatra neuspelim.
LOG_PROB_THRESHOLD = -1.0

#: Iznad ove verovatnoće "nema govora" segment se odbacuje kao tišina.
NO_SPEECH_THRESHOLD = 0.6

#: Lestvica temperatura; koristi se redom kad segment padne na gornjim pragovima.
TEMPERATURE_FALLBACK = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

#: ISKLJUČENO NAMERNO. Ne uključuj bez merenja pokrivenosti.
#:
#: Filter je 2026-08-11 uhvaćen kako guta ceo rep snimka: pokrivenost 55.9%,
#: rečenice staju na 79.74s od 90s, nestao ceo poslednji VAD odsečak. Sa
#: `None`: pokrivenost 83.8%, reči 155 -> 173, rep se vratio.
#:
#: Uzrok je u `faster_whisper/transcribe.py`, grana `if silence_before and
#: silence_after:` — radi `current_segments[si:] = []`, a kad je sporni segment
#: blizu kraja sadržaja i `seek = content_frames`, tj. preskok DO KRAJA FAJLA.
#: Segment se proglasi sumnjivim ako mu prvih 8 reči skupe skor >= 3 ili
#: skor + 0.01 >= broj reči (reč sa verovatnoćom < 0.15, kraća od 133 ms ili
#: duža od 2 s). Kratak i tih odgovor to ispuni bez muke. Ta grana NE LOGUJE
#: NIŠTA, pa se gubitak ne vidi ni u `asr.rejections`.
#:
#: Protiv petlji ponavljanja glavna odbrana je `CONDITION_ON_PREVIOUS_TEXT =
#: False`, uz `COMPRESSION_RATIO_THRESHOLD` i `TEMPERATURE_FALLBACK`. Ovaj
#: filter nam nije potreban za to.
HALLUCINATION_SILENCE_THRESHOLD: float | None = None

#: Vrednost koja se koristi kad filter neko IZRIČITO uključi
#: (`subtitis transcribe --hallucination-filter`). Whisperov podrazumevani izbor.
HALLUCINATION_SILENCE_SECONDS = 2.0

# --- VAD --------------------------------------------------------------------
#
# Silero VAD sa podrazumevanim `threshold=0.5` jede tiši govor — na testnom
# snimku je pojeo ~17 s od 90, uključujući ceo odgovor deteta. Sve što VAD
# odbaci nikad ne stigne do modela, pa se gubitak ne vidi ni u jednoj drugoj
# metrici osim pokrivenosti (`scripts/bench.py`).

#: Prag verovatnoće govora. Niže = manje agresivno, više zadržanog tišeg govora.
VAD_THRESHOLD = 0.35

#: Koliko tišine mora da prođe pre nego što se govorni odsečak zatvori (ms).
#: Uz `VAD_SPEECH_PAD_MS=400` sa svake strane, pauze do ~800 ms ne gube ništa.
VAD_MIN_SILENCE_DURATION_MS = 700

#: Svaki zadržani odsečak se produži za ovoliko sa obe strane (ms).
VAD_SPEECH_PAD_MS = 400

#: Imenovani izbori praga, za dashboard. Broj 0.35 korisniku ne znači ništa —
#: niko ne zna da li je dobar dok ne pusti ceo prepis i ne prebroji šta fali.
#: Naziv je odluka koja se može doneti unapred, pre nego što se čuje rezultat.
#: `None` znači da se VAD ne pušta uopšte.
#:
#: IZMERENO 2026-08-19 na tihom snimku bez bubice (sr, 94 s), da se ne meri
#: ponovo. Silero na istom fajlu, direktno:
#:
#:     prag 0.50   17 odsečaka   govor 80.13 s   85.2%
#:     prag 0.35   13 odsečaka   govor 81.98 s   87.1%   <- „normalno"
#:     prag 0.20   10 odsečaka   govor 86.77 s   92.2%   <- „tih"
#:     prag 0.10    4 odsečka    govor 90.19 s   95.9%
#:
#: Prag STIŽE do detekcije i menja rezultat — provereno i kroz ceo ASR na istom
#: fajlu: 0.35 daje 148 reči, 0.20 daje 151 i drugačiji tekst („barbecue sos"
#: naspram „barbekju sos", „kečapa" naspram „kećapa"). Ako dva prolaza daju
#: identičan prepis do poslednje reči, nije prag bez dejstva nego su prolazi
#: išli sa istim pragom; zato se prag od 2026-08-19 upisuje u `job.json`
#: (`vad_threshold` u detaljima faze `asr`) i ispisuje posle transkripcije.
VAD_PRESETS: dict[str, float | None] = {
    "normalno": VAD_THRESHOLD,
    "tih": 0.20,
    "iskljuceno": None,
}

#: PODRAZUMEVANO ISKLJUČENO od 2026-08-19.
#:
#: Tri prolaza nad istim tihim snimkom (sr, 94 s, bez bubice), čisto stanje:
#:
#:     prag 0.35   179 reči   pokrivenost 85.6%   označenih 2/11
#:     prag 0.20   180 reči   pokrivenost 87.9%   označenih 3/8
#:     isključeno  189 reči   pokrivenost 88.9%   označenih 1/10
#:
#: Isključeno je bolje po svakoj meri, i tekstualno osetno tačnije: „Delikates
#: burger od 200 grama" naspram „deli kates burdžer 200 grama", „bez smokija"
#: naspram „gre smokin", „barbecue" naspram „BBQ", i dve odvojene rečenice sa
#: znakom pitanja umesto spojenog fragmenta.
#:
#: Protiv halucinacija na tišini ostaju `CONDITION_ON_PREVIOUS_TEXT = False`,
#: `COMPRESSION_RATIO_THRESHOLD` i `NO_SPEECH_THRESHOLD` — iste one odbrane
#: zbog kojih je `HALLUCINATION_SILENCE_THRESHOLD` već isključen.
#:
#: Šta OVO merenje NE pokriva: svi mereni snimci su gust govor (VAD bi odbacio
#: 4.8-15.0%). Snimak sa dugom tišinom — pauza u intervjuu, predavanje sa
#: prekidima — nije isproban ni jednom, a to je upravo slučaj u kom VAD zarađuje
#: svoj hleb. Zato `vad_advice()` pri isključenom VAD-u ispisuje koliko snimka
#: nije govor: kad takav snimak jednom naiđe, brojka će biti tu i pre nego što
#: iko primeti halucinaciju.
DEFAULT_VAD_PRESET = "iskljuceno"


def vad_preset(name: str | None) -> tuple[bool, float | None]:
    """Naziv izbora -> (da li se VAD pušta, prag). Nepoznat naziv je podrazumevani."""
    key = (name or "").strip().lower()
    if key not in VAD_PRESETS:
        key = DEFAULT_VAD_PRESET
    threshold = VAD_PRESETS[key]
    return threshold is not None, threshold


# --- Hvatanje razloga odbacivanja -------------------------------------------
#
# faster-whisper NE izlaže odbačene segmente ni kroz povratnu vrednost ni kroz
# `Segment` objekte — samo ih loguje na DEBUG, na loggeru "faster_whisper".
# Ovde ih presrećemo.
#
# VAŽNO, i to je rupa u pokrivenosti: filter halucinacija
# (`hallucination_silence_threshold`) odbacuje segmente POTPUNO TIHO, bez
# ijednog log poziva — vidi `transcribe.py`, granu `if silence_before and
# silence_after:` koja radi `current_segments[si:] = []` i, kad je segment blizu
# kraja sadržaja, `seek = content_frames` (preskok do kraja fajla).
# Taj gubitak se ne može prebrojati; jedini način da se utvrdi je poređenje
# prolaza sa filterom i bez njega (`scripts/asr_probe.py --compare`).

LOGGER_NAME = "faster_whisper"

#: (ključ, obrazac, da li stvarno odbacuje audio)
REJECTION_PATTERNS: tuple[tuple[str, str, bool], ...] = (
    # Ceo prozor od 30 s se preskače: `seek += segment_size; continue`.
    ("no_speech_dropped", "No speech threshold is met", True),
    # Ovi samo pokreću ponovni pokušaj na višoj temperaturi.
    ("compression_ratio_fallback", "Compression ratio threshold is not met", False),
    ("log_prob_fallback", "Log probability threshold is not met", False),
)

#: Ključevi koji znače stvarno odbačen audio (a ne retry).
DROP_KEYS = tuple(key for key, _, drops in REJECTION_PATTERNS if drops)


class RejectionCapture(logging.Handler):
    """Prebroj razloge odbacivanja iz DEBUG loga faster-whisper-a.

    Koristi se kao kontekst menadžer. `counts` su brojači po ključu iz
    `REJECTION_PATTERNS`, `messages` su sirove poruke redom kako su stigle.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.counts: dict[str, int] = {}
        self.messages: list[str] = []
        self._logger = logging.getLogger(LOGGER_NAME)
        self._prev_level: int | None = None
        self._prev_propagate: bool | None = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — logovanje ne sme da obori posao
            return
        for key, pattern, _ in REJECTION_PATTERNS:
            if pattern in msg:
                self.counts[key] = self.counts.get(key, 0) + 1
                self.messages.append(msg)
                return

    def __enter__(self) -> RejectionCapture:
        self._prev_level = self._logger.level
        self._prev_propagate = self._logger.propagate
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False  # ne zatrpavaj korisnikov izlaz
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._logger.removeHandler(self)
        if self._prev_level is not None:
            self._logger.setLevel(self._prev_level)
        if self._prev_propagate is not None:
            self._logger.propagate = self._prev_propagate

    @property
    def dropped(self) -> int:
        """Broj prozora koje je Whisper stvarno odbacio (bez retry-ja)."""
        return sum(self.counts.get(k, 0) for k in DROP_KEYS)


def vad_parameters(threshold: float | None = None) -> dict[str, Any]:
    """Parametri za Silero VAD; prosleđuju se kao `vad_parameters` dict."""
    return {
        "threshold": VAD_THRESHOLD if threshold is None else float(threshold),
        "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
        "speech_pad_ms": VAD_SPEECH_PAD_MS,
    }


@dataclass
class SpeechMeasure:
    """Koliko zvuka Silero smatra govorom, mereno na celom fajlu."""

    duration: float = 0.0
    speech: float = 0.0
    segments: int = 0
    threshold: float = VAD_THRESHOLD

    @property
    def kept_share(self) -> float:
        return self.speech / self.duration if self.duration else 0.0

    @property
    def discarded_share(self) -> float:
        """Udeo snimka koji bi VAD bacio pre nego što model išta vidi."""
        return 1.0 - self.kept_share if self.duration else 0.0

    def line(self) -> str:
        return (
            f"VAD bi na pragu {self.threshold:.2f} zadržao "
            f"{100 * self.kept_share:.1f}% snimka u {self.segments} odsečaka "
            f"(odbacio {100 * self.discarded_share:.1f}%)"
        )


def measure_speech(
    wav_path: str | Path, threshold: float | None = None
) -> SpeechMeasure:
    """Pusti Silero nad celim fajlom i izmeri koliko bi zadržao.

    Meri se UVEK, i kad je VAD isključen — to je jedini način da se zna šta bi
    VAD uradio, a da se ne pušta ceo ASR dvaput. Cena je zanemarljiva: izmereno
    0.15 s na snimak od 94 s i 0.39 s na 181 s, naspram 34-50 s koliko traje
    sam ASR.

    Ova mera odgovara na pitanje na koje pokrivenost NE odgovara. Pokrivenost je
    ograničena time koliko snimak uopšte sadrži govora, pa i pri isključenom
    VAD-u ide do ~89%; udeo odbačenog zvuka je svojstvo VAD-a nad tim snimkom i
    razdvaja dobar snimak od tihog (4.8% naspram 12.9%) tamo gde ih pokrivenost
    ne razdvaja (88.2% naspram 85.6%).
    """
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    th = VAD_THRESHOLD if threshold is None else float(threshold)
    audio = decode_audio(str(wav_path), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE
    spans = get_speech_timestamps(audio, VadOptions(**vad_parameters(th)))
    speech = sum(s["end"] - s["start"] for s in spans) / SAMPLE_RATE
    return SpeechMeasure(
        duration=duration, speech=speech, segments=len(spans), threshold=th
    )


#: Relativna težina interpunkcijskog tokena pri deljenju vremena reči.
PUNCT_TIME_WEIGHT = 0.5
#: Najduže vreme koje interpunkcija sme da uzme (sekunde).
PUNCT_MAX_SECONDS = 0.08

_model = None
_model_key: tuple[str, str, str] | None = None


# --------------------------------------------------------------------------
# CUDA DLL-ovi na Windows-u
# --------------------------------------------------------------------------


def _nvidia_bin_dirs() -> list[Path]:
    """site-packages/nvidia/*/bin — tu pip paketi drže cuBLAS/cuDNN DLL-ove."""
    roots: list[Path] = []
    candidates = [Path(sys.prefix) / "Lib" / "site-packages"]
    try:
        candidates += [Path(p) for p in site.getsitepackages()]
        candidates.append(Path(site.getusersitepackages()))
    except Exception:  # noqa: BLE001 — site zna da fali u embedded interpreteru
        pass
    for sp in candidates:
        p = sp / "nvidia"
        if p.is_dir() and p not in roots:
            roots.append(p)
    dirs: list[Path] = []
    for root in roots:
        for bindir in sorted(root.glob("*/bin")):
            if bindir.is_dir() and bindir not in dirs:
                dirs.append(bindir)
    return dirs


def _add_nvidia_dll_dirs() -> list[Path]:
    """ctranslate2 na Windows-u učitava cublas64_12.dll običnim LoadLibrary.

    `os.add_dll_directory` tu NIJE dovoljno — taj mehanizam važi samo za
    pretragu koju pokreće sam Python. Zato dirove guramo i u PATH procesa.
    Bez ovoga na `device="cuda"` dobiješ:
        RuntimeError: Library cublas64_12.dll is not found or cannot be loaded
    """
    if sys.platform != "win32":
        return []
    dirs = _nvidia_bin_dirs()
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    for bindir in dirs:
        s = str(bindir)
        try:
            os.add_dll_directory(s)
        except (OSError, AttributeError):
            pass
        if s not in path_parts:
            path_parts.insert(0, s)
    os.environ["PATH"] = os.pathsep.join(path_parts)
    return dirs


# --------------------------------------------------------------------------
# Učitavanje / oslobađanje modela
# --------------------------------------------------------------------------


def _runtime_error(exc: BaseException, device: str, compute_type: str) -> SubtitisError:
    """Pretvori mutnu ctranslate2 grešku u poruku iz koje se vidi šta da radiš."""
    msg = str(exc)
    hint = ""
    if "cublas" in msg.lower() or "cudnn" in msg.lower():
        found = ", ".join(str(d) for d in _nvidia_bin_dirs()) or "(nijedan)"
        hint = (
            "\nCUDA biblioteke nisu nađene. Pronađeni nvidia bin dirovi:\n"
            f"  {found}\n"
            "  .venv\\Scripts\\pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )
    elif "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower():
        hint = (
            "\nNema dovoljno VRAM-a. Proveri da LLM nije učitan (`ollama ps`, pa "
            "`ollama stop <model>`), ili probaj --compute-type int8_float16."
        )
    return SubtitisError(
        f"Whisper je pukao na {device}/{compute_type}.\n"
        f"Uzrok: {type(exc).__name__}: {msg}{hint}\n"
        "Za proveru bez GPU-a: --device cpu --compute-type int8"
    )


def load_model(
    model_size: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
):
    global _model, _model_key
    key = (model_size, device, compute_type)
    if _model is not None and _model_key == key:
        return _model
    unload_model()

    _add_nvidia_dll_dirs()
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SubtitisError(
            "faster-whisper nije instaliran.\n"
            "  .venv\\Scripts\\pip install -r requirements.txt"
        ) from exc

    try:
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as exc:  # noqa: BLE001 — pretvaramo u čitljivu poruku
        raise _runtime_error(exc, device, compute_type) from exc
    _model_key = key
    return _model


def unload_model() -> None:
    """Oslobodi VRAM. Zovi OBAVEZNO pre pokretanja LLM-a."""
    global _model, _model_key
    if _model is None:
        return
    del _model
    _model = None
    _model_key = None
    gc.collect()
    try:  # torch nije zavisnost Faze 1; ako postoji, dodatno očisti keš
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — čišćenje keša nikad ne sme da obori posao
        pass


# --------------------------------------------------------------------------
# Transkripcija
# --------------------------------------------------------------------------


def _split_word_times(text: str, start: float, end: float) -> list[tuple[str, float, float]]:
    """Podeli Whisperov token na reč + interpunkciju, svakom svoje vreme."""
    toks = split_token(text)
    if not toks:
        return []
    if len(toks) == 1:
        return [(toks[0], start, end)]

    span = max(end - start, 0.0)
    weights = [
        PUNCT_TIME_WEIGHT if not any(c.isalnum() for c in t) else float(len(t))
        for t in toks
    ]
    total = sum(weights) or 1.0
    out: list[tuple[str, float, float]] = []
    pos = start
    for tok, w in zip(toks, weights):
        share = span * w / total
        if not any(c.isalnum() for c in tok):
            share = min(share, PUNCT_MAX_SECONDS)
        out.append((tok, round(pos, 3), round(pos + share, 3)))
        pos += share
    # poslednji token dotakne pravi kraj reči
    tok, s, _ = out[-1]
    out[-1] = (tok, s, round(max(end, s), 3))
    return out


# --------------------------------------------------------------------------
# Francuska elizija — spajanje razbijenih tokena
# --------------------------------------------------------------------------
#
# Whisper za francuski redovno vraća eliziju kao DVA tokena, sa apostrofom uz
# drugi: `j` + `'aimerais`, `aujourd` + `'hui`. Ispisano sa razmacima to je
# „j 'aimerais" — razmak pre apostrofa umesto posle.
#
# Šteta nije kozmetička. `linebreak.py` vidi `j` i `'aimerais` kao dve reči, pa
# pravilo o eliziji (`keep_elisions`) ne radi i cue ume da počne sa `'espère`,
# odvojenim od svog `j`. Isto tako pada i restauracija interpunkcije: model
# normalizuje `j 'aimerais` u `j'aimerais`, a to je promena niza reči, pa
# `punctuate.validate` odbaci ceo čank.
#
# Zato se spaja ovde, odmah posle transkripcije, pre svega što čita reči.
# Po uzoru na `ekavica.py`: kurirani spisak osnova, bez širokog pravila.

#: Oba oblika apostrofa — Whisper vraća čas jedan, čas drugi.
APOSTROPHES = "'’"

#: Jezici za koje se elizija spaja. Isključivo francuski.
ELISION_LANGUAGES: frozenset[str] = frozenset({"fr"})

#: Osnove duže od `MAX_ELIDED_STEM` koje ipak jesu elizija. Bez ovog spiska
#: `aujourd'hui`, `qu'il`, `jusqu'à` i `lorsqu'on` ostaju razbijeni.
ELIDED_STEMS: frozenset[str] = frozenset({
    "qu",        # qu'il, qu'elle, qu'on
    "jusqu",     # jusqu'à, jusqu'ici
    "lorsqu",    # lorsqu'il
    "aujourd",   # aujourd'hui
    "puisqu",    # puisqu'il
    "quoiqu",    # quoiqu'il
    "quelqu",    # quelqu'un
    "presqu",    # presqu'île
})

#: Osnova do ovoliko slova se spaja i bez spiska: j', d', l', c', n', s', t', m'.
#: Duže osnove moraju biti u `ELIDED_STEMS` — inače bi svaki navodnik posle
#: reči progutao sledeću reč.
MAX_ELIDED_STEM = 2


@dataclass(frozen=True)
class Elision:
    """Jedno spajanje — za log."""

    index: int  # indeks prve reči u words.json, pre prenumeracije
    start: float  # vreme, sekunde
    before: str  # kako je bilo ispisano, sa razmakom
    after: str

    def __str__(self) -> str:
        return f"[{self.index}] {self.start:.2f}s  {self.before!r} -> {self.after!r}"


def elision_language(language: str | None) -> bool:
    return bool(language) and str(language).strip().lower() in ELISION_LANGUAGES


def _is_elided_stem(token: str) -> bool:
    """Da li token sme da bude LEVA strana elizije: `j`, `qu`, `aujourd`."""
    core = token.rstrip(APOSTROPHES)
    if not core or not core.isalpha():
        return False
    low = core.lower()
    return low in ELIDED_STEMS or len(low) <= MAX_ELIDED_STEM


def _elision_pair(left: Word, right: Word) -> str | None:
    """Vrati spojen ispis ili `None` ako par nije elizija."""
    a, b = left.w, right.w
    if not a or not b:
        return None
    # Apostrof je ili na kraju prvog tokena ili na početku drugog — Whisper
    # razbija na oba načina, a viđeno je i oboje odjednom.
    if not (a.endswith(tuple(APOSTROPHES)) or b.startswith(tuple(APOSTROPHES))):
        return None
    if not _is_elided_stem(a):
        return None
    rest = b.lstrip(APOSTROPHES)
    if not rest or not rest[0].isalpha():
        return None  # `d` + `'` samo, ili apostrof pa znak — nije elizija
    mark = a[-1] if a.endswith(tuple(APOSTROPHES)) else b[0]
    return a.rstrip(APOSTROPHES) + mark + rest


def merge_elisions(words: list[Word], language: str | None) -> list[Elision]:
    """Spoji razbijenu eliziju NA MESTU. Vrati spisak spajanja za log.

    Spojena reč uzima `s` prve i `e` druge reči. Verovatnoća je manja od dve —
    spajanje ne sme da sakrije nesigurnost ASR-a u editoru. Indeksi se
    prenumerišu; rečenice se prave posle ovoga, pa je to bezbedno.
    """
    if not elision_language(language) or len(words) < 2:
        return []

    merges: list[Elision] = []
    out: list[Word] = []
    i = 0
    while i < len(words):
        cur = words[i]
        if i + 1 < len(words):
            nxt = words[i + 1]
            joined = _elision_pair(cur, nxt)
            if joined is not None:
                merges.append(
                    Elision(
                        index=cur.i,
                        start=cur.s,
                        before=f"{cur.w} {nxt.w}",
                        after=joined,
                    )
                )
                out.append(
                    cur.model_copy(
                        update={
                            "w": joined,
                            "e": round(max(nxt.e, cur.s), 3),
                            "p": round(min(cur.p, nxt.p), 4),
                        }
                    )
                )
                i += 2
                continue
        out.append(cur)
        i += 1

    if merges:
        words[:] = [w.model_copy(update={"i": k}) for k, w in enumerate(out)]
    return merges


def transcribe(
    wav_path: str | Path,
    *,
    job_id: str,
    source_file: str,
    language: str,
    duration: float,
    glossary_terms: list[str] | None = None,
    model_size: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    compute_type: str = DEFAULT_COMPUTE_TYPE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    vad: bool = False,
    vad_threshold: float | None = None,
    no_speech_threshold: float | None = NO_SPEECH_THRESHOLD,
    log_prob_threshold: float | None = LOG_PROB_THRESHOLD,
    hallucination_silence_threshold: float | None = HALLUCINATION_SILENCE_THRESHOLD,
    progress: Any = None,
) -> WordsFile:
    """Transkribuj WAV i vrati popunjen `WordsFile` (uvek latinica).

    `vad=False` isključuje Silero VAD u potpunosti — sporije je i podložnije
    halucinacijama na tišini, ali ništa ne može biti odbačeno pre modela.
    Korisno za poređenje kad sumnjaš da VAD jede govor.
    """
    if not language:
        raise SubtitisError(
            "Jezik mora biti eksplicitno zadat. Za srpski uvek --lang sr; "
            "auto-detekcija ga meša sa hrvatskim i vraća ijekavicu."
        )
    # Auto-detekcija se traži IZRIČITO, nazivom. Prazan jezik i dalje je greška
    # — razlika je između „ne znam" i „neka model pogodi".
    auto = language.strip().lower() == AUTO_LANGUAGE

    initial_prompt = ", ".join(glossary_terms) if glossary_terms else None

    params: dict[str, Any] = {
        "language": None if auto else language,
        "word_timestamps": True,
        "vad_filter": vad,
        "vad_parameters": vad_parameters(vad_threshold) if vad else None,
        "beam_size": beam_size,
        "initial_prompt": initial_prompt,
        # `initial_prompt` uz condition_on_previous_text=False važi samo za prvi
        # prozor od 30 s — prompt se posle svakog segmenta resetuje. `hotwords`
        # se ubacuje u SVAKI prozor, pa glosar tako preživi ceo fajl.
        "hotwords": initial_prompt,
        "condition_on_previous_text": CONDITION_ON_PREVIOUS_TEXT,
        "compression_ratio_threshold": COMPRESSION_RATIO_THRESHOLD,
        "log_prob_threshold": log_prob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "temperature": list(TEMPERATURE_FALLBACK),
        "hallucination_silence_threshold": hallucination_silence_threshold,
    }

    model = load_model(model_size, device, compute_type)

    words: list[Word] = []
    raw_texts: list[str] = []
    idx = 0
    with RejectionCapture() as capture:
        try:
            segments, info = model.transcribe(str(wav_path), **params)
            for seg in segments:  # lenj generator — greške pucaju tek ovde
                for w in seg.words or []:
                    for tok, s, e in _split_word_times(w.word, float(w.start), float(w.end)):
                        raw_texts.append(tok)
                        words.append(
                            Word(
                                i=idx,
                                s=round(s, 3),
                                e=round(max(e, s), 3),
                                w=tok,
                                p=round(float(getattr(w, "probability", 1.0)), 4),
                            )
                        )
                        idx += 1
                if progress is not None:
                    progress(float(seg.end), duration)
        except RuntimeError as exc:
            raise _runtime_error(exc, device, compute_type) from exc

    original_script = detect_script(" ".join(raw_texts))
    if original_script != "Latn":
        for w in words:
            w.w = to_latin(w.w)

    detected = getattr(info, "language", None)
    if auto and not detected:
        raise SubtitisError(
            "Auto-detekcija nije vratila jezik. Zadaj ga izričito."
        )

    return WordsFile(
        job_id=job_id,
        source_file=source_file,
        duration=round(duration, 3),
        language=detected if auto else language,
        script="Latn",
        asr=AsrMeta(
            model=f"faster-whisper/{model_size}",
            params={
                **params,
                "compute_type": compute_type,
                "device": device,
                "original_script": original_script,
            },
            detected_language=getattr(info, "language", None),
            detected_probability=round(float(getattr(info, "language_probability", 0.0)), 4),
            rejections=dict(capture.counts),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        ),
        words=words,
    )
