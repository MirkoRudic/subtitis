"""FastAPI dashboard.

HTTP sloj ne obrađuje ništa što traži VRAM — samo upisuje zadatke u red koji
čita worker. Jedini posao koji radi sam je čitanje/pisanje JSON-a i isecanje
audio isečaka ffmpeg-om.

Od 2026-08-12 worker podrazumevano radi kao NIT u ovom istom procesu (`serve`
pali `WORKER_ENV`), da korisnik ne mora da pokreće dva prozora. Red ostaje isti
SQLite red, jedan zadatak odjednom; menja se samo ko ga vrti. `serve --no-worker`
vraća staru podelu na dva procesa.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import asr
from .. import config as cfg
from ..config import SubtitisError
from ..fixedcues import CASES, output_lines
from ..hallucination import EDITOR_NOTE as HALLUC_NOTE
from ..hallucination import word_key, zero_runs
from ..linebreak import align_text_to_words
from ..models import SentencesFile, WordsFile
from ..pipeline import default_profile_for, is_srt_job, resolve_profile
from ..segment import NO_SPACE_BEFORE
from ..srt import timestamp
from ..translit import BISCRIPTAL_LANGUAGES
from ..untranslated import plural
from . import db

HERE = Path(__file__).resolve().parent

#: Ispod ove verovatnoće reč se boji žuto u editoru.
LOW_CONFIDENCE = 0.6

#: Najduži isečak koji endpoint za slušanje sme da iseče (sekunde).
MAX_CLIP_SECONDS = 60.0

#: Na koliko sekundi editor snima u pozadini kad ima neupisanih izmena.
AUTOSAVE_SECONDS = 30

#: Izvorni jezici, redom kojim se nude. `auto` prepušta detekciju modelu.
SOURCE_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("sr", "srpski"),
    ("en", "engleski"),
    ("fr", "francuski"),
    ("de", "nemački"),
    (asr.AUTO_LANGUAGE, "neka model prepozna"),
)

#: Ciljni jezici prevoda, redom kojim se nude. Kod -> ispis u meniju.
TARGET_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("sr", "srpski"),
    ("en", "engleski"),
    ("fr", "francuski"),
    ("de", "nemački"),
)

#: Najveći upload. `source.wav` je ~115 MB po satu audija posle konverzije, pa
#: 2 GB je oko 17 sati zvuka — daleko iznad svega što se ovde radi, a i dalje
#: brana od slučajnog povlačenja pogrešnog fajla u prozor.
MAX_UPLOAD_BYTES = 2 * 1024**3

#: Nazivi izbora VAD-a. Broj (0.35) korisniku ne znači ništa dok ne pusti ceo
#: prepis; naziv je odluka koju može doneti unapred.
VAD_PRESET_LABELS: tuple[tuple[str, str, str], ...] = (
    ("normalno", "normalno", "za snimke sa jasnim govorom"),
    ("tih", "tih snimak", "zadržava tiši govor, propušta više šuma"),
    ("iskljuceno", "isključeno", "ništa se ne odbacuje pre modela; sporije"),
)

#: `BISCRIPTAL_LANGUAGES` (uvezeno gore) su jezici sa dva pisma — samo za njih
#: ima smisla birati pismo. Prelamanje ćirilicom nad francuskim tekstom nije
#: opcija nego kvar, pa se izbor pisma za takav cilj i ne prikazuje, a `Cyrl` se
#: odbija na ulazu. Isti spisak odlučuje i šta `pipeline` proverava na kraju
#: prevoda i prelamanja, pa definicija stoji na jednom mestu — u `translit`.

#: Promenljiva okruženja kojom `subtitis serve` traži workera U SVOM procesu.
#: Podrazumevano je ISKLJUČENO: `app` uvoze i testovi, a oni ne smeju da dobiju
#: nit koja im obrađuje red. Pali ga `serve`, ne modul.
WORKER_ENV = "SUBTITIS_WORKER_IN_PROCESS"


def worker_in_process() -> bool:
    return os.environ.get(WORKER_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _migrate_profiles() -> None:
    """Imena profila posle kruga 8 — jednom po poslu, pre prvog zahteva."""
    from ..migrate import migrate_all

    for m in migrate_all(on_migrated=db.migrate_queued_profiles):
        if m.status == "skipped":
            cfg.echo(f"UPOZORENJE: {m.job_id}: nepoznato ime profila {m.unknown} — nije migriran")


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    _migrate_profiles()
    worker = thread = None
    if worker_in_process():
        from .worker import SHUTDOWN_JOIN_SECONDS, start_worker_thread

        worker, thread = start_worker_thread()
    try:
        yield
    finally:
        if worker is not None:
            worker.stop()
            # Kratak `join` samo da worker koji miruje stigne da obriše otkucaj
            # — inače indikator ostane zelen posle gašenja. Nit je `daemon`, pa
            # zadatak koji je usred obrade ne može da zadrži gašenje.
            thread.join(timeout=SHUTDOWN_JOIN_SECONDS)


app = FastAPI(title="Subtitis", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(HERE / "templates"))


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> Any:
    """Greška iz obrasca je poruka, ne pad.

    Podrazumevani rukovalac vraća goli JSON, pa korisnik koji je poslao obrazac
    dobije crn ekran sa `{"detail": ...}`. Poziv iz JavaScripta i dalje dobija
    JSON — razlikuje se po `Accept` zaglavlju.
    """
    if "text/html" not in request.headers.get("accept", ""):
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status": exc.status_code, "detail": exc.detail},
        status_code=exc.status_code,
    )


# ==========================================================================
# Pomoćno
# ==========================================================================


def job_or_404(job_id: str):
    row = db.get_job(job_id)
    if row is None:
        raise HTTPException(404, f"Nema projekta {job_id}")
    return row


def sentences_path(job_id: str, lang: str) -> Path:
    return cfg.job_dir(job_id) / f"sentences.{lang}.json"


def load_sentences(job_id: str, lang: str) -> SentencesFile | None:
    p = sentences_path(job_id, lang)
    if not p.exists():
        return None
    return SentencesFile.model_validate(cfg.read_json(p))


def derived_language(sf: SentencesFile | None) -> str | None:
    """`sentences.sr.json` -> `sr`. Iz čega je ovaj prevod nastao."""
    if sf is None:
        return None
    parts = sf.derived_from.split(".")
    return parts[1] if len(parts) >= 3 and parts[0] == "sentences" else None


def available_languages(job_id: str) -> list[str]:
    d = cfg.job_dir(job_id)
    if not d.exists():
        return []
    return sorted(p.name.split(".")[1] for p in d.glob("sentences.*.json"))


# --- zauzeće diska ---------------------------------------------------------


def human_size(n: int | float) -> str:
    size = float(n)
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            out = f"{size:.0f} {unit}" if unit in ("B", "kB") else f"{size:.1f} {unit}"
            return out.replace(".", ",")  # decimalni zarez, kao svaki broj na ekranu
        size /= 1024
    return f"{size:.1f} TB".replace(".", ",")  # pragma: no cover — petlja izlazi na TB


def dir_size(path: Path) -> int:
    """Zbir veličina svih datoteka ispod `path`. Nepostojeći folder je 0."""
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:  # pragma: no cover — fajl nestao usred obilaska
            continue
    return total


def source_wav(job_id: str) -> Path:
    """Radni WAV. Iz njega dugme ▶ seče isečke; posle korekture nije potreban."""
    return cfg.job_dir(job_id) / "source.wav"


def job_disk(job_id: str) -> dict[str, Any]:
    src = source_wav(job_id)
    src_bytes = src.stat().st_size if src.exists() else 0
    total = dir_size(cfg.job_dir(job_id))
    return {
        "total": total,
        "total_h": human_size(total),
        "source_bytes": src_bytes,
        "source_h": human_size(src_bytes),
        "has_source": src_bytes > 0,
    }


def detection(job_id: str, lang: str | None) -> dict[str, Any] | None:
    """Šta je Whisper prepoznao, sa sigurnošću. `None` ako nema `words.json`.

    Vredi prikazati uvek, ne samo kod `auto`: kad se zada `sr` a model tvrdi
    `hr`, to je jedini trag zašto prepis izgleda ijekavski.
    """
    if not lang:
        return None
    path = cfg.job_dir(job_id) / f"words.{lang}.json"
    if not path.exists():
        return None
    try:
        meta = WordsFile.model_validate(cfg.read_json(path)).asr
    except Exception:  # noqa: BLE001 — prikaz ne sme da obori stranicu
        return None
    if not meta.detected_language:
        return None
    return {
        "language": meta.detected_language,
        "probability": meta.detected_probability,
        "pct": round(100 * (meta.detected_probability or 0.0)),
        "matches": meta.detected_language == lang,
    }


#: `job.json` po projektu, dok se fajl ne promeni (vreme izmene + veličina).
#: Jedan prikaz stranice ga je čitao tri puta; svako otvaranje je prilika da
#: worker baš tada ne uspe da ga zameni na Windowsu (krug 11).
_JOB_CACHE: dict[str, tuple[tuple[int, int], Any]] = {}


def job_record(job_id: str) -> Any:
    """`cfg.load_job`, ali najviše jedno čitanje dok se fajl ne promeni."""
    path = cfg.job_dir(job_id) / "job.json"
    try:
        st = path.stat()
    except OSError:
        return cfg.load_job(job_id)  # prijavi nedostatak kao i do sada
    key = (st.st_mtime_ns, st.st_size)
    hit = _JOB_CACHE.get(job_id)
    if hit is not None and hit[0] == key:
        return hit[1]
    job = cfg.load_job(job_id)
    st = path.stat()  # lenja migracija ume da ga prepiše pri čitanju
    _JOB_CACHE[job_id] = ((st.st_mtime_ns, st.st_size), job)
    return job


def last_build(job_id: str, lang: str | None) -> dict[str, Any] | None:
    """Kojim profilom i pismom je NAPRAVLJEN postojeći SRT za taj jezik.

    Sva tri profila pišu u isti fajl (`ime.fr.srt`) — ime po BCP-47 ne nosi
    profil. Bez ovoga se posle prelamanja ne vidi šta je primenjeno, a meni se
    vraća na podrazumevani profil kao da izbor nije ni postojao.
    """
    try:
        job = job_record(job_id)
    except Exception:  # noqa: BLE001 — prikaz ne sme da obori stranicu
        return None
    for rec in reversed(job.history):
        if rec.phase != "build" or rec.status != "ok":
            continue
        det = rec.detail
        if lang and det.get("language") not in (None, lang):
            continue
        return {
            "profile": det.get("profile"),
            "script": det.get("script", "Latn"),
            "case": det.get("case", "normal"),
            "drop_punct": bool(det.get("drop_punct", False)),
            "cues": det.get("cues"),
            "at": rec.finished_at,
            # samo posao iz SRT-a: cue-ovi koji probijaju profil
            "overflow": det.get("overflow") or [],
            "overflow_count": det.get("overflow_count"),
            "worse_count": (
                sum(1 for o in det.get("overflow") or [] if o.get("worse_than_source", True))
                if det.get("overflow") is not None and det.get("fixed_cues") else None
            ),
            "italic_lost": det.get("italic_lost") or [],
            "profile_suggestions": det.get("profile_suggestions") or [],
            "too_short": [
                {**c, "at": timestamp(c["s"])} for c in det.get("too_short") or []
            ],
            "min_duration": _profile_min_duration(det.get("profile")),
            "silence_timing": det.get("silence_timing"),
        }
    return None


def _profile_min_duration(name: str | None) -> float | None:
    try:
        return cfg.load_profile(name).min_duration if name else None
    except Exception:  # noqa: BLE001 — izvorni profil (`original`) nema fajl
        return None


def untranslated(job_id: str, langs: list[str]) -> list[dict[str, Any]]:
    """Po jeziku: rečenice koje su ostale na originalu ili liče na njega.

    Prikazuje se na vrhu stranice bez obzira na otvoren jezik — neprimećena
    neprevedena rečenica u gotovom titlu je gora od pada. Broje se i cue-ovi:
    jedna duga rečenica ume da pokrije deset cue-ova.
    """
    from .. import untranslated as u
    from ..pipeline import untranslated_report

    out = []
    for lang in langs:
        sf = load_sentences(job_id, lang)
        report = untranslated_report(job_id, sf) if sf else None
        if not report or not report.all:
            continue
        groups = []
        for items, forms, why in (
            (report.failed, u.NOT_TRANSLATED, "model je odustao, u titlu je original"),
            (report.suspect, u.LOOKS_ORIGINAL, "model je vratio izvorni tekst"),
        ):
            if not items:
                continue
            n = len(items)
            cues = sorted({c for i in items for c in i.cues})
            groups.append({
                "head": f"{n} {u.plural(n, 'rečenica', 'rečenice', 'rečenica')} "
                        f"{u.plural(n, *forms)}",
                "cues": (f"{len(cues)} {u.plural(len(cues), 'cue', 'cue-a', 'cue-ova')} "
                         f"({u.cue_ranges(cues)})") if cues else "",
                "why": why,
                "sentences": [i.as_dict() for i in items],
            })
        out.append({
            "lang": lang,
            "groups": groups,
            "ids": {i.sentence: i.reason for i in report.all},
        })
    return out


def overflow_by_sentence(built: dict[str, Any] | None) -> dict[int, list[dict[str, Any]]]:
    """id rečenice -> prekoračenja cue-ova u kojima ona stoji.

    `worse` kaže da li je prevod na toj meri gori od istog cue-a u originalu —
    samo takvi se boje crveno; ostalo je bilo isto i u originalu.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    if not built:
        return out
    for o in built["overflow"]:
        bits = []
        if o.get("chars_over"):
            src = f"; original {o['source_chars']}" if "source_chars" in o else ""
            bits.append(f"red {o['chars']} zn. (+{o['chars_over']}{src})")
        if o.get("cps_over"):
            src = f"; original {o['source_cps']:.1f}" if "source_cps" in o else ""
            bits.append(f"{o['cps']:.1f} CPS (+{o['cps_over']:.1f}{src})")
        if o.get("lines_over"):
            bits.append(f"+{o['lines_over']} red(a)")
        note = {
            "text": f"cue #{o['cue']}: " + ", ".join(bits),
            # stari zapisi nemaju poređenje sa originalom — tretiraju se kao gori
            "worse": o.get("worse_than_source", True),
        }
        for sid in o.get("sentences", []):
            out.setdefault(int(sid), []).append(note)
    return out


#: Koliko dugačka rečenica se uzima za primer preloma. Prekratka se ne bi
#: prelomila ni po jednom profilu, pa ne bi ništa pokazala.
PREVIEW_MIN_CHARS = 60


def profile_previews(job_id: str, lang: str | None) -> list[dict[str, Any]]:
    """Za svaki profil: brojevi, čemu služi, i ISTI cue prelomljen po njemu.

    Primer se uzima iz samog posla, ne izmišlja — inače pokazuje kako bi
    izgledao neki drugi tekst. Bira se najduža rečenica, jer se na kratkoj
    profili ne razlikuju.
    """
    from ..linebreak import build_cues

    sf = load_sentences(job_id, lang) if lang else None
    sample = None
    words: list = []
    if sf and sf.sentences:
        longest = max(sf.sentences, key=lambda s: len(s.text))
        if len(longest.text) >= PREVIEW_MIN_CHARS:
            sample = longest
            wp = cfg.job_dir(job_id) / f"words.{sf.language}.json"
            if not sf.derived_from.startswith("sentences.") and wp.exists():
                by_index = {
                    w.i: w
                    for w in WordsFile.model_validate(cfg.read_json(wp)).words
                }
                lo, hi = sample.words
                words = [by_index[k] for k in range(lo, hi + 1) if k in by_index]

    out: list[dict[str, Any]] = []
    # Posao iz SRT-a: prvo profili izmereni na samom ulazu (podrazumevani).
    from ..fixedcues import SOURCE_PROFILES

    names = (list(SOURCE_PROFILES) if is_srt_job(job_id) else []) + sorted(
        p.stem for p in cfg.PROFILES_DIR.glob("*.json")
    )
    for name in names:
        try:
            prof = resolve_profile(job_id, name)
        except SubtitisError:  # pragma: no cover — profil nestao usred rada
            continue
        card: dict[str, Any] = {
            "name": prof.name,
            "note": prof.note,
            "max_chars": prof.max_chars_per_line,
            "min_chars": getattr(prof, "min_chars_per_line", 0),
            "max_cps": prof.max_cps,
            "max_lines": prof.max_lines,
            "min_duration": prof.min_duration,
            "max_duration": prof.max_duration,
            # podrazumevani izlaz koji izbor profila postavlja (krug 8)
            "case": getattr(prof, "case", "normal"),
            "drop_punctuation": getattr(prof, "drop_punctuation", False),
            "lines": [],
        }
        if sample is not None:
            try:
                cues = build_cues(
                    [sample.model_copy(deep=True)], words, prof,
                    language=sf.language, log=lambda m: None,
                    interpolate_times=not words,
                )
                card["lines"] = [
                    line for c in cues
                    for line in output_lines(c.lines, card["case"], card["drop_punctuation"])
                ]
                card["cues"] = len(cues)
            except Exception:  # noqa: BLE001 — primer nije vredan pada stranice
                card["lines"] = []
        out.append(card)
    return out


#: Ime faze za ekran i jedan red šta radi. Ključ je ime faze iz `job.json`
#: (deo pre „:" — `translate:sr->en` je `translate`). Faza bez unosa se
#: prikazuje pod svojim imenom, bez opisa.
PHASE_LABELS: dict[str, tuple[str, str]] = {
    "audio": ("Priprema zvuka", "Pretvaranje u radnu kopiju 16 kHz mono."),
    "asr": ("Prepoznavanje govora", "Najduži korak. Oko 7 minuta po satu snimka."),
    "elizija": ("Jezičke normalizacije",
                "Svođenje na latinicu, ekavizacija, spajanje francuskih elizija."),
    "ekavica": ("Jezičke normalizacije",
                "Svođenje na latinicu, ekavizacija, spajanje francuskih elizija."),
    "punctuate": ("Interpunkcija", "Pokreće se samo ako je prepis siromašan znacima."),
    "segment": ("Segmentacija", "Reči se sastavljaju u rečenice."),
    "build": ("Prelamanje", "Rečenice se seku u cue-ove po profilu."),
    "translate": ("Prevod", "Pokreće se ručno, posle korekture."),
    "srt_import": ("Uvoz SRT-a",
                   "Čitanje ulaznog titla i sastavljanje rečenica iz cue-ova po interpunkciji."),
    "shorten": ("Skraćivanje",
                "Drugi prolaz kroz model, samo nad rečenicama koje su posle prevoda "
                "teže za čitanje nego original."),
}


def phase_label(name: str) -> tuple[str, str]:
    return PHASE_LABELS.get(name.split(":", 1)[0], (name, ""))


def speech_gaps(job_id: str, lang: str | None, duration: float | None) -> list[dict[str, Any]]:
    """Praznine duže od 3 s u prepisu — samo za prikaz uz pokrivenost.

    Računa se iz sačuvanih rečenica istim postupkom kao pri segmentaciji;
    do sada su bile samo u dnevniku.
    """
    from ..segment import coverage

    if not lang or not duration:
        return []
    sf = load_sentences(job_id, lang)
    if sf is None or sf.derived_from == "cues.json":
        return []
    try:
        _, gaps = coverage(sf.sentences, duration)
    except Exception:  # noqa: BLE001 — prikaz ne sme da obori stranicu
        return []
    return [{"s": s, "e": e, "len": round(e - s, 1)} for s, e in gaps]


def job_glossary_info(job_id: str, lang: str | None) -> dict[str, Any]:
    """Šta je glosar posla — za prikaz, jer ulazi u prepoznavanje."""
    own = cfg.job_glossary_path(job_id).exists()
    gloss = cfg.job_glossary(job_id)
    asr_terms = cfg.recording_terms(gloss, lang) if lang else []
    return {
        "own": own,
        "asr_terms": asr_terms,
        "recording_terms": cfg.recording_terms(gloss, lang) if lang else [],
        "pairs": {k: v for k, v in gloss.items() if "->" in k},
    }


def job_summary(job_id: str) -> dict[str, Any] | None:
    """Šta je urađeno, u brojevima. Iz `job.json`, ne iz loga.

    Postoji da se ne bi tražilo po `log.txt`: kojim modelom, koliko je trajalo,
    da li je punktuacija radila i koliko je ustupaka napravila.
    """
    try:
        job = job_record(job_id)
    except Exception:  # noqa: BLE001 — sažetak ne sme da obori stranicu
        return None

    phases: dict[str, Any] = {}
    for rec in job.history:  # poslednji prolaz svake faze pobeđuje
        if rec.status == "ok":
            phases[rec.phase] = rec

    if not phases:
        return None

    out: dict[str, Any] = {
        "total_s": round(sum(r.duration_s for r in job.history), 1),
        "steps": [],
    }
    for name, rec in phases.items():
        label, hint = phase_label(name)
        out["steps"].append({
            "phase": name,
            "label": label,
            "hint": hint,
            "duration_s": round(rec.duration_s, 1),
            "detail": rec.detail,
        })

    asr_rec = phases.get("asr")
    if asr_rec:
        out["model"] = asr_rec.detail.get("model")
        out["words"] = asr_rec.detail.get("words")
        out["vad"] = asr_rec.detail.get("vad")
        # Isti prag kao savet u dnevniku (`segment.vad_advice`), sada i na
        # ekranu — red u log.txt korisnik nije video (Radionica draft 1, 30,3% negovora).
        from ..segment import HIGH_SILENCE_SHARE

        share = asr_rec.detail.get("vad_discarded_share")
        if share is not None:
            out["nonspeech_pct"] = round(100 * share, 1)
        if not out["vad"] and share is not None and share >= HIGH_SILENCE_SHARE:
            out["silence_pct"] = round(100 * share)
    seg = phases.get("segment")
    if seg:
        out["sentences"] = seg.detail.get("sentences")
        out["coverage_pct"] = seg.detail.get("coverage_pct")
        out["low_confidence"] = seg.detail.get("low_confidence")
        out["gaps"] = speech_gaps(job_id, job.language, job.duration)
    punct = phases.get("punctuate")
    if punct:
        d = punct.detail
        out["punctuation"] = {
            "model": d.get("model"),
            "marks_added": d.get("marks_added"),
            "chunks_ok": d.get("chunks_ok"),
            "chunks_failed": d.get("chunks_failed"),
            "seconds_total": d.get("seconds_total"),
            "seconds_wasted": d.get("seconds_wasted"),
            "concessions": d.get("concessions") or {},
            "failed_ranges": len(d.get("failed_ranges") or []),
        }
    eliz = phases.get("elizija")
    if eliz:
        out["elisions"] = eliz.detail.get("merges")
    imp = phases.get("srt_import")
    if imp:
        d = imp.detail
        out["srt"] = {
            "cues": d.get("cues"),
            "sentences": d.get("sentences"),
            "multi_cue": d.get("multi_cue_sentences"),
            "shared_cues": d.get("shared_cues"),
            "encoding": d.get("encoding"),
            "original_script": d.get("original_script"),
            "forced_breaks": len(d.get("forced_breaks") or []),
        }
    return out


SCRIPT_NAMES = {"Latn": "latinica", "Cyrl": "ćirilica"}


_SRT_START_RE = re.compile(r"^(\d+):(\d\d):(\d\d),(\d{3}) -->", re.M)


def _cue_starts(text: str) -> list[float]:
    return [int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000
            for h, m, s, ms in _SRT_START_RE.findall(text)]


def output_files(
    job_id: str, invented: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Gotovi fajlovi: ime, oznaka jezika i pisma iz BCP-47 dela imena, broj
    cue-ova i veličina. Uz `invented` (`invented_sentences`) i broj natpisa
    koji počinju u izmišljenoj rečenici — prevod je na istim vremenima."""
    out = cfg.job_dir(job_id) / "out"
    if not out.exists():
        return []
    files = []
    for p in sorted(out.glob("*.srt")):
        tag = p.name[:-4].rsplit(".", 1)[-1] if p.name.count(".") >= 2 else ""
        lang, _, script = tag.partition("-")
        label = f"{lang} · {SCRIPT_NAMES.get(script, script)}" if script else lang
        try:
            text = p.read_text(encoding="utf-8-sig", errors="replace")
        except OSError:  # pragma: no cover
            text = None
        starts = _cue_starts(text) if text is not None and invented else []
        files.append({
            "name": p.name, "size_kb": round(p.stat().st_size / 1024, 1),
            "label": label or "srt", "cues": text.count("-->") if text is not None else None,
            "invented": sum(
                1 for t in starts
                if any(v["s"] - 0.001 <= t <= v["e"] + 0.001 for v in invented or [])
            ),
        })
    return files


def halluc_items(
    sf: SentencesFile | None, tokens: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Reči bez zvuka iza sebe, po nizu: tekst i vreme rečenice — za crvenu
    traku na vrhu, da se vidi ŠTA je izmišljeno, ne samo koliko."""
    out: list[dict[str, Any]] = []
    if not sf:
        return out

    def close(run: list[dict[str, Any]], sid: int) -> None:
        while run and not any(ch.isalnum() for ch in run[-1]["w"]):
            run.pop()  # zarez iza poslednje izmišljene reči nije deo niza
        if run:
            out.append({
                "text": tokens_to_text(run),
                "words": sum(1 for t in run if any(ch.isalnum() for ch in t["w"])),
                "at": timecode(run[0].get("s", 0.0)), "sentence": sid,
            })

    for sent in sf.sentences:
        run: list[dict[str, Any]] = []
        for t in tokens.get(sent.id) or []:
            if t.get("halluc"):
                run.append(t)
            elif run and not any(ch.isalnum() for ch in t["w"]):
                run.append(t)  # interpunkcija ne prekida niz: „kamp, OKC, Divčibare"
            else:
                close(run, sent.id)
                run = []
        close(run, sent.id)
    return out


def invented_sentences(
    sf: SentencesFile | None, tokens: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Rečenice sa bar jednim nizom reči bez zvuka — CELE, ne samo uhvaćene
    reči. Pravilo hvata reči nultog trajanja; izmišljena rečenica ume da ima i
    reči sa razvučenim vremenima (krug 23, Video-4 draft 2: 13 reči, uhvaćeno
    10), a izmišljena je cela. Ista brojka ide u traku i u „Za preuzimanje"."""
    out: list[dict[str, Any]] = []
    if not sf:
        return out
    for sent in sf.sentences:
        toks = tokens.get(sent.id) or []
        words = [t for t in toks if any(ch.isalnum() for ch in t["w"])]
        caught = sum(1 for t in words if t.get("halluc"))
        if caught:
            out.append({
                "sentence": sent.id, "words": len(words), "caught": caught,
                "at": timecode(toks[0].get("s", 0.0)),
                "s": toks[0].get("s", 0.0), "e": max(t.get("e", 0.0) for t in toks),
            })
    return out


def job_invented(job_id: str, language: str | None) -> list[dict[str, Any]]:
    """`invented_sentences` izvornog prepisa — i kad je na ekranu prevod:
    prevod izmišljene rečenice je isto izmišljen."""
    if not language or is_srt_job(job_id):
        return []
    sf = load_sentences(job_id, language)
    return invented_sentences(sf, marked_tokens(sf, job_id)) if sf else []


def tokens_to_text(tokens: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for k, t in enumerate(tokens):
        if k and t["w"][:1] not in NO_SPACE_BEFORE:
            parts.append(" ")
        parts.append(t["w"])
    return "".join(parts)


def marked_tokens(sf: SentencesFile, job_id: str) -> dict[int, list[dict[str, Any]]]:
    """Za svaku rečenicu: tokeni sa oznakom da li im je ASR verovatnoća niska.

    Za prevedene rečenice vraća prazno — prevod nema verovatnoće po rečima i
    lažno bojenje bi bilo gore od nikakvog.
    """
    if sf.derived_from.startswith("sentences."):
        return {}
    wp = cfg.job_dir(job_id) / f"words.{sf.language}.json"
    if not wp.exists():
        return {}
    wf = WordsFile.model_validate(cfg.read_json(wp))
    by_index = {w.i: w for w in wf.words}
    # Nizovi se traže u ORIGINALNOM ASR izlazu, ne u poravnatom tekstu: reči
    # koju je čovek umetnuo poravnanje izmisli vreme (ume i nulto), pa bi bila
    # lažno obojena. Boje se samo tokeni koji su bili u nizu — kad ih čovek
    # izbaci, crveno nestaje.
    halluc_keys = {
        word_key(w) for run in zero_runs(wf.words) for w in run
    }

    out: dict[int, list[dict[str, Any]]] = {}
    for sent in sf.sentences:
        lo, hi = sent.words
        sw = [by_index[k] for k in range(lo, hi + 1) if k in by_index]
        if not sw:
            continue
        aligned, _ = align_text_to_words(sent.text, sw)
        out[sent.id] = [
            {"w": w.w, "p": w.p, "s": w.s, "e": w.e, "low": w.p < LOW_CONFIDENCE,
             "halluc": word_key(w) in halluc_keys}
            for w in aligned
        ]
    return out


def _is_punct_token(tok: str) -> bool:
    return not any(ch.isalnum() for ch in tok)


def tokens_to_html(tokens: list[dict[str, Any]]) -> str:
    """Tekst rečenice: žuto nesigurne reči, crveno i precrtano izmišljene.

    Crveno ima prednost. Zarez između dve izmišljene reči je deo niza — inače
    žuti zarezi rasparčaju crveni niz i ceo izgleda kao nesiguran (Radionica draft 1, krug 9).
    """
    parts: list[str] = []
    for k, t in enumerate(tokens):
        tok = t["w"]
        if k and tok[:1] not in NO_SPACE_BEFORE:
            parts.append(" ")
        esc = html.escape(tok)
        inside_run = (
            _is_punct_token(tok) and 0 < k < len(tokens) - 1
            and tokens[k - 1].get("halluc") and tokens[k + 1].get("halluc")
        )
        if t.get("halluc") or inside_run:
            parts.append(
                f'<span class="halluc" title="{html.escape(HALLUC_NOTE)}">{esc}</span>'
            )
        elif t["low"]:
            parts.append(
                f'<span class="low" title="sigurnost modela {t["p"]:.2f}">{esc}</span>'
            )
        else:
            parts.append(esc)
    return "".join(parts)


#: Oznake rečenice na ekranu. Ime polja iz koda (`low_confidence`) nije reč
#: za korisnika; nepoznata oznaka se prikazuje kakva jeste.
FLAG_LABELS: dict[str, str] = {
    "low_confidence": "nesigurne reči",
    "translation_failed": "prevod nije uspeo",
    "alignment_degraded": "vremena procenjena",
    "shortened": "skraćeno",
    "hallucinated": "izmišljeno — bez zvuka",
}


def flag_label(flag: str) -> str:
    return FLAG_LABELS.get(flag, flag.replace("_", " "))


templates.env.filters["flag_label"] = lambda f: flag_label(f)


def timecode(seconds: float) -> str:
    """`01:02,34` — decimalni zarez, kao svaki broj na ekranu (`dec`)."""
    s = max(0.0, seconds)
    return f"{int(s // 60):02d}:{s % 60:05.2f}".replace(".", ",")


def dec(value: Any) -> str:
    """Broj za ekran: decimalni zarez, ne tačka (87,4%, 0,8 kB)."""
    return str(value).replace(".", ",") if value is not None else ""


templates.env.filters["timecode"] = timecode
templates.env.filters["dec"] = dec
templates.env.filters["plural"] = lambda n, one, few, many: plural(n, one, few, many)

from .. import __version__ as VERSION  # noqa: E402 — potpis u podnožju (base.html)
from .. import version_string  # noqa: E402

templates.env.globals["version"] = VERSION
templates.env.globals["signature"] = version_string()


# ==========================================================================
# Ekran 1: lista poslova + upload
# ==========================================================================


#: Status projekta za pilulu: (tekst, klasa). Pilula gori samo kad treba —
#: obrada žuto i pulsira, greška crveno sa stvarnim razlogom (vidi
#: `status_pill`), gotov projekat mirno.
STATUS_PILLS: dict[str, tuple[str, str]] = {
    "uploaded": ("u redu", "run"),
    "transcribing": ("prepoznavanje", "run"),
    "transcribed": ("prepis za korekturu", ""),
    "translating": ("prevod u toku", "run"),
    "done": ("titl spreman", "done"),
}

MONTHS = ("jan", "feb", "mar", "apr", "maj", "jun", "jul", "avg", "sep", "okt", "nov", "dec")


def status_pill(status: str, error: str | None) -> dict[str, str]:
    if status == "error":
        # stvarni razlog umesto reči `error` — prvi red poruke je dovoljan
        reason = (error or "greška").strip().splitlines()[0] if (error or "").strip() else "greška"
        return {"text": reason, "cls": "err", "title": error or ""}
    text, cls = STATUS_PILLS.get(status, (status, ""))
    return {"text": text, "cls": cls, "title": ""}


def human_date(iso: str | None) -> str:
    """`danas 17:50`, `juče 09:12`, `30. sep`, `30. sep 2025`."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso).astimezone()
    except ValueError:
        return iso
    today = datetime.now().astimezone().date()
    if dt.date() == today:
        return f"danas {dt:%H:%M}"
    if (today - dt.date()).days == 1:
        return f"juče {dt:%H:%M}"
    out = f"{dt.day}. {MONTHS[dt.month - 1]}"
    return out if dt.year == today.year else f"{out} {dt.year}"


def project_name(source_name: str) -> str:
    """Ime fajla bez nastavka — naslov projekta. `job_id` se ne prikazuje."""
    return Path(source_name).stem or source_name


def languages_label(job_id: str, source: str) -> str:
    """`sr` ili `sr → fr, en` — izvorni jezik pa prevodi."""
    others = [l for l in available_languages(job_id) if l != source]
    return f"{source} → {', '.join(others)}" if others else source


def job_rows() -> list[dict[str, Any]]:
    jobs = db.rows_to_dicts(db.list_jobs())
    for j in jobs:
        j["pending"] = db.pending_task(j["id"]) is not None
        j["disk"] = job_disk(j["id"])
        j["name"] = project_name(j["source_name"])
        j["date"] = human_date(j.get("created_at"))
        j["langs"] = languages_label(j["id"], j["language"])
        j["pill"] = status_pill(j["status"], j.get("error"))
    return jobs


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request, obrisan: str | None = None, oslobodjeno: str | None = None,
) -> Any:
    rows = job_rows()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "jobs": rows,
            "any_active": any(j["pending"] or j["pill"]["cls"] == "run" for j in rows),
            "deleted": {"name": obrisan, "freed": oslobodjeno} if obrisan else None,
            "disk_total": human_size(dir_size(cfg.JOBS_DIR)),
            "statuses": db.STATUSES,
            "wav_ttl_days": cfg.source_wav_ttl_days(),
            "source_langs": SOURCE_LANGUAGES,
            "vad_presets": VAD_PRESET_LABELS,
            "default_vad": asr.DEFAULT_VAD_PRESET,
            "max_upload": human_size(MAX_UPLOAD_BYTES),
            "default_glossary_terms": len(cfg.glossary_keys(cfg.load_glossary(), "sr")),
            "target_langs": TARGET_LANGUAGES,
            "default_target": cfg.default_target_language(),
        },
    )


@app.get("/kako", response_class=HTMLResponse)
def how_it_works(request: Request) -> Any:
    """Jedna stranica: oba toka, šta alat ne radi, česti problemi."""
    return templates.TemplateResponse(
        request, "kako.html", {"wav_ttl_days": cfg.source_wav_ttl_days()}
    )


@app.get("/jobs/list", response_class=HTMLResponse)
def jobs_list(request: Request) -> Any:
    """HTMX osvežava samo tabelu."""
    return templates.TemplateResponse(
        request,
        "_jobs.html",
        {
            "jobs": job_rows(),
            "wav_ttl_days": cfg.source_wav_ttl_days(),
            "disk_total": human_size(dir_size(cfg.JOBS_DIR)),
        },
    )


@app.get("/worker/status", response_class=HTMLResponse)
def worker_status(request: Request) -> Any:
    """Indikator u zaglavlju. HTMX ga vuče na 3s."""
    return templates.TemplateResponse(
        request,
        "_worker.html",
        {
            "worker": db.worker_status(),
            "disk_total": human_size(dir_size(cfg.JOBS_DIR)),
        },
    )


@app.post("/jobs")
async def create_job(
    audio: UploadFile,
    language: str = Form("sr"),
    vad_preset: str = Form(asr.DEFAULT_VAD_PRESET),
    glossary_mode: str | None = Form(None),
    terms: str = Form(""),
) -> Any:
    """Otpremi i stavi transkripciju u red.

    Whisper model se NE bira ovde. Spuštanje na `medium` daje osetno lošiji
    srpski prepis, a ništa na ekranu to ne bi objasnilo — rizik je veći od
    koristi. Ostaje kao CLI opcija (`subtitis transcribe --model`).
    """
    if not audio.filename:
        raise HTTPException(400, "Nije izabran fajl.")
    if vad_preset not in asr.VAD_PRESETS:
        raise HTTPException(400, f"Nepoznat izbor filtera tišine: {vad_preset!r}")

    src = Path(audio.filename)
    job_id = cfg.make_job_id(src)
    jdir = cfg.job_dir(job_id)
    (jdir / "upload").mkdir(parents=True, exist_ok=True)
    dst = jdir / "upload" / src.name

    # Piše se u komadima da se fajl od par GB ne učita u memoriju, i da se
    # prekoračenje uhvati usred pisanja, a ne tek kad prođe ceo.
    written = 0
    try:
        with dst.open("wb") as fh:
            while chunk := await audio.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"Fajl je veći od dozvoljenih {human_size(MAX_UPLOAD_BYTES)}.",
                    )
                fh.write(chunk)
    except HTTPException:
        dst.unlink(missing_ok=True)  # nedovršen upload se ne ostavlja na disku
        raise

    # Početna strana više ne nudi izbor: šta nov projekat dobija bira se u
    # podešavanjima (`new_project_glossary`). Polje ostaje za CLI i testove.
    glossary_mode = glossary_mode or cfg.new_project_glossary()
    if glossary_mode not in ("default", "none"):
        dst.unlink(missing_ok=True)
        raise HTTPException(400, f"Nepoznat izbor glosara: {glossary_mode!r}")
    # Glosar posla: kopija podrazumevanog ili prazan, plus termini snimka.
    # Pravi se PRE prepoznavanja — `hotwords` ide u svaki prozor od 30 s.
    job_gloss = cfg.make_job_glossary(
        language, use_default=glossary_mode == "default",
        recording_terms=terms.splitlines(),
    )
    cfg.write_json(cfg.job_glossary_path(job_id), job_gloss)
    db.create_job(job_id, src.name, str(dst.resolve()), language)
    db.add_event(job_id, "log", f"otpremljeno: {src.name} ({human_size(written)})")
    db.add_event(
        job_id, "log",
        f"glosar projekta: {'kopija podrazumevanog' if glossary_mode == 'default' else 'prazan'}"
        f", termina snimka {len(cfg.recording_terms(job_gloss, language))}",
    )
    db.enqueue(
        job_id, "transcribe",
        {"model": cfg.DEFAULT_ASR_MODEL, "vad_preset": vad_preset},
    )
    db.add_event(job_id, "status", "uploaded")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


#: Najveći SRT. Sat titla je ~100 kB; ovo je brana od pogrešnog fajla.
MAX_SRT_BYTES = 20 * 1024**2


@app.post("/jobs/srt")
async def create_srt_job(
    srt_file: UploadFile,
    language: str = Form("sr"),
    target: str = Form(...),
) -> Any:
    """Postojeći SRT -> posao u stanju `transcribed`, prevod odmah u redu.

    Nema zvuka ni words.json. Uvoz je čitanje i pisanje JSON-a, bez modela, pa
    ga HTTP sloj radi sam; u red ide samo prevod.

    NAMERNO odstupa od pauze za pregled posle `transcribed`: ta pauza postoji
    zato što prevod neispravljenog ASR teksta prevodi i greške, a ovaj ulaz je
    već ručno doteran titl, pa razlog za pauzu ne postoji. Ne „ispravljati".
    """
    from .. import pipeline

    if not srt_file.filename:
        raise HTTPException(400, "Nije izabran fajl.")
    known = {c for c, _ in TARGET_LANGUAGES}
    if language not in known:
        raise HTTPException(400, f"Nepoznat izvorni jezik: {language!r}")
    if target not in known:
        raise HTTPException(400, f"Nepoznat ciljni jezik: {target!r}")
    if target == language:
        raise HTTPException(400, "Izvorni i ciljni jezik su isti.")

    data = await srt_file.read(MAX_SRT_BYTES + 1)
    if len(data) > MAX_SRT_BYTES:
        raise HTTPException(413, f"SRT je veći od {human_size(MAX_SRT_BYTES)}.")

    src = Path(srt_file.filename)
    job_id = cfg.make_job_id(src)
    jdir = cfg.job_dir(job_id)
    (jdir / "upload").mkdir(parents=True, exist_ok=True)
    dst = jdir / "upload" / src.name
    dst.write_bytes(data)

    events: list[str] = []
    try:
        r = pipeline.run_import_srt(
            dst, language, job_id=job_id, on_log=events.append
        )
    except SubtitisError as exc:
        shutil.rmtree(jdir, ignore_errors=True)
        raise HTTPException(400, f"SRT nije učitan: {exc}") from exc

    db.create_job(job_id, src.name, str(dst.resolve()), language)
    db.update_job(job_id, status="transcribed", duration=r.duration)
    db.add_event(job_id, "log", f"otpremljeno: {src.name} ({human_size(len(data))})")
    for msg in events:
        db.add_event(job_id, "log", msg)
    db.add_event(job_id, "status", "transcribed")
    db.enqueue(
        job_id, "translate",
        {"to": target, "model": None, "pivot": None, "profile": "", "script": "Latn"},
    )
    db.add_event(job_id, "log", f"prevod na {target} stavljen u red")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/delete")
def remove_job(job_id: str) -> Any:
    """Obriši posao iz liste I sa diska.

    Do sada su datoteke ostajale, pa je „obriši" ostavljalo 40 MB po poslu i
    korisnik nije imao nijedan način da ih se reši osim ručno kroz Explorer.
    """
    job_or_404(job_id)
    # Zadatak u redu se briše zajedno sa poslom (`delete_job` briše i red).
    # Zadatak u toku to primeti na sledećoj prirodnoj tački — pre poziva
    # modelu ili posle ASR segmenta — i stane; worker posle njega počisti
    # ono što je eventualno upisao posle brisanja (`worker.purge_job_dir`).
    running = db.pending_task(job_id)
    jdir = cfg.job_dir(job_id)
    name = project_name(db.get_job(job_id)["source_name"])
    freed = dir_size(jdir) if jdir.exists() else 0
    db.delete_job(job_id)
    if jdir.exists():
        # Na Windows-u fajl koji worker drži otvoren (source.wav) ne može da
        # se obriše odmah; ostatak briše worker kad zadatak stane.
        shutil.rmtree(jdir, ignore_errors=True)
    if running is not None:
        cfg.echo(
            f"posao {job_id} obrisan uz zadatak {running['kind']} ({running['state']}) "
            "— otkazan"
        )
    # Potvrda na početnoj, kao kod prelamanja (krug 11): bez nje se ne vidi da
    # je projekat stvarno nestao.
    from urllib.parse import urlencode

    query = urlencode({"obrisan": name, "oslobodjeno": human_size(freed)})
    return RedirectResponse(f"/?{query}#projekti", status_code=303)


@app.get("/api/jobs/status")
def jobs_status() -> Any:
    """Pilule svih projekata, samo iz baze — bez otvaranja ijednog fajla.

    Početna ovo pita na 3 s SAMO dok bar jedan projekat ima zadatak u toku ili
    u redu, i menja samo pilule; lista, skrol i otvorene trake ostaju gde jesu.
    """
    out = {}
    for row in db.list_jobs():
        pending = db.pending_task(row["id"]) is not None
        pill = status_pill(row["status"], row["error"])
        out[row["id"]] = {**pill, "pending": pending,
                          "active": pending or pill["cls"] == "run"}
    return {"jobs": out, "active": any(j["active"] for j in out.values())}


# ==========================================================================
# Podešavanja
# ==========================================================================


def glossary_rows() -> list[dict[str, Any]]:
    gloss = cfg.load_glossary()
    return [
        {"pair": pair, "term": term, "translation": tr}
        for pair in sorted(gloss)
        for term, tr in sorted(gloss[pair].items())
    ]


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, saved: str | None = None) -> Any:
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "values": {
                "default_profile": cfg.default_profile(),
                "default_target_language": cfg.default_target_language(),
                "llm_model": cfg.llm_model(),
                "source_wav_ttl_days": cfg.source_wav_ttl_days(),
                "auto_punctuate": cfg.auto_punctuate(),
                "new_project_glossary": cfg.new_project_glossary(),
                "translation_pivot": cfg.translation_pivot(),
            },
            "default_glossary_terms": len(cfg.glossary_keys(cfg.load_glossary(), "sr")),
            "profiles": sorted(p.stem for p in cfg.PROFILES_DIR.glob("*.json")),
            "target_langs": TARGET_LANGUAGES,
            "glossary": glossary_rows(),
            "config_file": str(cfg.CONFIG_FILE),
            "glossary_file": str(cfg.glossary_path()),
            "saved": saved,
            # Promenljiva okruženja pobeđuje fajl — ako je postavljena, ekran to
            # mora da kaže, inače korisnik menja vrednost koja se ne primenjuje.
            "env_overrides": {
                name: os.environ[env]
                for name, env in (
                    ("default_profile", "SUBTITIS_PROFILE"),
                    ("llm_model", "SUBTITIS_LLM_MODEL"),
                    ("auto_punctuate", "SUBTITIS_AUTO_PUNCTUATE"),
                )
                if os.environ.get(env)
            },
        },
    )


@app.post("/settings")
def save_settings_form(
    default_profile: str = Form(...),
    default_target_language: str = Form(...),
    llm_model: str = Form(...),
    source_wav_ttl_days: int = Form(...),
    auto_punctuate: str = Form(""),
    new_project_glossary: str = Form("default"),
    translation_pivot: str = Form(""),
) -> Any:
    known = {p.stem for p in cfg.PROFILES_DIR.glob("*.json")}
    if default_profile not in known:
        raise HTTPException(400, f"Nepoznat profil: {default_profile!r}")
    if default_target_language not in {c for c, _ in TARGET_LANGUAGES}:
        raise HTTPException(400, f"Nepoznat jezik: {default_target_language!r}")
    if not llm_model.strip():
        raise HTTPException(400, "Model za prevod ne sme biti prazan.")
    if source_wav_ttl_days < 0:
        raise HTTPException(400, "Broj dana ne može biti negativan.")
    if new_project_glossary not in ("default", "none"):
        raise HTTPException(400, f"Nepoznat izbor glosara: {new_project_glossary!r}")
    if translation_pivot not in ("", "en"):
        raise HTTPException(400, f"Nepoznat pivot: {translation_pivot!r}")

    cfg.save_settings({
        "default_profile": default_profile,
        "default_target_language": default_target_language,
        "llm_model": llm_model.strip(),
        "source_wav_ttl_days": source_wav_ttl_days,
        "auto_punctuate": auto_punctuate == "on",
        "new_project_glossary": new_project_glossary,
        "translation_pivot": translation_pivot,
    })
    return RedirectResponse("/settings?saved=podesavanja", status_code=303)


@app.post("/settings/glossary")
def save_glossary_entry(
    pair: str = Form(...),
    term: str = Form(...),
    # Prazan prevod je dozvoljen na ULAZU jer brisanju ne treba; da je obavezan,
    # obrazac za brisanje bi padao na 422 umesto da obriše unos.
    translation: str = Form(""),
    remove: str = Form(""),
) -> Any:
    """Dodaj ili obriši jedan unos glosara."""
    pair, term = pair.strip(), term.strip()
    if not pair or "->" not in pair:
        raise HTTPException(400, "Smer mora izgledati kao `sr->en`.")
    if not term:
        raise HTTPException(400, "Termin ne sme biti prazan.")

    gloss = {k: dict(v) for k, v in cfg.load_glossary().items()}
    entries = gloss.setdefault(pair, {})
    if remove == "on":
        if entries.pop(term, None) is None:
            raise HTTPException(404, f"Nema unosa {term!r} za {pair}.")
        what = "obrisan"
    else:
        if not translation.strip():
            raise HTTPException(400, "Prevod ne sme biti prazan.")
        entries[term] = translation.strip()
        what = "sačuvan"
    cfg.save_glossary(gloss)
    return RedirectResponse(f"/settings?saved=glosar-{what}", status_code=303)


# ==========================================================================
# Ekran 2: editor (glavni ekran)
# ==========================================================================


def _cue_line_html(line: str) -> str:
    """Red cue-a kako je u fajlu; `<i>` se prikazuje kao kurziv, ostalo kao tekst."""
    esc = html.escape(line)
    return esc.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")


def preview_by_sentence(job_id: str, lang: str) -> dict[int, list[dict[str, Any]]]:
    """Cue-ovi gotovog titla po rečenici (`pipeline.write_preview`).

    Samo ako fajl na koji se pregled odnosi i dalje postoji — inače bi editor
    pokazivao prelom titla kojeg nema.
    """
    from ..pipeline import preview_path

    p = preview_path(job_id, lang)
    if not p.exists():
        return {}
    try:
        data = cfg.read_json(p)
    except SubtitisError:
        return {}
    if not (cfg.job_dir(job_id) / "out" / data.get("srt", "")).exists():
        return {}
    out: dict[int, list[dict[str, Any]]] = {}
    for c in data.get("cues", []):
        item = {**c, "html": "\n".join(_cue_line_html(l) for l in c.get("lines", []))}
        for sid in c.get("sentences", []):
            out.setdefault(sid, []).append(item)
    return out


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_page(request: Request, job_id: str, lang: str | None = None) -> Any:
    job = dict(job_or_404(job_id))
    langs = available_languages(job_id)
    active = lang or job["language"]
    sf = load_sentences(job_id, active) if active in langs else None
    tokens = marked_tokens(sf, job_id) if sf else {}

    # Kad je otvoren prevod, uz njega ide original u istoj vrsti — inače se pri
    # ispravljanju mora pamtiti šta je pisalo. Veže se po `id`, koji prevod
    # zadržava iz izvornog fajla.
    is_translation = bool(sf and sf.derived_from.startswith("sentences."))
    # Izvor se čita iz `derived_from`, ne iz jezika posla: kod pivota
    # (sr -> en -> de) original za `de` je `en`, a ne `sr`.
    source_lang = derived_language(sf) if is_translation else None
    if source_lang == active:
        source_lang = None
    source_sf = load_sentences(job_id, source_lang) if source_lang else None
    source_tokens = marked_tokens(source_sf, job_id) if source_sf else {}
    source_by_id = {s.id: s for s in source_sf.sentences} if source_sf else {}

    # Posao iz SRT-a nema ni zvuka ni verovatnoća: bez dugmeta ▶ i bez žutog.
    # Umesto toga se obeležavaju rečenice čiji cue-ovi probijaju profil.
    srt_job = is_srt_job(job_id)
    built = last_build(job_id, active)
    untr = untranslated(job_id, langs)
    untr_active = next((u["ids"] for u in untr if u["lang"] == active), {})
    over = overflow_by_sentence(built) if srt_job else {}
    preview = preview_by_sentence(job_id, active) if sf else {}

    rows = []
    if sf:
        for s in sf.sentences:
            toks = tokens.get(s.id)
            row = {
                "id": s.id,
                "s": s.s,
                "e": s.e,
                "flags": s.flags,
                "html": tokens_to_html(toks) if toks else html.escape(s.text),
                "low": sum(1 for t in toks or [] if t["low"]),
                "source_html": "",
                "source_low": 0,
                "over": over.get(s.id, []),
                "halluc": any(t.get("halluc") for t in toks or []),
                "untranslated": untr_active.get(s.id),
                "cues": [p.cue + 1 for p in s.cue_parts],
                "preview": preview.get(s.id, []),
            }
            src = source_by_id.get(s.id)
            if src is not None:
                stoks = source_tokens.get(s.id)
                # Bojenje važi SAMO za original: prevod nema verovatnoće po
                # rečima, pa bi žuto na njemu bilo izmišljeno.
                row["source_html"] = (
                    tokens_to_html(stoks) if stoks else html.escape(src.text)
                )
                row["source_low"] = sum(1 for t in stoks or [] if t["low"])
                # izmišljen original -> i prevod je prevod izmišljenog
                row["halluc"] = row["halluc"] or any(t.get("halluc") for t in stoks or [])
            rows.append(row)

    halluc = halluc_items(sf, tokens) if sf and not is_translation else []
    invented = (invented_sentences(sf, tokens) if not is_translation
                else job_invented(job_id, job["language"]))
    return templates.TemplateResponse(
        request,
        "job.html",
        {
            "job": job,
            "name": project_name(job["source_name"]),
            "pill": status_pill(job["status"], job.get("error")),
            "langs_label": languages_label(job_id, job["language"]),
            "halluc_items": halluc,
            "halluc_words": sum(h["words"] for h in halluc),
            "invented": invented,
            "status_pills": STATUS_PILLS,
            "retry": [TASK_LABELS.get(t["kind"], t["kind"]) for t in db.retryable_tasks(job_id)]
                     if job["status"] == "error" else [],
            "vad_presets": VAD_PRESET_LABELS,
            "recording_terms": cfg.recording_terms(cfg.job_glossary(job_id), job["language"]),
            "langs": langs,
            "active": active,
            "sf": sf,
            "rows": rows,
            "low_rows": sum(1 for r in rows if "low_confidence" in r["flags"]),
            "outputs": output_files(job_id, invented),
            "pending": db.pending_task(job_id) is not None,
            "task_summary": db.job_task_summary(job_id),
            "bad_tasks": db.rows_to_dicts(db.failed_tasks(job_id)),
            "events": db.rows_to_dicts(db.recent_events(job_id)),
            "last_event_id": db.last_event_id(job_id),
            # Bojenje zavisi od POREKLA rečenica, ne od toga da li slučajno
            # postoji words.<lang>.json — prevod nema verovatnoće po rečima.
            "is_translation": is_translation,
            "source_lang": source_lang if source_sf else None,
            "low_source_rows": sum(1 for r in rows if r["source_low"]),
            # Izvorni jezik nije ciljni — prevod sr -> sr pukne u pajplajnu.
            "target_langs": [
                (code, name) for code, name in TARGET_LANGUAGES
                if code != job["language"]
            ],
            "biscriptal": sorted(BISCRIPTAL_LANGUAGES),
            "default_target": cfg.default_target_language(),
            "disk": job_disk(job_id),
            "detected": detection(job_id, active),
            "profile_cards": profile_previews(job_id, active),
            # Šta je STVARNO primenjeno na postojeći SRT — meni se preselektuje
            # po ovome, ne po podrazumevanom profilu.
            "built": built,
            "srt_job": srt_job,
            "halluc_note": HALLUC_NOTE,
            "glossary_info": job_glossary_info(job_id, job["language"]),
            "halluc_rows": sum(1 for r in rows if r["halluc"]),
            "untranslated": untr,
            "shorten": shorten_proposal(job_id, active) if srt_job else None,
            "over_rows": sum(1 for r in rows if any(o["worse"] for o in r["over"])),
            "default_profile": default_profile_for(job_id),
            "llm_model": cfg.llm_model(),
            "summary": job_summary(job_id),
            "versions": version_list(job_id, active) if sf else [],
            "autosave_seconds": AUTOSAVE_SECONDS,
            "phase_labels": {k: v[0] for k, v in PHASE_LABELS.items()},
        },
    )


@app.post("/jobs/{job_id}/sentences")
async def save_sentences(job_id: str, request: Request) -> Any:
    """Snimi ručne ispravke. Postavlja `edited_by_human: true`.

    `autosave: true` je snimanje u pozadini: ne pravi `.bak` na svakih 30 s i
    ne stavlja prelamanje u red. Prazna rečenica se pri automatskom snimanju
    PRESKAČE (usred kucanja je normalno stanje), a pri ručnom je greška.
    """
    job_or_404(job_id)
    payload = await request.json()
    lang = payload.get("lang")
    autosave = bool(payload.get("autosave"))
    edits = {int(e["id"]): str(e["text"]).strip() for e in payload.get("sentences", [])}

    path = sentences_path(job_id, lang)
    if not path.exists():
        raise HTTPException(404, f"Nema rečenica za jezik {lang!r}.")
    sf = SentencesFile.model_validate(cfg.read_json(path))

    changed = 0
    skipped_empty = 0
    fixed_failed: list[int] = []
    for sent in sf.sentences:
        new = edits.get(sent.id)
        if new is None or new == sent.text:
            continue
        if not new:
            if autosave:
                skipped_empty += 1
                continue
            raise HTTPException(400, f"Rečenica {sent.id} je prazna — nije snimljeno.")
        sent.text = new
        changed += 1
        # Čovek je prepisao neprevedenu rečenicu — više nije original.
        if "translation_failed" in sent.flags:
            sent.flags = [f for f in sent.flags if f != "translation_failed"]
            fixed_failed.append(sent.id)

    made_backup = None
    if changed:
        made_backup = make_backup(path, forced=not autosave)
        sf.edited_by_human = True
        cfg.write_model(path, sf)
        kind = "automatsko snimanje" if autosave else "editor"
        cfg.log(job_id, f"{kind}: {changed} rečenica izmenjeno, edited_by_human=true")
        if fixed_failed:
            cfg.log(job_id, f"{kind}: translation_failed skinut ručnom izmenom: {fixed_failed}")
        if not autosave:
            db.add_event(job_id, "log", f"editor: {changed} rečenica izmenjeno")
    return JSONResponse({
        "changed": changed,
        "edited_by_human": sf.edited_by_human,
        "skipped_empty": skipped_empty,
        "backup": made_backup.name if made_backup else None,
        "saved_at": db.now(),
    })


# ==========================================================================
# Verzije: povratak na prethodno snimljeno
# ==========================================================================

#: Automatsko snimanje ne pravi novu `.bak` kopiju češće od ovoga.
#: Bez ovog praga bi snimanje na 30 s ostavljalo stotine kopija dnevno.
MIN_BACKUP_INTERVAL_SECONDS = 300.0

#: Koliko verzija se nudi u interfejsu. Starije ostaju na disku.
MAX_VERSIONS_SHOWN = 20


def backup_stamp(path: Path) -> tuple[str, int]:
    """(`20260812-101500`, redni broj u toj sekundi) iz imena kopije.

    Sortiranje po golom imenu ne radi otkad postoji brojač: `...-101500-2.bak`
    i `...-101500.bak` se razlikuju u znaku `-` (0x2D) naspram `.` (0x2E), pa
    bi novija kopija ispala starija.
    """
    stamp = path.name.rsplit(".", 2)[-2]
    when, counter = stamp[:15], stamp[15:].lstrip("-")
    return when, int(counter) if counter.isdigit() else 1


def backup_files(path: Path) -> list[Path]:
    """`.bak` kopije za dati fajl, najnovija prva."""
    return sorted(
        path.parent.glob(f"{path.name}.*.bak"),
        key=backup_stamp,
        reverse=True,
    )


def make_backup(path: Path, *, forced: bool) -> Path | None:
    """Napravi `.bak`, ali ne češće od `MIN_BACKUP_INTERVAL_SECONDS`.

    Ručno snimanje je čovekova tačka oslonca i uvek pravi kopiju. Automatsko
    pravi samo ako je poslednja dovoljno stara.
    """
    if not forced:
        existing = backup_files(path)
        if existing:
            try:
                age = time.time() - existing[0].stat().st_mtime
            except OSError:  # pragma: no cover
                age = MIN_BACKUP_INTERVAL_SECONDS + 1
            if age < MIN_BACKUP_INTERVAL_SECONDS:
                return None
    return cfg.backup(path)


def version_list(job_id: str, lang: str) -> list[dict[str, Any]]:
    """Snimljene verzije, najnovija prva, sa vremenom i brojem rečenica."""
    path = sentences_path(job_id, lang)
    out: list[dict[str, Any]] = []
    for bak in backup_files(path)[:MAX_VERSIONS_SHOWN]:
        # `sentences.sr.json.20260812-101500.bak`
        # `sentences.sr.json.20260812-101500.bak` ili `...-101500-2.bak`
        # (brojač za dva snimanja u istoj sekundi). Prvih 15 znakova je uvek
        # `%Y%m%d-%H%M%S`.
        stamp = bak.name.rsplit(".", 2)[-2]
        try:
            when = datetime.strptime(stamp[:15], "%Y%m%d-%H%M%S").replace(
                tzinfo=timezone.utc
            )
            when_text = when.astimezone().strftime("%d.%m. %H:%M:%S")
        except ValueError:  # pragma: no cover — ručno preimenovan fajl
            when_text = stamp
        try:
            data = cfg.read_json(bak)
            count = len(data.get("sentences", []))
        except Exception:  # noqa: BLE001 — pokvarena kopija se ne nudi
            continue
        out.append({
            "name": bak.name,
            "when": when_text,
            "sentences": count,
            "size_kb": round(bak.stat().st_size / 1024, 1),
        })
    return out


@app.get("/jobs/{job_id}/versions")
def versions(job_id: str, lang: str) -> Any:
    job_or_404(job_id)
    return JSONResponse({"versions": version_list(job_id, lang)})


@app.post("/jobs/{job_id}/restore")
async def restore_version(job_id: str, request: Request) -> Any:
    """Vrati `sentences.<lang>.json` na izabranu `.bak` verziju.

    Tekuće stanje se PRE toga snima kao nova kopija — povratak se i sam može
    poništiti. Bez toga bi „vrati" bilo isto tako nepovratno kao i brisanje
    koje ispravlja.
    """
    job_or_404(job_id)
    payload = await request.json()
    lang = str(payload.get("lang") or "")
    name = str(payload.get("name") or "")

    path = sentences_path(job_id, lang)
    if not path.exists():
        raise HTTPException(404, f"Nema rečenica za jezik {lang!r}.")

    chosen = next((b for b in backup_files(path) if b.name == name), None)
    if chosen is None:
        raise HTTPException(404, f"Nema verzije {name!r}.")

    # Provera pre pisanja: pokvarena kopija ne sme da pregazi ispravan fajl.
    try:
        restored = SentencesFile.model_validate(cfg.read_json(chosen))
    except Exception as exc:  # noqa: BLE001 — i loš JSON i loša šema
        raise HTTPException(
            400, f"Verzija {name} je neispravna i nije vraćena: {exc}"
        ) from exc

    cfg.backup(path)
    restored.edited_by_human = True
    cfg.write_model(path, restored)
    cfg.log(job_id, f"editor: vraćeno na verziju {chosen.name}")
    db.add_event(job_id, "log", f"vraćeno na verziju od {name.rsplit('.', 2)[-2]}")
    db.enqueue(job_id, "build", {"language": lang})
    return JSONResponse({"restored": chosen.name, "sentences": len(restored.sentences)})


# ==========================================================================
# Pokretanje faza — eksplicitno, nikad lančano
# ==========================================================================


@app.post("/jobs/{job_id}/translate")
def start_translate(
    job_id: str,
    to: str = Form(...),
    model: str = Form(""),
    pivot: str = Form(""),
    profile: str = Form(""),
    script: str = Form("Latn"),
    case: str = Form("normal"),
    punct: str = Form("keep"),
) -> Any:
    job_or_404(job_id)
    if db.pending_task(job_id):
        raise HTTPException(409, "Za ovaj projekat već postoji zadatak u redu.")
    if script not in ("Latn", "Cyrl"):
        raise HTTPException(400, "Pismo mora biti latinica ili ćirilica.")
    # Meni pismo za jednopismene jezike i ne prikazuje; ovo je brana za ručno
    # poslat obrazac. Ćirilica nad francuskim je kvar, ne izbor.
    if script == "Cyrl" and to not in BISCRIPTAL_LANGUAGES:
        raise HTTPException(400, f"Ćirilica nema smisla za ciljni jezik {to!r}.")
    if case not in CASES:
        raise HTTPException(400, f"Nepoznata veličina slova: {case!r}")
    if punct not in ("keep", "drop"):
        raise HTTPException(400, f"Nepoznat izbor interpunkcije: {punct!r}")
    db.enqueue(
        job_id, "translate",
        {
            "to": to, "model": model or None, "pivot": pivot or None,
            "profile": profile, "script": script, "case": case,
            "drop_punct": punct == "drop",
        },
    )
    db.add_event(job_id, "log", f"prevod na {to}/{script} stavljen u red")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/build")
def start_build(
    job_id: str,
    language: str = Form(...),
    profile: str = Form(""),
    script: str = Form("Latn"),
    case: str = Form("normal"),
    punct: str = Form("keep"),
) -> Any:
    job_or_404(job_id)
    if case not in CASES:
        raise HTTPException(400, f"Nepoznata veličina slova: {case!r}")
    if punct not in ("keep", "drop"):
        raise HTTPException(400, f"Nepoznat izbor interpunkcije: {punct!r}")
    db.enqueue(job_id, "build", {
        "language": language, "profile": profile, "script": script, "case": case,
        "drop_punct": punct == "drop",
    })
    db.add_event(job_id, "log", f"prelamanje {language}/{profile}/{script}/{case} u redu")
    # Zadatak u redu ostaje za CLI i stare forme; editor prelama sinhrono
    # (`/rebuild`, krug 13).
    return RedirectResponse(f"/jobs/{job_id}?lang={language}", status_code=303)


#: Ime koraka za dugme „Pokušaj ponovo" (title) i dnevnik.
TASK_LABELS = {
    "transcribe": "prepoznavanje", "translate": "prevod",
    "build": "prelamanje", "shorten": "skraćivanje",
}


@app.post("/jobs/{job_id}/rebuild", response_class=HTMLResponse)
def rebuild_now(
    request: Request,
    job_id: str,
    language: str = Form(...),
    profile: str = Form(""),
    script: str = Form("Latn"),
    case: str = Form("normal"),
    punct: str = Form("keep"),
) -> Any:
    """Prelomi ODMAH i vrati samo ono što se menja (krug 13).

    Prelamanje je determinističko, bez modela, oko sekunde — nema razloga da
    ide kroz red. Odgovor nosi delove sa `hx-swap-oob`: prelom ispod svake
    rečenice i „Za preuzimanje"; ostatak strane, skrol i fokus ostaju.
    Dok projekat ima zadatak u toku, ne prelama se: dva pisca istog
    `job.json` bi jedan drugom pojela istoriju.
    """
    from .. import pipeline

    job = dict(job_or_404(job_id))
    error = None
    if case not in CASES or punct not in ("keep", "drop") or script not in ("Latn", "Cyrl"):
        error = "Nepoznat izbor."
    elif db.pending_task(job_id) is not None:
        error = "Obrada je u toku — prelomi ponovo kad završi."
    else:
        try:
            pipeline.run_build(
                job_id, profile=profile or None, script=script, language=language,
                case=case, drop_punct=punct == "drop",
            )
        except SubtitisError as exc:
            error = str(exc).splitlines()[0]
        except Exception as exc:  # noqa: BLE001 — greška mora na ekran, ne u 500
            error = f"{type(exc).__name__}: {exc}"
    if error:
        db.add_event(job_id, "log", f"prelamanje nije uspelo: {error}")
    sf = load_sentences(job_id, language)
    preview = preview_by_sentence(job_id, language) if sf else {}
    srt_job = is_srt_job(job_id)
    invented = job_invented(job_id, job["language"])
    return templates.TemplateResponse(request, "_rebuild.html", {
        "oob": True,
        "error": error,
        "job": job,
        "active": language,
        "rows": [{"id": x.id, "preview": preview.get(x.id, [])} for x in (sf.sentences if sf else [])],
        "outputs": output_files(job_id, invented),
        "invented": invented,
        "built": last_build(job_id, language),
        "srt_job": srt_job,
        "disk": job_disk(job_id),
        "is_translation": bool(sf and sf.derived_from.startswith("sentences.")),
        "shorten": shorten_proposal(job_id, language) if srt_job else None,
        "pending": db.pending_task(job_id) is not None,
    })


@app.post("/jobs/{job_id}/retry")
def retry_failed(job_id: str) -> Any:
    """Ponovo pusti korak koji je pao, i ono što je zbog njega preskočeno.

    Projekat se vraća u stanje pre pada: bez prepisa na „u redu", sa prepisom
    na „prepis za korekturu". Inače bi worker zadatak palog projekta preskočio.
    """
    job = job_or_404(job_id)
    if db.pending_task(job_id):
        raise HTTPException(409, "Za ovaj projekat već postoji zadatak u redu.")
    tasks = db.retryable_tasks(job_id)
    if not tasks:
        raise HTTPException(400, "Nema koraka koji je pao.")
    has_transcript = sentences_path(job_id, job["language"]).exists()
    status = "transcribed" if tasks[0]["kind"] != "transcribe" and has_transcript else "uploaded"
    db.update_job(job_id, status=status, error=None)
    for t in tasks:
        db.enqueue(job_id, t["kind"], json.loads(t["params"] or "{}"))
    names = ", ".join(TASK_LABELS.get(t["kind"], t["kind"]) for t in tasks)
    db.add_event(job_id, "log", f"pokušaj ponovo: {names}")
    db.add_event(job_id, "status", status)
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/retranscribe")
def start_retranscribe(
    job_id: str,
    model: str = Form(cfg.DEFAULT_ASR_MODEL),
    vad_preset: str = Form(asr.DEFAULT_VAD_PRESET),
    terms: str | None = Form(None),
) -> Any:
    """Ponovi prepoznavanje — sa drugim filterom tišine ili terminima snimka.

    Ovo je jedino mesto gde se filter bira. `terms` (ako je poslato) ZAMENJUJE
    termine snimka u glosaru projekta; ostatak glosara projekta ostaje.
    """
    job = job_or_404(job_id)
    if db.pending_task(job_id):
        raise HTTPException(409, "Za ovaj projekat već postoji zadatak u redu.")
    if is_srt_job(job_id):
        raise HTTPException(409, "Projekat je nastao iz SRT-a — nema zvuka za transkripciju.")
    if vad_preset not in asr.VAD_PRESETS:
        raise HTTPException(400, f"Nepoznat izbor filtera tišine: {vad_preset!r}")
    sf = load_sentences(job_id, job["language"])
    if sf and sf.edited_by_human:
        raise HTTPException(
            409,
            "Tvoje ispravke prepisa su sačuvane, a ponovno prepoznavanje bi ih "
            "obrisalo. Napravi kopiju projekta ako ti stvarno treba.",
        )
    if terms is not None:
        gloss = {k: dict(v) for k, v in cfg.job_glossary(job_id).items()}
        key = cfg.RECORDING_TERMS_KEY.format(lang=job["language"])
        new = list(dict.fromkeys(t.strip() for t in terms.splitlines() if t.strip()))
        if new:
            gloss[key] = {t: t for t in new}
        else:
            gloss.pop(key, None)
        cfg.write_json(cfg.job_glossary_path(job_id), gloss)
    db.enqueue(job_id, "transcribe", {"model": model, "vad_preset": vad_preset})
    label = {v: l for v, l, _ in VAD_PRESET_LABELS}.get(vad_preset, vad_preset)
    db.add_event(job_id, "log", f"ponovno prepoznavanje u redu (filter tišine: {label})")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


# ==========================================================================
# Audio isečak za dugme „play"
# ==========================================================================


@app.get("/jobs/{job_id}/clip")
def clip(job_id: str, start: float = 0.0, end: float = 0.0) -> Any:
    """Iseci deo `source.wav`. Isecanje je jeftinije od slanja celog fajla."""
    job_or_404(job_id)
    wav = source_wav(job_id)
    if not wav.exists():
        raise HTTPException(
            404,
            f"Nema radne kopije zvuka — ili prepoznavanje još nije gotovo, ili je "
            f"obrisana posle {cfg.source_wav_ttl_days()} dana. "
            "Rečenice i SRT-ovi su netaknuti; za slušanje otpremi snimak ponovo.",
        )
    start = max(0.0, start)
    duration = min(max(0.05, end - start), MAX_CLIP_SECONDS)
    from ..audio import ffmpeg_path

    try:
        exe = ffmpeg_path()
    except SubtitisError as exc:
        raise HTTPException(500, str(exc)) from exc

    proc = subprocess.run(
        [exe, "-hide_banner", "-loglevel", "error", "-nostdin",
         "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(wav),
         "-ac", "1", "-ar", "16000", "-f", "wav", "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise HTTPException(500, f"ffmpeg: {proc.stderr.decode('utf-8', 'replace')[:300]}")
    return Response(content=proc.stdout, media_type="audio/wav")


# ==========================================================================
# Preuzimanje
# ==========================================================================


@app.get("/jobs/{job_id}/download/{name}")
def download(job_id: str, name: str) -> Any:
    job_or_404(job_id)
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "Neispravno ime datoteke.")
    path = cfg.job_dir(job_id) / "out" / name
    if not path.exists():
        raise HTTPException(404, f"Nema {name}")
    return FileResponse(path, filename=name, media_type="application/x-subrip")


# ==========================================================================
# SSE
# ==========================================================================


@app.get("/jobs/{job_id}/events")
async def stream_events(job_id: str, last_id: int = 0) -> Any:
    job_or_404(job_id)

    async def gen():
        cursor = last_id
        idle = 0
        while True:
            rows = db.events_since(job_id, cursor)
            for r in rows:
                cursor = r["id"]
                payload = json.dumps(
                    {
                        "id": r["id"], "kind": r["kind"], "message": r["message"],
                        "progress": r["progress"], "ts": r["ts"],
                    },
                    ensure_ascii=False,
                )
                yield f"event: {r['kind']}\ndata: {payload}\n\n"
            if rows:
                idle = 0
            else:
                idle += 1
                if idle % 20 == 0:  # ping da posrednici ne zatvore vezu
                    yield ": ping\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


#: Šta `/api/identity` vraća u polju `app`. Po tome `desktop.is_subtitis`
#: razlikuje Subtitis od tuđe aplikacije na istom portu.
IDENTITY = "subtitis"


#: Ikonica u kartici pregledača: ISTA datoteka kao prečica i sistemska traka
#: (`scripts/make_icon.py`, boje iz `app.css`) — ne kopija.
ICON_FILE = HERE / "static" / "subtitis.ico"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Any:
    """Pregledač traži baš ovu putanju kad u HTML-u nema oznake; `base.html`
    je navodi sa `?v=<verzija>`, pa nova verzija zaobilazi keš."""
    return FileResponse(
        ICON_FILE, media_type="image/x-icon",
        headers={"Cache-Control": "public, max-age=604800"},
    )


@app.get("/api/identity")
def identity() -> Any:
    """Ko sluša na ovom portu. Otvoren port sam po sebi ne znači Subtitis."""
    from .. import __version__

    return {"app": IDENTITY, "version": __version__, "jobs_dir": str(cfg.JOBS_DIR)}


@app.get("/api/jobs/{job_id}/status")
def job_status(job_id: str) -> Any:
    job = job_or_404(job_id)
    return {
        "status": job["status"],
        "error": job["error"],
        "pending": db.pending_task(job_id) is not None,
        "outputs": output_files(job_id),
        "languages": available_languages(job_id),
    }


# ==========================================================================
# Skraćivanje prekoračenja: predlog -> pregled -> upis izabranog
# ==========================================================================

#: Flag na rečenici čiji je skraćen tekst čovek prihvatio u editoru.
SHORTENED_FLAG = "shortened"


def diff_html(old: str, new: str) -> tuple[str, str]:
    """Stara i nova verzija sa izbačenim (<del>) i dodatim (<ins>) rečima."""
    import difflib

    a, b = old.split(), new.split()
    left: list[str] = []
    right: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            left.append(html.escape(" ".join(a[i1:i2])))
            right.append(html.escape(" ".join(b[j1:j2])))
            continue
        if i2 > i1:
            left.append(f"<del>{html.escape(' '.join(a[i1:i2]))}</del>")
        if j2 > j1:
            right.append(f"<ins>{html.escape(' '.join(b[j1:j2]))}</ins>")
    return " ".join(left), " ".join(right)


def shorten_proposal(job_id: str, lang: str | None) -> dict[str, Any] | None:
    """Predlog skraćivanja za jezik, spreman za prikaz; None ako ga nema."""
    from ..pipeline import shorten_path

    if not lang:
        return None
    path = shorten_path(job_id, lang)
    if not path.exists():
        return None
    try:
        data = cfg.read_json(path)
    except Exception:  # noqa: BLE001 — pokvaren predlog ne sme da obori stranicu
        return None
    for it in data.get("items", []):
        if it.get("new"):
            it["old_html"], it["new_html"] = diff_html(it["old"], it["new"])
        else:
            it["old_html"], it["new_html"] = html.escape(it["old"]), ""
    return data


@app.post("/jobs/{job_id}/shorten")
def start_shorten(job_id: str, language: str = Form(...)) -> Any:
    """Drugi prolaz kroz model za cue-ove gore od originala. Samo na dugme."""
    job_or_404(job_id)
    if db.pending_task(job_id):
        raise HTTPException(409, "Za ovaj projekat već postoji zadatak u redu.")
    if not is_srt_job(job_id):
        raise HTTPException(400, "Skraćivanje postoji samo za projekat nastao iz SRT-a.")
    sf = load_sentences(job_id, language)
    if sf is None or not sf.derived_from.startswith("sentences."):
        raise HTTPException(400, f"{language} nije prevod — nema šta da se skraćuje.")
    db.enqueue(job_id, "shorten", {"language": language})
    db.add_event(job_id, "log", f"skraćivanje prekoračenja ({language}) stavljeno u red")
    return RedirectResponse(f"/jobs/{job_id}?lang={language}", status_code=303)


@app.post("/jobs/{job_id}/shorten/apply")
async def apply_shorten(job_id: str, request: Request) -> Any:
    """Upiši SAMO izabrane skraćene verzije. Ostatak predloga se odbacuje.

    Rečenica koju je čovek u međuvremenu menjao se preskače — predlog je
    napravljen nad starim tekstom i upisao bi se preko ispravke.
    """
    from ..pipeline import shorten_path

    job_or_404(job_id)
    payload = await request.json()
    lang = str(payload.get("lang") or "")
    chosen = {int(i) for i in payload.get("ids", [])}
    proposal = shorten_proposal(job_id, lang)
    if proposal is None:
        raise HTTPException(404, f"Nema predloga skraćivanja za {lang!r}.")
    path = sentences_path(job_id, lang)
    sf = SentencesFile.model_validate(cfg.read_json(path))
    by_id = {s.id: s for s in sf.sentences}

    applied: list[int] = []
    stale: list[int] = []
    for it in proposal["items"]:
        sid = it["sentence"]
        if sid not in chosen or it.get("status") != "accepted":
            continue
        sent = by_id.get(sid)
        if sent is None or sent.text != it["old"]:
            stale.append(sid)
            continue
        sent.text = it["new"]
        if SHORTENED_FLAG not in sent.flags:
            sent.flags.append(SHORTENED_FLAG)
        applied.append(sid)

    offered = sum(1 for it in proposal["items"] if it.get("status") == "accepted")
    if applied:
        make_backup(path, forced=True)
        sf.edited_by_human = True
        cfg.write_model(path, sf)
    # Predlog se čuva uz odluku, pa se ne pravi ponovo nad istim tekstom.
    done = path.with_name(f"shorten.{lang}.applied.json")
    proposal_path = shorten_path(job_id, lang)
    data = cfg.read_json(proposal_path)
    data["applied"], data["stale"] = applied, stale
    data["declined"] = [
        it["sentence"] for it in data["items"]
        if it.get("status") == "accepted" and it["sentence"] not in applied
        and it["sentence"] not in stale
    ]
    cfg.write_json(done, data)
    proposal_path.unlink(missing_ok=True)

    msg = (
        f"skraćivanje: upisano {len(applied)} od {offered} predloženih"
        + (f", preskočeno {len(stale)} (tekst menjan posle predloga)" if stale else "")
        + f", odbijeno ručno {len(data['declined'])}"
    )
    cfg.log(job_id, msg)
    db.add_event(job_id, "log", msg)
    if applied:
        db.enqueue(job_id, "build", {"language": lang})
    return JSONResponse({"applied": applied, "stale": stale, "declined": data["declined"]})


@app.post("/jobs/{job_id}/shorten/discard")
async def discard_shorten(job_id: str, request: Request) -> Any:
    from ..pipeline import shorten_path

    job_or_404(job_id)
    lang = str((await request.json()).get("lang") or "")
    path = shorten_path(job_id, lang)
    if not path.exists():
        raise HTTPException(404, f"Nema predloga skraćivanja za {lang!r}.")
    path.unlink()
    cfg.log(job_id, f"skraćivanje: predlog za {lang} odbačen ceo")
    db.add_event(job_id, "log", f"skraćivanje: predlog za {lang} odbačen ceo")
    return JSONResponse({"discarded": True})
