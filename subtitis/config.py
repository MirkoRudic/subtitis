"""Putanje, učitavanje profila/glosara, logovanje po poslu.

Sve izlazne JSON/tekst datoteke pišemo kao UTF-8 bez BOM-a.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import JOB_SCHEMA_VERSION, Job, PhaseRecord, Profile

# --- Putanje --------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
PROFILES_DIR = ROOT_DIR / "profiles"
GLOSSARY_DIR = ROOT_DIR / "glossary"
JOBS_DIR = Path(os.environ.get("SUBTITIS_JOBS_DIR", ROOT_DIR / "jobs"))

# --- Podrazumevane vrednosti ----------------------------------------------

DEFAULT_PROFILE = "tv"

#: Verzija imena profila u `subtitis.toml` (`default_profile`). Krug 8 je
#: promenio značenje imena `web`, pa vrednost bez ove oznake ne može da se
#: protumači — vidi `migrate.py`. Upisuje je svako snimanje podešavanja.
PROFILES_VERSION = 2
DEFAULT_ASR_MODEL = "large-v3"

#: Jezici za koje se ijekavica automatski prevodi u ekavicu (`force_ekavica`).
#: Whisper provuče ijekavicu i uz `language="sr"`.
FORCE_EKAVICA_LANGUAGES: tuple[str, ...] = ("sr",)


def force_ekavica_default(language: str) -> bool:
    return language in FORCE_EKAVICA_LANGUAGES


#: Pusti LLM restauraciju interpunkcije SAM, kad dijagnostika pokaže da ASR
#: nije vratio znakove (ili ih je posuo po svakoj reči). Bez interpunkcije su i
#: segmentacija i prelamanje loši, pa je ovo podrazumevano uključeno.
#: Isključuje se sa `--no-punctuate` ili `auto_punctuate = false` u subtitis.toml.
AUTO_PUNCTUATE = True


def auto_punctuate() -> bool:
    val = setting("auto_punctuate", env="SUBTITIS_AUTO_PUNCTUATE", default=AUTO_PUNCTUATE)
    if isinstance(val, str):
        return val.strip().lower() not in ("0", "false", "no", "off")
    return bool(val)


#: Pivot jezik po jezičkom paru: `"sr->fr": "en"` znači sr -> en -> fr.
#: `None` (ili odsustvo ključa) znači direktan prevod.
#:
#: NAMERNO PRAZNO. Za `sr->fr` je dvostruki prolaz preko engleskog često bolji
#: jer je `en<->fr` mnogo bolje pokriveno, ali to treba IZMERITI na svom
#: materijalu (`scripts/eval_translation.py`), a ne pretpostaviti.
PIVOT_VIA: dict[str, str | None] = {}


#: Pravopisna varijanta ciljnog jezika. Ide u prevodni prompt kao pravilo 13.
#: Postavljeno na `en-GB` jer su reference u `eval/pairs.json` britanske
#: (`socialise`, `Centre`), a model podrazumevano piše američki. Isprazni ovaj
#: rečnik ako ne želiš nikakvo pravilo o pravopisu.
TARGET_VARIANT: dict[str, str] = {"en": "en-GB"}


def target_variant(tgt_lang: str) -> str | None:
    return TARGET_VARIANT.get(tgt_lang)


def translation_pivot() -> str:
    """Pivot za sve projekte iz podešavanja: `""` (direktno) ili `"en"`.

    Bira se jednom, u podešavanjima — ne po projektu (krug 10). `PIVOT_VIA`
    po paru ima prednost kad je popunjen.
    """
    val = str(setting("translation_pivot", default="")).strip().lower()
    return val if val in ("", "en") else ""


def resolve_pivot(src_lang: str, tgt_lang: str) -> str | None:
    """Vrati pivot jezik za par, ili `None` za direktan prevod."""
    pivot = PIVOT_VIA.get(f"{src_lang}->{tgt_lang}") or translation_pivot() or None
    if pivot in (None, "", src_lang, tgt_lang):
        return None
    return pivot


DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
#: Izmereno na 9 parova (`scripts/eval_translation.py`): gemma3:12b daje
#: podjednak ili bolji kvalitet, 36.9 naspram 34.0 tokena/s, 2 GB manje VRAM-a
#: i sažetiji izlaz (1.03x naspram 1.06x u odnosu na referencu).
#: qwen3:14b ostaje dostupan preko `--model`.
DEFAULT_LLM_MODEL = os.environ.get("SUBTITIS_LLM_MODEL", "gemma3:12b")


class SubtitisError(RuntimeError):
    """Greška koju treba prikazati korisniku kao jasnu poruku, bez traceback-a."""


class JobCancelled(Exception):
    """Posao je obrisan dok je njegov zadatak radio.

    NAMERNO nije `SubtitisError` (ni `RuntimeError`): pajplajn na više mesta
    hvata `SubtitisError` da bi nastavio (restauracija interpunkcije je
    „poboljšanje, ne uslov"), a ASR petlja hvata `RuntimeError`. Prekid ne sme
    da bude progutan ni na jednom od tih mesta.
    """


# --- Prekid zadatka kad je posao obrisan -----------------------------------
#
# Worker postavi proveru pre zadatka; kod je poziva na prirodnim tačkama —
# pre svakog poziva LLM-u i posle svakog ASR segmenta — i nikad usred poziva
# modelu. Nit-lokalno, jer `serve` vrti workera kao nit uz HTTP sloj.

_cancel = threading.local()


@contextmanager
def cancellation(check) -> Iterator[None]:
    """Za trajanje bloka `checkpoint()` pita `check()` da li treba stati."""
    prev = getattr(_cancel, "check", None)
    _cancel.check = check
    try:
        yield
    finally:
        _cancel.check = prev


def checkpoint() -> None:
    """Prirodna tačka prekida: ako je posao obrisan, podigni `JobCancelled`."""
    check = getattr(_cancel, "check", None)
    if check is not None and check():
        raise JobCancelled()


# --- subtitis.toml --------------------------------------------------------

CONFIG_FILE = Path(os.environ.get("SUBTITIS_CONFIG", ROOT_DIR / "subtitis.toml"))

_settings_cache: dict[str, Any] | None = None


def load_settings(reload: bool = False) -> dict[str, Any]:
    """Učitaj `subtitis.toml`. Nedostatak fajla nije greška."""
    global _settings_cache
    if _settings_cache is not None and not reload:
        return _settings_cache
    data: dict[str, Any] = {}
    if CONFIG_FILE.exists():
        import tomllib

        try:
            with CONFIG_FILE.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception as exc:  # noqa: BLE001 — loš TOML mora da se vidi
            raise SubtitisError(
                f"{CONFIG_FILE} nije ispravan TOML: {exc}"
            ) from exc
    _settings_cache = data
    return data


def setting(name: str, env: str | None = None, default: Any = None) -> Any:
    """Vrednost podešavanja: promenljiva okruženja pa `subtitis.toml`."""
    if env:
        val = os.environ.get(env)
        if val:
            return val
    return load_settings().get(name, default)


# --- Podešavanja koja dashboard sme da menja -------------------------------
#
# Svako od ovih ima pristupnu funkciju, a ne golu konstantu — inače bi vrednost
# iz `subtitis.toml` bila pročitana jednom pri uvozu i izmena kroz ekran ne bi
# imala efekta do ponovnog pokretanja.


def default_profile() -> str:
    return str(setting("default_profile", env="SUBTITIS_PROFILE", default=DEFAULT_PROFILE))


def new_project_glossary() -> str:
    """Šta nov projekat dobija kao glosar: `default` (kopija podrazumevanog)
    ili `none`. Bira se u podešavanjima, ne na početnoj strani."""
    val = str(setting("new_project_glossary", default="default"))
    return val if val in ("default", "none") else "default"


def default_target_language() -> str:
    return str(setting("default_target_language", default="en"))


def llm_model() -> str:
    return str(setting("llm_model", env="SUBTITIS_LLM_MODEL", default=DEFAULT_LLM_MODEL))


#: 8000 drže mnoge razvojne alatke; 8010 je ređe zauzet. Menja se ključem
#: `port` u `subtitis.toml` ili sa SUBTITIS_PORT, bez izmene koda.
DEFAULT_PORT = 8010


def port() -> int:
    """Port dashboarda. Ako je zauzet tuđom aplikacijom, `desktop.choose_port`
    ide na sledeći slobodan."""
    try:
        return int(setting("port", env="SUBTITIS_PORT", default=DEFAULT_PORT))
    except (TypeError, ValueError):
        return DEFAULT_PORT


def source_wav_ttl_days() -> int:
    try:
        return int(setting("source_wav_ttl_days", default=SOURCE_WAV_TTL_DAYS))
    except (TypeError, ValueError):
        return SOURCE_WAV_TTL_DAYS


#: Ključevi koje ekran sa podešavanjima sme da upiše, sa tipom.
EDITABLE_SETTINGS: dict[str, type] = {
    "default_profile": str,
    "default_target_language": str,
    "llm_model": str,
    "source_wav_ttl_days": int,
    "auto_punctuate": bool,
    "profiles_version": int,
    "new_project_glossary": str,
    "translation_pivot": str,
}


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def save_settings(values: dict[str, Any]) -> Path:
    """Upiši podešavanja u `subtitis.toml`, čuvajući nepoznate ključeve.

    Pisac je namerno prost — pokriva samo skalarne vrednosti iz
    `EDITABLE_SETTINGS`. **Komentari u ručno pisanom fajlu se gube**; fajl je od
    trenutka kad se ekran upotrebi mašinski, ne ručno održavan.
    """
    merged = dict(load_settings())
    for key, value in values.items():
        if key not in EDITABLE_SETTINGS:
            raise SubtitisError(f"Nepoznato podešavanje: {key!r}")
        merged[key] = value
    # ime profila upisano sada je po današnjim imenima (`migrate.py`)
    merged["profiles_version"] = PROFILES_VERSION

    lines = [
        "# Podešavanja Subtitisa. Piše ih ekran „Podešavanja“ u dashboardu.",
        "# Promenljive okruženja imaju prednost nad ovim vrednostima.",
        "",
    ]
    for key in sorted(merged):
        lines.append(f"{key} = {_toml_value(merged[key])}")
    write_text_atomic(CONFIG_FILE, "\n".join(lines) + "\n")
    load_settings(reload=True)
    return CONFIG_FILE


# --- JSON I/O -------------------------------------------------------------


def read_json(path: Path) -> Any:
    if not path.exists():
        raise SubtitisError(f"Nedostaje datoteka: {path}")
    # utf-8-sig, a ne utf-8: ove fajlove ljudi otvaraju ručno, a Notepad i
    # PowerShell `Out-File -Encoding utf8` umeju da ostave BOM.
    with path.open("r", encoding="utf-8-sig") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            raise SubtitisError(
                f"{path} nije ispravan JSON: {exc.msg} (red {exc.lineno}, kolona {exc.colno}).\n"
                "Ako si fajl menjao ručno, proveri zareze i navodnike."
            ) from exc


#: Pauze između pokušaja zamene fajla (krug 11). Na Windowsu `os.replace`
#: pada sa `PermissionError` dok je ciljni fajl otvoren bilo gde — čitanje
#: drži fajl milisekundama, pa je ponavljanje standardno rešenje, ne
#: zaobilaženje. Posle poslednje pauze greška ide dalje.
REPLACE_RETRY_DELAYS: tuple[float, ...] = (0.02, 0.04, 0.08, 0.16, 0.32)


def replace_file(tmp: Path, path: Path) -> None:
    """`tmp` -> `path` atomično, uz ponavljanje dok je `path` zauzet."""
    import time

    for delay in (*REPLACE_RETRY_DELAYS, None):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def write_text_atomic(path: Path, text: str, *, newline: str | None = "\n") -> None:
    """Upiši tekst preko postojećeg fajla: prvo u `.tmp`, pa zamena.

    Čitalac nikad ne vidi pola fajla, a zauzet cilj ne obara upis
    (`replace_file`). Svako pisanje PREKO postojećeg fajla ide ovuda.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)
    replace_file(tmp, path)


def write_json(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_model(path: Path, model: Any) -> None:
    write_json(path, json.loads(model.model_dump_json()))


def backup(path: Path) -> Path | None:
    """Napravi `.bak` kopiju pre prepisivanja. Vrati putanju kopije.

    Vremenska oznaka ima rezoluciju od sekunde, pa dva snimanja u istoj sekundi
    daju isto ime — bez brojača bi drugo tiho pregazilo prvo i verzija bi
    nestala. Brojač se dodaje NA KRAJ oznake, pa prvih 15 znakova ostaje
    `%Y%m%d-%H%M%S` i za stare i za nove kopije.
    """
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dst = path.with_suffix(path.suffix + f".{stamp}.bak")
    counter = 2
    while dst.exists():
        dst = path.with_suffix(path.suffix + f".{stamp}-{counter}.bak")
        counter += 1
    dst.write_bytes(path.read_bytes())
    return dst


# --- Profili i glosar -----------------------------------------------------


def load_profile(name_or_path: str) -> Profile:
    candidate = Path(name_or_path)
    if not candidate.exists():
        candidate = PROFILES_DIR / f"{name_or_path}.json"
    if not candidate.exists():
        available = sorted(p.stem for p in PROFILES_DIR.glob("*.json"))
        raise SubtitisError(
            f"Nepoznat profil '{name_or_path}'. Dostupni: {', '.join(available) or '(nijedan)'}"
        )
    return Profile.model_validate(read_json(candidate))


def glossary_path(path: str | Path | None = None) -> Path:
    return Path(path) if path else GLOSSARY_DIR / "default.json"


def load_glossary(path: str | Path | None = None) -> dict[str, dict[str, str]]:
    p = glossary_path(path)
    if not p.exists():
        return {}
    return read_json(p)


def save_glossary(
    data: dict[str, dict[str, str]], path: str | Path | None = None
) -> Path:
    """Upiši glosar, uz `.bak` kopiju prethodnog.

    Glosar je ručno kuriran i ulazi u ASR (`hotwords`) i u prevodni prompt —
    slučajno brisanje kroz ekran mora da se može vratiti.
    """
    p = glossary_path(path)
    backup(p)
    write_json(p, {k: dict(v) for k, v in data.items() if v})
    return p


# --- Glosar posla -----------------------------------------------------------
#
# Svaki posao ima svoj glosar (`jobs/<id>/glossary.json`), kopiran iz
# podrazumevanog pri otvaranju. Globalni glosar je ulazio u SVAKI posao, pa je
# snimak sa džinglom (Radionica draft 1, 2026-10-01) dobio poslednjih 7 s prepisa iz
# glosara prethodnog projekta — `hotwords` ide u svaki prozor, a na muzici
# model vrati ono što mu je dato. Prazan glosar je legitiman izbor.

JOB_GLOSSARY = "glossary.json"

#: Ključ pod kojim stoje termini snimka: samo za prepoznavanje (`hotwords`),
#: bez prevoda. `glossary_keys` ih vidi jer je izvorni jezik isti; prevodni
#: smerovi (`sr->en`) ih ne vide.
RECORDING_TERMS_KEY = "{lang}"


def job_glossary_path(job_id: str) -> Path:
    return job_dir(job_id) / JOB_GLOSSARY


def job_glossary(job_id: str) -> dict[str, dict[str, str]]:
    """Glosar posla. Posao napravljen pre glosara po poslu nema fajl i čita
    podrazumevani, kao i do sada. Prazan fajl je prazan glosar, ne „nema fajla"."""
    p = job_glossary_path(job_id)
    if p.exists():
        return read_json(p)
    return load_glossary()


def make_job_glossary(
    language: str, *, use_default: bool, recording_terms: list[str]
) -> dict[str, dict[str, str]]:
    """Glosar za nov posao: kopija podrazumevanog (ili ništa) + termini snimka."""
    gloss = {k: dict(v) for k, v in load_glossary().items()} if use_default else {}
    terms = list(dict.fromkeys(t.strip() for t in recording_terms if t.strip()))
    if terms:
        key = RECORDING_TERMS_KEY.format(lang=language)
        gloss.setdefault(key, {}).update({t: t for t in terms})
    return gloss


def recording_terms(glossary: dict[str, dict[str, str]], language: str) -> list[str]:
    return list(glossary.get(RECORDING_TERMS_KEY.format(lang=language), {}))


def glossary_pairs(
    glossary: dict[str, dict[str, str]], src_lang: str, tgt_lang: str
) -> dict[str, str]:
    """Parovi za zadati smer; idu celi u prevodni prompt."""
    return dict(glossary.get(f"{src_lang}->{tgt_lang}", {}))


def derived_glossary_pairs(
    glossary: dict[str, dict[str, str]], origin: str, pivot: str, target: str
) -> dict[str, str]:
    """Izvedi `pivot->target` parove iz `origin->pivot` i `origin->target`.

    Kod pivota `sr -> en -> fr` drugi skok traži `en->fr` unose, kojih u glosaru
    po pravilu nema — pa bi druga polovina puta išla bez rečnika i termin
    ugovoren za `sr->fr` bi se izgubio. Ako za isti izvorni termin postoje oba
    prevoda, iz njih se izvodi `en->fr` par.
    """
    to_pivot = glossary.get(f"{origin}->{pivot}", {})
    to_target = glossary.get(f"{origin}->{target}", {})
    return {
        to_pivot[term]: to_target[term]
        for term in to_pivot.keys() & to_target.keys()
        if to_pivot[term] and to_target[term]
    }


def hop_glossary(
    glossary: dict[str, dict[str, str]],
    hop_src: str,
    hop_tgt: str,
    *,
    origin: str | None = None,
) -> tuple[dict[str, str], int]:
    """Rečnik za jedan skok. Vrati (parovi, koliko ih je izvedeno).

    Eksplicitni `hop_src->hop_tgt` unosi imaju prednost nad izvedenima.
    """
    direct = glossary_pairs(glossary, hop_src, hop_tgt)
    if origin is None or origin == hop_src:
        return direct, 0
    derived = derived_glossary_pairs(glossary, origin, hop_src, hop_tgt)
    merged = {**derived, **direct}
    return merged, len(set(derived) - set(direct))


def glossary_keys(glossary: dict[str, dict[str, str]], src_lang: str) -> list[str]:
    """Ključevi svih parova gde je `src_lang` izvorni — idu u Whisperov initial_prompt."""
    keys: list[str] = []
    for pair, mapping in glossary.items():
        if pair.split("->")[0].strip() == src_lang:
            keys.extend(mapping.keys())
    # bez duplikata, stabilan redosled
    return list(dict.fromkeys(keys))


def glossary_terms(glossary: dict[str, dict[str, str]], language: str) -> list[str]:
    """Svi termini glosara napisani na `language`: ključevi `language->*` i
    prevodi `*->language`. Prelamanje ih ne seče (`linebreak`)."""
    terms: list[str] = []
    for pair, mapping in glossary.items():
        src, _, tgt = pair.partition("->")
        if src.strip() == language:
            terms.extend(mapping.keys())
        if tgt.strip() == language:
            terms.extend(mapping.values())
    return list(dict.fromkeys(terms))


# --- Poslovi --------------------------------------------------------------

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def job_dir(job_id: str) -> Path:
    if not _JOB_ID_RE.match(job_id):
        raise SubtitisError(
            f"Neispravan job_id '{job_id}': dozvoljena su samo slova, cifre, '.', '_' i '-'."
        )
    return JOBS_DIR / job_id


def make_job_id(source: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", source.stem).strip("-") or "job"
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{stem}"


def load_job(job_id: str) -> Job:
    data = read_json(job_dir(job_id) / "job.json")
    if int(data.get("schema_version") or 1) < JOB_SCHEMA_VERSION:
        # posao sa starim imenima profila (pre kruga 8) — vidi `migrate.py`
        from .migrate import migrate_job

        if migrate_job(job_dir(job_id), warn=False).status == "migrated":
            data = read_json(job_dir(job_id) / "job.json")
    return Job.model_validate(data)


def save_job(job: Job) -> None:
    job.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_model(job_dir(job.job_id) / "job.json", job)


# --- Čišćenje diska --------------------------------------------------------

#: Posle ovoliko dana se radni `source.wav` briše sam.
#:
#: On je najveća stavka na disku (~115 MB po satu audija) a potreban je samo
#: dok traje korektura — iz njega dugme ▶ seče isečke. Sve što je rezultat
#: (`words.json`, `sentences.json`, SRT) ostaje netaknuto.
#:
#: `0` isključuje čišćenje.
SOURCE_WAV_TTL_DAYS = 7


def purge_old_sources(ttl_days: int | None = None) -> list[tuple[str, int]]:
    """Obriši `source.wav` poslova starijih od `ttl_days`.

    Vrati [(job_id, oslobođeno bajtova)]. Beleži se u log svakog posla, da se
    posle ne pitaš gde je nestao WAV kad dugme ▶ prestane da radi.
    """
    ttl = source_wav_ttl_days() if ttl_days is None else ttl_days
    if ttl <= 0 or not JOBS_DIR.exists():
        return []

    cutoff = time.time() - ttl * 86400.0
    freed: list[tuple[str, int]] = []
    for wav in sorted(JOBS_DIR.glob("*/source.wav")):
        try:
            stat = wav.stat()
            if stat.st_mtime >= cutoff:
                continue
            size = stat.st_size
            wav.unlink()
        except OSError:  # pragma: no cover — fajl zaključan ili već obrisan
            continue
        job_id = wav.parent.name
        freed.append((job_id, size))
        log(
            job_id,
            f"čišćenje: source.wav obrisan posle {ttl} dana "
            f"({size / 1024 / 1024:.0f} MB). Slušanje isečaka više ne radi; "
            "rečenice, JSON-ovi i SRT-ovi su netaknuti.",
        )
    return freed


# --- Log ------------------------------------------------------------------


def echo(line: str) -> None:
    """Ispiši red na konzolu tako da ispis NIKAD ne obori posao.

    Windows konzola je podrazumevano cp1252, pa `print("vraćeno")` baci
    `UnicodeEncodeError` i sruši ceo HTTP zahtev usred snimanja. Log fajl je
    UTF-8 i uvek je tačan; konzola je pomoćni ispis i sme da izgubi kvačicu, ali
    ne sme da obori rad.

    Drugi slučaj je `pythonw.exe` (desktop režim), gde `stdout` ne postoji
    uopšte. Njega rešava `bootstrap.ensure_streams()` na ulaznoj tački — ali
    provera `is None` ostaje i ovde, jer se `echo` zove i iz koda koji nije
    prošao kroz našu ulaznu tačku (ugrađen server, testovi, `python -c`).
    """
    import sys

    stream = sys.stdout
    if stream is None:  # pragma: no cover — neke GUI ljuske nemaju stdout
        return
    try:
        stream.write(line + "\n")
        stream.flush()
        return
    except UnicodeEncodeError:
        pass
    except (ValueError, OSError):  # pragma: no cover — zatvoren stream
        return
    try:
        encoding = getattr(stream, "encoding", None) or "ascii"
        stream.write(line.encode(encoding, "replace").decode(encoding) + "\n")
        stream.flush()
    except Exception:  # noqa: BLE001 — ispis ne sme da obori posao ni ovde
        pass


def log(job_id: str, message: str) -> None:
    """Dopiši red u jobs/<id>/log.txt i odštampaj ga."""
    d = job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    with (d / "log.txt").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    echo(line)


@contextmanager
def phase(job: Job, name: str, **detail: Any) -> Iterator[dict[str, Any]]:
    """Meri trajanje faze, upisuje ga u log i u job.history."""
    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    log(job.job_id, f"FAZA {name}: start {detail or ''}".rstrip())
    extra: dict[str, Any] = dict(detail)
    status = "ok"
    try:
        yield extra
    except BaseException as exc:  # noqa: BLE001 — beležimo pa propuštamo dalje
        status = "error"
        extra["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        dt = time.perf_counter() - t0
        job.history.append(
            PhaseRecord(
                phase=name,
                status=status,  # type: ignore[arg-type]
                started_at=started.isoformat(timespec="seconds"),
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                duration_s=round(dt, 3),
                detail=extra,
            )
        )
        save_job(job)
        log(job.job_id, f"FAZA {name}: {status} za {dt:.1f}s {extra}".rstrip())
