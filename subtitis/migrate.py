"""Migracija imena profila u krugu 8 (`job.json` verzija 1 -> 2).

Krug 8 je promenio ZNAČENJE imena `web`: do tada 42 znaka u dva reda, od tada
jedan red do 20 znakova. Pseudonim to ne može da reši — ne razlikuje stari
`web` od novog — pa se stara imena jednom prepisuju na disku:

    netflix -> tv, tv -> tv, web -> tv (staro značenje), digital -> web

Dira se samo posao čiji `job.json` ima `schema_version` manji od 2; posao koji
je već na novoj verziji se preskače. Ako posao nosi ime koje nije ni staro ni
izvorno (`original`, `original-max`), ne pogađa se: posao ostaje kakav je, a u
njegov dnevnik ide upozorenje.

Mesta na kojima stoji ime profila:
  - `job.json`: `history[].detail.profile` i `detail.profile_suggestions[].profile`
  - `shorten.*.json`: ključ `profile`
  - zadaci u redu (baza dashboarda) — preko `on_migrated`
  - `subtitis.toml`: `default_profile` (oznaka `profiles_version`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config as cfg
from .models import JOB_SCHEMA_VERSION

RENAMES: dict[str, str] = {"netflix": "tv", "tv": "tv", "web": "tv", "digital": "web"}

#: Profili izvedeni iz ulaznog SRT-a — ime im se ne menja.
KEEP: frozenset[str] = frozenset({"original", "original-max"})

WHY = (
    "krug 8: `web` sada znači jedan red do 20 znakova; stari `web` i `netflix` "
    "(42 zn., dva reda) zovu se `tv`, a stari `digital` (jednoredni) zove se `web`"
)


@dataclass
class JobMigration:
    job_id: str
    #: `migrated`, `current` (već na novoj verziji) ili `skipped` (nepoznato ime)
    status: str
    #: (gde, staro, novo) za svako promenjeno ime
    changes: list[tuple[str, str, str]] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)


def _rename(name: Any) -> Any:
    return RENAMES.get(name, name) if isinstance(name, str) else name


def _profile_slots(job: dict[str, Any], shorten: dict[Path, dict[str, Any]]):
    """(gde, kontejner, ključ) za svako mesto koje nosi ime profila."""
    for k, rec in enumerate(job.get("history") or []):
        det = rec.get("detail") or {}
        where = f"{rec.get('phase', '?')} #{k}"
        if "profile" in det:
            yield where, det, "profile"
        for sug in det.get("profile_suggestions") or []:
            if isinstance(sug, dict) and "profile" in sug:
                yield f"{where} predlog", sug, "profile"
    for path, data in shorten.items():
        if "profile" in data:
            yield path.name, data, "profile"


def migrate_job(
    jdir: Path, on_migrated: Callable[[str], None] | None = None, *, warn: bool = True,
) -> JobMigration:
    """Prevedi imena profila jednog posla. Idempotentno.

    `warn=False` je za lenju migraciju pri učitavanju posla: preskočen posao
    se ne prijavljuje pri svakom otvaranju, nego jednom, pri pregledu svih.
    """
    job_path = jdir / "job.json"
    job = cfg.read_json(job_path)
    job_id = jdir.name
    if int(job.get("schema_version") or 1) >= JOB_SCHEMA_VERSION:
        return JobMigration(job_id, "current")

    shorten = {p: cfg.read_json(p) for p in sorted(jdir.glob("shorten.*.json"))}
    slots = list(_profile_slots(job, shorten))
    unknown = sorted({
        c[key] for _, c, key in slots
        if c[key] and c[key] not in RENAMES and c[key] not in KEEP
    })
    if unknown:
        if not warn:
            return JobMigration(job_id, "skipped", unknown=unknown)
        cfg.log(
            job_id,
            f"UPOZORENJE: migracija profila preskočena — nepoznato ime "
            f"{', '.join(repr(u) for u in unknown)}; projekat je ostao na verziji "
            f"{job.get('schema_version', 1)}. Ne pogađa se ({WHY}).",
        )
        return JobMigration(job_id, "skipped", unknown=unknown)

    changes = []
    for where, container, key in slots:
        old = container[key]
        new = _rename(old)
        if new != old:
            changes.append((where, old, new))
        container[key] = new
    job["schema_version"] = JOB_SCHEMA_VERSION
    for path, data in shorten.items():
        cfg.write_json(path, data)
    cfg.write_json(job_path, job)
    if changes:
        summary = "; ".join(f"{w}: {o} -> {n}" for w, o, n in changes)
        cfg.log(job_id, f"migracija profila na verziju {JOB_SCHEMA_VERSION}: {summary}. Razlog: {WHY}.")
    else:
        cfg.log(job_id, f"migracija profila na verziju {JOB_SCHEMA_VERSION}: nijedno ime nije promenjeno.")
    if on_migrated:
        on_migrated(job_id)
    return JobMigration(job_id, "migrated", changes=changes)


def migrate_settings() -> tuple[str, str] | None:
    """`default_profile` u `subtitis.toml` bez oznake verzije -> novo ime.

    Vraća (staro, novo) ako je nešto promenjeno.
    """
    data = cfg.load_settings(reload=True)
    if int(data.get("profiles_version") or 1) >= cfg.PROFILES_VERSION:
        return None
    old = data.get("default_profile")
    if old is None:
        if cfg.CONFIG_FILE.exists():
            cfg.save_settings({})  # samo oznaka verzije
        return None
    if old not in RENAMES and old not in KEEP:
        cfg.echo(
            f"UPOZORENJE: default_profile = {old!r} u {cfg.CONFIG_FILE} nije poznato "
            "ime — migracija podešavanja preskočena, proveri ručno."
        )
        return None
    new = _rename(old)
    cfg.save_settings({"default_profile": new})
    return (old, new) if new != old else None


def migrate_all(on_migrated: Callable[[str], None] | None = None) -> list[JobMigration]:
    """Svi poslovi na disku + podešavanja. Bezbedno je zvati pri svakom startu."""
    migrate_settings()
    out = []
    if not cfg.JOBS_DIR.exists():
        return out
    for jdir in sorted(cfg.JOBS_DIR.iterdir()):
        if (jdir / "job.json").exists():
            out.append(migrate_job(jdir, on_migrated))
    return out
