"""JSON API za MCP server (krug 28).

MCP sloj (`subtitis.mcp_server`) ne radi ništa svoje: zove ove rute. Ako
nešto nedostaje, dodaje se ovde, ne tamo. Sve ide na 127.0.0.1 — server
ionako sluša samo tamo, a rute koje čitaju putanju sa diska ili gase server
to još jednom proveravaju.

Ograničenja veličine (`TEXT_MAX`, `FLAGS_MAX`) nisu zbog privatnosti nego
zbog veličine odgovora: dvesta rečenica troši kontekst.
"""

from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, Request
from pydantic import BaseModel

from .. import config as cfg
from ..config import SubtitisError
from . import db
from .app import (
    CASES,
    MAX_UPLOAD_BYTES,
    app,
    available_languages,
    is_srt_job,
    job_invented,
    job_or_404,
    last_build,
    load_sentences,
    output_files,
    project_name,
    timecode,
    untranslated,
)

#: Najviše rečenica po pozivu `text`.
TEXT_MAX = 50
#: Najviše upozorenja po pozivu `flags`.
FLAGS_MAX = 20
#: Vrste upozorenja u `job` i `flags`.
FLAG_KINDS = ("izmisljeno", "neprevedeno", "lici_na_original", "nesigurno",
              "prekratko", "prekoracenje", "preskoceno")

#: Gašenje servera, ako ga način pokretanja podržava — postavlja ga
#: `desktop.run` i `cli.serve`. Bez toga (`TestClient`, ručni uvicorn)
#: `/api/shutdown` kaže da ne može.
SHUTDOWN: Callable[[], None] | None = None

LOCAL = {"127.0.0.1", "::1", "localhost"}


def _local_only(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in LOCAL and host != "testclient":
        raise HTTPException(403, "Samo sa ovog računara.")


def _running() -> dict[str, Any] | None:
    row = db.running_task()
    if row is None:
        return None
    job = db.get_job(row["job_id"])
    ev = [e for e in db.recent_events(row["job_id"], 40) if e["kind"] == "progress"]
    return {
        "job": row["job_id"],
        "name": project_name(job["source_name"]) if job else row["job_id"],
        "kind": row["kind"],
        "phase": ev[0]["message"] if ev else None,
        "progress": ev[0]["progress"] if ev else None,
    }


def queue_info() -> dict[str, Any]:
    """Za `/api/identity`: koliko čeka i šta se upravo radi."""
    return {"queue": db.queued_count(), "running": _running(), "pid": os.getpid()}


# --- gašenje -----------------------------------------------------------------


class ShutdownIn(BaseModel):
    force: bool = False


@app.post("/api/shutdown")
def shutdown(body: ShutdownIn, request: Request) -> Any:
    """Uredno gašenje (kao „Izađi" u traci). Dok se obrađuje zadatak, odbija
    — gašenje usred prepoznavanja baca minute rada — osim uz `force`."""
    _local_only(request)
    run = _running()
    if run and not body.force:
        raise HTTPException(409, {
            "poruka": "Obrada je u toku — gašenje bi je prekinulo. Ponovi sa force.",
            "running": run,
        })
    if SHUTDOWN is None:
        raise HTTPException(503, "Ovaj način pokretanja ne podržava gašenje preko API-ja.")
    # Posle odgovora: inače klijent dobije prekinutu vezu umesto potvrde.
    threading.Timer(0.3, SHUTDOWN).start()
    return {"shutting_down": True, "interrupted": run}


# --- poslovi -----------------------------------------------------------------


def _languages(job_id: str) -> list[str]:
    return available_languages(job_id)


@app.get("/api/jobs")
def api_jobs(limit: int = 20, state: str | None = None) -> Any:
    rows = db.list_jobs()
    out = []
    for r in rows:
        pending = db.pending_task(r["id"])
        st = "u_redu" if pending is not None and r["status"] not in ("transcribing", "translating") else r["status"]
        if state and state not in (r["status"], st):
            continue
        out.append({
            "id": r["id"], "name": project_name(r["source_name"]), "status": r["status"],
            "pending": pending["kind"] if pending is not None else None,
            "languages": _languages(r["id"]), "created": r["created_at"],
        })
    total = len(out)
    return {"jobs": out[: max(1, limit)], "total": total, "more": max(0, total - limit)}


def _flags(job_id: str) -> dict[str, list[dict[str, Any]]]:
    """Sva upozorenja posla po vrsti, sa tekstom — `job` broji, `flags` seče."""
    job = job_or_404(job_id)
    lang = job["language"]
    out: dict[str, list[dict[str, Any]]] = {k: [] for k in FLAG_KINDS}
    src = load_sentences(job_id, lang) if lang else None
    by_id = {s.id: s for s in src.sentences} if src else {}

    for v in job_invented(job_id, lang):
        s = by_id.get(v["sentence"])
        out["izmisljeno"].append({"sentence": v["sentence"], "at": v["at"], "words": v["words"],
                                  "caught": v["caught"], "text": s.text if s else ""})
    for u in untranslated(job_id, _languages(job_id)):
        tf = load_sentences(job_id, u["lang"])
        tby = {s.id: s for s in tf.sentences} if tf else {}
        for g in u["groups"]:
            for i in g["sentences"]:
                kind = "neprevedeno" if i["reason"] == "translation_failed" else "lici_na_original"
                s = tby.get(i["sentence"])
                out[kind].append({"lang": u["lang"], "sentence": i["sentence"], "cues": i["cues"],
                                  "reason": i["reason"], "text": s.text if s else ""})
    if src:
        for s in src.sentences:
            if "low_confidence" in s.flags:
                out["nesigurno"].append({"sentence": s.id, "at": timecode(s.s), "text": s.text})
    for lg in _languages(job_id):
        b = last_build(job_id, lg)
        if not b:
            continue
        for c in b["too_short"]:
            out["prekratko"].append({"lang": lg, "cue": c["cue"], "at": c["at"], "dur": c["dur"]})
        for o in b["overflow"]:
            out["prekoracenje"].append({"lang": lg, **{k: o.get(k) for k in ("cue", "chars", "cps", "worse_than_source")}})
    for t in db.rows_to_dicts(db.failed_tasks(job_id, 50)):
        if t.get("state") == "skipped":
            out["preskoceno"].append({"kind": t["kind"], "error": t.get("error")})
    return out


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> Any:
    job = dict(job_or_404(job_id))
    pending = db.pending_task(job_id)
    ev = [e for e in db.recent_events(job_id, 40) if e["kind"] == "progress"]
    return {
        "id": job_id, "name": project_name(job["source_name"]), "status": job["status"],
        "error": job["error"], "language": job["language"], "srt_job": is_srt_job(job_id),
        "pending": pending["kind"] if pending is not None else None,
        "progress": ({"phase": ev[0]["message"], "fraction": ev[0]["progress"]} if ev and pending else None),
        "languages": _languages(job_id),
        "outputs": [{"name": f["name"], "cues": f["cues"]} for f in output_files(job_id)],
        "warnings": {k: len(v) for k, v in _flags(job_id).items() if v},
    }


@app.get("/api/jobs/{job_id}/flags")
def api_flags(job_id: str, kind: str | None = None) -> Any:
    if kind and kind not in FLAG_KINDS:
        raise HTTPException(400, f"Nepoznata vrsta {kind!r}; postoje: {', '.join(FLAG_KINDS)}.")
    all_flags = _flags(job_id)
    items = [{"kind": k, **it} for k, lst in all_flags.items() if not kind or k == kind for it in lst]
    return {"flags": items[:FLAGS_MAX], "total": len(items), "more": max(0, len(items) - FLAGS_MAX)}


@app.get("/api/jobs/{job_id}/text")
def api_text(job_id: str, lang: str | None = None, start: int = 1, end: int | None = None) -> Any:
    """Rečenice `start`..`end` (id-jevi, uključivo), najviše `TEXT_MAX`."""
    job = job_or_404(job_id)
    lang = lang or job["language"]
    sf = load_sentences(job_id, lang)
    if sf is None:
        raise HTTPException(404, f"Posao {job_id} nema tekst na jeziku {lang!r}; ima: {', '.join(_languages(job_id)) or 'nijedan'}.")
    sel = [s for s in sf.sentences if s.id >= start and (end is None or s.id <= end)]
    shown = sel[:TEXT_MAX]
    return {
        "lang": lang, "total_sentences": len(sf.sentences),
        "sentences": [{"id": s.id, "at": timecode(s.s), "text": s.text, "flags": s.flags} for s in shown],
        "more": max(0, len(sel) - len(shown)),
        "next": shown[-1].id + 1 if len(sel) > len(shown) else None,
    }


# --- rad -----------------------------------------------------------------------


class TranscribeIn(BaseModel):
    path: str
    language: str = "sr"
    profile: str | None = None
    vad_preset: str | None = None
    terms: list[str] = []


@app.post("/api/jobs")
def api_transcribe(body: TranscribeIn, request: Request) -> Any:
    """Nov posao iz fajla NA DISKU — bez otpremanja kroz pregledač (zato su
    u krugu 28 mp3-ovi završili u `static/`). Fajl se kopira u posao."""
    from .. import asr

    _local_only(request)
    src = Path(body.path)
    if not src.is_file():
        raise HTTPException(404, f"Nema fajla: {src}")
    if src.stat().st_size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Fajl je veći od dozvoljenih {MAX_UPLOAD_BYTES // 1024**3} GB.")
    vad = body.vad_preset or asr.DEFAULT_VAD_PRESET
    if vad not in asr.VAD_PRESETS:
        raise HTTPException(400, f"Nepoznat izbor filtera tišine: {vad!r}")
    if body.profile:
        try:
            cfg.load_profile(body.profile)
        except SubtitisError as exc:
            raise HTTPException(400, str(exc)) from exc
    job_id = cfg.make_job_id(src)
    up = cfg.job_dir(job_id) / "upload"
    up.mkdir(parents=True, exist_ok=True)
    dst = up / src.name
    shutil.copy2(src, dst)
    gloss = cfg.make_job_glossary(body.language, use_default=cfg.new_project_glossary() == "default",
                                  recording_terms=body.terms)
    cfg.write_json(cfg.job_glossary_path(job_id), gloss)
    db.create_job(job_id, src.name, str(dst.resolve()), body.language)
    db.add_event(job_id, "log", f"sa diska: {src}")
    params: dict[str, Any] = {"model": cfg.DEFAULT_ASR_MODEL, "vad_preset": vad}
    if body.profile:
        params["profile"] = body.profile
    db.enqueue(job_id, "transcribe", params)
    db.add_event(job_id, "status", "uploaded")
    return {"id": job_id, "queued": "transcribe"}


class TranslateIn(BaseModel):
    languages: list[str]
    script: str = "Latn"
    profile: str = ""
    case: str = "normal"
    punct: str = "keep"


@app.post("/api/jobs/{job_id}/translate")
def api_translate(job_id: str, body: TranslateIn) -> Any:
    from ..translit import BISCRIPTAL_LANGUAGES

    job = job_or_404(job_id)
    if not body.languages:
        raise HTTPException(400, "Nije zadat nijedan jezik.")
    if db.pending_task(job_id):
        raise HTTPException(409, "Za ovaj projekat već postoji zadatak u redu.")
    if body.case not in CASES or body.punct not in ("keep", "drop") or body.script not in ("Latn", "Cyrl"):
        raise HTTPException(400, "Nepoznat izbor slova, interpunkcije ili pisma.")
    bad = [lg for lg in body.languages if lg == job["language"]]
    if bad:
        raise HTTPException(400, f"Izvorni jezik nije ciljni: {', '.join(bad)}.")
    for lg in body.languages:
        script = body.script if lg in BISCRIPTAL_LANGUAGES else "Latn"
        db.enqueue(job_id, "translate", {
            "to": lg, "model": None, "pivot": None, "profile": body.profile, "script": script,
            "case": body.case, "drop_punct": body.punct == "drop",
        })
        db.add_event(job_id, "log", f"prevod na {lg}/{script} stavljen u red")
    return {"id": job_id, "queued": [f"translate:{lg}" for lg in body.languages]}


class RebuildIn(BaseModel):
    language: str | None = None
    profile: str | None = None
    script: str = "Latn"
    case: str = "normal"
    punct: str = "keep"


@app.post("/api/jobs/{job_id}/rebuild")
def api_rebuild(job_id: str, body: RebuildIn) -> Any:
    """Sinhrono, kao editor (`/rebuild`)."""
    from .. import pipeline

    job = job_or_404(job_id)
    if body.case not in CASES or body.punct not in ("keep", "drop") or body.script not in ("Latn", "Cyrl"):
        raise HTTPException(400, "Nepoznat izbor slova, interpunkcije ili pisma.")
    if db.pending_task(job_id) is not None:
        raise HTTPException(409, "Obrada je u toku — prelomi ponovo kad završi.")
    lang = body.language or job["language"]
    try:
        pipeline.run_build(job_id, profile=body.profile or None, script=body.script, language=lang,
                           case=body.case, drop_punct=body.punct == "drop")
    except SubtitisError as exc:
        raise HTTPException(400, str(exc).splitlines()[0]) from exc
    b = last_build(job_id, lang) or {}
    return {"id": job_id, "language": lang, "profile": b.get("profile"), "cues": b.get("cues"),
            "too_short": len(b.get("too_short") or []), "silence_timing": b.get("silence_timing"),
            "path": api_export(job_id, lang, body.script)["path"]}


@app.get("/api/jobs/{job_id}/export")
def api_export(job_id: str, lang: str | None = None, script: str | None = None) -> Any:
    """Putanja do gotovog SRT-a — ne sadržaj."""
    job = job_or_404(job_id)
    lang = lang or job["language"]
    out = cfg.job_dir(job_id) / "out"
    names = [f["name"] for f in output_files(job_id)]
    want = [n for n in names if n[:-4].rsplit(".", 1)[-1].split("-")[0] == lang
            and (script is None or n[:-4].endswith(f"-{script}") or "-" not in n[:-4].rsplit(".", 1)[-1])
            and "_lead" not in n]
    if not want:
        raise HTTPException(404, f"Posao {job_id} nema SRT za {lang!r}; ima: {', '.join(names) or 'ništa'}.")
    return {"id": job_id, "language": lang, "path": str((out / want[0]).resolve())}
