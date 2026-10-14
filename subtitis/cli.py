"""Typer CLI — tanak omotač nad `pipeline.py`.

    subtitis transcribe <audio> --lang sr [--job-id NAME]
    subtitis translate <job_id> --to en
    subtitis build <job_id> --profile tv --script Latn
    subtitis info <job_id>
    subtitis serve
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import config as cfg
from . import pipeline
from .config import SubtitisError

app = typer.Typer(
    add_completion=False,
    help="Subtitis — SRT titlovi iz audija.",
    no_args_is_help=True,
)


def _version(value: bool) -> None:
    if value:
        from . import version_string

        typer.echo(version_string())
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True,
        help="Verzija, krug i commit koda.",
    ),
) -> None:
    """Subtitis — SRT titlovi iz audija."""


def _fail(msg: str) -> None:
    typer.secho(f"\nGREŠKA: {msg}\n", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def parse_threshold(value: str | None, default: float | None, name: str) -> float | None:
    """`None` -> podrazumevano iz asr.py, 'off'/'none' -> isključeno, inače broj."""
    if value is None:
        return default
    v = value.strip().lower()
    if v in ("off", "none", "null", "-"):
        return None
    try:
        return float(v)
    except ValueError:
        raise SubtitisError(
            f"{name}: očekivan broj ili 'off', dobijeno {value!r}."
        ) from None


# --------------------------------------------------------------------------


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., help="Ulazni audio ili video fajl."),
    lang: str = typer.Option(
        ..., "--lang",
        help="Jezik izvora, npr. 'sr'. Obavezan. 'auto' prepušta detekciju "
             "modelu — ali za srpski zadaj 'sr': detekcija ga meša sa hrvatskim.",
    ),
    job_id: str = typer.Option(None, "--job-id", help="Ime posla; podrazumevano iz imena fajla."),
    model: str = typer.Option(cfg.DEFAULT_ASR_MODEL, "--model", help="Whisper model."),
    device: str = typer.Option("cuda", "--device", help="cuda ili cpu."),
    compute_type: str = typer.Option("float16", "--compute-type", help="float16 / int8 / int8_float16."),
    beam_size: int = typer.Option(5, "--beam-size"),
    vad: bool = typer.Option(
        False, "--vad/--no-vad",
        help=(
            "Silero VAD. Podrazumevano ISKLJUČEN — na merenim snimcima je "
            "isključen dao bolji prepis (vidi asr.DEFAULT_VAD_PRESET)."
        ),
    ),
    vad_threshold: float = typer.Option(
        None, "--vad-threshold",
        help="Prag govora za VAD; podrazumevano iz asr.py. "
             "Niže = manje agresivno, za tiše snimke.",
    ),
    no_speech_threshold: str = typer.Option(
        None, "--no-speech-threshold", help="Broj ili 'off'. Podrazumevano iz asr.py."
    ),
    log_prob_threshold: str = typer.Option(
        None, "--log-prob-threshold", help="Broj ili 'off'. Podrazumevano iz asr.py."
    ),
    hallucination_filter: bool = typer.Option(
        False, "--hallucination-filter/--no-hallucination-filter",
        help="ISKLJUČEN podrazumevano: tiho odbacuje segmente i ume da preskoči do kraja fajla.",
    ),
    force_ekavica: bool = typer.Option(
        None, "--force-ekavica/--no-force-ekavica",
        help="Ijekavica -> ekavica nad words.json. Podrazumevano uključeno za sr.",
    ),
    punctuate: bool = typer.Option(
        None, "--punctuate/--no-punctuate",
        help="LLM restauracija interpunkcije. Podrazumevano se pali sama kad "
             "dijagnostika pokaže da ASR nije vratio znakove.",
    ),
    llm_model: str = typer.Option(
        None, "--llm-model", help=f"Model za interpunkciju; podrazumevano {cfg.DEFAULT_LLM_MODEL}."
    ),
    glossary: Path = typer.Option(None, "--glossary", help="Putanja do glosara."),
    force: bool = typer.Option(False, "--force", help="Prepiši i ručno menjane fajlove."),
) -> None:
    """Audio -> words.<lang>.json + sentences.<lang>.json."""
    from . import asr, segment as segment_mod

    try:
        nst = parse_threshold(no_speech_threshold, asr.NO_SPEECH_THRESHOLD, "--no-speech-threshold")
        lpt = parse_threshold(log_prob_threshold, asr.LOG_PROB_THRESHOLD, "--log-prob-threshold")
        hst = (
            asr.HALLUCINATION_SILENCE_SECONDS if hallucination_filter
            else asr.HALLUCINATION_SILENCE_THRESHOLD
        )

        r = pipeline.run_transcribe(
            audio, lang, job_id=job_id, model=model, device=device,
            compute_type=compute_type, beam_size=beam_size, vad=vad,
            vad_threshold=vad_threshold,
            no_speech_threshold=nst, log_prob_threshold=lpt,
            hallucination_silence_threshold=hst,
            force_ekavica=force_ekavica, punctuate=punctuate, llm_model=llm_model,
            glossary=glossary, force=force,
        )

        pct = 100 * r.low_confidence_words / max(1, r.words)
        typer.echo("")
        typer.secho(f"Gotovo. job_id = {r.job_id}", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  audio        : {r.duration / 60:.1f} min")
        typer.echo(f"  reči         : {r.words}  (p<0.6: {r.low_confidence_words}, {pct:.1f}%)")
        typer.echo(f"  rečenice     : {r.sentences}")
        typer.echo(
            f"  jezik        : zadat={lang}  detektovan={r.detected_language} "
            f"({r.detected_probability})"
            + (f"  -> radi se kao {r.language}" if lang != r.language else "")
        )
        typer.echo(f"  pismo ASR-a  : {r.original_script} -> Latn")

        st, before = r.structure, r.structure_before
        typer.echo(
            f"  struktura    : {st.avg_words_per_sentence:.1f} reči/rečenici, "
            f"{st.sentence_end_marks} znakova kraja "
            f"({st.marks_per_100_words:.1f}/100 reči), {st.commas} zareza, "
            f"velikih početnih {100 * st.capitalized_share:.0f}%"
        )
        # Merenje gustine se ispisuje uvek, i kad kapija ne okine — pragovi se
        # podešavaju po izmerenom.
        typer.echo(f"  {before.density_line(r.language)}")
        # Minimum, ne prosek: artefakt od 0.06 s se u proseku ne vidi.
        if r.pace:
            typer.echo(f"  {r.pace.line()}")
            pw = r.pace.warning()
            if pw:
                typer.secho(f"  {pw}", fg=typer.colors.RED)
        # Raspodela verovatnoća — merenje pred odluku o pragu bojenja.
        if r.confidence:
            for cl in r.confidence.lines():
                typer.echo(f"  {cl}")
            # Oznaka rečenice ide po UDELU sumnjivih reči, ne po „bar jednoj".
            # Dva broja stoje jedan pored drugog namerno: udeo obeleženih reči
            # i udeo obeleženih rečenica moraju ostati istog reda veličine, jer
            # oznaka rečenice vodi navigaciju (Alt+N). Ako se raziđu, pravilo
            # je ponovo postalo pojačivač.
            spct = 100 * r.low_confidence_sentences / max(1, r.sentences)
            if r.would_flag:
                typer.echo(f"  {segment_mod.would_flag_line(r.would_flag)}")
            typer.echo(
                f"  označenih rečenica: {r.low_confidence_sentences}/{r.sentences} "
                f"({spct:.0f}%)  [udeo >= "
                f"{100 * segment_mod.LOW_CONFIDENCE_SENTENCE_SHARE:.0f}%, "
                f"ispod {segment_mod.MIN_WORDS_FOR_SHARE_RULE} reči bar jedna]"
            )
        if r.punctuation:
            p = r.punctuation
            typer.echo(
                f"  interpunkcija: dodato {p.marks_added} znakova, "
                f"čankova ok/pao {p.chunks_ok}/{p.chunks_failed}, "
                f"{p.calls.requests} zahteva"
            )
            typer.echo(
                f"                 pre: {before.avg_words_per_sentence:.1f} reči/rečenici, "
                f"{before.sentence_end_marks} znakova kraja"
            )
            typer.echo(f"  {p.timing_line()}")
            typer.echo(f"  {p.retreat_line()}")
            conc = p.concessions_line()
            if conc:
                typer.echo(f"  {conc}")
            wp = p.would_pass_line()
            if wp:
                typer.echo(f"  {wp}")
            if p.failed_ranges:
                typer.secho(
                    f"  bez interpunkcije ostalo: {len(p.failed_ranges)} opsega reči "
                    f"{p.failed_ranges[:5]}",
                    fg=typer.colors.YELLOW,
                )
        warn = st.warning(r.language)
        if warn:
            typer.secho(f"  {warn}", fg=typer.colors.YELLOW)
        if r.ekavica_applied:
            typer.echo(f"  ekavica      : {len(r.ekavica_changes)} izmena")
            for ch in r.ekavica_changes[:10]:
                typer.echo(f"      {ch}")
            if len(r.ekavica_changes) > 10:
                typer.echo(f"      … još {len(r.ekavica_changes) - 10}; ceo spisak u log.txt")
        else:
            typer.echo("  ekavica      : isključeno")
        if r.elisions:
            typer.echo(f"  elizija      : {len(r.elisions)} spajanja")
            for el in r.elisions[:10]:
                typer.echo(f"      {el}")
            if len(r.elisions) > 10:
                typer.echo(f"      … još {len(r.elisions) - 10}; ceo spisak u log.txt")

        color = (
            typer.colors.YELLOW if r.coverage_pct < segment_mod.LOW_COVERAGE_WARN
            else typer.colors.GREEN
        )
        typer.secho(
            f"  pokrivenost  : {r.coverage_pct:.1f}%  (VAD: "
            + (f"da, prag {r.vad_threshold:.2f}" if vad else "NE")
            + ")",
            fg=color,
        )
        if r.gaps:
            total = sum(b - a for a, b in r.gaps)
            typer.secho(
                f"  praznine >{segment_mod.MIN_REPORTED_GAP:.0f}s : {len(r.gaps)}, "
                f"ukupno {total:.1f}s",
                fg=typer.colors.YELLOW,
            )
            for a, b in r.gaps[:10]:
                typer.echo(f"      {a:8.2f} -> {b:8.2f}  ({b - a:.2f}s)")
            if len(r.gaps) > 10:
                typer.echo(f"      … još {len(r.gaps) - 10}; ceo spisak u log.txt")
            if vad:
                typer.echo("  Ako u prazninama ima govora, VAD ga je pojeo. Uporedi sa --no-vad.")

        if r.speech:
            typer.echo(f"  {r.speech.line()}")
        advice = segment_mod.vad_advice(
            r.speech.discarded_share if r.speech else 0.0,
            r.coverage_pct,
            vad_on=vad,
        )
        if advice:
            typer.secho(f"  {advice}", fg=typer.colors.YELLOW)

        dropped = sum(r.rejections.get(k, 0) for k in asr.DROP_KEYS)
        if r.rejections:
            typer.secho(
                f"  odbacio Whisper: {dropped} prozora  {dict(r.rejections)}",
                fg=typer.colors.YELLOW if dropped else typer.colors.RESET,
            )
        if hst is not None:
            typer.secho(
                "  POZOR: filter halucinacija je UKLJUČEN. Tiho briše segmente i ume da\n"
                "  preskoči do kraja fajla; ne broji se u 'odbacio Whisper' gore.",
                fg=typer.colors.YELLOW,
            )
        typer.echo(f"\nSledeći korak:  subtitis build {r.job_id} --profile tv --script Latn\n")

    except SubtitisError as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def translate(
    job_id: str = typer.Argument(..., help="ID posla."),
    to: str = typer.Option(..., "--to", help="Ciljni jezik, npr. 'en'."),
    source: str = typer.Option(None, "--from", help="Izvorni jezik; podrazumevano jezik posla."),
    model: str = typer.Option(None, "--model", help=f"Ollama model; podrazumevano {cfg.DEFAULT_LLM_MODEL}."),
    url: str = typer.Option(None, "--url", help=f"Ollama URL; podrazumevano {cfg.DEFAULT_OLLAMA_URL}."),
    pivot: str = typer.Option(None, "--pivot", help="Pivot jezik ('en') ili 'off'."),
    variant: str = typer.Option(None, "--variant", help="'en-GB' / 'en-US' / 'off'."),
    chunk_size: int = typer.Option(None, "--chunk-size", help="Rečenica po zahtevu."),
    glossary: Path = typer.Option(None, "--glossary", help="Putanja do glosara."),
    force: bool = typer.Option(False, "--force", help="Prepiši i ručno menjan ciljni fajl."),
) -> None:
    """sentences.<src>.json -> sentences.<tgt>.json preko Ollame."""
    from . import translate as tr

    try:
        hops = pipeline.run_translate(
            job_id, to, source=source, model=model, url=url, pivot=pivot,
            variant=variant, chunk_size=chunk_size, glossary=glossary, force=force,
        )
        for h in hops:
            typer.echo("")
            typer.secho(f"{h.src} -> {h.tgt}: {h.path.name}", fg=typer.colors.GREEN, bold=True)
            typer.echo(f"  rečenica          : {h.sentences}")
            typer.echo(
                f"  glosar / varijanta: {h.glossary_entries} unosa"
                + (f" ({h.glossary_derived} izvedeno)" if h.glossary_derived else "")
                + f" / {h.variant or '—'}"
            )
            typer.echo(f"  zahteva           : {h.requests}")
            typer.echo(f"  čankova ok/pao    : {h.chunks_ok}/{h.chunks_failed}")
            if h.repeats_src is not None and h.repeats_tgt is not None:
                from . import repeats as rp

                typer.echo(
                    f"  oklevanje         : {h.repeats_src.hesitation} u "
                    f"{h.src} -> {h.repeats_tgt.hesitation} u {h.tgt}"
                )
                typer.echo(
                    f"  pojačavanje       : {h.repeats_src.emphasis} u "
                    f"{h.src} -> {h.repeats_tgt.emphasis} u {h.tgt} "
                    "(treba da se prenese)"
                )
                for lab, r in ((h.src, h.repeats_src), (h.tgt, h.repeats_tgt)):
                    for kind in (rp.HESITATION, rp.EMPHASIS, rp.UNCLASSIFIED):
                        if r.of_kind(kind):
                            typer.echo(
                                f"    {lab} {kind}: {r.examples_line(kind)}"
                            )
                    if not rp.known_language(lab):
                        typer.secho(
                            f"    (za '{lab}' nema spiska reči — sve je nerazvrstano)",
                            fg=typer.colors.YELLOW,
                        )
            if h.warning:
                typer.secho(f"  UPOZORENJE: {h.warning}", fg=typer.colors.YELLOW)
            if h.thinking_responses:
                typer.secho(
                    f"  POZOR: {h.thinking_responses} odgovora sa razmišljanjem uprkos think=false.",
                    fg=typer.colors.YELLOW,
                )
            if h.failed_ids:
                typer.secho(
                    f"  NEPREVEDENO: {len(h.failed_ids)} rečenica "
                    f"(ID: {', '.join(map(str, h.failed_ids[:20]))}) — upisan original "
                    f"sa flagom '{tr.TRANSLATION_FAILED_FLAG}'.",
                    fg=typer.colors.YELLOW,
                )
        typer.echo(f"\nSledeći korak:  subtitis build {job_id} --profile tv --lang {to}\n")

    except SubtitisError as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def build(
    job_id: str = typer.Argument(..., help="ID posla."),
    profile: str = typer.Option(
        None, "--profile",
        help="Prelomni profil. Podrazumevano: iz podešavanja; za posao iz SRT-a "
             "'original' (granice izmerene na ulaznom SRT-u).",
    ),
    script: str = typer.Option("Latn", "--script", help="Latn ili Cyrl (samo za srpski)."),
    case: str = typer.Option(
        None, "--case",
        help="Veličina slova u izlaznom SRT-u: normal, upper ili lower. "
             "Podrazumevano iz profila (digital: upper). Rečenice se ne menjaju.",
    ),
    drop_punct: bool = typer.Option(
        None, "--bez-interpunkcije/--sa-interpunkcijom",
        help="Izostavi interpunkciju u izlaznom SRT-u (apostrof i crtica unutar reči "
             "ostaju). Podrazumevano iz profila (digital: bez).",
    ),
    lang: str = typer.Option(None, "--lang", help="Jezik SRT-a; podrazumevano jezik posla."),
    out: Path = typer.Option(None, "--out", help="Izlazna putanja SRT-a."),
) -> None:
    """words/sentences -> SRT. Ne pokreće ASR ponovo."""
    try:
        r = pipeline.run_build(
            job_id, profile=profile, script=script, language=lang, out=out, case=case,
            drop_punct=drop_punct,
        )
        typer.echo("")
        typer.secho(f"SRT: {r.path}", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"  cue-ova            : {r.cues}")
        typer.echo(f"  redova preko limita: {r.over_limit}")
        typer.echo(f"  preko max_cps      : {r.over_cps}")
        typer.echo(f"  sa low_confidence  : {r.low_confidence}")
        from . import untranslated as untranslated_mod

        shown_lang = lang or cfg.load_job(job_id).language or "?"
        for line in untranslated_mod.lines(r.untranslated, shown_lang) if r.untranslated else []:
            typer.secho(f"  {line}", fg=typer.colors.YELLOW)
        if r.degraded:
            typer.secho(
                f"  POZOR: {len(r.degraded)} rečenica sa degradiranim poravnanjem "
                f"(ID: {', '.join(map(str, r.degraded[:10]))}).\n"
                "  Vremena u njima su uglavnom interpolirana. Detalji u log.txt.",
                fg=typer.colors.YELLOW,
            )
        if r.interpolated:
            typer.echo(
                "  Vremena unutar rečenice su interpolirana srazmerno broju znakova —\n"
                "  prevod nema vremena po rečima. Granice rečenica su prave."
            )
        typer.echo("")
    except SubtitisError as exc:
        _fail(str(exc))


# --------------------------------------------------------------------------


@app.command()
def info(job_id: str = typer.Argument(..., help="ID posla.")) -> None:
    """Prikaži status posla i istoriju faza."""
    try:
        job = cfg.load_job(job_id)
        jdir = cfg.job_dir(job_id)
        typer.secho(f"\n{job.job_id}", bold=True)
        typer.echo(f"  izvor    : {job.source_file}")
        typer.echo(f"  status   : {job.status}")
        typer.echo(f"  jezik    : {job.language}")
        if job.duration:
            typer.echo(f"  trajanje : {job.duration / 60:.1f} min")
        typer.echo("\n  faze:")
        for rec in job.history:
            typer.echo(f"    {rec.phase:24s} {rec.status:5s} {rec.duration_s:8.1f}s  {rec.detail}")
        typer.echo("\n  datoteke:")
        for p in sorted(jdir.rglob("*")):
            if p.is_file():
                typer.echo(f"    {p.relative_to(jdir)}  ({p.stat().st_size / 1024:.0f} kB)")
        typer.echo("")
    except SubtitisError as exc:
        _fail(str(exc))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(
        None, "--port",
        help="Podrazumevano `port` iz subtitis.toml (8010). Ako ga drži tuđa "
             "aplikacija, ide se na sledeći slobodan.",
    ),
    worker: bool = typer.Option(
        True, "--worker/--no-worker",
        help="Digni i obradu, kao nit u istom procesu. `--no-worker` je za "
             "razvoj, kad workera pokrećeš posebno.",
    ),
    open_browser: bool = typer.Option(
        False, "--open/--no-open", help="Otvori pregledač kad server krene."
    ),
) -> None:
    """Pokreni dashboard; obrada ide uz njega u istom procesu."""
    import os
    import threading
    import uvicorn
    import webbrowser

    from .desktop import choose_port, url_for
    from .web.app import WORKER_ENV

    try:
        choice = choose_port(host, port if port is not None else cfg.port())
    except SubtitisError as exc:
        _fail(str(exc))
    port = choice.port
    url = url_for(host, port)
    if choice.running:
        # Druga instanca bi značila dva workera nad istom bazom.
        typer.secho(choice.line(host) + " — ne dižem drugu.", fg=typer.colors.YELLOW)
        if open_browser:
            webbrowser.open(url)
        return
    if choice.foreign:
        typer.secho(choice.line(host), fg=typer.colors.YELLOW, bold=True)

    os.environ[WORKER_ENV] = "1" if worker else "0"
    typer.secho(
        f"\nDashboard: {url}\n"
        + (
            "Obrada radi u ovom istom procesu.\n"
            if worker
            else "Obrada NIJE dignuta — pokreni je posebno:  subtitis worker\n"
        ),
        fg=typer.colors.GREEN,
    )
    if open_browser:
        # Odloženo: pregledač sme da pozove server tek kad server sluša.
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("subtitis.web.app:app", host=host, port=port, log_level="info")


@app.command()
def desktop(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(
        None, "--port", help="Podrazumevano `port` iz subtitis.toml (8010)."
    ),
    worker: bool = typer.Option(True, "--worker/--no-worker"),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Otvori pregledač kad server krene."
    ),
) -> None:
    """Pokreni kao desktop aplikaciju: bez terminala, sa ikonicom u traci."""
    from .desktop import run

    try:
        code = run(host, port, worker=worker, open_browser=open_browser, say=typer.echo)
    except SubtitisError as exc:
        _fail(str(exc))
    raise typer.Exit(code=code)


@app.command()
def shortcut(
    name: str = typer.Option("Subtitis", "--name", help="Ime prečice."),
    desktop_dir: Path = typer.Option(
        None, "--dir", help="Gde da je napravi; podrazumevano Desktop."
    ),
) -> None:
    """Napravi prečicu na desktopu koja pokreće Subtitis bez prozora."""
    from .shortcut import create_shortcut

    try:
        path = create_shortcut(name=name, target_dir=desktop_dir)
    except SubtitisError as exc:
        _fail(str(exc))
    typer.secho(f"Prečica napravljena: {path}", fg=typer.colors.GREEN)


@app.command()
def worker(
    poll: float = typer.Option(1.0, "--poll", help="Sekundi između provera reda."),
    once: bool = typer.Option(False, "--once", help="Odradi jedan zadatak pa izađi."),
) -> None:
    """Radni proces: uzima zadatke iz reda i obrađuje ih."""
    from .web.worker import run_worker

    run_worker(poll_seconds=poll, once=once)


@app.command()
def migrate() -> None:
    """Prevedi imena profila u svim poslovima na imena posle kruga 8.

    Dashboard i worker to rade sami pri startu; ova komanda samo ispisuje šta
    je urađeno. Posao koji je već preveden se ne dira.
    """
    from .migrate import migrate_all
    from .web import db

    db.init_db()
    results = migrate_all(on_migrated=db.migrate_queued_profiles)
    for m in results:
        if m.status == "migrated":
            what = "; ".join(f"{w}: {o} -> {n}" for w, o, n in m.changes) or "bez promene imena"
            typer.echo(f"  {m.job_id}: {what}")
        elif m.status == "skipped":
            typer.secho(f"  {m.job_id}: PRESKOČEN — nepoznato ime {m.unknown}", fg=typer.colors.RED)
    counts = {s: sum(1 for m in results if m.status == s) for s in ("migrated", "current", "skipped")}
    typer.echo(
        f"migrirano {counts['migrated']}, već na novoj verziji {counts['current']}, "
        f"preskočeno {counts['skipped']}"
    )


@app.command()
def doctor() -> None:
    """Provera okruženja: ffmpeg, GPU, faster-whisper, Ollama."""
    from . import audio as audio_mod
    from . import translate as tr

    ok = True
    try:
        exe, origin = audio_mod.find_ffmpeg(use_cache=False)
        if exe:
            typer.echo(f"ffmpeg          : {exe}   [{origin}]")
            if origin == audio_mod.ORIGIN_KNOWN:
                escaped = exe.replace("\\", "\\\\")
                typer.secho(
                    "  Nađen VAN PATH-a. Ako pokrećeš iz VS Code-a, on drži kopiju PATH-a\n"
                    "  od svog pokretanja i ne vidi kasnije instalacije. Da ovo ne zavisi\n"
                    f"  od toga, upiši putanju u {cfg.CONFIG_FILE}:\n"
                    f'      ffmpeg_path = "{escaped}"',
                    fg=typer.colors.YELLOW,
                )
        else:
            ok = False
            audio_mod.ffmpeg_path()  # baca poruku sa uputstvom
    except SubtitisError as exc:
        ok = False
        typer.secho(f"ffmpeg          : {exc}", fg=typer.colors.RED)
    try:
        import faster_whisper  # noqa: F401

        typer.echo("faster-whisper  : ok")
    except ImportError:
        ok = False
        typer.secho("faster-whisper  : NEDOSTAJE (pip install -r requirements.txt)", fg=typer.colors.RED)
    try:
        tr.OllamaClient(model=cfg.DEFAULT_LLM_MODEL).check()
        typer.echo(f"ollama          : ok ({cfg.DEFAULT_LLM_MODEL})")
    except SubtitisError as exc:
        typer.secho(f"ollama          : {exc}", fg=typer.colors.YELLOW)
    typer.echo(f"python          : {sys.version.split()[0]}")
    typer.echo(f"jobs dir        : {cfg.JOBS_DIR}")
    raise typer.Exit(code=0 if ok else 1)


def main() -> None:
    # Za slučaj da se ušlo preko konzolne skripte `subtitis`, a ne preko
    # `python -m subtitis`, koji ovo već uradi pre uvoza. Poziv je idempotentan.
    from .bootstrap import bootstrap, crash_guard

    bootstrap()
    with crash_guard("komanda"):
        app()


if __name__ == "__main__":
    main()
