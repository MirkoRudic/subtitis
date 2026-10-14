"""Jezgro pajplajna, bez ijednog ispisa.

CLI i web worker zovu ISTE funkcije. Da su pisali svoje verzije, razišle bi se
prvom promenom — a razlika između onoga što radi terminal i onoga što radi
dashboard je greška koja se otkriva tek na gotovom titlu.

Sav izlaz ide kroz `on_log` i `on_progress`; ovaj modul ne zna za `typer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import config as cfg
from . import hallucination as halluc_mod
from . import repeats
from . import translit
from . import typography
from . import untranslated as untranslated_mod
from .config import SubtitisError
from .models import Job, SentencesFile, WordsFile

Log = Callable[[str], None]
Progress = Callable[[str, float], None]  # (faza, 0..1)


def _noop_log(msg: str) -> None:
    pass


def _noop_progress(phase: str, fraction: float) -> None:
    pass


# ==========================================================================
# Pismo: jedno po datoteci, provereno pre nego što datoteka nastane
# ==========================================================================

#: Pismo u kom se DRŽE `sentences.<lang>.json` i `words.<lang>.json`.
#: Ćirilica postoji samo u gotovom SRT-u; sve unutra je latinica, pa se
#: poređenja, glosar i prelamanje nikad ne oslanjaju na pismo.
STORAGE_SCRIPT = "Latn"


def assert_single_script(
    items: list[tuple[Any, str]], script: str, *, what: str
) -> None:
    """Pukni glasno ako išta u `items` nije u traženom pismu.

    Mešano pismo u gotovoj datoteci je greška koja tiho prođe kroz sve dalje
    korake i vidi se tek na ekranu, kad je titl već isporučen. Zato se proverava
    na kraju svakog koraka koji piše tekst — i za `sentences.*.json` i za SRT.
    """
    bad = translit.script_violations(items, script)
    if not bad:
        return
    primeri = ", ".join(
        f"{label} ({''.join(sorted(foreign))})" for label, foreign in bad[:10]
    )
    raise SubtitisError(
        f"{what}: {len(bad)} od {len(items)} stavki nije u pismu {script} — "
        f"datoteka bi imala oba pisma.\n"
        f"  gde: {primeri}" + (" …" if len(bad) > 10 else "") + "\n"
        "  Ovo je greška u pajplajnu, ne u ulazu: transliteracija ide jednim "
        "prolazom nad celim tekstom i mora sve da pokrije."
    )


# ==========================================================================
# Transkripcija
# ==========================================================================


@dataclass
class TranscribeResult:
    job_id: str
    duration: float
    words: int
    low_confidence_words: int
    sentences: int
    low_confidence_sentences: int
    coverage_pct: float
    gaps: list[tuple[float, float]]
    language: str = ""               # razrešen jezik; kod `auto` ono detektovano
    structure: Any = None            # segment.Structure posle segmentacije
    structure_before: Any = None     # ista mera pre restauracije interpunkcije
    pace: Any = None                 # segment.Pace: najkraća i najbrža rečenica
    confidence: Any = None           # segment.Confidence: raspodela verovatnoća
    would_flag: Any = None           # merenje izuzeća funkcijskih reči
    punctuation: Any = None          # punctuate.PunctuateStats ili None
    ekavica_changes: list[Any] = field(default_factory=list)
    ekavica_applied: bool = False
    elisions: list[Any] = field(default_factory=list)  # asr.Elision, samo fr
    detected_language: str | None = None
    detected_probability: float | None = None
    original_script: str | None = None
    rejections: dict[str, int] = field(default_factory=dict)
    vad: bool = False
    #: Prag koji je STVARNO otišao u Silero (ne ono što je prosleđeno kao None).
    vad_threshold: float | None = None
    #: Šta bi VAD uradio nad ovim snimkom — meri se i kad je isključen.
    speech: Any = None
    #: Nizovi reči nultog trajanja (`hallucination.run_line`), za ispis.
    hallucinated: list[str] = field(default_factory=list)


def run_transcribe(
    audio_path: str | Path,
    language: str,
    *,
    job_id: str | None = None,
    model: str = cfg.DEFAULT_ASR_MODEL,
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad: bool = False,
    vad_threshold: float | None = None,
    no_speech_threshold: float | None = None,
    log_prob_threshold: float | None = None,
    hallucination_silence_threshold: float | None = None,
    force_ekavica: bool | None = None,
    punctuate: bool | None = None,
    llm_model: str | None = None,
    glossary: str | Path | None = None,
    force: bool = False,
    on_log: Log = _noop_log,
    on_progress: Progress = _noop_progress,
) -> TranscribeResult:
    """Audio -> words.<lang>.json + sentences.<lang>.json."""
    from . import asr, audio as audio_mod, ekavica as ekavica_mod, segment as segment_mod

    if no_speech_threshold is None:
        no_speech_threshold = asr.NO_SPEECH_THRESHOLD
    if log_prob_threshold is None:
        log_prob_threshold = asr.LOG_PROB_THRESHOLD
    if hallucination_silence_threshold is None:
        hallucination_silence_threshold = asr.HALLUCINATION_SILENCE_THRESHOLD

    audio_mod.check_environment()
    src = Path(audio_path)
    if not src.exists():
        raise SubtitisError(f"Ulazna datoteka ne postoji: {src}")

    jid = job_id or cfg.make_job_id(src)
    jdir = cfg.job_dir(jid)
    jdir.mkdir(parents=True, exist_ok=True)

    # Kod auto-detekcije jezik se zna tek posle ASR-a, pa se i provera ručnih
    # izmena radi tek tada. Cena je jedan uzaludan prolaz ASR-a u retkom
    # slučaju; alternativa bi bila da se ispravke pregaze, što ne dolazi u obzir.
    auto_language = language.strip().lower() == asr.AUTO_LANGUAGE

    def guard_human_edits(lang: str) -> Path:
        path = jdir / f"sentences.{lang}.json"
        if path.exists():
            existing = SentencesFile.model_validate(cfg.read_json(path))
            if existing.edited_by_human and not force:
                raise SubtitisError(
                    f"{path.name} je označen kao ručno menjan (edited_by_human=true).\n"
                    "Ponovno pokretanje bi pregazilo tvoje ispravke.\n"
                    "Ako to stvarno želiš, dodaj --force (napraviću .bak kopiju)."
                )
        return path

    sent_path = jdir / f"sentences.{language}.json"
    if not auto_language:
        sent_path = guard_human_edits(language)

    job = Job(job_id=jid, source_file=str(src.resolve()), language=language, status="created")
    cfg.save_job(job)
    cfg.log(jid, f"ulaz: {src}  model: {model}  device: {device}/{compute_type}")

    # --- 1. audio ---
    wav = jdir / "source.wav"
    job.status = "normalizing"
    on_progress("audio", 0.0)
    with cfg.phase(job, "audio") as det:
        duration = audio_mod.normalize(src, wav)
        det["duration_s_audio"] = round(duration, 2)
    job.duration = duration
    on_progress("audio", 1.0)

    # --- 2. ASR ---
    if glossary is None:
        ensure_job_glossary(jid)
    gloss = cfg.load_glossary(glossary) if glossary else cfg.job_glossary(jid)
    # U prepoznavanje (`hotwords`) idu SAMO termini ovog snimka. Prevodni
    # parovi (`sr->fr`...) idu samo u prevod: kao nagoveštaj su izlazili preko
    # tišine na kraju snimka (krug 23, Video-4 draft 2: 13 izmišljenih reči
    # iz `sr->fr` rečnika drugog projekta). Uz auto-detekciju termini ne mogu
    # da se proslede: jezik se zna tek posle prolaza.
    terms = [] if auto_language else cfg.recording_terms(gloss, language)
    cfg.log(jid, f"glosar projekta: {len(terms)} termina ide u prepoznavanje (hotwords)")
    job.status = "transcribing"
    cfg.save_job(job)

    def asr_progress(pos: float, total: float) -> None:
        on_progress("asr", min(1.0, pos / total) if total else 0.0)
        cfg.checkpoint()  # posle svakog segmenta — posao je možda obrisan

    # Šta bi VAD uradio nad ovim snimkom. Meri se i kad je isključen: to je
    # jedina brojka koja kaže koliko snimak sadrži tišine, a upravo je tišina
    # ono što isključen VAD pušta pred model. Košta ~0.15 s na 94 s zvuka.
    speech = asr.measure_speech(wav, vad_threshold)
    cfg.log(jid, speech.line())

    # `vad_threshold` MORA u zapis. Bez njega su dva prolaza sa različitim
    # pragom u job.json nerazlučiva, pa se identičan prepis ne može pročitati
    # ni kao „prag nema dejstva" ni kao „prolazi su išli sa istim pragom".
    with cfg.phase(
        job, "asr", model=model, device=device, compute_type=compute_type, vad=vad,
        hotwords=len(terms),
        vad_threshold=(asr.vad_parameters(vad_threshold)["threshold"] if vad else None),
        no_speech_threshold=no_speech_threshold, log_prob_threshold=log_prob_threshold,
        hallucination_silence_threshold=hallucination_silence_threshold,
    ) as det:
        wf: WordsFile = asr.transcribe(
            wav,
            job_id=jid,
            source_file=str(src.resolve()),
            language=language,
            duration=duration,
            glossary_terms=terms,
            model_size=model,
            device=device,
            compute_type=compute_type,
            beam_size=beam_size,
            vad=vad,
            vad_threshold=vad_threshold,
            no_speech_threshold=no_speech_threshold,
            log_prob_threshold=log_prob_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            progress=asr_progress,
        )
        det["words"] = len(wf.words)
        det["detected_language"] = wf.asr.detected_language
        det["detected_probability"] = wf.asr.detected_probability
        det["original_script"] = wf.asr.params.get("original_script")
        det["rejections"] = dict(wf.asr.rejections)
        det["vad_discarded_share"] = round(speech.discarded_share, 4)
        det["vad_segments"] = speech.segments

    # Prazan rezultat je GREŠKA, ne tih uspeh. Bez ovoga posao prođe do kraja i
    # javi „gotovo" nad praznim SRT-om.
    if not wf.words:
        raise SubtitisError(
            f"ASR nije vratio nijednu reč za '{src.name}'.\n"
            "  Proveri da fajl nije prazan ili bez govora (`ffprobe` trajanje, "
            "poslušaj ga), pa probaj --no-vad."
        )

    # VRAM se oslobađa odmah — Whisper i LLM nikad zajedno.
    asr.unload_model()
    cfg.log(jid, "Whisper model oslobođen iz VRAM-a")
    on_progress("asr", 1.0)

    # --- 2a. razrešavanje auto-detekcije ---
    if auto_language:
        language = wf.language
        job.language = language
        cfg.save_job(job)
        detected = (
            f"jezik detektovan: {language} "
            f"(sigurnost {wf.asr.detected_probability})"
        )
        cfg.log(jid, detected)
        on_log(detected)
        if language == "sr":
            note = (
                "Detekcija srpski meša sa hrvatskim i vraća ijekavicu. "
                "Ako je snimak srpski, zadaj jezik izričito."
            )
            cfg.log(jid, f"NAPOMENA: {note}")
            on_log(f"NAPOMENA: {note}")
        sent_path = guard_human_edits(language)

    # --- 2b. francuska elizija, PRE svega što čita reči ---
    #
    # Mora ovde: razbijena elizija kvari i prelome (`linebreak` vidi dve reči)
    # i restauraciju interpunkcije (spajanje je promena niza reči, pa validacija
    # odbaci čank).
    elisions: list = []
    if asr.elision_language(language):
        with cfg.phase(job, "elizija") as det:
            elisions = asr.merge_elisions(wf.words, language)
            det["merges"] = len(elisions)
        cfg.log(jid, f"elizija: spojeno {len(elisions)} parova tokena")
        for el in elisions:
            cfg.log(jid, f"elizija {el}")

    # --- 2c. ijekavica -> ekavica, PRE segmentacije ---
    do_ekavica = (
        cfg.force_ekavica_default(language) if force_ekavica is None else force_ekavica
    )
    changes: list = []
    if do_ekavica:
        with cfg.phase(job, "ekavica") as det:
            changes = ekavica_mod.convert_words(wf.words)
            det["changes"] = len(changes)
        for ch in changes:
            cfg.log(jid, f"ekavica {ch}")
    wf.ekavica = {"applied": do_ekavica, "changes": len(changes)}

    # --- 2d. dijagnostika strukture + restauracija interpunkcije ---
    structure_before = segment_mod.analyze(wf.words)
    cfg.log(jid, f"struktura: {structure_before.summary()}")
    # Merenje gustine ide u log posle SVAKE transkripcije, bez obzira na odluku
    # — pragovi se podešavaju po izmerenom, a ne po pretpostavci.
    density = structure_before.density_line(language)
    cfg.log(jid, density)
    on_log(density)
    warning = structure_before.warning(language)
    if warning:
        cfg.log(jid, f"UPOZORENJE: {warning}")
        on_log(f"UPOZORENJE: {warning}")

    do_punct = (
        (cfg.auto_punctuate() and structure_before.needs_punctuation(language))
        if punctuate is None
        else punctuate
    )
    punct_stats = None
    if do_punct:
        from . import punctuate as punct_mod
        from . import translate as tr

        on_progress("punctuate", 0.0)
        client = tr.OllamaClient(
            model=llm_model or cfg.llm_model(),
            log=lambda m: (cfg.log(jid, m), on_log(m)),
        )
        try:
            client.check()
            with cfg.phase(job, "punctuate", model=client.model) as det:
                new_words, punct_stats = punct_mod.punctuate_words(
                    wf.words, language, client=client,
                    log=lambda m: (cfg.log(jid, m), on_log(m)),
                )
                det["marks_added"] = punct_stats.marks_added
                det["chunks_ok"] = punct_stats.chunks_ok
                det["chunks_failed"] = punct_stats.chunks_failed
                det["failed_ranges"] = punct_stats.failed_ranges
                det["seconds_total"] = round(punct_stats.seconds_total, 1)
                det["seconds_wasted"] = round(punct_stats.seconds_wasted, 1)
                det["would_pass"] = dict(punct_stats.would_pass)
                det["retreats"] = punct_stats.retreats
                det["retreat_words"] = punct_stats.retreat_words
                det["concessions"] = dict(punct_stats.concessions)
                det["decapitalized"] = punct_stats.decapitalized
            wf.words = new_words
            # Koliko je koštalo i koliko je otišlo na bačene odgovore — bez
            # ovoga se ne vidi isplati li se restauracija uopšte.
            for line in (
                punct_stats.timing_line(),
                punct_stats.retreat_line(),
                punct_stats.concessions_line(),
                punct_stats.would_pass_line(),
            ):
                if line:
                    cfg.log(jid, line)
                    on_log(line)
        except SubtitisError as exc:
            # Restauracija je poboljšanje, ne uslov. Transkript je već zapisan.
            cfg.log(jid, f"UPOZORENJE: restauracija interpunkcije nije uspela: {exc}")
            on_log(f"UPOZORENJE: restauracija interpunkcije nije uspela: {exc}")
        finally:
            client.unload()
        on_progress("punctuate", 1.0)

    cfg.write_model(jdir / f"words.{language}.json", wf)

    # Tišine iz zvuka (krug 19), dok zvuk postoji: `source.wav` se posle roka
    # briše, a `pauses.json` ostaje. Prelamanje ih zasad ne koristi
    # (`linebreak.pauses_enabled`); čuvaju se za ponovno merenje.
    from . import pauses as pauses_mod

    def _say(m: str) -> None:
        cfg.log(jid, m)
        on_log(m)

    pauses_mod.ensure(jid, _say)

    # --- 3. segmentacija ---
    with cfg.phase(job, "segment") as det:
        sentences = segment_mod.segment(wf.words)
        cov, gaps = segment_mod.coverage(sentences, duration)
        structure = segment_mod.analyze(wf.words, sentences)
        pace = segment_mod.analyze_pace(sentences)
        confidence = segment_mod.analyze_confidence(wf.words)
        # Merenje, ne odluka: koliko bi rečenica bilo označeno kad bi se
        # funkcijske reči izuzele iz udela. Isto kao `punctuate.would_pass` —
        # broj se skuplja dok je pravilo još nepromenjeno.
        by_index = {w.i: w for w in wf.words}
        chunks = [
            [by_index[k] for k in range(s.words[0], s.words[1] + 1) if k in by_index]
            for s in sentences
        ]
        would_flag = segment_mod.would_flag(chunks, language)
        # Reči bez zvuka iza sebe: izmišljene, a verovatnoća ih ne vidi
        # (Radionica draft 1: `kamp` 0,98, `solidarnosti` 0,99). Vidi `hallucination`.
        halluc_runs = halluc_mod.zero_runs(wf.words)
        halluc_ids = halluc_mod.flag_sentences(sentences, halluc_runs)
        det["hallucinated_runs"] = [halluc_mod.run_line(r) for r in halluc_runs]
        det["hallucinated_sentences"] = halluc_ids
        det["sentences"] = len(sentences)
        det["low_confidence"] = sum(1 for s in sentences if "low_confidence" in s.flags)
        det["coverage_pct"] = round(cov, 1)
        det["gaps_over_3s"] = len(gaps)
        det["avg_words_per_sentence"] = round(structure.avg_words_per_sentence, 2)
        det["sentence_end_marks"] = structure.sentence_end_marks
        det["shortest_sentence_s"] = round(pace.shortest_duration, 3)
        det["fastest_s_per_word"] = round(pace.fastest_per_word, 3)
        det["impossible_sentences"] = list(pace.impossible)
        det["confidence_buckets"] = list(confidence.counts)
        det["confidence_percentiles"] = dict(confidence.percentiles)
        det["would_flag"] = dict(would_flag)
    if halluc_runs:
        msg = (
            f"UPOZORENJE: {len(halluc_runs)} niz(a) reči BEZ ZVUKA iza njih — gotovo "
            f"sigurno izmišljene (rečenice {', '.join(map(str, halluc_ids))}, crvene "
            "u editoru): " + "; ".join(halluc_mod.run_line(r) for r in halluc_runs)
        )
        cfg.log(jid, msg)
        on_log(msg)
    cfg.log(jid, f"struktura posle segmentacije: {structure.summary()}")
    cfg.log(jid, pace.line())
    cfg.log(jid, segment_mod.would_flag_line(would_flag))
    for line in confidence.lines():
        cfg.log(jid, line)
    # Rečenica kraća nego što je fizički moguće izgovoriti je artefakt, ne
    # govor. Mreža bi je spojila; ako je preživela, to mora da se vidi odmah.
    pace_warning = pace.warning()
    if pace_warning:
        cfg.log(jid, f"UPOZORENJE: {pace_warning}")
        on_log(f"UPOZORENJE: {pace_warning}")
    final_warning = structure.warning(language)
    if final_warning:
        cfg.log(jid, f"UPOZORENJE: {final_warning}")
        on_log(f"UPOZORENJE: {final_warning}")
    if sent_path.exists():
        bak = cfg.backup(sent_path)
        if bak:
            cfg.log(jid, f"prethodne rečenice sačuvane u {bak.name}")
    cfg.write_model(
        sent_path,
        SentencesFile(
            job_id=jid, language=language, script="Latn",
            derived_from=f"words.{language}.json",
            edited_by_human=False, sentences=sentences,
        ),
    )
    if gaps:
        cfg.log(jid, f"praznine >{segment_mod.MIN_REPORTED_GAP}s: {gaps}")
    advice = segment_mod.vad_advice(speech.discarded_share, cov, vad_on=vad)
    if advice:
        cfg.log(jid, advice)
        on_log(advice)

    job.status = "transcribed"
    cfg.save_job(job)
    on_progress("segment", 1.0)

    return TranscribeResult(
        job_id=jid,
        duration=duration,
        words=len(wf.words),
        low_confidence_words=sum(1 for w in wf.words if w.p < 0.6),
        sentences=len(sentences),
        low_confidence_sentences=sum(1 for s in sentences if "low_confidence" in s.flags),
        coverage_pct=cov,
        gaps=gaps,
        language=language,
        structure=structure,
        structure_before=structure_before,
        pace=pace,
        confidence=confidence,
        would_flag=would_flag,
        punctuation=punct_stats,
        ekavica_changes=changes,
        ekavica_applied=do_ekavica,
        elisions=elisions,
        detected_language=wf.asr.detected_language,
        detected_probability=wf.asr.detected_probability,
        original_script=wf.asr.params.get("original_script"),
        rejections=dict(wf.asr.rejections),
        vad=vad,
        vad_threshold=(asr.vad_parameters(vad_threshold)["threshold"] if vad else None),
        speech=speech,
        hallucinated=[halluc_mod.run_line(r) for r in halluc_runs],
    )


# ==========================================================================
# Uvoz postojećeg SRT-a (prevod sa očuvanim tajmingom)
# ==========================================================================

#: Posao nastao iz SRT-a ima ovaj fajl; po njemu se prevod i prelamanje
#: prebacuju u režim sa fiksnim cue-ovima.
CUES_FILE = "cues.json"


def ensure_job_glossary(job_id: str) -> None:
    """Posao bez sopstvenog glosara dobija ono što kaže podešavanje
    `new_project_glossary` — prazan ili kopiju podrazumevanog.

    Web pravi glosar posla pri otpremanju (uz termine snimka); ovo pokriva sve
    ostale puteve — CLI, uvoz SRT-a — po ISTOM podešavanju. Do kruga 23 su
    uvek kopirali podrazumevani, mimo podešavanja i bez traga na ekranu.
    Postojeći glosar posla (i prazan) se ne dira.
    """
    p = cfg.job_glossary_path(job_id)
    if not p.exists():
        use_default = cfg.new_project_glossary() == "default"
        cfg.write_json(p, cfg.load_glossary() if use_default else {})


def _snap_to_silences(job_id: str, cues: list, prof: Any, log: Callable[[str], None]) -> dict:
    """Vreme natpisa po tišini (krug 24, `linebreak.snap_to_silences`).

    Samo iz POSTOJEĆEG `pauses.json`: ovde se ne meri zvuk i ništa se ne
    upisuje. Bez fajla (stariji posao, zvuk obrisan pre kruga 19) mehanizam
    se preskače, a to piše u dnevniku i ispod „Za preuzimanje".
    """
    from . import linebreak
    from . import pauses as pauses_mod

    try:
        pf = pauses_mod.load(job_id)
    except Exception as exc:  # noqa: BLE001 — oštećen fajl nije razlog za pad
        log(f"vreme po tišini: preskočeno — {pauses_mod.PAUSES_FILE} se ne može pročitati ({exc})")
        return {"skipped": f"{pauses_mod.PAUSES_FILE} se ne može pročitati"}
    if pf is None:
        log(f"vreme po tišini: preskočeno — posao nema {pauses_mod.PAUSES_FILE}")
        return {"skipped": f"posao nema {pauses_mod.PAUSES_FILE}"}
    rep = linebreak.snap_to_silences(cues, pf.silences, prof, min_blank=linebreak.SNAP_MIN_BLANK)
    log(
        f"vreme po tišini: početak pomeren {len(rep.starts)}, kraj skraćen {len(rep.ends)}; "
        "odbijeno " + ", ".join(f"{k} {v}" for k, v in rep.rejected.items())
    )
    return {"starts": len(rep.starts), "ends": len(rep.ends), "rejected": rep.rejected}


def is_srt_job(job_id: str) -> bool:
    return (cfg.job_dir(job_id) / CUES_FILE).exists()


def default_profile_for(job_id: str) -> str:
    """Posao iz SRT-a meri prevod granicama samog originala; ostali — podešavanje."""
    from .fixedcues import ORIGINAL_PROFILE

    return ORIGINAL_PROFILE if is_srt_job(job_id) else cfg.default_profile()


def resolve_profile(job_id: str, name: str | None) -> Any:
    """Ime profila -> `Profile`. `original`/`original-max` se izvode iz
    ulaznog SRT-a i postoje samo za posao iz SRT-a."""
    from . import fixedcues

    name = name or default_profile_for(job_id)
    if name in fixedcues.SOURCE_PROFILES:
        if not is_srt_job(job_id):
            raise SubtitisError(
                f"Profil '{name}' se izvodi iz ulaznog SRT-a — ovaj projekat nije "
                "nastao iz SRT-a."
            )
        return fixedcues.source_profile(load_cues(job_id), name)
    return cfg.load_profile(name)


def load_cues(job_id: str):
    from .models import CuesFile

    return CuesFile.model_validate(cfg.read_json(cfg.job_dir(job_id) / CUES_FILE))


@dataclass
class ImportResult:
    job_id: str
    language: str
    cues: int
    sentences: int
    encoding: str
    original_script: str
    stats: Any = None  # fixedcues.ImportStats
    duration: float = 0.0


def run_import_srt(
    srt_path: str | Path,
    language: str,
    *,
    job_id: str | None = None,
    display_name: str | None = None,
    on_log: Log = _noop_log,
) -> ImportResult:
    """SRT -> cues.json + sentences.<lang>.json. Posao je odmah `transcribed`.

    Nema zvuka ni words.json. Ekavizacija i restauracija interpunkcije se NE
    pokreću — ulaz je ručno doteran. Za srpski se samo pismo prebacuje u
    latinicu, kao i svuda u pajplajnu (`STORAGE_SCRIPT`).
    """
    from . import fixedcues
    from . import srt as srt_mod
    from .models import CuesFile

    src = Path(srt_path)
    if not src.exists():
        raise SubtitisError(f"Ulazna datoteka ne postoji: {src}")
    name = display_name or src.name
    jid = job_id or cfg.make_job_id(Path(name))
    jdir = cfg.job_dir(jid)
    jdir.mkdir(parents=True, exist_ok=True)
    ensure_job_glossary(jid)

    text, encoding = srt_mod.decode(src.read_bytes())
    cues = [fixedcues.split_prefix(c) for c in srt_mod.parse(text)]
    original_script = translit.detect_script(
        " ".join(fixedcues.strip_markup(l) for c in cues for l in c.lines)
    )
    normalize = (
        (lambda t: translit.convert(t, STORAGE_SCRIPT))
        if translit.is_biscriptal(language) else None
    )
    duration = max((c.end_ms for c in cues), default=0) / 1000

    job = Job(
        job_id=jid, source_file=str(src.resolve()), language=language,
        status="created", duration=duration,
    )
    cfg.save_job(job)
    cfg.log(jid, f"ulaz: {src} (SRT, {encoding}), jezik {language}")

    with cfg.phase(job, "srt_import", encoding=encoding) as det:
        sentences, stats = fixedcues.assemble(cues, normalize=normalize)
        if translit.is_biscriptal(language):
            assert_single_script(
                [(s.id, s.text) for s in sentences], STORAGE_SCRIPT,
                what=f"sentences.{language}.json",
            )
        det["cues"] = len(cues)
        det["sentences"] = len(sentences)
        det["original_script"] = original_script
        det["forced_breaks"] = list(stats.forced_breaks)
        det["duration_breaks"] = list(stats.duration_breaks)
        det["gap_breaks"] = list(stats.gap_breaks)
        det["italic_cues"] = stats.italic_cues
        det["partial_italic"] = list(stats.partial_italic)
        det["dropped_tags"] = dict(stats.dropped_tags)
        det["empty_cues"] = list(stats.empty_cues)
        det["multi_cue_sentences"] = sum(1 for s in sentences if len(s.cue_parts) > 1)
        per_cue: dict[int, int] = {}
        for s in sentences:
            for p in s.cue_parts:
                per_cue[p.cue] = per_cue.get(p.cue, 0) + 1
        det["shared_cues"] = sum(1 for n in per_cue.values() if n > 1)
        measured = CuesFile(job_id=jid, source_file=str(src), language=language, cues=cues)
        det["source_profile"] = {
            name: {
                "max_chars_per_line": pr.max_chars_per_line,
                "max_cps": pr.max_cps,
                "max_lines": pr.max_lines,
            }
            for name in fixedcues.SOURCE_PROFILES
            for pr in [fixedcues.source_profile(measured, name)]
        }

    cfg.write_model(
        jdir / CUES_FILE,
        CuesFile(
            job_id=jid, source_file=str(src.resolve()), language=language,
            script=original_script if original_script != "mixed" else "Latn",
            encoding=encoding, cues=cues,
        ),
    )
    cfg.write_model(
        jdir / f"sentences.{language}.json",
        SentencesFile(
            job_id=jid, language=language, script=STORAGE_SCRIPT,
            derived_from=CUES_FILE, edited_by_human=False, sentences=sentences,
        ),
    )

    def say(msg: str) -> None:
        cfg.log(jid, msg)
        on_log(msg)

    say(
        f"SRT: {len(cues)} cue-ova -> {len(sentences)} rečenica "
        f"({det['multi_cue_sentences']} preko više cue-ova, "
        f"{det['shared_cues']} cue-ova sa više rečenica); kodiranje {encoding}"
    )
    if encoding not in ("utf-8", "utf-8-sig"):
        say(f"NAPOMENA: fajl nije UTF-8, pročitan kao {encoding}")
    if original_script == "Cyrl" or original_script == "mixed":
        say(f"pismo ulaza {original_script}; rečenice se drže u latinici")
    say(fixedcues.source_profile_line(measured))
    if stats.forced_breaks:
        say(
            f"UPOZORENJE: {len(stats.forced_breaks)} rečenica bez završnog znaka "
            f"presečeno posle {fixedcues.MAX_CUES_PER_SENTENCE} cue-ova "
            f"(cue {', '.join(map(str, stats.forced_breaks[:10]))})"
        )
    if stats.duration_breaks:
        say(
            f"{len(stats.duration_breaks)} rečenica duža od "
            f"{fixedcues.SRT_MAX_SENTENCE_DURATION:g} s presečeno na najvećoj pauzi "
            f"(posle cue {', '.join(map(str, stats.duration_breaks[:10]))})"
        )
    if stats.gap_breaks:
        say(
            f"{len(stats.gap_breaks)} rečenica presečeno na pauzi ≥ "
            f"{fixedcues.SRT_MAX_GAP:g} s (posle cue "
            f"{', '.join(map(str, stats.gap_breaks[:10]))})"
        )
    if stats.partial_italic:
        say(
            f"UPOZORENJE: kurziv na delu rečenice u cue-ovima "
            f"{', '.join(map(str, stats.partial_italic[:10]))} — posle prevoda ga "
            "ne mogu vratiti na isto mesto; te cue-ove proveri ručno"
        )
    if stats.dropped_tags:
        say(
            "UPOZORENJE: oznake koje nisu kurziv se ne prenose: "
            + ", ".join(f"<{k}> ×{n}" for k, n in sorted(stats.dropped_tags.items()))
        )

    job.status = "transcribed"
    cfg.save_job(job)
    return ImportResult(
        job_id=jid, language=language, cues=len(cues), sentences=len(sentences),
        encoding=encoding, original_script=original_script, stats=stats,
        duration=duration,
    )


# ==========================================================================
# Prevod
# ==========================================================================


#: Flag koji `translate` stavlja na rečenicu koja je ostala na originalu.
TRANSLATION_FAILED_FLAG = untranslated_mod.TRANSLATION_FAILED_FLAG


def assert_failed_is_original(translated: list[Any], source: list[Any]) -> None:
    """Rečenica sa `translation_failed` mora nositi CEO original, bez ijedne reči
    iz pokušaja prevoda. Pomešan tekst u titlu je gori od neprevedenog, pa ovo
    puca pre upisa umesto da prođe tiho."""
    src = {s.id: s.text for s in source}
    bad = [
        s.id for s in translated
        if TRANSLATION_FAILED_FLAG in s.flags and s.text != src.get(s.id, s.text)
    ]
    if bad:
        raise SubtitisError(
            f"Interna greška: rečenice {bad} nose {TRANSLATION_FAILED_FLAG}, a tekst "
            "im nije original — pokušaj prevoda je procureo. Ništa nije upisano."
        )


def untranslated_report(
    job_id: str, sf: SentencesFile, cue_map: dict[int, list[int]] | None = None,
) -> untranslated_mod.Report | None:
    """Neprevedene i sumnjive rečenice prevoda `sf`; None ako `sf` nije prevod."""
    if not sf.derived_from.startswith("sentences."):
        return None
    src_path = cfg.job_dir(job_id) / sf.derived_from
    if not src_path.exists():
        return None
    src = SentencesFile.model_validate(cfg.read_json(src_path))
    return untranslated_mod.check(
        src.sentences, sf.sentences, src.language, sf.language, cue_map
    )


@dataclass
class HopResult:
    src: str
    tgt: str
    path: Path
    sentences: int
    requests: int
    chunks_ok: int
    chunks_failed: int
    failed_ids: list[int]
    glossary_entries: int
    glossary_derived: int
    variant: str | None
    thinking_responses: int
    #: Uzastopna ponavljanja reči/fraza, izmerena pre i posle prevoda. Samo
    #: merenje — ništa u pajplajnu se po njemu ne ponaša drugačije.
    repeats_src: repeats.Repeats | None = None
    repeats_tgt: repeats.Repeats | None = None
    warning: str | None = None
    #: untranslated.Report: odustajanja + rečenice koje liče na original
    untranslated: Any = None


def run_translate(
    job_id: str,
    target: str,
    *,
    source: str | None = None,
    model: str | None = None,
    url: str | None = None,
    pivot: str | None = None,
    variant: str | None = None,
    chunk_size: int | None = None,
    glossary: str | Path | None = None,
    force: bool = False,
    on_log: Log = _noop_log,
    on_progress: Progress = _noop_progress,
) -> list[HopResult]:
    """sentences.<src>.json -> sentences.<tgt>.json, sa opcionim pivotom."""
    from . import translate as tr

    job = cfg.load_job(job_id)
    src = source or job.language or "sr"
    if src == target:
        raise SubtitisError(f"Izvorni i ciljni jezik su isti ({src}).")

    if pivot is None:
        via = cfg.resolve_pivot(src, target)
    elif str(pivot).strip().lower() in ("off", "none", "-", ""):
        via = None
    else:
        via = str(pivot).strip().lower()
    if via in (src, target):
        via = None

    client = tr.OllamaClient(
        url=url or cfg.DEFAULT_OLLAMA_URL,
        model=model or cfg.llm_model(),
        log=lambda m: (cfg.log(job_id, m), on_log(m)),
    )
    client.check()

    gloss = cfg.load_glossary(glossary) if glossary else cfg.job_glossary(job_id)
    hops = [(src, via), (via, target)] if via else [(src, target)]
    # Posao iz SRT-a: prevod se upisuje u zadate cue-ove, pa model dobija
    # pravilo da drži dužinu originala. Važi samo ovde, nikad za prevod govora.
    fit_length = is_srt_job(job_id)
    if fit_length:
        cfg.log(job_id, "SRT režim: prevod se upisuje u postojeće cue-ove")
    cfg.log(
        job_id,
        f"prevod {src} -> {target}" + (f" preko {via}" if via else " (direktno)")
        + f", model {client.model}",
    )

    out: list[HopResult] = []
    try:
        for n, (hop_src, hop_tgt) in enumerate(hops):
            on_progress("translate", n / len(hops))
            sf = SentencesFile.model_validate(
                cfg.read_json(cfg.job_dir(job_id) / f"sentences.{hop_src}.json")
            )
            out_path = cfg.job_dir(job_id) / f"sentences.{hop_tgt}.json"
            if out_path.exists():
                existing = SentencesFile.model_validate(cfg.read_json(out_path))
                if existing.edited_by_human and not force:
                    raise SubtitisError(
                        f"{out_path.name} je označen kao ručno menjan "
                        "(edited_by_human=true). Ponovni prevod bi pregazio ispravke."
                    )
                bak = cfg.backup(out_path)
                if bak:
                    cfg.log(job_id, f"prethodni prevod sačuvan u {bak.name}")

            hop_gloss, derived = cfg.hop_glossary(gloss, hop_src, hop_tgt, origin=src)
            if derived:
                cfg.log(
                    job_id,
                    f"glosar {hop_src}->{hop_tgt}: {derived} unosa izvedeno iz "
                    f"{src}->{hop_src} i {src}->{hop_tgt}",
                )
            if variant is None:
                hop_variant = cfg.target_variant(hop_tgt)
            elif str(variant).strip().lower() in ("off", "none", "-"):
                hop_variant = None
            else:
                hop_variant = str(variant).strip()

            warning = None
            if not hop_gloss:
                warning = (
                    f"glosar nema nijedan unos za {hop_src}->{hop_tgt}; "
                    "prevodni prompt ide bez rečnika."
                )
                if via and hop_src == via:
                    warning += (
                        f" Ovo je DRUGA polovina pivota — termini ugovoreni za "
                        f"{src}->{hop_tgt} se ovde gube."
                    )
                cfg.log(job_id, f"UPOZORENJE: {warning}")
                on_log(f"UPOZORENJE: {warning}")

            with cfg.phase(
                job, f"translate:{hop_src}->{hop_tgt}", model=client.model,
                fit_length=fit_length,
            ) as det:
                translated, stats = tr.translate_sentences(
                    sf.sentences, hop_src, hop_tgt,
                    client=client,
                    glossary_pairs=hop_gloss,
                    variant=hop_variant,
                    fit_length=fit_length,
                    chunk_size=chunk_size or tr.CHUNK_SIZE,
                    log=lambda m: (cfg.log(job_id, m), on_log(m)),
                )
                det["sentences"] = stats.sentences
                det["chunks_ok"] = stats.chunks_ok
                det["chunks_failed"] = stats.chunks_failed
                det["failed_ids"] = stats.failed_ids
                det["quotes_rejected"] = stats.quotes_rejected
                if stats.quotes_rejected:
                    msg = (
                        f"navodnici: {stats.quotes_rejected} prevod(a) odbijeno "
                        "zbog neuravnoteženih navodnika i ponovljeno"
                    )
                    cfg.log(job_id, msg)
                    on_log(msg)
                det["glossary_entries"] = len(hop_gloss)
                det["variant"] = hop_variant

                # Transliteracija ide OVDE: jednim prolazom nad celim prevodom,
                # kad su svi čankovi gotovi. Model pravilo „piši latinicom"
                # poštuje po odgovoru, pa ume da vrati ćirilicu za jedan čank a
                # latinicu za sledeći — a čank je devet rečenica, pa presek
                # ispadne oštar i liči na to da je transliteracija preskočila
                # deo teksta.
                if translit.is_biscriptal(hop_tgt):
                    fixed = translit.convert_all(
                        [s.text for s in translated], STORAGE_SCRIPT
                    )
                    changed = sum(
                        1 for s, t in zip(translated, fixed) if s.text != t
                    )
                    if changed:
                        msg = (
                            f"model je za {changed} od {len(translated)} "
                            f"rečenica vratio drugo pismo; ceo prevod prebačen "
                            f"u {STORAGE_SCRIPT} jednim prolazom"
                        )
                        cfg.log(job_id, msg)
                        on_log(msg)
                    det["script_normalized"] = changed
                    translated = [
                        s.model_copy(update={"text": t})
                        for s, t in zip(translated, fixed)
                    ]
                    assert_single_script(
                        [(s.id, s.text) for s in translated],
                        STORAGE_SCRIPT,
                        what=f"sentences.{hop_tgt}.json",
                    )

                # Uzastopna udvajanja ("jer, jer"): mere se i na ulazu i na
                # izlazu, jer sam broj na prevodu ne kaže ništa. Pitanje je da
                # li model govorno OKLEVANJE prenosi, uklanja ili dodaje svoje;
                # udvojeno pojačavanje ("veoma, veoma") se broji odvojeno jer
                # ono treba da se prenese i nije kvar.
                rep_src = repeats.measure_all(
                    [s.text for s in sf.sentences], hop_src
                )
                rep_tgt = repeats.measure_all(
                    [s.text for s in translated], hop_tgt
                )
                det["repeats_src"] = rep_src.hesitation
                det["repeats_tgt"] = rep_tgt.hesitation
                det["emphasis_src"] = rep_src.emphasis
                det["emphasis_tgt"] = rep_tgt.emphasis
                line = repeats.compare_line(
                    rep_src, rep_tgt, src_label=hop_src, tgt_label=hop_tgt
                )
                cfg.log(job_id, line)
                on_log(line)
                # Bez spiska reči za jezik sve ispada nerazvrstano; tad brojka
                # iznad ne meri ono što tvrdi i to mora da se vidi.
                for lang in (hop_src, hop_tgt):
                    if not repeats.known_language(lang):
                        note = (
                            f"udvajanja: nema spiska reči za '{lang}', pa se "
                            "oklevanje i pojačavanje ne razdvajaju"
                        )
                        cfg.log(job_id, note)
                        on_log(note)

                assert_failed_is_original(translated, sf.sentences)
                report = untranslated_mod.check(
                    sf.sentences, translated, hop_src, hop_tgt
                )
                det["untranslated"] = report.as_dict()
                for line in untranslated_mod.lines(report, hop_tgt):
                    cfg.log(job_id, line)
                    on_log(line)

            cfg.write_model(
                out_path,
                SentencesFile(
                    job_id=job_id, language=hop_tgt, script=STORAGE_SCRIPT,
                    derived_from=f"sentences.{hop_src}.json",
                    edited_by_human=False, sentences=translated,
                ),
            )
            out.append(
                HopResult(
                    src=hop_src, tgt=hop_tgt, path=out_path,
                    sentences=stats.sentences, requests=stats.calls.requests,
                    chunks_ok=stats.chunks_ok, chunks_failed=stats.chunks_failed,
                    failed_ids=list(stats.failed_ids),
                    glossary_entries=len(hop_gloss), glossary_derived=derived,
                    variant=hop_variant,
                    thinking_responses=stats.calls.thinking_responses,
                    repeats_src=rep_src, repeats_tgt=rep_tgt,
                    untranslated=report,
                    warning=warning,
                )
            )
    finally:
        client.unload()
        cfg.log(job_id, "Ollama model oslobođen iz VRAM-a")

    on_progress("translate", 1.0)
    return out


# ==========================================================================
# Prelamanje u SRT
# ==========================================================================


@dataclass
class BuildResult:
    path: Path
    cues: int
    over_limit: int
    over_cps: int
    low_confidence: int
    translation_failed: int
    degraded: list[int]
    interpolated: bool
    #: Samo za posao iz SRT-a: cue-ovi koji probijaju profil, najgori prvi.
    overflows: list[Any] = field(default_factory=list)
    italic_lost: list[int] = field(default_factory=list)
    untranslated: Any = None  # untranslated.Report ili None za izvorni jezik


def fixed_build(
    source: Any, sf: SentencesFile, prof: Any, lang: str, script: str = "Latn",
    *, sentences: list[Any] | None = None, log: Log = _noop_log,
) -> Any:
    """Cue-ovi posla iz SRT-a, tačno kako idu u fajl (i pravopis jezika).

    `sentences` zamenjuje rečenice iz `sf` — za proveru predloga pre upisa.
    """
    from . import fixedcues

    built = fixedcues.build(
        source, sentences if sentences is not None else sf.sentences, prof,
        language=sf.language,
        script=script if translit.is_biscriptal(lang) else None,
        exact=sf.derived_from == CUES_FILE,
        glossary_terms=cfg.glossary_terms(cfg.job_glossary(sf.job_id), sf.language),
        log=log,
    )
    # Pravopisni razmaci PRE izveštaja: nerazlomljiv razmak je znak, pa se
    # prekoračenja računaju nad konačnim tekstom.
    for cue in built.cues:
        cue.lines = typography.apply(cue.lines, lang)
    return built


def _run_build_fixed(
    job_id: str, job: Job, prof: Any, sf: SentencesFile, lang: str, script: str,
    out: str | Path | None, on_log: Log, case: str = "normal", drop_punct: bool = False,
) -> BuildResult:
    """Prelamanje za posao iz SRT-a: cue-ovi i vremena su iz ulaza, nepromenjeni."""
    from . import fixedcues, srt as srt_mod

    source = load_cues(job_id)

    def logger(m: str) -> None:
        cfg.log(job_id, m)
        on_log(m)

    biscriptal = translit.is_biscriptal(lang)

    def build_with(p: Any, log: Log) -> Any:
        return fixed_build(source, sf, p, lang, script, log=log)

    with cfg.phase(
        job, "build", profile=prof.name, script=script, language=lang,
        fixed_cues=True, case=case, drop_punct=drop_punct,
    ) as det:
        built = build_with(prof, logger)
        if biscriptal:
            assert_single_script(
                [(c.index, fixedcues.strip_markup(l)) for c in built.cues for l in c.lines],
                script,
                what=f"SRT {lang}/{script}",
            )
        cues = fixedcues.with_prefixes(built.cues, source)
        # Tvrda garancija režima: isti broj cue-ova, ista vremena. Ako ovo
        # ikad pukne, greška je u kodu — i bolje da pukne nego da izađe titl
        # koji tiho beži od slike.
        if srt_mod.timing(cues) != srt_mod.timing(source.cues):
            raise SubtitisError(
                "Interna greška: tajming izlaza se razlikuje od ulaznog SRT-a. "
                "SRT nije zapisan."
            )
        found = fixedcues.overflows(built, prof, source)
        det["cues"] = len(cues)
        det["unchanged"] = built.unchanged
        det["overflow_count"] = len(found)
        det["overflow"] = [o.as_dict() for o in found]
        det["italic_lost"] = list(built.italic_lost)
        det["empty"] = list(built.empty)
        report = untranslated_report(job_id, sf)
        det["untranslated"] = report.as_dict() if report else None

        # Kad probija SAMO brzina, a ne dužina reda, izbor profila je stvarna
        # odluka — pa se broj za blaži profil RAČUNA punim prelamanjem po
        # njemu, ne procenjuje iz postojećih cue-ova.
        suggestions: list[dict[str, Any]] = []
        if found and all(not o.chars_over and not o.lines_over for o in found):
            names = list(fixedcues.SOURCE_PROFILES) + sorted(
                p.stem for p in cfg.PROFILES_DIR.glob("*.json")
            )
            for name in names:
                other = resolve_profile(job_id, name)
                if other.name == prof.name or other.max_cps <= prof.max_cps:
                    continue
                alt = build_with(other, lambda m: None)
                suggestions.append({
                    "profile": other.name, "max_cps": other.max_cps,
                    "max_chars": other.max_chars_per_line,
                    "overflow": len(fixedcues.overflows(alt, other, source)),
                })
        det["profile_suggestions"] = suggestions

    for line in untranslated_mod.lines(report, lang) if report else []:
        logger(line)
    for line in fixedcues.overflow_lines(found, len(cues), prof):
        logger(line)
    for sug in suggestions:
        logger(
            f"predlog: sva prekoračenja su CPS — profil {sug['profile']} "
            f"({sug['max_cps']:g} CPS, {sug['max_chars']} zn./red) bi ih smanjio "
            f"sa {len(found)} na {sug['overflow']} (izračunato prelamanjem po njemu)"
        )
    if built.italic_lost:
        logger(
            f"kurziv nije vraćen (bio je na delu rečenice): cue "
            f"{', '.join(map(str, built.italic_lost[:10]))}"
            + (" …" if len(built.italic_lost) > 10 else "")
        )
    if built.empty:
        logger(
            f"UPOZORENJE: {len(built.empty)} cue-ova je ostalo bez teksta "
            f"(prevod ima manje reči nego original cue-ova): "
            f"{', '.join(map(str, built.empty[:10]))}"
        )

    stem = srt_mod.srt_stem(Path(job.source_file).name)
    path = Path(out) if out else (
        cfg.job_dir(job_id) / "out" / srt_mod.srt_name(stem, lang, script)
    )
    final = _output(cues, case, drop_punct)
    srt_mod.write_srt(path, final)
    if not out:
        by_cue: dict[int, list[int]] = {}
        for s in sf.sentences:
            for part in s.cue_parts:
                by_cue.setdefault(part.cue, []).append(s.id)
        marks = {o.cue - 1: ("over" if o.worse_than_source else "over-same") for o in found}
        write_preview(job_id, lang, path, final,
                      lambda i, c: by_cue.get(i, []), lambda i, c: marks.get(i))
    return BuildResult(
        path=path,
        cues=len(cues),
        over_limit=sum(1 for o in found if o.chars_over),
        over_cps=sum(1 for o in found if o.cps_over),
        low_confidence=0,
        translation_failed=len(report.failed) if report else 0,
        degraded=[],
        interpolated=False,
        overflows=found,
        italic_lost=list(built.italic_lost),
        untranslated=report,
    )


#: Prelom po rečenici za editor (krug 12): cue-ovi TAČNO kako su u fajlu, uz
#: rečenicu iz koje su nastali. Editor uređuje rečenice, a titl se iz njih
#: pravi iznova — bez ovoga se profil, pismo i slova nigde ne vide.
PREVIEW_FILE = "preview.{lang}.json"


def preview_path(job_id: str, lang: str) -> Path:
    return cfg.job_dir(job_id) / PREVIEW_FILE.format(lang=lang)


def write_preview(
    job_id: str, lang: str, srt_path: Path, cues: list[Any],
    sentences_of: Any, mark_of: Any,
) -> None:
    """`sentences_of(i, cue)` -> lista id-jeva rečenica; `mark_of(i, cue)` ->
    `over` / `over-same` / `short` / None (boje kao u izveštaju)."""
    cfg.write_json(preview_path(job_id, lang), {
        "srt": srt_path.name,
        "cues": [
            {"n": i + 1, "s": c.s, "e": c.e, "lines": list(c.lines),
             "sentences": sentences_of(i, c), "mark": mark_of(i, c)}
            for i, c in enumerate(cues)
        ],
    })


def _output(cues: list[Any], case: str, drop_punct: bool) -> list[Any]:
    """Kopije cue-ova sa izlaznim transformacijama (`fixedcues.output_lines`):
    veličina slova, pa izostavljena interpunkcija. Oznake (`<i>`) se ne diraju."""
    from .fixedcues import output_lines

    if case == "normal" and not drop_punct:
        return cues
    return [
        c.model_copy(update={"lines": output_lines(c.lines, case, drop_punct)}) for c in cues
    ]


def run_build(
    job_id: str,
    *,
    profile: str | None = None,
    script: str = "Latn",
    language: str | None = None,
    out: str | Path | None = None,
    on_log: Log = _noop_log,
    case: str | None = None,
    drop_punct: bool | None = None,
) -> BuildResult:
    """words/sentences -> SRT. Ne pokreće ASR ponovo.

    `case` (normal/upper/lower) i `drop_punct` menjaju samo izgled izlaznog
    fajla — primenjuju se pri pisanju, posle transliteracije i brojanja
    prekoračenja. Kad nisu zadati, uzimaju podrazumevano iz profila
    (`Profile.case`, `Profile.drop_punctuation`).
    """
    from . import linebreak, srt as srt_mod
    from .translit import convert

    job = cfg.load_job(job_id)
    lang = language or job.language or "sr"
    if script not in ("Latn", "Cyrl"):
        raise SubtitisError("script mora biti 'Latn' ili 'Cyrl'.")
    from .fixedcues import CASES

    prof = resolve_profile(job_id, profile)
    case = case or getattr(prof, "case", "normal")
    drop_punct = getattr(prof, "drop_punctuation", False) if drop_punct is None else drop_punct
    if case not in CASES:
        raise SubtitisError(f"veličina slova mora biti jedno od: {', '.join(CASES)}.")
    sf = SentencesFile.model_validate(
        cfg.read_json(cfg.job_dir(job_id) / f"sentences.{lang}.json")
    )
    if is_srt_job(job_id):
        return _run_build_fixed(
            job_id, job, prof, sf, lang, script, out, on_log, case, drop_punct
        )
    is_translation = sf.derived_from.startswith("sentences.")
    words: list = []
    if not is_translation:
        wf = WordsFile.model_validate(
            cfg.read_json(cfg.job_dir(job_id) / f"words.{lang}.json")
        )
        words = wf.words

    def logger(m: str) -> None:
        cfg.log(job_id, m)
        on_log(m)

    silences = None
    if not is_translation and linebreak.pauses_enabled():
        from . import pauses as pauses_mod

        pf = pauses_mod.ensure(job_id, logger)
        silences = pf.silences if pf is not None else None

    with cfg.phase(
        job, "build", profile=prof.name, script=script, language=lang,
        interpolated=is_translation, case=case, drop_punct=drop_punct,
    ) as det:
        cues = linebreak.build_cues(
            sf.sentences, words, prof, language=sf.language,
            log=logger, interpolate_times=is_translation,
            glossary_terms=cfg.glossary_terms(cfg.job_glossary(job_id), sf.language),
            silences=silences,
        )
        if linebreak.pauses_enabled() and not is_translation:
            det["pauses"] = silences is not None
        if linebreak.SNAP_MIN_BLANK is not None and not is_translation:
            det["silence_timing"] = _snap_to_silences(job_id, cues, prof, logger)
        # I za `Latn`, ne samo za `Cyrl`: ulazna datoteka ume da bude mešana
        # (stariji poslovi, ručne ispravke), a `convert` je idempotentan — nad
        # tekstom koji je već u traženom pismu ne menja ništa.
        if translit.is_biscriptal(lang):
            for cue in cues:
                cue.lines = [convert(l, script) for l in cue.lines]
            assert_single_script(
                [(c.index, l) for c in cues for l in c.lines],
                script,
                what=f"SRT {lang}/{script}",
            )
        # Posle prelamanja, pre brojanja prekoračenja (vidi `typography`).
        for cue in cues:
            cue.lines = typography.apply(cue.lines, lang)
        det["cues"] = len(cues)
        degraded = [
            s.id for s in sf.sentences if linebreak.ALIGNMENT_DEGRADED_FLAG in s.flags
        ]
        det["alignment_degraded"] = len(degraded)
        short = linebreak.too_short(cues, prof)
        det["too_short"] = [
            {"cue": c.index, "s": c.s, "e": c.e, "dur": round(c.e - c.s, 3)} for c in short
        ]
        cue_map: dict[int, list[int]] = {}
        for c in cues:
            if c.sentence_id is not None:
                cue_map.setdefault(c.sentence_id, []).append(c.index)
        report = untranslated_report(job_id, sf, cue_map)
        det["untranslated"] = report.as_dict() if report else None

    for line in untranslated_mod.lines(report, lang) if report else []:
        logger(line)
    if short:
        logger(
            f"prekratki cue-ovi: {len(short)} od {len(cues)} kraći od "
            f"{prof.min_duration:g} s (profil {prof.name}) i posle produžavanja — "
            "nema mesta ni napred ni nazad: "
            + "; ".join(
                f"#{c.index} {srt_mod.timestamp(c.s)} ({c.e - c.s:.2f} s)" for c in short
            )
        )

    if degraded:
        cfg.write_model(cfg.job_dir(job_id) / f"sentences.{lang}.json", sf)

    stem = Path(job.source_file).stem
    path = Path(out) if out else (
        cfg.job_dir(job_id) / "out" / srt_mod.srt_name(stem, lang, script)
    )
    final = _output(cues, case, drop_punct)
    srt_mod.write_srt(path, final)
    if not out:
        short_ids = {id(c) for c in short}

        def mark(i: int, c: Any) -> str | None:
            src = cues[i]
            if any(len(l) > prof.max_chars_per_line for l in src.lines) or src.cps > prof.max_cps:
                return "over"
            return "short" if id(src) in short_ids else None

        write_preview(job_id, lang, path, final,
                      lambda i, c: [c.sentence_id] if c.sentence_id is not None else [], mark)

    return BuildResult(
        path=path,
        cues=len(cues),
        over_limit=sum(
            1 for c in cues if any(len(l) > prof.max_chars_per_line for l in c.lines)
        ),
        over_cps=sum(1 for c in cues if c.cps > prof.max_cps),
        low_confidence=sum(1 for c in cues if "low_confidence" in c.flags),
        translation_failed=sum(1 for c in cues if "translation_failed" in c.flags),
        degraded=degraded,
        interpolated=is_translation,
        untranslated=report,
    )


# ==========================================================================
# Skraćivanje prekoračenja — predlog, ne upis
# ==========================================================================

#: Predlozi skraćivanja za jezik. Prevod se NE menja dok čovek ne potvrdi
#: pojedinačne izmene u editoru — skraćivanje je gubitak sadržaja.
SHORTEN_FILE = "shorten.{lang}.json"


def shorten_path(job_id: str, lang: str) -> Path:
    return cfg.job_dir(job_id) / SHORTEN_FILE.format(lang=lang)


@dataclass
class ShortenResult:
    path: Path
    attempted: int
    accepted: int
    rejected: dict[str, int]
    #: cue-ovi gori od originala koji sa prihvaćenim predlozima više nisu preko profila
    cues_fixed: int
    cues_worse_before: int
    avg_cut: float  # prosečno skraćenje prihvaćenih, u znakovima
    avg_cut_pct: float
    items: list[dict[str, Any]] = field(default_factory=list)


def _has_term(text: str, term: str) -> bool:
    norm = lambda t: t.lower().replace("’", "'")  # noqa: E731
    return norm(term) in norm(text)


def _last_build_profile(job: Job, lang: str) -> str | None:
    for rec in reversed(job.history):
        if rec.phase == "build" and rec.status == "ok" and rec.detail.get("language") == lang:
            return rec.detail.get("profile")
    return None


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_shorten(
    job_id: str,
    language: str,
    *,
    profile: str | None = None,
    model: str | None = None,
    url: str | None = None,
    out: str | Path | None = None,
    on_log: Log = _noop_log,
) -> ShortenResult:
    """Drugi prolaz kroz model, samo za rečenice u cue-ovima gorim od originala.

    Ništa ne upisuje u `sentences.<lang>.json`: rezultat je `shorten.<lang>.json`
    sa starom i novom verzijom, koji čovek pregleda u editoru. Predlog je
    „prihvaćen" samo ako je stvarno kraći I ako je najveći CPS u cue-ovima te
    rečenice pao — provereno ponovnim prelamanjem, ne procenom.
    """
    import math

    from . import fixedcues
    from . import translate as tr

    if not is_srt_job(job_id):
        raise SubtitisError("Skraćivanje postoji samo za projekat nastao iz SRT-a.")
    job = cfg.load_job(job_id)
    sf = SentencesFile.model_validate(
        cfg.read_json(cfg.job_dir(job_id) / f"sentences.{language}.json")
    )
    if not sf.derived_from.startswith("sentences."):
        raise SubtitisError(f"{language} nije prevod — nema šta da se skraćuje.")
    src_sf = SentencesFile.model_validate(
        cfg.read_json(cfg.job_dir(job_id) / sf.derived_from)
    )
    source = load_cues(job_id)
    prof = resolve_profile(job_id, profile or _last_build_profile(job, language))

    def say(m: str) -> None:
        cfg.log(job_id, m)
        on_log(m)

    built = fixed_build(source, sf, prof, language)
    worse = [o for o in fixedcues.overflows(built, prof, source) if o.worse_than_source]

    # N iz viška CPS-a: koliko znakova cue ima preko `max_cps * trajanje`.
    # Cue sa više rečenica deli višak na jednake delove.
    need: dict[int, float] = {}
    for o in worse:
        cue = built.cues[o.cue - 1]
        excess = max(o.cps_over * (cue.e - cue.s), float(o.chars_over))
        for sid in o.sentences:
            need[sid] = need.get(sid, 0.0) + excess / len(o.sentences)

    by_id = {s.id: s for s in sf.sentences}
    src_by_id = {s.id: s for s in src_sf.sentences}
    items = [
        tr.ShortenItem(
            id=sid, text=by_id[sid].text,
            source=src_by_id[sid].text if sid in src_by_id else "",
            need=max(1, math.ceil(n)),
        )
        for sid, n in sorted(need.items())
        if sid in by_id and tr.TRANSLATION_FAILED_FLAG not in by_id[sid].flags
    ]

    proposals: dict[int, str] = {}
    if items:
        client = tr.OllamaClient(
            url=url or cfg.DEFAULT_OLLAMA_URL, model=model or cfg.llm_model(),
            log=lambda m: (cfg.log(job_id, m), on_log(m)),
        )
        client.check()
        gloss, _ = cfg.hop_glossary(
            cfg.job_glossary(job_id), src_sf.language, language, origin=src_sf.language
        )
        try:
            with cfg.phase(job, f"shorten:{language}", model=client.model,
                           profile=prof.name) as det:
                proposals, _stats = tr.shorten_sentences(
                    items, src_sf.language, language, client=client,
                    glossary_pairs=gloss, log=say,
                )
                det["attempted"] = len(items)
                det["returned"] = len(proposals)
        finally:
            client.unload()

    def cue_cps(b: Any, sid: int) -> float:
        return max(
            fixedcues.measure_cue(b.cues[p.cue].lines, b.cues[p.cue].s, b.cues[p.cue].e).cps
            for p in by_id[sid].cue_parts
        )

    terms = cfg.glossary_terms(cfg.job_glossary(job_id), language)

    # Provera prelamanjem: svaki predlog pojedinačno, pa svi prihvaćeni zajedno.
    records: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for it in items:
        new = proposals.get(it.id)
        rec: dict[str, Any] = {
            "sentence": it.id,
            "cues": [p.cue + 1 for p in by_id[it.id].cue_parts],
            "old": it.text, "new": new, "source": it.source, "need": it.need,
            "cps_before": round(cue_cps(built, it.id), 2),
        }
        lost = [t for t in terms if _has_term(it.text, t) and not _has_term(new or "", t)]
        if new is None:
            rec["status"], rec["reason"] = "rejected", "model nije vratio ispravan odgovor"
        elif lost:
            # Termin iz glosara je ugovoren prevod; skraćen ili izbačen je
            # izgubljen sadržaj (`Open Door Workshop` -> `Open Door`).
            rec["status"], rec["reason"] = "rejected", "izgubljen termin iz glosara"
            rec["lost_terms"] = lost
        elif len(new) >= len(it.text):
            rec["status"], rec["reason"] = "rejected", "nije kraće"
        else:
            trial = [s.model_copy(update={"text": new}) if s.id == it.id else s
                     for s in sf.sentences]
            after = cue_cps(fixed_build(source, sf, prof, language, sentences=trial), it.id)
            rec["cps_after"] = round(after, 2)
            if after >= rec["cps_before"] - 1e-9:
                rec["status"], rec["reason"] = "rejected", "CPS nije pao"
            else:
                rec["status"], rec["reason"] = "accepted", ""
                rec["cut"] = len(it.text) - len(new)
                rec["short_of_need"] = rec["cut"] < it.need
                # Preko traženog opsega. Izmereno 2026-09-28: gotovo sav gubitak
                # značenja je u ovim predlozima, a opseg u promptu ih smanjuje
                # tek malo (medijana 2,7x -> 2,4x traženog). U editoru su zato
                # bez kvačice — rizičnu izmenu čovek mora sam da uključi.
                rec["overcut"] = rec["cut"] > it.need + tr.shorten_slack(it.need)
        if rec["status"] == "rejected":
            rejected[rec["reason"]] = rejected.get(rec["reason"], 0) + 1
        records.append(rec)

    accepted = [r for r in records if r["status"] == "accepted"]
    applied = {r["sentence"]: r["new"] for r in accepted}
    trial = [s.model_copy(update={"text": applied[s.id]}) if s.id in applied else s
             for s in sf.sentences]
    after_all = fixed_build(source, sf, prof, language, sentences=trial)
    still = {o.cue for o in fixedcues.overflows(after_all, prof, source)}
    worse_cues = {o.cue for o in worse}
    for r in accepted:
        r["cues_fixed"] = [c for c in r["cues"] if c in worse_cues and c not in still]

    cuts = [r["cut"] for r in accepted]
    pcts = [r["cut"] / len(r["old"]) for r in accepted]
    result = ShortenResult(
        path=Path(out) if out else shorten_path(job_id, language),
        attempted=len(items), accepted=len(accepted), rejected=rejected,
        cues_fixed=len(worse_cues - still), cues_worse_before=len(worse_cues),
        avg_cut=sum(cuts) / len(cuts) if cuts else 0.0,
        avg_cut_pct=100 * sum(pcts) / len(pcts) if pcts else 0.0,
        items=records,
    )
    cfg.write_json(result.path, {
        "job_id": job_id, "language": language, "profile": prof.name,
        "created_at": _now(), "attempted": result.attempted,
        "accepted": result.accepted, "rejected": rejected,
        "cues_fixed": result.cues_fixed, "cues_worse_before": result.cues_worse_before,
        "avg_cut": round(result.avg_cut, 1), "avg_cut_pct": round(result.avg_cut_pct, 1),
        "items": records,
    })
    for line in shorten_lines(result):
        say(line)
    return result


def shorten_lines(r: ShortenResult) -> list[str]:
    if not r.attempted:
        return ["skraćivanje: nema cue-ova gorih od originala — nema šta da se skraćuje"]
    rej = ", ".join(f"{why}: {n}" for why, n in sorted(r.rejected.items())) or "nijedan"
    return [
        f"skraćivanje: pokušano {r.attempted} rečenica, prihvaćeno {r.accepted} "
        f"(odbijeno — {rej})",
        f"skraćivanje: iz prekoračenja izlazi {r.cues_fixed} od {r.cues_worse_before} "
        f"cue-ova gorih od originala; prosečno skraćenje {r.avg_cut:.1f} zn. "
        f"({r.avg_cut_pct:.0f}%)",
        "skraćivanje: NIŠTA NIJE UPISANO — predlozi čekaju pregled u editoru",
    ]
