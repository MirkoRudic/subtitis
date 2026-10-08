"""Radna petlja. Vrti se kao nit unutar `subtitis serve` (podrazumevano) ili
kao zaseban proces:

    subtitis worker

HTTP sloj NIKAD ne obrađuje — samo upisuje zadatak u red. Razlog nije samo
odzivnost: jedan worker znači da se dva modela nikad ne nađu u VRAM-u
istovremeno — zadaci idu redom, jedan po jedan. To važi isto i kad je worker
nit: red je i dalje SQLite, `claim_next_task` je atomičan, a `pipeline`
oslobađa Whisper pre nego što pusti LLM.

Tok je EKSPLICITAN. Posle transkripcije worker NE pokreće prevod. Status ostaje
`transcribed` dok čovek ne pregleda original i ne pritisne dugme.

Zadatak čiji je posao pao se NE izvršava. Bez toga se posle neuspele
transkripcije pusti prelamanje nad nepostojećim rečenicama i prijavi „gotovo".
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any

from .. import config as cfg
from .. import pipeline
from ..config import JobCancelled, SubtitisError
from . import db

#: Jedan Ctrl+C samo naoruža zaustavljanje; drugi u ovom roku ga potvrđuje.
#: Zalutali konzolni događaj (Windows ume da ga isporuči celoj grupi procesa)
#: tako ne obara workera koji treba da stoji i čeka nove zadatke.
STOP_CONFIRM_SECONDS = 5.0

#: Razmak između otkucaja. Kuca zasebna nit, pa otkucaj ne staje ni dok traje
#: transkripcija od minut i po — inače bi indikator pocrveneo usred posla.
HEARTBEAT_SECONDS = 5.0

#: Koliko `serve` čeka workera pri gašenju. Taman da worker koji miruje obriše
#: otkucaj; prekratko da bi zadatak u toku mogao da zadrži gašenje servera.
SHUTDOWN_JOIN_SECONDS = 2.0

#: Koliko često se proverava ima li zastarelih radnih WAV-ova.
PURGE_EVERY_SECONDS = 6 * 3600.0


def purge_job_dir(job_id: str) -> None:
    """Obriši direktorijum posla koji više ne postoji u bazi. Nikad ne puca."""
    import shutil

    try:
        jdir = cfg.job_dir(job_id)
    except SubtitisError:
        return
    if jdir.exists():
        shutil.rmtree(jdir, ignore_errors=True)
        if jdir.exists():
            say(f"    (ostaci posla {job_id} nisu obrisani: {jdir})")


def say(msg: str) -> None:
    """Ispis koji se ne gubi u baferu kad je izlaz preusmeren.

    Ide kroz `cfg.echo`, koji ne puca na cp1252 konzoli ni pod `pythonw`.
    """
    cfg.echo(msg)


class Worker:
    """Petlja koja uzima zadatke iz reda. Stanje je na instanci, ne globalno."""

    def __init__(
        self,
        poll_seconds: float = 1.0,
        once: bool = False,
        *,
        mode: str = "proces",
    ) -> None:
        self.poll_seconds = poll_seconds
        self.once = once
        self.mode = mode  # „proces" ili „nit"; ide u indikator, ne u logiku
        self.stop_requested = False
        self.empty_cycles = 0
        self._stop_armed_at: float | None = None
        self._wake = threading.Event()  # zamena za sleep, da gašenje bude trenutno
        self._beat_thread: threading.Thread | None = None
        self._last_purge = 0.0

    # ---- signali ----

    def request_stop(self, source: str = "program") -> bool:
        """Zatraži zaustavljanje. Vrati True ako je stvarno zaustavljeno."""
        now = time.monotonic()
        if self._stop_armed_at is not None and now - self._stop_armed_at <= STOP_CONFIRM_SECONDS:
            self.stop_requested = True
            say(f"Zaustavljam posle tekućeg zadatka… ({source})")
            return True
        self._stop_armed_at = now
        say(
            f"Primljen prekid ({source}). Worker OSTAJE da radi.\n"
            f"  Za zaustavljanje ponovi u roku od {STOP_CONFIRM_SECONDS:.0f}s."
        )
        return False

    def stop(self) -> None:
        """Bezuslovno zaustavljanje. Za gašenje servera — bez potvrđivanja."""
        self.stop_requested = True
        self._wake.set()

    def install_signal_handlers(self) -> None:
        def handler(signum: int, frame: Any) -> None:
            try:
                name = signal.Signals(signum).name
            except ValueError:  # pragma: no cover
                name = str(signum)
            self.request_stop(name)

        # SIGTERM se na Windows-u ne isporučuje kao događaj koji handler može da
        # obradi (TerminateProcess), pa ga tamo i ne registrujemo.
        sigs = [signal.SIGINT]
        if sys.platform != "win32":
            sigs.append(signal.SIGTERM)
        else:
            brk = getattr(signal, "SIGBREAK", None)
            if brk is not None:
                sigs.append(brk)
        for sig in sigs:
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError) as exc:  # pragma: no cover
                say(f"ne mogu da registrujem {sig}: {exc}")

    # ---- čišćenje ----

    def purge_old_sources(self) -> None:
        """Obriši zastarele radne WAV-ove. Nikad ne sme da obori workera."""
        self._last_purge = time.monotonic()
        try:
            freed = cfg.purge_old_sources()
        except Exception as exc:  # noqa: BLE001
            say(f"(čišćenje diska nije uspelo: {exc})")
            return
        for job_id, size in freed:
            db.add_event(
                job_id, "log",
                f"čišćenje: source.wav obrisan posle {cfg.source_wav_ttl_days()} "
                f"dana ({size / 1024 / 1024:.0f} MB)",
            )
        if freed:
            total = sum(s for _, s in freed) / 1024 / 1024
            say(f"Čišćenje: obrisano {len(freed)} radnih WAV-ova, {total:.0f} MB")

    # ---- petlja ----

    def tick(self) -> bool:
        """Jedan prolaz. Vrati True ako je nešto odrađeno."""
        task = db.claim_next_task()
        if task is None:
            self.empty_cycles += 1
            return False

        self.empty_cycles = 0
        job_id, kind = task["job_id"], task["kind"]

        job = db.get_job(job_id)
        if job is None:
            db.finish_task(task["id"], "skipped", "projekat ne postoji")
            say(f"[{kind}] {job_id} preskočen — posao ne postoji")
            return True
        if job["status"] == "error":
            reason = (
                "prethodni korak je pao, pa ovaj zadatak nema nad čim da radi"
            )
            db.finish_task(task["id"], "skipped", reason)
            db.add_event(job_id, "error", f"{kind} preskočen: {reason}")
            say(f"[{kind}] {job_id} PRESKOČEN — {reason}")
            return True

        say(f"[{kind}] {job_id} …")
        try:
            # Posao sme da se obriše dok zadatak radi: `checkpoint()` na
            # prirodnim tačkama (pre poziva LLM-u, posle ASR segmenta) tada
            # podiže `JobCancelled`.
            with cfg.cancellation(lambda: db.get_job(job_id) is None):
                run_task(task)
            db.finish_task(task["id"], "done")
            say(f"[{kind}] {job_id} gotovo")
        except JobCancelled:
            self._cancelled(task, job_id)
        except Exception as exc:  # noqa: BLE001 — worker ne sme da padne zbog jednog posla
            # Obrisan posao ume da pukne i PRE sledeće tačke provere —
            # nestali fajlovi, `None` umesto reda iz baze. To nije greška
            # zadatka, nego posledica brisanja.
            if db.get_job(job_id) is None:
                self._cancelled(task, job_id)
            elif isinstance(exc, SubtitisError):
                self._fail(task, job_id, str(exc))
            else:
                self._fail(task, job_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
        finally:
            # Zadatak koji je radio dok je posao brisan ume da upiše nešto
            # posle brisanja (dnevnik, job.json) i vrati direktorijum. Briše se
            # ponovo ovde, kad ništa više ne piše — bez ostataka.
            if db.get_job(job_id) is None:
                purge_job_dir(job_id)
                # i događaji koje je zadatak upisao posle brisanja
                db.delete_job(job_id)
        return True

    def _cancelled(self, task: Any, job_id: str) -> None:
        """Posao je obrisan tokom zadatka. Uredna poruka, bez traga u poslu."""
        say(f"[{task['kind']}] {job_id} prekinut — posao je obrisan; prelazim na sledeći zadatak")
        # Whisper ne sme da ostane u VRAM-u uz LLM sledećeg zadatka. Ollama se
        # oslobađa sama (`finally: client.unload()` u pajplajnu).
        try:
            from .. import asr

            asr.unload_model()
        except Exception as exc:  # noqa: BLE001
            say(f"    (Whisper nije oslobođen: {exc})")

    def _fail(self, task: Any, job_id: str, message: str) -> None:
        """Zabeleži neuspeh. Ova metoda NIKAD ne sme sama da baci izuzetak —
        inače jedan pokvaren posao obori celu petlju."""
        say(f"[{task['kind']}] {job_id} GREŠKA: {message}")
        for step in (
            lambda: db.finish_task(task["id"], "error", message),
            lambda: db.update_job(job_id, status="error", error=message),
            lambda: db.add_event(job_id, "error", message),
            lambda: cfg.log(job_id, f"GREŠKA: {message}"),
        ):
            try:
                step()
            except Exception as exc:  # noqa: BLE001
                say(f"    (ne mogu da zabeležim neuspeh: {exc})")

        # Sve što je čekalo u redu za ovaj posao nema više nad čim da radi.
        try:
            skipped = db.skip_queued_tasks(job_id, "prethodni korak je pao")
            if skipped:
                db.add_event(job_id, "error", f"preskočeno zadataka u redu: {skipped}")
                say(f"    preskočeno zadataka u redu: {skipped}")
        except Exception as exc:  # noqa: BLE001
            say(f"    (ne mogu da preskočim ostatak reda: {exc})")

    # ---- otkucaj ----

    def _pulse(self) -> None:
        """Kuca dok worker živi, u zasebnoj niti.

        Zasebna nit je nužna: `tick()` blokira koliko traje transkripcija, a
        indikator sme da pocrveni samo kad obrade STVARNO nema.
        """
        while not self.stop_requested:
            try:
                db.beat(self.mode, os.getpid())
            except Exception as exc:  # noqa: BLE001 — otkucaj ne sme da obori posao
                say(f"(otkucaj nije zapisan: {exc})")
            if self._wake.wait(HEARTBEAT_SECONDS):
                return

    def start_pulse(self) -> None:
        if self._beat_thread is not None:
            return
        db.beat(self.mode, os.getpid())  # prvi otkucaj odmah, ne za 5s
        self._beat_thread = threading.Thread(
            target=self._pulse, name="subtitis-beat", daemon=True
        )
        self._beat_thread.start()

    def run(self) -> None:
        db.init_db()
        from ..migrate import migrate_all

        migrate_all(on_migrated=db.migrate_queued_profiles)
        requeued = db.reset_running_tasks()
        if requeued:
            say(f"Vraćeno u red posle prethodnog pada: {requeued} zadataka")
        say(f"Worker radi. Baza: {db.db_path()}   Poslovi: {cfg.JOBS_DIR}")
        if self.mode == "proces":
            say("Ctrl+C dvaput za zaustavljanje.")
        self.start_pulse()
        self.purge_old_sources()

        try:
            while not self.stop_requested:
                did = self.tick()
                if self.once and did:
                    return
                if not did:
                    if self.once:
                        return
                    # Server ume da radi nedeljama; čišćenje samo na pokretanju
                    # bi tada značilo nikad.
                    if time.monotonic() - self._last_purge > PURGE_EVERY_SECONDS:
                        self.purge_old_sources()
                    # Event umesto sleep: gašenje servera ne čeka pun ciklus.
                    self._wake.wait(self.poll_seconds)
            say("Worker zaustavljen.")
        finally:
            self._wake.set()
            try:
                db.clear_beat()  # indikator pocrveni odmah, ne za 20s
            except Exception:  # noqa: BLE001
                pass


def run_worker(poll_seconds: float = 1.0, once: bool = False) -> None:
    w = Worker(poll_seconds=poll_seconds, once=once)
    w.install_signal_handlers()
    w.run()


def start_worker_thread(poll_seconds: float = 1.0) -> tuple[Worker, threading.Thread]:
    """Digni workera kao nit u tekućem procesu. Za `subtitis serve`.

    Nit je `daemon` i petlja čeka na `Event`, pa gašenje servera ne visi. Ono
    što nit NE radi je registracija signala — to sme samo glavna nit, a i ne
    treba joj: `serve` gasi workera preko `lifespan`.

    Model i dalje ide u VRAM i iz njega kao i pre: `pipeline` oslobađa Whisper
    pre LLM-a bez obzira na to ko ga je pozvao. Jedna nit znači jedan zadatak
    odjednom, isto kao jedan proces.
    """
    w = Worker(poll_seconds=poll_seconds, mode="nit")
    t = threading.Thread(target=w.run, name="subtitis-worker", daemon=True)
    t.start()
    return w, t


# ==========================================================================
# Zadaci
# ==========================================================================


def _emit(job_id: str, kind: str, message: str = "", progress: float | None = None) -> None:
    db.add_event(job_id, kind, message, progress)


def run_task(task: Any) -> None:
    """Odradi jedan zadatak. Izuzeci se hvataju u pozivaocu."""
    job_id = task["job_id"]
    kind = task["kind"]
    params: dict[str, Any] = json.loads(task["params"] or "{}")

    def on_log(msg: str) -> None:
        _emit(job_id, "log", msg)

    def on_progress(phase: str, fraction: float) -> None:
        _emit(job_id, "progress", phase, round(float(fraction), 4))

    if kind == "transcribe":
        if pipeline.is_srt_job(job_id):
            raise SubtitisError(
                "Projekat je nastao iz SRT-a — nema zvuka za transkripciju."
            )
        db.update_job(job_id, status="transcribing", error=None)
        _emit(job_id, "status", "transcribing")
        job = db.get_job(job_id)
        from .. import asr as asr_mod

        use_vad, vad_threshold = asr_mod.vad_preset(params.get("vad_preset"))
        r = pipeline.run_transcribe(
            job["source_path"], job["language"], job_id=job_id,
            model=params.get("model", cfg.DEFAULT_ASR_MODEL),
            device=params.get("device", "cuda"),
            compute_type=params.get("compute_type", "float16"),
            vad=use_vad,
            vad_threshold=vad_threshold,
            force=True,
            on_log=on_log, on_progress=on_progress,
        )
        # Kod `auto` jezik se zna tek sada — bez ovoga bi u listi zauvek stajalo
        # „auto" i editor ne bi našao rečenice.
        if r.language and r.language != job["language"]:
            db.update_job(job_id, language=r.language)
            _emit(job_id, "log", f"jezik detektovan: {r.language}")
        db.update_job(job_id, status="transcribed", duration=r.duration)
        _emit(job_id, "status", "transcribed")
        _emit(
            job_id, "log",
            f"gotovo: {r.words} reči, {r.sentences} rečenica, "
            f"pokrivenost {r.coverage_pct:.1f}%, praznina >3s: {len(r.gaps)}",
        )
        # Original SRT je za preuzimanje ODMAH — ne čeka prevod.
        _build(job_id, params.get("profile") or cfg.default_profile(), on_log, required=False)
        _emit(job_id, "done", "transcribed")

    elif kind == "translate":
        db.update_job(job_id, status="translating", error=None)
        _emit(job_id, "status", "translating")
        hops = pipeline.run_translate(
            job_id, params["to"],
            model=params.get("model"), pivot=params.get("pivot"),
            variant=params.get("variant"), force=True,
            on_log=on_log, on_progress=on_progress,
        )
        for h in hops:
            _emit(
                job_id, "log",
                f"{h.src}->{h.tgt}: {h.sentences} rečenica, "
                f"{h.glossary_entries} unosa glosara, " + (
                    f"NEPREVEDENO {len(h.untranslated.all)} rečenica / "
                    f"{h.untranslated.cue_count} cue-ova (vidi UPOZORENJE iznad)"
                    if h.untranslated and h.untranslated.all else "sve prevedeno"
                ),
            )
            if h.warning:
                _emit(job_id, "log", f"UPOZORENJE: {h.warning}")
        _build(job_id, params.get("profile") or None, on_log,
               language=params["to"], script=params.get("script", "Latn"),
               case=params.get("case"),
               drop_punct=params.get("drop_punct"), required=False)
        db.update_job(job_id, status="done")
        _emit(job_id, "status", "done")
        _emit(job_id, "done", "done")

    elif kind == "shorten":
        # Samo predlog: prevod se menja tek kad čovek potvrdi izmene u editoru.
        r = pipeline.run_shorten(job_id, params["language"], on_log=on_log)
        _emit(job_id, "log", f"skraćivanje: {r.accepted} predloga čeka pregled")
        _emit(job_id, "done", "shorten")

    elif kind == "build":
        # Samostalno prelamanje MORA da pukne ako nema nad čim da radi.
        _build(
            job_id, params.get("profile") or None, on_log,
            language=params.get("language"), script=params.get("script", "Latn"),
            case=params.get("case"),
            drop_punct=params.get("drop_punct"), required=True,
        )
        _emit(job_id, "done", "build")

    else:
        raise SubtitisError(f"Nepoznata vrsta zadatka: {kind!r}")


def _build(
    job_id: str, profile: str | None, on_log: Any, *,
    language: str | None = None, script: str = "Latn", case: str | None = None,
    drop_punct: bool | None = None, required: bool,
) -> None:
    """Prelomi u SRT.

    `required=False` je za prelamanje koje ide uz uspešnu transkripciju ili
    prevod — glavni posao je odrađen i zapisan, pa neuspeh prelamanja ne
    poništava rezultat, ali se prijavljuje kao greška u dnevniku, ne kao usputna
    beleška. `required=True` je za zadatak koji NIŠTA drugo ne radi: tamo
    neuspeh mora da obori zadatak, inače worker javlja „gotovo" za posao koji
    se nije desio.
    """
    try:
        r = pipeline.run_build(
            job_id, profile=profile, script=script, language=language, on_log=on_log,
            case=case, drop_punct=drop_punct,
        )
        _emit(job_id, "log", f"SRT: {r.path.name} ({r.cues} cue-ova)")
    except SubtitisError as exc:
        if required:
            raise
        _emit(job_id, "error", f"SRT nije napravljen: {exc}")


if __name__ == "__main__":  # pragma: no cover
    run_worker()
