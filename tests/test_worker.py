"""Testovi workera: preživljavanje praznog reda, prekid, i preskakanje
zadataka posle neuspelog koraka.
"""

from __future__ import annotations

import time

import pytest

from subtitis.config import SubtitisError


@pytest.fixture()
def wenv(tmp_path, monkeypatch):
    from subtitis import config as cfg
    from subtitis.web import db, worker

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    db.init_db()
    return db, worker, cfg


# ==========================================================================
# Prazan red ne sme da obori workera
# ==========================================================================


def test_prezivljava_dva_prazna_ciklusa(wenv):
    """Regresija: worker se gasio sam posle praznog reda."""
    db, worker, _ = wenv
    w = worker.Worker(poll_seconds=0)
    assert w.tick() is False
    assert w.tick() is False
    assert w.stop_requested is False
    assert w.empty_cycles == 2


def test_prezivljava_deset_praznih_ciklusa(wenv):
    db, worker, _ = wenv
    w = worker.Worker(poll_seconds=0)
    for _ in range(10):
        assert w.tick() is False
    assert w.stop_requested is False


def test_petlja_ne_izlazi_dok_se_ne_zatrazi(wenv):
    """Pusti pravu petlju u niti i proveri da posle praznih ciklusa još radi."""
    import threading

    db, worker, _ = wenv
    w = worker.Worker(poll_seconds=0.02)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.4)  # >> 2 ciklusa
    assert t.is_alive(), "worker je izašao sam od sebe"
    assert w.empty_cycles >= 2
    w.stop_requested = True
    t.join(timeout=2)
    assert not t.is_alive()


def test_prazan_red_ne_dira_zastavicu_prekida(wenv):
    db, worker, _ = wenv
    w = worker.Worker(poll_seconds=0)
    for _ in range(5):
        w.tick()
    assert w._stop_armed_at is None


# ==========================================================================
# Prekid traži potvrdu
# ==========================================================================


def test_jedan_prekid_ne_zaustavlja(wenv):
    """Zalutali konzolni događaj ne sme da obori workera."""
    _, worker, _ = wenv
    w = worker.Worker()
    assert w.request_stop("SIGINT") is False
    assert w.stop_requested is False


def test_drugi_prekid_zaustavlja(wenv):
    _, worker, _ = wenv
    w = worker.Worker()
    w.request_stop("SIGINT")
    assert w.request_stop("SIGINT") is True
    assert w.stop_requested is True


def test_prekid_posle_isteka_roka_ponovo_samo_naoruzava(wenv, monkeypatch):
    _, worker, _ = wenv
    w = worker.Worker()
    w.request_stop("SIGINT")
    w._stop_armed_at = time.monotonic() - worker.STOP_CONFIRM_SECONDS - 1
    assert w.request_stop("SIGINT") is False
    assert w.stop_requested is False


def test_sigterm_se_ne_registruje_na_windowsu(wenv, monkeypatch):
    import signal

    _, worker, _ = wenv
    registered = []
    monkeypatch.setattr(signal, "signal", lambda s, h: registered.append(s))
    monkeypatch.setattr(worker.sys, "platform", "win32")
    worker.Worker().install_signal_handlers()
    assert signal.SIGINT in registered
    assert signal.SIGTERM not in registered, "SIGTERM na Windowsu ne stiže do handlera"


# ==========================================================================
# Zadaci posle neuspelog koraka
# ==========================================================================


def make_job(db, job_id="p1", status="uploaded"):
    db.create_job(job_id, "a.wav", "/a.wav", "sr")
    db.update_job(job_id, status=status)
    return job_id


def test_zadatak_palog_posla_se_preskace(wenv):
    db, worker, _ = wenv
    jid = make_job(db, status="error")
    db.enqueue(jid, "build")
    w = worker.Worker(poll_seconds=0)
    assert w.tick() is True
    with db.connect() as conn:
        row = conn.execute("SELECT state, error FROM tasks WHERE job_id=?", (jid,)).fetchone()
    assert row["state"] == "skipped"
    assert "pao" in row["error"]


def test_preskocen_zadatak_ne_javlja_gotovo(wenv, capsys):
    db, worker, _ = wenv
    jid = make_job(db, status="error")
    db.enqueue(jid, "build")
    worker.Worker(poll_seconds=0).tick()
    out = capsys.readouterr().out
    assert "PRESKOČEN" in out
    assert "gotovo" not in out


def test_neuspeh_preskace_ostatak_reda(wenv, monkeypatch):
    db, worker, _ = wenv
    jid = make_job(db)
    db.enqueue(jid, "transcribe")
    db.enqueue(jid, "build")
    db.enqueue(jid, "translate", {"to": "en"})

    def boom(task):
        raise SubtitisError("ffmpeg nije pronađen")

    monkeypatch.setattr(worker, "run_task", boom)
    w = worker.Worker(poll_seconds=0)
    w.tick()

    with db.connect() as conn:
        states = [r["state"] for r in conn.execute(
            "SELECT state FROM tasks WHERE job_id=? ORDER BY id", (jid,)).fetchall()]
    assert states == ["error", "skipped", "skipped"]
    assert db.get_job(jid)["status"] == "error"
    assert "ffmpeg" in db.get_job(jid)["error"]


def test_posle_neuspeha_worker_nastavlja_sa_drugim_poslom(wenv, monkeypatch):
    db, worker, _ = wenv
    make_job(db, "los")
    make_job(db, "dobar")
    db.enqueue("los", "build")
    db.enqueue("dobar", "build")

    done = []

    def flaky(task):
        if task["job_id"] == "los":
            raise SubtitisError("puk")
        done.append(task["job_id"])

    monkeypatch.setattr(worker, "run_task", flaky)
    w = worker.Worker(poll_seconds=0)
    w.tick()
    w.tick()
    assert done == ["dobar"]
    assert db.get_job("dobar")["status"] != "error"


def test_beleska_o_neuspehu_ne_obara_petlju(wenv):
    """`_fail` sam ume da pukne (npr. neispravan job_id) — to ne sme da izađe."""
    db, worker, _ = wenv
    db.create_job("ime sa razmakom", "a.wav", "/a.wav", "sr")
    db.enqueue("ime sa razmakom", "build")

    def boom(task):
        raise SubtitisError("puk")

    import pytest as _pytest

    monkey = _pytest.MonkeyPatch()
    monkey.setattr(worker, "run_task", boom)
    try:
        w = worker.Worker(poll_seconds=0)
        assert w.tick() is True  # ne sme da propusti izuzetak
        assert w.tick() is False  # petlja i dalje radi
    finally:
        monkey.undo()


def test_zadatak_bez_posla_se_preskace(wenv):
    db, worker, _ = wenv
    db.enqueue("nepostojeci", "build")
    w = worker.Worker(poll_seconds=0)
    assert w.tick() is True
    with db.connect() as conn:
        assert conn.execute("SELECT state FROM tasks").fetchone()["state"] == "skipped"


def test_skip_queued_tasks_ne_dira_zavrsene(wenv):
    db, _, _ = wenv
    jid = make_job(db)
    t1 = db.enqueue(jid, "transcribe")
    db.enqueue(jid, "build")
    db.finish_task(t1, "done")
    assert db.skip_queued_tasks(jid, "razlog") == 1
    with db.connect() as conn:
        states = [r["state"] for r in conn.execute(
            "SELECT state FROM tasks WHERE job_id=? ORDER BY id", (jid,)).fetchall()]
    assert states == ["done", "skipped"]


# ==========================================================================
# Prelamanje: obavezno vs usputno
# ==========================================================================


def test_samostalno_prelamanje_pada_ako_nema_nad_cim(wenv, monkeypatch):
    """Regresija: build je javljao „gotovo" nad nepostojećim rečenicama."""
    db, worker, _ = wenv
    jid = make_job(db, status="transcribed")

    def boom(*a, **k):
        raise SubtitisError("Nedostaje datoteka: sentences.sr.json")

    monkeypatch.setattr(worker.pipeline, "run_build", boom)
    with pytest.raises(SubtitisError):
        worker._build(jid, "tv", lambda m: None, required=True)


def test_usputno_prelamanje_ne_obara_glavni_posao(wenv, monkeypatch):
    db, worker, _ = wenv
    jid = make_job(db, status="transcribed")

    def boom(*a, **k):
        raise SubtitisError("nema profila")

    monkeypatch.setattr(worker.pipeline, "run_build", boom)
    worker._build(jid, "tv", lambda m: None, required=False)  # ne diže izuzetak
    kinds = [e["kind"] for e in db.recent_events(jid)]
    assert "error" in kinds, "neuspeh se ipak mora videti kao greška, ne kao beleška"


def test_build_zadatak_pada_kroz_run_task(wenv, monkeypatch):
    db, worker, _ = wenv
    jid = make_job(db, status="transcribed")
    db.enqueue(jid, "build")

    def boom(*a, **k):
        raise SubtitisError("Nedostaje datoteka")

    monkeypatch.setattr(worker.pipeline, "run_build", boom)
    w = worker.Worker(poll_seconds=0)
    w.tick()
    with db.connect() as conn:
        row = conn.execute("SELECT state FROM tasks WHERE job_id=?", (jid,)).fetchone()
    assert row["state"] == "error"
    assert db.get_job(jid)["status"] == "error"


# ==========================================================================
# Otkucaj: jedini način da se sa ekrana vidi da obrada uopšte radi
# ==========================================================================


def test_bez_workera_nema_otkucaja(wenv):
    db, _, _ = wenv
    st = db.worker_status()
    assert st["alive"] is False
    assert st["age"] is None


def test_worker_kuca_dok_radi(wenv):
    db, worker, _ = wenv
    w = worker.Worker(poll_seconds=0.02, mode="nit")
    w.start_pulse()
    try:
        st = db.worker_status()
        assert st["alive"] is True
        assert st["mode"] == "nit"
    finally:
        w.stop()


def test_otkucaj_se_brise_pri_urednom_gasenju(wenv):
    """Indikator sme da pocrveni odmah, ne tek kad otkucaj zastari."""
    import threading

    db, worker, _ = wenv
    w = worker.Worker(poll_seconds=0.02)
    t = threading.Thread(target=w.run, daemon=True)
    t.start()
    time.sleep(0.2)
    assert db.worker_status()["alive"] is True
    w.stop()
    t.join(timeout=2)
    assert db.worker_status()["alive"] is False


def test_zastareo_otkucaj_znaci_da_obrada_ne_radi(wenv, monkeypatch):
    """Worker koji je ubijen ne stigne da obriše otkucaj — starost ga otkriva."""
    from datetime import datetime, timedelta, timezone

    db, _, _ = wenv
    db.beat("proces", 123)
    stale = datetime.now(timezone.utc) - timedelta(
        seconds=db.HEARTBEAT_STALE_SECONDS + 5
    )
    with db.connect() as conn:
        conn.execute("UPDATE worker_beat SET ts=? WHERE id=1",
                     (stale.isoformat(timespec="seconds"),))
    assert db.worker_status()["alive"] is False


def test_status_broji_zadatke_u_redu(wenv):
    """Crveno + „zadaci čekaju" je cela poenta indikatora."""
    db, _, _ = wenv
    jid = make_job(db)
    db.enqueue(jid, "transcribe")
    db.enqueue(jid, "build")
    st = db.worker_status()
    assert st["alive"] is False
    assert st["queued"] == 2


def test_stop_ne_traži_potvrdu(wenv):
    """`request_stop` traži dva Ctrl+C; gašenje servera ne sme da čeka na to."""
    _, worker, _ = wenv
    w = worker.Worker()
    w.stop()
    assert w.stop_requested is True


def test_nit_ne_zadrzava_gasenje(wenv):
    """Petlja čeka na Event, ne na sleep — `stop` je trenutan."""
    db, worker, _ = wenv
    w, t = worker.start_worker_thread(poll_seconds=30.0)
    try:
        time.sleep(0.2)
        assert t.is_alive()
        started = time.monotonic()
        w.stop()
        t.join(timeout=worker.SHUTDOWN_JOIN_SECONDS)
        assert not t.is_alive(), "nit nije izašla iako je poll 30s"
        assert time.monotonic() - started < 2.0
    finally:
        w.stop()


# ==========================================================================
# Brisanje posla dok mu zadatak radi
# ==========================================================================


def _job(db, cfg, jid="brisan"):
    jdir = cfg.job_dir(jid)
    jdir.mkdir(parents=True)
    (jdir / "sentences.sr.json").write_text("{}", encoding="utf-8")
    db.create_job(jid, "x.wav", str(jdir / "x.wav"), "sr")
    return jdir


def _obrisi(db, cfg, jid):
    """Isto što radi ruta za brisanje: baza i direktorijum zajedno."""
    import shutil

    db.delete_job(jid)
    shutil.rmtree(cfg.job_dir(jid), ignore_errors=True)


def test_zadatak_u_toku_staje_na_prirodnoj_tacki(wenv, monkeypatch):
    db, worker, cfg = wenv
    _job(db, cfg)
    db.enqueue("brisan", "translate", {"to": "en"})
    db.enqueue("drugi", "build", {})  # sledeći zadatak drugog posla
    _job(db, cfg, "drugi")
    koraci: list[str] = []

    def run_task(task):
        if task["job_id"] != "brisan":
            koraci.append("drugi posao odrađen")
            return
        koraci.append("čank 1")
        cfg.checkpoint()
        _obrisi(db, cfg, "brisan")  # korisnik briše usred prevoda
        # zadatak još piše pre sledeće tačke — vrati direktorijum
        (cfg.job_dir("brisan")).mkdir(parents=True, exist_ok=True)
        (cfg.job_dir("brisan") / "log.txt").write_text("posle brisanja", encoding="utf-8")
        db.add_event("brisan", "log", "zakasneli događaj")
        cfg.checkpoint()  # sledeći čank — tu staje
        koraci.append("čank 2 — NE SME")

    monkeypatch.setattr(worker, "run_task", run_task)
    rečeno: list[str] = []
    monkeypatch.setattr(worker, "say", rečeno.append)
    w = worker.Worker(poll_seconds=0)
    assert w.tick() is True
    assert koraci == ["čank 1"]
    assert any("prekinut — posao je obrisan" in m for m in rečeno)
    assert not any("GREŠKA" in m or "Traceback" in m for m in rečeno)
    # bez ostataka: ni direktorijum, ni zakasneli događaji, ni zadatak
    assert not cfg.job_dir("brisan").exists()
    assert db.recent_events("brisan") == []
    assert db.pending_task("brisan") is None
    # worker ide dalje
    assert w.tick() is True and koraci[-1] == "drugi posao odrađen"


def test_nestali_fajlovi_obrisanog_posla_nisu_sirova_greska(wenv, monkeypatch):
    """Pukne PRE sledeće tačke provere (fajl nestao) — i dalje uredna poruka."""
    db, worker, cfg = wenv
    _job(db, cfg)
    db.enqueue("brisan", "build", {})

    def run_task(task):
        _obrisi(db, cfg, "brisan")
        cfg.read_json(cfg.job_dir("brisan") / "sentences.sr.json")  # SubtitisError

    monkeypatch.setattr(worker, "run_task", run_task)
    rečeno: list[str] = []
    monkeypatch.setattr(worker, "say", rečeno.append)
    worker.Worker(poll_seconds=0).tick()
    assert any("prekinut — posao je obrisan" in m for m in rečeno)
    assert not any("GREŠKA" in m for m in rečeno)
    assert db.get_job("brisan") is None


def test_neobrisan_posao_i_dalje_pada_kao_greska(wenv, monkeypatch):
    db, worker, cfg = wenv
    _job(db, cfg)
    db.enqueue("brisan", "build", {})

    def run_task(task):
        raise SubtitisError("prava greška")

    monkeypatch.setattr(worker, "run_task", run_task)
    worker.Worker(poll_seconds=0).tick()
    assert db.get_job("brisan")["status"] == "error"
    assert cfg.job_dir("brisan").exists()


def test_poziv_modelu_ne_krece_za_obrisan_posao():
    """Tačka prekida je PRE poziva, pa usred poziva nema šta da se prekida."""
    from subtitis import config as cfg
    from subtitis.translate import OllamaClient

    client = OllamaClient(url="http://127.0.0.1:9")  # ne sme ni da pokuša

    def post(*a, **k):  # pragma: no cover — poziv bi bio greška
        raise AssertionError("poziv modelu posle brisanja")

    client._post = post
    with cfg.cancellation(lambda: True):
        with pytest.raises(cfg.JobCancelled):
            client.chat("s", "u")


def test_prekid_se_ne_guta_kao_obicna_greska():
    """Pajplajn hvata SubtitisError da bi nastavio, ASR hvata RuntimeError."""
    from subtitis.config import JobCancelled

    assert not issubclass(JobCancelled, RuntimeError)
    assert not issubclass(JobCancelled, SubtitisError)


def test_asr_staje_posle_segmenta_i_prekid_prolazi_kroz_asr(monkeypatch):
    """Posle svakog segmenta — prekid izlazi iz `asr.transcribe` kao prekid,
    ne kao „greška CUDA/modela" (ASR petlja hvata `RuntimeError`)."""
    from subtitis import asr
    from subtitis import config as cfg

    class Word:
        def __init__(self, w, s, e):
            self.word, self.start, self.end, self.probability = w, s, e, 0.9

    class Seg:
        def __init__(self, k):
            self.words = [Word(f"reč{k}", k, k + 0.5)]
            self.end = k + 0.5

    procitano = []

    def segments():
        for k in range(10):
            procitano.append(k)
            yield Seg(k)

    class Model:
        def transcribe(self, path, **params):
            return segments(), type("Info", (), {"language": "sr", "language_probability": 1.0})()

    monkeypatch.setattr(asr, "load_model", lambda *a, **k: Model())
    obrisan = {"da": False}

    def progress(pos, total):
        if pos > 2:
            obrisan["da"] = True  # posao obrisan dok je tekao treći segment
        cfg.checkpoint()

    with cfg.cancellation(lambda: obrisan["da"]):
        with pytest.raises(cfg.JobCancelled):
            asr.transcribe("x.wav", job_id="p", source_file="p.wav", language="sr",
                           duration=10.0, progress=progress)
    # segment 2 završava na 2,5 s: tu je brisanje primećeno i dalje se ne čita
    assert procitano == [0, 1, 2]


def test_build_zadatak_prenosi_velicinu_slova(wenv, monkeypatch):
    db, worker, _ = wenv
    jid = make_job(db, status="transcribed")
    db.enqueue(jid, "build", {"language": "sr", "profile": "tv", "script": "Cyrl", "case": "upper"})
    seen = {}

    class R:
        path = __import__("pathlib").Path("x.srt")
        cues = 1

    def fake(job_id, **kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(worker.pipeline, "run_build", fake)
    worker.Worker(poll_seconds=0).tick()
    assert (seen["case"], seen["script"]) == ("upper", "Cyrl")


def test_prevod_prenosi_velicinu_slova_do_prelamanja(wenv, monkeypatch):
    db, worker, _ = wenv
    jid = make_job(db, status="transcribed")
    db.enqueue(jid, "translate", {"to": "en", "case": "lower"})
    monkeypatch.setattr(worker.pipeline, "run_translate", lambda *a, **k: [])
    seen = {}
    monkeypatch.setattr(worker, "_build", lambda *a, **k: seen.update(k))
    worker.Worker(poll_seconds=0).tick()
    assert seen["case"] == "lower" and seen["language"] == "en"
