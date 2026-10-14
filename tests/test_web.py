"""Testovi dashboarda: baza, red zadataka i HTTP sloj.

Ne diraju ni Whisper ni Ollamu — worker se ne pokreće, zadaci samo stoje u redu.
Naglasak je na dve stvari iz specifikacije: da tok NIJE automatski lančan i da
editor ispravno rukuje `edited_by_human`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from subtitis.models import AsrMeta, Sentence, SentencesFile, Word, WordsFile


@pytest.fixture()
def web(tmp_path, monkeypatch):
    """Izolovan JOBS_DIR i baza po testu."""
    from subtitis import config as cfg
    from subtitis.web import app as app_mod
    from subtitis.web import db

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    # Bez ovoga bi testovi podešavanja pisali u pravi `subtitis.toml` i pravi
    # glosar u radnom stablu.
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    db.init_db()

    from fastapi.testclient import TestClient

    with TestClient(app_mod.app) as client:
        yield client, db, cfg


def make_job(cfg, db, job_id="posao", lang="sr", *, edited=False, translated=False):
    """Napravi posao na disku i u bazi, bez pokretanja ičega."""
    jdir = cfg.job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    (jdir / "upload").mkdir(exist_ok=True)
    src = jdir / "upload" / "snimak.wav"
    src.write_bytes(b"RIFF")

    words = [
        Word(i=0, s=0.0, e=0.5, w="Dobar", p=0.95),
        Word(i=1, s=0.5, e=1.0, w="dan", p=0.42),   # ispod praga -> žuto
        Word(i=2, s=1.0, e=1.05, w=".", p=0.99),
    ]
    cfg.write_model(
        jdir / f"words.{lang}.json",
        WordsFile(
            job_id=job_id, source_file=str(src), duration=2.0, language=lang,
            asr=AsrMeta(model="test"), words=words,
        ),
    )
    cfg.write_model(
        jdir / f"sentences.{lang}.json",
        SentencesFile(
            job_id=job_id, language=lang,
            derived_from=("sentences.sr.json" if translated else f"words.{lang}.json"),
            edited_by_human=edited,
            sentences=[
                Sentence(id=1, s=0.0, e=1.05, text="Dobar dan.", words=(0, 2),
                         flags=["low_confidence"])
            ],
        ),
    )
    from subtitis.models import Job

    cfg.save_job(Job(job_id=job_id, source_file=str(src), language=lang,
                     status="transcribed", duration=2.0))
    db.create_job(job_id, "snimak.wav", str(src), lang)
    db.update_job(job_id, status="transcribed", duration=2.0)
    return job_id


# ==========================================================================
# Baza i red
# ==========================================================================


def test_red_je_fifo_i_atomican(web):
    _, db, _ = web
    db.create_job("a", "a.wav", "/a.wav", "sr")
    t1 = db.enqueue("a", "transcribe")
    t2 = db.enqueue("a", "build")
    assert db.claim_next_task()["id"] == t1
    assert db.claim_next_task()["id"] == t2
    assert db.claim_next_task() is None


def test_preuzet_zadatak_se_ne_preuzima_dvaput(web):
    _, db, _ = web
    db.create_job("a", "a.wav", "/a.wav", "sr")
    db.enqueue("a", "transcribe")
    assert db.claim_next_task() is not None
    assert db.claim_next_task() is None, "drugi worker ne sme uzeti isti zadatak"


def test_zaostali_running_se_vraca_u_red(web):
    _, db, _ = web
    db.create_job("a", "a.wav", "/a.wav", "sr")
    db.enqueue("a", "transcribe")
    db.claim_next_task()
    assert db.reset_running_tasks() == 1
    assert db.claim_next_task() is not None


def test_pending_task_vidi_i_queued_i_running(web):
    _, db, _ = web
    db.create_job("a", "a.wav", "/a.wav", "sr")
    db.enqueue("a", "transcribe")
    assert db.pending_task("a") is not None
    db.claim_next_task()
    assert db.pending_task("a") is not None


def test_dogadjaji_se_citaju_od_kursora(web):
    _, db, _ = web
    db.create_job("a", "a.wav", "/a.wav", "sr")
    db.add_event("a", "log", "prvi")
    db.add_event("a", "log", "drugi")
    assert [e["message"] for e in db.events_since("a", 0)] == ["prvi", "drugi"]
    first = db.events_since("a", 0)[0]["id"]
    assert [e["message"] for e in db.events_since("a", first)] == ["drugi"]


# ==========================================================================
# Tok NIJE automatski lančan
# ==========================================================================


def test_upload_stavlja_samo_transkripciju_u_red(web, tmp_path):
    client, db, cfg = web
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF____WAVE")
    r = client.post(
        "/jobs",
        files={"audio": ("test.wav", wav.read_bytes(), "audio/wav")},
        data={"language": "sr", "model": "large-v3", "vad": "on"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    job_id = r.headers["location"].rsplit("/", 1)[-1]
    with db.connect() as conn:
        kinds = [x["kind"] for x in conn.execute(
            "SELECT kind FROM tasks WHERE job_id=?", (job_id,)).fetchall()]
    assert kinds == ["transcribe"], "prevod se NE sme staviti u red automatski"
    assert db.get_job(job_id)["status"] == "uploaded"


def test_prevod_se_pokrece_tek_posebnim_dugmetom(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM tasks WHERE job_id=?", (jid,)).fetchone()["c"] == 0
    r = client.post(f"/jobs/{jid}/translate", data={"to": "en"}, follow_redirects=False)
    assert r.status_code == 303
    with db.connect() as conn:
        rows = conn.execute("SELECT kind, params FROM tasks WHERE job_id=?", (jid,)).fetchall()
    assert [x["kind"] for x in rows] == ["translate"]
    assert json.loads(rows[0]["params"])["to"] == "en"


def test_srpski_je_ponudjen_kao_ciljni_jezik(web):
    client, db, cfg = web
    jid = make_job(cfg, db, lang="fr")
    page = client.get(f"/jobs/{jid}").text
    assert 'value="sr"' in page
    # Izvorni jezik nije ciljni — prevod fr -> fr pukne u pajplajnu.
    assert 'value="fr"' not in page.split('name="to"')[1].split("</select>")[0]


def test_izbor_pisma_ide_uz_prevod_na_srpski(web):
    client, db, cfg = web
    jid = make_job(cfg, db, lang="fr")
    r = client.post(
        f"/jobs/{jid}/translate", data={"to": "sr", "script": "Cyrl"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    with db.connect() as conn:
        row = conn.execute("SELECT params FROM tasks WHERE job_id=?", (jid,)).fetchone()
    params = json.loads(row["params"])
    assert params["to"] == "sr" and params["script"] == "Cyrl"


def test_cirilica_za_nedvopismeni_jezik_se_odbija(web):
    """Meni je ne nudi; ovo je brana za ručno poslat obrazac."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(
        f"/jobs/{jid}/translate", data={"to": "fr", "script": "Cyrl"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    with db.connect() as conn:
        assert conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE job_id=?", (jid,)
        ).fetchone()["c"] == 0


def test_prevod_bez_pisma_ostaje_latinica(web):
    client, db, cfg = web
    jid = make_job(cfg, db, lang="fr")
    client.post(f"/jobs/{jid}/translate", data={"to": "sr"}, follow_redirects=False)
    with db.connect() as conn:
        row = conn.execute("SELECT params FROM tasks WHERE job_id=?", (jid,)).fetchone()
    assert json.loads(row["params"])["script"] == "Latn"


def test_dva_zadatka_odjednom_se_odbijaju(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    client.post(f"/jobs/{jid}/translate", data={"to": "en"}, follow_redirects=False)
    r = client.post(f"/jobs/{jid}/translate", data={"to": "fr"}, follow_redirects=False)
    assert r.status_code == 409


def test_original_srt_je_dostupan_pre_prevoda(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Latn.srt").write_text("1\r\n", encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    assert "snimak.sr-Latn.srt" in page
    r = client.get(f"/jobs/{jid}/download/snimak.sr-Latn.srt")
    assert r.status_code == 200


def test_status_stranica_prikazuje_pauzu_za_pregled(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    # pauza za pregled: prevod se ne pokreće sam, prvo se ispravlja prepis
    assert "Prvo isprati prepis, pa onda prevodi." in page
    assert "Prevedi" in page


# ==========================================================================
# Editor
# ==========================================================================


def test_editor_boji_reci_ispod_praga(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert '<span class="low"' in page
    assert ">dan</span>" in page, "reč sa p=0.42 mora biti obojena"
    assert ">Dobar<" not in page.split('<span class="low"')[1][:40]


def test_editor_objasnjava_boje_u_traci(web):
    """Opšte objašnjenje je u traci „Kako se koristi editor" (krug 4b), ne u
    napomeni iznad liste."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert "Kako se koristi editor" in page
    assert "isto što i greška" in page
    assert '<p class="notice">' not in page.split('id="editor"')[1].split("<table")[0]


def test_prevedene_recenice_nemaju_bojenje(web):
    client, db, cfg = web
    jid = make_job(cfg, db, lang="en", translated=True)
    page = client.get(f"/jobs/{jid}?lang=en").text
    assert '<span class="low"' not in page


def test_snimanje_postavlja_edited_by_human(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(
        f"/jobs/{jid}/sentences",
        json={"lang": "sr", "sentences": [{"id": 1, "text": "Dobar dan svima."}]},
    )
    assert r.status_code == 200 and r.json()["changed"] == 1
    sf = SentencesFile.model_validate(
        cfg.read_json(cfg.job_dir(jid) / "sentences.sr.json")
    )
    assert sf.edited_by_human is True
    assert sf.sentences[0].text == "Dobar dan svima."


def test_snimanje_bez_izmena_ne_dira_fajl(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(
        f"/jobs/{jid}/sentences",
        json={"lang": "sr", "sentences": [{"id": 1, "text": "Dobar dan."}]},
    )
    body = r.json()
    assert body["changed"] == 0
    assert body["edited_by_human"] is False
    assert body["backup"] is None, "bez izmena se ne pravi kopija"


def test_snimanje_pravi_bak_kopiju(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    client.post(
        f"/jobs/{jid}/sentences",
        json={"lang": "sr", "sentences": [{"id": 1, "text": "Izmenjeno."}]},
    )
    assert list(cfg.job_dir(jid).glob("sentences.sr.json.*.bak"))


def test_prazna_recenica_se_odbija(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(
        f"/jobs/{jid}/sentences",
        json={"lang": "sr", "sentences": [{"id": 1, "text": "   "}]},
    )
    assert r.status_code == 400
    sf = SentencesFile.model_validate(cfg.read_json(cfg.job_dir(jid) / "sentences.sr.json"))
    assert sf.text if False else sf.sentences[0].text == "Dobar dan."


def test_snimanje_ne_stavlja_prelamanje_u_red(web):
    """Krug 13: editor posle snimanja prelama SINHRONO, izborima iz trake
    (/rebuild). Red je ranije prelamao podrazumevanim profilom, ne izabranim."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    client.post(
        f"/jobs/{jid}/sentences",
        json={"lang": "sr", "sentences": [{"id": 1, "text": "Izmenjeno."}]},
    )
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM tasks WHERE job_id=?", (jid,)).fetchone()["c"] == 0


def test_rucno_menjan_fajl_blokira_ponovnu_transkripciju(web):
    client, db, cfg = web
    jid = make_job(cfg, db, edited=True)
    r = client.post(f"/jobs/{jid}/retranscribe", data={"model": "large-v3"},
                    follow_redirects=False)
    assert r.status_code == 409
    assert "Tvoje ispravke prepisa su sačuvane" in r.text


def test_editor_pokazuje_oznaku_rucno_menjanog(web):
    client, db, cfg = web
    jid = make_job(cfg, db, edited=True)
    assert "Tvoje ispravke su sačuvane." in client.get(f"/jobs/{jid}").text


# ==========================================================================
# Ostalo
# ==========================================================================


def test_nepostojeci_posao_daje_404(web):
    client, _, _ = web
    assert client.get("/jobs/nema-ga").status_code == 404


def test_download_ne_izlazi_iz_direktorijuma(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    assert client.get(f"/jobs/{jid}/download/..%2F..%2Fjob.json").status_code in (400, 404)


def test_lista_poslova_se_prikazuje(web):
    client, db, cfg = web
    make_job(cfg, db, "prvi")
    make_job(cfg, db, "drugi")
    page = client.get("/").text
    assert "prvi" in page and "drugi" in page


def test_status_api(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    data = client.get(f"/api/jobs/{jid}/status").json()
    assert data["status"] == "transcribed"
    assert data["pending"] is False
    assert data["languages"] == ["sr"]


def test_brisanje_posla_brise_i_disk(web):
    """Ranije su datoteke ostajale, pa je „obriši" ostavljalo 40 MB po poslu."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    (cfg.job_dir(jid) / "source.wav").write_bytes(b"x" * 4096)
    client.post(f"/jobs/{jid}/delete", follow_redirects=False)
    assert db.get_job(jid) is None
    assert not cfg.job_dir(jid).exists()


def test_brisanje_otkazuje_zadatak_u_redu(web):
    """Obrada pokrenuta greškom mora moći da se ukloni — i dok čeka u redu."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    db.enqueue(jid, "transcribe")
    r = client.post(f"/jobs/{jid}/delete", follow_redirects=False)
    assert r.status_code == 303
    assert db.get_job(jid) is None
    assert db.pending_task(jid) is None
    assert not cfg.job_dir(jid).exists()
    assert db.claim_next_task() is None, "otkazan zadatak ne sme da ostane u redu"


def test_dugme_obrisi_nije_sivo_i_upozorava_na_otkazivanje(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    db.enqueue(jid, "translate", {"to": "en"})
    html = client.get("/jobs/list").text
    forma = html.split(f"/jobs/{jid}/delete")[1].split("</form>")[0]
    assert "disabled" not in forma
    # krug 12: naš prozor, dugme kaže šta radi, bez „SVE"
    assert 'data-confirm-ok="Obriši projekat"' in forma and 'data-confirm-danger="true"' in forma
    assert "biće otkazan" in forma and "SVE" not in forma


def test_htmx_je_vendorovan():
    """Bez CDN-a i bez build koraka."""
    p = Path(__file__).resolve().parents[1] / "subtitis" / "web" / "static" / "htmx.min.js"
    assert p.exists() and p.stat().st_size > 10_000


# ==========================================================================
# Upload: model se ne bira, VAD ima imena, jezik sme da bude `auto`
# ==========================================================================


def test_model_se_ne_bira_na_ekranu(web):
    """Spuštanje na `medium` daje lošiji srpski prepis, bez ijednog objašnjenja."""
    client, _, _ = web
    page = client.get("/").text
    assert "large-v3" not in page
    assert "medium" not in page


def test_upload_uvek_salje_podrazumevani_model(web):
    client, db, _ = web
    r = client.post(
        "/jobs", files={"audio": ("t.wav", b"RIFF")},
        data={"language": "sr", "vad_preset": "normalno"}, follow_redirects=False,
    )
    assert r.status_code == 303
    jid = r.headers["location"].rsplit("/", 1)[-1]
    from subtitis import config as cfg

    with db.connect() as conn:
        row = conn.execute("SELECT params FROM tasks WHERE job_id=?", (jid,)).fetchone()
    assert json.loads(row["params"])["model"] == cfg.DEFAULT_ASR_MODEL


def test_vad_je_imenovani_izbor(web):
    """Krug 8: filter se bira u izveštaju projekta (ponovno prepoznavanje),
    ne na početnoj; endpoint za nov projekat i dalje prima izbor."""
    client, db, cfg = web
    assert "tih snimak" not in client.get("/").text
    page = client.get(f"/jobs/{make_job(cfg, db, job_id='vad')}").text
    form = page.split('id="retranscribe"')[1].split("</form>")[0]
    assert "tih snimak" in form and "isključeno" in form
    assert "0.35" not in form, "broj korisniku ne znači ništa"

    r = client.post(
        "/jobs", files={"audio": ("t.wav", b"RIFF")},
        data={"language": "sr", "vad_preset": "tih"}, follow_redirects=False,
    )
    jid = r.headers["location"].rsplit("/", 1)[-1]
    with db.connect() as conn:
        row = conn.execute("SELECT params FROM tasks WHERE job_id=?", (jid,)).fetchone()
    assert json.loads(row["params"])["vad_preset"] == "tih"


def test_izmisljen_vad_izbor_se_odbija(web):
    client, db, _ = web
    r = client.post(
        "/jobs", files={"audio": ("t.wav", b"RIFF")},
        data={"language": "sr", "vad_preset": "kako-god"}, follow_redirects=False,
    )
    assert r.status_code == 400


def test_auto_jezik_je_ponudjen(web):
    client, _, _ = web
    page = client.get("/").text
    assert 'value="auto"' in page
    # Savet uz polje, ne paragraf ispod forme.
    assert "prepozna kao hrvatski" in page
    assert "subtitis worker" not in page, "uputstvo o workeru je nestalo sa ekrana"


def test_upload_veci_od_ogranicenja_se_odbija(web, monkeypatch):
    client, db, cfg = web
    from subtitis.web import app as app_mod

    monkeypatch.setattr(app_mod, "MAX_UPLOAD_BYTES", 16)
    r = client.post(
        "/jobs", files={"audio": ("veliki.wav", b"x" * 4096)},
        data={"language": "sr", "vad_preset": "normalno"}, follow_redirects=False,
    )
    assert r.status_code == 413
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM jobs").fetchone()["c"] == 0
    # Nedovršen upload se ne ostavlja na disku.
    assert not list((cfg.JOBS_DIR).rglob("veliki.wav"))


# ==========================================================================
# Indikator obrade
# ==========================================================================


def test_indikator_je_crven_kad_worker_ne_radi(web):
    client, _, _ = web
    frag = client.get("/worker/status").text
    assert "dot off" in frag
    assert "obrada ne radi" in frag


def test_indikator_javlja_koliko_zadataka_ceka(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    db.enqueue(jid, "transcribe")
    frag = client.get("/worker/status").text
    assert "dot off" in frag
    assert "1 zadataka čeka" in frag
def test_stranica_prikazuje_sta_je_model_prepoznao(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    from subtitis.models import AsrMeta, WordsFile

    path = cfg.job_dir(jid) / "words.sr.json"
    wf = WordsFile.model_validate(cfg.read_json(path))
    wf.asr = AsrMeta(model="test", detected_language="sr", detected_probability=0.97)
    cfg.write_model(path, wf)

    page = client.get(f"/jobs/{jid}").text
    assert "jezik: sr (97%)" in page


def test_neslaganje_zadatog_i_detektovanog_se_istice(web):
    """Zadato `sr`, model tvrdi `hr` — jedini trag zašto je prepis ijekavski."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    from subtitis.models import AsrMeta, WordsFile

    path = cfg.job_dir(jid) / "words.sr.json"
    wf = WordsFile.model_validate(cfg.read_json(path))
    wf.asr = AsrMeta(model="test", detected_language="hr", detected_probability=0.8)
    cfg.write_model(path, wf)

    page = client.get(f"/jobs/{jid}").text
    assert "jezik: hr (80%)" in page
    assert "zadato je sr" in page


# ==========================================================================
# Profil: izbor se pamti, i vidi se šta je primenjeno
# ==========================================================================


def build_record(cfg, jid, profile="tv", script="Latn", lang="sr", cues=12):
    """Upiši u job.json da je SRT napravljen tim profilom."""
    from subtitis.models import PhaseRecord

    job = cfg.load_job(jid)
    job.history.append(
        PhaseRecord(
            phase="build", status="ok",
            started_at="2026-08-12T10:00:00+00:00",
            finished_at="2026-08-12T10:00:01+00:00",
            duration_s=1.0,
            detail={"profile": profile, "script": script, "language": lang, "cues": cues},
        )
    )
    cfg.save_job(job)


def selected_option(page: str, field: str) -> str | None:
    """Vrednost `<option ... selected>` u zadatom `<select>`."""
    import re

    blok = page.split(f'name="{field}"')[1].split("</select>")[0]
    for opt in re.findall(r"<option[^>]*>", blok):
        if "selected" in opt:
            m = re.search(r'value="([^"]*)"', opt)
            return m.group(1) if m else None
    return None


def test_izabrani_profil_ostaje_izabran(web):
    """Regresija: meni se vraćao na netflix i posle prelamanja po `web`."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    build_record(cfg, jid, profile="tv")
    page = client.get(f"/jobs/{jid}").text
    assert selected_option(page, "profile") == "tv"


def test_stranica_kaze_cime_je_srt_napravljen(web):
    """Svi profili pišu u isti fajl — bez ovoga se ne zna šta je primenjeno."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    build_record(cfg, jid, profile="tv", script="Cyrl", cues=44)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Cyrl.srt").write_text("1\r\n", encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    assert "profilom <strong>tv</strong>" in page
    assert "44 cue-ova" in page


def test_bez_prelamanja_stoji_podrazumevani_profil(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert selected_option(page, "profile") == cfg.DEFAULT_PROFILE


def test_izabrano_pismo_ostaje_izabrano(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    build_record(cfg, jid, script="Cyrl")
    page = client.get(f"/jobs/{jid}").text
    assert selected_option(page, "script") == "Cyrl"


def test_profili_imaju_brojeve_i_opis(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert "znakova/red" in page and "CPS" in page
    # Opis dolazi iz samog profila, pa ne može da se raziđe sa brojevima.
    assert cfg.load_profile("tv").note[:20] in page


def test_primer_preloma_je_iz_samog_posla(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    # Dovoljno duga rečenica da se profili razlikuju.
    from subtitis.models import Sentence, SentencesFile, Word, WordsFile

    text = (
        "Ovo je namerno duga rečenica koja mora da se prelomi na više redova "
        "da bi se razlika između profila uopšte videla u primeru."
    )
    words, t = [], 0.0
    for k, w in enumerate(text.split()):
        words.append(Word(i=k, s=round(t, 2), e=round(t + 0.3, 2), w=w, p=0.9))
        t += 0.35
    cfg.write_model(
        cfg.job_dir(jid) / "words.sr.json",
        WordsFile(job_id=jid, source_file="x", duration=t, language="sr",
                  asr=AsrMeta(model="test"), words=words),
    )
    cfg.write_model(
        cfg.job_dir(jid) / "sentences.sr.json",
        SentencesFile(job_id=jid, language="sr", derived_from="words.sr.json",
                      sentences=[Sentence(id=1, s=0.0, e=t, text=text,
                                          words=(0, len(words) - 1))]),
    )
    page = client.get(f"/jobs/{jid}").text
    assert "cue-preview" in page
    assert "namerno duga" in page, "primer mora biti iz OVOG posla"


# ==========================================================================
# Pismo se ne nudi kad nema smisla
# ==========================================================================


def test_izbor_pisma_je_sakriven_css_om_a_ne_samo_atributom(web):
    """Regresija: `.row-form label{display:flex}` je pobeđivao `[hidden]`."""
    import re

    css = Path(__file__).resolve().parents[1] / "subtitis" / "web" / "static" / "app.css"
    text = re.sub(r"/\*.*?\*/", "", css.read_text(encoding="utf-8"), flags=re.S)
    m = re.search(r"\[hidden\]\s*\{([^}]*)\}", text)
    assert m, "nema pravila za [hidden]"
    assert "display" in m.group(1) and "!important" in m.group(1)


def test_greska_iz_obrasca_je_poruka_a_ne_sirov_json(web):
    client, db, cfg = web
    jid = make_job(cfg, db, lang="fr")
    r = client.post(
        f"/jobs/{jid}/translate",
        data={"to": "fr", "script": "Cyrl"},
        headers={"accept": "text/html"},
        follow_redirects=False,
    )
    assert r.status_code == 400
    assert "text/html" in r.headers["content-type"]
    assert "Nije prošlo" in r.text
    assert "Ćirilica nema smisla" in r.text
    assert '{"detail"' not in r.text


def test_poziv_iz_javascripta_i_dalje_dobija_json(web):
    client, db, cfg = web
    r = client.post(
        "/jobs/nepostojeci/sentences",
        json={"lang": "sr", "sentences": []},
        headers={"accept": "application/json"},
    )
    assert r.status_code == 404
    assert r.json()["detail"]


# ==========================================================================
# Editor: original i prevod uporedo
# ==========================================================================


def make_translation(cfg, db, jid, tgt="en", src="sr"):
    """Dodaj prevod vezan po istom `id` kao original."""
    cfg.write_model(
        cfg.job_dir(jid) / f"sentences.{tgt}.json",
        SentencesFile(
            job_id=jid, language=tgt, derived_from=f"sentences.{src}.json",
            sentences=[Sentence(id=1, s=0.0, e=1.05, text="Good day.", words=(0, 2))],
        ),
    )


def test_prevod_prikazuje_original_u_istom_redu(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    make_translation(cfg, db, jid)
    page = client.get(f"/jobs/{jid}?lang=en").text
    assert "original (sr)" in page and "prevod (en)" in page
    # Original ide kroz bojenje, pa je razbijen na tokene.
    assert "Dobar" in page, "original mora biti na ekranu"
    assert "Good day." in page


def test_original_je_samo_za_citanje(web):
    """Ispravke se čuvaju u fajlu OTVORENOG jezika — izmena originala bi nestala."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    make_translation(cfg, db, jid)
    page = client.get(f"/jobs/{jid}?lang=en").text
    red = page.split('<tr data-id="1"')[1].split("</tr>")[0]
    assert red.count("contenteditable") == 1, "uređuje se samo prevod"


def test_bojenje_vazi_samo_za_original(web):
    """Prevod nema verovatnoće po rečima — žuto na njemu bi bilo izmišljeno."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    make_translation(cfg, db, jid)
    page = client.get(f"/jobs/{jid}?lang=en").text
    red = page.split('<tr data-id="1"')[1].split("</tr>")[0]
    src_cell = red.split('class="src-col source"')[1].split("</td>")[0]
    tgt_cell = red.split("contenteditable")[1].split("</div>")[0]
    assert '<span class="low"' in src_cell, "`dan` ima p=0.42"
    assert '<span class="low"' not in tgt_cell


def test_original_bez_prevoda_nema_drugu_kolonu(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert "src-col" not in page
    assert "prevod (" not in page


def test_pivot_uzima_pravi_original(web):
    """sr -> en -> de: original za `de` je `en`, ne `sr`."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    make_translation(cfg, db, jid, tgt="en", src="sr")
    cfg.write_model(
        cfg.job_dir(jid) / "sentences.de.json",
        SentencesFile(
            job_id=jid, language="de", derived_from="sentences.en.json",
            sentences=[Sentence(id=1, s=0.0, e=1.05, text="Guten Tag.", words=(0, 2))],
        ),
    )
    page = client.get(f"/jobs/{jid}?lang=de").text
    assert "original (en)" in page
    assert "Good day." in page
    assert "Dobar dan." not in page


# ==========================================================================
# Automatsko čišćenje radnih WAV-ova
# ==========================================================================


def test_stari_wav_se_brise(web):
    import os
    import time

    client, db, cfg = web
    jid = make_job(cfg, db)
    wav = cfg.job_dir(jid) / "source.wav"
    wav.write_bytes(b"x" * 2048)
    star = time.time() - (cfg.SOURCE_WAV_TTL_DAYS + 1) * 86400
    os.utime(wav, (star, star))

    freed = cfg.purge_old_sources()
    assert [j for j, _ in freed] == [jid]
    assert not wav.exists()
    # Rezultat ostaje.
    assert (cfg.job_dir(jid) / "sentences.sr.json").exists()
    assert "čišćenje" in (cfg.job_dir(jid) / "log.txt").read_text(encoding="utf-8")


def test_svez_wav_se_ne_dira(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    wav = cfg.job_dir(jid) / "source.wav"
    wav.write_bytes(b"x" * 2048)
    assert cfg.purge_old_sources() == []
    assert wav.exists()


def test_ttl_nula_iskljucuje_ciscenje(web):
    import os
    import time

    client, db, cfg = web
    jid = make_job(cfg, db)
    wav = cfg.job_dir(jid) / "source.wav"
    wav.write_bytes(b"x")
    star = time.time() - 999 * 86400
    os.utime(wav, (star, star))
    assert cfg.purge_old_sources(ttl_days=0) == []
    assert wav.exists()


# ==========================================================================
# Verzije i povratak
# ==========================================================================


def edit(client, jid, text, *, lang="sr", auto=False):
    return client.post(
        f"/jobs/{jid}/sentences",
        json={"lang": lang, "sentences": [{"id": 1, "text": text}], "autosave": auto},
    )


def test_rucno_snimanje_uvek_pravi_verziju(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    assert edit(client, jid, "Prva izmena.").json()["backup"]
    assert edit(client, jid, "Druga izmena.").json()["backup"]
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    assert len(vers) == 2


def test_automatsko_snimanje_ne_pravi_verziju_svaki_put(web):
    """Snimanje na 30 s bi inace ostavljalo stotine kopija dnevno."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    prva = edit(client, jid, "A.", auto=True).json()
    druga = edit(client, jid, "B.", auto=True).json()
    assert prva["backup"], "prva kopija se pravi jer je nema"
    assert druga["backup"] is None, "druga je unutar praga"
    assert druga["changed"] == 1, "sadrzaj se ipak snima"


def test_automatsko_snimanje_ne_stavlja_prelamanje_u_red(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    edit(client, jid, "Izmena.", auto=True)
    edit(client, jid, "Druga.")
    with db.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) c FROM tasks WHERE job_id=?", (jid,)
        ).fetchone()["c"]
    assert n == 0, "prelama editor, sinhrono (krug 13)"


def test_automatsko_snimanje_preskace_praznu_recenicu(web):
    """Usred kucanja je prazan red normalno stanje, ne greska."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = edit(client, jid, "", auto=True)
    assert r.status_code == 200
    assert r.json()["skipped_empty"] == 1
    assert r.json()["changed"] == 0
    sf = cfg.read_json(cfg.job_dir(jid) / "sentences.sr.json")
    assert sf["sentences"][0]["text"] == "Dobar dan.", "stari tekst ostaje"


def test_rucno_snimanje_prazne_recenice_je_greska(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    assert edit(client, jid, "").status_code == 400


def test_povratak_na_verziju_vraca_tekst(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    edit(client, jid, "Pokvareno.")
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    r = client.post(
        f"/jobs/{jid}/restore", json={"lang": "sr", "name": vers[0]["name"]}
    )
    assert r.status_code == 200
    sf = cfg.read_json(cfg.job_dir(jid) / "sentences.sr.json")
    assert sf["sentences"][0]["text"] == "Dobar dan."


def test_povratak_i_sam_moze_da_se_ponisti(web):
    """Tekuce stanje se snima PRE vracanja."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    edit(client, jid, "Pokvareno.")
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    client.post(f"/jobs/{jid}/restore", json={"lang": "sr", "name": vers[0]["name"]})

    posle = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    assert len(posle) == len(vers) + 1
    novi = cfg.job_dir(jid) / posle[0]["name"]
    data = json.loads(novi.read_text(encoding="utf-8"))
    assert data["sentences"][0]["text"] == "Pokvareno."


def test_povratak_regenerise_srt(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    edit(client, jid, "Pokvareno.")
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    with db.connect() as conn:
        conn.execute("DELETE FROM tasks")
    client.post(f"/jobs/{jid}/restore", json={"lang": "sr", "name": vers[0]["name"]})
    with db.connect() as conn:
        kinds = [r["kind"] for r in conn.execute("SELECT kind FROM tasks").fetchall()]
    assert kinds == ["build"]


def test_nepostojeca_verzija_se_odbija(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(f"/jobs/{jid}/restore", json={"lang": "sr", "name": "nema.bak"})
    assert r.status_code == 404


def test_verzije_su_najnovija_prva(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    path = cfg.job_dir(jid) / "sentences.sr.json"
    for stamp in ("20260810-100000", "20260812-100000", "20260811-100000"):
        (path.parent / f"{path.name}.{stamp}.bak").write_bytes(path.read_bytes())
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    assert [v["name"].rsplit(".", 2)[-2] for v in vers] == [
        "20260812-100000", "20260811-100000", "20260810-100000",
    ]
    assert all(v["sentences"] == 1 for v in vers)


def test_pokvarena_kopija_se_ne_nudi(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    path = cfg.job_dir(jid) / "sentences.sr.json"
    bad = path.parent / f"{path.name}.20260812-100000.bak"
    bad.write_text("{ nije json", encoding="utf-8")
    assert client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"] == []


def test_pokvarena_kopija_ne_moze_da_pregazi_fajl(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    path = cfg.job_dir(jid) / "sentences.sr.json"
    ime = f"{path.name}.20260812-100000.bak"
    (path.parent / ime).write_text('{"schema_version": 1}', encoding="utf-8")
    r = client.post(f"/jobs/{jid}/restore", json={"lang": "sr", "name": ime})
    assert r.status_code >= 400
    sf = cfg.read_json(path)
    assert sf["sentences"][0]["text"] == "Dobar dan.", "original netaknut"


# ==========================================================================
# Sazetak: sta je uradjeno
# ==========================================================================


def phase(name, seconds, **detail):
    from subtitis.models import PhaseRecord

    return PhaseRecord(
        phase=name, status="ok",
        started_at="2026-08-12T10:00:00+00:00",
        finished_at="2026-08-12T10:00:20+00:00",
        duration_s=seconds, detail=detail,
    )


def test_sazetak_prikazuje_model_i_recenice(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    job = cfg.load_job(jid)
    job.history += [
        phase("asr", 20.0, model="large-v3", words=442, vad=True),
        phase("segment", 1.0, sentences=27, coverage_pct=92.5, low_confidence=10),
    ]
    cfg.save_job(job)
    page = client.get(f"/jobs/{jid}").text
    assert "Izveštaj o obradi" in page
    assert "large-v3" in page and "<dt>reči</dt><dd>442</dd>" in page
    assert "<dd>27 · 10 nesigurnih</dd>" in page and "92,5%" in page


def test_sazetak_prikazuje_punktuaciju_i_ustupke(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    job = cfg.load_job(jid)
    job.history.append(phase(
        "punctuate", 60.0, model="gemma3:12b", marks_added=48, chunks_ok=8,
        chunks_failed=1, seconds_total=60.0, seconds_wasted=6.0,
        failed_ranges=[], concessions={"split": 1, "inflection": 1},
    ))
    cfg.save_job(job)
    page = client.get(f"/jobs/{jid}").text
    assert "48 znakova" in page
    assert "gemma3:12b" in page
    assert "8/9 čankova" in page
    assert "6,0 s bačeno" in page
    assert "split 1" in page and "inflection 1" in page


def test_sazetak_kaze_kad_punktuacija_nije_pokretana(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    job = cfg.load_job(jid)
    job.history.append(phase("asr", 20.0, model="large-v3", words=10, vad=True))
    cfg.save_job(job)
    assert "nije pokretana" in client.get(f"/jobs/{jid}").text


def test_sazetak_ne_pada_bez_istorije(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    assert client.get(f"/jobs/{jid}").status_code == 200


def test_dva_snimanja_u_istoj_sekundi_daju_dve_verzije(web):
    """Ime kopije ima rezoluciju od sekunde — bez brojaca bi druga pregazila prvu."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    edit(client, jid, "Prva.")
    edit(client, jid, "Druga.")
    edit(client, jid, "Treca.")
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]
    assert len(vers) == 3, "nijedna verzija ne sme da nestane"
    texts = [
        json.loads((cfg.job_dir(jid) / v["name"]).read_text(encoding="utf-8"))
        ["sentences"][0]["text"]
        for v in vers
    ]
    # Najnovija kopija drzi stanje pre poslednjeg snimanja.
    assert texts == ["Druga.", "Prva.", "Dobar dan."]


# ==========================================================================
# Podesavanja
# ==========================================================================


def test_ekran_prikazuje_tekuce_vrednosti(web):
    client, db, cfg = web
    page = client.get("/settings").text
    assert "Podrazumevane vrednosti" in page
    assert cfg.default_profile() in page
    assert cfg.llm_model() in page


def _drugi_profil(cfg, tmp_path, monkeypatch, name="drugi"):
    """Od kruga 6 postoji jedan stvarni profil; podesavanja traze drugi."""
    profiles = tmp_path / "profiles"
    profiles.mkdir(exist_ok=True)
    (profiles / "tv.json").write_text((cfg.PROFILES_DIR / "tv.json").read_text(encoding="utf-8"), encoding="utf-8")
    cfg.write_json(profiles / f"{name}.json", {"name": name, "max_chars_per_line": 37})
    monkeypatch.setattr(cfg, "PROFILES_DIR", profiles)
    return name


def test_snimanje_podesavanja_menja_ponasanje(web, tmp_path, monkeypatch):
    """Podesavanje koje se ne cita nigde je samo ukras."""
    client, db, cfg = web
    drugi = _drugi_profil(cfg, tmp_path, monkeypatch)
    assert cfg.default_profile() != drugi
    r = client.post("/settings", data={
        "default_profile": drugi,
        "default_target_language": "de",
        "llm_model": "moj-model",
        "source_wav_ttl_days": "3",
        "auto_punctuate": "on",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert cfg.default_profile() == drugi
    assert cfg.default_target_language() == "de"
    assert cfg.llm_model() == "moj-model"
    assert cfg.source_wav_ttl_days() == 3
    assert cfg.auto_punctuate() is True
    assert cfg.CONFIG_FILE.exists()


def test_podesavanja_stizu_do_ekrana_posla(web, tmp_path, monkeypatch):
    client, db, cfg = web
    drugi = _drugi_profil(cfg, tmp_path, monkeypatch)
    jid = make_job(cfg, db)
    client.post("/settings", data={
        "default_profile": drugi, "default_target_language": "de",
        "llm_model": "moj-model", "source_wav_ttl_days": "7",
    }, follow_redirects=False)
    page = client.get(f"/jobs/{jid}").text
    assert selected_option(page, "profile") == drugi
    assert selected_option(page, "to") == "de"
    # krug 10: model i pivot se biraju samo u podešavanjima, ne po projektu
    assert 'name="model"' not in page and 'name="pivot"' not in page
    assert "moj-model" in client.get("/settings").text


def test_iskljuceno_auto_punctuate_se_snima(web):
    client, db, cfg = web
    client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "en",
        "llm_model": "m", "source_wav_ttl_days": "7",
    }, follow_redirects=False)
    assert cfg.auto_punctuate() is False


def test_nepoznat_profil_se_odbija(web):
    client, db, cfg = web
    r = client.post("/settings", data={
        "default_profile": "izmisljen", "default_target_language": "en",
        "llm_model": "m", "source_wav_ttl_days": "7",
    }, follow_redirects=False)
    assert r.status_code == 400
    assert not cfg.CONFIG_FILE.exists(), "nista se ne upisuje kad ulaz nije dobar"


def test_negativan_broj_dana_se_odbija(web):
    client, db, cfg = web
    r = client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "en",
        "llm_model": "m", "source_wav_ttl_days": "-1",
    }, follow_redirects=False)
    assert r.status_code == 400


def test_prazan_model_se_odbija(web):
    client, db, cfg = web
    r = client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "en",
        "llm_model": "   ", "source_wav_ttl_days": "7",
    }, follow_redirects=False)
    assert r.status_code == 400


def test_nepoznat_kljuc_se_ne_upisuje(web):
    client, db, cfg = web
    with pytest.raises(Exception):
        cfg.save_settings({"nepoznato": 1})


def test_ekran_javlja_da_promenljiva_okruzenja_pobedjuje(web, monkeypatch):
    client, db, cfg = web
    monkeypatch.setenv("SUBTITIS_LLM_MODEL", "iz-okruzenja")
    page = client.get("/settings").text
    assert "SUBTITIS_LLM_MODEL=iz-okruzenja" in page


def test_ttl_iz_podesavanja_utice_na_ciscenje(web):
    import os
    import time

    client, db, cfg = web
    jid = make_job(cfg, db)
    wav = cfg.job_dir(jid) / "source.wav"
    wav.write_bytes(b"x" * 16)
    star = time.time() - 5 * 86400
    os.utime(wav, (star, star))

    assert cfg.purge_old_sources() == [], "podrazumevanih 7 dana jos nije proslo"
    client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "en",
        "llm_model": "m", "source_wav_ttl_days": "3",
    }, follow_redirects=False)
    assert [j for j, _ in cfg.purge_old_sources()] == [jid]


# ==========================================================================
# Glosar kroz interfejs
# ==========================================================================


def test_dodavanje_unosa_u_glosar(web):
    client, db, cfg = web
    r = client.post("/settings/glossary", data={
        "pair": "sr->en", "term": "radionica", "translation": "workshop",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert cfg.load_glossary()["sr->en"]["radionica"] == "workshop"
    assert "radionica" in client.get("/settings").text


def test_brisanje_unosa_iz_glosara(web):
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"a": "b", "c": "d"}})
    r = client.post("/settings/glossary", data={
        "pair": "sr->en", "term": "a", "translation": "b", "remove": "on",
    }, follow_redirects=False)
    assert r.status_code == 303
    assert cfg.load_glossary()["sr->en"] == {"c": "d"}


def test_glosar_pravi_bak_kopiju(web):
    """Glosar je rucno kuriran i ulazi u ASR i prevod — brisanje mora da se vrati."""
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"a": "b"}})
    client.post("/settings/glossary", data={
        "pair": "sr->en", "term": "a", "translation": "b", "remove": "on",
    }, follow_redirects=False)
    kopije = list(cfg.GLOSSARY_DIR.glob("default.json.*.bak"))
    assert kopije, "prethodna verzija mora da postoji"
    stara = json.loads(kopije[-1].read_text(encoding="utf-8"))
    assert stara["sr->en"]["a"] == "b"


def test_los_smer_se_odbija(web):
    client, db, cfg = web
    r = client.post("/settings/glossary", data={
        "pair": "srpski", "term": "a", "translation": "b",
    }, follow_redirects=False)
    assert r.status_code == 400


def test_brisanje_nepostojeceg_unosa_je_404(web):
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"a": "b"}})
    r = client.post("/settings/glossary", data={
        "pair": "sr->en", "term": "nema", "translation": "", "remove": "on",
    }, follow_redirects=False)
    assert r.status_code == 404


def test_prazan_prevod_se_odbija(web):
    client, db, cfg = web
    r = client.post("/settings/glossary", data={
        "pair": "sr->en", "term": "a", "translation": "  ",
    }, follow_redirects=False)
    assert r.status_code == 400


# ==========================================================================
# Ispis na konzolu ne sme da obori posao
#
# Windows konzola je cp1252, a `pythonw.exe` (desktop rezim) ima isti takav
# stdout. `print("vraceno")` sa nasim slovima baca UnicodeEncodeError i rusi
# ceo HTTP zahtev usred snimanja.
# ==========================================================================


class Cp1252Stream:
    """Stream koji puca na sve van cp1252, kao prava Windows konzola."""

    encoding = "cp1252"

    def __init__(self):
        self.written = []

    def write(self, text):
        text.encode("cp1252")   # baca UnicodeEncodeError
        self.written.append(text)

    def flush(self):
        pass


def test_echo_ne_puca_na_cp1252(monkeypatch):
    import sys

    from subtitis import config as cfg

    stream = Cp1252Stream()
    monkeypatch.setattr(sys, "stdout", stream)
    cfg.echo("vraćeno na verziju: rečenica, čišćenje, đačko")
    assert stream.written, "nesto je ipak ispisano"
    assert "?" in stream.written[0], "kvacice su zamenjene, ne izgubljene tiho"


def test_echo_prezivljava_stdout_none(monkeypatch):
    import sys

    from subtitis import config as cfg

    monkeypatch.setattr(sys, "stdout", None)
    cfg.echo("bilo šta")   # ne sme da baci


def test_echo_prezivljava_zatvoren_stream(monkeypatch):
    import io
    import sys

    from subtitis import config as cfg

    stream = io.StringIO()
    stream.close()
    monkeypatch.setattr(sys, "stdout", stream)
    cfg.echo("bilo šta")   # ne sme da baci


def test_log_upisuje_utf8_i_kad_konzola_ne_moze(web, monkeypatch):
    """Fajl je UTF-8 i uvek tacan; konzola sme da izgubi kvacicu."""
    import sys

    client, db, cfg = web
    jid = make_job(cfg, db)
    monkeypatch.setattr(sys, "stdout", Cp1252Stream())
    cfg.log(jid, "vraćeno na verziju, čišćenje đubreta")
    text = (cfg.job_dir(jid) / "log.txt").read_text(encoding="utf-8")
    assert "vraćeno na verziju, čišćenje đubreta" in text


def test_povratak_na_verziju_radi_na_cp1252_konzoli(web, monkeypatch):
    """Regresija: `cfg.log` u `restore` je rusio zahtev sa 500."""
    import sys

    client, db, cfg = web
    jid = make_job(cfg, db)
    edit(client, jid, "Pokvareno.")
    vers = client.get(f"/jobs/{jid}/versions?lang=sr").json()["versions"]

    monkeypatch.setattr(sys, "stdout", Cp1252Stream())
    r = client.post(
        f"/jobs/{jid}/restore", json={"lang": "sr", "name": vers[0]["name"]}
    )
    assert r.status_code == 200
    sf = cfg.read_json(cfg.job_dir(jid) / "sentences.sr.json")
    assert sf["sentences"][0]["text"] == "Dobar dan."


def test_snimanje_radi_na_cp1252_konzoli(web, monkeypatch):
    import sys

    client, db, cfg = web
    jid = make_job(cfg, db)
    monkeypatch.setattr(sys, "stdout", Cp1252Stream())
    r = edit(client, jid, "Izmenjena rečenica sa čkvačicama.")
    assert r.status_code == 200
    assert r.json()["changed"] == 1


def test_editor_boji_reci_bez_zvuka_crveno_i_samo_njih(web):
    """Krug 5: niz reči nultog trajanja je crven (ne žut), sa napomenom; reč
    koju je čovek umetnuo ne boji se ni kad joj poravnanje izmisli nulto vreme."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    jdir = cfg.job_dir(jid)
    words = [
        Word(i=0, s=0.0, e=0.5, w="Dobar", p=0.95),
        Word(i=1, s=0.5, e=1.0, w="dan", p=0.95),
        Word(i=2, s=1.0, e=1.0, w="OKC", p=0.99),
        Word(i=3, s=1.0, e=1.0, w="Divčibare", p=0.99),
    ]
    cfg.write_model(jdir / "words.sr.json", WordsFile(
        job_id=jid, source_file="x", duration=2.0, language="sr",
        asr=AsrMeta(model="test"), words=words,
    ))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[
            Sentence(id=1, s=0.0, e=1.0, text="Dobar dan", words=(0, 1)),
            Sentence(id=2, s=1.0, e=1.0, text="OKC Divčibare", words=(2, 3)),
        ],
    ))
    page = client.get(f"/jobs/{jid}").text
    assert '<span class="halluc"' in page and ">OKC</span>" in page
    row1 = page.split('<tr data-id="1"')[1].split("</tr>")[0]
    row2 = page.split('<tr data-id="2"')[1].split("</tr>")[0]
    assert "flag-halluc" not in row1.split(">")[0]
    assert "flag-halluc" in row2.split(">")[0]
    assert "Gotovo sigurno su izmišljene" in row2

    # čovek izbaci izmišljeno i upiše svoje — crveno nestaje
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    sf.sentences[1].text = "Hvala"
    cfg.write_model(jdir / "sentences.sr.json", sf)
    page = client.get(f"/jobs/{jid}").text
    assert '<span class="halluc"' not in page


def test_izmisljena_recenica_cela_u_traci_i_u_panelu_za_preuzimanje(web):
    """Krug 23: traka kaže da je sumnjiva CELA rečenica (i reči sa razvučenim
    vremenom, koje pravilo ne hvata), a panel za preuzimanje nosi istu brojku
    i broj natpisa po fajlu — bez blokiranja preuzimanja."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    jdir = cfg.job_dir(jid)
    words = [
        Word(i=0, s=0.0, e=0.5, w="Dobar", p=0.95),
        Word(i=1, s=0.5, e=1.0, w="dan", p=0.95),
        Word(i=2, s=3.0, e=3.0, w="OKC", p=0.99),
        Word(i=3, s=3.0, e=3.0, w="Divčibare", p=0.99),
        Word(i=4, s=3.0, e=4.2, w="Srbija", p=0.2),  # razvučena — pravilo je ne hvata
    ]
    cfg.write_model(jdir / "words.sr.json", WordsFile(
        job_id=jid, source_file="x", duration=5.0, language="sr",
        asr=AsrMeta(model="test"), words=words,
    ))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[
            Sentence(id=1, s=0.0, e=1.0, text="Dobar dan", words=(0, 1)),
            Sentence(id=2, s=3.0, e=4.2, text="OKC Divčibare Srbija", words=(2, 4)),
        ],
    ))
    (jdir / "out").mkdir(exist_ok=True)
    srt = ("1\n00:00:00,000 --> 00:00:01,000\nDobar dan\n\n"
           "2\n00:00:03,000 --> 00:00:03,500\nOKC Divčibare\n\n"
           "3\n00:00:03,583 --> 00:00:04,200\nSrbija\n")
    (jdir / "out" / "x.sr-Latn.srt").write_text(srt, encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    bar = page.split('class="flagbar err"')[1].split("</div>")[0]
    red = bar.split('<span class="flagtxt">')[1].split("</span>")[0]
    assert "1 rečenica bez zvuka iza sebe (3 reči)" in red, "brojka se vidi bez klika"
    assert "rečenica 2 u 00:03,00: 3 reči, od toga 2 bez trajanja" in bar.split("<template>")[1]
    panel = page.split('id="downloads"')[1].split("</section>")[0]
    assert "1 rečenica bez zvuka iza sebe" in panel and "(3 reči)" in panel
    assert "2 izmišljena" in panel  # natpisi 2 i 3, natpis 1 je govor
    assert "Fajl se preuzima kakav jeste" in panel
    assert client.get(f"/jobs/{jid}/download/x.sr-Latn.srt").status_code == 200

    # izbačena rečenica -> nestaje i traka i oznaka u panelu
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    sf.sentences[1].text = "Hvala"
    cfg.write_model(jdir / "sentences.sr.json", sf)
    page = client.get(f"/jobs/{jid}").text
    assert "bez zvuka iza sebe" not in page.split('id="downloads"')[1].split("</section>")[0]


# ==========================================================================
# Glosar posla (krug 5)
# ==========================================================================


def _upload(client, **form):
    data = {"language": "sr", "vad_preset": "iskljuceno", **form}
    return client.post("/jobs", files={"audio": ("radionica.wav", b"RIFF", "audio/wav")},
                       data=data, follow_redirects=False)


def test_nov_posao_podrazumevano_dobija_prazan_glosar(web):
    """Krug 23: glosar je alat za jedan snimak, ne stalna postavka."""
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"Radionica otvorenih vrata": "Open Door Workshop"}})
    assert cfg.new_project_glossary() == "none"
    jid = _upload(client).headers["location"].rsplit("/", 1)[-1]
    assert cfg.job_glossary(jid) == {}


def test_nov_posao_dobija_kopiju_podrazumevanog_glosara_kad_se_izabere(web):
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"Radionica otvorenih vrata": "Open Door Workshop"}})
    jid = _upload(client, glossary_mode="default").headers["location"].rsplit("/", 1)[-1]
    assert cfg.job_glossary(jid) == {"sr->en": {"Radionica otvorenih vrata": "Open Door Workshop"}}
    # izmena podrazumevanog posle otvaranja ne menja posao
    cfg.save_glossary({"sr->en": {"Nešto drugo": "Something else"}})
    assert "Radionica otvorenih vrata" in cfg.glossary_keys(cfg.job_glossary(jid), "sr")


def test_prazan_glosar_je_legitiman_izbor(web):
    """Radionica draft 1: glosar prethodnog projekta je završio u prepisu preko džingla."""
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"OKC": "OKC", "Divčibare": "Divčibare"}})
    jid = _upload(client, glossary_mode="none").headers["location"].rsplit("/", 1)[-1]
    assert cfg.job_glossary(jid) == {}
    assert cfg.glossary_keys(cfg.job_glossary(jid), "sr") == []


def test_termini_snimka_idu_u_prepoznavanje_ne_u_prevod(web):
    client, db, cfg = web
    jid = _upload(client, glossary_mode="none",
                  terms="Valjevo\n\n  Mileva \nValjevo").headers["location"].rsplit("/", 1)[-1]
    g = cfg.job_glossary(jid)
    assert cfg.glossary_keys(g, "sr") == ["Valjevo", "Mileva"]
    assert cfg.recording_terms(g, "sr") == ["Valjevo", "Mileva"]
    assert cfg.hop_glossary(g, "sr", "en", origin="sr") == ({}, 0)


def test_stari_posao_bez_glosara_cita_podrazumevani(web):
    client, db, cfg = web
    cfg.save_glossary({"sr->en": {"Đurđević": "Đurđević"}})
    jid = make_job(cfg, db)
    assert not cfg.job_glossary_path(jid).exists()
    assert cfg.job_glossary(jid) == {"sr->en": {"Đurđević": "Đurđević"}}


def test_nepoznat_izbor_glosara_je_greska(web):
    client, db, cfg = web
    assert _upload(client, glossary_mode="svašta").status_code == 400


# ==========================================================================
# Upozorenje o tišini na ekranu (krug 5)
# ==========================================================================


def _with_asr(cfg, jid, *, vad, share):
    from subtitis.models import PhaseRecord

    job = cfg.load_job(jid)
    job.history.append(PhaseRecord(
        phase="asr", status="ok", started_at="x", finished_at="x", duration_s=1.0,
        detail={"model": "large-v3", "words": 3, "vad": vad, "vad_discarded_share": share},
    ))
    cfg.save_job(job)


@pytest.mark.parametrize("vad,share,shown", [
    (False, 0.303, True),   # Radionica draft 1
    (False, 0.10, False),   # gust govor
    (True, 0.40, False),    # detekcija je uključena — tišina je već isečena
])
def test_upozorenje_o_tisini(web, vad, share, shown):
    client, db, cfg = web
    jid = make_job(cfg, db)
    _with_asr(cfg, jid, vad=vad, share=share)
    page = client.get(f"/jobs/{jid}").text
    # muzika izričito, ne samo tišina — džingl i špica su čest slučaj (krug 5)
    # tekst iz mokapa kruga 8; muzika i dalje izričito
    assert ("snimka nije govor</b>" in page) is shown
    if shown:
        assert "<b>30% snimka nije govor</b>" in page
        bar = page.split('class="flagbar warn"')[1].split("</div>")[0]
        assert "Ponovi sa filterom" in bar and 'value="normalno"' in bar


def test_izbor_profila_ima_tv_web_i_digital(web):
    """Krug 8: tri profila za ASR posao."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    blok = page.split('name="profile"')[1].split("</select>")[0]
    import re

    assert re.findall(r'<option[^>]*value="([^"]*)"', blok) == ["digital", "tv", "web"]


def test_prekratki_cue_ovi_se_vide_na_stranici(web):
    """Ne smeju proći tiho: broj, redni broj i vreme svakog."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    build_record(cfg, jid, profile="web")
    job = cfg.load_job(jid)
    job.history[-1].detail["too_short"] = [
        {"cue": 7, "s": 62.34, "e": 62.65, "dur": 0.31},
        {"cue": 19, "s": 125.0, "e": 125.42, "dur": 0.42},
    ]
    cfg.save_job(job)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Latn.srt").write_text("1\r\n", encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    assert "2 cue-ova kraće od 0,5 s" in " ".join(page.split())
    assert "#7 (00:01:02,340, 0,31 s)" in page
    assert "#19 (00:02:05,000, 0,42 s)" in page


def test_velicina_slova_ide_u_zadatak_i_ostaje_izabrana(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(f"/jobs/{jid}/build", data={
        "language": "sr", "profile": "tv", "script": "Latn", "case": "upper",
    }, follow_redirects=False)
    assert r.status_code == 303
    import json

    assert json.loads(db.pending_task(jid)["params"])["case"] == "upper"
    assert client.post(f"/jobs/{jid}/build", data={
        "language": "sr", "profile": "tv", "script": "Latn", "case": "naslovna",
    }).status_code == 400

    page = client.get(f"/jobs/{jid}").text
    assert selected_option(page, "case") == "normal", "bez prelamanja: normalno"
    assert "Menja samo izgled izlaznog titla." in page
    assert "Mala slova</strong> brišu velika u imenima" in page

    build_record(cfg, jid)
    job = cfg.load_job(jid)
    job.history[-1].detail["case"] = "upper"
    cfg.save_job(job)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Latn.srt").write_text("1\r\n", encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    assert selected_option(page, "case") == "upper"
    assert "Latn, VELIKA SLOVA" in page


def test_velicina_slova_se_bira_i_pri_prevodu(web):
    """Kao pismo: ko hoće prevod u VELIKIM slovima ne treba dva koraka."""
    import json

    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    forma = page.split(f'action="/jobs/{jid}/translate"')[1].split("</form>")[0]
    assert 'name="case"' in forma and 'value="upper"' in forma
    r = client.post(f"/jobs/{jid}/translate", data={"to": "en", "case": "upper"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert json.loads(db.pending_task(jid)["params"])["case"] == "upper"
    db.finish_task(db.pending_task(jid)["id"], "done")
    assert client.post(f"/jobs/{jid}/translate", data={"to": "en", "case": "x"}).status_code == 400


def test_digital_postavlja_izlaz_a_izbor_ostaje_slobodan(web):
    """Krug 8: profil nosi podrazumevani izlaz u opciji; oba izbora se šalju ručno."""
    import json
    import re

    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    blok = page.split('name="profile"')[1].split("</select>")[0]
    opt = re.search(r'<option value="digital"[^>]*>', blok).group(0)
    assert 'data-case="upper"' in opt and 'data-punct="drop"' in opt
    assert selected_option(page, "punct") == "keep", "podrazumevani tv zadržava interpunkciju"

    r = client.post(f"/jobs/{jid}/build", data={
        "language": "sr", "profile": "digital", "script": "Latn", "case": "normal", "punct": "keep",
    }, follow_redirects=False)
    assert r.status_code == 303
    params = json.loads(db.pending_task(jid)["params"])
    assert (params["case"], params["drop_punct"]) == ("normal", False), "ručni izbor ima prednost"

    build_record(cfg, jid, profile="digital")
    job = cfg.load_job(jid)
    job.history[-1].detail.update(case="upper", drop_punct=True)
    cfg.save_job(job)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Latn.srt").write_text("1\r\n", encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    assert "Latn, VELIKA SLOVA, bez interpunkcije" in page
    assert selected_option(page, "punct") == "drop"


# ==========================================================================
# Krug 8: nov interfejs
# ==========================================================================


def test_pocetna_ima_dva_ulaza_bez_profila_glosara_i_filtera(web):
    client, db, cfg = web
    page = client.get("/").text
    assert "Transkribuj" in page and "Prevod SRT-a" in page and "Baza projekata" in page
    forma = page.split('action="/jobs"')[1].split("</form>")[0]
    for polje in ('name="profile"', 'name="glossary_mode"', 'name="vad_preset"'):
        assert polje not in forma, polje
    # termini iz snimka su tu, ali sakriveni iza veze (krug 8)
    zatvoreno = forma.split('<details class="terms-toggle"')[1].split("</details>")[0]
    assert "dodaj termine iz snimka" in zatvoreno and 'name="terms"' in zatvoreno
    assert "Ne snimaju se posebno." in zatvoreno, "krug 11: šta se dešava sa upisanim"
    assert "terms-toggle\" open" not in forma, "zatvoreno po podrazumevanom stanju"
    assert 'name="audio"' in forma and 'name="language"' in forma


def test_masinski_id_se_ne_prikazuje(web):
    client, db, cfg = web
    jid = make_job(cfg, db, job_id="20261007-120000-snimak")
    import re

    lista = client.get("/jobs/list").text
    # samo u adresama, nikad kao tekst
    vidljivo = re.sub(r'(href|action|id)="[^"]*"', "", lista)
    assert jid not in vidljivo and ">snimak</a>" in lista
    page = client.get(f"/jobs/{jid}").text
    assert f"<title>snimak — Subtitis</title>" in page
    assert '<h1 title="snimak.wav">snimak</h1>' in page


def test_pilula_greske_nosi_razlog(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    db.update_job(jid, status="error", error="ffmpeg nije pronađen\nTraceback …")
    lista = client.get("/jobs/list").text
    assert '<span class="pill err"' in lista and "ffmpeg nije pronađen" in lista
    assert ">error<" not in lista


def test_traka_izmisljenih_reci_kaze_sta_i_kad(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    jdir = cfg.job_dir(jid)
    cfg.write_model(jdir / "words.sr.json", WordsFile(
        job_id=jid, source_file="x", duration=2.0, language="sr", asr=AsrMeta(model="test"),
        words=[Word(i=0, s=0.0, e=0.5, w="Dobar", p=0.95),
               Word(i=1, s=32.26, e=32.26, w="OKC", p=0.99),
               Word(i=2, s=32.26, e=32.26, w="Divčibare", p=0.99)],
    ))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[Sentence(id=1, s=0.0, e=0.5, text="Dobar", words=(0, 0)),
                   Sentence(id=2, s=32.26, e=32.26, text="OKC Divčibare", words=(1, 2))],
    ))
    page = client.get(f"/jobs/{jid}").text
    bar = page.split('class="flagbar err"')[1].split("</div>")[0]
    assert "1 rečenica bez zvuka iza sebe (2 reči)" in bar
    assert "rečenica 2 u 00:32,26: 2 reči</li>" in bar  # sve uhvaćene — bez „od toga"
    assert "<code>OKC Divčibare</code> u 00:32,26" in bar


def test_ponovi_prepoznavanje_sa_filterom_i_terminima(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    cfg.write_json(cfg.job_glossary_path(jid), {"sr->en": {"Radionica otvorenih vrata": "Open Door"}})
    r = client.post(f"/jobs/{jid}/retranscribe",
                    data={"vad_preset": "tih", "terms": "Valjevo\nMile\n"},
                    follow_redirects=False)
    assert r.status_code == 303
    params = json.loads(db.pending_task(jid)["params"])
    assert params["vad_preset"] == "tih"
    gloss = cfg.read_json(cfg.job_glossary_path(jid))
    assert gloss["sr"] == {"Valjevo": "Valjevo", "Mile": "Mile"}
    assert gloss["sr->en"] == {"Radionica otvorenih vrata": "Open Door"}, "ostatak glosara ostaje"
    db.finish_task(db.pending_task(jid)["id"], "done")
    assert client.post(f"/jobs/{jid}/retranscribe", data={"vad_preset": "x"}).status_code == 400


def test_ponovno_prepoznavanje_ne_gazi_ispravke(web):
    client, db, cfg = web
    jid = make_job(cfg, db, edited=True)
    page = client.get(f"/jobs/{jid}").text
    assert 'id="retranscribe"' not in page, "ručno menjan prepis: ne nudi se"
    assert "Ponovno prepoznavanje bi ih\n        obrisalo" in _info(page, "Editor"), "u ikonici uz EDITOR, ne kao traka"
    assert client.post(f"/jobs/{jid}/retranscribe", data={"vad_preset": "normalno"}).status_code == 409


def test_glosar_novog_projekta_se_bira_u_podesavanjima(web):
    client, db, cfg = web
    client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "en",
        "llm_model": "m", "source_wav_ttl_days": "7", "new_project_glossary": "none",
    }, follow_redirects=False)
    assert cfg.new_project_glossary() == "none"
    import re

    assert re.search(r'value="none"\s+checked', client.get("/settings").text)
    r = client.post("/jobs", files={"audio": ("t.wav", b"RIFF")}, data={"language": "sr"},
                    follow_redirects=False)
    jid = r.headers["location"].rsplit("/", 1)[-1]
    assert cfg.read_json(cfg.job_glossary_path(jid)) == {}, "bez glosara, po podešavanju"


def test_preuzimanje_ima_jezik_pismo_i_broj_cue_ova(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Cyrl.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nДа.\n\n2\n00:00:01,000 --> 00:00:02,000\nНе.\n",
        encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    row = page.split('class="dlrow"')[1].split("</a>")[0]
    assert '<span class="lang">sr · ćirilica</span>' in row and "2 cue-ova" in row


def test_zivi_red_bez_primera_iz_titla(web):
    """Krug 10: primer se sekao na pola rečenice — živi red završava na slovima."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert 'id="edwhat"' in page
    assert "u ovom titlu stoji" not in page and "EXAMPLES" not in page


@pytest.mark.parametrize("url", ["/", "/settings", "/kako"])
def test_potpis_u_podnozju_svake_strane(web, url):
    """Krug 8: verzija se čita sa jednog mesta; krug 25: uz krug i commit."""
    from subtitis import ROUND, __version__

    client, db, cfg = web
    page = client.get(url).text
    sign = page.split('<footer class="sign">')[1].split("</footer>")[0]
    assert f"Mirko + Claude · Subtitis {__version__} · krug {ROUND}" in sign
    assert 'href="/kako"' in sign


def test_verzija_je_na_jednom_mestu():
    import tomllib

    p = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(p.read_text(encoding="utf-8"))["project"]
    assert "version" not in project and "version" in project["dynamic"]


def test_termini_sa_pocetne_ulaze_u_glosar_projekta(web):
    client, db, cfg = web
    r = client.post("/jobs", files={"audio": ("t.wav", b"RIFF")},
                    data={"language": "sr", "terms": "Valjevo\nMile"}, follow_redirects=False)
    jid = r.headers["location"].rsplit("/", 1)[-1]
    assert cfg.recording_terms(cfg.read_json(cfg.job_glossary_path(jid)), "sr") == ["Valjevo", "Mile"]


def test_nista_se_ne_ucitava_van_racunara(web):
    """Fontovi i htmx su lokalni: alat obećava da ništa ne napušta računar."""
    import re

    client, db, cfg = web
    for url in ("/", "/settings", "/kako"):
        page = client.get(url).text
        spolja = re.findall(r'(?:src|href)="(https?://[^"]+)"', page)
        assert spolja == [], spolja
    css = (Path(__file__).resolve().parents[1] / "subtitis" / "web" / "static" / "fonts.css").read_text(encoding="utf-8")
    assert "http" not in css.replace("/static/", "")
    for f in re.findall(r"/static/(fonts/[^)]+)", css):
        assert (Path(__file__).resolve().parents[1] / "subtitis" / "web" / "static" / f).exists(), f


# ==========================================================================
# Krug 9: ispod jedne trake najviše jedan red; ostalo iza „zašto ovo"
# ==========================================================================


def _deo(page, start, end):
    return page.split(start)[1].split(end)[0]


def _info(html, za):
    """Sadržaj prozorčića ikonice za dato polje."""
    return html.split(f'aria-label="Objašnjenje: {za}"')[1].split("</template>")[0]


def test_ispod_trake_editora_samo_zivi_red(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    traka = _deo(page, 'id="build-form"', "</form>")
    for polje, prvi in (("Profil", "Određuje koliko znakova"), ("Slova", "Menja samo izgled"),
                        ("Interpunkcija", "Izostavljena interpunkcija")):
        # ikonica odmah iza oznake, u istom redu; tekst je u prozorčiću
        assert f'<span class="field-label">{polje}<span class="why-ic"' in traka.replace("\n", "").replace("      ", ""), polje
        assert prvi in _info(traka, polje), polje
    assert "Kako se profili razlikuju" in _info(traka, "Profil")
    ispod = _deo(page, 'id="edwhat"', "<table")
    assert 'class="hint"' not in ispod and "<details" not in ispod


def test_prevod_i_preuzimanje_imaju_po_jedan_red(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    build_record(cfg, jid)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Latn.srt").write_text("1\r\n", encoding="utf-8")
    page = client.get(f"/jobs/{jid}").text
    prevod = _deo(page, "<h3>Prevod</h3>", "</section>")
    vidljivo = prevod.split("</form>")[1]
    assert vidljivo.count('class="note"') == 1 and 'class="hint"' not in vidljivo
    assert "Prvo ispravi prepis. Prevod neispravljenog teksta prevodi i greške." in vidljivo
    assert "rečenica sa kontekstom" in _info(prevod, "Prevod")
    preuzimanje = _deo(page, 'id="downloads"', "</section>")
    assert "Fajl se prepisuje" in _info(preuzimanje, "Za preuzimanje")
    assert preuzimanje.split("</h3>")[1].count('class="note"') == 1


def test_pocetna_bez_suvisnih_redova(web):
    client, db, cfg = web
    page = client.get("/").text
    srt = _deo(page, 'action="/jobs/srt"', "</form>")
    assert 'class="hint"' not in srt and "<details" not in srt
    snimak = _deo(page, 'action="/jobs"', "</form>")
    assert 'class="hint"' not in snimak
    # krug 12: polje za fajl ima oznaku kao svako drugo, ikonica uz nju
    assert '<span class="lbl field-label">Snimak<span class="why-ic"' in snimak
    assert "Audio ili video" in _info(snimak, "Snimak")
    assert "Za srpski obavezno" in _info(snimak, "Jezik u snimku")
    assert '<span class="lbl field-label">SRT fajl<span class="why-ic"' in srt


def test_filter_i_termini_tek_u_razvijenom_ponovi(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    izvestaj = _deo(page, "<h3>Izveštaj o obradi</h3>", "</section>")
    razvijeno = _deo(izvestaj, '<details class="guide retr" id="ponovi">', "</form>")
    assert 'name="vad_preset"' in razvijeno and 'name="terms"' in razvijeno
    assert izvestaj.count('name="vad_preset"') == 1
    # krug 12: meni nosi ime, ne opis; potvrda kroz naš prozor
    assert ">isključeno</option>" in razvijeno and "sporije</option>" not in razvijeno
    assert 'data-confirm-ok="Pokreni ponovo"' in razvijeno


def test_izmisljeno_crveno_i_imenovano_za_ekran(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    jdir = cfg.job_dir(jid)
    cfg.write_model(jdir / "words.sr.json", WordsFile(
        job_id=jid, source_file="x", duration=2.0, language="sr", asr=AsrMeta(model="test"),
        words=[Word(i=0, s=0.0, e=0.5, w="Dobar", p=0.2),
               Word(i=1, s=1.0, e=1.0, w="OKC", p=0.0),
               Word(i=2, s=1.0, e=1.0, w=",", p=0.0),
               Word(i=3, s=1.0, e=1.0, w="Divčibare", p=0.0)],
    ))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[Sentence(id=1, s=0.0, e=1.0, text="Dobar OKC, Divčibare", words=(0, 3),
                            flags=["low_confidence"])],
    ))
    row = _deo(client.get(f"/jobs/{jid}").text, '<tr data-id="1"', "</tr>")
    assert '<span class="low"' in row and ">Dobar</span>" in row
    # zarez u nizu je crven, ne žut — inače niz izgleda nesigurno
    assert '<span class="halluc"' in row.split(">OKC</span>")[1].split("Divčibare")[0]
    assert ">nesigurne reči<" in row and "low_confidence<" not in row
    assert ">izmišljeno — bez zvuka<" in row and 'class="flag-note"' in row


def test_decimalni_zarez_svuda(web):
    """Krug 9: 87,4% u tabeli i 87.4% red ispod je isti broj u dva zapisa."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    job = cfg.load_job(jid)
    job.history += [
        phase("asr", 20.0, model="large-v3", words=442, vad=False, vad_discarded_share=0.123),
        phase("segment", 1.0, sentences=27, coverage_pct=87.4, low_confidence=10),
    ]
    cfg.save_job(job)
    page = client.get(f"/jobs/{jid}").text
    assert "87,4%" in page and "87.4" not in page
    assert "12,3%" in page and "12.3" not in page
    assert "00:00,00" in page, "vremena u editoru"


def test_izbor_u_traci_prelama_odmah(web):
    """Krug 13: bez dugmeta „Prelomi" — promena izbora prelama sinhrono i vraća
    samo prelom ispod rečenica i „Za preuzimanje"."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    forma = page.split('id="build-form"')[1].split("</form>")[0]
    assert 'hx-post="/jobs/' + jid + '/rebuild"' in page
    assert 'hx-trigger="change delay:300ms, rebuild"' in page
    assert "Prelomi</button>" not in page and "<button" not in forma
    r = client.post(f"/jobs/{jid}/rebuild", data={
        "language": "sr", "profile": "tv", "script": "Cyrl", "case": "upper", "punct": "drop"})
    assert r.status_code == 200
    assert '<div class="cue-prev" id="prev-1" aria-label="prelom u titlu" hx-swap-oob="true">' in r.text
    assert "ДОБАР ДАН" in r.text
    assert 'id="downloads" hx-swap-oob="true"' in r.text and "sr · ćirilica" in r.text
    status = r.text.split('id="build-status"')[1].split("</span>")[0]
    assert "Prelamanje nije uspelo" not in status, "uspeh nema poruku"
    assert "Titl ponovo prelomljen" not in r.text and "<html" not in r.text
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM tasks WHERE job_id=?", (jid,)).fetchone()["c"] == 0


def test_neuspelo_prelamanje_se_kaze_uz_traku(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    (cfg.job_dir(jid) / "sentences.sr.json").unlink()
    r = client.post(f"/jobs/{jid}/rebuild", data={"language": "sr", "profile": "tv"})
    status = r.text.split('id="build-status"')[1].split("</span>")[0]
    assert "Prelamanje nije uspelo: Nedostaje datoteka" in status


def test_prelamanje_ceka_dok_obrada_radi(web):
    """Dva pisca istog job.json bi jedan drugom pojela istoriju."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    db.enqueue(jid, "translate", {"to": "en"})
    r = client.post(f"/jobs/{jid}/rebuild", data={"language": "sr", "profile": "tv"})
    assert "Obrada je u toku" in r.text
    assert not (cfg.job_dir(jid) / "out").exists()


# ==========================================================================
# Krug 10: zaglavlje, jezici, prečice, pivot
# ==========================================================================


def test_zaglavlje_logo_vodi_nazad_zupcanik_bez_teksta(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    for url in ("/", f"/jobs/{jid}", "/settings", "/kako"):
        page = client.get(url).text
        nav = page.split('<header class="topbar">')[1].split("</header>")[0]
        assert 'href="/kako"' not in nav, "kako ovo radi je u podnožju"
        assert 'class="gear" title="podešavanja" aria-label="podešavanja"' in nav
        assert '<a class="brand" href="/" title="baza projekata">' in nav
        assert "← baza projekata" not in page
        assert 'href="/kako"' in page.split('<footer class="sign">')[1]


def test_jezici_bez_skracenice(web):
    import re

    client, db, cfg = web
    jid = make_job(cfg, db)
    for url in ("/", f"/jobs/{jid}", "/settings"):
        opcije = re.findall(r"<option[^>]*>\s*([^<]*?)\s*</option>", client.get(url).text)
        assert "engleski" in opcije
        assert not [o for o in opcije if re.match(r"^(sr|en|fr|de|auto) — ", o)], url


def test_precice_rade_ali_su_u_uputstvu(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    assert "Prečice na tastaturi" not in page
    assert 'ev.key.toLowerCase() === "s"' in page, "Ctrl+S i dalje radi"
    kako = client.get("/kako").text
    assert "Prečice u editoru" in kako and "<kbd>Alt</kbd>+<kbd>N</kbd>" in kako


def test_trake_se_zatvaraju_same(web):
    """Klik van, Esc, najviše jedna otvorena — zajedničko za sve strane."""
    client, db, cfg = web
    page = client.get("/").text
    skript = page.split('<footer class="sign">')[1]
    assert 'document.addEventListener("toggle"' in skript and '"Escape"' in skript


def test_pivot_se_bira_u_podesavanjima(web):
    client, db, cfg = web
    assert cfg.resolve_pivot("sr", "fr") is None
    client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "fr", "llm_model": "m",
        "source_wav_ttl_days": "7", "translation_pivot": "en",
    }, follow_redirects=False)
    assert cfg.translation_pivot() == "en"
    assert cfg.resolve_pivot("sr", "fr") == "en"
    assert cfg.resolve_pivot("sr", "en") is None, "pivot jednak cilju nije pivot"
    assert client.post("/settings", data={
        "default_profile": "tv", "default_target_language": "fr", "llm_model": "m",
        "source_wav_ttl_days": "7", "translation_pivot": "de",
    }).status_code == 400


# ==========================================================================
# Krug 11: zauzet fajl na Windowsu, lista koja miruje, brisanje sa potvrdom
# ==========================================================================


def test_zamena_zauzetog_fajla_se_ponavlja(tmp_path, monkeypatch):
    """`os.replace` na Windowsu pada dok je cilj otvoren — milisekunde."""
    from subtitis import config as cfg

    pauze: list[float] = []
    monkeypatch.setattr("time.sleep", pauze.append)
    stvarno = Path.replace
    pokusaji = {"n": 0}

    def zauzeto(self, target):
        pokusaji["n"] += 1
        if pokusaji["n"] <= 3:
            raise PermissionError(5, "Access is denied")
        return stvarno(self, target)

    monkeypatch.setattr(Path, "replace", zauzeto)
    cfg.write_json(tmp_path / "job.json", {"a": 1})
    assert cfg.read_json(tmp_path / "job.json") == {"a": 1}
    assert pauze == [0.02, 0.04, 0.08]

    pokusaji["n"] = -100  # zauzeto zauvek
    pauze.clear()
    with pytest.raises(PermissionError):
        cfg.write_json(tmp_path / "job.json", {"a": 2})
    assert pauze == [0.02, 0.04, 0.08, 0.16, 0.32], "pet pauza, pa greška"
    assert cfg.read_json(tmp_path / "job.json") == {"a": 1}, "stari sadržaj ostaje ceo"


def test_svaki_upis_preko_fajla_ide_kroz_zamenu(tmp_path, monkeypatch):
    """Rečenice, titlovi, podešavanja i glosar — isti kvar čeka na svakom."""
    from subtitis import config as cfg, srt
    from subtitis.models import Cue

    zamene: list[str] = []
    stvarna = cfg.replace_file
    monkeypatch.setattr(cfg, "replace_file", lambda t, p: (zamene.append(p.name), stvarna(t, p)))
    srt.write_srt(tmp_path / "a.srt", [Cue(s=0, e=1, lines=["Da."])])
    cfg.save_settings({"llm_model": "m"})
    cfg.save_glossary({"sr->en": {"Da": "Yes"}})
    assert cfg.GLOSSARY_DIR != cfg.ROOT_DIR / "glossary", "nikad pravi glosar"
    assert "a.srt" in zamene and cfg.CONFIG_FILE.name in zamene and cfg.glossary_path().name in zamene


def test_stranica_projekta_cita_job_json_jednom(web, monkeypatch):
    client, db, cfg = web
    jid = make_job(cfg, db)
    build_record(cfg, jid)
    citanja = {"n": 0}
    stvarno = cfg.load_job

    def broji(job_id):
        citanja["n"] += 1
        return stvarno(job_id)

    monkeypatch.setattr(cfg, "load_job", broji)
    client.get(f"/jobs/{jid}?posle=prelom")
    assert citanja["n"] <= 1


def test_lista_miruje_kad_nista_ne_radi(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get("/").text
    assert 'hx-get="/jobs/list"' not in page and 'data-active="false"' in page
    db.enqueue(jid, "build", {"language": "sr"})
    page = client.get("/").text
    assert 'data-active="true"' in page and f'id="pill-{jid}"' in page
    st = client.get("/api/jobs/status").json()
    assert st["active"] is True and st["jobs"][jid]["pending"] is True


def test_brisanje_stvarno_brise_i_kaze_koliko(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(f"/jobs/{jid}/delete", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/?obrisan=snimak")
    page = client.get(r.headers["location"]).text
    assert "Projekat „snimak“ obrisan — oslobođeno" in page
    assert db.get_job(jid) is None and not cfg.job_dir(jid).exists()


# ==========================================================================
# Krug 12: šest pravila, provereno kroz SVE šablone i iscrtane strane
# ==========================================================================

TEMPLATES = Path(__file__).resolve().parents[1] / "subtitis" / "web" / "templates"
#: <details> sme samo za SADRŽAJ, nikad za objašnjenje: polje za termine,
#: verzije, forma „Ponovi prepoznavanje", dnevnik.
DETAILS_SADRZAJ = {"termini", "versions-box", "ponovi", "dnevnik"}


def _sve_strane(client, cfg, db):
    jid = make_job(cfg, db)
    build_record(cfg, jid)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "snimak.sr-Latn.srt").write_text("1\r\n", encoding="utf-8")
    return {u: client.get(u).text for u in ("/", "/jobs/list", "/settings", "/kako", f"/jobs/{jid}", "/jobs/nema")}


def test_pravilo1_details_samo_za_sadrzaj_u_svim_sablonima():
    import re

    for t in sorted(TEMPLATES.glob("*.html")):
        for tag in re.findall(r"<details\b[^>]*>", t.read_text(encoding="utf-8")):
            m = re.search(r'id="([^"]+)"', tag)
            assert m and m.group(1) in DETAILS_SADRZAJ, f"{t.name}: {tag}"


def test_pravilo1_i_2_na_iscrtanim_stranama(web):
    import re

    client, db, cfg = web
    for url, page in _sve_strane(client, cfg, db).items():
        for tag in re.findall(r"<details\b[^>]*>", page):
            assert re.search(r'id="(termini|versions-box|ponovi|dnevnik)"', tag), (url, tag)
        assert 'class="hint"' not in page.replace('class="hint err"', ""), url
        # svaka ikonica nosi ime polja i nikad ne stoji sama u svom redu
        for label in re.findall(r'class="why-ic"[^>]*aria-label="([^"]*)"', page, re.S):
            assert label.startswith("Objašnjenje: ") and len(label) > 14, (url, label)
        assert not re.search(r'<(div|p)[^>]*>\s*<span class="why-ic"', page), url


def test_pravilo2_ikonica_14px_tanka_6px():
    css = (TEMPLATES.parent / "static" / "app.css").read_text(encoding="utf-8")
    blok = css.split("\n.why-ic {")[1].split("}")[0]
    assert "width: 14px" in blok and "height: 14px" in blok and "margin-left: 6px" in blok
    assert "border: 1px solid" in blok


def test_pravilo4_meni_profila_nosi_ime(web):
    import re

    client, db, cfg = web
    page = client.get(f"/jobs/{make_job(cfg, db)}").text
    blok = page.split('name="profile"')[1].split("</select>")[0]
    assert re.findall(r"<option[^>]*>([^<]*)</option>", blok) == ["digital", "tv", "web"]


def test_pravilo5_pogled_se_cuva_kroz_osvezavanje(web):
    client, db, cfg = web
    page = client.get("/").text
    assert 'addEventListener("pagehide"' in page and "scrollRestoration" in page


def test_pravilo6_nigde_pregledacev_confirm():
    import re

    for t in sorted(TEMPLATES.glob("*.html")):
        txt = t.read_text(encoding="utf-8")
        poziva = [m for m in re.findall(r"(?<![\w.])confirm\(", txt)]
        assert not poziva, t.name


# --- prelom ispod rečenice ---------------------------------------------------


def test_prelom_ispod_recenice(web):
    """Cue-ovi gotovog titla iz rečenice, tačno kako su u fajlu; editor i dalje uređuje rečenicu."""
    from subtitis import pipeline

    client, db, cfg = web
    jid = make_job(cfg, db)
    page = client.get(f"/jobs/{jid}").text
    prazan = page.split('id="prev-1"')[1].split("</div>")[0]
    assert "cp " not in prazan, "dok titla nema, stoji samo polje"
    pipeline.run_build(jid, profile="tv", script="Cyrl", case="upper")
    page = client.get(f"/jobs/{jid}").text
    row = page.split('<tr data-id="1"')[1].split("</tr>")[0]
    prev = row.split('class="cue-prev"')[1].split("</div>\n")[0]
    assert '<span class="n">1</span>' in prev and "ДОБАР ДАН." in prev
    assert row.index('class="text"') < row.index('class="cue-prev"'), "ispod polja"
    assert 'contenteditable' not in prev, "samo prikaz"
    assert 'id="toggle-prev" checked' in page and "pokaži prelom" in page


def test_prelom_obelezava_prekratak_cue(web):
    from subtitis import pipeline

    client, db, cfg = web
    jid = make_job(cfg, db)
    pipeline.run_build(jid, profile="tv")
    data = cfg.read_json(pipeline.preview_path(jid, "sr"))
    data["cues"][0]["mark"] = "short"
    cfg.write_json(pipeline.preview_path(jid, "sr"), data)
    page = client.get(f"/jobs/{jid}").text
    assert '<div class="cp short" title="kraći od najkraćeg trajanja">' in page


def test_pokusaj_ponovo_pusta_pali_korak_i_preskocene(web):
    """Pao je korak, ne projekat: zvuk i prepis su u redu (Izjava — direktorka)."""
    import json

    client, db, cfg = web
    jid = make_job(cfg, db)
    t1 = db.enqueue(jid, "build", {"language": "sr", "profile": "tv"})
    db.enqueue(jid, "translate", {"to": "en"})
    db.finish_task(t1, "error", "PermissionError: [WinError 5]")
    db.skip_queued_tasks(jid, "prethodni korak je pao")
    db.update_job(jid, status="error", error="PermissionError: [WinError 5]")

    page = client.get(f"/jobs/{jid}").text
    glava = page.split('class="pjhead"')[1].split("</div>")[0]
    assert 'class="pill err"' in glava and "Pokušaj ponovo" in glava
    assert 'title="ponovo: prelamanje, prevod"' in glava

    r = client.post(f"/jobs/{jid}/retry", follow_redirects=False)
    assert r.status_code == 303
    row = db.get_job(jid)
    assert (row["status"], row["error"]) == ("transcribed", None), "prepis postoji"
    with db.connect() as conn:
        red = [(t["kind"], json.loads(t["params"])) for t in conn.execute(
            "SELECT * FROM tasks WHERE job_id=? AND state='queued' ORDER BY id", (jid,))]
    assert red == [("build", {"language": "sr", "profile": "tv"}), ("translate", {"to": "en"})]
    assert "Pokušaj ponovo" not in client.get(f"/jobs/{jid}").text, "više nije pao"
    assert client.post(f"/jobs/{jid}/retry").status_code == 409


def test_bez_palog_koraka_nema_dugmeta(web):
    client, db, cfg = web
    jid = make_job(cfg, db)
    assert "Pokušaj ponovo" not in client.get(f"/jobs/{jid}").text
    assert client.post(f"/jobs/{jid}/retry").status_code == 400


# ==========================================================================
# Krug 13: manje klikova, manje čitanja
# ==========================================================================


def test_stanje_snimanja_je_disketa_uz_dugme(web):
    client, db, cfg = web
    jid = make_job(cfg, db, edited=True)
    page = client.get(f"/jobs/{jid}").text
    glava = page.split('class="edhead"')[1].split("</div>")[0]
    assert 'id="save-icon" class="disk clean" title="sve izmene su sačuvane"' in glava
    assert glava.index('id="save-icon"') < glava.index('id="save"'), "disketa uz dugme"
    assert 'id="save-state"' in glava, "poruka nema izmena stoji uz dugme"
    # „Tvoje ispravke su sačuvane" samo u ikonici uz EDITOR — ne traka, ne dva puta
    assert page.count("Tvoje ispravke su sačuvane") == 1
    assert 'class="badge edited"' not in page
    assert "Tvoje ispravke su sačuvane" in _info(page, "Editor")


def test_zivi_red_samo_oblik_reda(web):
    client, db, cfg = web
    page = client.get(f"/jobs/{make_job(cfg, db)}").text
    karte = json.loads(page.split("const CARDS = ")[1].split(";\n")[0])
    web_karta = next(k for k in karte if k["name"] == "web")
    assert web_karta["min_chars"] == 14 and web_karta["max_lines"] == 1
    js = page.split("function renderWhat()")[1].split("\n}\n")[0]
    assert "reda po najviše" in js and "Jedan red od" in js
    assert "brzina čitanja" not in js and "Latinica" not in js and "SVA SLOVA" not in js
    # zakucan odmah ispod trake
    assert page.split("</form>")[0].count('id="build-form"') == 1
    posle_trake = page.split('id="build-form"')[1].split("</form>", 1)[1].lstrip()
    assert posle_trake.startswith('<p class="edwhat" id="edwhat">') or \
        posle_trake.split("\n", 1)[0].startswith('<p class="edwhat"')


def test_ikonica_centrirana_po_visini():
    """Krug 13b: omotač centrira flex-om; vertical-align ikonicu spušta ispod verzala."""
    css = (TEMPLATES.parent / "static" / "app.css").read_text(encoding="utf-8")
    omot = css.split("\n.field-label {")[1].split("}")[0]
    assert "display: inline-flex" in omot and "align-items: center" in omot and "gap: 6px" in omot
    assert "vertical-align: middle" not in css.split("\n.why-ic {")[1].split("}")[0]


def _roditelji_ikonica(html):
    """Klasa neposrednog roditelja svake ikonice."""
    from html.parser import HTMLParser

    VOID = {"input", "br", "img", "meta", "link", "path", "circle"}

    class P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stek, self.out = [], []

        def handle_starttag(self, tag, attrs):
            a = dict(attrs)
            if "why-ic" in (a.get("class") or "").split():
                self.out.append(self.stek[-1] if self.stek else None)
            if tag not in VOID:
                self.stek.append(a.get("class") or "")

        def handle_endtag(self, tag):
            if tag not in VOID and self.stek:
                self.stek.pop()

    p = P()
    p.feed(html)
    return p.out


def test_svaka_ikonica_u_omotacu_sa_oznakom(web):
    client, db, cfg = web
    for url, page in _sve_strane(client, cfg, db).items():
        page = re.sub(r"<template>.*?</template>", "", page, flags=re.S)
        roditelji = _roditelji_ikonica(page)
        for r in roditelji:
            assert "field-label" in (r or "").split(), (url, r)


@pytest.mark.parametrize("profil", ["tv", "web", "digital"])
def test_posle_prelamanja_za_preuzimanje_nosi_ime_profila(web, profil):
    """Prijava iz kruga 13: „Napravljeno profilom , pismo Cyrl" posle /rebuild."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    client.post(f"/jobs/{jid}/rebuild", data={"language": "sr", "profile": "tv", "script": "Cyrl"})
    r = client.post(f"/jobs/{jid}/rebuild", data={"language": "sr", "profile": profil, "script": "Cyrl"})
    deo = r.text.split('id="downloads"')[1].split("</section>")[0]
    assert f"Napravljeno profilom <strong>{profil}</strong>, pismo" in deo


def test_meniji_u_traci_imaju_stalno_ime(web):
    import re

    client, db, cfg = web
    page = client.get(f"/jobs/{make_job(cfg, db)}").text
    traka = page.split('id="build-form"')[1].split("</form>")[0]
    imena = re.findall(r'<select name="(\w+)"[^>]*aria-label="([^"]+)"', traka)
    assert imena == [("profile", "Profil"), ("script", "Pismo"), ("case", "Slova"), ("punct", "Interpunkcija")]


def test_izlazak_iz_izmenjenog_polja_snima_i_prelama(web):
    """Krug 14: tekst je bio jedino mesto koje je čekalo dugme."""
    client, db, cfg = web
    page = client.get(f"/jobs/{make_job(cfg, db)}").text
    js = page.split('cell.addEventListener("blur"')[1].split("function rebuildNow")[0]
    assert "cell.innerText !== cell.dataset.before" in js, "samo kad je tekst promenjen"
    assert "save({ auto: true, rebuild: true })" in js, "kopija po pravilu automatskog snimanja"
    assert "if (!auto || rebuild) rebuildNow();" in page
    assert "queuedSave" in page, "snimanje usred drugog se ne gubi"


def test_automatsko_snimanje_kopija_najvise_na_pet_minuta(web):
    """Izlazak iz polja ide kao automatsko snimanje — ne kopija za svaku reč."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    jdir = cfg.job_dir(jid)
    edit(client, jid, "Prva.", auto=True)
    edit(client, jid, "Druga.", auto=True)
    edit(client, jid, "Treća.", auto=True)
    assert len(list(jdir.glob("sentences.sr.json.*.bak"))) == 1


def test_ikonica_u_kartici_je_ista_datoteka(web):
    """Krug 15: bez oznake i rute pregledač je dobijao 404. Krug 25: `?v=` je
    otisak sadržaja ikonice, ne verzija."""
    from subtitis.web.app import static_version

    client, db, cfg = web
    r = client.get("/favicon.ico")
    ico = TEMPLATES.parent / "static" / "subtitis.ico"
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/")
    assert r.content == ico.read_bytes(), "bajt u bajt ista datoteka"
    assert "max-age" in r.headers["cache-control"]
    for url in ("/", "/settings", "/kako"):
        assert f'<link rel="icon" href="/favicon.ico?v={static_version("subtitis.ico")}"' in client.get(url).text


def test_vreme_po_tisini_bez_pauses_json_se_preskace_i_to_se_vidi(web, monkeypatch):
    """Krug 24: mehanizam čita samo POSTOJEĆI `pauses.json` — ne pravi ga i ne
    meri zvuk. Bez fajla se preskače i to piše ispod „Za preuzimanje"."""
    from subtitis import linebreak as lb
    from subtitis import pauses, pipeline

    client, db, cfg = web
    jid = make_job(cfg, db)
    monkeypatch.setattr(lb, "SNAP_MIN_BLANK", 0.4)
    pipeline.run_build(jid, profile="tv")
    page = " ".join(client.get(f"/jobs/{jid}").text.split())
    assert "Vreme po tišini: preskočeno — posao nema pauses.json." in page
    assert not pauses.path(jid).exists(), "ne pravi se sam"
    assert "vreme po tišini: preskočeno" in (cfg.job_dir(jid) / "log.txt").read_text(encoding="utf-8")

    # sa fajlom: tišina na početku je tu, ali bi natpis ostao kraći od 0,833 s
    cfg.write_json(pauses.path(jid), pauses.PausesFile(silences=[(0.0, 0.4)]).model_dump())
    pipeline.run_build(jid, profile="tv")
    page = " ".join(client.get(f"/jobs/{jid}").text.split())
    assert "Vreme po tišini: 0 početaka pomereno na kraj tišine, 0 krajeva skraćeno" in page


def test_bez_ukljucenog_mehanizma_nema_reda_o_vremenu(web, monkeypatch):
    from subtitis import linebreak as lb
    from subtitis import pipeline

    monkeypatch.setattr(lb, "SNAP_MIN_BLANK", None)
    client, db, cfg = web
    jid = make_job(cfg, db)
    pipeline.run_build(jid, profile="tv")
    assert "Vreme po tišini" not in client.get(f"/jobs/{jid}").text


def test_staticki_fajlovi_nose_otisak_sadrzaja(web, monkeypatch, tmp_path):
    """Krug 25: bez `?v=` brauzer drži stari CSS uz nov HTML i stranica se
    raspadne. Otisak je iz sadržaja — izmena fajla menja adresu."""
    import re

    from subtitis.web import app as app_mod

    client, db, cfg = web
    page = client.get("/").text
    for name in ("app.css", "fonts.css", "htmx.min.js"):
        m = re.search(rf'/static/{re.escape(name)}\?v=([0-9a-f]{{10}})"', page)
        assert m, name
        assert client.get(f"/static/{name}?v={m.group(1)}").status_code == 200
    assert re.search(r'/favicon\.ico\?v=[0-9a-f]{10}"', page)
    assert '"/static/app.css"' not in page, "nijedna adresa bez otiska"

    static = tmp_path / "static"
    static.mkdir()
    (static / "app.css").write_text("a{}", encoding="utf-8")
    monkeypatch.setattr(app_mod, "HERE", tmp_path)
    prvi = app_mod.static_url("app.css")
    (static / "app.css").write_text("a{color:red}", encoding="utf-8")
    assert app_mod.static_url("app.css") != prvi
    assert app_mod.static_url("nema.css") == "/static/nema.css"


def test_traka_je_jedan_red_spisak_je_u_prozorcicu(web):
    """Krug 27: traka nosi ikonicu, šta i koliko, radnju i ikonicu za detalje.
    Spisak, `pre` i pasusi su u <template> prozorčića — ne u samoj traci."""
    import re

    client, db, cfg = web
    jid = make_job(cfg, db)
    jdir = cfg.job_dir(jid)
    cfg.write_model(jdir / "words.sr.json", WordsFile(
        job_id=jid, source_file="x", duration=40.0, language="sr", asr=AsrMeta(model="test"),
        words=[Word(i=0, s=0.0, e=0.5, w="Dobar", p=0.95),
               Word(i=1, s=32.26, e=32.26, w="OKC", p=0.99),
               Word(i=2, s=32.26, e=32.26, w="Divčibare", p=0.99)],
    ))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[Sentence(id=1, s=0.0, e=0.5, text="Dobar", words=(0, 0)),
                   Sentence(id=2, s=32.26, e=32.26, text="OKC Divčibare", words=(1, 2))],
    ))
    db.update_job(jid, status="error", error="prvi red greške\ndrugi red\ntreći red")
    page = client.get(f"/jobs/{jid}").text
    bars = re.findall(r'<div class="flagbar [a-z]+">(.*?)\n  </div>', page, re.S)
    assert len(bars) >= 2
    for bar in bars:
        red = re.sub(r"<template>.*?</template>", "", bar, flags=re.S)
        assert not re.search(r"<(ul|li|p|pre)\b", red), red
        assert '<span class="ic">' in red and 'class="flagtxt"' in red and 'class="why-ic"' in red
    # greška u redu je samo prvi red; ceo tekst je u prozorčiću
    assert not any("drugi red" in re.sub(r"<template>.*?</template>", "", b, flags=re.S) for b in bars)
    assert any("drugi red" in b for b in bars)
