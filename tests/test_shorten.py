"""Skraćivanje prekoračenja: drugi prolaz kroz model, predlog — ne upis.

Ollama se ne dira. Lažni model za svaku rečenicu vraća ono što mu test zada.
"""

from __future__ import annotations

import re

import pytest

from subtitis import pipeline
from subtitis.models import SentencesFile
from subtitis.translate import ShortenItem, build_shorten_prompt, shorten_sentences


class FakeClient:
    """Po izvornom tekstu linije vraća zadati odgovor; ostalo vraća neizmenjeno."""

    model = "lazni"

    def __init__(self, answers: dict[str, str] | None = None, raw: str | None = None):
        from subtitis.translate import CallStats

        self.stats = CallStats()
        self.answers = answers or {}
        self.raw = raw
        self.prompts: list[str] = []

    def check(self):
        pass

    def unload(self):
        pass

    def chat(self, system, user):
        self.prompts.append(user)
        if self.raw is not None:
            return self.raw
        block = user.split("TITLOVI:\n", 1)[1]
        out = []
        for line in block.splitlines():
            n, text = line.split("|", 1)
            out.append(f"{n}|{self.answers.get(text, text)}")
        return "\n".join(out)


def test_prompt_nosi_original_i_zahtev_u_znakovima():
    it = ShortenItem(id=42, text="Nous sommes rentrés à la maison ensemble.", source="Vratili smo se kući.", need=12)
    p = build_shorten_prompt([(1, it)], "sr")
    assert "ORIGINAL (sr):\n1: Vratili smo se kući." in p
    # apsolutna granica, ne „skrati za N": relativni zahtev je vukao na sečenje
    assert "DUŽINE:\n1: najviše 29 znakova" in p
    assert "skrati" not in p.lower() and "sada 41" not in p
    assert "TITLOVI:\n1|Nous sommes rentrés" in p
    assert "42" not in p  # pravi id ne ide modelu — lokalna numeracija


def test_neispravan_odgovor_ne_daje_predlog():
    items = [ShortenItem(id=7, text="Dugačka rečenica.", source="x", need=3)]
    out, stats = shorten_sentences(items, "sr", "fr", client=FakeClient(raw="smeće"), log=lambda m: None)
    assert out == {} and stats.failed_ids == [7]


def test_neuravnotezeni_navodnici_se_odbijaju():
    items = [ShortenItem(id=1, text="Il dit «oui» ici.", source="x", need=3)]
    client = FakeClient({"Il dit «oui» ici.": "Il dit «oui"})
    out, stats = shorten_sentences(items, "sr", "fr", client=client, log=lambda m: None)
    assert out == {} and stats.failed_ids == [1]


# ==========================================================================
# run_shorten: provera prelamanjem, ništa se ne upisuje
# ==========================================================================

SRT = (
    "1\n00:00:01,000 --> 00:00:02,000\nDa.\n\n"
    "2\n00:00:03,000 --> 00:00:04,000\nNe.\n\n"
    "3\n00:00:05,000 --> 00:00:06,000\nMožda.\n\n"
    "4\n00:00:07,000 --> 00:00:10,000\nDobro.\n"
)
DUGO = {
    1: "Oui, oui, bien sûr, absolument, mon ami.",    # 40 zn. u 1 s
    2: "Non, non, pas du tout, jamais de la vie.",     # 40 zn. u 1 s
    3: "Peut-être que oui, peut-être que non, hein.",  # 43 zn. u 1 s
    4: "D'accord.",                                    # nije preko
}


@pytest.fixture()
def posao(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    p = tmp_path / "film.sr.srt"
    p.write_text(SRT, encoding="utf-8")
    pipeline.run_import_srt(p, "sr", job_id="skr")
    jdir = cfg.job_dir("skr")
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    cfg.write_model(jdir / "sentences.fr.json", sf.model_copy(update={
        "language": "fr", "derived_from": "sentences.sr.json",
        "sentences": [s.model_copy(update={"text": DUGO[s.id]}) for s in sf.sentences],
    }))
    return cfg, jdir


def run(monkeypatch, client, **kw):
    from subtitis import translate as tr

    monkeypatch.setattr(tr, "OllamaClient", lambda **k: client)
    return pipeline.run_shorten("skr", "fr", profile="tv", **kw)


def test_prihvata_samo_krace_sa_nizim_cps(posao, monkeypatch):
    cfg, jdir = posao
    before = (jdir / "sentences.fr.json").read_text(encoding="utf-8")
    client = FakeClient({
        DUGO[1]: "Oui, bien sûr.",                                   # kraće -> prihvaćeno
        DUGO[2]: "Non, non, pas du tout, jamais de la vie, jamais.",  # duže -> odbijeno
    })
    logs: list[str] = []
    r = run(monkeypatch, client, on_log=logs.append)

    by = {it["sentence"]: it for it in r.items}
    assert set(by) == {1, 2, 3}, "rečenica 4 nije preko profila i ne šalje se"
    assert by[1]["status"] == "accepted" and by[1]["cut"] == len(DUGO[1]) - len("Oui, bien sûr.")
    assert by[1]["cps_after"] < by[1]["cps_before"] and by[1]["cues_fixed"] == [1]
    assert (by[2]["status"], by[2]["reason"]) == ("rejected", "nije kraće")
    assert (by[3]["status"], by[3]["reason"]) == ("rejected", "nije kraće")  # vraćeno isto
    assert (r.attempted, r.accepted, r.cues_fixed, r.cues_worse_before) == (3, 1, 1, 3)
    assert r.rejected == {"nije kraće": 2}

    # ništa nije upisano u prevod; predlog je u shorten.fr.json
    assert (jdir / "sentences.fr.json").read_text(encoding="utf-8") == before
    saved = cfg.read_json(pipeline.shorten_path("skr", "fr"))
    assert saved["accepted"] == 1 and saved["items"][0]["old"] == DUGO[1]
    assert any("pokušano 3 rečenica, prihvaćeno 1" in m for m in logs)
    assert any("NIŠTA NIJE UPISANO" in m for m in logs)



def test_izgubljen_termin_iz_glosara_se_odbija(posao, monkeypatch):
    """Stvaran predlog: `Open Door Workshop` -> `Open Door`."""
    cfg, _ = posao
    # glosar POSLA (kopija pri otvaranju), ne podrazumevani
    cfg.write_json(cfg.job_glossary_path("skr"), {"sr->fr": {"Da": "Oui, oui, bien sûr"}})
    client = FakeClient({DUGO[1]: "Bien sûr, mon ami."})
    r = run(monkeypatch, client)
    by = {it["sentence"]: it for it in r.items}
    assert (by[1]["status"], by[1]["reason"]) == ("rejected", "izgubljen termin iz glosara")
    assert by[1]["lost_terms"] == ["Oui, oui, bien sûr"]


# ==========================================================================
# Editor: pregled pre upisa, upis samo izabranog
# ==========================================================================


@pytest.fixture()
def web(posao, monkeypatch):
    from fastapi.testclient import TestClient

    from subtitis.web import app as app_mod
    from subtitis.web import db

    cfg, jdir = posao
    db.init_db()
    db.create_job("skr", "film.sr.srt", str(jdir / "upload"), "sr")
    db.update_job("skr", status="done")
    with TestClient(app_mod.app) as client:
        yield client, db, cfg, jdir


def predlog(monkeypatch):
    return run(monkeypatch, FakeClient({
        DUGO[1]: "Oui, bien sûr.",
        DUGO[2]: "Non, jamais.",
        DUGO[3]: "Peut-être que oui, peut-être que non, hein, oui.",  # duže
    }))


def test_dugme_stavlja_skracivanje_u_red(web):
    client, db, _, _ = web
    r = client.post("/jobs/skr/shorten", data={"language": "fr"}, follow_redirects=False)
    assert r.status_code == 303
    assert db.pending_task("skr")["kind"] == "shorten"
    # izvorni jezik nije prevod
    db.finish_task(db.pending_task("skr")["id"], "done")
    assert client.post("/jobs/skr/shorten", data={"language": "sr"}).status_code == 400


def test_pregled_pokazuje_staro_i_novo(web, monkeypatch):
    client, _, _, _ = web
    predlog(monkeypatch)
    page = client.get("/jobs/skr?lang=fr").text
    assert "Predlozi skraćivanja — fr" in page
    assert "Ništa nije upisano" in page or "ništa nije upisano" in page.lower()
    assert "<del>" in page
    # „Oui, bien sûr." seče 26 znakova za traženih 20 (opseg do 30) — izabran
    assert re.search(r'value="1"\s+checked', page)
    assert "nije kraće" in page  # odbijeni se vide, bez kvačice
    assert 'value="3"' not in page


def test_upisuje_samo_izabrano(web, monkeypatch):
    client, db, cfg, jdir = web
    predlog(monkeypatch)
    r = client.post("/jobs/skr/shorten/apply", json={"lang": "fr", "ids": [1]})
    assert r.status_code == 200
    assert r.json() == {"applied": [1], "stale": [], "declined": [2]}
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.fr.json"))
    by = {s.id: s for s in sf.sentences}
    assert by[1].text == "Oui, bien sûr." and "shortened" in by[1].flags
    assert by[2].text == DUGO[2] and "shortened" not in by[2].flags  # odbijeno ručno
    assert sf.edited_by_human
    assert list(jdir.glob("sentences.fr.json.*.bak")), "pre upisa se pravi verzija"
    assert not pipeline.shorten_path("skr", "fr").exists()
    assert cfg.read_json(jdir / "shorten.fr.applied.json")["declined"] == [2]
    assert db.pending_task("skr")["kind"] == "build"


def test_menjan_tekst_se_ne_pregazi(web, monkeypatch):
    client, _, cfg, jdir = web
    predlog(monkeypatch)
    p = jdir / "sentences.fr.json"
    sf = SentencesFile.model_validate(cfg.read_json(p))
    sf.sentences[0].text = "Ručno ispravljeno posle predloga."
    cfg.write_model(p, sf)
    r = client.post("/jobs/skr/shorten/apply", json={"lang": "fr", "ids": [1, 2]})
    assert r.json()["stale"] == [1] and r.json()["applied"] == [2]
    sf = SentencesFile.model_validate(cfg.read_json(p))
    assert sf.sentences[0].text == "Ručno ispravljeno posle predloga."


def test_odbaci_ceo_predlog(web, monkeypatch):
    client, _, cfg, jdir = web
    before = (jdir / "sentences.fr.json").read_text(encoding="utf-8")
    predlog(monkeypatch)
    assert client.post("/jobs/skr/shorten/discard", json={"lang": "fr"}).status_code == 200
    assert not pipeline.shorten_path("skr", "fr").exists()
    assert (jdir / "sentences.fr.json").read_text(encoding="utf-8") == before


def test_prekomerno_skracen_predlog_nije_izabran(web, monkeypatch):
    """Preko traženog opsega = gde se značenje najčešće gubi: bez kvačice."""
    client, _, _, _ = web
    r = run(monkeypatch, FakeClient({DUGO[1]: "Oui."}))  # 36 zn. za traženih 20
    by = {it["sentence"]: it for it in r.items}
    assert by[1]["status"] == "accepted" and by[1]["overcut"] is True
    page = client.get("/jobs/skr?lang=fr").text
    assert re.search(r'value="1"\s+>', page), "nije izabran"
    assert "više od traženog" in page


def test_neispravan_odgovor_u_celom_toku_se_odbija(posao, monkeypatch):
    """Krug 16: grana u `run_shorten`, ne samo u `shorten_sentences`."""
    r = run(monkeypatch, FakeClient(raw="smeće"))
    assert {it["reason"] for it in r.items} == {"model nije vratio ispravan odgovor"}
    assert r.accepted == 0
