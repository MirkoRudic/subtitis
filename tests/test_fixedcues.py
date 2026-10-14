"""Prevod postojećeg SRT-a: cue-ovi i vremena ostaju, menja se samo tekst.

Jedina tvrda garancija režima je `test_tajming_izlaza_je_identican_ulazu`:
isti broj cue-ova i ista vremena, poređenjem ulaznog i izlaznog fajla. Ostalo
su pravila sastavljanja rečenica i raspoređivanja prevoda nazad u cue-ove.

Ollama se ne dira — klijent se zamenjuje lažnim.
"""

from __future__ import annotations

import pytest

from subtitis import fixedcues, linebreak, pipeline, srt
from subtitis.models import CuePart, CuesFile, Profile, Sentence, SentencesFile, SourceCue, Word
from subtitis.segment import tokenize_text
from subtitis.translate import FIT_LENGTH_RULE, build_system_prompt

# Namerno ružan ulaz: BOM, mešani prelomi reda, cue od tri reda, rečenica
# preko tri cue-a, cue sa krajem jedne i početkom druge rečenice, dijalog,
# kurziv, pozicija, nedostaje prazan red na kraju.
ULAZ = (
    "﻿1\r\n"
    "00:00:01,000 --> 00:00:03,200\r\n"
    "Juče sam otišao u grad\r\n"
    "sa svojim bratom\r\n"
    "\r\n"
    "2\r\n"
    "00:00:03,500 --> 00:00:06,000\r\n"
    "i kupili smo novi auto. A onda\r\n"
    "\r\n"
    "3\n"
    "00:00:06,100 --> 00:00:08,040\n"
    "<i>smo se vratili kući.</i>\n"
    "\n"
    "4\n"
    "00:00:08,500 --> 00:00:10,000\n"
    "- Da li je lep?\n"
    "- Jeste, veoma.\n"
    "\n"
    "5\n"
    "00:00:11,000 --> 00:00:14,999\n"
    "{\\an8}Ovo je dugačak cue\n"
    "koji ima tri reda\n"
    "i ne završava se\n"
    "\n"
    "6\n"
    "00:00:15,000 --> 00:00:16,500\n"
    "tačkom nego ovde.\n"
    "\n"
    "7\n"
    "00:01:02,345 --> 00:01:04,001\n"
    "Kraj."
)


@pytest.fixture()
def prof() -> Profile:
    return Profile(name="tv")


def cues_of(text: str = ULAZ) -> list[SourceCue]:
    return [fixedcues.split_prefix(c) for c in srt.parse(text)]


def words_of(text: str) -> list[Word]:
    return [Word(i=k, s=0.0, e=0.0, w=t, p=1.0) for k, t in enumerate(tokenize_text(text))]


# ==========================================================================
# Parser
# ==========================================================================


def test_parser_je_tolerantan():
    cues = cues_of()
    assert len(cues) == 7
    assert (cues[0].start_ms, cues[0].end_ms) == (1000, 3200)
    assert cues[0].lines == ["Juče sam otišao u grad", "sa svojim bratom"]
    # cue od tri reda, pozicija izdvojena
    assert cues[4].lines == ["Ovo je dugačak cue", "koji ima tri reda", "i ne završava se"]
    assert cues[4].prefix == "{\\an8}"
    # kurziv ostaje u redu
    assert cues[2].lines == ["<i>smo se vratili kući.</i>"]
    # poslednji cue bez praznog reda na kraju
    assert (cues[6].start_ms, cues[6].end_ms, cues[6].lines) == (62345, 64001, ["Kraj."])


def test_parser_bez_praznog_reda_izmedju_i_bez_rednog_broja():
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nPrvi.\n"
        "2\n00:00:02,500 --> 00:00:03,000\nDrugi.\n\n\n\n"
        "00:00:04,000 --> 00:00:05,000 X1:10 X2:20\nTreći.\n"
    )
    cues = srt.parse(text)
    assert [c.lines for c in cues] == [["Prvi."], ["Drugi."], ["Treći."]]
    assert srt.timing(cues) == [(1000, 2000), (2500, 3000), (4000, 5000)]


def test_parser_razlomak_sekunde_i_tacka():
    cues = srt.parse("1\n00:00:01.5 --> 00:00:02,25\nA.\n")
    assert srt.timing(cues) == [(1500, 2250)]


def test_nije_srt():
    from subtitis.config import SubtitisError

    with pytest.raises(SubtitisError, match="nije SRT"):
        srt.parse("samo tekst\nbez vremena\n")


def test_kodiranje():
    assert srt.decode("Juče".encode("utf-8-sig")) == ("Juče", "utf-8-sig")
    assert srt.decode("Juče".encode("utf-8")) == ("Juče", "utf-8")
    assert srt.decode("Juče smo išli".encode("cp1250")) == ("Juče smo išli", "cp1250")
    # cp1251 se dekodira i kao cp1250 (u đubre) — bira se po slovima
    assert srt.decode("Јуче смо ишли".encode("cp1251")) == ("Јуче смо ишли", "cp1251")


@pytest.mark.parametrize("name,stem", [
    ("film.srt", "film"),
    ("film.sr.srt", "film"),
    ("film.sr-Latn.srt", "film"),
    ("moj.film.2024.srt", "moj.film.2024"),
])
def test_ime_izlaza(name, stem):
    assert srt.srt_stem(name) == stem
    assert srt.srt_name(srt.srt_stem(name), "fr") == f"{stem}.fr.srt"


# ==========================================================================
# Sastavljanje rečenica
# ==========================================================================


def test_recenica_preko_vise_cue_ova_pamti_udele():
    sents, _ = fixedcues.assemble(cues_of())
    prva = sents[0]
    assert prva.text == "Juče sam otišao u grad sa svojim bratom i kupili smo novi auto."
    assert [p.cue for p in prva.cue_parts] == [0, 1]
    assert [p.chars for p in prva.cue_parts] == [39, 23]
    assert sum(p.share for p in prva.cue_parts) == pytest.approx(1.0, abs=1e-3)
    assert prva.s == 1.0 and prva.e == 6.0


def test_cue_sa_krajem_jedne_i_pocetkom_druge():
    sents, stats = fixedcues.assemble(cues_of())
    druga = sents[1]
    assert druga.text == "A onda smo se vratili kući."
    assert [(p.cue, p.chars, p.italic) for p in druga.cue_parts] == [
        (1, 6, ""), (2, 20, "all"),
    ]
    assert stats.partial_italic == []


def test_dijalog_je_dve_recenice_sa_prelomom_reda():
    sents, _ = fixedcues.assemble(cues_of())
    d1, d2 = sents[2], sents[3]
    assert (d1.text, d2.text) == ("- Da li je lep?", "- Jeste, veoma.")
    assert d1.cue_parts[0].line_break is True
    assert d2.cue_parts[0].line_break is False


def test_crta_na_pocetku_reda_otvara_novu_recenicu_i_bez_tacke():
    text = "1\n00:00:01,000 --> 00:00:02,000\n- Ideš li\n- Ne idem\n"
    sents, _ = fixedcues.assemble(srt.parse(text))
    assert [s.text for s in sents] == ["- Ideš li", "- Ne idem"]


def test_tri_tacke_i_malo_slovo_su_nastavak():
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nMislio sam da...\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n...ćemo stići.\n\n"
        "3\n00:00:03,000 --> 00:00:04,000\nOnda... Ništa.\n"
    )
    sents, _ = fixedcues.assemble(srt.parse(text))
    assert [s.text for s in sents] == [
        "Mislio sam da... ćemo stići.", "Onda...", "Ništa.",
    ]


def test_skracenica_ne_zavrsava_recenicu():
    text = "1\n00:00:01,000 --> 00:00:02,000\nRazgovarao sam sa dr.\n\n" \
           "2\n00:00:02,000 --> 00:00:03,000\nPetrovićem juče.\n"
    sents, _ = fixedcues.assemble(srt.parse(text))
    assert len(sents) == 1


def test_recenica_bez_kraja_se_sece_posle_granice():
    n = fixedcues.MAX_CUES_PER_SENTENCE + 3
    text = "".join(
        f"{k}\n00:00:{k:02d},000 --> 00:00:{k:02d},900\nreč broj {k}\n\n"
        for k in range(1, n + 1)
    )
    sents, stats = fixedcues.assemble(srt.parse(text))
    assert len(sents) == 2
    assert len(sents[0].cue_parts) == fixedcues.MAX_CUES_PER_SENTENCE
    assert stats.forced_breaks == [fixedcues.MAX_CUES_PER_SENTENCE]


def test_kurziv_na_delu_recenice_se_prijavljuje():
    text = "1\n00:00:01,000 --> 00:00:02,000\nGledali smo <i>Kazablanku</i> juče.\n"
    sents, stats = fixedcues.assemble(srt.parse(text))
    assert sents[0].cue_parts[0].italic == "partial"
    assert stats.partial_italic == [1]


def test_ostale_oznake_se_skidaju_i_broje():
    text = "1\n00:00:01,000 --> 00:00:02,000\n<b>Glasno</b> <font color=\"red\">crveno</font>.\n"
    sents, stats = fixedcues.assemble(srt.parse(text))
    assert sents[0].text == "Glasno crveno."
    assert stats.dropped_tags == {"b": 2, "font": 2}


# ==========================================================================
# Raspoređivanje — postojeće skorovanje, fiksan broj delova
# ==========================================================================


def test_deljenje_daje_tacno_zadat_broj_delova(prof):
    words = words_of("Hier je suis allé en ville avec mon frère et nous avons acheté une voiture.")
    for k in (1, 2, 3, 4):
        ranges = linebreak.split_by_shares(words, [1.0 / k] * k, prof, linebreak.rules_for("fr"))
        assert len(ranges) == k
        assert ranges[0][0] == 0 and ranges[-1][1] == len(words)
        assert all(a < b for a, b in ranges)
        assert all(r1[1] == r2[0] for r1, r2 in zip(ranges, ranges[1:]))


def test_deljenje_postuje_funkcijske_reci(prof):
    """Udeo pada tačno iza `dans`; iste kazne kao pri prelamanju ga pomeraju."""
    words = words_of("Nous avons marché longtemps dans la forêt sombre.")
    ranges = linebreak.split_by_shares(
        words, [0.62, 0.38], prof, linebreak.rules_for("fr")
    )
    prvi = " ".join(w.w for w in words[ranges[0][0]:ranges[0][1]])
    assert not prvi.endswith(("dans", "la"))


def test_deljenje_prati_udele(prof):
    words = words_of("jedan dva tri četiri pet šest sedam osam devet deset jedanaest")
    ranges = linebreak.split_by_shares(words, [0.2, 0.8], prof)
    assert ranges[0][1] <= 4


def test_manje_reci_nego_cue_ova_ostavlja_prazan_deo(prof):
    ranges = linebreak.split_by_shares(words_of("Da."), [0.5, 0.5], prof)
    assert len(ranges) == 2
    assert sum(b - a for a, b in ranges) == 2


# ==========================================================================
# Sklapanje nazad u cue-ove
# ==========================================================================


def source_file(cues) -> CuesFile:
    return CuesFile(job_id="x", source_file="x.srt", language="sr", cues=cues)


def test_neizmenjen_izvor_se_vraca_doslovno(prof):
    cues = cues_of()
    sents, _ = fixedcues.assemble(cues)
    built = fixedcues.build(source_file(cues), sents, prof, language="sr", exact=True)
    assert [c.lines for c in built.cues] == [c.lines for c in cues]
    assert built.unchanged == len(cues)


FR = {
    1: "Hier, je suis allé en ville avec mon frère et nous avons acheté une nouvelle voiture.",
    2: "Et puis nous sommes rentrés à la maison.",
    3: "- Est-elle belle ?",
    4: "- Oui, très.",
    5: "C'est un long sous-titre qui a trois lignes et ne se termine pas par un point mais ici.",
    6: "Fin.",
}


def translated(sents):
    return [s.model_copy(update={"text": FR[s.id]}) for s in sents]


def test_prevod_ide_u_iste_cue_ove(prof):
    cues = cues_of()
    sents, _ = fixedcues.assemble(cues)
    built = fixedcues.build(source_file(cues), translated(sents), prof, language="fr")
    assert len(built.cues) == len(cues)
    # sav tekst je tu, redom — ništa izgubljeno, ništa skraćeno
    flat = " ".join(fixedcues.strip_markup(l) for c in built.cues for l in c.lines)
    assert tokenize_text(flat) == tokenize_text(" ".join(FR[k] for k in sorted(FR)))
    # dijalog zadržava svoja dva reda
    assert built.cues[3].lines == ["- Est-elle belle?", "- Oui, très."]
    # kurziv je ceo drugi deo druge rečenice -> ceo cue 3
    assert built.cues[2].lines[0].startswith("<i>")
    assert built.cues[2].lines[-1].endswith("</i>")
    assert built.cue_sentences[1] == [1, 2]


def test_delimicni_kurziv_se_prijavljuje_na_cue_u(prof):
    text = "1\n00:00:01,000 --> 00:00:03,000\nGledali smo <i>Kazablanku</i> juče.\n"
    cues = srt.parse(text)
    sents, _ = fixedcues.assemble(cues)
    sents = [s.model_copy(update={"text": "Nous avons vu Casablanca hier."}) for s in sents]
    built = fixedcues.build(source_file(cues), sents, prof, language="fr")
    assert built.italic_lost == [1]
    assert fixedcues.ITALIC_LOST_FLAG in built.cues[0].flags


def test_cirilica_ne_dira_oznake():
    assert fixedcues.convert_markup("<i>Da</i>, {\\an8}jeste", "Cyrl") == "<i>Да</i>, {\\an8}јесте"


# ==========================================================================
# Prekoračenja: prijava, ne popravka
# ==========================================================================


def test_prekoracenja_se_prijavljuju_sa_rednim_brojevima(prof):
    cues = cues_of()
    sents, _ = fixedcues.assemble(cues)
    long = [
        s.model_copy(update={"text": FR[s.id] + " Et encore beaucoup de mots en plus ici."})
        for s in sents
    ]
    built = fixedcues.build(source_file(cues), long, prof, language="fr")
    found = fixedcues.overflows(built, prof)
    assert found, "francuski duži od srpskog mora da probije CPS"
    assert all(o.cue >= 1 for o in found)
    assert [o.severity for o in found] == sorted((o.severity for o in found), reverse=True)
    head, worst = fixedcues.overflow_lines(found, len(built.cues), prof)
    assert f"{len(found)} od {len(cues)}" in head
    assert "Ništa nije skraćeno" in head
    assert worst.count("#") == min(len(found), fixedcues.WORST_SHOWN)
    # ništa nije skraćeno: sve reči prevoda su u izlazu
    flat = " ".join(fixedcues.strip_markup(l) for c in built.cues for l in c.lines)
    assert tokenize_text(flat) == tokenize_text(" ".join(s.text for s in long))


# ==========================================================================
# Prompt
# ==========================================================================


def test_pravilo_o_duzini_samo_u_srt_rezimu():
    obican = build_system_prompt("sr", "fr", {"Beograd": "Belgrade"})
    srt_rez = build_system_prompt("sr", "fr", {"Beograd": "Belgrade"}, fit_length=True)
    assert FIT_LENGTH_RULE not in obican
    assert f"13. {FIT_LENGTH_RULE}" in srt_rez
    assert "Beograd -> Belgrade" in srt_rez  # glosar radi kao i inače
    sa_varijantom = build_system_prompt("sr", "en", {}, "en-GB", fit_length=True)
    assert f"14. {FIT_LENGTH_RULE}" in sa_varijantom


# ==========================================================================
# Ceo tok kroz pajplajn — i tvrda garancija tajminga
# ==========================================================================


@pytest.fixture()
def jobs(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    return cfg, tmp_path


class FakeClient:
    """„Prevodi" na francuski tako što tekst produži — kao pravi francuski."""

    model = "lazni"

    def __init__(self) -> None:
        from subtitis.translate import CallStats

        self.stats = CallStats()
        self.systems: list[str] = []

    def check(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def chat(self, system: str, user: str) -> str:
        self.systems.append(system)
        block = user.split("ZA_PREVOD:\n", 1)[1].split("\n\nKONTEKST")[0]
        out = []
        for line in block.splitlines():
            if not line.strip():
                continue
            i, text = line.split("|", 1)
            out.append(f"{i}|{text} et voilà, c'est plus long")
        return "\n".join(out)


def uvezi(cfg, tmp_path, text=ULAZ, name="film.sr.srt", language="sr", encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return pipeline.run_import_srt(p, language, job_id="srt-posao")


def test_tajming_izlaza_je_identican_ulazu(jobs, monkeypatch):
    """TVRDA GARANCIJA: broj cue-ova i sva vremena izlaza == ulaz.

    Poređenje je nad dva fajla, parsiranjem oba — ne nad internim strukturama.
    """
    cfg, tmp_path = jobs
    from subtitis import translate as tr

    r = uvezi(cfg, tmp_path)
    assert r.cues == 7
    client = FakeClient()
    monkeypatch.setattr(tr, "OllamaClient", lambda **kw: client)
    pipeline.run_translate("srt-posao", "fr")
    b = pipeline.run_build("srt-posao", language="fr", profile="tv")

    assert b.path.name == "film.fr.srt"
    ulaz = srt.parse((tmp_path / "film.sr.srt").read_text(encoding="utf-8-sig"))
    izlaz = srt.parse(b.path.read_text(encoding="utf-8"))
    assert len(izlaz) == len(ulaz)
    assert srt.timing(izlaz) == srt.timing(ulaz)
    # i doslovno isti vremenski redovi u fajlu
    vremena = lambda t: [l for l in t.replace("\r\n", "\n").split("\n") if "-->" in l]
    assert vremena(b.path.read_text(encoding="utf-8")) == vremena(ULAZ)

    # pravilo o dužini je stiglo do modela
    assert all(FIT_LENGTH_RULE in s for s in client.systems)
    # pozicija vraćena ispred cue-a
    assert izlaz[4].lines[0].startswith("{\\an8}")
    # prekoračenja zapisana za editor
    job = cfg.load_job("srt-posao")
    build = [h for h in job.history if h.phase == "build"][-1]
    assert build.detail["overflow_count"] == len(b.overflows)
    assert build.detail["overflow"] == [o.as_dict() for o in b.overflows]


def test_izvorni_jezik_bez_izmena_daje_isti_titl(jobs):
    cfg, tmp_path = jobs
    uvezi(cfg, tmp_path)
    b = pipeline.run_build("srt-posao", language="sr", profile="tv")
    ulaz = srt.parse(ULAZ)
    izlaz = srt.parse(b.path.read_text(encoding="utf-8"))
    assert [c.lines for c in izlaz] == [c.lines for c in ulaz]
    assert srt.timing(izlaz) == srt.timing(ulaz)


def test_cirilicni_izvor_se_drzi_u_latinici_bez_ekavizacije(jobs):
    """Normalizacija pisma da; ekavizacija ne — ulaz je ručno doteran."""
    cfg, tmp_path = jobs
    text = (
        "1\n00:00:01,000 --> 00:00:03,000\n<i>Лијепо смо се\n"
        "провели у Београду.</i>\n"
    )
    uvezi(cfg, tmp_path, text=text, encoding="cp1251")  # nije UTF-8, ni cp1250
    sf = SentencesFile.model_validate(
        cfg.read_json(cfg.job_dir("srt-posao") / "sentences.sr.json")
    )
    assert sf.sentences[0].text == "Lijepo smo se proveli u Beogradu."
    assert sf.derived_from == pipeline.CUES_FILE
    assert not (cfg.job_dir("srt-posao") / "words.sr.json").exists()

    b = pipeline.run_build("srt-posao", language="sr", script="Cyrl", profile="tv")
    izlaz = srt.parse(b.path.read_text(encoding="utf-8"))
    assert "<i>" in izlaz[0].lines[0] and "</i>" in izlaz[0].lines[-1]
    assert "<и>" not in "".join(izlaz[0].lines)


def test_uvoz_je_transcribed_bez_zvuka(jobs):
    cfg, tmp_path = jobs
    uvezi(cfg, tmp_path)
    job = cfg.load_job("srt-posao")
    assert job.status == "transcribed"
    assert job.duration == pytest.approx(64.001)
    jdir = cfg.job_dir("srt-posao")
    assert (jdir / "cues.json").exists()
    assert not (jdir / "source.wav").exists()
    assert pipeline.is_srt_job("srt-posao")


# ==========================================================================
# Dashboard
# ==========================================================================


@pytest.fixture()
def web(tmp_path, monkeypatch):
    from subtitis import config as cfg
    from subtitis.web import app as app_mod
    from subtitis.web import db

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    db.init_db()

    from fastapi.testclient import TestClient

    with TestClient(app_mod.app) as client:
        yield client, db, cfg


def otpremi(client, text=ULAZ, name="film.sr.srt", language="sr", target="fr"):
    return client.post(
        "/jobs/srt",
        files={"srt_file": (name, text.encode("utf-8-sig"), "application/x-subrip")},
        data={"language": language, "target": target},
        follow_redirects=False,
    )


def test_otpremanje_srt_pravi_posao_i_stavlja_prevod_u_red(web):
    client, db, cfg = web
    r = otpremi(client)
    assert r.status_code == 303
    job_id = r.headers["location"].rsplit("/", 1)[-1]
    row = db.get_job(job_id)
    assert row["status"] == "transcribed"
    assert row["language"] == "sr"
    task = db.pending_task(job_id)
    assert task["kind"] == "translate"
    import json

    assert json.loads(task["params"])["to"] == "fr"
    jdir = cfg.job_dir(job_id)
    assert (jdir / "cues.json").exists()
    assert (jdir / "sentences.sr.json").exists()
    assert not list(jdir.glob("words.*.json"))


def test_otpremanje_neispravnog_fajla_ne_ostavlja_posao(web):
    client, db, cfg = web
    r = otpremi(client, text="ovo nije titl")
    assert r.status_code == 400
    assert "nije SRT" in r.json()["detail"]
    assert db.list_jobs() == []
    assert not any(cfg.JOBS_DIR.glob("*/cues.json"))


def test_isti_izvorni_i_ciljni_jezik_se_odbija(web):
    client, _, _ = web
    assert otpremi(client, target="sr").status_code == 400


def test_editor_podnosi_posao_bez_zvuka(web):
    client, _, _ = web
    job_id = otpremi(client).headers["location"].rsplit("/", 1)[-1]
    r = client.get(f"/jobs/{job_id}")
    assert r.status_code == 200
    assert 'class="play"' not in r.text
    assert "<b>Vremena iz tvog fajla ostaju netaknuta</b>" in r.text
    assert "U ovom režimu nema zvuka" in r.text
    assert 'class="low"' not in r.text
    assert "Juče sam otišao u grad sa svojim bratom" in r.text
    # sažetak uvoza, bez lažnog „interpunkcija nije pokretana"
    assert "7 cue-ova → 6 rečenica" in r.text
    # nema reda „interpunkcija nije pokretana" u sažetku (reč se javlja u ograničenjima)
    assert '<span class="muted small">interpunkcija</span>' not in r.text


def test_editor_obelezava_cue_ove_preko_profila(web, monkeypatch):
    client, db, cfg = web
    from subtitis import translate as tr

    job_id = otpremi(client).headers["location"].rsplit("/", 1)[-1]
    monkeypatch.setattr(tr, "OllamaClient", lambda **kw: FakeClient())
    pipeline.run_translate(job_id, "fr")
    b = pipeline.run_build(job_id, language="fr", profile="tv")
    assert b.overflows, "produžen prevod mora da probije profil"

    r = client.get(f"/jobs/{job_id}?lang=fr")
    assert r.status_code == 200
    assert 'class="play"' not in r.text
    assert "flag-over" in r.text
    worst = b.overflows[0]
    assert f"cue #{worst.cue}:" in r.text
    assert f"{len(b.overflows)} cue-ova probija profil" in r.text
    # dve kolone: original levo
    assert "original (sr)" in r.text


def test_srt_posao_se_ne_transkribuje(web):
    client, db, _ = web
    job_id = otpremi(client).headers["location"].rsplit("/", 1)[-1]
    db.finish_task(db.pending_task(job_id)["id"], "done")
    r = client.post(f"/jobs/{job_id}/retranscribe", follow_redirects=False)
    assert r.status_code == 409

    from subtitis.config import SubtitisError
    from subtitis.web import worker

    with pytest.raises(SubtitisError, match="nema zvuka"):
        worker.run_task({"job_id": job_id, "kind": "transcribe", "params": "{}"})


def test_cue_koji_u_prevodu_ostane_isti_ne_unosi_pismo_izvora(prof):
    """Ćirilični izvor, cue sa samim imenom: u francuskom mora ostati latinica."""
    text = "1\n00:00:01,000 --> 00:00:02,000\nМарко!\n\n2\n00:00:02,500 --> 00:00:04,000\nДођи овамо.\n"
    cues = srt.parse(text)
    from subtitis import translit

    sents, _ = fixedcues.assemble(cues, normalize=lambda t: translit.convert(t, "Latn"))
    fr = {1: "Marko!", 2: "Viens ici."}
    sents = [s.model_copy(update={"text": fr[s.id]}) for s in sents]
    built = fixedcues.build(source_file(cues), sents, prof, language="fr")
    assert built.cues[0].lines == ["Marko!"]
    assert built.unchanged == 0


def test_izvor_sa_delimicnim_kurzivom_nije_izgubio_kurziv(prof):
    text = "1\n00:00:01,000 --> 00:00:03,000\nGledali smo <i>Kazablanku</i> juče.\n"
    cues = srt.parse(text)
    sents, _ = fixedcues.assemble(cues)
    built = fixedcues.build(source_file(cues), sents, prof, language="sr", exact=True)
    assert built.cues[0].lines == ["Gledali smo <i>Kazablanku</i> juče."]
    assert built.italic_lost == []


def test_nastavak_sa_tri_tacke_u_izvoru_ostaje_doslovno(prof):
    text = (
        "1\n00:00:01,000 --> 00:00:02,000\nMislio sam da...\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\n...ćemo stići.\n"
    )
    cues = srt.parse(text)
    sents, _ = fixedcues.assemble(cues)
    built = fixedcues.build(source_file(cues), sents, prof, language="sr", exact=True)
    assert [c.lines for c in built.cues] == [["Mislio sam da..."], ["...ćemo stići."]]


def test_cue_ceo_u_kurzivu_ostaje_u_kurzivu(prof):
    """Glas van kadra: ceo cue u <i>, dva reda, kraj jedne i cela druga rečenica."""
    text = (
        "1\n00:00:01,000 --> 00:00:03,000\nNeko je došao\n\n"
        "2\n00:00:03,000 --> 00:00:06,000\n<i>u ponoć. Niko ga\nnije video.</i>\n\n"
        "3\n00:00:06,000 --> 00:00:08,000\n<i>NASLOV</i>\n"
    )
    cues = srt.parse(text)
    sents, stats = fixedcues.assemble(cues)
    assert stats.partial_italic == []
    fr = {1: "Quelqu'un est venu à minuit.", 2: "Personne ne l'a vu.", 3: "TITRE"}
    sents = [s.model_copy(update={"text": fr[s.id]}) for s in sents]
    built = fixedcues.build(source_file(cues), sents, prof, language="fr")
    assert built.italic_lost == []
    assert not built.cues[0].lines[0].startswith("<i>")
    for c in built.cues[1:]:
        plain = " ".join(c.lines)
        assert plain.startswith("<i>") and plain.endswith("</i>")
        # nijedan deo teksta van kurziva
        assert fixedcues.strip_markup(plain) == plain.replace("<i>", "").replace("</i>", "")
        assert all(l.startswith("<i>") and l.endswith("</i>") for l in c.lines)
    assert built.cues[2].lines == ["<i>TITRE</i>"]


# ==========================================================================
# Termini iz glosara se ne seku
# ==========================================================================

TERM = "Open Door Workshop"


def test_termini_glosara_po_jeziku():
    from subtitis.config import glossary_terms

    g = {"sr->en": {"Radionica otvorenih vrata": TERM}, "en->fr": {"activity leader": "éducateur"}}
    assert glossary_terms(g, "en") == [TERM, "activity leader"]
    assert glossary_terms(g, "sr") == ["Radionica otvorenih vrata"]


def test_rez_izmedju_cue_ova_ne_sece_termin(prof):
    """Slučaj iz merenja, cue 11/12: udeo pada usred termina."""
    ws = words_of(f"I am the coordinator from the {TERM}, from the Town Culture Centre.")
    shares = [0.47, 0.53]
    rules = linebreak.rules_for("en")
    bez = linebreak.split_by_shares(ws, shares, prof, rules)
    sa = linebreak.split_by_shares(ws, shares, prof, rules, linebreak.term_tokens([TERM]))
    cut = lambda r: " ".join(w.w for w in ws[: r[0][1]])
    assert "Open Door" in cut(bez) and "Workshop" not in cut(bez)
    assert TERM in cut(sa) or "Open" not in cut(sa)


def test_prelom_reda_ne_sece_termin():
    """Bez kazne: `We came from the Open / Door Workshop.` (21 zn./red)."""
    prof = Profile(name="t", max_chars_per_line=21)
    ws = words_of(f"We came from the {TERM}.")
    rules = linebreak.rules_for("en")
    render = lambda lines: [" ".join(w.w for w in ws[a:b]) for a, b in lines]
    bez = render(linebreak.best_lines(ws, prof, rules))
    sa = render(linebreak.best_lines(ws, prof, rules, linebreak.term_tokens([TERM])))
    assert not any(TERM in l for l in bez)
    assert any(TERM in l for l in sa), sa


def test_termin_se_prepoznaje_bez_obzira_na_apostrof_i_velicinu_slova(prof):
    ws = words_of("We visited the children’s home yesterday with everyone.")
    terms = linebreak.term_tokens(["Children's Home"])
    span = linebreak._Span(ws, prof, linebreak.rules_for("en"), terms)
    k = [w.w for w in ws].index("children’s")
    assert span.break_pen[k] >= linebreak.PENALTY_SPLIT_GLOSSARY_TERM


# ==========================================================================
# Neprevedena rečenica mora da se vidi
# ==========================================================================


class FailingClient(FakeClient):
    """Za rečenicu `bad` svaki put vrati odgovor bez nje — model „spoji" linije."""

    def __init__(self, bad: str) -> None:
        # po tekstu, ne po broju: model vidi lokalne brojeve 1..n
        super().__init__()
        self.bad = bad

    def chat(self, system: str, user: str) -> str:
        out = super().chat(system, user)
        return "\n".join(l for l in out.splitlines() if not l.split("|", 1)[1].startswith(self.bad))


def test_neprevedena_recenica_se_prijavljuje_glasno(web, monkeypatch):
    client, db, cfg = web
    from subtitis import translate as tr

    job_id = otpremi(client).headers["location"].rsplit("/", 1)[-1]
    logs: list[str] = []
    monkeypatch.setattr(tr, "OllamaClient", lambda **kw: FailingClient(bad="Juče sam otišao"))
    pipeline.run_translate(job_id, "fr", on_log=logs.append)
    assert any("1 rečenica neprevedena — 2 cue-a (1–2) na fr" in m for m in logs), logs

    logs.clear()
    b = pipeline.run_build(job_id, language="fr", profile="tv", on_log=logs.append)
    assert b.translation_failed == 1
    assert any("1 rečenica neprevedena — 2 cue-a (1–2)" in m for m in logs)
    job = cfg.load_job(job_id)
    build = [h for h in job.history if h.phase == "build"][-1]
    assert build.detail["untranslated"]["failed"] == [
        {"sentence": 1, "cues": [1, 2], "reason": "translation_failed"}
    ]

    # stranica: blok na vrhu i na drugom jeziku, crveni red u editoru
    page = client.get(f"/jobs/{job_id}").text
    # krug 27: u redu trake broj i jezik, bez klika; spisak u prozorčiću
    red = page.split('<span class="flagtxt">')[1].split("</span>")[0]
    assert "<b>fr: 1 rečenica neprevedena</b> (2 cue-a)" in red
    assert "1 rečenica neprevedena — 2 cue-a (1–2)" in page.split("<template>")[1]
    assert "rečenica 1 — cue 1, 2" in page
    page = client.get(f"/jobs/{job_id}?lang=fr").text
    assert "flag-failed" in page

    # ručna ispravka skida oznaku
    r = client.post(f"/jobs/{job_id}/sentences", json={
        "lang": "fr",
        "sentences": [{"id": 1, "text": "Hier je suis allé en ville avec mon frère."}],
    })
    assert r.status_code == 200
    sf = SentencesFile.model_validate(cfg.read_json(cfg.job_dir(job_id) / "sentences.fr.json"))
    assert "translation_failed" not in sf.sentences[0].flags
    assert "rečenica neprevedena" not in client.get(f"/jobs/{job_id}").text


# ==========================================================================
# Predlog profila kad je prekoračenje isključivo CPS
# ==========================================================================


def _srt_posao(cfg, tmp_path, text, fr_texts, job_id="cps"):
    src = tmp_path / "film.sr.srt"
    src.write_text(text, encoding="utf-8")
    pipeline.run_import_srt(src, "sr", job_id=job_id)
    jdir = cfg.job_dir(job_id)
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    cfg.write_model(jdir / "sentences.fr.json", sf.model_copy(update={
        "language": "fr", "derived_from": "sentences.sr.json",
        "sentences": [s.model_copy(update={"text": t}) for s, t in zip(sf.sentences, fr_texts)],
    }))
    return job_id


def test_predlog_blazeg_profila_je_izracunat(jobs, monkeypatch):
    cfg, tmp_path = jobs
    # od kruga 6 postoji jedan profil za višeredni titl; blaži se ovde pravi
    # privremeno, da predlog ima šta da izračuna
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "tv.json").write_text((cfg.PROFILES_DIR / "tv.json").read_text(encoding="utf-8"), encoding="utf-8")
    cfg.write_json(profiles / "blazi.json", {"name": "blazi", "max_chars_per_line": 47, "max_cps": 23})
    monkeypatch.setattr(cfg, "PROFILES_DIR", profiles)
    # 22 znaka u sekundi: preko tv (20), ispod blazi (23)
    job = _srt_posao(cfg, tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nDa.\n",
                     ["Oui, bien sûr mon ami."])
    logs: list[str] = []
    b = pipeline.run_build(job, language="fr", profile="tv", on_log=logs.append)
    assert len(b.overflows) == 1 and not b.overflows[0].chars_over
    direct = pipeline.run_build(job, language="fr", profile="blazi", out=tmp_path / "w.srt")
    line = next(m for m in logs if m.startswith("predlog:"))
    assert f"profil blazi (23 CPS, 47 zn./red) bi ih smanjio sa 1 na {len(direct.overflows)}" in line
    build = [h for h in cfg.load_job(job).history if h.phase == "build" and h.detail["profile"] == "tv"][-1]
    assert build.detail["profile_suggestions"] == [
        {"profile": "blazi", "max_cps": 23.0, "max_chars": 47, "overflow": len(direct.overflows)}
    ]


def test_bez_predloga_kad_probija_i_duzina(jobs):
    cfg, tmp_path = jobs
    dugo = "Oui " * 30
    job = _srt_posao(cfg, tmp_path, "1\n00:00:01,000 --> 00:00:02,000\nDa.\n", [dugo.strip()])
    logs: list[str] = []
    b = pipeline.run_build(job, language="fr", profile="tv", on_log=logs.append)
    assert any(o.chars_over or o.lines_over for o in b.overflows)
    assert not any(m.startswith("predlog:") for m in logs)


# ==========================================================================
# Profil izmeren na ulaznom SRT-u
# ==========================================================================


def _cues(spec):
    """[(trajanje u s, [redovi])] -> CuesFile."""
    out, t = [], 0
    for dur, lines in spec:
        out.append(SourceCue(start_ms=t, end_ms=t + int(dur * 1000), lines=lines))
        t += int(dur * 1000) + 100
    return CuesFile(job_id="x", source_file="x.srt", language="sr", cues=out)


def test_profil_iz_originala_percentil_i_maksimum():
    # 20 cue-ova od po jednog reda dužine 20..39 znakova, po 2 s; jedan ekstrem
    spec = [(2.0, ["x" * n]) for n in range(20, 40)]
    spec.append((1.0, ["y" * 51, "z" * 10]))  # najduži red 51, 62 zn./1 s
    src = _cues(spec)
    p = fixedcues.source_profile(src, fixedcues.ORIGINAL_PROFILE)
    m = fixedcues.source_profile(src, fixedcues.ORIGINAL_MAX_PROFILE)
    assert (m.max_chars_per_line, m.max_cps, m.max_lines) == (51, 62.0, 2)
    # 95. percentil ne vidi jedan ekstrem
    assert p.max_chars_per_line == 39
    assert p.max_cps == 19.5
    assert p.max_lines == 2
    line = fixedcues.source_profile_line(src)
    assert line.startswith("profil iz originala: 39 zn./red, 19.5 CPS, do 2 reda")


def test_gore_nego_isti_cue_u_originalu(prof):
    """Cue koji je i u originalu bio brz nije mesto za skraćivanje."""
    # cue 1: original 30 zn./1 s = 30 CPS (već brz); cue 2: original spor
    src = _cues([(1.0, ["a" * 30]), (1.0, ["Da."])])
    sents = [
        Sentence(id=1, s=0, e=1, text="b" * 29 + ".", words=(0, -1),
                 cue_parts=[CuePart(cue=0, chars=30, share=1.0)]),
        Sentence(id=2, s=1, e=2, text="Oui, bien sûr, mon ami.", words=(0, -1),
                 cue_parts=[CuePart(cue=1, chars=3, share=1.0)]),
    ]
    built = fixedcues.build(src, sents, prof, language="fr")
    found = fixedcues.overflows(built, prof, src)
    assert [(o.cue, o.worse_than_source) for o in found] == [(2, True), (1, False)]
    lines = fixedcues.overflow_lines(found, 2, prof)
    assert lines[1] == (
        "od toga GORE NEGO ISTI CUE U ORIGINALU: 1 — tu skraćuj; "
        "kao u originalu (isto toliko dugo ili brzo): 1"
    )
    assert lines[2].startswith("najgori od njih: #2:") and "original 3.0" in lines[2]


def test_srt_posao_podrazumevano_meri_originalom(jobs):
    cfg, tmp_path = jobs
    logs: list[str] = []
    p = tmp_path / "film.sr.srt"
    p.write_bytes(ULAZ.encode("utf-8"))
    pipeline.run_import_srt(p, "sr", job_id="orig", on_log=logs.append)
    assert any(m.startswith("profil iz originala: ") for m in logs)
    imp = [h for h in cfg.load_job("orig").history if h.phase == "srt_import"][-1]
    assert set(imp.detail["source_profile"]) == {"original", "original-max"}

    b = pipeline.run_build("orig", language="sr")
    build = [h for h in cfg.load_job("orig").history if h.phase == "build"][-1]
    assert build.detail["profile"] == "original"
    # izvorni jezik meren svojim profilom — nijedan cue nije gori od sebe
    assert all(not o.worse_than_source for o in b.overflows)
    # ostali profili i dalje na izbor
    pipeline.run_build("orig", language="sr", profile="tv")
    build = [h for h in cfg.load_job("orig").history if h.phase == "build"][-1]
    assert build.detail["profile"] == "tv"


def test_profil_original_samo_za_srt_posao(jobs):
    from subtitis.config import SubtitisError
    from subtitis.models import Job

    cfg, _ = jobs
    cfg.job_dir("govor").mkdir(parents=True)
    cfg.save_job(Job(job_id="govor", source_file="x.wav", language="sr"))
    with pytest.raises(SubtitisError, match="nije nastao iz SRT-a"):
        pipeline.resolve_profile("govor", "original")
    assert pipeline.resolve_profile("govor", None).name == cfg.default_profile()


# ==========================================================================
# Granica rečenice mimo interpunkcije: trajanje (uključeno), pauza (isključena)
# ==========================================================================


def _srt(blocks):
    """[(početak ms, kraj ms, tekst)] -> SRT tekst."""
    return "".join(
        f"{k}\n{srt.timestamp(a / 1000)} --> {srt.timestamp(b / 1000)}\n{t}\n\n"
        for k, (a, b, t) in enumerate(blocks, start=1)
    )


#: Prijava 2026-09-28, cue 30–34: bez tačke, 15,96 s, 3,27 s tišine pred poslednjim.
PRIJAVA = [  # prava vremena iz posla; tekst izmišljen, iste strukture
    (171_160, 172_293, "Ovde je trener Vuk"),
    (172_626, 175_091, "Lana, Teodora,"),
    (176_724, 179_190, "Ilija, Nevena, Ognjen,"),
    (179_823, 181_422, "Pored nas je grupa Zeleni bor..."),
    (184_687, 187_120, "...posle podne idemo svi zajedno na jezero."),
]


def test_predugacka_recenica_se_sece_na_najvecoj_pauzi():
    cues = srt.parse(_srt(PRIJAVA))
    assert (cues[-1].end_ms - cues[0].start_ms) / 1000 > fixedcues.SRT_MAX_SENTENCE_DURATION
    sents, stats = fixedcues.assemble(cues)
    assert [s.text for s in sents] == [
        "Ovde je trener Vuk Lana, Teodora, Ilija, Nevena, Ognjen, Pored nas je grupa Zeleni bor...",
        "posle podne idemo svi zajedno na jezero.",
    ]
    assert stats.duration_breaks == [4]
    assert all(s.e - s.s <= fixedcues.SRT_MAX_SENTENCE_DURATION for s in sents)


def test_bez_trajanja_ostaje_spojeno():
    sents, _ = fixedcues.assemble(srt.parse(_srt(PRIJAVA)), max_duration=None)
    assert len(sents) == 1


def test_pri_jednakim_pauzama_sece_najblize_sredini():
    blocks = [(k * 4_000, k * 4_000 + 3_500, f"reč{k}") for k in range(6)]  # 23,5 s, pauze po 0,5 s
    sents, stats = fixedcues.assemble(srt.parse(_srt(blocks)))
    assert [len(s.cue_parts) for s in sents] == [3, 3]
    assert stats.duration_breaks == [3]


def test_jedan_predug_cue_se_ne_sece():
    """Unutar cue-a nema vremena po rečima — nema ni gde da se preseče."""
    sents, stats = fixedcues.assemble(srt.parse(_srt([(0, 20_000, "Jedan dug cue bez kraja")])))
    assert len(sents) == 1 and stats.duration_breaks == []


def test_pauza_je_podrazumevano_iskljucena():
    """3 s tišine u kratkoj rečenici bez tačke: bez dokaza da pauza pomaže, ne seče se."""
    blocks = [(0, 2_000, "Kad stignemo do jezera,"), (5_000, 7_000, "pravimo pauzu za ručak.")]
    assert fixedcues.SRT_MAX_GAP is None
    sents, stats = fixedcues.assemble(srt.parse(_srt(blocks)))
    assert len(sents) == 1 and stats.gap_breaks == []
    # mehanizam postoji i radi kad se izričito uključi
    sents, stats = fixedcues.assemble(srt.parse(_srt(blocks)), max_gap=2.0)
    assert len(sents) == 2 and stats.gap_breaks == [1]


def test_uvoz_prijavljuje_presek_po_trajanju(jobs):
    cfg, tmp_path = jobs
    p = tmp_path / "film.sr.srt"
    p.write_text(_srt(PRIJAVA), encoding="utf-8")
    logs: list[str] = []
    r = pipeline.run_import_srt(p, "sr", job_id="traj", on_log=logs.append)
    assert r.sentences == 2
    assert any("duža od 15 s presečeno na najvećoj pauzi (posle cue 4)" in m for m in logs)
    imp = [h for h in cfg.load_job("traj").history if h.phase == "srt_import"][-1]
    assert imp.detail["duration_breaks"] == [4] and imp.detail["gap_breaks"] == []


def test_uvoz_srt_koji_ne_postoji(jobs):
    """Krug 16: jedina nepokrivena grana `run_import_srt` koja nije poruka."""
    cfg, tmp_path = jobs
    with pytest.raises(cfg.SubtitisError, match="Ulazna datoteka ne postoji"):
        pipeline.run_import_srt(tmp_path / "nema.srt", "sr")
