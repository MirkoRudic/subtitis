"""Pivot prevoda (krug 16): `off` direktno, `en` u dva skoka, glosar drugog skoka.

Lažni model prevodu dodaje redni broj poziva — ako drugi skok dobije tekst sa
oznakom prvog, francuski je stvarno nastao iz engleskog, a ne iz srpskog.
"""

from __future__ import annotations

import pytest

from subtitis import pipeline
from subtitis import config as cfg
from subtitis.models import Job, SentencesFile, Sentence


class Model:
    model = "lazni"

    def __init__(self) -> None:
        from subtitis.translate import CallStats

        self.stats = CallStats()
        self.pozivi: list[str] = []  # ulazni blok svakog poziva

    def check(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def chat(self, system: str, user: str) -> str:
        block = user.split("ZA_PREVOD:\n", 1)[1].split("\n\nKONTEKST")[0]
        self.pozivi.append(block)
        n = len(self.pozivi)
        return "\n".join(
            f"{i}|{text} [{n}]" for i, text in
            (line.split("|", 1) for line in block.splitlines() if line.strip())
        )


@pytest.fixture()
def posao(monkeypatch):
    from subtitis import translate as tr

    jid = "piv"
    jdir = cfg.job_dir(jid)
    jdir.mkdir(parents=True)
    cfg.save_job(Job(job_id=jid, source_file="a.wav", language="sr", status="transcribed"))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id=jid, language="sr", derived_from="words.sr.json",
        sentences=[Sentence(id=1, s=0.0, e=1.0, text="Dobar dan.", words=(0, 2))],
    ))
    model = Model()
    monkeypatch.setattr(tr, "OllamaClient", lambda **kw: model)
    return jid, jdir, model


def tekst(jdir, lang):
    sf = SentencesFile.model_validate(cfg.read_json(jdir / f"sentences.{lang}.json"))
    return sf.derived_from, sf.sentences[0].text


def test_off_ide_direktno_jednim_skokom(posao):
    jid, jdir, model = posao
    cfg.save_settings({"translation_pivot": "en"})  # i kad podešavanje kaže „preko en"
    hops = pipeline.run_translate(jid, "fr", pivot="off")
    assert [(h.src, h.tgt) for h in hops] == [("sr", "fr")]
    assert len(model.pozivi) == 1 and not (jdir / "sentences.en.json").exists()
    assert tekst(jdir, "fr") == ("sentences.sr.json", "Dobar dan. [1]")
    assert "prevod sr -> fr (direktno)" in (jdir / "log.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("kako", ["parametar", "podešavanje"])
def test_en_ide_u_dva_skoka_tacnim_redom(posao, kako):
    jid, jdir, model = posao
    if kako == "podešavanje":
        cfg.save_settings({"translation_pivot": "en"})
        hops = pipeline.run_translate(jid, "fr")
    else:
        hops = pipeline.run_translate(jid, "fr", pivot="en")
    assert [(h.src, h.tgt) for h in hops] == [("sr", "en"), ("en", "fr")]
    # drugi skok je dobio IZLAZ prvog, ne srpski original
    assert model.pozivi[0].endswith("|Dobar dan.") and model.pozivi[1].endswith("|Dobar dan. [1]")
    assert tekst(jdir, "en") == ("sentences.sr.json", "Dobar dan. [1]")
    assert tekst(jdir, "fr") == ("sentences.en.json", "Dobar dan. [1] [2]")
    log = (jdir / "log.txt").read_text(encoding="utf-8")
    assert "prevod sr -> fr preko en" in log
    faze = [h.phase for h in cfg.load_job(jid).history]
    assert faze == ["translate:sr->en", "translate:en->fr"]


def test_glosar_drugog_skoka_izveden_iz_postojecih_smerova(posao):
    jid, jdir, model = posao
    cfg.write_json(cfg.job_glossary_path(jid), {
        "sr->en": {"Radionica otvorenih vrata": "Open Door"},
        "sr->fr": {"Radionica otvorenih vrata": "Centre d'accueil"},
    })
    hops = pipeline.run_translate(jid, "fr", pivot="en")
    assert hops[1].glossary_entries == 1 and hops[1].glossary_derived == 1
    assert hops[1].warning is None
    assert "glosar en->fr: 1 unosa izvedeno iz sr->en i sr->fr" in (jdir / "log.txt").read_text(encoding="utf-8")


def test_prazan_glosar_drugog_skoka_upozorava(posao):
    jid, jdir, model = posao
    cfg.write_json(cfg.job_glossary_path(jid), {"sr->en": {"Radionica otvorenih vrata": "Open Door"}})
    logs: list[str] = []
    hops = pipeline.run_translate(jid, "fr", pivot="en", on_log=logs.append)
    assert hops[0].warning is None and hops[0].glossary_entries == 1
    assert "DRUGA polovina pivota" in hops[1].warning
    assert any("UPOZORENJE: glosar nema nijedan unos za en->fr" in m and "DRUGA polovina" in m for m in logs)


def test_pivot_jednak_cilju_nije_pivot(posao):
    jid, jdir, model = posao
    hops = pipeline.run_translate(jid, "en", pivot="en")
    assert [(h.src, h.tgt) for h in hops] == [("sr", "en")]


# --- ostale nepokrivene grane, jeftine ---------------------------------------


def test_rucni_prevod_se_ne_gazi(posao):
    jid, jdir, model = posao
    pipeline.run_translate(jid, "fr", pivot="off")
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.fr.json"))
    cfg.write_model(jdir / "sentences.fr.json", sf.model_copy(update={"edited_by_human": True}))
    with pytest.raises(cfg.SubtitisError, match="ručno menjan"):
        pipeline.run_translate(jid, "fr", pivot="off")


def test_isti_izvorni_i_ciljni_jezik(posao):
    jid, _, _ = posao
    with pytest.raises(cfg.SubtitisError, match="isti"):
        pipeline.run_translate(jid, "sr")
