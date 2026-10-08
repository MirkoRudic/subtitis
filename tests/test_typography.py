"""Francuski pravopisni razmaci: nerazlomljiv razmak ispred `? ! : ; »` i iza `«`.

Običan razmak bi plejeru dozvolio da prelomi red između reči i znaka.
"""

from __future__ import annotations

import json

import pytest

from subtitis import linebreak, pipeline, srt, typography
from subtitis.models import Job, Profile, Sentence, SentencesFile, Word
from subtitis.segment import tokenize_text
from subtitis.typography import NBSP

N = NBSP


@pytest.mark.parametrize("raw,expected", [
    ("Est-elle belle ?", f"Est-elle belle{N}?"),
    ("Est-elle belle?", f"Est-elle belle{N}?"),
    ("Quoi ?!", f"Quoi{N}?!"),
    ("Attention : danger ; fuyez !", f"Attention{N}: danger{N}; fuyez{N}!"),
    ("« Bonjour »", f"«{N}Bonjour{N}»"),
    ("«Bonjour»", f"«{N}Bonjour{N}»"),
    ("Tu viens ?»", f"Tu viens{N}?{N}»"),
    ("<i>belle</i> ?", f"<i>belle</i>{N}?"),
    ("Rendez-vous à 10:30.", "Rendez-vous à 10:30."),
    ("Voir https://exemple.fr", "Voir https://exemple.fr"),
    ("? au début", "? au début"),
])
def test_francuski_razmaci(raw, expected):
    assert typography.french(raw) == expected


def test_idempotentno():
    once = typography.french("« Tu viens ?! » Attention : oui ;")
    assert typography.french(once) == once


def test_samo_za_francuski():
    lines = ["Da li je lep ?", "Is it nice ?"]
    assert typography.apply(lines, "sr") == lines
    assert typography.apply(lines, "en") == lines
    assert typography.apply(lines, "fr") == [f"Da li je lep{N}?", f"Is it nice{N}?"]


def words(text: str) -> list[Word]:
    return [Word(i=k, s=0.0, e=0.0, w=t, p=1.0) for k, t in enumerate(tokenize_text(text))]


def test_prelamanje_racuna_nerazlomljiv_razmak_u_duzinu():
    prof = Profile(name="t")
    ws = words("« Tu viens ? »")
    fr = linebreak._Span(ws, prof, linebreak.rules_for("fr"))
    sr = linebreak._Span(ws, prof, linebreak.rules_for("sr"))
    assert sr.length(0, len(ws)) == len("«Tu viens?»")
    assert fr.length(0, len(ws)) == len(typography.french("«Tu viens?»"))


def test_red_se_ne_zavrsava_otvorenim_navodnikom():
    prof = Profile(name="t")
    ws = words("Il a dit « bonjour » et puis il est parti.")
    span = linebreak._Span(ws, prof, linebreak.rules_for("fr"))
    k = [w.w for w in ws].index("«")
    assert not span.can_break[k]


def test_prelamanje_ne_prelazi_profil_zbog_razmaka():
    """Bez računanja razmaka red od tačno `max` znakova postane `max + 1`."""
    # "Qui est ce monsieur?" ima tačno 20 znakova; sa razmakom 21. Bez
    # računanja razmaka ostane u jednom redu i posle obrade probije profil.
    prof = Profile(name="t", max_chars_per_line=20, max_lines=2, max_cps=99)
    sent = Sentence(id=1, s=0.0, e=20.0, text="Qui est ce monsieur ?", words=(0, 0))
    cues = linebreak.build_cues(
        [sent], [], prof, language="fr", log=lambda m: None, interpolate_times=True
    )
    lines = [l for c in cues for l in typography.apply(c.lines, "fr")]
    assert all(len(l) <= 20 for l in lines), lines
    assert lines == ["Qui est", f"ce monsieur{N}?"]


# ==========================================================================
# Pajplajn: i običan prevod i prevod postojećeg SRT-a
# ==========================================================================


@pytest.fixture()
def jobs(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    return cfg, tmp_path


def test_obican_prevod_dobija_razmake(jobs):
    cfg, _ = jobs
    jdir = cfg.job_dir("govor")
    jdir.mkdir(parents=True)
    cfg.save_job(Job(job_id="govor", source_file=str(jdir / "film.wav"), language="sr",
                     status="done", duration=5.0))
    cfg.write_model(jdir / "sentences.fr.json", SentencesFile(
        job_id="govor", language="fr", derived_from="sentences.sr.json",
        sentences=[Sentence(id=1, s=0.0, e=3.0, text="Est-elle belle ?", words=(0, 1))],
    ))
    b = pipeline.run_build("govor", language="fr", profile="tv")
    out = b.path.read_text(encoding="utf-8")
    assert f"Est-elle belle{N}?" in out
    assert "belle ?" not in out and "belle?" not in out


def test_srt_izvestaj_racuna_nad_konacnim_tekstom(jobs, monkeypatch):
    cfg, tmp_path = jobs
    src = tmp_path / "film.sr.srt"
    src.write_text("1\n00:00:01,000 --> 00:00:03,000\nDa li je lep?\n", encoding="utf-8")
    pipeline.run_import_srt(src, "sr", job_id="srt")
    jdir = cfg.job_dir("srt")
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    cfg.write_model(jdir / "sentences.fr.json", sf.model_copy(update={
        "language": "fr", "derived_from": "sentences.sr.json",
        "sentences": [s.model_copy(update={"text": "Est-elle belle ?"}) for s in sf.sentences],
    }))
    prof = tmp_path / "uzak.json"
    prof.write_text(json.dumps({"name": "uzak", "max_chars_per_line": 12, "max_lines": 1}))

    b = pipeline.run_build("srt", language="fr", profile=str(prof))
    line = srt.parse(b.path.read_text(encoding="utf-8"))[0].lines[0]
    assert line == f"Est-elle belle{N}?"
    (o,) = b.overflows
    assert o.chars == len(line) == 16  # sa nerazlomljivim razmakom
    assert o.chars_over == 4


# ==========================================================================
# Apostrofi: jedan oblik po fajlu
# ==========================================================================


@pytest.mark.parametrize("raw,expected", [
    ("I'm 17 and we’re here.", "I’m 17 and we’re here."),
    ("the kids' toys", "the kids’ toys"),
    ("'Hello,' she said", "'Hello,' she said"),  # navodnici, ne apostrofi
    ("<i>don't</i>", "<i>don’t</i>"),
])
def test_apostrofi(raw, expected):
    assert typography.apostrophes(raw) == expected


def test_apostrofi_za_svaki_jezik_i_uz_francuske_razmake():
    assert typography.apply(["I'm here."], "en") == ["I’m here."]
    assert typography.apply(["C'est vrai ?"], "fr") == [f"C’est vrai{N}?"]


# ==========================================================================
# Navodnici
# ==========================================================================


@pytest.mark.parametrize("lines,expected", [
    # stvaran cue 156 iz prvog francuskog posla
    (['c\'est : " Comment avez-vous passé l\'été ?"'],
     [f"c’est{N}: «{N}Comment avez-vous passé l’été{N}?{N}»"]),
    (["Il a dit “bonjour” hier."], [f"Il a dit «{N}bonjour{N}» hier."]),
    (["Il a dit „bonjour“ hier."], [f"Il a dit «{N}bonjour{N}» hier."]),
    # citat preko dva reda
    (['Il a dit : "Je viens', 'demain."'], [f"Il a dit{N}: «{N}Je viens", f"demain.{N}»"]),
])
def test_francuski_navodnici(lines, expected):
    assert typography.apply(lines, "fr") == expected


def test_francuski_navodnici_idempotentni():
    once = typography.apply(['Il a dit : " Oui ?"'], "fr")
    assert typography.apply(once, "fr") == once


def test_engleski_navodnici_bez_razmaka_unutra():
    # prav navodnik -> tipografski (2026-09-28), razmak unutra se skida
    assert typography.apply(['She said " Hello there "'], "en") == ["She said “Hello there”"]
    assert typography.apply(["She said “Hello”."], "en") == ["She said “Hello”."]


def test_sirina_reda_racuna_navodnike_i_razmak_oko_zatvorenog():
    """Stvaran cue 156: `?»` dobija DVA nerazlomljiva razmaka, `«` jedan.
    Pre merenja kroz `typography` red je izlazio sa 43 znaka na profilu 42."""
    prof = Profile(name="t", max_chars_per_line=42)
    text = 'Et puis, c\'est : " Comment avez-vous passé l\'été ?"'
    toks = tokenize_text(text)
    ws = [Word(i=k, s=0.0, e=0.0, w=t, p=1.0) for k, t in enumerate(toks)]

    def lines_with(rules):
        return [
            typography.apply([" ".join(toks[a:b])], "fr")[0]
            for a, b in linebreak.best_lines(ws, prof, rules)
        ]

    ok = lines_with(linebreak.rules_for("fr"))
    assert all(len(l) <= 42 for l in ok), ok
    # Fajl ima tri nerazlomljiva razmaka više od golog spoja tokena: ispred
    # `:`, ispred `?` i ispred `»`. Staro merenje je znalo za prva dva — treći
    # je bio onaj +1 na cue-u 156.
    from subtitis.segment import join_words

    real_span = linebreak._Span(ws, prof, linebreak.rules_for("fr"))
    final = typography.apply([join_words(ws)], "fr")[0]
    assert real_span.length(0, len(ws)) == len(final) == len(join_words(ws)) + 3


# ==========================================================================
# Pravi navodnici: strana pre prelamanja, uravnoteženost
# ==========================================================================


@pytest.mark.parametrize("lang,raw,expected", [
    ("en", '"Ringe ringe raja" "Ringe raja sa"', "“Ringe ringe raja” “Ringe raja sa”"),
    ("en", 'He said " hello " to me.', "He said “hello” to me."),
    ("fr", 'Il dit "oui".', "Il dit «oui»."),
    ("sr", 'Rekao je "da".', "Rekao je „da“."),
    ("en", "Already “curly” stays.", "Already “curly” stays."),
    ("sr", "Srpski „original“ i “engleski” ostaju.", "Srpski „original“ i “engleski” ostaju."),
])
def test_usmereni_navodnici(lang, raw, expected):
    assert typography.directional_quotes(raw, lang) == expected


def test_engleski_izlaz_nema_pravih_navodnika():
    assert typography.apply(['She said "hi" and "bye".'], "en") == ["She said “hi” and “bye”."]


@pytest.mark.parametrize("text,ok", [
    ('"Ringe ringe raja" "Ringe raja sa"', True),
    ('"Ringe ringe raja" "', False),
    ("“Ringe” “Raja”", True),
    ("Raja” “sa", False),          # počinje zatvaranjem
    ("” Ringe Ringe", False),
    ("„Da“, rekao je.", True),     # srpski: “ zatvara
    ("«Oui», dit-il.", True),
    ("Bez navodnika.", True),
])
def test_uravnotezenost(text, ok):
    assert typography.quotes_balanced(text) is ok


def test_rez_izmedju_cue_ova_ne_odvaja_navodnik_od_reci():
    """Stvaran kvar, cue 66–71: `"Ringe ringe raja" "` / `Ringe raja sa ""`. Model je
    vratio uravnotežen tekst; navodnike je razneo rez između cue-ova."""
    from subtitis import fixedcues
    from subtitis.models import CuePart, CuesFile, Sentence, SourceCue

    chant = ["“Ringe ringe raja”", "“Ringe raja sa”"] * 3
    src = CuesFile(job_id="x", source_file="x.srt", language="sr", cues=[
        SourceCue(start_ms=k * 1000, end_ms=k * 1000 + 900, lines=[t])
        for k, t in enumerate(chant)
    ])
    sent = Sentence(
        id=1, s=0, e=6, words=(0, -1),
        text='"Ringe ringe raja" "Ringe raja sa" "Ringe ringe raja" "Ringe raja sa" "Ringe ringe raja" "Ringe raja sa"',
        cue_parts=[CuePart(cue=k, chars=14, share=1 / 6) for k in range(6)],
    )
    built = fixedcues.build(src, [sent], Profile(name="t"), language="en")
    lines = [typography.apply(c.lines, "en") for c in built.cues]
    assert lines == [[t] for t in chant]
