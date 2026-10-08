"""Testovi segmentacije reči -> rečenice."""

from __future__ import annotations

import pytest
from conftest import make_words

from subtitis.models import Sentence, Word
from subtitis.segment import (
    coverage,
    join_words,
    segment,
    split_token,
    tokenize_text,
)


def texts(sentences):
    return [s.text for s in sentences]


# --- Tokenizacija ----------------------------------------------------------


def test_split_token():
    assert split_token("Dobar") == ["Dobar"]
    assert split_token("došli.") == ["došli", "."]
    assert split_token("dan,") == ["dan", ","]
    assert split_token("-") == ["-"]
    assert split_token("stvarno?!") == ["stvarno", "?!"]


def test_tokenize_i_join_su_inverzni():
    text = "Dobar dan, hvala što ste došli."
    toks = tokenize_text(text)
    words = [Word(i=k, s=k, e=k + 1, w=t, p=1.0) for k, t in enumerate(toks)]
    assert join_words(words) == text


# --- Granice rečenica ------------------------------------------------------


def test_osnovna_granica():
    s = segment(make_words("Dobar dan. Hvala što ste došli."))
    assert texts(s) == ["Dobar dan.", "Hvala što ste došli."]


def test_upitnik_i_uzvicnik():
    s = segment(make_words("Šta je bilo? Ništa! Baš ništa…"))
    assert texts(s) == ["Šta je bilo?", "Ništa!", "Baš ništa…"]


def test_skracenice_ne_lome_recenicu():
    s = segment(make_words("Kupili smo hleb, mleko itd. za doručak."))
    assert len(s) == 1


def test_titula_ne_lomi_ni_pred_velikim_slovom():
    s = segment(make_words("Došao je dr. Petrović u pola devet."))
    assert len(s) == 1


def test_opsta_skracenica_pred_velikim_slovom_lomi():
    s = segment(make_words("Bilo je hleba, mleka itd. Onda smo krenuli."))
    assert len(s) == 2


def test_redni_broj_ne_lomi():
    s = segment(make_words("To se desilo 23. maja prošle godine."))
    assert len(s) == 1


def test_broj_na_kraju_recenice_lomi():
    s = segment(make_words("Bilo ih je 23. Onda su otišli."))
    assert len(s) == 2


def test_npr_ne_lomi():
    s = segment(make_words("Uzmi npr. ovaj slučaj."))
    assert len(s) == 1


# --- Raspon indeksa --------------------------------------------------------


def test_raspon_indeksa_je_inkluzivan():
    words = make_words("Dobar dan. Hvala.")
    s = segment(words)
    lo, hi = s[0].words
    assert [w.w for w in words[lo : hi + 1]] == ["Dobar", "dan", "."]
    assert s[0].s == words[0].s
    assert s[0].e == words[hi].e


def test_vremena_dolaze_iz_reci():
    words = make_words("Dobar dan.")
    s = segment(words)
    assert s[0].s == words[0].s
    assert s[0].e == words[-1].e


# --- Zastavice -------------------------------------------------------------


def test_low_confidence_flag():
    words = make_words("Nesigurna rec ovde.")
    words[1].p = 0.42
    s = segment(words)
    assert s[0].flags == ["low_confidence"]


def test_bez_flaga_kad_je_sve_sigurno():
    s = segment(make_words("Sve je jasno."))
    assert s[0].flags == []


# --- Lomljenje predugačkih rečenica ---------------------------------------


def test_duga_recenica_se_lomi_na_zarezu():
    words = make_words(
        "Prvi deo ove veoma duge recenice, drugi deo ove veoma duge recenice",
        dur_per_char=0.3,
    )
    assert words[-1].e - words[0].s > 15.0
    s = segment(words)
    assert len(s) >= 2
    assert s[0].text.endswith(",")


def test_duga_recenica_bez_zareza_lomi_na_pauzi():
    words = make_words("Jedan dva tri cetiri pet sest sedam osam", dur_per_char=0.5)
    # ubaci jasnu pauzu posle 4. tokena
    for w in words[4:]:
        w.s = round(w.s + 1.5, 3)
        w.e = round(w.e + 1.5, 3)
    assert words[-1].e - words[0].s > 15.0
    s = segment(words)
    assert len(s) >= 2


def test_kratka_recenica_se_ne_lomi():
    s = segment(make_words("Kratko i jasno."))
    assert len(s) == 1


def test_prazan_ulaz():
    assert segment([]) == []


# --- Pokrivenost audija ----------------------------------------------------


def sents(*spans: tuple[float, float]) -> list[Sentence]:
    return [
        Sentence(id=k, s=s, e=e, text=f"r{k}", words=(k, k))
        for k, (s, e) in enumerate(spans, start=1)
    ]


def test_puna_pokrivenost():
    cov, gaps = coverage(sents((0.0, 10.0)), 10.0)
    assert cov == 100.0
    assert gaps == []


def test_polovicna_pokrivenost():
    cov, _ = coverage(sents((0.0, 5.0)), 10.0)
    assert cov == 50.0


def test_praznina_u_sredini():
    cov, gaps = coverage(sents((0.0, 54.18), (61.60, 90.0)), 90.0)
    assert gaps == [(54.18, 61.6)]
    assert cov == pytest.approx(100.0 * (54.18 + 28.4) / 90.0)


def test_praznina_na_kraju_se_prijavljuje():
    _, gaps = coverage(sents((0.0, 79.74)), 90.14)
    assert gaps == [(79.74, 90.14)]


def test_praznina_na_pocetku_se_prijavljuje():
    _, gaps = coverage(sents((5.0, 20.0)), 20.0)
    assert gaps == [(0.0, 5.0)]


def test_kratke_pauze_se_ne_prijavljuju():
    _, gaps = coverage(sents((0.0, 10.0), (11.5, 20.0)), 20.0)
    assert gaps == []


def test_prag_praznine_je_podesiv():
    _, gaps = coverage(sents((0.0, 10.0), (11.5, 20.0)), 20.0, min_gap=1.0)
    assert gaps == [(10.0, 11.5)]


def test_preklopljene_recenice_se_ne_broje_dvaput():
    cov, _ = coverage(sents((0.0, 6.0), (4.0, 10.0)), 10.0)
    assert cov == 100.0


def test_neuredjen_redosled():
    cov, gaps = coverage(sents((60.0, 90.0), (0.0, 50.0)), 90.0)
    assert gaps == [(50.0, 60.0)]
    assert cov == pytest.approx(100.0 * 80.0 / 90.0)


def test_bez_recenica():
    cov, gaps = coverage([], 90.0)
    assert cov == 0.0
    assert gaps == [(0.0, 90.0)]


def test_nulto_trajanje_ne_deli_nulom():
    assert coverage(sents((0.0, 1.0)), 0.0) == (0.0, [])
