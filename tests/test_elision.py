"""Testovi spajanja razbijene francuske elizije.

Whisper vraća `j` + `'aimerais` kao dva tokena. Negativni primeri su ovde
važniji od pozitivnih: pravilo „spoji sve oko apostrofa" bi prošlo sve
pozitivne testove i pojelo navodnike, engleski genitiv i srpski tekst.
"""

from __future__ import annotations

import pytest
from conftest import make_words

from subtitis.asr import (
    ELIDED_STEMS,
    MAX_ELIDED_STEM,
    elision_language,
    merge_elisions,
)
from subtitis.models import Word


def words(*tokens: str) -> list[Word]:
    """Tokeni tačno onako kako ih ASR upisuje — bez ponovnog cepanja."""
    out: list[Word] = []
    t = 0.0
    for k, tok in enumerate(tokens):
        d = max(0.05, len(tok) * 0.06)
        out.append(Word(i=k, s=round(t, 3), e=round(t + d, 3), w=tok, p=0.9))
        t += d + 0.02
    return out


def text(ws: list[Word]) -> str:
    return " ".join(w.w for w in ws)


# ==========================================================================
# Oblici iz zahteva — svi moraju da se spoje
# ==========================================================================

COVERED = [
    ("j", "'aimerais", "j'aimerais"),
    ("d", "'autres", "d'autres"),
    ("l", "'ordinateur", "l'ordinateur"),
    ("c", "'est", "c'est"),
    ("n", "'est", "n'est"),
    ("s", "'il", "s'il"),
    ("t", "'il", "t'il"),
    ("m", "'appelle", "m'appelle"),
    ("qu", "'il", "qu'il"),
    ("jusqu", "'à", "jusqu'à"),
    ("lorsqu", "'on", "lorsqu'on"),
    ("aujourd", "'hui", "aujourd'hui"),
]


@pytest.mark.parametrize("left,right,expected", COVERED)
def test_spaja_trazene_oblike(left: str, right: str, expected: str) -> None:
    ws = words(left, right)
    merges = merge_elisions(ws, "fr")
    assert len(merges) == 1
    assert text(ws) == expected


def test_veliko_slovo_ostaje() -> None:
    ws = words("C", "'est", "vrai")
    merge_elisions(ws, "fr")
    assert text(ws) == "C'est vrai"


def test_tipografski_apostrof() -> None:
    ws = words("j", "’ai")
    merge_elisions(ws, "fr")
    assert text(ws) == "j’ai"


def test_apostrof_na_kraju_prvog_tokena() -> None:
    """Whisper razbija i ovako: `j'` + `ai`."""
    ws = words("j'", "ai")
    merge_elisions(ws, "fr")
    assert text(ws) == "j'ai"


def test_apostrof_sa_obe_strane_ne_udvaja() -> None:
    ws = words("j'", "'ai")
    merge_elisions(ws, "fr")
    assert text(ws) == "j'ai"


# ==========================================================================
# NEGATIVNI PRIMERI — ništa od ovoga se ne sme spojiti
# ==========================================================================


def test_srpski_se_ne_dira() -> None:
    ws = words("j", "'ai", "31", "an")
    assert merge_elisions(ws, "sr") == []
    assert text(ws) == "j 'ai 31 an"


def test_bez_jezika_se_ne_dira() -> None:
    ws = words("j", "'ai")
    assert merge_elisions(ws, None) == []


def test_duga_osnova_van_spiska_ne_spaja() -> None:
    """`Paul` + `'ovo'` je navodnik, ne elizija."""
    ws = words("Paul", "'ovo")
    assert merge_elisions(ws, "fr") == []
    assert text(ws) == "Paul 'ovo"


def test_apostrof_bez_slova_iza_ne_spaja() -> None:
    ws = words("d", "'", "autres")
    assert merge_elisions(ws, "fr") == []


def test_bez_apostrofa_ne_spaja() -> None:
    """Dve kratke reči jedna do druge nisu elizija."""
    ws = words("de", "la", "rue")
    assert merge_elisions(ws, "fr") == []
    assert text(ws) == "de la rue"


def test_interpunkcija_izmedju_prekida_par() -> None:
    ws = words("j", ",", "'ai")
    assert merge_elisions(ws, "fr") == []


def test_cifra_nije_osnova() -> None:
    ws = words("26", "'ans")
    assert merge_elisions(ws, "fr") == []


# ==========================================================================
# Vremena, verovatnoća, indeksi
# ==========================================================================


def test_vreme_uzima_s_prvog_i_e_drugog() -> None:
    ws = words("j", "'aimerais")
    a, b = ws[0].s, ws[1].e
    merge_elisions(ws, "fr")
    assert ws[0].s == a
    assert ws[0].e == b


def test_verovatnoca_je_manja_od_dve() -> None:
    ws = words("j", "'aimerais")
    ws[1] = ws[1].model_copy(update={"p": 0.31})
    merge_elisions(ws, "fr")
    assert ws[0].p == 0.31


def test_indeksi_se_prenumerisu_bez_rupa() -> None:
    ws = words("Je", "m", "'appelle", "Mathieu", "j", "'ai", "31", "an")
    merges = merge_elisions(ws, "fr")
    assert len(merges) == 2
    assert [w.i for w in ws] == list(range(len(ws)))
    assert text(ws) == "Je m'appelle Mathieu j'ai 31 an"


def test_bez_spajanja_lista_ostaje_ista() -> None:
    ws = words("de", "la", "rue")
    before = [w.model_copy() for w in ws]
    merge_elisions(ws, "fr")
    assert ws == before


def test_uzastopne_elizije() -> None:
    ws = words("c", "'est", "d", "'autres")
    assert len(merge_elisions(ws, "fr")) == 2
    assert text(ws) == "c'est d'autres"


def test_log_zapis_ima_pre_i_posle() -> None:
    ws = words("j", "'aimerais")
    (m,) = merge_elisions(ws, "fr")
    assert m.before == "j 'aimerais"
    assert m.after == "j'aimerais"
    assert "j'aimerais" in str(m)


# ==========================================================================
# Tabele i granica
# ==========================================================================


def test_spisak_osnova_nema_kracih_od_granice() -> None:
    """Osnova kraća od granice u spisku je mrtvo slovo — granica je već hvata.

    `qu` je tačno na granici i sme da stoji, kao dokumentacija oblika `qu'il`.
    """
    assert all(len(s) >= MAX_ELIDED_STEM for s in ELIDED_STEMS)


def test_elision_language_samo_francuski() -> None:
    assert elision_language("fr")
    assert elision_language(" FR ")
    assert not elision_language("sr")
    assert not elision_language("en")
    assert not elision_language("")


# ==========================================================================
# Zašto ovo postoji: bez spajanja `linebreak` vidi dve reči
# ==========================================================================


def test_cue_ne_pocinje_sirocetom(profil) -> None:
    """`'espère` se nikad ne sme naći odvojen od svog `j`."""
    from subtitis.linebreak import build_cues
    from subtitis.segment import segment

    ws = words(*"et je remercie les techniciens pour tout le matériel prêté hier et".split())
    ws += words("j", "'espère", "pouvoir", "revenir", "dans", "cette", "grande", "salle")
    for k, w in enumerate(ws):  # vremena posle spajanja dve grupe
        ws[k] = w.model_copy(update={"i": k, "s": k * 0.5, "e": k * 0.5 + 0.45})

    merge_elisions(ws, "fr")
    cues = build_cues(segment(ws), ws, profil, language="fr")
    assert not any(line.startswith("'") for c in cues for line in c.lines)
