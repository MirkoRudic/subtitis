"""Reči bez zvuka iza sebe (krug 5). Vremena su iz merenja, reči izmišljene."""

from __future__ import annotations

from subtitis import hallucination as h
from subtitis.models import Sentence, Word


def w(i, s, e, t, p=0.9):
    return Word(i=i, s=s, e=e, w=t, p=p)


# Radionica draft 1, poslednji tokeni (vremena iz words.sr.json; reči izmišljene)
PRIMER = [
    w(93, 30.48, 31.32, "Otvorena", 0.34), w(94, 31.32, 32.16, "vrata", 0.98),
    w(95, 32.16, 32.26, ",", 0.98), w(96, 32.26, 32.26, "letnja", 0.87),
    w(97, 32.26, 32.26, "radionica", 0.98), w(98, 32.26, 32.26, "mladih", 0.99),
    w(99, 32.26, 32.26, ","), w(100, 32.26, 32.26, "OKC", 0.0),
    w(101, 32.26, 32.26, ","), w(102, 32.26, 32.26, "Divčibare", 0.2),
    w(103, 32.26, 32.26, ","), w(104, 32.26, 32.26, "Srbija", 0.23),
]


def test_niz_iz_glosara():
    (run,) = h.zero_runs(PRIMER)
    assert [x.w for x in run] == ["letnja", "radionica", "mladih", "OKC", "Divčibare", "Srbija"]
    assert "32.26–32.26 s" in h.run_line(run)


def test_verovatnoca_ne_hvata_ono_sto_trajanje_hvata():
    """`radionica` 0,98, `mladih` 0,99 — sigurne, a izmišljene."""
    (run,) = h.zero_runs(PRIMER)
    assert max(x.p for x in run) >= 0.98


def test_pojedinacna_nulta_rec_nije_niz():
    """Proba B: `Top 10 najboljih uvijeka na svijetu` — `10` stoji samo."""
    words = [
        w(0, 30.00, 30.54, "Top"), w(1, 30.54, 30.54, "10"), w(2, 30.54, 31.24, "najboljih"),
        w(3, 31.24, 32.04, "uvijeka"), w(4, 32.04, 32.04, "na"), w(5, 32.04, 32.04, "svijetu"),
    ]
    runs = h.zero_runs(words)
    assert [[x.w for x in r] for r in runs] == [["na", "svijetu"]]


def test_interpunkcija_nije_rec_i_ne_prekida_niz():
    words = [w(0, 1, 1, "a"), w(1, 1, 1, ","), w(2, 1, 1, "b"), w(3, 1, 1.5, "c")]
    assert [[x.w for x in r] for r in h.zero_runs(words)] == [["a", "b"]]
    assert h.zero_runs([w(0, 1, 1, "."), w(1, 1, 1, ",")]) == []


def test_kratke_ali_ne_nulte_reci_nisu_niz():
    """Brz govor: reči od tačno 0,020 i 0,060 s nisu bez zvuka (vremena sa
    urednog snimka iz merenja; reči izmišljene — bez odlomaka pravog govora)."""
    words = [w(0, 67.86, 67.88, "x"), w(1, 67.88, 67.94, "y"), w(2, 67.94, 68.0, "z")]
    assert h.zero_runs(words) == []


def test_oznaka_ide_na_recenicu_koja_sadrzi_niz_i_ne_dira_low_confidence():
    sents = [
        Sentence(id=1, s=30.48, e=32.16, text="Otvorena vrata,", words=(93, 95), flags=["low_confidence"]),
        Sentence(id=2, s=32.26, e=32.26, text="letnja radionica mladih, OKC, Divčibare, Srbija",
                 words=(96, 104)),
    ]
    ids = h.flag_sentences(sents, h.zero_runs(PRIMER))
    assert ids == [2]
    assert sents[1].flags == [h.HALLUCINATED_FLAG]
    assert sents[0].flags == ["low_confidence"]
