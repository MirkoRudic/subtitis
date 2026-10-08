"""Testovi prelamanja."""

from __future__ import annotations

import pytest
from conftest import make_words

from subtitis.linebreak import (
    ALIGNMENT_DEGRADED_FLAG,
    GENERIC_RULES,
    LANGUAGE_RULES,
    MIN_ALIGNED_TOKEN_SHARE,
    align_text_to_words,
    apply_timing,
    build_cues,
    layout_sentence,
    rules_for,
)
from subtitis.models import Cue, Profile
from subtitis.segment import segment


def all_lines(cues):
    return [l for c in cues for l in c.lines]


# --- Osnovna ograničenja ---------------------------------------------------


def test_kratka_recenica_je_jedan_red(profil):
    cues = layout_sentence(make_words("Dobar dan."), profil)
    assert len(cues) == 1
    assert cues[0].lines == ["Dobar dan."]


def test_nijedan_red_ne_prelazi_max_chars(profil):
    text = (
        "Onda smo seli i razgovarali otvoreno o svemu sto nas je muvilo "
        "tih dana, jer je bilo jasno da moramo da promenimo pristup."
    )
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    for line in all_lines(cues):
        assert len(line) <= profil.max_chars_per_line, line


def test_najvise_max_lines_po_cue(profil):
    text = "Prva recenica koja je dovoljno duga da mora da se prelomi na vise redova i cue-ova."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    for c in cues:
        assert 1 <= len(c.lines) <= profil.max_lines


def test_uzi_profil_daje_krace_redove():
    text = "Onda smo seli i razgovarali otvoreno o svemu sto nas je mucilo tih dana."
    words = make_words(text, dur_per_char=0.08)
    tv = Profile(name="tv", max_chars_per_line=37, max_cps=17, min_duration=1.0, max_duration=6.0, min_gap=0.12)
    for line in all_lines(layout_sentence(words, tv)):
        assert len(line) <= 37


# --- Vremena ---------------------------------------------------------------


def test_vremena_dolaze_iz_reci_a_ne_iz_proporcije(profil):
    words = make_words("Prva rec pa duga pauza pa jos reci ovde", dur_per_char=0.1)
    cues = layout_sentence(words, profil)
    assert cues[0].s == words[0].s
    assert cues[-1].e == words[-1].e


def test_svaki_cue_pocinje_i_zavrsava_na_granici_reci(profil):
    words = make_words(
        "Ovo je dovoljno dugacka recenica da se sigurno prelomi u vise cue-ova jer traje predugo",
        dur_per_char=0.12,
    )
    starts = {w.s for w in words}
    ends = {w.e for w in words}
    for c in layout_sentence(words, profil):
        assert c.s in starts and c.e in ends


# --- Lingvističke kazne ----------------------------------------------------


def _line_pairs(cues):
    """(kraj gornjeg reda, početak donjeg reda) za svaki prelom."""
    pairs = []
    for c in cues:
        for a, b in zip(c.lines, c.lines[1:]):
            pairs.append((a.split()[-1].lower(), b.split()[0].lower()))
    return pairs


def test_predlog_se_ne_odvaja_od_imenice(profil):
    text = "Razgovarali smo veoma dugo i otvoreno o svemu sto se dogodilo u toku prosle godine."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    for last, _ in _line_pairs(cues):
        assert last.strip(",.") not in {"u", "o", "na", "za", "sa", "od", "do"}, last


def test_enklitika_ne_zavrsava_red(profil):
    text = "Marko Petrovic iz Novog Sada je dosao na sastanak tacno u devet sati ujutru."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    for last, first in _line_pairs(cues):
        assert last not in {"je", "su", "bi", "ce", "cu"}, last
        assert first not in {"je", "su", "bi"}, first


def test_prelom_preferira_zarez(profil):
    # 49 znakova: mora u dva reda, ali staje u jedan cue
    text = "Kada smo zavrsili sa poslom, otisli smo kuci."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    assert len(cues) == 1 and len(cues[0].lines) == 2
    assert cues[0].lines[0].endswith(","), cues[0].lines


def test_nema_usamljene_reci_u_redu(profil):
    text = "Ovo je recenica srednje duzine koja se prelama u dva reda bez viska."
    for c in layout_sentence(make_words(text, dur_per_char=0.08), profil):
        if len(c.lines) > 1:
            for line in c.lines:
                assert len(line.split()) > 1, c.lines


def test_nema_bliceva_kratkih_cue_ova(profil):
    """Cue od par znakova otcepljen od nastavka je greška; cela kratka rečenica nije."""
    text = "Bilo je jasno da moramo da promenimo pristup, i to se pokazalo kao dobra odluka."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    assert len(cues) > 1
    for c in cues:
        assert len(" ".join(c.lines)) >= 20, c.lines


def test_kratka_cela_recenica_sme_biti_kratak_cue(profil):
    cues = layout_sentence(make_words("Da."), profil)
    assert len(cues) == 1 and cues[0].lines == ["Da."]


def test_redovi_su_priblizno_uravnotezeni(profil):
    text = "Danas cemo pricati o jednoj veoma vaznoj temi."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), profil)
    for c in cues:
        if len(c.lines) == 2:
            assert abs(len(c.lines[0]) - len(c.lines[1])) <= 14, c.lines


def test_interpunkcija_ne_pocinje_red(profil):
    text = "Prvo smo pricali o poslu, zatim o porodici, pa onda o planovima za leto."
    for line in all_lines(layout_sentence(make_words(text, dur_per_char=0.08), profil)):
        assert line[0] not in ",.!?…:;"


# --- Jezičke tabele funkcijskih reči --------------------------------------


def breaks(cues, rules):
    """(poslednja reč gornjeg reda, prva reč donjeg) za SVE prelome — i one
    unutar cue-a i one između cue-ova."""
    from subtitis.linebreak import _norm, _norm_keep_apostrophe

    lines = [l for c in cues for l in c.lines]
    out = []
    for a, b in zip(lines, lines[1:]):
        out.append(
            (_norm(a.split()[-1]), _norm_keep_apostrophe(a.split()[-1]), _norm(b.split()[0]))
        )
    return out


def assert_funkcijske_reci_ne_lome(text, lang, profil, dur=0.075):
    rules = LANGUAGE_RULES[lang]
    words = make_words(text, dur_per_char=dur)
    cues = build_cues(segment(words), words, profil, language=lang)
    assert len(all_lines(cues)) > 1, f"nije se prelomilo: {text}"
    for last, last_apo, first in breaks(cues, rules):
        assert last not in rules.no_line_end, f"{lang}: red se završava na {last!r}"
        assert last_apo not in rules.no_line_end, f"{lang}: red se završava na {last_apo!r}"
        assert first not in rules.no_line_start, f"{lang}: red počinje sa {first!r}"
        if rules.keep_elisions:
            assert not last_apo.endswith("'"), f"{lang}: red se završava elizijom {last_apo!r}"
    return cues


#: Svaka rečenica je izabrana tako da bez pravila jezika (opšta pravila)
#: prelom padne baš na funkcijsku reč — u komentaru je taj loš rez. Ako
#: izmena rečenice ili profila to pomeri, test više ne meri ništa.
EN_PRIMERI = [
    # prelom posle `the` (regresija iz prevedenog SRT-a)
    "Slide the cover off and remove the old battery from the camera slowly.",  # the / old
    "Tomorrow there will be rain in the morning and strong wind in the evening.",  # the / morning
    "Please check that the power cable is plugged into the back of the recorder.",  # is / plugged
]

FR_PRIMERI = [
    "Fixez les deux pieds courts sous la table et serrez bien toutes les vis.",  # la / table
    "Demain matin, il fera froid et venteux sur la côte dans le nord du pays.",  # sur / la
    "Tournez le bouton jusqu'à ce que la petite lampe sur l'appareil s'éteigne.",  # la / petite
]

SR_PRIMERI = [
    "Okrecite dugme polako sve dok se mala lampica na prednjoj strani ne ugasi.",  # se / mala
    "Kabl za napajanje uvek ukljucite u uticnicu tek posle povezivanja kamere.",  # u / uticnicu
    "Sutra ujutru ce biti hladno i vetrovito na obali i u planinama na severu.",  # na / obali
]


@pytest.mark.parametrize("text", EN_PRIMERI)
def test_engleske_funkcijske_reci(text, profil):
    assert_funkcijske_reci_ne_lome(text, "en", profil)


@pytest.mark.parametrize("text", FR_PRIMERI)
def test_francuske_funkcijske_reci(text, profil):
    assert_funkcijske_reci_ne_lome(text, "fr", profil)


@pytest.mark.parametrize("text", SR_PRIMERI)
def test_srpske_funkcijske_reci(text, profil):
    assert_funkcijske_reci_ne_lome(text, "sr", profil)


def test_clan_the_ne_zavrsava_red(profil):
    """Regresija: u prevedenom SRT-u je prelom pao posle `the`."""
    cues = assert_funkcijske_reci_ne_lome(EN_PRIMERI[0], "en", profil)
    assert not any(l.rstrip().endswith(" the") for l in all_lines(cues))


def test_francuska_elizija_ostaje_uz_sledecu_rec(profil):
    # bez pravila: `jusqu' / au bout`
    words = make_words("Il faut tourner le bouton gris jusqu' au bout avant d' ouvrir le couvercle.")
    cues = build_cues(segment(words), words, profil, language="fr")
    for line in all_lines(cues):
        assert not line.rstrip().endswith("'"), line


def test_srpska_pravila_se_ne_primenjuju_na_engleski():
    """`je` je srpska enklitika, ali obična engleska reč nije u en tabeli."""
    assert "je" in LANGUAGE_RULES["sr"].no_line_start
    assert "je" not in LANGUAGE_RULES["en"].no_line_start
    assert "the" in LANGUAGE_RULES["en"].no_line_end
    assert "the" not in LANGUAGE_RULES["sr"].no_line_end


def test_tabele_pokrivaju_trazene_reci():
    en = LANGUAGE_RULES["en"].no_line_end
    for w in ("a", "an", "the", "in", "on", "at", "to", "of", "for", "with",
              "from", "by", "into", "about", "over", "under", "between",
              "through", "is", "are", "was", "were", "has", "have", "had",
              "will", "would", "can", "could", "should", "must", "do",
              "does", "did"):
        assert w in en, w
    fr = LANGUAGE_RULES["fr"].no_line_end
    for w in ("le", "la", "les", "un", "une", "des", "du", "de", "à", "dans",
              "sur", "pour", "avec", "par", "sans", "sous", "entre", "chez",
              "vers", "est", "sont", "était", "a", "ont", "avait", "sera",
              "peut", "doit", "l'", "d'", "qu'", "n'"):
        assert w in fr, w


def test_nepoznat_jezik_daje_upozorenje_i_opsta_pravila():
    poruke = []
    rules = rules_for("de", poruke.append)
    assert rules is GENERIC_RULES
    assert poruke and "nema jezičkih pravila" in poruke[0]
    assert "de" in poruke[0]


def test_nepoznat_jezik_i_dalje_lomi_po_duzini(profil):
    poruke = []
    text = "Dies ist ein ziemlich langer deutscher Satz der umgebrochen werden muss."
    words = make_words(text, dur_per_char=0.075)
    cues = build_cues(segment(words), words, profil, language="de", log=poruke.append)
    assert cues
    for line in all_lines(cues):
        assert len(line) <= profil.max_chars_per_line
    assert any("nema jezičkih pravila" in m for m in poruke)


def test_poznati_jezik_ne_dize_upozorenje(profil):
    poruke = []
    words = make_words("Dobar dan, hvala sto ste dosli.")
    build_cues(segment(words), words, profil, language="sr", log=poruke.append)
    assert not any("nema jezičkih pravila" in m for m in poruke)


def test_bcp47_oznaka_se_svodi_na_jezik():
    assert rules_for("sr-Latn") is LANGUAGE_RULES["sr"]
    assert rules_for("EN") is LANGUAGE_RULES["en"]


# --- Vremenska ograničenja profila ----------------------------------------


def test_min_duration_se_produzava(profil):
    cues = [Cue(s=1.0, e=1.2, lines=["Da."]), Cue(s=10.0, e=12.0, lines=["Dobro."])]
    apply_timing(cues, profil)
    assert cues[0].e - cues[0].s >= profil.min_duration - 1e-6


def test_min_gap_se_postuje(profil):
    cues = [Cue(s=0.0, e=2.0, lines=["Prvi."]), Cue(s=2.01, e=4.0, lines=["Drugi."])]
    apply_timing(cues, profil)
    assert cues[1].s - cues[0].e >= profil.min_gap - 1e-6


def test_produzavanje_ne_gazi_sledeci_cue(profil):
    cues = [Cue(s=1.0, e=1.1, lines=["Da."]), Cue(s=1.5, e=3.0, lines=["Dobro."])]
    apply_timing(cues, profil)
    assert cues[1].s - cues[0].e >= profil.min_gap - 1e-6


def test_indeksi_su_redom(profil):
    cues = [Cue(s=0.0, e=2.0, lines=["A"]), Cue(s=3.0, e=5.0, lines=["B"])]
    apply_timing(cues, profil)
    assert [c.index for c in cues] == [1, 2]


# --- Ceo lanac -------------------------------------------------------------


def test_build_cues_od_recenica(profil):
    words = make_words(
        "Dobar dan, hvala sto ste dosli. Danas cemo pricati o jednoj vaznoj temi.",
        dur_per_char=0.09,
    )
    cues = build_cues(segment(words), words, profil, language="sr")
    assert cues
    assert all(c.e > c.s for c in cues)
    assert [c.index for c in cues] == list(range(1, len(cues) + 1))
    for a, b in zip(cues, cues[1:]):
        assert b.s - a.e >= profil.min_gap - 1e-6


def test_cue_cuva_flagove_recenice(profil):
    words = make_words("Nesigurna recenica ovde je.")
    words[1].p = 0.3
    cues = build_cues(segment(words), words, profil, language="sr")
    assert all("low_confidence" in c.flags for c in cues)


# --- Poravnanje ručne ispravke --------------------------------------------


def test_poravnanje_neizmenjenog_teksta_vraca_original():
    words = make_words("Dobar dan, hvala.")
    out, share = align_text_to_words("Dobar dan, hvala.", words)
    assert out is words
    assert share == 1.0


def test_poravnanje_cuva_vremena_neizmenjenih_reci():
    words = make_words("Dobar dan, hvala sto ste dosli.")
    out, share = align_text_to_words("Dobar dan, hvala što ste došli.", words)
    assert [w.w for w in out] == ["Dobar", "dan", ",", "hvala", "što", "ste", "došli", "."]
    assert out[0].s == words[0].s
    assert out[-1].e == words[-1].e
    assert share > MIN_ALIGNED_TOKEN_SHARE


def test_poravnanje_prijavljuje_degradaciju():
    words = make_words("Dobar dan, hvala sto ste dosli.")
    _, share = align_text_to_words("Potpuno drugaciji sadrzaj bez ijedne zajednicke reci", words)
    assert share < MIN_ALIGNED_TOKEN_SHARE


def test_degradirano_poravnanje_dize_flag_i_logira(profil):
    words = make_words("Dobar dan, hvala sto ste dosli.")
    sents = segment(words)
    sents[0].text = "Potpuno drugaciji sadrzaj bez ijedne zajednicke reci ovde"
    poruke: list[str] = []
    cues = build_cues(sents, words, profil, language="sr", log=poruke.append)
    assert ALIGNMENT_DEGRADED_FLAG in sents[0].flags
    assert all(ALIGNMENT_DEGRADED_FLAG in c.flags for c in cues)
    assert poruke and "degradirano" in poruke[0]


def test_mala_ispravka_ne_dize_flag(profil):
    words = make_words("Dobar dan, hvala sto ste dosli.")
    sents = segment(words)
    sents[0].text = "Dobar dan, hvala što ste došli."
    poruke: list[str] = []
    cues = build_cues(sents, words, profil, language="sr", log=poruke.append)
    assert ALIGNMENT_DEGRADED_FLAG not in sents[0].flags
    assert not poruke
    assert all(ALIGNMENT_DEGRADED_FLAG not in c.flags for c in cues)


def test_rucna_ispravka_stize_do_srt_a(profil):
    words = make_words("Dobar dan, hvala sto ste dosli.")
    sents = segment(words)
    sents[0].text = "Dobar dan, hvala što ste došli."
    cues = build_cues(sents, words, profil, language="sr")
    assert "došli" in " ".join(all_lines(cues))


# ==========================================================================
# Strana veznika i determinatori (en, fr) — izmereno na prvom SRT poslu
# ==========================================================================

from subtitis.linebreak import (  # noqa: E402
    PENALTY_BREAK_AFTER_CONJUNCTION,
    PENALTY_SPLIT_PREPOSITION,
    _Span,
    best_lines,
    rules_for,
)
from subtitis.models import Profile as _Profile, Word as _Word  # noqa: E402
from subtitis.segment import tokenize_text as _tok  # noqa: E402


def _span(text: str, lang: str) -> tuple[list[str], _Span]:
    toks = _tok(text)
    ws = [_Word(i=k, s=0.0, e=0.0, w=t, p=1.0) for k, t in enumerate(toks)]
    return toks, _Span(ws, _Profile(name="t"), rules_for(lang))


@pytest.mark.parametrize("lang,text,conj", [
    ("en", "The old camera is very slow and when I turn it on", "and"),
    ("fr", "Je suis content mais je suis fatigué", "mais"),
])
def test_en_fr_lome_ispred_veznika(lang, text, conj):
    toks, span = _span(text, lang)
    k = toks.index(conj)
    assert span.break_pen[k - 1] == PENALTY_BREAK_AFTER_CONJUNCTION  # ispred
    assert span.break_pen[k] > PENALTY_BREAK_AFTER_CONJUNCTION       # iza


def test_srpski_i_dalje_lomi_iza_veznika():
    toks, span = _span("Bio sam umoran i onda sam otišao kući", "sr")
    k = toks.index("i")
    assert span.break_pen[k] == PENALTY_BREAK_AFTER_CONJUNCTION
    assert span.break_pen[k - 1] > PENALTY_BREAK_AFTER_CONJUNCTION


@pytest.mark.parametrize("lang,text,det", [
    ("en", "I think their biggest problem will be the battery", "their"),
    ("en", "We went to see those beautiful mountains", "those"),
    ("fr", "Ils ont vu leurs amis dans cette belle ville", "leurs"),
    ("fr", "Ils ont vu leurs amis dans cette belle ville", "cette"),
])
def test_determinator_ne_zavrsava_red(lang, text, det):
    toks, span = _span(text, lang)
    assert span.break_pen[toks.index(det)] >= PENALTY_SPLIT_PREPOSITION


def test_that_nije_determinator_nego_veznik():
    """Češće veznik (`I hope that / in the years`) nego determinator — bez kazne."""
    toks, span = _span("I hope that in the years to come we meet", "en")
    assert span.break_pen[toks.index("that")] < PENALTY_SPLIT_PREPOSITION


def test_red_na_engleskom_pocinje_veznikom():
    prof = _Profile(name="t", max_chars_per_line=30)
    # bez pravila: `Hi, the old camera is very slow and / when ...`
    toks = _tok("Hi, the old camera is very slow and when I turn it on the screen goes dark.")
    ws = [_Word(i=k, s=0.0, e=0.0, w=t, p=1.0) for k, t in enumerate(toks)]
    lines = [" ".join(toks[a:b]) for a, b in best_lines(ws, prof, rules_for("en"))]
    assert not any(l.endswith(" and") for l in lines), lines


@pytest.mark.parametrize("lang,text,word", [
    ("en", "I wanted to explain that in winter we need more batteries", "that"),
    ("en", "I called those people who sold us the old camera", "who"),
    ("en", "We visited the town where my grandmother was born", "where"),
    ("en", "We took the bus which was very slow and old", "which"),
    ("fr", "Je connais les gens qui habitent dans cette maison", "qui"),
    ("fr", "Voici le livre dont je parlais avec toi hier", "dont"),
    ("fr", "Il pleuvait lorsque nous sommes arrivés à la gare", "lorsque"),
])
def test_zavisni_veznici_se_lome_ispred(lang, text, word):
    toks, span = _span(text, lang)
    k = toks.index(word)
    assert span.break_pen[k - 1] == PENALTY_BREAK_AFTER_CONJUNCTION
    assert span.break_pen[k] > PENALTY_BREAK_AFTER_CONJUNCTION


def test_parce_que_se_ne_razdvaja():
    toks, span = _span("Je suis resté à la maison parce que il pleuvait", "fr")
    k = toks.index("parce")
    assert span.break_pen[k - 1] == PENALTY_BREAK_AFTER_CONJUNCTION  # / parce que
    assert span.break_pen[k] >= PENALTY_SPLIT_PREPOSITION             # parce / que


# --- Jednoredni profil (krug 6: `digital`, od kruga 8: `web`) ----------------


@pytest.fixture
def digital():
    """Jednoredni prelom; ime fixture-a je iz kruga 6, profil se sada zove `web`."""
    from subtitis import config as cfg

    return cfg.load_profile("web")


def _cue_pairs(cues):
    """(kraj cue-a, početak sledećeg) — u jednom redu je prelom između cue-ova."""
    return [
        (a.lines[-1].split()[-1].lower(), b.lines[0].split()[0].lower())
        for a, b in zip(cues, cues[1:])
    ]


def test_digital_jedan_red_do_20_znakova(digital):
    text = "Onda smo seli i razgovarali otvoreno o svemu sto nas je mucilo tih dana."
    cues = layout_sentence(make_words(text, dur_per_char=0.08), digital)
    assert len(cues) > 2
    for c in cues:
        assert len(c.lines) == 1 and len(c.lines[0]) <= 20, c.lines


def test_digital_enklitika_i_predlog_i_dalje_vezu(digital):
    """Kazna za enklitiku i predlog ostaje: u jednom redu je taj rez još vidljiviji."""
    for text in (
        "Marko Petrovic iz Novog Sada je dosao na sastanak tacno u devet sati ujutru.",
        "Razgovarali smo veoma dugo i otvoreno o svemu sto se dogodilo u toku prosle godine.",
    ):
        cues = layout_sentence(make_words(text, dur_per_char=0.08), digital)
        for last, first in _cue_pairs(cues):
            assert last.strip(",.") not in {"u", "o", "na", "za", "sa", "od", "do", "je", "su"}, last
            assert first not in {"je", "su", "bi"}, first


def _dspan(text, profile):
    from subtitis.linebreak import _Span

    return _Span(make_words(text), profile, LANGUAGE_RULES["sr"])


def test_kratak_red_se_kaznjava_po_znaku(digital):
    from subtitis.linebreak import PENALTY_BELOW_MIN_CHARS

    span = _dspan("Da li ces doci sutra uvece?", digital)
    # "Da li" = 5 znakova, a ostatak rečenice je dovoljno dug
    assert span.line_cost(0, 2, False) == 5 * PENALTY_BELOW_MIN_CHARS
    assert span.line_cost(0, 3, False) == 1 * PENALTY_BELOW_MIN_CHARS  # "Da li ces" = 9


def test_kratka_cela_recenica_nije_kaznjena(digital):
    """Izuzetak važi samo kad je CELA rečenica kraća od granice (krug 7)."""
    assert _dspan("Da.", digital).line_cost(0, 2, False) == 0
    assert _dspan("Ne znam.", digital).line_cost(0, 3, False) == 0


def test_kratak_ostatak_duze_recenice_jeste_kaznjen(digital):
    """Krug 6 je štedeo kratak ostatak — a on je kratak zbog lošeg reza pre njega."""
    from subtitis.linebreak import PENALTY_BELOW_MIN_CHARS

    span = _dspan("Čao, Nikola, šta ima?", digital)
    # "ima?" = 4 znaka, ostatak je kratak, ali rečenica nije
    assert span.line_cost(span.n - 2, span.n, False) == 6 * PENALTY_BELOW_MIN_CHARS


def test_bez_donje_granice_nema_kazne(profil):
    assert profil.min_chars_per_line == 0
    assert _dspan("Da li ces doci?", profil).line_cost(0, 2, False) == 0


def test_usamljena_rec_se_ne_kaznjava_u_jednom_redu(digital):
    """Reč sama u jednorednom cue-u je nastavak, ne visi ispod drugog reda."""
    from subtitis.linebreak import PENALTY_ORPHAN_WORD, _best_lines

    bez_granice = digital.model_copy(update={"min_chars_per_line": 0})
    jedan = _dspan("Sutradan smo krenuli rano.", bez_granice)
    assert _best_lines(jedan, 0, 1)[0] == 0
    dva = _dspan("Sutradan smo krenuli rano.", bez_granice.model_copy(update={"max_lines": 2}))
    assert _best_lines(dva, 0, 1)[0] == PENALTY_ORPHAN_WORD


def test_prekratki_cue_se_prijavljuje(digital):
    from subtitis.linebreak import too_short

    cues = [
        Cue(s=0.0, e=0.3, lines=["Da."]),
        Cue(s=0.383, e=1.5, lines=["Dobro, idemo."]),
    ]
    out = too_short(apply_timing(cues, digital), digital)
    assert [c.lines for c in out] == [["Da."]], "prvi nema mesta do drugog"


# --- Negacija (krug 7) -------------------------------------------------------


def _pen_after(text, word, lang="sr"):
    from subtitis.linebreak import _Span

    words = make_words(text)
    span = _Span(words, Profile(name="t"), LANGUAGE_RULES[lang])
    k = next(i for i, w in enumerate(words) if w.w.lower() == word)
    return span.break_pen[k]


def test_ne_ne_zavrsava_red_kao_predlog():
    from subtitis.linebreak import PENALTY_SPLIT_PREPOSITION

    assert _pen_after("zato što mi ne smemo da kasnimo", "ne") >= PENALTY_SPLIT_PREPOSITION
    # isto kao predlog
    assert _pen_after("zato što mi ne smemo", "ne") == _pen_after("kabl sa punjača", "sa")


def test_ne_kao_cela_recenica_nije_problem(profil):
    cues = layout_sentence(make_words("Ne."), profil)
    assert [c.lines for c in cues] == [["Ne."]]


def test_francusko_ne_i_n_apostrof():
    from subtitis.linebreak import PENALTY_SPLIT_PREPOSITION

    assert _pen_after("je ne vois pas le câble", "ne", "fr") >= PENALTY_SPLIT_PREPOSITION
    # `n'ai` je jedan token — ne može se ni preseći
    assert "n'ai" in [w.w for w in make_words("je n'ai rien vu")]
    # a kad ga prepoznavanje da odvojeno, hvata ga pravilo o apostrofu
    from subtitis.linebreak import _Span
    from subtitis.models import Word

    words = [Word(i=k, s=k, e=k + 0.5, w=t, p=0.9) for k, t in enumerate(["je", "n'", "ai", "rien"])]
    span = _Span(words, Profile(name="t"), LANGUAGE_RULES["fr"])
    assert span.break_pen[1] >= PENALTY_SPLIT_PREPOSITION
