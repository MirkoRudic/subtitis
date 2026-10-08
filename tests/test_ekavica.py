"""Testovi ijekavica -> ekavica.

Negativni primeri su ovde važniji od pozitivnih: pravilo „ije/je -> e" bi
prošlo sve pozitivne testove i uništilo pola rečnika.
"""

from __future__ import annotations

import pytest
from conftest import make_words

from subtitis.ekavica import (
    ANY,
    EXACT,
    PREFIX,
    convert_text,
    convert_word,
    convert_words,
    is_protected,
)

# ==========================================================================
# NEGATIVNI PRIMERI — ništa od ovoga se ne sme dirati
# ==========================================================================

NEGATIVE = [
    # iz zahteva
    "jesen", "jedan", "jezik", "jeftino", "prijem", "sjajno", "Jelena",
    # strano ime: `mj` nije ijekavizam
    "Mjanmar", "Mjanmara", "Mjanmaru",
    # izvedenice istih
    "jeseni", "jesenji", "jedna", "jedno", "jednom", "jezika", "jeziku",
    "jezički", "jeftiniji", "prijema", "prijemni", "sjaj", "sjajan",
    "Jelenu", "Jelenin",
    # opasne po pojedinačna pravila
    "odjek", "odjeci",        # `djec` bi ih pokvario da je ANY
    "želje", "želja", "željama",  # imenica; `željet`/`željel` je glagol
    "dijeta", "dijete_ishrana".replace("_ishrana", "tni"),  # dijetni
    "prijatelj", "prijateljstvo",
    "radio", "studio", "video",   # `dio` -> `deo` samo kao cela reč
    "umio", "umivaonik",          # umio se
    # obične ekavske reči koje već jesu ispravne
    "jer", "je", "jesu", "jednostavno", "jedinstven", "medij", "serija",
]


@pytest.mark.parametrize("word", NEGATIVE)
def test_ne_dira_se(word):
    assert convert_word(word) == word


def test_strano_ime_sa_mj_je_zasticeno():
    """Mjanmar nije ijekavizam. Nijedno današnje pravilo ga ne pogađa —
    zaštita je brana za slučaj da neko doda širu osnovu tipa `mje`."""
    for w in ("Mjanmar", "Mjanmara", "Mjanmaru", "mjanmarski"):
        assert convert_word(w) == w


def test_dio_se_menja_samo_kao_cela_rec():
    assert convert_word("dio") == "deo"
    assert convert_word("Dio") == "Deo"
    for w in ("radio", "studio", "video", "audio", "dionica", "dioba"):
        assert convert_word(w) == w


def test_negativni_primeri_iz_zahteva_poimence():
    for w in ("jesen", "jedan", "jezik", "jeftino", "prijem", "sjajno",
              "Jelena", "Mjanmar"):
        assert convert_word(w) == w, w


# ==========================================================================
# POZITIVNI PRIMERI
# ==========================================================================

POSITIVE = [
    # iz testnog snimka
    ("posjeduju", "poseduju"),
    ("ovdje", "ovde"),
    # tražena lista
    ("ondje", "onde"),
    ("gdje", "gde"),
    ("djeca", "deca"),
    ("dijete", "dete"),
    ("vrijeme", "vreme"),
    ("mjesto", "mesto"),
    ("mjesec", "mesec"),
    ("vjera", "vera"),
    ("vjeruje", "veruje"),
    ("svijet", "svet"),
    ("tijelo", "telo"),
    ("lijep", "lep"),
    ("bijel", "bel"),
    ("cvijet", "cvet"),
    ("rijeka", "reka"),
    ("snijeg", "sneg"),
    ("mlijeko", "mleko"),
    ("sjeme", "seme"),
    ("sjediti", "sediti"),
    ("sjetiti", "setiti"),
    ("htjeti", "hteti"),
    ("željeti", "želeti"),
    ("vidjeti", "videti"),
    ("živjeti", "živeti"),
    ("razumjeti", "razumeti"),
    ("posjeduje", "poseduje"),
    ("prije", "pre"),
    ("poslije", "posle"),
    ("uvijek", "uvek"),
    ("nedjelja", "nedelja"),
    ("srijeda", "sreda"),
    ("pjesma", "pesma"),
    ("pjeva", "peva"),
    ("čovjek", "čovek"),
    ("dvije", "dve"),
]


@pytest.mark.parametrize("jek,ek", POSITIVE)
def test_osnovni_oblici(jek, ek):
    assert convert_word(jek) == ek


PADEZI_I_GLAGOLI = [
    # padeži
    ("djece", "dece"), ("djeci", "deci"), ("djecu", "decu"),
    ("djeteta", "deteta"), ("djetetu", "detetu"),
    ("vremena", "vremena"),          # isto u oba narečja
    ("mjesta", "mesta"), ("mjestu", "mestu"), ("umjesto", "umesto"),
    ("mjeseca", "meseca"), ("mjesečni", "mesečni"),
    ("vjere", "vere"), ("vjerovatno", "verovatno"), ("povjerenje", "poverenje"),
    ("svijeta", "sveta"), ("svjetski", "svetski"), ("svjetlo", "svetlo"),
    ("tijela", "tela"), ("tjelesni", "telesni"),
    ("lijepa", "lepa"), ("lijepo", "lepo"), ("ljepši", "lepši"),
    ("ljepota", "lepota"),
    ("bijela", "bela"), ("bjelina", "belina"),
    ("cvijeta", "cveta"), ("cvjetni", "cvetni"),
    ("rijeke", "reke"), ("riječ", "reč"), ("riječi", "reči"),
    ("snijega", "snega"), ("snježni", "snežni"),
    ("mlijeka", "mleka"), ("mliječni", "mlečni"),
    ("sjemena", "semena"), ("sjemenka", "semenka"),
    ("nedjelje", "nedelje"), ("srijedom", "sredom"),
    ("pjesme", "pesme"), ("pjevati", "pevati"), ("pjevač", "pevač"),
    ("čovjeka", "čoveka"), ("čovječanstvo", "čovečanstvo"),
    # glagolski oblici
    ("vidjela", "videla"), ("vidjeli", "videli"), ("vidio", "video"),
    ("živjela", "živela"), ("živio", "živeo"),
    ("htjela", "htela"), ("htjeli", "hteli"), ("htio", "hteo"),
    ("željela", "želela"), ("željeli", "želeli"), ("želio", "želeo"),
    ("razumio", "razumeo"), ("razumjela", "razumela"),
    ("sjedio", "sedio"), ("sjednica", "sednica"), ("sjedište", "sedište"),
    ("sjećanje", "sećanje"), ("osjećaj", "osećaj"), ("osjetiti", "osetiti"),
    ("posjeta", "poseta"), ("posjedovati", "posedovati"),
    # prefiksirani glagoli — zato `vidje` mora biti ANY, ne PREFIX
    ("predvidjeti", "predvideti"), ("uvidjeti", "uvideti"),
    ("umjetnost", "umetnost"), ("umjetnik", "umetnik"),
    # ostalo visoke frekvencije
    ("djevojka", "devojka"), ("djelo", "delo"), ("dijeliti", "deliti"),
    ("primjer", "primer"), ("namjera", "namera"), ("promjena", "promena"),
    ("mjenjačnica", "menjačnica"), ("mjenjati", "menjati"), ("izmjena", "izmena"),
    ("cijeli", "celi"), ("vrijednost", "vrednost"), ("smjer", "smer"),
    ("negdje", "negde"), ("nigdje", "nigde"), ("svugdje", "svugde"),
    ("prijedlog", "predlog"), ("prijevoz", "prevoz"),
]


@pytest.mark.parametrize("jek,ek", PADEZI_I_GLAGOLI)
def test_padezni_i_glagolski_oblici(jek, ek):
    assert convert_word(jek) == ek


# ==========================================================================
# Velika slova
# ==========================================================================


def test_cuva_veliko_pocetno_slovo():
    assert convert_word("Ovdje") == "Ovde"
    assert convert_word("Vrijeme") == "Vreme"
    assert convert_word("Čovjek") == "Čovek"


def test_cuva_sva_velika_slova():
    assert convert_word("OVDJE") == "OVDE"
    assert convert_word("VRIJEME") == "VREME"


def test_jedno_slovo_nije_verzal():
    assert convert_word("A") == "A"


# ==========================================================================
# Interpunkcija i granični slučajevi
# ==========================================================================


def test_interpunkcija_prolazi_netaknuta():
    for tok in (".", ",", "…", "?!", "", "123", "2024."):
        assert convert_word(tok) == tok


def test_convert_text():
    assert (
        convert_text("Ovdje posjeduju lijepo mjesto")
        == "Ovde poseduju lepo mesto"
    )


def test_convert_text_ne_dira_negativne():
    text = "Jelena je u jesen naučila jedan jezik, prijem je bio sjajno"
    assert convert_text(text) == text


# ==========================================================================
# Primena na words.json
# ==========================================================================


def test_convert_words_menja_na_mestu_i_prijavljuje():
    words = make_words("Ovdje posjeduju kucu.")
    changes = convert_words(words)
    assert [w.w for w in words] == ["Ovde", "poseduju", "kucu", "."]
    assert len(changes) == 2
    assert changes[0].before == "Ovdje" and changes[0].after == "Ovde"
    assert changes[0].index == 0
    assert changes[1].index == 1


def test_convert_words_cuva_vremena_i_indekse():
    words = make_words("Ovdje je lijepo.")
    before = [(w.i, w.s, w.e, w.p) for w in words]
    convert_words(words)
    assert [(w.i, w.s, w.e, w.p) for w in words] == before


def test_convert_words_bez_izmena():
    words = make_words("Ovde je lepo.")
    assert convert_words(words) == []


def test_change_ima_citljiv_ispis():
    words = make_words("Ovdje.")
    ch = convert_words(words)[0]
    assert "Ovdje" in str(ch) and "Ovde" in str(ch)


# ==========================================================================
# Ceo lanac: words -> ekavica -> sentences -> cue-ovi
# ==========================================================================


def test_sentences_i_srt_su_cisti(profil):
    """Konverzija pre segmentacije znači da ništa nizvodno ne vidi ijekavicu."""
    from subtitis.linebreak import build_cues
    from subtitis.segment import segment
    from subtitis.srt import render

    words = make_words(
        "Oni posjeduju kucu ovdje. Ovdje je uvijek lijepo vrijeme."
    )
    convert_words(words)
    sentences = segment(words)
    tekst = " ".join(s.text for s in sentences)
    for jek in ("posjeduju", "ovdje", "Ovdje", "uvijek", "lijepo", "vrijeme"):
        assert jek not in tekst
    assert "poseduju" in tekst and "Ovde" in tekst

    srt = render(build_cues(sentences, words, profil, language="sr"))
    assert "vreme" in srt and "vrijeme" not in srt


def test_konfiguracija_podrazumevano_ukljucuje_za_sr():
    from subtitis.config import force_ekavica_default

    assert force_ekavica_default("sr") is True
    assert force_ekavica_default("en") is False
    assert force_ekavica_default("fr") is False


# ==========================================================================
# Doslednost tabela
# ==========================================================================


def test_nijedna_osnova_nije_istovremeno_zasticena():
    for table in (EXACT, PREFIX, ANY):
        for stem in table:
            assert not is_protected(stem), stem


def test_zamene_zaista_nesto_menjaju():
    for table in (EXACT, PREFIX, ANY):
        for jek, ek in table.items():
            assert jek != ek, jek


def test_rezultat_konverzije_je_stabilan():
    """Druga primena ne sme ništa da promeni."""
    for jek, _ in POSITIVE + PADEZI_I_GLAGOLI:
        once = convert_word(jek)
        assert convert_word(once) == once, jek
