"""Testovi transliteracije — težište na digrafima preko granice morfeme."""

from __future__ import annotations

import pytest

from subtitis.translit import (
    convert,
    convert_all,
    detect_script,
    foreign_letters,
    is_biscriptal,
    script_violations,
    to_cyrillic,
    to_latin,
)

# --- Slučajevi iz specifikacije: digraf se MORA razdvojiti -----------------

SPLIT_CASES = [
    ("nadživeti", "надживети"),
    ("injekcija", "инјекција"),
    ("konjugacija", "конјугација"),
    ("podžanr", "поджанр"),
    ("tanjug", "танјуг"),
]


@pytest.mark.parametrize("lat,cyr", SPLIT_CASES)
def test_digraf_se_razdvaja(lat, cyr):
    assert to_cyrillic(lat) == cyr


def test_tanjug_velikim_slovom():
    assert to_cyrillic("Tanjug") == "Танјуг"
    assert to_cyrillic("TANJUG") == "ТАНЈУГ"


# --- Kontraprimeri: digraf je pravi i ostaje jedno slovo -------------------

KEEP_CASES = [
    ("odžak", "оџак"),
    ("odžačar", "оџачар"),
    ("konjak", "коњак"),
    ("kanjon", "кањон"),
    ("inje", "иње"),
    ("hodža", "хоџа"),
    ("narandža", "наранџа"),
    ("menadžer", "менаџер"),
    ("nadljudski", "надљудски"),
    ("ljubav", "љубав"),
    ("njegov", "његов"),
    ("konji", "коњи"),
]


@pytest.mark.parametrize("lat,cyr", KEEP_CASES)
def test_pravi_digraf_ostaje(lat, cyr):
    assert to_cyrillic(lat) == cyr


# --- Padeži i izvedenice ---------------------------------------------------


@pytest.mark.parametrize(
    "lat,cyr",
    [
        ("injekciju", "инјекцију"),
        ("injekcije", "инјекције"),
        ("reinjekcija", "реинјекција"),
        ("konjunktiv", "конјунктив"),
        ("konjunktura", "конјунктура"),
        ("nadživljavanje", "надживљавање"),
        ("nadživeo", "надживео"),
        ("podžupan", "поджупан"),
        ("odžvakati", "оджвакати"),
        ("vanjezički", "ванјезички"),
        ("anjon", "анјон"),
        ("injicirati", "инјицирати"),
    ],
)
def test_izvedenice(lat, cyr):
    assert to_cyrillic(lat) == cyr


# --- Velika slova i digrafi ------------------------------------------------


def test_veliko_slovo_digrafa():
    assert to_cyrillic("Džak") == "Џак"
    assert to_cyrillic("DŽAK") == "ЏАК"
    assert to_cyrillic("Ljubav") == "Љубав"
    assert to_cyrillic("NJEGOV") == "ЊЕГОВ"
    assert to_cyrillic("Nadživeti") == "Надживети"


def test_cir_u_lat_veliko():
    assert to_latin("Џак") == "Džak"
    assert to_latin("ЏАК") == "DŽAK"
    assert to_latin("Његов") == "Njegov"
    assert to_latin("ЊЕГОВ") == "NJEGOV"


# --- Osnovna 1:1 mapa ------------------------------------------------------


def test_cir_u_lat_recenica():
    assert (
        to_latin("Добар дан, хвала што сте дошли.")
        == "Dobar dan, hvala što ste došli."
    )


def test_lat_u_cir_recenica():
    assert (
        to_cyrillic("Dobar dan, hvala što ste došli.")
        == "Добар дан, хвала што сте дошли."
    )


def test_povratna_konverzija():
    for lat, cyr in SPLIT_CASES + KEEP_CASES:
        assert to_latin(cyr) == lat


# --- Detekcija pisma -------------------------------------------------------


def test_detekcija():
    assert detect_script("Dobar dan") == "Latn"
    assert detect_script("Добар дан") == "Cyrl"
    assert detect_script("Добар dan") == "mixed"
    assert detect_script("123 ...") == "Latn"


def test_normalizacija_mesanog_teksta():
    assert to_latin("Добар dan, хвала.") == "Dobar dan, hvala."


def test_convert():
    assert convert("Добар дан", "Latn") == "Dobar dan"
    assert convert("Dobar dan", "Cyrl") == "Добар дан"
    # mešano -> ćirilica ide preko latinice, bez duple konverzije
    assert convert("Добар dan", "Cyrl") == "Добар дан"
    with pytest.raises(ValueError):
        convert("x", "Arab")


def test_brojevi_i_interpunkcija_prolaze():
    assert to_cyrillic("U 2024. godini, 15%") == "У 2024. години, 15%"


# --- Idempotencija ---------------------------------------------------------

IDEMPOTENT_CASES = [
    "Dobar dan, hvala što ste došli.",
    "Добар дан, хвала што сте дошли.",
    "Добар dan, хвала što ste došli.",      # mešano, kao iz modela
    "Nadživeo je Tanjug i injekcije.",
    "Gledam Netflix u 2024. godini.",       # x nema ćirilični par
    "",
]


@pytest.mark.parametrize("tekst", IDEMPOTENT_CASES)
@pytest.mark.parametrize("pismo", ["Latn", "Cyrl"])
def test_ponovljena_primena_ne_menja_nista(tekst, pismo):
    jednom = convert(tekst, pismo)
    assert convert(jednom, pismo) == jednom
    assert convert(convert(jednom, pismo), pismo) == jednom


def test_convert_all_je_isto_sto_i_convert_po_stavci():
    tekstovi = IDEMPOTENT_CASES
    assert convert_all(tekstovi, "Cyrl") == [convert(t, "Cyrl") for t in tekstovi]


# --- Provera pisma nad gotovim tekstom -------------------------------------


def test_strana_slova_se_prijavljuju():
    assert foreign_letters("Добар дан", "Cyrl") == set()
    assert foreign_letters("Добар dan", "Cyrl") == {"d", "a", "n"}
    assert foreign_letters("Dobar dan", "Latn") == set()
    assert foreign_letters("Dobar дан", "Latn") == {"д", "а", "н"}


def test_slova_bez_cirilicnog_para_nisu_greska():
    """`q`, `w`, `x`, `y` ostaju i u ispravnom ćiriličnom titlu."""
    assert convert("Netflix", "Cyrl") == "Нетфлиx"
    assert foreign_letters(convert("Netflix", "Cyrl"), "Cyrl") == set()
    assert foreign_letters(convert("New York Times", "Cyrl"), "Cyrl") == set()


def test_prekrsaji_nose_oznaku_stavke():
    stavke = [(1, "Добар дан"), (2, "Dobar dan"), (3, "Хвала")]
    assert [id_ for id_, _ in script_violations(stavke, "Cyrl")] == [2]
    assert [id_ for id_, _ in script_violations(stavke, "Latn")] == [1, 3]
    assert script_violations([], "Cyrl") == []


def test_dvopismeni_jezici():
    assert is_biscriptal("sr") is True
    assert is_biscriptal("fr") is False
    assert is_biscriptal(None) is False
