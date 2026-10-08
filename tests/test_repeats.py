"""Merenje uzastopnih udvajanja i podela na oklevanje / pojačavanje.

Povod: prevod fr -> sr je preneo govorno oklevanje ("parce que, parce que" ->
"jer, jer"). Pre nego što se zbog toga dira prompt, treba znati da li model to
radi u većini slučajeva ili je bio izolovan slučaj.

Drugi povod, iz istog posla: sirov zbir je mešao oklevanje sa pojačavanjem
("très, très" -> "veoma, veoma"), pa je ista merenja moglo da se pročita na dva
suprotna načina. Zato se meri po vrsti. Ovi testovi pokrivaju MERU, ne prevod.
"""

from __future__ import annotations

from subtitis import repeats


# ==========================================================================
# Udvojena reč
# ==========================================================================


def test_ista_rec_dvaput_za_redom():
    r = repeats.measure("To je važno za mene, jer, jer mi se sviđa.", "sr")
    assert r.words == 1
    assert r.phrases == 0
    assert r.total == 1


def test_zarez_izmedju_ne_prekida_udvajanje():
    """Interpunkcija se ne broji — inače bi 'jer, jer' promaklo."""
    assert repeats.measure("jer, jer", "sr").words == 1
    assert repeats.measure("jer jer", "sr").words == 1


def test_velika_slova_se_ne_racunaju():
    assert repeats.measure("Ne, ne mogu.", "sr").words == 1


def test_tri_iste_reci_daju_dva():
    assert repeats.measure("Da, da, da.", "sr").words == 2


def test_bez_udvajanja_je_nula():
    r = repeats.measure("Ovo je sasvim obična rečenica bez oklevanja.", "sr")
    assert r.total == 0
    assert r.texts_with_repeat == 0


def test_ista_rec_ali_ne_za_redom_se_ne_broji():
    """Udvajanje na razmaku je stil, ne oklevanje."""
    assert repeats.measure("Video sam ga, a onda sam ga opet video.", "sr").total == 0


# ==========================================================================
# Udvojena dvočlana fraza
# ==========================================================================


def test_ista_fraza_dvaput_za_redom():
    r = repeats.measure("C'est fermé, parce que, parce que je pars.", "fr")
    assert r.phrases == 1
    assert r.words == 0


def test_apostrof_drzi_rec_na_okupu():
    """Bez apostrofa u tokenu bi udvojena REČ ispala udvojena FRAZA."""
    assert repeats.tokens("c'est") == ["c'est"]
    r = repeats.measure("c'est c'est important", "fr")
    assert r.words == 1 and r.phrases == 0


def test_udvojena_rec_se_ne_broji_i_kao_fraza():
    """'a a a a' je jedno mucanje, ne i tri reči i jedna fraza."""
    r = repeats.measure("a a a a", "fr")
    assert r.words == 3
    assert r.phrases == 0


def test_fraza_tri_puta_ne_daje_preklapajuce_nalaze():
    """Prozori se preklapaju, pa bi naivno brojanje dalo previše."""
    assert repeats.measure("x y x y x y").phrases == 1
    assert repeats.measure("x y x y x y x y").phrases == 2


# ==========================================================================
# Podela: oklevanje vs pojačavanje
# ==========================================================================


def test_prilog_stepena_je_pojacavanje():
    """Ovo treba da se PRENESE u prevod — nije kvar."""
    for text, lang in (
        ("C'est très, très beau.", "fr"),
        ("Veoma, veoma darežljiv.", "sr"),
        ("It was so, so good.", "en"),
    ):
        r = repeats.measure(text, lang)
        assert r.emphasis == 1, text
        assert r.hesitation == 0, text


def test_veznik_i_zamenica_su_oklevanje():
    for text, lang in (
        ("Et et je me lève...", "fr"),
        ("qui me qui me filme", "fr"),
        ("parce que, parce que je pars", "fr"),
        ("jer, jer mi se sviđa", "sr"),
        ("I, i onda smo krenuli.", "sr"),
        ("and and then we left", "en"),
    ):
        r = repeats.measure(text, lang)
        assert r.hesitation == 1, text
        assert r.emphasis == 0, text


def test_puna_rec_ostaje_nerazvrstana():
    """'drugarima, drugarima' nije ni prilog stepena ni funkcijska reč — ne nagađa se."""
    r = repeats.measure("pozdrav drugarima, drugarima iz kluba", "sr")
    assert r.unclassified == 1
    assert r.hesitation == 0 and r.emphasis == 0


def test_pojacavanje_ima_prednost_nad_oklevanjem():
    """'so' je i veznik i prilog stepena; udvojeno je pojačavanje."""
    r = repeats.measure("It was so so good.", "en")
    assert r.emphasis == 1 and r.hesitation == 0


def test_fraza_je_oklevanje_samo_ako_su_sve_reci_funkcijske():
    assert repeats.measure("parce que parce que", "fr").hesitation == 1
    # "moj brat moj brat" — zamenica jeste funkcijska, imenica nije.
    assert repeats.measure("moj brat moj brat", "sr").unclassified == 1


def test_nepoznat_jezik_ne_razvrstava():
    """Poštenije nego nagađati: mera tad ne zna, i to se vidi."""
    assert not repeats.known_language("de")
    r = repeats.measure("und und ich gehe", "de")
    assert r.total == 1
    assert r.unclassified == 1
    assert r.hesitation == 0


def test_bez_jezika_sve_je_nerazvrstano():
    r = repeats.measure("jer, jer mi se sviđa")
    assert r.total == 1 and r.unclassified == 1


def test_poznati_jezici():
    assert all(repeats.known_language(l) for l in ("fr", "sr", "en"))
    assert repeats.known_language("SR ")


# ==========================================================================
# Granice
# ==========================================================================


def test_granica_recenice_se_ne_prelazi():
    """'Da. Da.' preko dve rečenice je normalan dijalog, ne oklevanje."""
    assert repeats.measure_all(["Da.", "Da."], "sr").total == 0
    assert repeats.measure("Da. Da.", "sr").words == 1


def test_prazan_tekst():
    r = repeats.measure("", "sr")
    assert r.total == 0 and r.total_words == 0
    assert r.per_100_words == 0.0


def test_measure_all_sabira():
    r = repeats.measure_all(
        ["jer, jer mi se sviđa", "sasvim obično", "veoma, veoma dobro"], "sr"
    )
    assert r.hesitation == 1
    assert r.emphasis == 1
    assert r.texts == 3
    assert r.texts_with_repeat == 2
    # Rečenica sa pojačavanjem se NE broji kao rečenica sa oklevanjem.
    assert r.texts_with_hesitation == 1


def test_gustina_broji_samo_oklevanje():
    """Pojačavanje nije kvar, pa ne sme da diže gustinu."""
    assert repeats.measure("veoma veoma", "sr").per_100_words == 0.0
    assert repeats.measure("jer jer", "sr").per_100_words == 50.0


# ==========================================================================
# Odnos original / prevod
# ==========================================================================


def test_compare_line_prenosi_oklevanje():
    src = repeats.measure("parce que, parce que je pars", "fr")
    tgt = repeats.measure("jer, jer mi se sviđa", "sr")
    assert "PRENOSI" in repeats.compare_line(src, tgt)


def test_compare_line_uklanja_oklevanje():
    src = repeats.measure("jer, jer mi se sviđa", "sr")
    tgt = repeats.measure("because I like it", "en")
    assert "uklanja" in repeats.compare_line(src, tgt)


def test_compare_line_preneto_pojacavanje_nije_prenos():
    """Slučaj [7] iz prvog posla: sabrano bi reklo 'prenosi', a nije."""
    src = repeats.measure("C'est très, très beau.", "fr")
    tgt = repeats.measure("Veoma, veoma dobro.", "sr")
    line = repeats.compare_line(src, tgt)
    assert "nijedno" in line
    assert "pojačavanje 1 -> 1" in line


def test_compare_line_dodaje_na_cisto():
    src = repeats.measure("zato mi se sviđa", "sr")
    tgt = repeats.measure("jer, jer mi se sviđa", "sr")
    assert "DODAJE" in repeats.compare_line(src, tgt)


def test_primeri_se_pamte_po_vrsti():
    r = repeats.measure_all(["jer, jer mi se sviđa", "veoma, veoma dobro"], "sr")
    assert r.examples_line(repeats.HESITATION) == '"jer jer"'
    assert r.examples_line(repeats.EMPHASIS) == '"veoma veoma"'
    assert repeats.HESITATION in r.examples_line()


def test_primeri_prijavljuju_koliko_ih_je_preskoceno():
    r = repeats.measure_all(["jer jer"] * 10, "sr")
    assert r.hesitation == 10
    assert "(+7)" in r.examples_line(repeats.HESITATION)


# ==========================================================================
# Presuda po rečenici (eval skripta)
# ==========================================================================


def _verdicts(src_texts, tgt_texts, kind=repeats.HESITATION):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from eval_translation import repeat_verdicts

    return repeat_verdicts(
        [repeats.measure(t, "sr") for t in src_texts],
        [repeats.measure(t, "sr") for t in tgt_texts],
        kind,
    )


def test_verdikti_po_recenici():
    src = ["jer, jer", "obično", "obično", "da, da"]
    tgt = ["jer, jer", "obično", "pa, pa", "dobro"]
    assert _verdicts(src, tgt) == (1, 1, 1, 2)


def test_verdikti_ne_mesaju_vrste():
    """Preneto pojačavanje ne sme da uđe u presudu o oklevanju."""
    src = ["veoma, veoma dobro", "jer, jer mi se sviđa"]
    tgt = ["veoma, veoma dobro", "zato mi se sviđa"]
    assert _verdicts(src, tgt, repeats.HESITATION) == (0, 1, 0, 1)
    assert _verdicts(src, tgt, repeats.EMPHASIS) == (1, 0, 0, 1)
