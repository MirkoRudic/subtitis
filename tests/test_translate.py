"""Testovi validacije prevoda, prompta i pada na manji čank.

Ollama se ne dira — klijent se zamenjuje lažnim.
"""

from __future__ import annotations

import pytest

from subtitis.config import (
    SubtitisError,
    derived_glossary_pairs,
    glossary_pairs,
    hop_glossary,
    resolve_pivot,
)
from subtitis.models import Sentence
from subtitis.translate import (
    CHUNK_SIZE,
    CONTEXT_AFTER,
    CONTEXT_BEFORE,
    MAX_ATTEMPTS_PER_SENTENCE,
    NUM_CTX,
    REPEAT_PENALTY,
    TRANSLATION_FAILED_FLAG,
    build_system_prompt,
    build_user_prompt,
    format_glossary,
    strip_thinking,
    translate_sentences,
    validate,
    variant_rule,
)

# ==========================================================================
# validate()
# ==========================================================================


def test_ispravan_odgovor():
    got, err = validate("12|Prvi\n13|Drugi", [12, 13])
    assert err is None
    assert got == {12: "Prvi", 13: "Drugi"}


def test_spojene_linije_se_hvataju():
    """Najčešći kvar: pošalješ 10, model vrati 9 jer je spojio dve."""
    got, err = validate("1|Prvi\n2|Drugi", [1, 2, 3])
    assert got is None
    assert "očekivano" in err


def test_razbijena_linija_se_hvata():
    got, err = validate("1|Prvi\n2|Drugi\n3|Visak", [1, 2])
    assert got is None and err


def test_pogresni_brojevi_se_hvataju():
    """Model ume da vrati 2| za obe linije — viđeno uživo."""
    got, err = validate("2|Prvi\n2|Drugi", [1, 2])
    assert got is None
    assert "dobijeno [2]" in err


def test_linija_bez_separatora():
    got, err = validate("1|Prvi\nDrugi bez separatora", [1, 2])
    assert got is None
    assert err == "linija bez separatora"


def test_broj_nije_cifra():
    got, err = validate("prvi|Tekst", [1])
    assert got is None
    assert err == "broj nije cifra"


def test_prazna_linija():
    got, err = validate("1|Prvi\n2|", [1, 2])
    assert got is None
    assert err == "prazna linija"


def test_prazni_redovi_se_preskacu():
    got, err = validate("\n1|Prvi\n\n2|Drugi\n\n", [1, 2])
    assert err is None and len(got) == 2


def test_visak_razmaka_se_seche():
    """Qwen ostavlja dva razmaka na kraju linije."""
    got, err = validate("1|Prvi  \n2|  Drugi  ", [1, 2])
    assert got == {1: "Prvi", 2: "Drugi"}


def test_tekst_sa_separatorom_u_sebi():
    got, err = validate("1|Ovo je a|b slucaj", [1])
    assert err is None
    assert got == {1: "Ovo je a|b slucaj"}


def test_redosled_linija_nije_bitan():
    got, err = validate("2|Drugi\n1|Prvi", [1, 2])
    assert err is None
    assert got == {1: "Prvi", 2: "Drugi"}


def test_uvod_pred_odgovorom_se_hvata():
    got, err = validate("Evo prevoda:\n1|Prvi", [1])
    assert got is None


# ==========================================================================
# strip_thinking()
# ==========================================================================


def test_uklanja_think_blok():
    assert strip_thinking("<think>razmisljam</think>1|Prvi") == "1|Prvi"


def test_nedovrsen_think_blok():
    assert strip_thinking("1|Prvi<think>zaglavio") == "1|Prvi"


def test_bez_think_bloka_nista_ne_menja():
    assert strip_thinking("1|Prvi") == "1|Prvi"


# ==========================================================================
# Prompt
# ==========================================================================


def test_sistemski_prompt_ima_jezike_i_recnik():
    p = build_system_prompt("sr", "en", {"javno preduzeće": "state-owned enterprise"})
    assert "sa srpskog na engleski" in p
    assert "javno preduzeće -> state-owned enterprise" in p
    assert "Prvi znak tvog odgovora je cifra." in p


def test_prazan_recnik():
    assert format_glossary({}) == "(nema unosa)"
    assert "(nema unosa)" in build_system_prompt("sr", "en", {})


def test_korisnicki_prompt_ima_sva_tri_bloka():
    p = build_user_prompt(["Pre jedan."], [(12, "Prva."), (13, "Druga.")], ["Posle."])
    assert "KONTEKST (pre):\nPre jedan." in p
    assert "ZA_PREVOD:\n12|Prva.\n13|Druga." in p
    assert "KONTEKST (posle):\nPosle." in p


def test_pravila_o_duzini_i_govornom_jeziku():
    p = build_system_prompt("sr", "en", {})
    assert "11. Titl mora biti kratak." in p
    assert "Prevod ne sme biti duži od originala." in p
    assert "12. Ovo je govorni jezik, ne pisani." in p


def test_primeri_uz_pravilo_12_su_po_ciljnom_jeziku():
    en = build_system_prompt("sr", "en", {})
    assert '"because" a ne "considering that"' in en
    assert '"penultimate"' in en
    # engleski primeri nemaju šta da traže u francuskom promptu
    fr = build_system_prompt("sr", "fr", {})
    assert "12. Ovo je govorni jezik, ne pisani." in fr
    assert "considering that" not in fr
    assert "penultimate" not in fr


def test_pravilo_13_samo_uz_varijantu():
    bez = build_system_prompt("sr", "en", {})
    assert "13." not in bez
    assert "britanskim" not in bez

    gb = build_system_prompt("sr", "en", {}, "en-GB")
    assert "13. Piši britanskim pravopisom: -ise, -our, centre, organisation." in gb

    us = build_system_prompt("sr", "en", {}, "en-US")
    assert "13. Piši američkim pravopisom" in us


def test_varijanta_nije_osetljiva_na_velika_slova():
    assert variant_rule("EN-gb") == variant_rule("en-GB")
    assert "britanskim" in variant_rule(" en-gb ")


def test_nepoznata_varijanta_ne_dodaje_pravilo():
    assert variant_rule("en-AU") == ""
    assert variant_rule(None) == ""
    assert variant_rule("") == ""
    assert "13." not in build_system_prompt("sr", "en", {}, "klingonski")


def test_varijanta_stize_do_prompta_kroz_prevod():
    client = FakeClient(good)
    seen: list[str] = []
    orig = client.chat
    client.chat = lambda s, u: (seen.append(s), orig(s, u))[1]
    translate_sentences(
        sents(2), "sr", "en", client=client, variant="en-GB", log=lambda m: None
    )
    assert "britanskim pravopisom" in seen[0]


def test_glosar_stize_do_prompta_kroz_prevod():
    """Provera cele žice: glossary_pairs -> translate_sentences -> sistemski prompt."""
    client = FakeClient(good)
    seen: list[str] = []
    orig = client.chat
    client.chat = lambda s, u: (seen.append(s), orig(s, u))[1]
    translate_sentences(
        sents(2), "sr", "en", client=client,
        glossary_pairs={"animator": "activity leader"}, log=lambda m: None,
    )
    assert "animator -> activity leader" in seen[0]


def test_korisnicki_prompt_bez_konteksta():
    p = build_user_prompt([], [(1, "Sama.")], [])
    assert p == "ZA_PREVOD:\n1|Sama."


# ==========================================================================
# Lažni klijent
# ==========================================================================


class FakeClient:
    """Beleži pozive i odgovara po scenariju."""

    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[list[int]] = []
        from subtitis.translate import CallStats

        self.stats = CallStats()

    def chat(self, system: str, user: str) -> str:
        block = user.split("ZA_PREVOD:\n", 1)[1].split("\n\nKONTEKST")[0]
        lines = [l for l in block.split("\n") if l.strip()]
        ids = [int(l.split("|", 1)[0]) for l in lines]
        self.calls.append(ids)
        self.stats.requests += 1
        return self.behaviour(ids, lines)


def good(ids, lines):
    return "\n".join(f"{i}|EN {l.split('|', 1)[1]}" for i, l in zip(ids, lines))


def sents(n: int, start: int = 1) -> list[Sentence]:
    return [
        Sentence(id=i, s=float(i), e=float(i) + 1, text=f"Rečenica {i}.", words=(i, i))
        for i in range(start, start + n)
    ]


# ==========================================================================
# Čankovanje i kontekst
# ==========================================================================


def test_cankuje_po_zadatoj_velicini():
    s = sents(20)
    client = FakeClient(good)
    out, stats = translate_sentences(s, "sr", "en", client=client, chunk_size=9, log=lambda m: None)
    assert [len(c) for c in client.calls] == [9, 9, 2]
    assert stats.chunks_ok == 3
    assert all(t.text.startswith("EN ") for t in out)


def capture_prompts(sentences, **kw) -> list[str]:
    """Pusti prevod i vrati sve korisničke promptove redom."""
    client = FakeClient(good)
    seen: list[str] = []
    orig = client.chat

    def chat(system, user):
        seen.append(user)
        return orig(system, user)

    client.chat = chat
    translate_sentences(sentences, "sr", "en", client=client, log=lambda m: None, **kw)
    return seen


def test_kontekst_se_salje_ali_se_ne_prevodi():
    prompts = capture_prompts(sents(20), chunk_size=9)
    srednji = prompts[1]  # ima susede sa obe strane
    assert srednji.count("KONTEKST (pre):") == 1
    assert srednji.count("KONTEKST (posle):") == 1
    # numeracija je lokalna (1..n), ne pravi id — vidi test_numeracija_je_lokalna
    assert "ZA_PREVOD:\n1|Rečenica 10." in srednji
    # kontekst nije numerisan -> model ga ne može pomešati sa gradivom
    before = srednji.split("KONTEKST (pre):\n")[1].split("\n\n")[0]
    assert "|" not in before
    assert before.splitlines() == ["Rečenica 8.", "Rečenica 9."]


def test_kontekst_je_tacno_dve_recenice_sa_svake_strane():
    prompts = capture_prompts(sents(20), chunk_size=9)
    before = prompts[1].split("KONTEKST (pre):\n")[1].split("\n\n")[0]
    after = prompts[1].split("KONTEKST (posle):\n")[1].strip()
    assert len(before.splitlines()) == CONTEXT_BEFORE
    assert len(after.splitlines()) == CONTEXT_AFTER


def test_prvi_cank_nema_kontekst_pre():
    prompts = capture_prompts(sents(5), chunk_size=9)
    assert "KONTEKST (pre):" not in prompts[0]
    assert "KONTEKST (posle):" not in prompts[0]


def test_id_jevi_vremena_i_rasponi_ostaju_netaknuti():
    s = sents(5)
    out, _ = translate_sentences(sents(5), "sr", "en", client=FakeClient(good), log=lambda m: None)
    for a, b in zip(s, out):
        assert (a.id, a.s, a.e, a.words) == (b.id, b.s, b.e, b.words)


# ==========================================================================
# Pad na manji čank
# ==========================================================================


def test_pad_na_polovinu_pa_na_pojedinacne():
    """Čank od 4 koji uvek pada mora da se prepolovi do pojedinačnih rečenica."""
    calls = []

    def always_bad(ids, lines):
        calls.append(list(ids))
        return "smeće bez separatora"

    client = FakeClient(always_bad)
    out, stats = translate_sentences(
        sents(4), "sr", "en", client=client, chunk_size=4, log=lambda m: None
    )
    sizes = [len(c) for c in calls]
    assert sizes[0] == 4
    assert 2 in sizes and 1 in sizes
    # svaka rečenica dobija MAX_ATTEMPTS_PER_SENTENCE pokušaja
    assert sizes.count(1) == 4 * MAX_ATTEMPTS_PER_SENTENCE
    assert stats.failed_ids == [1, 2, 3, 4]


def test_neuspeh_upisuje_original_sa_flagom():
    out, stats = translate_sentences(
        sents(1), "sr", "en",
        client=FakeClient(lambda ids, lines: "bez separatora"),
        log=lambda m: None,
    )
    assert out[0].text == "Rečenica 1."
    assert TRANSLATION_FAILED_FLAG in out[0].flags
    assert stats.failed_ids == [1]


def test_jedna_losa_recenica_ne_obara_ceo_posao():
    def bad_for_three(ids, lines):
        # po tekstu, ne po broju: model vidi lokalne brojeve 1..n
        if any(l.endswith("|Rečenica 3.") for l in lines):
            return "smeće"
        return good(ids, lines)

    out, stats = translate_sentences(
        sents(5), "sr", "en", client=FakeClient(bad_for_three),
        chunk_size=5, log=lambda m: None,
    )
    assert stats.failed_ids == [3]
    assert out[2].text == "Rečenica 3."
    assert TRANSLATION_FAILED_FLAG in out[2].flags
    for k in (0, 1, 3, 4):
        assert out[k].text.startswith("EN ")
        assert TRANSLATION_FAILED_FLAG not in out[k].flags


def test_oporavak_iz_drugog_pokusaja():
    state = {"n": 0}

    def flaky(ids, lines):
        state["n"] += 1
        if state["n"] == 1:
            return "smeće"
        return good(ids, lines)

    out, stats = translate_sentences(
        sents(2), "sr", "en", client=FakeClient(flaky), chunk_size=2, log=lambda m: None
    )
    assert stats.failed_ids == []
    assert all(t.text.startswith("EN ") for t in out)


def test_postojeci_flagovi_ostaju():
    s = sents(1)
    s[0].flags = ["low_confidence"]
    out, _ = translate_sentences(
        s, "sr", "en", client=FakeClient(lambda i, l: "smeće"), log=lambda m: None
    )
    assert out[0].flags == ["low_confidence", TRANSLATION_FAILED_FLAG]


def test_greska_veze_ne_izaziva_polovljenje():
    """Ako Ollama ne odgovara, polovljenje nema smisla — neka pukne jasno."""

    def dead(ids, lines):
        raise SubtitisError("Ollama ne odgovara")

    with pytest.raises(SubtitisError):
        translate_sentences(
            sents(8), "sr", "en", client=FakeClient(dead), chunk_size=8, log=lambda m: None
        )


def test_prazan_ulaz():
    out, stats = translate_sentences([], "sr", "en", client=FakeClient(good), log=lambda m: None)
    assert out == [] and stats.sentences == 0


def test_statistika_je_po_prolazu_a_ne_kumulativna():
    """Kod pivota se drugi skok inače prikazuje sa brojevima prvog."""
    client = FakeClient(good)
    _, prvi = translate_sentences(
        sents(5), "sr", "en", client=client, chunk_size=5, log=lambda m: None
    )
    _, drugi = translate_sentences(
        sents(5), "en", "fr", client=client, chunk_size=5, log=lambda m: None
    )
    assert prvi.calls.requests == 1
    assert drugi.calls.requests == 1
    assert client.stats.requests == 2


# ==========================================================================
# Konstante i konfiguracija
# ==========================================================================


def test_kljucne_konstante():
    assert NUM_CTX == 8192, "Ollamin default 4096 tiho seče prompt"
    assert REPEAT_PENALTY == 1.0, "preko 1.0 model izbegava ponavljanja normalna u titlovima"
    assert 8 <= CHUNK_SIZE <= 10
    assert CONTEXT_BEFORE == 2 and CONTEXT_AFTER == 2


def test_glosar_po_smeru():
    g = {"sr->en": {"a": "b"}, "sr->fr": {"c": "d"}}
    assert glossary_pairs(g, "sr", "en") == {"a": "b"}
    assert glossary_pairs(g, "sr", "de") == {}


def test_primer_glosara_ima_trazene_unose():
    """glossary/default.json je izmišljen primer (javni repo); mora da pokrije
    tri vrste unosa: naziv, reč koja se ne prevodi doslovno, ustaljen izraz."""
    from subtitis.config import load_glossary

    g = glossary_pairs(load_glossary(None), "sr", "en")
    assert g["Radionica otvorenih vrata"] == "Open Door Workshop"
    assert g["animator"] == "activity leader"
    assert g["s obzirom na to da"] == "because"


def test_izvedeni_glosar_za_drugi_skok_pivota():
    """sr->en->fr: en->fr unosa nema, ali se izvode iz sr->en i sr->fr."""
    g = {
        "sr->en": {"animator": "activity leader", "radionica otvorenih vrata": "open door workshop"},
        "sr->fr": {"animator": "éducateur spécialisé"},
    }
    assert derived_glossary_pairs(g, "sr", "en", "fr") == {
        "activity leader": "éducateur spécialisé"
    }


def test_izvodi_se_samo_gde_postoje_oba_prevoda():
    g = {"sr->en": {"a": "A", "b": "B"}, "sr->fr": {"a": "AA"}}
    assert derived_glossary_pairs(g, "sr", "en", "fr") == {"A": "AA"}


def test_prazan_prevod_se_ne_izvodi():
    g = {"sr->en": {"a": "A"}, "sr->fr": {"a": ""}}
    assert derived_glossary_pairs(g, "sr", "en", "fr") == {}


def test_hop_glossary_prvi_skok_je_direktan():
    g = {"sr->en": {"a": "A"}, "sr->fr": {"a": "AA"}}
    pairs, derived = hop_glossary(g, "sr", "en", origin="sr")
    assert pairs == {"a": "A"} and derived == 0


def test_hop_glossary_drugi_skok_izvodi():
    g = {"sr->en": {"a": "A"}, "sr->fr": {"a": "AA"}}
    pairs, derived = hop_glossary(g, "en", "fr", origin="sr")
    assert pairs == {"A": "AA"} and derived == 1


def test_eksplicitni_unos_pobedjuje_izvedeni():
    g = {
        "sr->en": {"a": "A"},
        "sr->fr": {"a": "AA"},
        "en->fr": {"A": "rucno"},
    }
    pairs, derived = hop_glossary(g, "en", "fr", origin="sr")
    assert pairs == {"A": "rucno"}
    assert derived == 0, "ručni unos se ne broji kao izveden"


def test_bez_pivota_nema_izvodjenja():
    g = {"sr->en": {"a": "A"}}
    assert hop_glossary(g, "sr", "en")[0] == {"a": "A"}


def test_glosar_bez_sr_fr_izvodi_prazno():
    """Regresija: dok `sr->fr` ne postoji, drugi skok pivota nema šta da izvede.

    Nekad je čitao stvarni `glossary/default.json`; otkad on ima `sr->fr`
    (2026-09-28), premisa važi samo za glosar bez njega — zato je ovde svoj.
    """
    g = {"sr->en": {"Radionica otvorenih vrata": "Open Door Workshop"}}
    pairs, derived = hop_glossary(g, "en", "fr", origin="sr")
    assert pairs == {} and derived == 0


def test_varijanta_iz_konfiguracije():
    from subtitis.config import target_variant

    assert target_variant("en") == "en-GB"
    assert target_variant("de") is None


def test_pivot_podrazumevano_iskljucen():
    assert resolve_pivot("sr", "fr") is None
    assert resolve_pivot("sr", "en") is None


def test_pivot_se_moze_ukljuciti(monkeypatch):
    from subtitis import config

    monkeypatch.setitem(config.PIVOT_VIA, "sr->fr", "en")
    assert resolve_pivot("sr", "fr") == "en"


def test_pivot_ne_sme_biti_izvor_ili_cilj(monkeypatch):
    from subtitis import config

    monkeypatch.setitem(config.PIVOT_VIA, "sr->fr", "sr")
    assert resolve_pivot("sr", "fr") is None


# ==========================================================================
# Neuravnoteženi navodnici: odbij, ponovi, pa original sa oznakom
# ==========================================================================


class QuoteClient:
    """Za rečenicu `bad` vraća neuravnotežene navodnike `fails` puta."""

    def __init__(self, bad: str, fails: int) -> None:
        from subtitis.translate import CallStats

        self.stats = CallStats()
        self.bad, self.fails = bad, fails  # `bad` je izvorni tekst rečenice

    def chat(self, system, user):
        block = user.split("ZA_PREVOD:\n", 1)[1].split("\n\nKONTEKST")[0]
        out = []
        for line in block.splitlines():
            if not line.strip():
                continue
            i, text = line.split("|", 1)
            if text == self.bad and self.fails > 0:
                self.fails -= 1
                out.append(f'{i}|"Ringe ringe raja" "')
            else:
                out.append(f"{i}|EN {text}")
        return "\n".join(out)


def _quote_sentences():
    return [
        Sentence(id=1, s=0, e=1, text="Prva.", words=(0, 0)),
        Sentence(id=2, s=1, e=2, text="“Ringe ringe raja” “Ringe raja sa”", words=(1, 1)),
        Sentence(id=3, s=2, e=3, text="Treća.", words=(2, 2)),
    ]


def test_neuravnotezeni_navodnici_se_ponavljaju():
    client = QuoteClient(bad="“Ringe ringe raja” “Ringe raja sa”", fails=1)
    out, stats = translate_sentences(_quote_sentences(), "sr", "en", client=client, log=lambda m: None)
    assert [s.text for s in out] == ["EN Prva.", "EN “Ringe ringe raja” “Ringe raja sa”", "EN Treća."]
    assert stats.quotes_rejected == 1 and stats.failed_ids == []


def test_posle_dva_neuspeha_ostaje_original_sa_oznakom():
    client = QuoteClient(bad="“Ringe ringe raja” “Ringe raja sa”", fails=2)
    out, stats = translate_sentences(_quote_sentences(), "sr", "en", client=client, log=lambda m: None)
    assert out[1].text == "“Ringe ringe raja” “Ringe raja sa”"
    assert TRANSLATION_FAILED_FLAG in out[1].flags
    assert stats.failed_ids == [2] and stats.quotes_rejected == 2
    assert out[0].text == "EN Prva." and out[2].text == "EN Treća."


def test_neuravnotezen_original_ne_trazi_uravnotezen_prevod():
    """Rečenica presečena usred citata — prevod ne može biti uravnotežen."""
    sents = [Sentence(id=1, s=0, e=1, text="“Ringe ringe raja” “Raja", words=(0, 0))]

    class C(QuoteClient):
        def chat(self, system, user):
            return '1|"Ringe ringe raja" "Raja'

    out, stats = translate_sentences(sents, "sr", "en", client=C("", 0), log=lambda m: None)
    assert out[0].text == '"Ringe ringe raja" "Raja' and stats.quotes_rejected == 0


def test_numeracija_je_lokalna_i_mapira_se_nazad():
    """Model je za odlomak [100..108] vraćao [1..9] — prenumerisao je sam.
    Sad vidi 1..n; pravi id-jevi se vraćaju u kodu."""
    s = sents(9, start=100)
    client = FakeClient(good)
    out, stats = translate_sentences(s, "sr", "en", client=client, chunk_size=9, log=lambda m: None)
    assert client.calls == [list(range(1, 10))]
    assert stats.chunks_failed == 0
    assert [(x.id, x.text) for x in out] == [(i, f"EN Rečenica {i}.") for i in range(100, 109)]


def test_provera_broja_linija_i_dalje_hvata_spajanje():
    def merged(ids, lines):
        return good(ids[:-1], lines[:-1])  # dve linije spojene u jednu

    out, stats = translate_sentences(
        sents(4, start=100), "sr", "en", client=FakeClient(merged), chunk_size=4, log=lambda m: None,
    )
    assert stats.chunks_failed > 0
