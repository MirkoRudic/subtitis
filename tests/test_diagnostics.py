"""Dijagnostika posle transkripcije: tempo rečenica i raspodela verovatnoća.

Sve u ovom fajlu potiče iz jednog posla sa lošim zvukom (2026-08-19, snimak bez
bubice):

1. Rečenica „8 cm." trajala je 31.41 -> 31.47, dakle 0.06 s za dve reči. Prošla
   je kroz `MIN_SENTENCE_WORDS` i `MIN_SENTENCE_DURATION` jer se završava
   tačkom, a izuzeće „ima jasnu interpunkciju" nije imalo gornju granicu.
2. Bojenje po pragu 0.6 označilo je skoro sve rečenice, pa je prestalo da nosi
   signal. Pre bilo kakve promene praga treba videti raspodelu.
"""

from __future__ import annotations

from subtitis.models import Sentence, Word
from subtitis.segment import (
    CONFIDENCE_EDGES,
    LOW_CONFIDENCE_SENTENCE_SHARE,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_DURATION_PER_WORD,
    MIN_WORDS_FOR_SHARE_RULE,
    analyze_confidence,
    analyze_pace,
    is_low_confidence,
    low_confidence_share,
    merge_short,
    percentile,
    segment,
)


def words(text: str, *, start=0.0, dur=0.35, gap=0.25, p=0.9) -> list[Word]:
    out, t = [], start
    for k, w in enumerate(text.split()):
        out.append(Word(i=k, s=round(t, 3), e=round(t + dur, 3), w=w, p=p))
        t += dur + gap
    return out


def squeezed(text: str, *, start: float, span: float, p=0.9) -> list[Word]:
    """Ceo tekst stisnut u `span` sekundi — artefakt, ne govor."""
    toks = text.split()
    step = span / max(1, len(toks))
    return [
        Word(i=k, s=round(start + k * step, 3), e=round(start + (k + 1) * step, 3),
             w=w, p=p)
        for k, w in enumerate(toks)
    ]


# ==========================================================================
# Zaštitna mreža: nemoguće trajanje probija izuzeće za interpunkciju
# ==========================================================================


def test_artefakt_sa_tackom_se_ipak_spaja():
    """Slučaj iz posla: „8 cm." u 0.06 s, sa tačkom na kraju.

    Pre popravke je izuzeće „ima jasnu interpunkciju" puštalo ovo kroz obe
    provere, iako je gustina znakova u fajlu bila sasvim uredna.
    """
    a = words("prva recenica ima dovoljno reci i traje dugo", start=26.0)
    b = squeezed("8 cm .", start=31.41, span=0.06)
    c = words("ovo je sledeca recenica posle tog kratkog artefakta", start=33.26)
    out = merge_short([a, b, c], trust_punctuation=True)
    assert len(out) == 2, "artefakt od 0.06 s ne sme da ostane zasebna rečenica"
    assert "8 cm" in " ".join(w.w for w in out[1]), "spaja se sa sledećom"


def test_prava_kratka_recenica_sa_tackom_prezivljava():
    """„Hvala puno." (0.61 s / 2 reči) je u redu i mora da ostane sama.

    Ovo je razlog zbog kog granica NIJE 0.3 s po reči: ta rečenica ima
    0.303 s po reči i pala bi za tri hiljadita dela sekunde.
    """
    a = words("hvala puno", start=0.0, dur=0.28, gap=0.02) + [
        Word(i=9, s=0.60, e=0.61, w=".", p=1.0)
    ]
    b = words("posle toga smo otisli kuci", start=5.0)
    out = merge_short([a, b], trust_punctuation=True)
    assert len(out) == 2
    assert (a[-1].e - a[0].s) / 2 < 0.31, "kontrola: ovo je zaista brza rečenica"


def test_granica_je_ispod_najbrzeg_pravog_govora():
    """Izmereni minimum pravog govora je 0.239 s/reč (fr, 16 reči)."""
    assert MIN_DURATION_PER_WORD < 0.239
    assert MIN_DURATION_PER_WORD > 0.03, "mora biti iznad izmerenog artefakta"


def test_granica_je_podesiva():
    a = words("prva recenica ima dovoljno reci i traje dugo", start=26.0)
    b = squeezed("da .", start=31.41, span=0.2)  # jedna reč, 0.2 s po reči
    c = words("i onda sledi jos jedna normalna recenica", start=33.0)
    assert len(merge_short([a, b, c], min_duration_per_word=0.0)) == 3
    assert len(merge_short([a, b, c], min_duration_per_word=0.3)) == 2


def test_poslednji_artefakt_daleko_od_prethodne_prezivi_ali_se_prijavi():
    """Granični slučaj koji mreža ne može da reši, pa mora da ga prijavi.

    Ako je artefakt POSLEDNJI komad, a prethodna rečenica je toliko udaljena da
    bi spajanje probilo `MAX_SENTENCE_DURATION`, spajanja nema — spojena
    rečenica bi trajala pola minuta i bila bi gora od artefakta. Tada ga hvata
    `Pace.warning()`, koji se ispisuje posle svake transkripcije.
    """
    a = words("prva recenica ima dovoljno reci i traje dugo", start=0.0)
    b = squeezed("8 cm .", start=31.41, span=0.06)
    out = merge_short([a, b], trust_punctuation=True)
    assert len(out) == 2, "spajanje bi napravilo rečenicu od 31 s"

    p = analyze_pace([sent(1, 0.0, 5.0, "prva recenica ima dosta reci"),
                      sent(2, 31.41, 31.47, "8 cm.")])
    assert p.impossible == [2]
    assert p.warning() is not None


def test_ceo_prolaz_kroz_segment_uklanja_artefakt():
    toks = words("ovo je prva recenica koja traje sasvim normalno") + [
        Word(i=98, s=5.0, e=5.05, w=".", p=1.0)
    ]
    toks += squeezed("8 cm .", start=5.05, span=0.06, p=0.08)
    toks += words("i onda sledi jos jedna normalna recenica", start=7.0)
    out = segment(toks)
    assert all(s.e - s.s > 0.5 for s in out), [(s.text, s.e - s.s) for s in out]
    assert not analyze_pace(out).impossible


# ==========================================================================
# Tempo: najkraća rečenica i najkraće trajanje po reči
# ==========================================================================


def sent(id_: int, s: float, e: float, text: str) -> Sentence:
    return Sentence(id=id_, s=s, e=e, text=text, words=(0, 1))


def test_pace_nalazi_najkracu_i_najbrzu():
    ss = [
        sent(1, 0.0, 10.0, "ovo je duga i spora recenica sa mnogo reci"),
        sent(2, 10.0, 11.2, "kratka ali normalna"),
        sent(3, 20.0, 26.0, "jedna dve tri cetiri pet sest sedam osam devet deset"),
    ]
    p = analyze_pace(ss)
    assert p.shortest_id == 2
    assert round(p.shortest_duration, 2) == 1.20
    # [1] 10.0s/9 = 1.11, [2] 1.2s/3 = 0.40, [3] 6.0s/10 = 0.60
    assert p.fastest_id == 2
    assert round(p.fastest_per_word, 2) == 0.40
    assert p.sentences == 3


def test_pace_najbrza_nije_uvek_najkraca():
    """Najkraća po trajanju i najbrža po reči su različite mere."""
    ss = [
        sent(1, 0.0, 0.9, "dve reci"),            # 0.45 s/reč, najkraća
        sent(2, 5.0, 8.0, "jedna dve tri cetiri pet sest sedam osam devet deset"),  # 0.30
    ]
    p = analyze_pace(ss)
    assert p.shortest_id == 1
    assert p.fastest_id == 2


def test_pace_prijavljuje_nemoguce():
    ss = [
        sent(1, 0.0, 10.0, "ovo je sasvim normalna recenica sa dosta reci"),
        sent(2, 31.41, 31.47, "8 cm."),
    ]
    p = analyze_pace(ss)
    assert p.impossible == [2]
    assert "NEMOGUĆE TRAJANJE" in p.warning()
    assert "[2] '8 cm.'" in p.line()


def test_pace_bez_upozorenja_kad_je_sve_u_redu():
    ss = [sent(1, 0.0, 5.0, "ovo je sasvim normalna recenica")]
    p = analyze_pace(ss)
    assert p.warning() is None
    assert "tempo:" in p.line()


def test_pace_bez_recenica():
    p = analyze_pace([])
    assert p.shortest_id is None
    assert "nema nijedne" in p.line()
    assert p.warning() is None


def test_pace_preskace_recenicu_bez_reci():
    ss = [sent(1, 0.0, 0.1, "..."), sent(2, 1.0, 5.0, "ovo su prave reci")]
    p = analyze_pace(ss)
    assert p.shortest_id == 2


# ==========================================================================
# Raspodela verovatnoća — merenje pred odluku o pragu
# ==========================================================================


def conf_words(ps: list[float]) -> list[Word]:
    return [
        Word(i=k, s=float(k), e=float(k) + 0.5, w=f"rec{k}", p=p)
        for k, p in enumerate(ps)
    ]


def test_opsezi_su_oni_koje_ocekujemo():
    assert CONFIDENCE_EDGES == (0.4, 0.6, 0.8)


def test_raspodela_po_opsezima():
    c = analyze_confidence(conf_words([0.1, 0.3, 0.5, 0.65, 0.7, 0.9, 0.95, 1.0]))
    assert c.words == 8
    assert c.counts == [2, 1, 2, 3]


def test_granica_opsega_ide_navise():
    """Reč sa p tačno 0.6 nije ispod praga 0.6."""
    c = analyze_confidence(conf_words([0.6]))
    assert c.counts == [0, 0, 1, 0]
    assert c.below_threshold == 0


def test_prag_broji_samo_opsege_ispod_sebe():
    c = analyze_confidence(conf_words([0.1, 0.5, 0.7, 0.9]))
    assert c.threshold == LOW_CONFIDENCE_THRESHOLD
    assert c.below_threshold == 2
    assert round(c.flagged_share, 2) == 0.5


def test_interpunkcija_se_ne_broji():
    """Znak nasledi verovatnoću susedne reči i samo bi naduvao gornji opseg."""
    ws = conf_words([0.2, 0.9]) + [Word(i=9, s=9.0, e=9.05, w=".", p=1.0)]
    c = analyze_confidence(ws)
    assert c.words == 2


def test_percentili():
    ps = [k / 100 for k in range(1, 101)]
    c = analyze_confidence(conf_words(ps))
    assert abs(c.percentiles[50] - 0.50) < 0.02
    assert abs(c.percentiles[15] - 0.15) < 0.02


def test_percentil_praznog_uzorka():
    assert percentile([], 15) == 0.0


def test_ispis_sadrzi_relativan_prag():
    """Red koji odgovara na pitanje: gde bi stajao prag „donjih 15%"."""
    c = analyze_confidence(conf_words([0.1, 0.2, 0.5, 0.7, 0.85, 0.9, 0.95, 1.0]))
    text = "\n".join(c.lines())
    assert "percentili" in text
    assert "donjih 15%" in text
    assert f"prag {LOW_CONFIDENCE_THRESHOLD:.2f}" in text


def test_ispis_praznog_uzorka():
    assert analyze_confidence([]).lines() == []


def test_los_zvuk_ima_pomerenu_raspodelu():
    """Sam oblik mere: gori zvuk mora dati veći udeo ispod praga."""
    dobar = analyze_confidence(conf_words([0.95] * 90 + [0.3] * 10))
    los = analyze_confidence(conf_words([0.95] * 50 + [0.3] * 50))
    assert dobar.flagged_share < los.flagged_share
    assert dobar.percentiles[15] > los.percentiles[15]


# ==========================================================================
# Oznaka rečenice: udeo sumnjivih reči, ne „bar jedna"
# ==========================================================================


def mixed(n_good: int, n_bad: int, *, good=0.95, bad=0.3) -> list[Word]:
    """Rečenica sa zadatim brojem sigurnih i sumnjivih reči."""
    ps = [good] * n_good + [bad] * n_bad
    return [
        Word(i=k, s=float(k), e=float(k) + 0.5, w=f"rec{k}", p=p)
        for k, p in enumerate(ps)
    ]


def test_konstante_su_izlozene():
    assert LOW_CONFIDENCE_SENTENCE_SHARE == 0.20
    assert MIN_WORDS_FOR_SHARE_RULE == 5
    assert LOW_CONFIDENCE_THRESHOLD == 0.6, "prag po reči se ovom izmenom NE dira"


def test_jedna_losa_rec_u_dugoj_recenici_ne_oznacava():
    """Srž izmene: 1 od 15 je 6.7%, ispod praga od 20%."""
    assert not is_low_confidence(mixed(14, 1))


def test_petina_losih_reci_oznacava():
    assert is_low_confidence(mixed(12, 3))          # 3/15 = 20%
    assert not is_low_confidence(mixed(13, 2))      # 2/15 = 13%


def test_prag_je_ukljuciv():
    """Tačno 20% se označava — „petina ili više"."""
    assert low_confidence_share(mixed(8, 2)) == 0.2
    assert is_low_confidence(mixed(8, 2))


def test_kratka_recenica_i_dalje_ide_po_bar_jedna():
    """U rečenici od 4 reči jedna loša je 25% — svaki udeo bi je uhvatio,
    ali pravilo mora da važi i za dve reči, gde je jedna loša pola."""
    assert is_low_confidence(mixed(3, 1))
    assert is_low_confidence(mixed(1, 1))
    assert not is_low_confidence(mixed(4, 0))


def test_granica_kratkog_pravila():
    # 4 reči -> „bar jedna"; 5 reči -> udeo (1/5 = 20%, tačno na pragu)
    assert is_low_confidence(mixed(3, 1))
    assert is_low_confidence(mixed(4, 1))
    assert not is_low_confidence(mixed(5, 1))       # 1/6 = 17%


def test_interpunkcija_ne_razblazuje_udeo():
    """Znak nasledi verovatnoću susedne reči; u imeniocu bi lažno smirivao."""
    chunk = mixed(4, 1) + [
        Word(i=90 + k, s=90.0 + k, e=90.5 + k, w=",", p=1.0) for k in range(10)
    ]
    assert low_confidence_share(chunk) == 0.2
    assert is_low_confidence(chunk)


def test_prazan_komad():
    assert not is_low_confidence([])
    assert low_confidence_share([]) == 0.0
    assert not is_low_confidence(
        [Word(i=0, s=0.0, e=0.1, w=",", p=0.1)]
    ), "sam znak nije rečenica"


def test_prag_i_udeo_su_podesivi():
    chunk = mixed(12, 3)                             # 20%
    assert is_low_confidence(chunk, share=0.20)
    assert not is_low_confidence(chunk, share=0.25)
    assert not is_low_confidence(chunk, threshold=0.2), "prag po reči nikog ne hvata"


def test_kroz_segment_duga_recenica_sa_jednom_losom_recju():
    from conftest import make_words

    words = make_words(
        "Ovo je duga recenica koja ima tacno jednu nesigurnu rec u sebi negde."
    )
    words[5].p = 0.3
    s = segment(words)
    assert s[0].flags == [], "1 od 12 reči ne sme da oboji celu rečenicu"


def test_kroz_segment_recenica_sa_mnogo_losih_reci():
    from conftest import make_words

    words = make_words(
        "Ovo je duga recenica koja ima tacno jednu nesigurnu rec u sebi negde."
    )
    for k in (2, 3, 4, 5):
        words[k].p = 0.3
    s = segment(words)
    assert s[0].flags == ["low_confidence"]


# ==========================================================================
# Kapija za restauraciju: zarez ulazi u odluku
# ==========================================================================


def structure(marks_per_100: float, commas_per_100: float, *, words=500, avg=12.0):
    from subtitis.segment import Structure

    return Structure(
        words=words,
        sentences=int(words / avg),
        sentence_end_marks=round(marks_per_100 * words / 100),
        commas=round(commas_per_100 * words / 100),
        punctuation_tokens=0,
        avg_words_per_sentence=avg,
        capitalized_share=0.5,
    )


def test_puno_zareza_malo_tacaka_ne_pokrece_restauraciju():
    """Slučaj iz posla: 3.4 znaka kraja i 17.6 zareza na sto reči.

    Restauracija je tu okinula bez potrebe i koštala 61 s, od čega 33 s na
    odbijene odgovore, uz dva sumnjiva ustupka na istoj reči.
    """
    st = structure(3.4, 17.6)
    assert not st.sparse_punctuation("sr")
    assert not st.needs_punctuation("sr")
    assert "zarezi drže" in st.gate_reason("sr")


def test_malo_i_tacaka_i_zareza_i_dalje_pokrece():
    """Pravi kvar iz fr posla 2026-08-17: 2.5 znaka kraja, 1.0 zarez."""
    st = structure(2.5, 1.0)
    assert st.sparse_punctuation("fr")
    assert st.needs_punctuation("fr")
    assert st.gate_reason("fr") == "obe gustine ispod praga"


def test_dovoljno_tacaka_ne_pokrece_bez_obzira_na_zareze():
    for commas in (0.0, 20.0):
        st = structure(6.0, commas)
        assert not st.needs_punctuation("sr")
        assert st.gate_reason("sr") == "znakova kraja dovoljno"


def test_bez_ijednog_znaka_kraja_zarez_ne_spasava():
    """Bez tačke nema granice rečenice ni da se traži — zarez je ne pravi."""
    st = structure(0.0, 20.0)
    assert st.no_punctuation
    assert st.needs_punctuation("sr")
    assert st.gate_reason("sr") == "nijedan znak kraja"


def test_prag_zareza_je_recnik_po_jeziku():
    from subtitis.segment import COMMA_DENSITY_THRESHOLD, comma_threshold

    assert comma_threshold("sr") == COMMA_DENSITY_THRESHOLD["default"] == 8.0
    assert comma_threshold("nepoznat") == 8.0
    assert comma_threshold(None) == 8.0


def test_obe_gustine_su_i_dalje_u_ispisu():
    line = structure(3.4, 17.6).density_line("sr")
    assert "3.4 znaka kraja" in line and "17.6 zareza" in line
    assert "prag 4.0" in line and "prag 8.0" in line


# ==========================================================================
# Predlog da se proba bez detekcije govora
# ==========================================================================


def test_predlog_kad_vad_odbaci_previse():
    """Tihi snimak, prag 0.35: odbačeno 12.9% uz pokrivenost 85.6%."""
    from subtitis.segment import vad_advice

    msg = vad_advice(0.129, 85.6, vad_on=True)
    assert msg is not None
    assert "12.9%" in msg and "85.6%" in msg


def test_bez_predloga_kad_vad_odbaci_malo():
    """Dobar snimak sa bubicom: odbačeno 4.8% uz pokrivenost 88.2%.

    Pokrivenost mu je skoro ista kao tihom snimku (88.2 naspram 85.6) — zato
    prag i NIJE na pokrivenost. Vidi `MAX_VAD_DISCARD_SHARE`.
    """
    from subtitis.segment import vad_advice

    assert vad_advice(0.048, 88.2, vad_on=True) is None


def test_prag_na_pokrivenost_bi_okinuo_uvek():
    """Zašto pokrivenost ne može biti kriterijum: ni jedan posao je ne prelazi.

    Najveća izmerena pokrivenost, i to sa POTPUNO isključenim VAD-om, je 88.9%.
    Svaki prag na 90% je zato „uvek uključen", a to je isto što i nikad.
    """
    from subtitis.segment import vad_advice

    izmerene_pokrivenosti = (88.2, 85.6, 87.9, 79.6, 79.2)
    assert all(c < 90.0 for c in izmerene_pokrivenosti)
    # Ista ta dva posla se po odbačenom udelu jasno razdvajaju:
    assert vad_advice(0.048, 88.2, vad_on=True) is None
    assert vad_advice(0.129, 85.6, vad_on=True) is not None


def test_pri_iskljucenom_vadu_prijavljuje_tisinu():
    """Drugi smer: isključen VAD nad snimkom sa mnogo tišine."""
    from subtitis.segment import vad_advice

    msg = vad_advice(0.40, 60.0, vad_on=False)
    assert msg is not None and "ISKLJUČENA" in msg and "40.0%" in msg


def test_pri_iskljucenom_vadu_cuti_na_gustom_govoru():
    """Svi mereni snimci su gusti (odbačeno 4.8-15.0%) — tu nema šta da se kaže."""
    from subtitis.segment import vad_advice

    for share in (0.048, 0.129, 0.150):
        assert vad_advice(share, 88.9, vad_on=False) is None


# ==========================================================================
# Merenje: izuzeće funkcijskih reči (would_flag)
# ==========================================================================


def sr_chunk(pairs: list[tuple[str, float]]) -> list[Word]:
    return [
        Word(i=k, s=float(k), e=float(k) + 0.5, w=w, p=p)
        for k, (w, p) in enumerate(pairs)
    ]


def test_funkcijske_reci_dolaze_iz_linebreak_tabela():
    from subtitis.segment import function_words

    sr = function_words("sr")
    assert {"i", "ali", "s", "od", "kod", "je", "se"} <= sr
    assert "burger" not in sr and "meso" not in sr
    assert "the" in function_words("en")
    assert "et" in function_words("fr")


def test_would_flag_broji_obe_varijante():
    from subtitis.segment import would_flag

    # 3 loše funkcijske reči + 7 dobrih sadržajnih = 30% sada, 0% bez njih.
    chunk = sr_chunk(
        [("ali", 0.3), ("i", 0.3), ("od", 0.3)]
        + [(f"rec{k}", 0.95) for k in range(7)]
    )
    out = would_flag([chunk], "sr")
    assert out == {"sada": 1, "bez_funkcijskih": 0, "rečenica": 1}


def test_would_flag_ne_menja_sud_kad_su_greske_sadrzajne():
    from subtitis.segment import would_flag

    chunk = sr_chunk(
        [("grekota", 0.3), ("barbikiju", 0.3), ("smokovanje", 0.3)]
        + [(f"rec{k}", 0.95) for k in range(7)]
    )
    out = would_flag([chunk], "sr")
    assert out["sada"] == 1 and out["bez_funkcijskih"] == 1


def test_would_flag_recenica_od_samih_funkcijskih_reci():
    """Nema šta da se meri — sudi se po starom, ne ispada tiho čista."""
    from subtitis.segment import would_flag

    out = would_flag([sr_chunk([("pa", 0.3), ("i", 0.3), ("to", 0.3)])], "sr")
    assert out["sada"] == 1 and out["bez_funkcijskih"] == 1


def test_would_flag_line_je_merenje_a_ne_odluka():
    from subtitis.segment import would_flag_line

    line = would_flag_line({"sada": 5, "bez_funkcijskih": 3, "rečenica": 12})
    assert "5/12" in line and "3/12" in line
    assert "pravilo se ne menja" in line


# ==========================================================================
# Podrazumevana detekcija govora je isključena
# ==========================================================================


def test_podrazumevano_je_iskljuceno_u_sva_tri_interfejsa():
    """Tri ulaza u isti posao ne smeju da se razilaze u podrazumevanoj vrednosti.

    Dashboard bira po `DEFAULT_VAD_PRESET`, CLI po zastavici `--vad/--no-vad`,
    a biblioteka po podrazumevanom argumentu. Da se jedan od njih zaboravi,
    isti snimak bi kroz dva ulaza dao dva različita prepisa.
    """
    import inspect

    from subtitis import asr, pipeline

    assert asr.DEFAULT_VAD_PRESET == "iskljuceno"
    assert asr.vad_preset(None) == (False, None)
    assert inspect.signature(asr.transcribe).parameters["vad"].default is False
    assert inspect.signature(pipeline.run_transcribe).parameters["vad"].default is False


def test_cli_zastavica_je_podrazumevano_iskljucena():
    from typer.models import OptionInfo

    from subtitis import cli

    default = cli.transcribe.__defaults__
    vad_opt = [d for d in default if isinstance(d, OptionInfo) and "--vad/--no-vad" in d.param_decls]
    assert vad_opt and vad_opt[0].default is False


def test_izbori_praga_su_i_dalje_dostupni():
    """Isključeno je podrazumevano, ali oba praga moraju ostati na izbor."""
    from subtitis import asr

    assert asr.vad_preset("normalno") == (True, asr.VAD_THRESHOLD)
    assert asr.vad_preset("tih")[1] < asr.VAD_THRESHOLD
