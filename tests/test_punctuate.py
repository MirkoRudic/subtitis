"""Testovi restauracije interpunkcije i dijagnostike strukture.

Težište je na tome da model NE sme da promeni nijednu reč. Model povremeno
pokuša da „popravi" formulaciju — to mora da bude odbijeno, ne prihvaćeno.
"""

from __future__ import annotations

import pytest
from conftest import make_words

from subtitis.config import SubtitisError
from subtitis.models import Word
from subtitis.punctuate import (
    MAX_ATTEMPTS_PER_CHUNK,
    MIN_CHUNK_WORDS,
    PunctuateStats,
    apply_tokens,
    build_system_prompt,
    normalize_token,
    punctuate_words,
    restore_lowercase,
    retreat_point,
    starts_new_sentence,
    validate,
    word_sequence,
    would_pass,
)
from subtitis.segment import (
    MIN_AVG_WORDS_PER_SENTENCE,
    MIN_SENTENCE_DURATION,
    MIN_SENTENCE_WORDS,
    MIN_WORDS_FOR_DIAGNOSIS,
    PUNCTUATION_DENSITY_THRESHOLD,
    analyze,
    density_threshold,
    merge_short,
    segment,
)


def plain(text: str, dur=0.35, gap=0.25, start=0.0) -> list[Word]:
    out, t = [], start
    for k, w in enumerate(text.split()):
        out.append(Word(i=k, s=round(t, 3), e=round(t + dur, 3), w=w, p=0.9))
        t += dur + gap
    return out


# ==========================================================================
# Validacija
# ==========================================================================


def test_dodata_interpunkcija_prolazi():
    orig = ["paul", "et", "lucas", "on", "est", "parti"]
    accepted, err = validate("Paul et Lucas, on est parti.", orig)
    assert err is None
    produced, owner, _ = accepted
    assert "".join(produced).count(",") == 1
    assert owner == [0, 1, 2, 3, 4, 5], "bez razdvajanja je jedan-na-jedan"


def test_promenjena_rec_se_odbija():
    """Model „popravlja" formulaciju — mora da bude uhvaćeno."""
    orig = ["on", "est", "parti", "tres", "tot"]
    got, err = validate("On est parti très tôt.", orig)
    assert got is None
    assert "promenjena" in err.message


def test_dodata_rec_se_odbija():
    """Dodata reč pada i sa upaljenim dozvolama — nijedna je ne pokriva."""
    orig = ["on", "est", "parti"]
    got, err = validate("On est vraiment parti.", orig)
    assert got is None
    assert err.kind == "word"


def test_izbacena_rec_se_odbija():
    orig = ["on", "est", "parti", "tot"]
    got, err = validate("On est parti.", orig)
    assert got is None and err.kind == "count"


def test_promenjen_redosled_se_odbija():
    orig = ["lucas", "et", "paul"]
    got, err = validate("Paul et Lucas.", orig)
    assert got is None


def test_velika_slova_ne_smetaju():
    orig = ["paul", "est", "la"]
    got, err = validate("PAUL EST LA!", orig)
    assert err is None


def test_prazan_odgovor_se_odbija():
    got, err = validate("   ", ["a", "b"])
    assert got is None and err.message == "prazan odgovor"


def test_apostrof_se_ne_broji_kao_izmena():
    orig = ["l'ordinateur", "est", "là"]
    got, err = validate("L'ordinateur est là.", orig)
    assert err is None


def test_dodat_akcenat_je_izmena_reci():
    """Rule 3: ni pravopis se ne ispravlja — `la` -> `là` je promena reči."""
    got, err = validate("La est là.", ["la", "est", "la"])
    assert got is None and "promenjena" in err.message


def test_normalizacija():
    assert normalize_token("L'ordinateur,") == "lordinateur"
    assert normalize_token("...") == ""
    assert word_sequence("Paul, et Lucas.") == ["paul", "et", "lucas"]


# ==========================================================================
# Primena na tokene — vremena se ne diraju
# ==========================================================================


def test_vremena_reci_ostaju_netaknuta():
    words = plain("paul et lucas")
    before = [(w.s, w.e, w.p) for w in words]
    out = apply_tokens(words, ["Paul", "et", "Lucas", "."])
    kept = [(w.s, w.e, w.p) for w in out if any(c.isalnum() for c in w.w)]
    assert kept[0][0] == before[0][0]
    assert kept[1] == before[1]
    assert [w.w for w in out] == ["Paul", "et", "Lucas", "."]


def test_znak_uzima_odsecak_sa_kraja_prethodne_reci():
    words = plain("paul")
    out = apply_tokens(words, ["Paul", "."])
    rec, znak = out
    assert rec.s == words[0].s
    assert znak.e == pytest.approx(words[0].e)
    assert znak.s == rec.e
    assert 0 < znak.e - znak.s <= 0.09


def test_nedozvoljeni_znakovi_se_brisu():
    words = plain("paul est la")
    out = apply_tokens(words, ["Paul", "*", "est", "là".replace("à", "a"), "#"])
    assert all(any(c.isalnum() for c in w.w) for w in out)


def test_postojeca_interpunkcija_originala_se_zamenjuje():
    words = plain("paul est") + [Word(i=9, s=9.0, e=9.05, w=".", p=1.0)]
    out = apply_tokens(words, ["Paul", "est", "?"])
    assert [w.w for w in out] == ["Paul", "est", "?"]


def test_znak_pre_prve_reci_se_ignorise():
    out = apply_tokens(plain("paul"), [",", "Paul"])
    assert [w.w for w in out] == ["Paul"]


# ==========================================================================
# Čankovanje i pad na manji čank
# ==========================================================================


class FakeClient:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls: list[list[str]] = []
        from subtitis.translate import CallStats

        self.stats = CallStats()

    def chat(self, system: str, user: str) -> str:
        toks = user.split()
        self.calls.append(toks)
        self.stats.requests += 1
        return self.behaviour(toks)


def good(toks):
    return " ".join(toks).capitalize() + "."


def test_cankuje_po_zadatoj_velicini():
    words = plain(" ".join(f"r{k}" for k in range(25)))
    client = FakeClient(good)
    out, stats = punctuate_words(words, "fr", client=client, chunk_words=10, log=lambda m: None)
    assert [len(c) for c in client.calls] == [10, 10, 5]
    assert stats.chunks_ok == 3
    assert stats.words_in == 25


def test_neuspeh_polovi_pa_ostavlja_original():
    words = plain(" ".join(f"r{k}" for k in range(20)))
    client = FakeClient(lambda toks: "potpuno drugacije reci ovde")
    out, stats = punctuate_words(words, "fr", client=client, chunk_words=20, log=lambda m: None)
    assert stats.chunks_failed > 0
    assert stats.failed_ranges, "mora da prijavi šta je ostalo bez interpunkcije"
    assert [w.w for w in out] == [w.w for w in words], "original ostaje netaknut"


def test_jedan_los_cank_ne_obara_ceo_posao():
    words = plain(" ".join(f"r{k}" for k in range(20)))

    def flaky(toks):
        if "r0" in toks:
            return "smece"
        return good(toks)

    client = FakeClient(flaky)
    out, stats = punctuate_words(words, "fr", client=client, chunk_words=10, log=lambda m: None)
    assert stats.chunks_ok >= 1
    assert len(out) >= len(words)


def test_greska_veze_ne_polovi():
    def dead(toks):
        raise SubtitisError("Ollama ne odgovara")

    with pytest.raises(SubtitisError):
        punctuate_words(plain("a b c d e"), "fr", client=FakeClient(dead), log=lambda m: None)


def test_indeksi_su_prenumerisani():
    words = plain("paul et lucas")
    out, _ = punctuate_words(words, "fr", client=FakeClient(good), log=lambda m: None)
    assert [w.i for w in out] == list(range(len(out)))


def test_prazan_ulaz():
    out, stats = punctuate_words([], "fr", client=FakeClient(good), log=lambda m: None)
    assert out == [] and stats.words_in == 0


def test_prompt_sadrzi_jezik_i_zabranu_izmena():
    p = build_system_prompt("fr")
    assert "francuski" in p
    assert "Nijednu reč ne menjaj" in p
    assert "Ne prevodi" in p


# ==========================================================================
# Dijagnostika strukture
# ==========================================================================


def test_bez_interpunkcije_se_prijavljuje():
    words = plain("paul et lucas on est parti tres tot le matin")
    st = analyze(words, segment(words))
    assert st.no_punctuation
    assert st.sentence_end_marks == 0
    assert "nije vratio nijedan znak" in st.warning()


#: 50 reči — dovoljno da pređe MAX_SENTENCE_DURATION i da uđe u dijagnostiku.
FRENCH_WORDS = (
    "paul et lucas et le technicien on est parti tres tot le matin pour "
    "arriver avant midi et on a tout installe dans la salle avant que les "
    "invites arrivent vers quatorze heures est ce que tu te souviens de ce "
    "jour la moi je m en souviens tres bien"
)


def dotted(text: str) -> list[Word]:
    """Svaka reč sa tačkom iza sebe — kako Whisper ume da uradi na lošem zvuku."""
    toks: list[Word] = []
    t = 0.0
    for w in text.split():
        toks.append(Word(i=len(toks), s=round(t, 3), e=round(t + 0.3, 3), w=w, p=0.9))
        toks.append(Word(i=len(toks), s=round(t + 0.3, 3), e=round(t + 0.33, 3), w=".", p=0.9))
        t += 0.5
    return toks


def test_znak_posle_svake_reci_je_takodje_kvar():
    """Nije samo odsustvo znakova problem — i znak po reči ruši segmentaciju."""
    toks = dotted(FRENCH_WORDS)
    st = analyze(toks, [])  # bez mreže: onoliko rečenica koliko je granica
    raw = len([w for w in toks if w.w == "."])
    st = analyze(toks, [None] * raw)
    assert not st.no_punctuation
    assert st.failed, "prosek reči po rečenici mora da padne ispod praga"
    assert "posuti po svakoj reči" in st.warning()


def test_kratak_uzorak_nije_dokaz_kvara():
    """Dve kratke rečenice imaju nizak prosek a savršenu interpunkciju."""
    words = make_words("Dobar dan. Hvala.")
    st = analyze(words, segment(words))
    assert st.avg_words_per_sentence < MIN_AVG_WORDS_PER_SENTENCE
    assert not st.failed
    assert st.warning() is None


def test_uredan_tekst_ne_dize_upozorenje():
    words = make_words("Dobar dan, hvala što ste došli. Danas pričamo o važnoj temi.")
    st = analyze(words, segment(words))
    assert st.warning() is None
    assert not st.needs_punctuation()
    assert st.sentence_end_marks == 2


def test_prosek_i_udeo_velikih_slova():
    words = make_words("Dobar dan. Hvala vam.")
    st = analyze(words, segment(words))
    assert st.sentences == 2
    assert st.avg_words_per_sentence == pytest.approx(4 / 2)
    assert st.capitalized_share == pytest.approx(0.5)


# ==========================================================================
# Kapija po gustini interpunkcije
#
# Treći kvar: Whisper punktira početak pa prestane. Znakova ima, prosek
# rečenice je pristojan — a tri četvrtine transkripta je bez ijednog znaka.
# ==========================================================================


def sparse(n_words: int, n_marks: int) -> list[Word]:
    """Tekst zadate dužine sa tačno `n_marks` tačaka, sve na početku."""
    toks: list[Word] = []
    t = 0.0
    for k in range(n_words):
        toks.append(Word(i=len(toks), s=round(t, 3), e=round(t + 0.3, 3), w="reč", p=0.9))
        t += 0.35
        if k < n_marks:
            toks.append(Word(i=len(toks), s=round(t, 3), e=round(t + 0.03, 3), w=".", p=0.9))
    return toks


def test_retka_interpunkcija_pusta_restauraciju():
    """Francuski test 2026-08-12: 10 znakova na 428 reči."""
    st = analyze(sparse(400, 10), [None] * 10)
    assert not st.no_punctuation, "znakova IMA — stara provera ovo ne hvata"
    assert not st.failed, "prosek rečenice je pristojan — ni ova ne hvata"
    assert st.sparse_punctuation("fr")
    assert st.needs_punctuation("fr")
    assert "PREMALO" in st.warning("fr")


def test_gusta_interpunkcija_ne_pusta_restauraciju():
    """Srpski test 2026-08-12: 19 znakova na 348 reči."""
    st = analyze(sparse(348, 19), [None] * 19)
    assert not st.sparse_punctuation("sr")
    assert not st.needs_punctuation("sr")
    assert st.warning("sr") is None


def test_kratak_uzorak_ne_pada_na_gustinu():
    """Isti prag na 10 reči bi obarao svaki kratak snimak."""
    st = analyze(sparse(10, 0), [None])
    assert st.words < MIN_WORDS_FOR_DIAGNOSIS
    assert not st.sparse_punctuation()


def test_prag_je_po_jeziku():
    PUNCTUATION_DENSITY_THRESHOLD["xx"] = 1.0
    try:
        st = analyze(sparse(400, 10), [None] * 10)  # 2.5/100 reči
        assert st.sparse_punctuation("fr"), "podrazumevani prag 4.0 ovo hvata"
        assert not st.sparse_punctuation("xx"), "prag 1.0 za xx ovo pušta"
    finally:
        del PUNCTUATION_DENSITY_THRESHOLD["xx"]


def test_nepoznat_jezik_uzima_podrazumevani_prag():
    assert density_threshold("klingonski") == PUNCTUATION_DENSITY_THRESHOLD["default"]
    assert density_threshold(None) == PUNCTUATION_DENSITY_THRESHOLD["default"]


def test_bez_znakova_i_dalje_pada_na_staroj_proveri():
    """Gustina 0 je poseban slučaj — poruka mora ostati konkretnija."""
    st = analyze(sparse(400, 0), [None])
    assert st.no_punctuation
    assert st.needs_punctuation("fr")
    assert "nije vratio nijedan znak" in st.warning("fr")


def test_dijagnostika_meri_i_zareze():
    """Zarezi su za `linebreak` važniji od tačaka, pa se mere i kad kapija ćuti."""
    words = make_words("Dobar dan, hvala što ste došli. Danas pričamo, opet, o temi.")
    st = analyze(words, segment(words))
    assert st.commas == 3
    assert st.commas_per_100_words == pytest.approx(100 * 3 / st.words)
    line = st.density_line("sr")
    assert "zareza/100 reči" in line
    assert "znaka kraja/100 reči" in line


def test_dijagnostika_ispisuje_i_kad_je_sve_u_redu():
    """Red se ispisuje posle SVAKE transkripcije — zato skuplja merenja."""
    words = make_words("Dobar dan, hvala što ste došli. Danas pričamo o važnoj temi.")
    st = analyze(words, segment(words))
    assert st.warning() is None
    assert "u redu" in st.density_line("sr")


def test_dijagnostika_kaze_sta_ce_se_desiti():
    st = analyze(sparse(400, 10), [None] * 10)
    assert "restauracija" in st.density_line("fr")


def test_prazan_ulaz_u_dijagnostiku():
    st = analyze([], [])
    assert st.words == 0
    assert "nije vratio nijednu reč" in st.warning()


# ==========================================================================
# Zaštitna mreža: minimalna dužina rečenice
# ==========================================================================


def chunks_of(text: str, **kw) -> list[list[Word]]:
    return [plain(part.strip(), **kw) for part in text.split("|")]


def test_kratke_recenice_se_spajaju():
    parts = merge_short(chunks_of("paul|et|lucas|on est parti tres tot"))
    assert len(parts) < 4
    for p in parts:
        assert len([w for w in p if any(c.isalnum() for c in w.w)]) >= MIN_SENTENCE_WORDS


def test_jasna_interpunkcija_izuzima_od_spajanja():
    a = plain("da") + [Word(i=9, s=1.0, e=1.05, w=".", p=1.0)]
    b = plain("posle toga smo otisli", start=2.0)
    out = merge_short([a, b], trust_punctuation=True)
    assert len(out) == 2, "„Da.” je legitimna kratka rečenica"


def test_izuzece_pada_kad_se_interpunkciji_ne_veruje():
    """Tačka posle svake reči nije „jasna interpunkcija”."""
    a = plain("paul") + [Word(i=9, s=1.0, e=1.05, w=".", p=1.0)]
    b = plain("et", start=2.0) + [Word(i=19, s=3.0, e=3.05, w=".", p=1.0)]
    c = plain("lucas", start=4.0) + [Word(i=29, s=5.0, e=5.05, w=".", p=1.0)]
    out = merge_short([a, b, c], trust_punctuation=False)
    assert len(out) == 1


def test_poslednji_kratak_komad_se_lepi_na_prethodni():
    a = plain("ovo je duza recenica koja stoji sama")
    b = plain("da", start=10.0)
    out = merge_short([a, b])
    assert len(out) == 1


def test_spajanje_ne_dira_dovoljno_duge():
    a = plain("prva recenica ima dovoljno reci")
    b = plain("druga recenica takodje ima dosta", start=10.0)
    assert len(merge_short([a, b])) == 2


def test_segment_ne_pravi_jednu_rec_po_recenici():
    """Regresija: prijavljeno je da svaka reč postaje zasebna rečenica."""
    toks = dotted(FRENCH_WORDS)
    s = segment(toks)
    words = sum(1 for x in toks if any(c.isalnum() for c in x.w))
    assert words / len(s) >= MIN_AVG_WORDS_PER_SENTENCE
    for x in s:
        assert len(x.text.split()) > 1, f"jednoreč rečenica: {x.text!r}"


def test_lomljenje_bez_pauza_ne_ljusti_rec_po_rec():
    """PRAVI uzrok bага „svaka reč je zasebna rečenica".

    Bez interpunkcije i sa ujednačenim pauzama, `max([(pauza, k), …])` je pri
    izjednačenju birao NAJVEĆI indeks, pa se rez pravio na poslednjoj reči i
    rekurzija je ljuštila po jednu reč sa kraja.
    """
    words = plain(FRENCH_WORDS, dur=0.32, gap=0.12)  # ravnomerne pauze, 22 s
    assert words[-1].e - words[0].s > 15.0, "mora da pređe MAX_SENTENCE_DURATION"
    s = segment(words)
    assert all(len(x.text.split()) > 1 for x in s), [x.text for x in s]
    assert len(s) <= 4, "rez mora biti kod sredine, ne na kraju"


@pytest.mark.parametrize("gap", [0.05, 0.12, 0.30, 0.50, 0.69])
def test_bez_interpunkcije_nijedna_recenica_nije_jednorec(gap):
    words = plain(FRENCH_WORDS, dur=0.32, gap=gap)
    for x in segment(words):
        assert len(x.text.split()) > 1, x.text


def test_rez_pada_blizu_sredine():
    words = plain(FRENCH_WORDS, dur=0.32, gap=0.12)
    s = segment(words)
    mid = (words[0].s + words[-1].e) / 2
    assert abs(s[0].e - mid) < 2.0, f"prvi rez na {s[0].e:.1f}s, sredina je {mid:.1f}s"


def test_konstante_su_izlozene():
    assert MIN_SENTENCE_WORDS == 3
    assert MIN_SENTENCE_DURATION == 1.0
    assert MIN_AVG_WORDS_PER_SENTENCE == 3.0
    assert MAX_ATTEMPTS_PER_CHUNK >= 1 and MIN_CHUNK_WORDS >= 1


# ==========================================================================
# Potpis greške: isti uzrok mora imati isti potpis bez obzira na veličinu čanka
# ==========================================================================


def test_isti_manjak_reci_ima_isti_potpis():
    """60->61 i 30->31 je ISTA greška — inače polovljenje ne zna da stane."""
    _, a = validate("a b c d", ["a", "b", "c"])
    _, b = validate("a b c", ["a", "b"])
    assert a.kind == b.kind == "count"
    assert a.signature == b.signature == "count:+1"


def test_ista_promenjena_rec_ima_isti_potpis():
    """Ista izmena je isto u čanku od 60 i u čanku od 15."""
    _, a = validate("x chat y", ["x", "chien", "y"])
    _, b = validate("chat z", ["chien", "z"])
    assert a.signature == b.signature == "word:chien->chat"


def test_razlicite_greske_imaju_razlicite_potpise():
    _, a = validate("a b c d", ["a", "b", "c"])
    _, b = validate("a x c", ["a", "b", "c"])
    assert a.signature != b.signature


# ==========================================================================
# ALLOW_WORD_SPLITS — `ya` -> `y a`
# ==========================================================================


def test_razdvajanje_se_odbija_kad_je_zabranjeno():
    _, err = validate(
        "s'il y a lieu.", ["s'il", "ya", "lieu"], allow_splits=False
    )
    assert err is not None
    assert err.kind == "count"


def test_razdvajanje_prolazi_podrazumevano():
    """`ALLOW_WORD_SPLITS` je upaljen od 2026-08-12."""
    accepted, err = validate("s'il y a lieu.", ["s'il", "ya", "lieu"])
    assert err is None
    produced, owner, concessions = accepted
    # `y` i `a` pripadaju istoj originalnoj reči (`ya`, indeks 1).
    assert owner == [0, 1, 1, 2]
    # Ustupak se BELEŽI i kad je dozvoljen — da se vidi u logu.
    assert [str(c) for c in concessions] == ["split: 'ya' -> 'y a'"]


def test_razdvajanje_koje_se_ne_poklapa_i_dalje_pada():
    """Provera je mehanička: spojeno mora biti tačno jednako razdvojenom."""
    _, err = validate("s'il y avait besoin.", ["s'il", "ya", "lieu"])
    assert err is not None


def test_razdvajanje_ne_dozvoljava_dodavanje_reci():
    _, err = validate("on est vraiment parti.", ["on", "est", "parti"])
    assert err is not None


def test_razdvojena_rec_deli_svoje_vreme():
    """Nova reč ne sme da izmisli vreme — uzima deo vremena originala."""
    words = [Word(i=0, s=1.0, e=2.0, w="ya", p=0.8)]
    accepted, err = validate("y a", ["ya"])
    assert err is None
    produced, owner, _ = accepted
    out = apply_tokens(words, produced, owner)
    assert [w.w for w in out] == ["y", "a"]
    assert out[0].s == 1.0
    assert out[-1].e == 2.0
    assert out[0].e == out[1].s, "delovi se nadovezuju bez rupe"
    assert all(w.p == 0.8 for w in out)


# ==========================================================================
# ALLOW_INFLECTION_CHANGES — `généreuse` -> `généreuses`
# ==========================================================================


def test_nastavak_se_odbija_kad_je_zabranjen():
    _, err = validate(
        "Personnes généreuses.", ["personnes", "généreuse"],
        allow_inflection=False,
    )
    assert err is not None


def test_nastavak_prolazi_podrazumevano():
    """`ALLOW_INFLECTION_CHANGES` je upaljen od 2026-08-12."""
    accepted, err = validate("Personnes généreuses.", ["personnes", "généreuse"])
    assert err is None
    assert [str(c) for c in accepted.concessions] == [
        "inflection: 'généreuse' -> 'généreuses'"
    ]


def test_nastavak_ne_pusta_dijakritiku():
    """Dve dozvole su odvojene: nastavak nije akcenat."""
    _, err = validate("très", ["tres"])
    assert err is not None


def test_nastavak_ne_pusta_drugu_rec():
    """`chien` -> `chat` nije nastavak nego druga reč."""
    _, err = validate("Le chat.", ["le", "chien"])
    assert err is not None


def test_nastavak_ne_pusta_predugu_razliku():
    _, err = validate("parlementaire", ["parle"])
    assert err is not None


def test_nastavak_ne_pusta_prekratku_osnovu():
    """Dvoslovna „osnova" nije osnova — `de` -> `des` bi prošlo svuda."""
    _, err = validate("des", ["de"])
    assert err is not None


def test_nastavak_zadrzava_model_ov_ispis():
    words = [Word(i=0, s=0.0, e=1.0, w="généreuse", p=0.9)]
    accepted, _ = validate("généreuses", ["généreuse"])
    produced, owner, _ = accepted
    out = apply_tokens(words, produced, owner)
    assert out[0].w == "généreuses"
    assert (out[0].s, out[0].e) == (0.0, 1.0), "vreme se ne dira"


# ==========================================================================
# Merenje dozvola dok su isključene
# ==========================================================================


def test_merenje_javlja_sta_bi_prosla_dozvola_spasila():
    wp = would_pass("s'il y a lieu.", ["s'il", "ya", "lieu"])
    assert wp["ALLOW_WORD_SPLITS"] is True
    assert wp["ALLOW_INFLECTION_CHANGES"] is False
    assert wp["oba"] is True


def test_merenje_javlja_i_kad_nista_ne_bi_pomoglo():
    wp = would_pass("Paul et Lucas sont partis.", ["lucas", "et", "paul"])
    assert not any(wp.values())


# ==========================================================================
# Polovljenje staje kad ne pomaže
#
# Iz loga 2026-08-12: model dosledno razdvaja `ya` u `y a`, pa ista greška ide
# 60->61, 30->31, 15->16. Šest poziva da bi se saznalo ono što je prvi rekao.
# ==========================================================================


def adds_one_word(toks):
    """Doda jednu reč, ali samo u čanku sa `ya` — kao u pravom logu.

    Dodata reč, za razliku od `ya` -> `y a`, ne prolazi ni uz jednu dozvolu:
    ovde se meri zaustavljanje polovljenja, ne dozvole.
    """
    if "ya" in toks:
        return " ".join([*toks, "visak"]).capitalize() + "."
    return good(toks)


def test_ista_greska_u_podcanku_zaustavlja_polovljenje():
    words = plain(" ".join(["ya" if k == 5 else f"r{k}" for k in range(40)]))
    client = FakeClient(adds_one_word)
    _, stats = punctuate_words(
        words, "fr", client=client, chunk_words=40, log=lambda m: None
    )
    assert stats.halvings == 1, "prvo polovljenje se dozvoljava"
    assert stats.halvings_skipped >= 1, "drugo ne — greška je ista"
    assert stats.failed_ranges


def test_zaustavljanje_stedi_pozive():
    """2 pokušaja na 0-40, polovljenje, 2 na polovini sa greškom, 1 na čistoj."""
    words = plain(" ".join(["ya" if k == 5 else f"r{k}" for k in range(40)]))
    client = FakeClient(adds_one_word)
    punctuate_words(words, "fr", client=client, chunk_words=40, log=lambda m: None)
    assert len(client.calls) == 5


def test_razlicita_greska_u_podcanku_nastavlja_polovljenje():
    """Zaustavlja se samo ISTA greška — nova greška zaslužuje novi pokušaj."""

    def flaky(toks):
        if len(toks) > 50:
            return " ".join(toks) + " visak."                  # count:+1
        if "r5" in toks:
            return " ".join(toks).replace("r5", "r9") + "."    # word:r5->r9
        return good(toks)

    words = plain(" ".join(f"r{k}" for k in range(100)))
    _, stats = punctuate_words(
        words, "fr", client=FakeClient(flaky), chunk_words=100, log=lambda m: None
    )
    # 0-100 padne na count:+1, polovi se; 0-50 padne na word:r5->r9 — DRUGI
    # potpis, pa se polovi ponovo. Tek 0-25 sa istim potpisom staje.
    assert stats.halvings == 2
    assert stats.halvings_skipped == 1


def test_potpuni_neuspeh_i_dalje_ostavlja_original():
    """Zaustavljanje polovljenja ne sme da izgubi nijednu reč."""
    words = plain(" ".join(["ya" if k == 3 else f"r{k}" for k in range(30)]))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(adds_one_word), chunk_words=30,
        log=lambda m: None,
    )
    assert stats.failed_ranges
    kept = [w.w.lower() for w in out if any(c.isalnum() for c in w.w)]
    assert kept == [w.w.lower() for w in words], "nijedna reč ne sme da nestane"
    # Opseg koji je odustao ostaje bez ijednog dodatog znaka.
    lo, hi = stats.failed_ranges[0]
    assert [w.w for w in out][lo:hi] == [w.w for w in words][lo:hi]


# ==========================================================================
# Merenje: koliko je trajalo i koliko je bačeno
# ==========================================================================


def test_meri_bacene_odgovore():
    """Lažni klijent odgovara trenutno, pa se meri odnos, ne apsolutno vreme."""
    words = plain(" ".join(["ya" if k == 5 else f"r{k}" for k in range(40)]))
    _, stats = punctuate_words(
        words, "fr", client=FakeClient(adds_one_word), chunk_words=40,
        log=lambda m: None,
    )
    assert stats.seconds_wasted <= stats.seconds_total
    assert stats.chunks_failed == 4, "4 bačena odgovora se broje kao potrošeni"


def test_ispis_vremena_kaze_koliko_je_baceno():
    stats = PunctuateStats(
        chunks_ok=10, chunks_failed=10, seconds_total=103.0, seconds_wasted=62.0
    )
    assert stats.wasted_share == pytest.approx(62 / 103)
    line = stats.timing_line()
    assert "103s ukupno" in line
    assert "62s (60%) na odbijene odgovore" in line


def test_ispis_vremena_bez_deljenja_nulom():
    assert PunctuateStats().wasted_share == 0.0
    assert "0s ukupno" in PunctuateStats().timing_line()


def test_bez_odbijanja_nema_bacenog_vremena():
    words = plain(" ".join(f"r{k}" for k in range(20)))
    _, stats = punctuate_words(
        words, "fr", client=FakeClient(good), chunk_words=10, log=lambda m: None
    )
    assert stats.seconds_wasted == 0.0
    assert stats.wasted_share == 0.0
    assert stats.would_pass_line() is None, "nema odbijenih — nema šta da se meri"


def test_dozvoljena_izmena_se_belezi_u_statistici():
    """Dozvola je upaljena — ali svaka izmena koju propusti mora u log.

    Ovo je druga strana `would_pass`: kad se dozvoli, mora da se vidi ŠTA je
    dozvoljeno, da se uhvati izmena koja nije pravopisna nego značenjska.
    """

    def splits_ya(toks):
        out = []
        for t in toks:
            out.extend(["y", "a"] if t == "ya" else [t])
        return " ".join(out).capitalize() + "."

    words = plain(" ".join(["ya" if k == 5 else f"r{k}" for k in range(20)]))
    _, stats = punctuate_words(
        words, "fr", client=FakeClient(splits_ya), chunk_words=20,
        log=lambda m: None,
    )
    assert stats.chunks_failed == 0, "dozvola ga pušta"
    assert stats.concessions == {"split": 1}
    assert stats.concession_examples == ["split: 'ya' -> 'y a'"]
    assert "'ya' -> 'y a'" in stats.concessions_line()


def test_bez_ustupaka_nema_reda_u_logu():
    words = plain(" ".join(f"r{k}" for k in range(10)))
    _, stats = punctuate_words(
        words, "fr", client=FakeClient(good), chunk_words=10, log=lambda m: None
    )
    assert stats.concessions == {}
    assert stats.concessions_line() is None


def test_merenje_javlja_i_kad_nijedna_dozvola_ne_pomaze():
    words = plain(" ".join(f"r{k}" for k in range(20)))
    _, stats = punctuate_words(
        words, "fr", client=FakeClient(lambda t: "potpuno druge reci ovde"),
        chunk_words=20, log=lambda m: None,
    )
    assert stats.chunks_failed
    assert "nijedno popuštanje" in stats.would_pass_line()


# ==========================================================================
# Povlačenje na poslednji znak
#
# Model svaki čank punktira kao zaokruženu celinu, pa granica čanka postane
# granica rečenice. Izmereno na francuskom testu: 4 od 6 granica dobile lom
# koji tamo ne pripada, broj rečenica skočio 21 -> 28.
# ==========================================================================


def marks_at(*positions: int):
    """Model koji stavi tačku posle zadatih reči (0-bazirano) i na kraj."""

    def behave(toks):
        out = []
        for k, t in enumerate(toks):
            out.append(t)
            if k in positions and k != len(toks) - 1:
                out.append(".")
        text = " ".join(out)
        return (text if text.endswith(".") else text + ".").replace(" .", ".")

    return behave


def test_rep_se_vraca_u_sledeci_poziv():
    """Čank 0-10 sa tačkom posle 6. reči: prihvata se 7, ostatak ide dalje."""
    words = plain(" ".join(f"r{k}" for k in range(20)))
    client = FakeClient(marks_at(6))
    _, stats = punctuate_words(
        words, "fr", client=client, chunk_words=10, log=lambda m: None
    )
    assert [len(c) for c in client.calls][0] == 10
    assert client.calls[1][0] == "r7", "sledeći čank počinje od repa"
    assert stats.retreats >= 1
    assert stats.retreat_words >= 3


def test_veštacka_tacka_sa_kraja_canka_se_odbacuje():
    """Ono zbog čega sve ovo postoji: lom na granici čanka."""
    words = plain(" ".join(f"r{k}" for k in range(20)))
    out, _ = punctuate_words(
        words, "fr", client=FakeClient(marks_at(6)), chunk_words=10,
        log=lambda m: None,
    )
    toks = [w.w for w in out]
    # Posle r9 (kraj prvog čanka) NE sme da stoji tačka — tamo je model
    # zatvorio čank, ne rečenicu.
    i = toks.index("r9")
    assert toks[i + 1] != ".", "veštačka granica čanka je ostala"


def test_poslednji_cank_se_ne_povlaci():
    """Iza poslednje reči stvarno nema ničega — tačka tamo je prava."""
    words = plain(" ".join(f"r{k}" for k in range(10)))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(marks_at(6)), chunk_words=10,
        log=lambda m: None,
    )
    assert stats.retreats == 0
    assert [w.w for w in out][-1] == "."


def test_nijedna_rec_se_ne_gubi_ni_dodaje():
    for chunk in (7, 10, 13):
        words = plain(" ".join(f"r{k}" for k in range(30)))
        out, _ = punctuate_words(
            words, "fr", client=FakeClient(marks_at(3, 6, 9)), chunk_words=chunk,
            log=lambda m: None,
        )
        kept = [w.w for w in out if any(c.isalnum() for c in w.w)]
        assert kept == [w.w for w in words], f"chunk_words={chunk}"


def test_vremena_ostaju_rastuca_posle_povlacenja():
    words = plain(" ".join(f"r{k}" for k in range(30)))
    out, _ = punctuate_words(
        words, "fr", client=FakeClient(marks_at(4, 8)), chunk_words=10,
        log=lambda m: None,
    )
    times = [(w.s, w.e) for w in out]
    assert all(a[0] <= b[0] for a, b in zip(times, times[1:])), "vremena se ne mešaju"


# --- zaštita od beskonačnog repa -------------------------------------------


def test_cank_bez_unutrasnjeg_znaka_se_prihvata_ceo():
    """Nema gde da se povuče — tačka je samo na kraju."""
    words = plain(" ".join(f"r{k}" for k in range(20)))
    client = FakeClient(good)  # tačka samo na kraju
    _, stats = punctuate_words(
        words, "fr", client=client, chunk_words=10, log=lambda m: None
    )
    assert stats.retreats == 0
    assert [len(c) for c in client.calls] == [10, 10], "nema vrtenja"


def test_predugacak_rep_se_ne_vraca():
    """Znak posle 1. reči bi vratio 90% čanka — zaštita to preseca."""
    words = plain(" ".join(f"r{k}" for k in range(20)))
    client = FakeClient(marks_at(0))
    logs: list[str] = []
    _, stats = punctuate_words(
        words, "fr", client=client, chunk_words=10, log=logs.append
    )
    assert stats.retreats == 0
    assert stats.retreats_forced >= 1
    assert [len(c) for c in client.calls] == [10, 10]
    assert any("prihvatam čank kakav jeste" in m for m in logs)


def test_rep_tacno_na_granici_se_jos_prihvata():
    """Rep od pola čanka je granični slučaj: preko pola se preseca."""
    words = plain(" ".join(f"r{k}" for k in range(30)))
    client = FakeClient(marks_at(4))  # prihvata 5 od 10, rep 5 = tačno pola
    _, stats = punctuate_words(
        words, "fr", client=client, chunk_words=10, log=lambda m: None
    )
    assert stats.retreats >= 1, "tačno pola nije 'preko pola'"


def test_povlacenje_uvek_napreduje():
    """Ma šta model vraćao, petlja mora da dođe do kraja."""
    import itertools

    for pos in itertools.chain(range(0, 6), [None]):
        words = plain(" ".join(f"r{k}" for k in range(25)))
        behave = good if pos is None else marks_at(pos)
        out, _ = punctuate_words(
            words, "fr", client=FakeClient(behave), chunk_words=8,
            log=lambda m: None,
        )
        kept = [w.w.lower() for w in out if any(c.isalnum() for c in w.w)]
        assert kept == [w.w for w in words], f"pozicija znaka {pos}"


def test_odustali_cank_ne_pravi_prazan_hod():
    """Kad čank odustane, potroši svoj opseg — inače bi se vrteo u mestu."""
    words = plain(" ".join(f"r{k}" for k in range(20)))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(lambda t: "potpuno druge reci ovde"),
        chunk_words=20, log=lambda m: None,
    )
    assert stats.failed_ranges == [(0, 20)]
    assert len([w for w in out if any(c.isalnum() for c in w.w)]) == 20


# --- sama funkcija ---------------------------------------------------------


def test_retreat_point_nalazi_poslednji_znak():
    produced = ["A", "b", ".", "C", "d", ".", "E", "f", "."]
    owner = [0, 1, 2, 3, 4, 5]
    point = retreat_point(produced, owner)
    assert point is not None
    tokens, taken = point
    assert produced[:tokens] == ["A", "b", ".", "C", "d", "."]
    assert taken == 4


def test_retreat_point_ignorise_znak_na_kraju():
    assert retreat_point(["A", "b", "c", "."], [0, 1, 2]) is None


def test_retreat_point_bez_znakova():
    assert retreat_point(["A", "b", "c"], [0, 1, 2]) is None


def test_retreat_point_prazan_ulaz():
    assert retreat_point([], []) is None


def test_retreat_point_zarez_nije_kraj_recenice():
    """Povlači se na kraj REČENICE, ne na bilo koji znak."""
    assert retreat_point(["A", ",", "b", "c", "."], [0, 1, 2]) is None


def test_retreat_point_racuna_razdvojene_reci():
    """Uz ALLOW_WORD_SPLITS jedna originalna reč može dati dva tokena."""
    produced = ["S'il", "y", "a", ".", "Lieu", "."]
    owner = [0, 1, 1, 2]          # `ya` -> `y a`
    tokens, taken = retreat_point(produced, owner)
    assert tokens == 4
    assert taken == 2, "tri dobijene reči pokrivaju dve originalne"


# ==========================================================================
# Malo slovo na prvoj reči čanka
#
# Povlačenje skida veštačku TAČKU sa kraja čanka, ali model i dalje piše prvu
# reč sledećeg čanka velikim slovom — pa usred rečenice ostane `revoir Ce
# genre` i `ce Genre de salle` (oblik izmeren na francuskom testu).
# ==========================================================================


def test_veliko_slovo_usred_recenice_se_vraca():
    assert restore_lowercase(["Ce", "genre"], ["ce", "genre"]) == ["ce", "genre"]


def test_vlastito_ime_ostaje_veliko():
    """`Paul`, `Serbie`, `France` su i u originalu veliki — ne diraju se."""
    for ime in ("Paul", "Serbie", "France"):
        assert restore_lowercase([ime, "est"], [ime, "est"]) == [ime, "est"]


def test_rec_koja_je_i_bila_mala_a_model_je_ostavio_malom():
    assert restore_lowercase(["ce", "genre"], ["ce", "genre"]) == ["ce", "genre"]


def test_dira_se_samo_prva_rec():
    """Veliko slovo dublje u čanku je model-ova odluka o rečenici, ne o granici."""
    out = restore_lowercase(["ce", "genre", ".", "Ce", "autre"], ["ce", "genre", "ce", "autre"])
    assert out == ["ce", "genre", ".", "Ce", "autre"]


def test_znak_ispred_prve_reci_se_preskace():
    assert restore_lowercase(["—", "Ce", "genre"], ["ce", "genre"]) == ["—", "ce", "genre"]


def test_prazan_ulaz_u_vracanje_malog_slova():
    assert restore_lowercase([], []) == []
    assert restore_lowercase(["Ce"], []) == ["Ce"]


def test_granica_recenice_se_prepoznaje():
    assert starts_new_sentence([]) is True
    assert starts_new_sentence(plain("paul") + [Word(i=1, s=1.0, e=1.05, w=".", p=1.0)])
    assert not starts_new_sentence(plain("paul"))
    assert not starts_new_sentence(plain("paul") + [Word(i=1, s=1.0, e=1.05, w=",", p=1.0)])


def test_prvi_cank_zadrzava_veliko_slovo():
    """Početak teksta JESTE početak rečenice."""
    words = plain(" ".join(f"r{k}" for k in range(20)))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(good), chunk_words=10, log=lambda m: None
    )
    assert [w.w for w in out][0] == "R0"
    assert stats.decapitalized == 0, "svaki čank je ovde počeo posle tačke"


def test_posle_povlacenja_veliko_slovo_ostaje():
    """Povlačenje seče TAČNO na granici rečenice — tamo je veliko slovo tačno."""

    def marks_inside(toks):
        out = []
        for k, t in enumerate(toks):
            out.append(t.capitalize() if k == 0 else t)
            if k == 6 and k != len(toks) - 1:
                out.append(".")
        text = " ".join(out).replace(" .", ".")
        return text if text.endswith(".") else text + "."

    words = plain(" ".join(f"r{k}" for k in range(20)))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(marks_inside), chunk_words=10,
        log=lambda m: None,
    )
    assert stats.retreats >= 1
    assert stats.decapitalized == 0


def test_ce_genre_ne_ostaje_posle_odustalog_canka():
    """Regresija na tačan slučaj iz francuskog testa.

    Odustali čank ostaje BEZ ijednog znaka, pa sledeći čank ne počinje na
    granici rečenice — a model mu ipak stavi veliko slovo. Tako je nastalo
    `...avec les techniciens donc on pourra revoir Ce genre...` (primer
    izmišljen, oblik isti).
    """

    def fails_first(toks):
        if "r0" in toks:
            return "potpuno druge reci ovde"
        return " ".join(toks).capitalize() + "."

    words = plain(" ".join(f"r{k}" for k in range(40)))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(fails_first), chunk_words=20,
        log=lambda m: None,
    )
    assert stats.failed_ranges == [(0, 20)]
    assert stats.decapitalized == 1
    toks = [w.w for w in out]
    for k, t in enumerate(toks[1:], start=1):
        if not t[:1].isupper():
            continue
        prev = next((x for x in reversed(toks[:k]) if x.strip()), "")
        assert prev in ".!?…", f"{t!r} je veliko a nije posle znaka: {toks[max(0, k - 3):k + 1]}"


def test_vlastito_ime_posle_odustalog_canka_ostaje_veliko():
    """Ista situacija, ali reč je i u originalu bila velika."""

    def fails_first(toks):
        if "r0" in toks:
            return "potpuno druge reci ovde"
        return " ".join(toks) + "."

    words = plain(" ".join(["Paul" if k == 20 else f"r{k}" for k in range(40)]))
    out, stats = punctuate_words(
        words, "fr", client=FakeClient(fails_first), chunk_words=20,
        log=lambda m: None,
    )
    assert stats.decapitalized == 0
    assert "Paul" in [w.w for w in out]
