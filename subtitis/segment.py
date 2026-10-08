"""Reči -> rečenice.

Ulaz je lista `Word` iz `words.<lang>.json` (interpunkcija su zasebne stavke sa
sopstvenim vremenom). Izlaz su `Sentence` objekti sa inkluzivnim rasponom
indeksa reči, preko koga se prelamanje vraća na tačna vremena.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Sentence, Word

# --- Konstante za štelovanje ----------------------------------------------

#: Znaci koji zatvaraju rečenicu.
SENTENCE_END_CHARS = ".!?…"

#: Rečenica duža od ovoga se nasilno lomi (sekunde).
MAX_SENTENCE_DURATION = 15.0

#: Pauza koja se smatra dovoljno jasnom granicom za nasilno lomljenje (sekunde).
HARD_SPLIT_MIN_PAUSE = 0.700

#: Ispod ove ASR verovatnoće reč je „sumnjiva". Ovaj prag upravlja BOJENJEM
#: pojedinačnih reči u pregledu i ne dira se — na izmerenim poslovima obeležava
#: 3.8% reči na dobrom i 14.5% na lošem zvuku, što je tačna gustina.
LOW_CONFIDENCE_THRESHOLD = 0.6

#: Udeo sumnjivih reči od kog se OZNAČAVA CELA REČENICA.
#:
#: Do 2026-08-19 je pravilo bilo „ima bar jednu sumnjivu reč". Pri prosečnoj
#: rečenici od 15 reči to je pojačivač, ne mera: 3.8% loših reči daje 41%
#: označenih rečenica, a 14.5% daje 80%. Oznaka rečenice vodi navigaciju
#: (skok na sledeću označenu); navigacija koja staje na četiri od pet rečenica
#: ne vodi nigde.
#:
#: Prag po reči je pri tome bio nevin — merenje raspodele je pokazalo da je
#: gustina po rečima sasvim razumna i da signal gubi baš `any()`.
LOW_CONFIDENCE_SENTENCE_SHARE = 0.20

#: Ispod ovoliko reči udeo nije mera: u rečenici od tri reči jedna loša je već
#: trećina, pa bi svaki prag ispod 0.33 bio isto što i „bar jedna". Za kratke
#: rečenice zato ostaje staro pravilo — ono tu i nije pravilo greške nego opis
#: stanja: kad je rečenica od dve reči, jedna sumnjiva reč jeste pola rečenice.
MIN_WORDS_FOR_SHARE_RULE = 5

#: Praznina duža od ovoliko sekundi se prijavljuje. Kraće su normalne pauze.
MIN_REPORTED_GAP = 3.0

#: Ispod ovog procenta pokrivenosti verovatno je VAD pojeo govor.
LOW_COVERAGE_WARN = 85.0


#: Koliko VAD sme da odbaci pre nego što se predloži poređenje bez njega.
#:
#: Prag NIJE na pokrivenost, iako je to bio prvi predlog. Pokrivenost ne meri
#: agresivnost VAD-a nego koliko snimak uopšte sadrži govora — i pri potpuno
#: isključenom VAD-u ide samo do 88.9%. Izmereno na svih sedam poslova sa
#: uključenim VAD-om, oba pragova na pokrivenost okidaju SVAKI put:
#:
#:     posao                    pokrivenost  odbačeno  pokr<85  pokr<90  odb>10%
#:     fr, bubica, dobar zvuk        88.2%      4.8%      -      OKINE      -
#:     sr, tih, prag 0.35            85.6%     12.9%      -      OKINE    OKINE
#:     sr, tih, prag 0.20            87.9%      7.8%      -      OKINE      -
#:     sr, tih, stariji prolazi   79.2-79.6%  12.9-15.0% OKINE   OKINE    OKINE
#:
#: „Ijedna praznina preko 3 s" okida 7/7 — svaki posao ima bar jednu. Savet koji
#: se uvek pojavi je isto što i savet koji se nikad ne pojavi; to je ista greška
#: koju smo upravo ispravili kod `low_confidence` (`any()` kao pojačivač).
#:
#: Udeo odbačenog zvuka razdvaja ono što treba: dobar snimak 4.8% naspram tihog
#: 12.9%, dok su im pokrivenosti skoro jednake (88.2% i 85.6%).
#:
#: 10% je izabrano da ostane razmak i naviše i naniže od jedinog poznatog dobrog
#: snimka (4.8%). Posledica koju treba znati: prag 0.20 na tihom snimku odbacuje
#: 7.8% i NE dobija savet, iako je i tu isključen VAD dao nešto bolji prepis
#: (189 naspram 180 reči). Ako se pokaže da to smeta, 7.0 hvata i taj slučaj —
#: ali sa svega jednim merenjem dobrog snimka, prag između dve susedne tačke bio
#: bi krojen po uzorku.
MAX_VAD_DISCARD_SHARE = 0.10

#: Koliko snimka sme biti ne-govor pre nego što se, pri ISKLJUČENOM VAD-u,
#: skrene pažnja na tišinu. Isključen VAD znači da model vidi i tišinu, a to je
#: prilika za halucinaciju; ovo je jedina odbrana koja ne košta ništa — brojka
#: koja kaže koliko je te tišine bilo. Raniji mereni snimci su gusti govor
#: (odbačeno 4.8-15.0%). Prvi izmeren slučaj preko praga: Radionica draft 1, 2026-10-01,
#: 30,3% negovora (džingl) — sa isključenim VAD-om model je preko džingla
#: izmislio tekst iz glosara (`merenja/dnevnik.md`). Od tada se upozorenje
#: prikazuje i na stranici posla, ne samo u dnevniku.
HIGH_SILENCE_SHARE = 0.25


def vad_advice(
    discarded_share: float,
    coverage_pct: float,
    *,
    vad_on: bool,
    max_discard: float = MAX_VAD_DISCARD_SHARE,
    high_silence: float = HIGH_SILENCE_SHARE,
) -> str | None:
    """Šta reći o detekciji govora posle transkripcije. `None` ako nema šta.

    Dva smera, jer se od 2026-08-19 podrazumeva ISKLJUČENA detekcija:

    UKLJUČENA i odbacuje mnogo -> predloži poređenje bez nje. Izmereno na tihom
    snimku bez bubice: isključen VAD je dao BOLJI prepis („Delikates burger od
    200 grama" umesto „deli kates burdžer 200 grama", „bez smokija" umesto „gre
    smokin", dve odvojene rečenice umesto spojenog fragmenta). Filter tišine na
    tihom snimku seče delove govora koje model inače rekonstruiše iz konteksta.

    ISKLJUČENA a snimak ima mnogo tišine -> reci koliko. To je jedini slučaj u
    kom isključena detekcija stvarno rizikuje, a nije izmeren ni na jednom
    snimku — svi dosadašnji su gust govor.
    """
    if vad_on:
        if discarded_share <= max_discard:
            return None
        return (
            f"detekcija govora je odbacila {100 * discarded_share:.1f}% snimka "
            f"(prag {100 * max_discard:.0f}%), pokrivenost {coverage_pct:.1f}% — "
            "probaj sa isključenom detekcijom i uporedi "
            "(dashboard: „isključeno“; CLI: --vad je uključuje)"
        )
    if discarded_share >= high_silence:
        return (
            f"detekcija govora je ISKLJUČENA, a {100 * discarded_share:.1f}% "
            "snimka nije govor (muzika, džingl, špica ili tišina) — model vidi i "
            "te delove. Ako se u prepisu pojavi tekst preko muzike ili tišine, "
            "uključi detekciju (CLI: --vad) i uporedi."
        )
    return None

# --- Zaštitna mreža za degenerisanu segmentaciju ---------------------------
#
# Ovo NIJE rešenje nego mreža. Pravo rešenje kad ASR ne vrati interpunkciju je
# `punctuate.py`; bez znakova su i prelomi loši, ne samo segmentacija, jer je
# interpunkcija glavni signal za tačku preloma u `linebreak.py`.

#: Rečenica sa manje reči od ovoga spaja se sa sledećom.
MIN_SENTENCE_WORDS = 3

#: Rečenica kraća od ovoliko sekundi spaja se sa sledećom.
MIN_SENTENCE_DURATION = 1.0

#: Najkraće trajanje po reči koje se uopšte smatra mogućim (sekunde).
#:
#: Ovo je jedini uslov koji „jasna interpunkcija" NE može da poništi, i zato
#: postoji odvojeno od `MIN_SENTENCE_DURATION`. Izmereno 2026-08-19, posao sa
#: lošim zvukom: rečenica „8 cm." traje 31.41 -> 31.47, dakle 0.06 s za dve
#: reči i tačku (0.03 s po reči). To je ostatak ponavljanja — poslednje dve
#: reči prethodne rečenice, ponovljene sa p=0.08 i nultim trajanjem.
#:
#: Mreža ga je propustila jer se komad završava tačkom, pa je bio izuzet od
#: obe provere (i od `MIN_SENTENCE_WORDS` i od `MIN_SENTENCE_DURATION`).
#: Gustina znakova u tom poslu je bila sasvim normalna (0.05 znakova po reči,
#: prag 0.5), pa `trust_punctuation` nije ni pao — interpunkcija je bila dobra,
#: rečenica nije. Tačka na kraju artefakta ne čini artefakt rečenicom.
#:
#: Vrednost je izabrana po izmerenom, ne po osećaju. Najbrže PRAVE rečenice u
#: dva merena posla (2026-08-19):
#:
#:     0.239 s/reč  francuska rečenica, brz govor                    (16 reči)
#:     0.264 s/reč  srpski uzvik pri pozdravu                       (6 reči)
#:     0.303 s/reč  "Merci beaucoup." / "Hvala puno."                  (2 reči)
#:     ---------------------------------------------------------------
#:     0.030 s/reč  "8 cm."                                    <- artefakt
#:
#: Prag 0.3 bi stao TAČNO na „Hvala puno." (0.303) — rečenica koja je u redu
#: prošla bi za 3 hiljadita dela sekunde, a sledeća takva ne bi. Zato 0.15:
#: pet puta iznad artefakta, oko 1.6 puta ispod najbržeg pravog govora. Rupa
#: između te dve pojave je ogromna (0.03 naspram 0.24) i prag treba da stoji u
#: njenoj sredini, a ne uz njenu ivicu.
MIN_DURATION_PER_WORD = 0.15

#: Ispod ovoliko reči po rečenici u proseku, segmentacija nije uspela.
MIN_AVG_WORDS_PER_SENTENCE = 3.0

#: Iznad ovoliko znakova kraja po reči interpunkcija je šum, ne struktura.
#: Znak posle svake reči je isto toliko beskoristan kao nijedan znak.
MAX_SENTENCE_END_MARKS_PER_WORD = 0.5

#: Ispod ovoliko reči se o strukturi ne sudi. „Dobar dan. Hvala." ima prosek
#: 1.5 reči po rečenici i savršeno je u redu — kratak uzorak nije dokaz kvara.
MIN_WORDS_FOR_DIAGNOSIS = 25

#: Prag gustine interpunkcije: znakova kraja na 100 reči. Ispod praga se pušta
#: LLM restauracija.
#:
#: Ovo je treći kvar, koji ni `MIN_AVG_WORDS_PER_SENTENCE` ni `no_punctuation`
#: ne hvataju: Whisper punktira prvih nekoliko rečenica pa prestane. Znakova
#: ima, prosek rečenice je pristojan, a tri četvrtine transkripta je bez ijednog
#: znaka. Francuski test 2026-08-12: 10 znakova na 428 reči, rečenica od 214
#: znakova bez zareza — sve tri stare provere su rekle „u redu".
#:
#: Prag je po jeziku, ne globalan: francuski i engleski imaju kraće rečenice od
#: srpskog, pa im je prirodna gustina viša. Zasad su poznata samo dva merenja
#: (fr 2.3 = kvar, sr 5.5 = u redu) — premalo da se iz njih izvede jedan broj
#: koji važi svuda. Zato rečnik: kad se iz dijagnostike skupi više merenja,
#: podešava se ključ po jeziku, a ne jedna konstanta za sve.
PUNCTUATION_DENSITY_THRESHOLD: dict[str, float] = {
    "default": 4.0,
}

#: Prag gustine ZAREZA: zareza na 100 reči. Rečnik po jeziku, kao i gornji.
#:
#: Dodato 2026-08-19, posle trećeg merenja. Do tada je odluka o restauraciji
#: išla samo po znacima kraja, iako se gustina zareza merila i ispisivala —
#: rupa koja je ostavljena svesno („kriterijum se dodaje ovde kad se merenja
#: skupe") i koju je treći posao naplatio:
#:
#:     interpunkcija: 3.4 znaka kraja/100 reči, 17.6 zareza/100 reči -> restauracija
#:
#: Znakova kraja jeste bilo malo, ali zareza je bilo 17.6 na sto reči — više
#: nego dovoljno za prelamanje. Restauracija je pokrenuta bez potrebe i koštala
#: je 61 s, od čega 33 s (55%) na odbijene odgovore, uz dva sumnjiva ustupka
#: (`curic -> curi` i `curic -> curici` u istom prolazu).
#:
#: Zarezi su ono što `linebreak.py` stvarno koristi: oni su glavni signal za
#: tačku preloma UNUTAR rečenice. Tekst sa malo tačaka a puno zareza lomi se
#: dobro; tekst sa puno tačaka a bez zareza ne.
#:
#: Vrednost 8.0 je predlog iz nalaza, ne izvedena iz uzorka — poznata merenja
#: su fr 8.4 (dobar zvuk) i sr 17.6 (loš zvuk, lažno okidanje). Oba su iznad
#: praga, dakle prag zasad razdvaja samo ono što treba da razdvaja i nema
#: nijedno merenje ISPOD sebe. Prvi posao sa malo i tačaka i zareza je onaj
#: koji će ga stvarno proveriti.
COMMA_DENSITY_THRESHOLD: dict[str, float] = {
    "default": 8.0,
}


def density_threshold(language: str | None) -> float:
    key = (language or "").strip().lower()
    return PUNCTUATION_DENSITY_THRESHOLD.get(
        key, PUNCTUATION_DENSITY_THRESHOLD["default"]
    )


def comma_threshold(language: str | None) -> float:
    key = (language or "").strip().lower()
    return COMMA_DENSITY_THRESHOLD.get(key, COMMA_DENSITY_THRESHOLD["default"])


#: Titule: iza njih tačka nikad ne završava rečenicu (sledi ime, veliko slovo).
TITLE_ABBREVIATIONS = frozenset({
    "dr", "mr", "g", "gdin", "gđa", "gđica", "prof", "akad", "ing", "arh",
    "sv", "gen", "pukovnik", "inž",
})

#: Opšte skraćenice: tačka završava rečenicu samo ako sledeća reč ide velikim slovom.
GENERAL_ABBREVIATIONS = frozenset({
    "itd", "npr", "tj", "tzv", "sl", "br", "ul", "god", "str", "odn", "uklj",
    "tel", "cca", "resp", "op", "cit", "sr", "hrv", "eng",
})

# --- Razmaci pri spajanju tokena -------------------------------------------

NO_SPACE_BEFORE = set(",.!?…:;%)]}»”’")
NO_SPACE_AFTER = set("([{«„‘")


def join_words(words: list[Word]) -> str:
    """Spoji tokene u tekst, bez razmaka ispred interpunkcije."""
    out: list[str] = []
    for idx, w in enumerate(words):
        tok = w.w
        if not tok:
            continue
        if out:
            prev = out[-1]
            glue = tok[0] in NO_SPACE_BEFORE or (prev and prev[-1] in NO_SPACE_AFTER)
            if not glue:
                out.append(" ")
        out.append(tok)
    return "".join(out).strip()


# --- Tokenizacija ----------------------------------------------------------

#: Interpunkcija koja se odvaja sa kraja tokena u zaseban token.
PUNCT_TAIL = ",.!?…:;\"»”’)]}%"
#: Interpunkcija koja se odvaja sa početka tokena.
PUNCT_HEAD = "([{«„‘\"-–—"


def split_token(raw: str) -> list[str]:
    """`"Dobar"` -> `["Dobar"]`, `"došli."` -> `["došli", "."]`, `"- Da"` -> `["-", "Da"]`.

    Isti postupak koristi `asr` pri upisu i `linebreak` pri poravnanju ručno
    izmenjenog teksta — moraju da se slažu, inače poravnanje puca.
    """
    out: list[str] = []
    raw = raw.strip()
    if not raw:
        return out
    i = 0
    while i < len(raw) - 1 and raw[i] in PUNCT_HEAD:
        out.append(raw[i])
        i += 1
    core = raw[i:]
    j = len(core)
    while j > 0 and core[j - 1] in PUNCT_TAIL:
        j -= 1
    if j > 0:
        out.append(core[:j])
    if j < len(core):
        out.append(core[j:])
    return out


def tokenize_text(text: str) -> list[str]:
    """Tekst -> tokeni, isto kako ih `asr` upisuje u words.json."""
    out: list[str] = []
    for raw in text.split():
        out.extend(split_token(raw))
    return out


# --- Detekcija granice rečenice --------------------------------------------


def _is_end_punct_token(tok: str) -> bool:
    t = tok.strip()
    return bool(t) and all(ch in SENTENCE_END_CHARS for ch in t)


def _strip_punct(tok: str) -> str:
    return tok.strip().strip(".,!?…:;\"'»«„”’()[]{}-–—").lower()


def _starts_upper(tok: str) -> bool:
    for ch in tok:
        if ch.isalpha():
            return ch.isupper()
        if ch.isdigit():
            return True
    return False


def _is_boundary(words: list[Word], i: int) -> bool:
    """Da li token na poziciji `i` (interpunkcija) zatvara rečenicu."""
    tok = words[i].w.strip()
    if not _is_end_punct_token(tok):
        return False

    # ! ? … nikad nisu deo skraćenice
    if set(tok) != {"."}:
        return True

    prev = _strip_punct(words[i - 1].w) if i > 0 else ""
    nxt = words[i + 1].w.strip() if i + 1 < len(words) else ""

    if prev in TITLE_ABBREVIATIONS:
        return False
    if prev in GENERAL_ABBREVIATIONS or (prev.isdigit() and prev):
        # "23. maja" nije granica; "...bilo ih je 23. Onda..." jeste
        return not nxt or _starts_upper(nxt)
    return True


# --- Nasilno lomljenje predugačkih rečenica --------------------------------


def _split_long(chunk: list[Word], max_duration: float) -> list[list[Word]]:
    """Rekurzivno razbij predugačku rečenicu na zarezu ili na dugoj pauzi."""
    if len(chunk) < 2 or (chunk[-1].e - chunk[0].s) <= max_duration:
        return [chunk]

    mid = (chunk[0].s + chunk[-1].e) / 2.0

    comma_cuts: list[tuple[float, int]] = []
    pause_cuts: list[tuple[float, int]] = []
    for k in range(len(chunk) - 1):
        if k == 0:
            continue  # ne seci tako da prvi deo ima jednu reč-interpunkciju
        if chunk[k].w.strip().startswith(","):
            comma_cuts.append((abs(chunk[k].e - mid), k))
        gap = chunk[k + 1].s - chunk[k].e
        if gap > HARD_SPLIT_MIN_PAUSE:
            pause_cuts.append((abs(chunk[k].e - mid), k))

    candidates = comma_cuts or pause_cuts
    if not candidates:
        # Nema nijedne prirodne granice — seci na najvećoj pauzi koja postoji,
        # a pri izjednačenju NAJBLIŽE SREDINI.
        #
        # Ovde je bio pravi kvar iza „svaka reč je zasebna rečenica": sa
        # `max([(pauza, k), …])` Python pri jednakim pauzama poredi drugi član
        # torke i bira NAJVEĆI indeks. U govoru bez interpunkcije su sve pauze
        # jednake, pa se rez uvek pravio na poslednjoj reči — rekurzija je
        # ljuštila po jednu reč sa kraja, do kraja snimka.
        interior = range(1, len(chunk) - 1)
        if not interior:
            return [chunk]
        cut = min(
            interior,
            key=lambda k: (
                -(chunk[k + 1].s - chunk[k].e),  # najveća pauza
                abs(chunk[k].e - mid),           # pa najbliža sredini
            ),
        )
    else:
        _, cut = min(candidates)

    left, right = chunk[: cut + 1], chunk[cut + 1 :]
    if not left or not right:
        return [chunk]
    return _split_long(left, max_duration) + _split_long(right, max_duration)


# --- Javni API -------------------------------------------------------------


def _word_count(chunk: list[Word]) -> int:
    return sum(1 for w in chunk if any(c.isalnum() for c in w.w))


def _ends_with_clear_punctuation(chunk: list[Word]) -> bool:
    for w in reversed(chunk):
        t = w.w.strip()
        if not t:
            continue
        return bool(t) and t[-1] in SENTENCE_END_CHARS
    return False


def _duration_per_word(chunk: list[Word]) -> float | None:
    """Sekundi po reči. None ako u komadu nema nijedne prave reči."""
    n = _word_count(chunk)
    if not n:
        return None
    return (chunk[-1].e - chunk[0].s) / n


def _impossibly_fast(chunk: list[Word], floor: float) -> bool:
    """Traje kraće nego što je fizički moguće izgovoriti.

    Ovo nije „kratko" nego „nemoguće", i zato je jedini uslov koji interpunkcija
    ne može da poništi. Vidi `MIN_DURATION_PER_WORD`.
    """
    per_word = _duration_per_word(chunk)
    return per_word is not None and per_word < floor


def low_confidence_share(
    chunk: list[Word], *, threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> float:
    """Udeo sumnjivih reči u komadu, od 0 do 1.

    Broji SAMO prave reči. Interpunkcija nasledi verovatnoću susedne reči i
    ulazila bi u imenilac kao da je izgovorena — rečenica sa mnogo zareza bi
    tako ispala sigurnija nego što jeste.
    """
    real = [w for w in chunk if any(c.isalnum() for c in w.w)]
    if not real:
        return 0.0
    return sum(1 for w in real if w.p < threshold) / len(real)


def is_low_confidence(
    chunk: list[Word],
    *,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
    share: float = LOW_CONFIDENCE_SENTENCE_SHARE,
    min_words: int = MIN_WORDS_FOR_SHARE_RULE,
) -> bool:
    """Da li cela rečenica nosi oznaku `low_confidence`.

    Duža rečenica se sudi po UDELU sumnjivih reči; kratka po pravilu „bar
    jedna", jer u njoj jedna reč ionako prelazi svaki razuman udeo. Vidi
    `LOW_CONFIDENCE_SENTENCE_SHARE` za razlog promene.
    """
    real = [w for w in chunk if any(c.isalnum() for c in w.w)]
    if not real:
        return False
    if len(real) < min_words:
        return any(w.p < threshold for w in real)
    return low_confidence_share(chunk, threshold=threshold) >= share


def merge_short(
    chunks: list[list[Word]],
    *,
    min_words: int = MIN_SENTENCE_WORDS,
    min_duration: float = MIN_SENTENCE_DURATION,
    max_duration: float = MAX_SENTENCE_DURATION,
    min_duration_per_word: float = MIN_DURATION_PER_WORD,
    trust_punctuation: bool = True,
) -> list[list[Word]]:
    """Spoji prekratke rečenice sa sledećom.

    `trust_punctuation=False` ukida izuzeće za rečenice koje se završavaju
    tačkom. Kad ASR stavi tačku posle svake reči, to „jasna interpunkcija"
    prestaje da bude — inače bi mreža propustila baš slučaj zbog kog postoji.

    Nezavisno od toga, izuzeće NIKAD ne važi za komad koji traje kraće nego
    što je fizički moguće izgovoriti. Tačka na kraju artefakta ne čini artefakt
    rečenicom, a gustina znakova u celom fajlu ume da bude uredna dok je jedan
    jedini komad smeće — pa `trust_punctuation` u tom slučaju i ne padne.
    """
    if not chunks:
        return chunks

    # Spajanje unapred: kratak komad se lepi na sledeći, dok ne poraste.
    merged: list[list[Word]] = []
    pending: list[Word] = []
    for chunk in chunks:
        candidate = pending + chunk
        pending = []
        too_short = (
            _word_count(candidate) < min_words
            or (candidate[-1].e - candidate[0].s) < min_duration
        )
        exempt = (
            trust_punctuation
            and _ends_with_clear_punctuation(candidate)
            and not _impossibly_fast(candidate, min_duration_per_word)
        )
        if too_short and not exempt:
            pending = candidate
            continue
        merged.append(candidate)

    if pending:
        # Poslednji komad nema sledećeg — lepi se na prethodni, ako smeju zajedno.
        if merged and (pending[-1].e - merged[-1][0].s) <= max_duration:
            merged[-1] = merged[-1] + pending
        else:
            merged.append(pending)
    return merged


def segment(
    words: list[Word],
    *,
    max_duration: float = MAX_SENTENCE_DURATION,
    low_confidence: float = LOW_CONFIDENCE_THRESHOLD,
    low_confidence_share: float = LOW_CONFIDENCE_SENTENCE_SHARE,
    min_words: int = MIN_SENTENCE_WORDS,
    min_sentence_duration: float = MIN_SENTENCE_DURATION,
    min_duration_per_word: float = MIN_DURATION_PER_WORD,
) -> list[Sentence]:
    """Pretvori niz reči u rečenice."""
    if not words:
        return []

    raw: list[list[Word]] = []
    current: list[Word] = []
    for i, w in enumerate(words):
        current.append(w)
        if _is_boundary(words, i):
            raw.append(current)
            current = []
    if current:
        raw.append(current)

    # Da li interpunkciji uopšte treba verovati? Kriterijum je gustina znakova
    # kraja po reči, ne prosečna dužina rečenice — kratak tekst sa kratkim
    # rečenicama („Dobar dan. Hvala.") ima nizak prosek a savršenu interpunkciju.
    total_words = _word_count(words)
    marks = sum(1 for w in words if _is_end_punct_token(w.w))
    trust = not (
        total_words >= MIN_WORDS_FOR_DIAGNOSIS
        and marks / total_words > MAX_SENTENCE_END_MARKS_PER_WORD
    )

    raw = merge_short(
        raw,
        min_words=min_words,
        min_duration=min_sentence_duration,
        max_duration=max_duration,
        min_duration_per_word=min_duration_per_word,
        trust_punctuation=trust,
    )

    chunks: list[list[Word]] = []
    for chunk in raw:
        chunks.extend(_split_long(chunk, max_duration))

    # Mreža i POSLE lomljenja: degenerisani komadi nastaju upravo tu.
    chunks = merge_short(
        chunks,
        min_words=min_words,
        min_duration=min_sentence_duration,
        max_duration=max_duration,
        min_duration_per_word=min_duration_per_word,
        trust_punctuation=trust,
    )

    sentences: list[Sentence] = []
    for n, chunk in enumerate(chunks, start=1):
        text = join_words(chunk)
        if not text:
            continue
        flags: list[str] = []
        if is_low_confidence(
            chunk, threshold=low_confidence, share=low_confidence_share
        ):
            flags.append("low_confidence")
        sentences.append(
            Sentence(
                id=n,
                s=round(chunk[0].s, 3),
                e=round(chunk[-1].e, 3),
                text=text,
                words=(chunk[0].i, chunk[-1].i),
                speaker=None,
                flags=flags,
            )
        )
    # renumeriši ako je neki prazan chunk odbačen
    for n, sent in enumerate(sentences, start=1):
        sent.id = n
    return sentences


# --- Dijagnostika strukture -------------------------------------------------


@dataclass
class Structure:
    """Ima li ASR izlaz ikakvu rečeničnu strukturu.

    Interpunkcija je važnija nego što izgleda: pored segmentacije, ona je glavni
    signal za tačku preloma u `linebreak.py`. Bez nje su i prelomi loši.
    """

    words: int
    sentences: int
    sentence_end_marks: int
    commas: int
    punctuation_tokens: int
    avg_words_per_sentence: float
    capitalized_share: float

    @property
    def marks_per_100_words(self) -> float:
        return 100.0 * self.sentence_end_marks / self.words if self.words else 0.0

    @property
    def commas_per_100_words(self) -> float:
        return 100.0 * self.commas / self.words if self.words else 0.0

    @property
    def no_punctuation(self) -> bool:
        return self.sentence_end_marks == 0

    def sparse_punctuation(self, language: str | None = None) -> bool:
        """Znakova ima, ali premalo — Whisper je punktirao pa odustao.

        Traži se da OBE gustine budu ispod svog praga. Ranije se sudilo samo po
        znacima kraja, pa je tekst sa 3.4 tačke i 17.6 zareza na sto reči išao
        na restauraciju iako se odlično lomio. Zarezi su ono što `linebreak.py`
        stvarno koristi — vidi `COMMA_DENSITY_THRESHOLD`.
        """
        return (
            self.words >= MIN_WORDS_FOR_DIAGNOSIS
            and not self.no_punctuation
            and self.marks_per_100_words < density_threshold(language)
            and self.commas_per_100_words < comma_threshold(language)
        )

    @property
    def failed(self) -> bool:
        """Segmentacija nije uspela — prosek rečenice je ispod praga.

        Sudi se samo na dovoljno velikom uzorku; dve kratke rečenice nisu kvar.
        """
        return (
            self.words >= MIN_WORDS_FOR_DIAGNOSIS
            and self.avg_words_per_sentence < MIN_AVG_WORDS_PER_SENTENCE
        )

    def needs_punctuation(self, language: str | None = None) -> bool:
        """Vredi pustiti LLM restauraciju interpunkcije."""
        return self.words > 0 and (
            self.no_punctuation or self.failed or self.sparse_punctuation(language)
        )

    def gate_reason(self, language: str | None = None) -> str:
        """Koji uslov je doneo odluku. Bez ovoga se iz brojeva ne vidi ko je sudio.

        `no_punctuation` i `failed` ostaju bezuslovni i zarez ih ne spasava:
        bez ijednog znaka kraja nema granice rečenice ni da se traži, a zarez
        je ne pravi. Zarez ulazi samo u ocenu GUSTINE, gde je i grešilo.
        """
        if self.words == 0:
            return "nema reči"
        if self.no_punctuation:
            return "nijedan znak kraja"
        if self.failed:
            return f"prosek {self.avg_words_per_sentence:.1f} reči/rečenici"
        if self.sparse_punctuation(language):
            return "obe gustine ispod praga"
        if self.marks_per_100_words < density_threshold(language):
            return (
                f"znakova kraja jeste malo, ali zarezi drže "
                f"({self.commas_per_100_words:.1f} >= {comma_threshold(language):.1f})"
            )
        return "znakova kraja dovoljno"

    def density_line(self, language: str | None = None) -> str:
        """Jedan red merenja, ispisuje se posle SVAKE transkripcije.

        Postoji da bi se pragovi podešavali po izmerenom, a ne po pretpostavci.
        """
        verdict = (
            "restauracija" if self.needs_punctuation(language) else "u redu"
        )
        return (
            f"interpunkcija: {self.marks_per_100_words:.1f} znaka kraja/100 reči "
            f"(prag {density_threshold(language):.1f}), "
            f"{self.commas_per_100_words:.1f} zareza/100 reči "
            f"(prag {comma_threshold(language):.1f}) "
            f"-> {verdict} [{self.gate_reason(language)}]"
        )

    def summary(self) -> str:
        return (
            f"reči {self.words}, rečenica {self.sentences}, "
            f"prosek {self.avg_words_per_sentence:.1f} reči/rečenici, "
            f"znakova kraja {self.sentence_end_marks} "
            f"({self.marks_per_100_words:.1f}/100 reči), zareza {self.commas}, "
            f"velikih početnih slova {100 * self.capitalized_share:.0f}%"
        )

    def warning(self, language: str | None = None) -> str | None:
        if self.words == 0:
            return "ASR nije vratio nijednu reč."
        if self.sparse_punctuation(language):
            return (
                f"INTERPUNKCIJE JE PREMALO: {self.sentence_end_marks} znakova kraja "
                f"na {self.words} reči ({self.marks_per_100_words:.1f}/100, prag "
                f"{density_threshold(language):.1f}), zareza {self.commas} "
                f"({self.commas_per_100_words:.1f}/100). Prelomi će biti loši. "
                "Pusti restauraciju interpunkcije: --punctuate."
            )
        if self.failed:
            return (
                f"SEGMENTACIJA NIJE USPELA: prosek {self.avg_words_per_sentence:.1f} "
                f"reči po rečenici (prag {MIN_AVG_WORDS_PER_SENTENCE:.0f}). "
                + (
                    "ASR nije vratio nijedan znak interpunkcije."
                    if self.no_punctuation
                    else f"Znakova kraja ima {self.sentence_end_marks} na "
                         f"{self.words} reči — verovatno posuti po svakoj reči."
                )
                + " Pusti restauraciju interpunkcije: --punctuate."
            )
        if self.no_punctuation:
            return (
                "ASR nije vratio nijedan znak interpunkcije. Prelomi će biti loši "
                "jer je interpunkcija glavni signal za tačku preloma. --punctuate."
            )
        return None


def analyze(words: list[Word], sentences: list[Sentence] | None = None) -> Structure:
    """Prebroj interpunkciju i izmeri prosečnu dužinu rečenice."""
    word_tokens = [w for w in words if any(c.isalnum() for c in w.w)]
    punct = [w for w in words if w.w.strip() and not any(c.isalnum() for c in w.w)]
    ends = [w for w in punct if _is_end_punct_token(w.w)]
    commas = [w for w in punct if w.w.strip().startswith(",")]

    alpha = [w for w in word_tokens if any(c.isalpha() for c in w.w)]
    caps = sum(1 for w in alpha if w.w[:1].isupper())

    n_sent = len(sentences) if sentences is not None else 0
    if sentences is None:
        n_sent = len(segment(words)) if words else 0

    return Structure(
        words=len(word_tokens),
        sentences=n_sent,
        sentence_end_marks=len(ends),
        commas=len(commas),
        punctuation_tokens=len(punct),
        avg_words_per_sentence=len(word_tokens) / n_sent if n_sent else 0.0,
        capitalized_share=caps / len(alpha) if alpha else 0.0,
    )


# --- Tempo: najkraća rečenica i najkraće trajanje po reči -------------------


def _text_word_count(text: str) -> int:
    return sum(1 for t in text.split() if any(c.isalnum() for c in t))


@dataclass
class Pace:
    """Najkraća rečenica i najbrža rečenica u poslu.

    Postoji zato što je jedan artefakt od 0.06 s prošao kroz sve provere i
    video se tek kad ga je čovek slučajno pogledao u SRT-u. Zbir i prosek ga
    ne pokazuju — pokazuje ga samo minimum, pa se minimum ispisuje uvek.
    """

    sentences: int = 0
    shortest_id: int | None = None
    shortest_duration: float = 0.0
    shortest_text: str = ""
    fastest_id: int | None = None
    fastest_per_word: float = 0.0
    fastest_text: str = ""
    #: Rečenice ispod fizičke granice. Posle segmentacije ovo mora biti prazno;
    #: ako nije, mreža nije uspela da ih spoji i to je kvar koji traži pogled.
    impossible: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.impossible is None:
            self.impossible = []

    def line(self, floor: float = MIN_DURATION_PER_WORD) -> str:
        if self.shortest_id is None:
            return "tempo: nema nijedne rečenice"
        return (
            f"tempo: najkraća {self.shortest_duration:.2f}s "
            f"[{self.shortest_id}] {self.shortest_text!r}, "
            f"najbrža {self.fastest_per_word:.2f} s/reč "
            f"[{self.fastest_id}] {self.fastest_text!r} (granica {floor:.2f})"
        )

    def warning(self, floor: float = MIN_DURATION_PER_WORD) -> str | None:
        if not self.impossible:
            return None
        return (
            f"NEMOGUĆE TRAJANJE: rečenice {self.impossible} traju kraće od "
            f"{floor:.2f} s po reči. To nije govor nego ostatak ASR ponavljanja; "
            "zaštitna mreža ih nije spojila — pogledaj words.json na tim vremenima."
        )


def analyze_pace(
    sentences: list[Sentence], *, floor: float = MIN_DURATION_PER_WORD
) -> Pace:
    """Izmeri najkraću i najbržu rečenicu. Rečenice bez reči se preskaču."""
    pace = Pace(sentences=len(sentences))
    best_dur: tuple[float, Sentence] | None = None
    best_rate: tuple[float, Sentence] | None = None
    for sent in sentences:
        n = _text_word_count(sent.text)
        if not n:
            continue
        dur = sent.e - sent.s
        rate = dur / n
        if best_dur is None or dur < best_dur[0]:
            best_dur = (dur, sent)
        if best_rate is None or rate < best_rate[0]:
            best_rate = (rate, sent)
        if rate < floor:
            pace.impossible.append(sent.id)
    if best_dur:
        pace.shortest_duration, sent = best_dur
        pace.shortest_id, pace.shortest_text = sent.id, sent.text
    if best_rate:
        pace.fastest_per_word, sent = best_rate
        pace.fastest_id, pace.fastest_text = sent.id, sent.text
    return pace


# --- Raspodela ASR verovatnoća ---------------------------------------------

#: Granice opsega za raspodelu. Prag bojenja (`LOW_CONFIDENCE_THRESHOLD`) je
#: jedna od njih namerno — da se vidi tačno koliko teksta on obuhvata.
CONFIDENCE_EDGES = (0.4, 0.6, 0.8)

#: Percentili koji se prijavljuju. p15 je tu zbog otvorenog pitanja: ako bi
#: prag bio RELATIVAN (donjih 15% reči tog posla), ovo je verovatnoća na kojoj
#: bi stajao — pa se čita uporedo sa apsolutnih 0.6 i vidi se koliko se
#: razlikuju na dobrom, a koliko na lošem zapisu.
REPORTED_PERCENTILES = (5, 15, 25, 50)


@dataclass
class Confidence:
    """Raspodela ASR verovatnoća po rečima.

    Broji SAMO prave reči; interpunkcija nasledi verovatnoću susedne reči i
    samo bi naduvala gornji opseg.

    Ovo je merenje pred odluku, ne odluka: da li prag bojenja treba da ostane
    apsolutan (0.6) ili da postane relativan (donjih N% reči tog posla). Da bi
    se to znalo, treba videti raspodelu na dobrom i na lošem zapisu.
    """

    words: int = 0
    counts: list[int] = None  # type: ignore[assignment]
    percentiles: dict[int, float] = None  # type: ignore[assignment]
    threshold: float = LOW_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        if self.counts is None:
            self.counts = [0] * (len(CONFIDENCE_EDGES) + 1)
        if self.percentiles is None:
            self.percentiles = {}

    @property
    def ranges(self) -> list[tuple[float, float]]:
        edges = (0.0,) + CONFIDENCE_EDGES + (1.0,)
        return [(edges[k], edges[k + 1]) for k in range(len(edges) - 1)]

    @property
    def below_threshold(self) -> int:
        """Reči koje prag bojenja trenutno obeležava."""
        return sum(
            n for (lo, hi), n in zip(self.ranges, self.counts)
            if hi <= self.threshold
        )

    @property
    def flagged_share(self) -> float:
        return self.below_threshold / self.words if self.words else 0.0

    def lines(self) -> list[str]:
        """Raspodela, spremna za ispis. Prazan spisak ako nema reči."""
        if not self.words:
            return []
        out = ["verovatnoće po rečima:"]
        for (lo, hi), n in zip(self.ranges, self.counts):
            pct = 100.0 * n / self.words
            bar = "#" * int(round(pct / 2))
            label = f"{lo:.1f}-{hi:.1f}" if hi < 1.0 else f"{lo:.1f}-1.0"
            out.append(f"  {label:>9s} {n:5d}  {pct:5.1f}%  {bar}")
        pcts = "  ".join(
            f"p{k}={self.percentiles.get(k, 0.0):.2f}" for k in REPORTED_PERCENTILES
        )
        out.append(f"  percentili: {pcts}")
        out.append(
            f"  prag {self.threshold:.2f} obeležava "
            f"{self.below_threshold}/{self.words} reči "
            f"({100 * self.flagged_share:.1f}%); "
            f"relativan prag „donjih 15%\" stajao bi na "
            f"{self.percentiles.get(15, 0.0):.2f}"
        )
        return out


def function_words(language: str | None) -> frozenset[str]:
    """Funkcijske reči jednog jezika, iz tabela u `linebreak.py`.

    Uvoz je odložen namerno: `linebreak` uvozi `segment`, pa bi uvoz na vrhu
    napravio ciklus. Tabele se NE dupliraju — ako se prelamanje predomisli o
    tome šta je funkcijska reč, ovo merenje se predomisli sa njim.
    """
    from .linebreak import rules_for

    rules = rules_for(language, lambda _msg: None)
    return frozenset(rules.no_line_end | rules.no_line_start | rules.conjunctions)


def would_flag(
    chunks: list[list[Word]],
    language: str | None,
    *,
    threshold: float = LOW_CONFIDENCE_THRESHOLD,
    share: float = LOW_CONFIDENCE_SENTENCE_SHARE,
) -> dict[str, int]:
    """Koliko bi rečenica bilo označeno sa i bez izuzeća funkcijskih reči.

    Za MERENJE, ne za odluku — isto kao `punctuate.would_pass`. Postoji zbog
    sumnje da udeo dižu kratke funkcijske reči (`ali`, `i`, `s`, `od`), na
    kojima je model nesiguran i kad ih tačno prepozna, a koje niko ne ispravlja.
    Dok se ne vidi razlika u brojevima, pravilo se ne dira.

    Vraća broj označenih rečenica po varijanti; `bez_funkcijskih` iz IMENIOCA i
    BROJIOCA izbacuje funkcijske reči — one prestaju da postoje za tu meru.
    """
    fw = function_words(language)

    def real(chunk: list[Word]) -> list[Word]:
        return [w for w in chunk if any(c.isalnum() for c in w.w)]

    def content(chunk: list[Word]) -> list[Word]:
        return [w for w in real(chunk) if _strip_punct(w.w).casefold() not in fw]

    out = {"sada": 0, "bez_funkcijskih": 0, "rečenica": len(chunks)}
    for chunk in chunks:
        if is_low_confidence(chunk, threshold=threshold, share=share):
            out["sada"] += 1
        rest = content(chunk)
        if not rest:
            # Rečenica od samih funkcijskih reči: nema šta da se meri, pa se
            # sudi po starom. Bez ovoga bi „Pa i to." tiho ispala uvek čista.
            if is_low_confidence(chunk, threshold=threshold, share=share):
                out["bez_funkcijskih"] += 1
            continue
        bad = sum(1 for w in rest if w.p < threshold)
        if len(rest) < MIN_WORDS_FOR_SHARE_RULE:
            hit = bad > 0
        else:
            hit = bad / len(rest) >= share
        out["bez_funkcijskih"] += int(hit)
    return out


def would_flag_line(counts: dict[str, int]) -> str:
    """Jedan red merenja, po uzoru na `punctuate.would_pass_line`."""
    n = counts.get("rečenica", 0) or 1
    sada, bez = counts.get("sada", 0), counts.get("bez_funkcijskih", 0)
    return (
        f"low_confidence: sada {sada}/{n} ({100 * sada / n:.0f}%), "
        f"bez funkcijskih reči {bez}/{n} ({100 * bez / n:.0f}%) "
        f"[merenje, pravilo se ne menja]"
    )


def percentile(values: list[float], pct: float) -> float:
    """Vrednost ispod koje je `pct` procenata uzorka (najbliži rang)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round(pct / 100.0 * len(ordered))) - 1))
    return ordered[k]


def analyze_confidence(
    words: list[Word], *, threshold: float = LOW_CONFIDENCE_THRESHOLD
) -> Confidence:
    """Prebroj reči po opsezima verovatnoće i izmeri percentile."""
    ps = [w.p for w in words if any(c.isalnum() for c in w.w)]
    conf = Confidence(words=len(ps), threshold=threshold)
    for p in ps:
        for k, edge in enumerate(CONFIDENCE_EDGES):
            if p < edge:
                conf.counts[k] += 1
                break
        else:
            conf.counts[-1] += 1
    conf.percentiles = {k: round(percentile(ps, k), 4) for k in REPORTED_PERCENTILES}
    return conf


# --- Pokrivenost audija ----------------------------------------------------


def coverage_of_spans(
    spans: list[tuple[float, float]],
    duration: float,
    *,
    min_gap: float = MIN_REPORTED_GAP,
) -> tuple[float, list[tuple[float, float]]]:
    """Koliko je `duration` pokriveno intervalima `spans`.

    Vraća (procenat pokrivenosti, praznine duže od `min_gap` kao (start, end)).
    Uključuje i prazninu na početku i onu na kraju snimka — tu VAD najčešće
    odseče, a bez ove metrike se gubitak ne vidi ni u jednom drugom broju.
    """
    if duration <= 0:
        return 0.0, []

    merged: list[list[float]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    covered = sum(b - a for a, b in merged)
    gaps: list[tuple[float, float]] = []
    prev = 0.0
    for a, b in merged:
        if a - prev >= min_gap:
            gaps.append((round(prev, 2), round(a, 2)))
        prev = b
    if duration - prev >= min_gap:
        gaps.append((round(prev, 2), round(duration, 2)))

    return 100.0 * min(covered, duration) / duration, gaps


def coverage(
    sentences: list[Sentence],
    duration: float,
    *,
    min_gap: float = MIN_REPORTED_GAP,
) -> tuple[float, list[tuple[float, float]]]:
    """Koliko je audija zaista pokriveno rečenicama."""
    return coverage_of_spans(
        [(x.s, x.e) for x in sentences], duration, min_gap=min_gap
    )
