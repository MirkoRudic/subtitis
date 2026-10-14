"""Prelamanje rečenica u cue-ove — deterministički, bez LLM-a.

Postupak: generiši sve kandidate za tačku preloma, svakom dodeli kaznene poene,
pa dinamičkim programiranjem izaberi kombinaciju sa najmanjim zbirom.

Vremena cue-a se IZVODE iz `s` prve i `e` poslednje reči u njemu. Trajanje
rečenice se nikad ne deli proporcionalno broju znakova — imamo prava vremena.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field, replace
from typing import Callable

from . import config as cfg
from .models import Cue, Profile, Sentence, Word
from .segment import (
    NO_SPACE_AFTER,
    NO_SPACE_BEFORE,
    SENTENCE_END_CHARS,
    join_words,
    tokenize_text,
)
from . import typography

# ==========================================================================
# KAZNENI POENI — sve za štelovanje je ovde
# ==========================================================================

#: Po znaku preko `max_chars_per_line`. Namerno ogroman: prelazak je skoro zabrana.
PENALTY_OVER_MAX_CHARS = 1000

#: Prelom koji ne pada na granicu klauze (nije iza zareza/veznika/tačke).
PENALTY_BREAK_NOT_AT_CLAUSE = 50

#: Prelom odmah iza zareza — idealan.
PENALTY_BREAK_AFTER_COMMA = 0

#: Prelom na vezniku. Srpski: IZA veznika (i, ali, da, jer, pa, te, ili).
#: Engleski i francuski: ISPRED veznika — vidi `LanguageRules.conjunction_side`.
PENALTY_BREAK_AFTER_CONJUNCTION = 10

#: Prelom razdvaja predlog od imenice koja mu pripada.
PENALTY_SPLIT_PREPOSITION = 300

#: Prelom razdvaja pridev od imenice (heuristika po nastavku).
PENALTY_SPLIT_ADJECTIVE = 200

#: Prelom razdvaja delove složenog glagola (bi, je, su, ću, ćeš...).
#: Važi u oba smera: enklitika ne sme ni da završi ni da počne red.
PENALTY_SPLIT_COMPOUND_VERB = 300

#: Prelom unutar višečlanog termina iz glosara ("Open Door / for Children").
#: Reda veličine kazne za predlog: termin je ugovoren kao celina i čita se kao ime.
PENALTY_SPLIT_GLOSSARY_TERM = 300

#: Jedna jedina reč sama u redu (samo kad cue ima više od jednog reda).
PENALTY_ORPHAN_WORD = 400

#: Gornji red duži od donjeg, a prelom nije na jasnoj granici klauze.
PENALTY_TOP_LONGER = 30

#: Po znaku razlike u dužini susednih redova. Tabela iz specifikacije kažnjava
#: samo smer "gornji duži", pa bi bez ovoga prelom "Danas ćemo / pričati o jednoj
#: važnoj temi" bio jeftiniji od uravnoteženog. Drži redove približno jednakim.
PENALTY_LINE_IMBALANCE = 2

#: Po jedinici CPS preko `max_cps`.
PENALTY_OVER_MAX_CPS = 100

# --- Dodato preko tabele iz specifikacije (nije bilo zadato, ali je nužno) ---

#: Po sekundi preko `max_duration`. Bez ovoga DP nema razlog da lomi duge pauze.
PENALTY_OVER_MAX_DURATION = 100

#: Fiksni trošak svakog novog cue-a; blago favorizuje popunjene cue-ove
#: umesto rafala kratkih. Spusti na 0 ako želiš čisto tabelarno ponašanje.
PENALTY_PER_CUE = 20

#: Ispod ovoliko znakova cue "blicne" i deluje isprekidano.
MIN_CUE_CHARS = 20

#: Po znaku ispod `MIN_CUE_CHARS`. Ne važi kad je cue cela rečenica —
#: "Da." je legitiman cue, "It was" otcepljen od nastavka nije.
PENALTY_SHORT_CUE = 20

#: Po znaku ispod `Profile.min_chars_per_line` (samo profili koji je imaju).
#: Ne važi samo kad je CELA rečenica kraća od granice (`Da.`, `Ne znam.`).
#: Krug 7: izuzetak za kratak OSTATAK rečenice (krug 6) je gasio kaznu za
#: svaki poslednji komad, pa se `šta / ima?` nikad nije poredilo sa
#: `Čao, Nikola, / šta ima?` — ostatak je kratak upravo zbog lošeg reza. Kod takvog profila
#: ona zamenjuje `MIN_CUE_CHARS`/`PENALTY_SHORT_CUE`: dve kazne za isto bi se
#: sabirale, a `MIN_CUE_CHARS` (20) je za `digital` ceo kapacitet cue-a.
#: Mereno 15/30/60 na 12 ASR poslova (`merenja/dnevnik.md`, 2026-10-07):
#: 30 je najbolje po rezovima kroz sintagmu (22 od 244), CPS-u i kratkom
#: trajanju; 60 prepolovi kratke redove bez izuzetka (15 -> 9) za dva reza
#: kroz sintagmu više — a u jednom redu se `u` i `kuću` pojave u različitim
#: trenucima, pa je taj rez teža greška nego u dvorednom titlu.
PENALTY_BELOW_MIN_CHARS = 30

#: Po znaku odstupanja dužine cue-a od ravnomernog udela rečenice, samo kad
#: je `max_lines` 1. Tada su uzastopni cue-ovi iste rečenice ono što je u
#: dvorednom titlu drugi red, pa neravnoteža meri njih, a ne redove u cue-u
#: (`PENALTY_LINE_IMBALANCE`). Meri se prema proseku delova koje je prelamanje
#: stvarno izabralo (`_layout_even`); skorovanje je isto kao u
#: `split_by_shares` (`share_deviation`). 0 = isključeno.
#: ISKLJUČENO, mereno (krug 7, `merenja/dnevnik.md`): na 13 poslova težina 2
#: menja 7 od 59 rečenica sa više cue-ova — tri bolje, dve gore, dve
#: neutralne — a zbirni brojevi ostaju isti. Mehanizam ostaje za ponovno
#: merenje, kao `fixedcues.SRT_MAX_GAP`.
PENALTY_CUE_SHARE_DEVIATION = 0

# --- Pauza u govoru (krug 19) ----------------------------------------------
# Tišine dolaze iz zvuka (`pauses.py`, Silero VAD), ne iz razmaka između reči
# u `words.json` — taj je na granici bez interpunkcije tačno 0 u 825 od 830
# slučajeva. Važe samo za prelamanje izvornog jezika sa pravim vremenima;
# prevod i uvoz gotovog SRT-a nemaju zvuk po rečima i rade kao ranije.
#
# SVE ISKLJUČENO, mereno (krug 19, `merenja/dnevnik.md`, 2026-10-12): na 15
# ASR poslova (13 sa zvukom), mreža prag 0,12/0,20/0,30/0,45 s × težina
# 50/100/200 × popust 25/50. Kazna za pauzu u cue-u ni u jednoj tački ne
# prolazi kriterijum zapisan pre merenja — cepa natpise, pa rastu
# prekoračenja CPS-a (`tv` 3 -> 4–5) ili kratki cue-ovi (`web` +1–2). Prolazi
# samo popust; najbolji (prag 0,30, popust 50) menja 10 rečenica: 4 bolje,
# 4 gore, 2 neutralne — u `tv` 4 : 1, u jednorednom `web` 0 : 3. Ispod
# traženih 3 : 1, pa se ne uvodi. Mehanizam ostaje za ponovno merenje, kao
# `PENALTY_CUE_SHARE_DEVIATION`; `pauses.json` se svejedno čuva pri
# prepoznavanju govora, jer posle brisanja zvuka tišina više nema odakle.

#: Tišina od ovoliko sekundi naviše je pauza. 0,30 je jedini prag na kome je
#: popust prošao kriterijum u sva tri profila sa padom većim od jednog cue-a.
PAUSE_MIN = 0.30

#: Po sekundi najveće pauze iznad `PAUSE_MIN` unutar cue-a: u isti natpis su
#: spojene reči koje govornik nije izgovorio u dahu. Najviše
#: `PENALTY_PAUSE_IN_CUE_MAX` — reda veličine `PENALTY_BREAK_NOT_AT_CLAUSE`,
#: nikad reda kazne za predlog, enklitiku ili termin (300): govornik ume da
#: udahne usred sintagme, i `u / kuću` ostaje gore od pauze u natpisu.
#: ISKLJUČENO: nijedna težina nije prošla kriterijum (vidi gore).
PENALTY_PAUSE_IN_CUE = 0
PENALTY_PAUSE_IN_CUE_MAX = 50

#: Popust na `PENALTY_BREAK_NOT_AT_CLAUSE` za rez na pauzi: pauza je granica
#: klauze, samo čujna umesto pisane. Ne dira kazne za predlog i ostale.
#: ISKLJUČENO: 50 prolazi brojke, ali ne i čitanje (4 : 4).
PAUSE_BREAK_DISCOUNT = 0

#: Kazna za rez van klauze BEZ pauze — ista informacija gledana naopako.
#: ISKLJUČENO: uz popust 50 dodaje 1 cue manje sa pauzom, a diže CPS (`tv`)
#: i kratke cue-ove (`web`).
PENALTY_BREAK_WITHOUT_PAUSE = 0


#: Pauza na granici na kojoj rez nije dozvoljen — iza predloga, enklitike,
#: negacije ili veznika, ili ispred enklitike — nije granica misli nego
#: oklevanje: govornik udahne, sintagma se nastavlja (krug 22: jača kazna ->
#: češći kraj natpisa na `i`, `ali`, `a`, `pa`, `da`). "keep" — ostaje;
#: "discard" — ne ulazi ni u jedan mehanizam pauze.
#: Krug 23 (dnevnik, 2026-10-13): lažnih je 35–45 % pauza; uz "discard" u
#: `tv` krajeva natpisa na vezniku 12 -> 0, čitanje 6 : 11 -> 9 : 5.
#: Treća varijanta, premeštanje pauze na granicu ispred te reči, merena je i
#: izbačena (krug 24): čita se isto (9 : 5), a ostavlja 1 kraj na vezniku
#: prema 0 — složenija, bez dobiti.
PAUSE_AT_FUNCTION_WORD = "keep"

#: Srpski veznik koji nije u `SR_CONJUNCTIONS` (strana preloma se za njega ne
#: meri), a govornik posle njega zastaje (krug 22: `a / posebno`).
SR_PAUSE_FALSE_EXTRA = frozenset({"a"})


def false_pause_at(words: list[Word], k: int, rules: LanguageRules) -> str | None:
    """Da li je pauza upisana na token k lažna: "left" ako je iza reči na kojoj
    natpis ne sme da se završi, "right" ako je ispred enklitike, inače None.
    Pauza upisana na interpunkciju je na zarezu — granica klauze, prava."""
    if k + 1 >= len(words) or _is_punct(words[k].w):
        return None
    left = _norm(words[k].w)
    extra = SR_PAUSE_FALSE_EXTRA if rules.code == "sr" else frozenset()
    if left in rules.no_line_end or left in rules.conjunctions or left in extra:
        return "left"
    if _norm(words[k + 1].w) in rules.no_line_start:
        return "right"
    return None


def adjust_pauses(
    words: list[Word], pauses: list[float], rules: LanguageRules, mode: str
) -> list[float]:
    """Primeni `PAUSE_AT_FUNCTION_WORD` na pauze po tokenu (`boundary_pauses`)."""
    if mode == "keep":
        return pauses
    return [0.0 if p and false_pause_at(words, k, rules) else p for k, p in enumerate(pauses)]


#: Paket pauze za DVOREDNI prelom (`max_lines >= 2`), uveden u krugu 25 —
#: vezan za oblik kao `conjunction_side_multiline` (krug 21). Jednoredni
#: (`web`, `digital`) ostaje na vrednostima iznad: tamo se čita 1 : 3.
#: Mereno u krugu 24 (dnevnik, 2026-10-14): uz "discard" krajeva natpisa na
#: vezniku 12 -> 0, uz kaznu za zamenicu čitanje 5 : 1 (ponovljeni snimak se
#: broji jednom, krug 25). Merne skripte ga isključuju (`None`) da bi stari
#: krugovi ostali ponovljivi.
PAUSE_MULTILINE: dict[str, float | str] | None = {
    "PAUSE_MIN": 0.12,
    "PENALTY_PAUSE_IN_CUE": 100,  # po sekundi, najviše PENALTY_PAUSE_IN_CUE_MAX
    "PAUSE_BREAK_DISCOUNT": 50,
    "PENALTY_BREAK_WITHOUT_PAUSE": 0,
    "PAUSE_AT_FUNCTION_WORD": "discard",
    "PENALTY_SPLIT_PRONOUN": 25,
}


def pause_params(profile: Profile | None = None) -> dict[str, float | str]:
    """Važeće vrednosti pauze za oblik: modulske, a za dvoredni `PAUSE_MULTILINE`."""
    pp: dict[str, float | str] = {
        "PAUSE_MIN": PAUSE_MIN,
        "PENALTY_PAUSE_IN_CUE": PENALTY_PAUSE_IN_CUE,
        "PAUSE_BREAK_DISCOUNT": PAUSE_BREAK_DISCOUNT,
        "PENALTY_BREAK_WITHOUT_PAUSE": PENALTY_BREAK_WITHOUT_PAUSE,
        "PAUSE_AT_FUNCTION_WORD": PAUSE_AT_FUNCTION_WORD,
        "PENALTY_SPLIT_PRONOUN": PENALTY_SPLIT_PRONOUN,
    }
    if profile is not None and profile.max_lines >= 2 and PAUSE_MULTILINE:
        pp.update(PAUSE_MULTILINE)
    return pp


def pauses_enabled(profile: Profile | None = None) -> bool:
    """Da li prelamanje za ovaj oblik uopšte koristi pauze — bez toga se zvuk
    ne čita. Bez profila: da li ih koristi ijedan oblik."""
    def on(pp: dict[str, float | str]) -> bool:
        return bool(pp["PENALTY_PAUSE_IN_CUE"] or pp["PAUSE_BREAK_DISCOUNT"]
                    or pp["PENALTY_BREAK_WITHOUT_PAUSE"])
    if profile is not None:
        return on(pause_params(profile))
    return on(pause_params()) or bool(PAUSE_MULTILINE and on({**pause_params(), **PAUSE_MULTILINE}))

#: Gornja granica pretrage — koliko puta više znakova od kapaciteta cue-a
#: sme da uđe u kandidat-raspon. Samo ograničava DP, ne utiče na izbor.
SEARCH_WIDTH_FACTOR = 1.6

#: Najmanji udeo tokena ručno izmenjene rečenice koji sme da bude poravnat sa
#: originalnim ASR vremenima. Ispod toga su vremena uglavnom interpolirana, pa
#: rečenica dobija flag `alignment_degraded` i upozorenje u logu — tiho loše
#: poravnanje se inače otkrije tek u plejeru.
MIN_ALIGNED_TOKEN_SHARE = 0.6

ALIGNMENT_DEGRADED_FLAG = "alignment_degraded"

INF = math.inf

# ==========================================================================
# Leksičke liste
# ==========================================================================

SR_PREPOSITIONS = frozenset({
    "u", "na", "o", "od", "do", "za", "sa", "s", "iz", "kod", "pre", "po",
    "uz", "niz", "kroz", "pored", "preko", "ispod", "iznad", "među", "nad",
    "pod", "pri", "prema", "protiv", "radi", "zbog", "bez",
})

#: Negacija se piše odvojeno, a izgovara kao jedna akcenatska reč sa
#: glagolom (`ne možemo`): kao predlog, ne sme da ZAVRŠI red (krug 7).
SR_NEGATION = frozenset({"ne"})

#: Veznici iz specifikacije. Strana preloma zavisi od OBLIKA (krugovi 20–21,
#: `merenja/dnevnik.md`, mereno 2026-10-12 na 15 ASR poslova):
#:
#: * jedan red (`web`, `digital`): IZA veznika (`conjunction_side` "after").
#:   `before` se tu čita 11 : 6 i diže rez iza predloga (3 -> 4) i kratke
#:   cue-ove (33 -> 36) — u redu od par reči rez nema kuda da se pomeri.
#: * dva reda (`tv`): ISPRED veznika (`conjunction_side_multiline`
#:   "before"), kao engleski i francuski — `treba / da bude`. Čitanje 16 : 1
#:   (bez ponovljenih snimaka 14 : 1). Viđena i prihvaćena granična
#:   pogoršanja: jedan cue od 19 znakova pri granici 20 (u rečenici koja je
#:   bolja) i jedan cue od 20,5 CPS pri granici 20.
#:
#: `da` izbačeno iz veznika (kao enklitika) nije prošlo ni u jednom obliku.
SR_CONJUNCTIONS = frozenset({"i", "ali", "da", "jer", "pa", "te", "ili"})

#: Enklitike pomoćnih glagola — ne smeju se odvojiti od participa/infinitiva.
#: Jedine u ovim tabelama koje ne smeju ni da POČNU red (enklitika po definiciji
#: nema svoj naglasak, pa ne može stajati na početku).
SR_CLITICS = frozenset({
    "bi", "bih", "bismo", "biste", "je", "su", "sam", "si", "smo", "ste",
    "ću", "ćeš", "će", "ćemo", "ćete", "se", "ga", "mu", "joj", "ih", "im",
})

#: Lične zamenice u nominativu (`mi`, `vi`, `ti` su i enklitike dativa —
#: i tada se drže reči uz koju idu). Krug 24.
SR_PERSONAL_PRONOUNS = frozenset({"ja", "ti", "on", "ona", "ono", "mi", "vi", "oni", "one"})

#: Rez odmah iza lične zamenice: zamenica ostaje bez svog glagola. Subjekat
#: koji nije zamenica (`cela ta priča / mora`) se ovako ne vidi — za to bi
#: trebala sintaksa, a ne lista reči. Mereno u krugu 24 (dnevnik).
#: NIJE SAMOSTALNA: sama, bez pauza, ne menja nijednu rečenicu ni na 100 —
#: rez iza zamenice bez pauze prelom ionako ne bira. Uključena je samo u
#: paketu pauze za dvoredni prelom (`PAUSE_MULTILINE`), jer je tamo pauza
#: iza zamenice vukla rez (`mi / nikome nećemo`). Bez paketa nema svrhu.
PENALTY_SPLIT_PRONOUN = 0

#: Gruba heuristika za pridev: tipični nastavci pridevske promene.
SR_ADJECTIVE_SUFFIXES = (
    "ski", "ška", "ško", "čki", "čka", "čko", "ni", "na", "no", "ne", "nu",
    "og", "om", "im", "ih", "oj", "ov", "in", "an", "ni", "ki",
)

# --- engleski ---

EN_ARTICLES = frozenset({"a", "an", "the"})

#: `to` je i predlog i marker infinitiva — isti token, isto pravilo.
EN_PREPOSITIONS = frozenset({
    "in", "on", "at", "to", "of", "for", "with", "from", "by", "into",
    "about", "over", "under", "between", "through",
})

EN_AUXILIARIES = frozenset({
    "is", "are", "was", "were", "has", "have", "had", "will", "would",
    "can", "could", "should", "must", "do", "does", "did",
})

#: Prisvojni i pokazni determinatori drže imenicu uz sebe, kao član
#: (`their / biggest`). Ista kazna kao član: `no_line_end`.
#: `that` NAMERNO nije ovde: izmereno 2026-09-25 na prvom SRT poslu, od 10
#: pojava samo 1 je determinator (`that chance`), 6 su veznik (`I hope that /
#: in the years`), ostalo zamenice — kazna bi uglavnom pogađala pogrešne.
#: `her` je i zamenica objekta (`saw her / yesterday`); nije mereno.
EN_DETERMINERS = frozenset({
    "my", "your", "his", "her", "its", "our", "their",
    "this", "these", "those",
})

#: Veznici. Engleski lomi ISPRED veznika (`conjunction_side="before"`): izmereno
#: 2026-09-25 na prvom SRT poslu, 26 od 92 reza je padalo iza `and`/`so`
#: dok je pravilo nagrađivalo rez iza veznika (pisano za srpski).
#: `that`, `which`, `who`, `where` uvode zavisnu rečenicu isto kao veznici i
#: lome se ispred njih (`to understand / that in Europe`); dok nisu bili na
#: spisku, rez je padao IZA njih.
EN_CONJUNCTIONS = frozenset({
    "and", "but", "or", "so", "because", "although", "while", "if", "when",
    "that", "which", "who", "where",
})

# --- francuski ---

FR_ARTICLES = frozenset({"le", "la", "les", "un", "une", "des", "du", "de"})

FR_PREPOSITIONS = frozenset({
    "à", "dans", "sur", "pour", "avec", "par", "sans", "sous", "entre",
    "chez", "vers",
})

FR_AUXILIARIES = frozenset({
    "est", "sont", "était", "a", "ont", "avait", "sera", "peut", "doit",
})

FR_DETERMINERS = frozenset({
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses",
    "notre", "nos", "votre", "vos", "leur", "leurs",
    "ce", "cet", "cette", "ces",
})

#: Kao i za engleski: lomi se ispred veznika.
#: `parce que` su dva tokena: `parce` je na spisku da bi se lomilo ispred
#: njega, a u `FR_NO_SPLIT_AFTER` da `parce / que` ne bi bio jeftin rez ispred
#: `que`.
FR_CONJUNCTIONS = frozenset({
    "et", "mais", "ou", "car", "donc", "que", "si", "quand", "comme",
    "qui", "dont", "lorsque", "parce",
})

#: Prvi deo višečlanog veznika — ne sme da završi red.
FR_NO_SPLIT_AFTER = frozenset({"parce"})

#: `ne` kao u srpskom (`je ne / sais` je rez kroz glagol); elidirano `n'`
#: već hvata pravilo o apostrofu (`FR_ELIDED`).
FR_NEGATION = frozenset({"ne"})

#: Elidirani oblici se nikad ne odvajaju od sledeće reči. Pored ovih, isto važi
#: za svaki token koji se završava apostrofom (`s'`, `j'`, `m'`, `c'`…).
FR_ELIDED = frozenset({"l'", "d'", "qu'", "n'", "s'", "j'", "m'", "t'", "c'"})

APOSTROPHES = "'’"


# ==========================================================================
# Jezičke tabele
# ==========================================================================


@dataclass(frozen=True)
class LanguageRules:
    """Funkcijske reči jednog jezika.

    `no_line_end`  — token ne sme da ZAVRŠI red (kazna `PENALTY_SPLIT_PREPOSITION`)
    `no_line_start`— token ne sme da POČNE red (kazna `PENALTY_SPLIT_COMPOUND_VERB`)
    """

    code: str
    no_line_end: frozenset[str] = frozenset()
    no_line_start: frozenset[str] = frozenset()
    conjunctions: frozenset[str] = frozenset()
    adjective_suffixes: tuple[str, ...] = ()
    #: Francuski: token koji se završava apostrofom drži sledeću reč uz sebe.
    keep_elisions: bool = False
    #: Pravopis jezika (`typography`) menja dužinu reda: francuski dodaje
    #: nerazlomljive razmake i `« »`. Kad je True, dužina reda se meri na redu
    #: provučenom kroz `typography.apply` — isto ono što ide u fajl — da red
    #: ne probije `max_chars_per_line` tek posle obrade.
    typographic_width: bool = False
    #: Na kojoj strani veznika je prelom jeftin (`PENALTY_BREAK_AFTER_CONJUNCTION`).
    #: "after" — srpski, iz specifikacije: `i / onda`.
    #: "before" — engleski i francuski lome ISPRED veznika: `old / and when`.
    conjunction_side: str = "after"
    #: Strana veznika kad profil ima dva reda ili više (`max_lines >= 2`);
    #: None = ista kao `conjunction_side`. Postoji zato što je strana veznika
    #: i svojstvo oblika, ne samo jezika — vidi `SR_CONJUNCTIONS`.
    conjunction_side_multiline: str | None = None


LANGUAGE_RULES: dict[str, LanguageRules] = {
    "sr": LanguageRules(
        code="sr",
        no_line_end=SR_PREPOSITIONS | SR_CLITICS | SR_NEGATION,
        no_line_start=SR_CLITICS,
        conjunctions=SR_CONJUNCTIONS,
        adjective_suffixes=SR_ADJECTIVE_SUFFIXES,
        conjunction_side_multiline="before",
    ),
    "en": LanguageRules(
        code="en",
        no_line_end=EN_ARTICLES | EN_DETERMINERS | EN_PREPOSITIONS | EN_AUXILIARIES,
        conjunctions=EN_CONJUNCTIONS,
        conjunction_side="before",
    ),
    "fr": LanguageRules(
        code="fr",
        no_line_end=(
            FR_ARTICLES | FR_DETERMINERS | FR_PREPOSITIONS | FR_AUXILIARIES | FR_ELIDED
            | FR_NO_SPLIT_AFTER | FR_NEGATION
        ),
        conjunctions=FR_CONJUNCTIONS,
        conjunction_side="before",
        keep_elisions=True,
        typographic_width=True,
    ),
}

#: Jezik bez tabele: samo opšta pravila (dužina, balans, CPS, trajanje).
GENERIC_RULES = LanguageRules(code="?")


def rules_for(language: str | None, log: Callable[[str], None] | None = None) -> LanguageRules:
    """Tabela za jezik; uz upozorenje ako je nema — da lomljenje bilo gde ne prođe tiho."""
    code = (language or "").split("-")[0].lower()
    rules = LANGUAGE_RULES.get(code)
    if rules is None:
        msg = (
            f"nema jezičkih pravila za '{language}' — prelamam samo po opštim "
            f"pravilima (dužina, balans, CPS). Funkcijske reči se neće poštovati. "
            f"Poznati jezici: {', '.join(sorted(LANGUAGE_RULES))}."
        )
        (log or (lambda m: cfg.echo(f"UPOZORENJE: {m}")))(msg)
        return GENERIC_RULES
    return rules


# ==========================================================================
# Pomoćne funkcije
# ==========================================================================


def _norm(tok: str) -> str:
    return tok.strip().strip(",.!?…:;\"»«„“”‘’()[]{}").lower().strip("'")


def _norm_keep_apostrophe(tok: str) -> str:
    """Kao `_norm`, ali zadržava završni apostrof (`l'`, `qu'`)."""
    return tok.strip().strip(",.!?…:;\"»«„“‘()[]{}").lower()


def _is_punct(tok: str) -> bool:
    t = tok.strip()
    return bool(t) and not any(ch.isalnum() for ch in t)


def _is_clause_boundary(tok: str, rules: LanguageRules, nxt: str = "") -> bool:
    """Da li rez iza `tok` (ispred `nxt`) pada na granicu klauze."""
    t = tok.strip()
    if t.endswith(",") or t == ",":
        return True
    if t and t[-1] in SENTENCE_END_CHARS:
        return True
    return _conjunction_cut(t, nxt, rules)


def _conjunction_cut(tok: str, nxt: str, rules: LanguageRules) -> bool:
    """Rez na pravoj strani veznika. Kod "before" veznik ne sme sam da završi
    red: `old and / when` je rez ispred `when`, ali ostavlja `and` na kraju."""
    if rules.conjunction_side == "before":
        return _norm(nxt) in rules.conjunctions and _norm(tok) not in rules.conjunctions
    return _norm(tok) in rules.conjunctions


def _looks_adjective(tok: str, rules: LanguageRules) -> bool:
    if not rules.adjective_suffixes:
        return False
    n = _norm(tok)
    return len(n) > 3 and n.endswith(rules.adjective_suffixes)


class _Span:
    """Predračun dužina i kazni za jednu rečenicu."""

    def __init__(
        self, words: list[Word], profile: Profile, rules: LanguageRules = GENERIC_RULES,
        terms: list[list[str]] | None = None, pauses: list[float] | None = None,
    ) -> None:
        self.words = words
        self.profile = profile
        if profile.max_lines >= 2 and rules.conjunction_side_multiline:
            rules = replace(rules, conjunction_side=rules.conjunction_side_multiline)
        self.rules = rules
        #: pauza posle tokena k (`pauses.boundary_pauses`); None = bez zvuka
        self.pauses = pauses
        #: vrednosti pauze i zamenice za OVAJ oblik (`pause_params`)
        self.pp = pause_params(profile)
        n = len(words)
        self.n = n

        # razmak ispred tokena k (k > 0)
        space = [0] * n
        for k in range(1, n):
            tok = words[k].w
            prev = words[k - 1].w
            glue = (tok[:1] in NO_SPACE_BEFORE) or (prev[-1:] in "([{«„‘")
            space[k] = 0 if glue else 1
        self.space = space

        self.len_cum = [0] * (n + 1)
        self.space_cum = [0] * (n + 1)
        self.wc_cum = [0] * (n + 1)  # kumulativni broj tokena koji nisu interpunkcija
        for k in range(n):
            self.len_cum[k + 1] = self.len_cum[k] + len(words[k].w)
            self.space_cum[k + 1] = self.space_cum[k] + space[k]
            self.wc_cum[k + 1] = self.wc_cum[k] + (0 if _is_punct(words[k].w) else 1)

        # prelom iza indeksa k je dozvoljen samo ako sledeći token nije
        # interpunkcija, a ovaj nije otvorena zagrada/navodnik (`«` na kraju reda)
        self.can_break = [
            k < n - 1
            and words[k + 1].w[:1] not in NO_SPACE_BEFORE
            and words[k].w[-1:] not in NO_SPACE_AFTER
            for k in range(n)
        ]

        # Navodnici menjaju dužinu u svakom jeziku (`" Comment` -> `"Comment`),
        # francuski pravopis i bez njih. Merenje preko `typography` je sporije,
        # pa samo tamo gde menja rezultat.
        self._render_lang = (
            rules.code
            if rules.typographic_width or any(
                q in w.w for w in words for q in typography.QUOTE_CHARS
            )
            else None
        )
        self._len_memo: dict[tuple[int, int], int] = {}

        self.break_pen = [self._break_penalty(k) for k in range(n)]
        for k in _term_interior_cuts(words, terms or []):
            if self.break_pen[k] < INF:
                self.break_pen[k] += PENALTY_SPLIT_GLOSSARY_TERM
        #: memo za `_lines_dp`, deljen između svih kandidat-cue-ova ove rečenice
        self.memo: dict[tuple[int, int, int], tuple[float, tuple[int, ...]]] = {}

    # ---- dužine ----

    def length(self, i: int, j: int) -> int:
        """Broj znakova za tokene [i, j) — onako kako će red izaći u fajl."""
        if j <= i:
            return 0
        if self._render_lang is not None:
            key = (i, j)
            hit = self._len_memo.get(key)
            if hit is None:
                line = join_words(self.words[i:j])
                hit = len(typography.apply([line], self._render_lang)[0])
                self._len_memo[key] = hit
            return hit
        return (self.len_cum[j] - self.len_cum[i]) + (
            self.space_cum[j] - self.space_cum[i + 1]
        )

    def word_count(self, i: int, j: int) -> int:
        return self.wc_cum[j] - self.wc_cum[i]

    # ---- kazne ----

    def _break_penalty(self, k: int) -> float:
        """Kazna za prelom odmah iza tokena k."""
        if not self.can_break[k]:
            return INF
        tok = self.words[k].w.strip()
        nxt = self.words[k + 1].w.strip()

        rules = self.rules
        t = tok.rstrip()
        if t.endswith(",") or t == ",":
            pen = float(PENALTY_BREAK_AFTER_COMMA)
        elif t and t[-1] in SENTENCE_END_CHARS:
            pen = float(PENALTY_BREAK_AFTER_COMMA)  # granica klauze, isto 0
        elif _conjunction_cut(t, nxt, rules):
            pen = float(PENALTY_BREAK_AFTER_CONJUNCTION)
        else:
            pen = float(PENALTY_BREAK_NOT_AT_CLAUSE)
            if self.pauses is not None:
                pp = self.pp
                if self.pauses[k] >= pp["PAUSE_MIN"]:
                    pen = max(0.0, pen - pp["PAUSE_BREAK_DISCOUNT"])
                else:
                    pen += pp["PENALTY_BREAK_WITHOUT_PAUSE"]

        left = _norm(tok)
        right = _norm(nxt)
        left_elided = _norm_keep_apostrophe(tok)

        # funkcijska reč ne sme da završi red: predlog, član, pomoćni glagol,
        # elidirani oblik
        if left in rules.no_line_end or left_elided in rules.no_line_end:
            pen += PENALTY_SPLIT_PREPOSITION
        elif rules.keep_elisions and left_elided.endswith(tuple(APOSTROPHES)):
            pen += PENALTY_SPLIT_PREPOSITION

        # ne sme da počne red (srpske enklitike)
        if right in rules.no_line_start:
            pen += PENALTY_SPLIT_COMPOUND_VERB

        # lična zamenica odvojena od svog glagola (`mi / nikome nećemo`)
        if (self.pp["PENALTY_SPLIT_PRONOUN"] and rules.code == "sr"
                and left in SR_PERSONAL_PRONOUNS and right and not _is_punct(nxt)):
            pen += self.pp["PENALTY_SPLIT_PRONOUN"]

        if (
            _looks_adjective(tok, rules)
            and right
            and not _is_punct(nxt)
            and left not in rules.no_line_end
        ):
            pen += PENALTY_SPLIT_ADJECTIVE
        return pen

    def line_cost(self, i: int, j: int, penalize_orphan: bool) -> float:
        cost = 0.0
        over = self.length(i, j) - self.profile.max_chars_per_line
        if over > 0:
            cost += over * PENALTY_OVER_MAX_CHARS
        if penalize_orphan and self.word_count(i, j) <= 1:
            cost += PENALTY_ORPHAN_WORD
        floor = self.profile.min_chars_per_line
        if floor:
            short = floor - self.length(i, j)
            if short > 0 and self.length(0, self.n) >= floor:
                cost += short * PENALTY_BELOW_MIN_CHARS
        return cost


# ==========================================================================
# Unutrašnji DP: raspored tokena [i, j) u najviše `max_lines` redova
# ==========================================================================


def _term_key(tok: str) -> str:
    return tok.lower().replace("’", "'")


def term_tokens(terms: list[str]) -> list[list[str]]:
    """Termini glosara -> tokeni za poređenje. Samo višečlani: jednu reč nema
    šta da preseče."""
    out = []
    for term in terms:
        toks = [_term_key(t) for t in tokenize_text(term)]
        if len(toks) > 1:
            out.append(toks)
    return out


def _term_interior_cuts(words: list[Word], terms: list[list[str]]) -> set[int]:
    """Indeksi k takvi da bi rez iza tokena k presekao neki termin."""
    if not terms:
        return set()
    keys = [_term_key(w.w) for w in words]
    cuts: set[int] = set()
    for term in terms:
        m = len(term)
        for i in range(len(keys) - m + 1):
            if keys[i : i + m] == term:
                cuts.update(range(i, i + m - 1))
    return cuts


def _cut_extra(span: _Span, cut: int, top: int, bottom: int) -> float:
    """Kazna za (ne)balans dva susedna reda razdvojena rezom `cut`."""
    extra = abs(top - bottom) * PENALTY_LINE_IMBALANCE
    nxt = span.words[cut + 1].w if cut + 1 < span.n else ""
    if top > bottom and not _is_clause_boundary(span.words[cut].w, span.rules, nxt):
        extra += PENALTY_TOP_LONGER
    return extra


def _lines_dp(span: _Span, pos: int, lines_left: int, j: int) -> tuple[float, tuple[int, ...]]:
    """Raspored [pos, j) u najviše `lines_left` redova; svaki red se kažnjava
    za usamljenu reč.

    Memo je vezan za rečenicu (`span`), pa se podproblemi dele između svih
    kandidat-cue-ova — bez toga je prelamanje jednog sata govora minutima sporo.
    """
    key = (pos, lines_left, j)
    hit = span.memo.get(key)
    if hit is not None:
        return hit
    if pos >= j:
        return INF, ()

    best: tuple[float, tuple[int, ...]] = (span.line_cost(pos, j, True), ())
    if lines_left >= 2:
        for cut in range(pos, j - 1):
            bp = span.break_pen[cut]
            if bp == INF:
                continue
            head = span.line_cost(pos, cut + 1, True)
            tail_cost, tail_cuts = _lines_dp(span, cut + 1, lines_left - 1, j)
            if tail_cost >= INF:
                continue
            nxt_end = tail_cuts[0] + 1 if tail_cuts else j
            extra = _cut_extra(
                span, cut, span.length(pos, cut + 1), span.length(cut + 1, nxt_end)
            )
            total = head + bp + extra + tail_cost
            if total < best[0]:
                best = (total, (cut,) + tail_cuts)

    span.memo[key] = best
    return best


def _best_lines(span: _Span, i: int, j: int) -> tuple[float, list[int]]:
    """Vrati (kazna, lista tačaka reza) za cue koji obuhvata tokene [i, j)."""
    if i >= j:
        return INF, []

    # Jedini red jednočlanog cue-a koji pokriva celu rečenicu ne broji se kao
    # "usamljena reč" — "Da." je sasvim ispravan titl. Kod jednog reda po cue-u
    # ne broji se nikad: reč sama u redu je tu nastavak, a ne visi ispod
    # drugog reda (vezano za broj redova, ne za ime profila).
    covers_whole_sentence = i == 0 and j == span.n
    best: tuple[float, tuple[int, ...]] = (
        span.line_cost(i, j, not covers_whole_sentence and span.profile.max_lines > 1),
        (),
    )

    if span.profile.max_lines >= 2:
        for cut in range(i, j - 1):
            bp = span.break_pen[cut]
            if bp == INF:
                continue
            head = span.line_cost(i, cut + 1, True)
            tail_cost, tail_cuts = _lines_dp(span, cut + 1, span.profile.max_lines - 1, j)
            if tail_cost >= INF:
                continue
            nxt_end = tail_cuts[0] + 1 if tail_cuts else j
            extra = _cut_extra(
                span, cut, span.length(i, cut + 1), span.length(cut + 1, nxt_end)
            )
            total = head + bp + extra + tail_cost
            if total < best[0]:
                best = (total, (cut,) + tail_cuts)

    return best[0], list(best[1])


# ==========================================================================
# Spoljašnji DP: rečenica -> niz cue-ova
# ==========================================================================


def pause_in_cue_penalty(pause: float, pp: dict[str, float | str] | None = None) -> float:
    """Kazna za cue čija je najveća unutrašnja pauza `pause` sekundi."""
    pp = pp or pause_params()
    w, lo = float(pp["PENALTY_PAUSE_IN_CUE"]), float(pp["PAUSE_MIN"])
    if not w or pause < lo:
        return 0.0
    return min(w * (pause - lo), PENALTY_PAUSE_IN_CUE_MAX)


def layout_sentence(
    words: list[Word], profile: Profile, sentence_id: int | None = None,
    flags: list[str] | None = None, rules: LanguageRules | None = None,
    terms: list[list[str]] | None = None,
    silences: list[tuple[float, float]] | None = None,
) -> list[Cue]:
    """Prelomi jednu rečenicu (njene reči) u cue-ove.

    `rules` podrazumevano srpski — to je izvorni jezik alata. Za prevedene
    rečenice `build_cues` prosleđuje tabelu ciljnog jezika. `silences` su
    tišine iz zvuka (`pauses.py`); bez njih se pauza ne uzima u obzir.
    """
    words = [w for w in words if w.w.strip()]
    if not words:
        return []

    pauses = None
    if silences is not None:
        from .pauses import boundary_pauses

        pauses = adjust_pauses(
            words, boundary_pauses(words, silences), rules or LANGUAGE_RULES["sr"],
            str(pause_params(profile)["PAUSE_AT_FUNCTION_WORD"]),
        )
    span = _Span(words, profile, rules or LANGUAGE_RULES["sr"], terms, pauses)
    n = span.n
    capacity = profile.max_chars_per_line * profile.max_lines
    max_search = capacity * SEARCH_WIDTH_FACTOR

    # Svi dozvoljeni cue-ovi [i, j) sa troškom koji zavisi samo od njih.
    edges: list[list[tuple[int, float, list[int]]]] = [[] for _ in range(n)]
    for i in range(n - 1, -1, -1):
        inner_pause = 0.0  # najveća pauza na granicama unutar [i, j)
        for j in range(i + 1, n + 1):
            if pauses is not None and j - 2 >= i:
                inner_pause = max(inner_pause, pauses[j - 2])
            if j - i > 1 and span.length(i, j) > max_search:
                break
            if j < n and span.break_pen[j - 1] >= INF:
                continue  # ne smemo prelomiti ovde
            lines_cost, cuts = _best_lines(span, i, j)
            if lines_cost >= INF:
                continue

            dur = words[j - 1].e - words[i].s
            chars = span.length(i, j)
            cost = lines_cost + PENALTY_PER_CUE
            if dur > 0:
                cps = chars / dur
                if cps > profile.max_cps:
                    cost += (cps - profile.max_cps) * PENALTY_OVER_MAX_CPS
            if dur > profile.max_duration:
                cost += (dur - profile.max_duration) * PENALTY_OVER_MAX_DURATION
            if (
                not profile.min_chars_per_line
                and not (i == 0 and j == n)
                and chars < MIN_CUE_CHARS
            ):
                cost += (MIN_CUE_CHARS - chars) * PENALTY_SHORT_CUE
            cost += pause_in_cue_penalty(inner_pause, span.pp)
            if j < n:
                cost += span.break_pen[j - 1]
            edges[i].append((j, cost, cuts))

    if profile.max_lines == 1 and PENALTY_CUE_SHARE_DEVIATION > 0:
        choice = _layout_even(span, edges, PENALTY_CUE_SHARE_DEVIATION)
    else:
        best: list[float] = [INF] * (n + 1)
        choice = [(0, [])] * (n + 1)
        best[n] = 0.0
        for i in range(n - 1, -1, -1):
            for j, cost, cuts in edges[i]:
                total = cost + best[j]
                # `<=` namerno: pri izjednačenju biramo veće j, tj. manje cue-ova
                if total <= best[i]:
                    best[i] = total
                    choice[i] = (j, cuts)

    cues: list[Cue] = []
    i = 0
    while i < n:
        j, cuts = choice[i]
        if j <= i:  # ne bi smelo, ali ne petljaj beskonačno
            j, cuts = n, []
        bounds = [i] + [c + 1 for c in cuts] + [j]
        lines = [
            join_words(words[a:b]) for a, b in zip(bounds, bounds[1:]) if b > a
        ]
        cues.append(
            Cue(
                s=round(words[i].s, 3),
                e=round(words[j - 1].e, 3),
                lines=[l for l in lines if l],
                sentence_id=sentence_id,
                flags=list(flags or []),
            )
        )
        i = j
    return cues


# ==========================================================================
# Vremenska ograničenja profila (primenjuju se na ceo fajl, ne po rečenici)
# ==========================================================================


#: Najkraći cue koji `apply_timing` ostavlja kad mesta nema (prijavljuje se
#: kao prekratak, `too_short`).
MIN_CUE_SECONDS = 0.05

#: Vremena se zaokružuju na ms; razlika ispod pola ms nije preklapanje.
ROUNDING_SLACK = 0.0005


def apply_timing(cues: list[Cue], profile: Profile) -> list[Cue]:
    """Produži prekratke cue-ove i obezbedi `min_gap` između susednih."""
    n = len(cues)
    for idx, cue in enumerate(cues):
        if cue.e - cue.s >= profile.min_duration:
            continue
        need = profile.min_duration
        # 1) produži unapred, na račun pauze koja sledi
        limit = cues[idx + 1].s - profile.min_gap if idx + 1 < n else cue.s + need
        cue.e = round(min(cue.s + need, max(cue.e, limit)), 3)
        # 2) ako i dalje nije dovoljno, uzmi unazad
        if cue.e - cue.s < profile.min_duration:
            floor = cues[idx - 1].e + profile.min_gap if idx > 0 else 0.0
            cue.s = round(max(floor, cue.e - need), 3)

    # Razmak, pa redosled bez preklapanja. Reči nultog trajanja (izmišljen tekst,
    # `hallucination`) daju više cue-ova na ISTOM trenutku: tu nema mesta ni
    # napred ni nazad, korak 2 gore ume da stavi početak iza kraja, a samo
    # skraćivanje prethodnog nije dovoljno. Bez guranja početka unapred
    # izlazila su dva cue-a sa istim vremenima preko trećeg (krug 23,
    # Video-4 draft 2: #17 i #18 `44,943–44,993`, #19 od 44,86).
    for idx in range(1, n):
        prev, cur = cues[idx - 1], cues[idx]
        if cur.s - prev.e < profile.min_gap:
            prev.e = round(max(prev.s + MIN_CUE_SECONDS, cur.s - profile.min_gap), 3)
        if cur.s < prev.e + profile.min_gap - ROUNDING_SLACK:
            cur.s = round(prev.e + profile.min_gap, 3)
        if cur.e < cur.s + MIN_CUE_SECONDS:
            cur.e = round(cur.s + MIN_CUE_SECONDS, 3)

    for idx, cue in enumerate(cues, start=1):
        cue.index = idx
        if cue.e <= cue.s:
            cue.e = round(cue.s + MIN_CUE_SECONDS, 3)
    return cues


# ==========================================================================
# Vreme natpisa po tišini (krug 24)
# ==========================================================================
#
# Whisper tišinu upija u reč (krug 23: 96 od 100 tišina ≥ 0,30 s leži u
# trajanju reči), pa natpis koji počinje ili se završava takvom reči stoji na
# ekranu nad tišinom. Prelom to ne može da popravi — rez je između reči. Ovde
# se pomera VREME: početak natpisa na kraj tišine koja ga otvara, kraj na
# početak tišine koja ga zatvara. Svaka ivica se pomera cela ili nikako.

#: Najkraća tišina koja pomera ivicu natpisa.
SNAP_SILENCE_MIN = 0.30

#: Najkraći prazan ekran koji pomeranje sme da napravi (kraće treperi), ili
#: None — mehanizam isključen. Mereno u krugu 24 (dnevnik, 2026-10-14):
#: ograničenja drži (CPS isti, preklapanja 0), natpisa sa pauzom `tv` 41 -> 36,
#: `web` 53 -> 42; mrtvo vreme raste tačno koliko teksta nad tišinom nestane
#: (isti broj s druge strane, ne cena). Uključeno u krugu 25, odlukom.
SNAP_MIN_BLANK: float | None = 0.40

#: Rezerva kojom natpis ulazi PRE kraja tišine po VAD-u. Kašnjenje natpisa za
#: govorom se oseti odmah (reč se čuje pre nego što se vidi), rani ulazak
#: retko; ako je VAD tihi početak reči proglasio tišinom, rezerva ga pokrije.
#: Krug 25: 0 / 0,08 / 0,15 uklanjaju isto natpisa sa pauzom (`tv` 29, `web`
#: 42 / 41 / 41 — dnevnik), pa je uzeta najveća. Izbor po slušanju je otvoren (tri SRT-a za preslušavanje).
SNAP_LEAD = 0.15

#: Tišina „otvara" natpis ako počinje najviše ovoliko posle njegovog početka
#: (i simetrično za kraj) — reč pre tišine ume da ima par desetinki zvuka.
SNAP_EDGE_SLACK = 0.10


@dataclass
class SnapReport:
    """Šta je `snap_to_silences` uradio — za dnevnik posla i merenje."""

    starts: list[tuple[int, float]] = field(default_factory=list)  # (cue, pomeraj s)
    ends: list[tuple[int, float]] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=lambda: {"trajanje": 0, "cps": 0, "praznina": 0})


def snap_to_silences(
    cues: list[Cue], silences: list[tuple[float, float]], profile: Profile,
    *, min_blank: float, min_silence: float = SNAP_SILENCE_MIN, lead: float | None = None,
) -> SnapReport:
    """Pomeri ivice natpisa na ivice tišine (posle `apply_timing`).

    Ivica se ne pomera ako bi natpis pao ispod `min_duration`, prešao
    `max_cps`, ili ako bi prazan ekran do susednog natpisa bio kraći od
    `min_blank` (treperenje je gore od teksta nad tišinom). Preklapanje ne
    može da nastane: početak ide samo napred, kraj samo nazad.
    """
    rep = SnapReport()
    lead = SNAP_LEAD if lead is None else lead
    chars = [sum(len(l) for l in c.lines) for c in cues]

    def why_not(k: int, s: float, e: float, edge: str) -> str | None:
        dur = e - s
        if dur < profile.min_duration - 1e-9:
            return "trajanje"
        if chars[k] / dur > profile.max_cps + 1e-9:
            return "cps"
        # samo praznina koju pravi OVO pomeranje; postojeći razmak sa druge
        # strane (`min_gap` unutar rečenice) nije njegov
        if edge == "s" and k > 0 and s - cues[k - 1].e < min_blank - 1e-9:
            return "praznina"
        if edge == "e" and k + 1 < len(cues) and cues[k + 1].s - e < min_blank - 1e-9:
            return "praznina"
        return None

    for k, c in enumerate(cues):
        for a, b in silences:
            if b - a < min_silence:
                continue
            # tišina otvara natpis: počinje pre (ili tik posle) početka, traje
            # bar prag u natpisu, a posle nje u natpisu ima govora
            if a <= c.s + SNAP_EDGE_SLACK and b - c.s >= min_silence and b < c.e:
                new_s = round(max(c.s, b - lead), 3)  # rezerva pred govorom
                reason = why_not(k, new_s, c.e, "s")
                if reason:
                    rep.rejected[reason] += 1
                else:
                    rep.starts.append((k, round(new_s - c.s, 3)))
                    c.s = new_s
            # tišina zatvara natpis
            elif b >= c.e - SNAP_EDGE_SLACK and c.e - a >= min_silence and a > c.s:
                reason = why_not(k, c.s, round(a, 3), "e")
                if reason:
                    rep.rejected[reason] += 1
                else:
                    rep.ends.append((k, round(c.e - a, 3)))
                    c.e = round(a, 3)
    return rep


def too_short(cues: list[Cue], profile: Profile) -> list[Cue]:
    """Cue-ovi koji su i posle `apply_timing` kraći od `min_duration` — oko
    njih nema mesta ni napred ni nazad. Ne smeju proći tiho: prijavljuju se u
    dnevniku i na stranici posla."""
    return [c for c in cues if c.e - c.s < profile.min_duration - 1e-9]


# ==========================================================================
# Poravnanje ručno izmenjenog teksta sa originalnim vremenima
# ==========================================================================


def align_text_to_words(text: str, orig: list[Word]) -> tuple[list[Word], float]:
    """Tekst rečenice (možda ručno ispravljen) -> tokeni sa vremenima.

    Neizmenjeni tokeni zadržavaju svoja prava ASR vremena. Samo za tokene koje
    je čovek dodao ili promenio vreme se interpolira unutar odgovarajućeg
    intervala — tu drugih podataka nema.

    Vraća i **udeo tokena koji su zadržali prava ASR vremena**. Kad taj udeo
    padne ispod `MIN_ALIGNED_TOKEN_SHARE`, vremena su uglavnom izmišljena i
    pozivalac to mora da prijavi, a ne da tiho isporuči loš titl.
    """
    new_toks = tokenize_text(text)
    orig_toks = [w.w for w in orig]
    if new_toks == orig_toks or not orig:
        return orig, 1.0
    if not new_toks:
        return orig, 1.0

    sm = difflib.SequenceMatcher(
        a=[t.lower() for t in orig_toks], b=[t.lower() for t in new_toks], autojunk=False
    )
    out: list[Word | None] = [None] * len(new_toks)
    base_i = orig[0].i

    def fill(j1: int, j2: int, t0: float, t1: float) -> None:
        toks = new_toks[j1:j2]
        total = sum(len(t) for t in toks) or 1
        span = max(t1 - t0, 0.0)
        pos = t0
        for k, tok in enumerate(toks):
            share = span * len(tok) / total
            out[j1 + k] = Word(
                i=base_i + j1 + k, s=round(pos, 3), e=round(pos + share, 3), w=tok, p=1.0
            )
            pos += share

    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # poklapanje je bez obzira na velika slova — tekst uzimamo novi,
            # vremena stara
            matched += j2 - j1
            for k in range(j2 - j1):
                out[j1 + k] = orig[i1 + k].model_copy(
                    update={"w": new_toks[j1 + k]}
                )
        elif j1 < j2:
            if i1 < i2:
                t0, t1 = orig[i1].s, orig[i2 - 1].e
            else:  # čisto umetanje: uzmi pauzu između suseda
                t0 = orig[i1 - 1].e if i1 > 0 else orig[0].s
                t1 = orig[i1].s if i1 < len(orig) else orig[-1].e
            fill(j1, j2, t0, max(t1, t0))

    result = [w for w in out if w is not None]
    # stabilni indeksi unutar rečenice (prava vremena su već sačuvana)
    for k, w in enumerate(result):
        result[k] = w.model_copy(update={"i": base_i + k})
    return result, matched / len(new_toks)


def synthesize_words(sentence: Sentence) -> list[Word]:
    """Vremena po rečima za rečenicu koja ih nema — PREVOD.

    Za izvorni jezik ovo se nikad ne koristi: tamo postoje prava ASR vremena i
    ona se koriste. Prevod ih po prirodi stvari nema — zna se samo početak i
    kraj rečenice — pa se raspoređuje srazmerno broju znakova. To je jedina
    dostupna informacija, i zato je odvojeno u zasebnu funkciju, da se nikad ne
    pomeša sa pravim vremenima.
    """
    toks = tokenize_text(sentence.text)
    if not toks:
        return []
    total = sum(len(t) for t in toks) or 1
    span = max(sentence.e - sentence.s, 0.0)
    out: list[Word] = []
    pos = sentence.s
    for k, tok in enumerate(toks):
        share = span * len(tok) / total
        out.append(
            Word(i=k, s=round(pos, 3), e=round(pos + share, 3), w=tok, p=1.0)
        )
        pos += share
    out[-1] = out[-1].model_copy(update={"e": round(sentence.e, 3)})
    return out


def build_cues(
    sentences: list[Sentence],
    words: list[Word],
    profile: Profile,
    *,
    language: str,
    log: Callable[[str], None] | None = None,
    interpolate_times: bool = False,
    glossary_terms: list[str] | None = None,
    silences: list[tuple[float, float]] | None = None,
) -> list[Cue]:
    """Ceo posao: rečenice + reči -> gotovi cue-ovi sa primenjenim ograničenjima.

    `language` bira tabelu funkcijskih reči. Bez toga bi se na engleskom i
    francuskom primenjivale srpske liste, pa bi prelom padao iza `the`.

    `silences` (tišine iz zvuka, `pauses.py`) važe samo uz prava vremena reči;
    uz `interpolate_times` se ne koriste — izmišljena vremena nemaju pauze.

    Rečenice čije poravnanje degradira dobijaju flag `alignment_degraded`
    (i na rečenici i na cue-ovima) i upozorenje kroz `log`.
    """
    warn = log or (lambda msg: cfg.echo(f"UPOZORENJE: {msg}"))
    rules = rules_for(language, warn)
    terms = term_tokens(glossary_terms or [])
    by_index = {w.i: w for w in words}
    cues: list[Cue] = []
    for sent in sentences:
        if interpolate_times:
            # Prevod: pravih vremena po rečima nema, pa se raspoređuje srazmerno.
            # Prav `"` postaje usmeren PRE tokenizacije — inače ga prelom odvoji
            # od citata (`typography.QUOTE_PAIRS`).
            sw = synthesize_words(sent.model_copy(update={
                "text": typography.directional_quotes(sent.text, language)
            }))
            if sw:
                cues.extend(layout_sentence(
                    sw, profile, sent.id, list(sent.flags), rules, terms
                ))
            continue

        lo, hi = sent.words
        sw = [by_index[k] for k in range(lo, hi + 1) if k in by_index]
        if not sw:
            continue
        # Tekst rečenice je merodavan: ako ga je čovek menjao, poštuj izmenu.
        sw, share = align_text_to_words(sent.text, sw)
        flags = list(sent.flags)
        if share < MIN_ALIGNED_TOKEN_SHARE:
            if ALIGNMENT_DEGRADED_FLAG not in flags:
                flags.append(ALIGNMENT_DEGRADED_FLAG)
            if ALIGNMENT_DEGRADED_FLAG not in sent.flags:
                sent.flags.append(ALIGNMENT_DEGRADED_FLAG)
            warn(
                f"rečenica {sent.id}: poravnanje ručne izmene degradirano — "
                f"samo {share:.0%} tokena je zadržalo prava ASR vremena "
                f"(prag {MIN_ALIGNED_TOKEN_SHARE:.0%}). Vremena u ovoj rečenici "
                f"su uglavnom interpolirana. Tekst: {sent.text[:80]!r}"
            )
        cues.extend(layout_sentence(sw, profile, sent.id, flags, rules, terms, silences))
    return apply_timing(cues, profile)


# ==========================================================================
# Deljenje rečenice na ZADAT broj delova (prevod postojećeg SRT-a)
# ==========================================================================

#: Po znaku odstupanja dela od ciljne dužine. Jedina nova kazna u ovom režimu:
#: sve ostalo (funkcijske reči, zarez, veznik) je `_Span.break_pen`, isto kao
#: pri prelamanju. Sa 4 po znaku rez se pomera ~6 znakova da bi pao na zarez
#: (dobit 50, trošak 2 dela × 6 × 4), a ~35 da ne bi razdvojio predlog od
#: imenice (300) — što se desi samo kad bliže nema nijednog pristojnog reza.
PENALTY_SHARE_DEVIATION = 4


def _layout_even(
    span: _Span, edges: list[list[tuple[int, float, list[int]]]], weight: float,
) -> list[tuple[int, list[int]]]:
    """Jednoredni profil: cue-ovi iste rečenice što ravnomerniji MEĐU SOBOM.

    Cilj nije zadat spolja: za svaki broj delova k meri se odstupanje od
    prosečne dužine tih k delova, pa se bira najjeftiniji k. Broj delova i
    dalje određuje `PENALTY_PER_CUE` (u troškovima `edges`), ne udeo.
    Prosek je `(dužina - (k - 1)) / k` — razmak na mestu reza ne ulazi ni u
    jedan deo.

    Pretraga po k staje kad sam trošak cue-ova (`k * PENALTY_PER_CUE`, a sve
    ostale kazne su nenegativne) premaši najbolje nađeno — rezultat je tačan.
    """
    n = span.n
    total_len = span.length(0, n)
    best_total, best_choice = INF, None
    for k in range(1, n + 1):
        if k * PENALTY_PER_CUE > best_total:
            break
        mean = (total_len - (k - 1)) / k
        # g[r][i]: najmanji trošak da tačno r cue-ova pokrije [i, n)
        g = [[INF] * (n + 1) for _ in range(k + 1)]
        back: list[list[tuple[int, list[int]]]] = [[(0, [])] * (n + 1) for _ in range(k + 1)]
        g[0][n] = 0.0
        for r in range(1, k + 1):
            for i in range(n - 1, -1, -1):
                for j, cost, cuts in edges[i]:
                    rest = g[r - 1][j]
                    if rest >= INF:
                        continue
                    total = cost + share_deviation(span.length(i, j), mean, weight) + rest
                    if total <= g[r][i]:
                        g[r][i] = total
                        back[r][i] = (j, cuts)
        if g[k][0] < best_total:
            best_total = g[k][0]
            choice: list[tuple[int, list[int]]] = [(0, [])] * (n + 1)
            i, r = 0, k
            while r > 0:
                choice[i] = back[r][i]
                i, r = back[r][i][0], r - 1
            best_choice = choice
    return best_choice or [(n, [])] * (n + 1)


def share_deviation(length: int, target: float, weight: float) -> float:
    """Kazna za deo rečenice čija dužina odstupa od ciljnog udela.

    Zajednička za `split_by_shares` (udeli zadati cue-ovima originala) i za
    jednoredni profil u `layout_sentence` (udeli ravnomerni).
    """
    return abs(length - target) * weight


def split_by_shares(
    words: list[Word], shares: list[float], profile: Profile,
    rules: LanguageRules = GENERIC_RULES, terms: list[list[str]] | None = None,
) -> list[tuple[int, int]]:
    """Podeli tokene na `len(shares)` uzastopnih delova, srazmerno `shares`.

    Za razliku od `layout_sentence`, broj delova i njihove ciljne dužine su
    zadati spolja (cue-ovi originala); bira se samo GDE se seče. Svaki rez se
    kažnjava istim `_Span.break_pen` kao prelom cue-a, a svaki deo odstupanjem
    od ciljne dužine. Vraća `[(i, j), ...]` — polu-otvorene raspone tokena.

    Kad tokena ima manje nego delova, neki delovi ostaju prazni `(i, i)`;
    pozivalac to prijavljuje, ne popravlja.
    """
    k = len(shares)
    n = len(words)
    if k == 0:
        return []
    if k == 1:
        return [(0, n)]
    total_share = sum(shares) or 1.0
    span = _Span(words, profile, rules, terms)
    total_len = span.length(0, n)
    targets = [total_len * s / total_share for s in shares]

    if n < k:
        return _split_proportional(n, shares)

    # dp[j][p]: najmanji trošak da prvih j delova pokrije tokene [0, p)
    dp = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[-1] * (n + 1) for _ in range(k + 1)]
    dp[0][0] = 0.0
    for j in range(1, k + 1):
        last = j == k
        # svaki deo ima bar jedan token, pa j-ti deo ne može da se završi pre j
        for p in ([n] if last else range(j, n - (k - j) + 1)):
            cut_pen = 0.0 if last else span.break_pen[p - 1]
            if cut_pen >= INF:
                continue
            best, arg = INF, -1
            for q in range(j - 1, p):
                prev = dp[j - 1][q]
                if prev >= INF:
                    continue
                cost = prev + share_deviation(
                    span.length(q, p), targets[j - 1], PENALTY_SHARE_DEVIATION
                )
                if cost < best:
                    best, arg = cost, q
            if arg >= 0:
                dp[j][p] = best + cut_pen
                back[j][p] = arg

    if dp[k][n] >= INF:
        # Nema dovoljno dozvoljenih mesta za rez (npr. niz interpunkcije).
        return _split_proportional(n, shares)

    bounds = [n]
    p = n
    for j in range(k, 0, -1):
        p = back[j][p]
        bounds.append(p)
    bounds.reverse()
    return list(zip(bounds, bounds[1:]))


def _split_proportional(n: int, shares: list[float]) -> list[tuple[int, int]]:
    """Rezervni put: tokeni po kumulativnom udelu, bez ikakvog skorovanja."""
    total = sum(shares) or 1.0
    out: list[tuple[int, int]] = []
    acc = 0.0
    start = 0
    for idx, s in enumerate(shares):
        acc += s
        end = n if idx == len(shares) - 1 else min(n, round(n * acc / total))
        end = max(end, start)
        out.append((start, end))
        start = end
    return out


def best_lines(
    words: list[Word], profile: Profile, rules: LanguageRules = GENERIC_RULES,
    terms: list[list[str]] | None = None,
) -> list[tuple[int, int]]:
    """Redovi jednog već zadatog cue-a — isti `_best_lines` kao pri prelamanju.

    Vraća raspone nad `words` kakvi su prosleđeni; prazni tokeni nisu dozvoljeni.
    """
    if not words:
        return []
    span = _Span(words, profile, rules, terms)
    _, cuts = _best_lines(span, 0, span.n)
    bounds = [0] + [c + 1 for c in cuts] + [span.n]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
