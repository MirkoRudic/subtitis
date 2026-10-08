"""Ijekavica -> ekavica, kurirani rečnik oblika.

Whisper za srpski povremeno provuče ijekavicu i uz `language="sr"` — `posjeduju`,
`ovdje`, `vrijeme`. Ovo se ispravlja nad `words.json` PRE segmentacije, da
`sentences.json` i sve izvedeno iz njega budu čisti.

NEMA pravila „ije/je -> e". Takvo pravilo uništava `jesen`, `jedan`, `jezik`,
`jeftino`, `prijem`, `sjajno`, `Jelena`. Zato ovde stoji kurirani rečnik osnova,
u tri sloja, po istoj disciplini kao izuzeci u `translit.py`:

* `PROTECTED_*`  — reči koje se nikad ne diraju; proverava se prvo
* `EXACT`        — cela reč, tačno poklapanje
* `PREFIX`       — reč POČINJE osnovom (padežni i glagolski nastavci variraju)
* `ANY`          — osnova bilo gde u reči (hvata i prefiksirane glagole:
                   `predvidjeti`, `uvidjeti`, `posjeduje`)

U `ANY` idu samo osnove koje su proverene i kao podniz druge reči. Kad osnova
nije bezbedna kao podniz, ide u `PREFIX` ili `EXACT`. Primeri zbog kojih je ta
podela nužna:

    prijem      — zato je `prije` samo u EXACT, ne PREFIX (inače `prijem`->`prem`)
    odjeci      — zato je `djec` u PREFIX, ne ANY (inače `odjeci`->`odeci`)
    želje       — zato su `željet`/`željel`, ne `želje` (inače imenica `želje`->`žele`)
    dijeta      — zato je `dijete` u EXACT, ne PREFIX `dijet` (inače `dijeta`->`deta`)
    umio se     — zato `razumio` stoji ceo, a gole osnove `umio` nema
    Mjanmar     — strano ime; `mj` tu nije ijekavizam
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Word

# ==========================================================================
# Zaštita — proverava se PRE svih zamena
# ==========================================================================

#: Cele reči koje se nikad ne diraju.
#: Imenica `želja` mora biti ovde kao TAČNI oblici, ne kao prefiks — prefiks
#: `želje` bi blokirao i glagol `željeti`/`željela`, koji se menja.
PROTECTED_WORDS: frozenset[str] = frozenset({
    "želja", "želje", "želji", "želju", "željom", "željama",
})

#: Reči koje počinju ovim se nikad ne diraju. Ovde su i oblici koje nijedno
#: pravilo trenutno ne pogađa — stoje kao brana za buduće dodavanje osnova.
PROTECTED_PREFIXES: tuple[str, ...] = (
    "jesen",      # jesen, jesenji
    "jedan", "jedn",  # jedan, jedna, jedno, jednom
    "jezi",       # jezik, jezika, jeziku, jezički
    "jeftin",     # jeftino, jeftiniji
    "prijem",     # prijem, prijema, prijemni
    "prijatelj",  # prijatelj, prijateljstvo
    "sjaj",       # sjaj, sjajno, sjajan
    "jelen",      # jelen; pokriva i ime Jelena
    # Strano ime gde `mj` nije ijekavizam. Nijedno današnje pravilo ga ne
    # pogađa (`mjen` traži `mje`, ovde je `mja`) — stoji kao brana za slučaj da
    # neko doda širu osnovu tipa `mje`.
    "mjanmar",
    "odjek", "odjec",  # odjek, odjeci
    "dijeta", "dijetn",  # dijeta (ishrana), dijetni — NE `dijet`, to bi
                         # blokiralo i `dijete` iz EXACT
    "umio", "umiv",  # umio se, umivati (`razumio` ne počinje sa `umio`)
)

# ==========================================================================
# Rečnik zamena
# ==========================================================================

#: Cela reč, tačno poklapanje.
EXACT: dict[str, str] = {
    "prije": "pre",        # PREFIX bi pokvario `prijem`
    "dijete": "dete",      # PREFIX `dijet` bi pokvario `dijeta`
    "razumio": "razumeo",  # gola osnova `umio` bi pokvarila `umio se`
    "razumjela": "razumela",
    "razumjeli": "razumeli",
    "htio": "hteo",
    "želio": "želeo",
    "vidio": "video",
    "živio": "živeo",
    "dvije": "dve",
    "obje": "obe",
}

#: Reč POČINJE ovom osnovom.
PREFIX: dict[str, str] = {
    "djec": "dec",          # djeca, djece, djeci, djecu  (ANY bi pokvario `odjeci`)
    "djetet": "detet",      # djeteta, djetetu
    "djetinj": "detinj",    # djetinjstvo, djetinji
    "prijedlog": "predlog",
    "prijedlož": "predlož",
    "prijevoz": "prevoz",
    "prijelaz": "prelaz",
    "prijenos": "prenos",
    "prijetnj": "pretnj",
    "prijeti": "preti",
}

#: Osnova bilo gde u reči. Svaka je proverena i kao podniz.
ANY: dict[str, str] = {
    # --- mesni i vremenski prilozi ---
    "ovdje": "ovde",
    "ondje": "onde",
    "gdje": "gde",          # gdje, negdje, nigdje, svugdje, igdje, ponegdje
    "poslije": "posle",
    "uvijek": "uvek",
    "naprijed": "napred",

    # --- vreme, dani ---
    "vrijem": "vrem",       # vrijeme
    "srijed": "sred",       # srijeda, srijedom
    "nedjelj": "nedelj",    # nedjelja, nedjeljom
    "mjesec": "mesec",      # mjesec, mjeseca
    "mjeseč": "meseč",      # mjesečni, mjesečno (palatalizacija c -> č)

    # --- mesto, mera ---
    "mjest": "mest",        # mjesto, mjesta, umjesto, mjesni
    "mjer": "mer",          # mjera, mjeriti, namjera, primjer, odmjeren
    "mjen": "men",          # promjena, izmjena, mjenjati, mjenjačnica

    # --- vera, nada ---
    "vjer": "ver",          # vjera, vjeruje, vjerovatno, povjerenje, nevjerica

    # --- svet, svetlost ---
    "svijet": "svet",       # svijet, svijeta, svijetu
    "svjet": "svet",        # svjetski, svjetlo, svjetlost, osvjetljenje

    # --- telo ---
    "tijel": "tel",         # tijelo, tijela
    "tjeles": "teles",      # tjelesni

    # --- lep, beo, cvet ---
    "lijep": "lep",         # lijep, lijepa, lijepo
    "ljepš": "lepš",        # ljepši, ljepša
    "ljepot": "lepot",      # ljepota
    "bijel": "bel",         # bijel, bijela, bijelo
    "bjel": "bel",          # bjelina, bjeloglavi
    "cvijet": "cvet",       # cvijet, cvijeta
    "cvjet": "cvet",        # cvjetni, cvjetovi

    # --- priroda ---
    "rijek": "rek",         # rijeka, rijeke, rijeku
    "riječ": "reč",         # riječ, riječi, riječima
    "snijeg": "sneg",       # snijeg, snijega
    "snjeg": "sneg",        # snjegovi
    "snjež": "snež",        # snježni, snježna
    "mlijek": "mlek",       # mlijeko, mlijeka
    "mliječ": "mleč",       # mliječni
    "zvijezd": "zvezd",     # zvijezda
    "korijen": "koren",
    "sjever": "sever",

    # --- seme, sedenje, sećanje, posed ---
    "sjeme": "seme",        # sjeme, sjemena, sjemenka
    "sjed": "sed",          # sjediti, sjednica, sjedište, posjed, posjeduje
    "sjet": "set",          # sjetiti, posjeta, osjetiti
    "sjeć": "seć",          # sjećanje, osjećaj, osjeća

    # --- glagoli ---
    "htje": "hte",          # htjeti, htjela, htjeli
    "željet": "želet",      # željeti  (`želje` je imenica, ne diraj)
    "željel": "želel",      # željela, željeli
    "vidje": "vide",        # vidjeti, vidjela, predvidjeti, uvidjeti
    "živje": "žive",        # živjeti, živjela, poživjeti
    "umje": "ume",          # umjeti, razumjeti, umjetnost, umjetnik
    "smje": "sme",          # smjeti, smjer, smješta, namjeravati
    "tjera": "tera",        # tjerati, otjerati
    "bjež": "bež",          # bježati, izbjeći
    "bjeg": "beg",          # bjegunac
    "ovjer": "over",        # ovjeriti, ovjera

    # --- čovek, deca, delo ---
    "čovje": "čove",        # čovjek, čovjeka
    "čovječ": "čoveč",      # čovječanstvo, čovječe
    "djev": "dev",          # djevojka, djevojčica
    "djed": "ded",          # djed, djeda
    "djel": "del",          # djelo, djela, djelovati, udjel, odjel
    "dijel": "del",         # dijeliti, dijelovi, podijeliti

    # --- pesma ---
    "pjesm": "pesm",        # pjesma, pjesme
    "pjev": "pev",          # pjevati, pjevač, pjeva
    "pješ": "peš",          # pješak, pješice
    "pjen": "pen",          # pjena

    # --- ostalo visoke frekvencije ---
    "cijel": "cel",         # cijeli, cijela, cijelo
    "cjel": "cel",          # cjelina, cjelokupan
    "vrijed": "vred",       # vrijedan, vrijednost, vrijedi
    "liječ": "leč",         # liječiti, liječenje
    "lječ": "leč",          # lječilište
    "dio": "deo",           # dio  -> deo   (v. napomenu ispod)
}

#: `dio` -> `deo` je jedini unos koji nije bezbedan kao podniz (`radio`,
#: `studio`, `video`), pa se primenjuje samo kao cela reč.
ANY_WHOLE_WORD_ONLY: frozenset[str] = frozenset({"dio"})


def _check_tables() -> None:
    """Sanity provera pri uvozu: nijedna osnova ne sme biti i zaštićena."""
    for table in (EXACT, PREFIX, ANY):
        for stem in table:
            if stem in PROTECTED_WORDS or stem.startswith(PROTECTED_PREFIXES):
                raise ValueError(
                    f"Osnova {stem!r} je istovremeno u rečniku i zaštićena."
                )


_check_tables()

# Duže osnove prvo — deterministički i bez delimičnih preklapanja.
_ANY_SORTED: tuple[tuple[str, str], ...] = tuple(
    sorted(ANY.items(), key=lambda kv: (-len(kv[0]), kv[0]))
)
_PREFIX_SORTED: tuple[tuple[str, str], ...] = tuple(
    sorted(PREFIX.items(), key=lambda kv: (-len(kv[0]), kv[0]))
)


# ==========================================================================
# Konverzija
# ==========================================================================


@dataclass(frozen=True)
class Change:
    """Jedna izmena — za log."""

    index: int  # stabilan indeks reči u words.json
    start: float  # vreme, sekunde
    before: str
    after: str

    def __str__(self) -> str:
        return f"[{self.index}] {self.start:.2f}s  {self.before!r} -> {self.after!r}"


def is_protected(word_lower: str) -> bool:
    return word_lower in PROTECTED_WORDS or word_lower.startswith(PROTECTED_PREFIXES)


def _match_case(original: str, converted: str) -> str:
    """`Ovdje` -> `Ovde`, `OVDJE` -> `OVDE`, `ovdje` -> `ovde`."""
    if len(original) > 1 and original.isupper():
        return converted.upper()
    if original[:1].isupper():
        return converted[:1].upper() + converted[1:]
    return converted


def convert_word(word: str) -> str:
    """Vrati ekavski oblik reči; nepromenjenu reč ako nema pravila."""
    if not word or not any(ch.isalpha() for ch in word):
        return word

    low = word.lower()
    if is_protected(low):
        return word

    if low in EXACT:
        return _match_case(word, EXACT[low])

    for stem, repl in _PREFIX_SORTED:
        if low.startswith(stem):
            return _match_case(word, repl + low[len(stem) :])

    out = low
    for stem, repl in _ANY_SORTED:
        if stem not in out:
            continue
        if stem in ANY_WHOLE_WORD_ONLY:
            if out == stem:
                out = repl
            continue
        out = out.replace(stem, repl)

    if out == low:
        return word
    return _match_case(word, out)


def convert_text(text: str) -> str:
    """Konvertuj slobodan tekst. Za `words.json` koristi `convert_words`."""
    return " ".join(convert_word(tok) for tok in text.split(" "))


def convert_words(words: list[Word]) -> list[Change]:
    """Konvertuj listu reči NA MESTU. Vrati spisak izmena za log."""
    changes: list[Change] = []
    for w in words:
        new = convert_word(w.w)
        if new != w.w:
            changes.append(Change(index=w.i, start=w.s, before=w.w, after=new))
            w.w = new
    return changes
