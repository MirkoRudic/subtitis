"""Transliteracija srpskog: ćirilica <-> latinica.

Ćirilica -> latinica je trivijalno 1:1.

Latinica -> ćirilica je zamka: `lj`, `nj` i `dž` nisu uvek digrafi. Kad padnu
preko granice morfema, to su dva zasebna slova:

    nadživeti   -> надживети   (nad + živeti),  ne наџивети
    podžanr     -> поджанр     (pod + žanr)
    injekcija   -> инјекција
    konjugacija -> конјугација
    tanjug      -> Танјуг

Zato ovde postoji eksplicitna, u kodu verzionisana lista izuzetaka. Biblioteka
`cyrtranslit` nema ništa od ovoga i pravi tačno ove greške — ne koristiti je.

Kontraprimeri koji pokazuju zašto lista MORA biti eksplicitna (ovde je digraf
pravi i ne sme se razdvojiti): odžak (оџак), konjak (коњак), kanjon (кањон),
inje (иње), hodža (хоџа), narandža (наранџа), nadljudski (надљудски).
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# --------------------------------------------------------------------------
# Osnovne tabele
# --------------------------------------------------------------------------

DIGRAPHS: dict[str, str] = {"lj": "љ", "nj": "њ", "dž": "џ"}

_LAT_SINGLE: dict[str, str] = {
    "a": "а", "b": "б", "c": "ц", "č": "ч", "ć": "ћ", "d": "д", "đ": "ђ",
    "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "ј", "k": "к",
    "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "r": "р", "s": "с",
    "š": "ш", "t": "т", "u": "у", "v": "в", "z": "з", "ž": "ж",
}

_CYR_SINGLE: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e",
    "ж": "ž", "з": "z", "и": "i", "ј": "j", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "ћ": "ć",
    "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č", "ш": "š",
}
_CYR_DIGRAPH: dict[str, str] = {"љ": "lj", "њ": "nj", "џ": "dž"}

SRB_LATIN_LETTERS = set(_LAT_SINGLE) | {c.upper() for c in _LAT_SINGLE}
SRB_CYR_LETTERS = (
    set(_CYR_SINGLE)
    | {c.upper() for c in _CYR_SINGLE}
    | set(_CYR_DIGRAPH)
    | {c.upper() for c in _CYR_DIGRAPH}
)

# Reč = niz slova (naša + ASCII, zbog stranih imena) plus apostrof unutar reči.
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# --------------------------------------------------------------------------
# Izuzeci: mesta gde se digraf MORA razdvojiti na dva ćirilična slova
# --------------------------------------------------------------------------

# Osnove koje se traže bilo gde u reči. Svaka mora sadržati tačno jedan digraf;
# razdvaja se PRVI digraf u osnovi. Dodaj samo osnove koje su bezbedne i kao
# podniz druge reči (npr. `anjon` NIJE bezbedan zbog `kanjon` — vidi listu ispod).
SPLIT_STEMS_ANY: frozenset[str] = frozenset({
    # inj- internacionalizmi
    "injek",      # injekcija, injektor, reinjekcija
    "injici",     # injicirati
    "injunk",     # injunkcija
    # konj- internacionalizmi (konjak/konjic ostaju sa њ jer ih ne dodirujemo)
    "konjug",     # konjugacija, konjugovati
    "konjunk",    # konjunkcija, konjunktiv, konjunktura, konjunktivitis
    "konjekt",    # konjektura
    # skraćenice / imena
    "tanjug",     # TANJUG -> Танјуг
    # van- složenice (generalno pravilo za `van-` NE postoji zbog imena Vanja)
    "vanjezič",   # vanjezički
    "vanjezik",
    # dž preko granice morfeme, izvan prefiksnog pravila
    "nadživ",     # nadživeti, nadživljavanje
    "podžanr",
    "nadžanr",
    "podžupan",
    "odžvak",     # odžvakati
})

# Osnove koje se traže SAMO na početku reči — nisu bezbedne kao podniz.
SPLIT_STEMS_START: frozenset[str] = frozenset({
    "anjon",  # анјон (hemija); kao podniz bi pogodio `kanjon` = кањон
})

# Prefiksi kod kojih poslednje slovo prefiksa + prvo slovo korena slučajno
# daju digraf. Pravilo se primenjuje samo na početku reči.
#   nad|živeti, pod|žanr, pred|životni, od|živeti
# `iz-` je ovde radi potpunosti: sa trenutnim skupom digrafa nikad ne okida
# (iz + j daje "zj", što nije digraf). Ostaje kao dokumentacija namere.
# `van-` NAMERNO nije ovde: `Vanja`, `Vanjin` bi bili pokvareni. Vidi SPLIT_STEMS_ANY.
SPLIT_PREFIXES: tuple[str, ...] = ("nad", "pod", "pred", "od", "iz")

# Reči kod kojih prefiksno pravilo NE važi jer je digraf pravi.
PREFIX_RULE_EXCEPTIONS: tuple[str, ...] = (
    "odžak",   # оџак
    "odžač",   # odžačar, odžačiti
)


def _compile_stems(stems: frozenset[str]) -> tuple[tuple[str, int], ...]:
    """Za svaku osnovu nađi offset prvog digrafa u njoj."""
    out: list[tuple[str, int]] = []
    for stem in stems:
        for i in range(len(stem) - 1):
            if stem[i : i + 2] in DIGRAPHS:
                out.append((stem, i))
                break
        else:
            raise ValueError(f"Osnova bez digrafa u listi izuzetaka: {stem!r}")
    return tuple(sorted(out))


_STEMS_ANY = _compile_stems(SPLIT_STEMS_ANY)
_STEMS_START = _compile_stems(SPLIT_STEMS_START)


def forced_splits(word_lower: str) -> set[int]:
    """Indeksi (početak digrafa) na kojima digraf treba čitati kao dva slova."""
    splits: set[int] = set()

    for stem, off in _STEMS_ANY:
        start = 0
        while True:
            pos = word_lower.find(stem, start)
            if pos < 0:
                break
            splits.add(pos + off)
            start = pos + 1

    for stem, off in _STEMS_START:
        if word_lower.startswith(stem):
            splits.add(off)

    if not word_lower.startswith(PREFIX_RULE_EXCEPTIONS):
        for prefix in SPLIT_PREFIXES:
            if word_lower.startswith(prefix):
                k = len(prefix) - 1  # poslednje slovo prefiksa
                if word_lower[k : k + 2] in DIGRAPHS:
                    splits.add(k)

    return splits


# --------------------------------------------------------------------------
# Latinica -> ćirilica
# --------------------------------------------------------------------------


def _cyr_digraph_cased(cyr: str, first: str, second: str) -> str:
    """`Dž`/`DŽ` -> `Џ`, `dž` -> `џ`."""
    return cyr.upper() if first.isupper() else cyr


def _word_to_cyrillic(word: str) -> str:
    lower = word.lower()
    splits = forced_splits(lower)
    out: list[str] = []
    i = 0
    n = len(word)
    while i < n:
        pair = lower[i : i + 2]
        if i not in splits and pair in DIGRAPHS:
            out.append(_cyr_digraph_cased(DIGRAPHS[pair], word[i], word[i + 1]))
            i += 2
            continue
        ch = word[i]
        mapped = _LAT_SINGLE.get(ch.lower())
        if mapped is None:
            out.append(ch)  # strano slovo (q, w, x, y) ostaje kako jeste
        else:
            out.append(mapped.upper() if ch.isupper() else mapped)
        i += 1
    return "".join(out)


def to_cyrillic(text: str) -> str:
    """Latinicu prevedi u ćirilicu. Već ćirilični delovi ostaju netaknuti."""
    return _WORD_RE.sub(lambda m: _word_to_cyrillic(m.group(0)), text)


# --------------------------------------------------------------------------
# Ćirilica -> latinica
# --------------------------------------------------------------------------


def _lat_digraph_cased(lat: str, ch: str, nxt: str | None, prev: str | None) -> str:
    """`џ` -> `dž`; `Џ` -> `Dž` ili `DŽ` u zavisnosti od okoline."""
    if not ch.isupper():
        return lat
    neighbour = nxt if (nxt and nxt.isalpha()) else prev
    if neighbour and neighbour.isalpha() and neighbour.isupper():
        return lat.upper()
    return lat.capitalize()


def to_latin(text: str) -> str:
    """Ćirilicu prevedi u latinicu. Već latinični delovi ostaju netaknuti."""
    out: list[str] = []
    n = len(text)
    for i, ch in enumerate(text):
        low = ch.lower()
        if low in _CYR_DIGRAPH:
            nxt = text[i + 1] if i + 1 < n else None
            prev = text[i - 1] if i > 0 else None
            out.append(_lat_digraph_cased(_CYR_DIGRAPH[low], ch, nxt, prev))
        elif low in _CYR_SINGLE:
            lat = _CYR_SINGLE[low]
            out.append(lat.upper() if ch.isupper() else lat)
        else:
            out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# Detekcija pisma
# --------------------------------------------------------------------------


def script_counts(text: str) -> tuple[int, int]:
    """(broj ćiriličnih slova, broj slova specifičnih za latinicu)."""
    cyr = sum(1 for ch in text if ch in SRB_CYR_LETTERS)
    lat = sum(1 for ch in text if ch.isalpha() and ch not in SRB_CYR_LETTERS)
    return cyr, lat


def detect_script(text: str) -> str:
    """Vrati 'Cyrl', 'Latn' ili 'mixed'. Prazan/bez slova -> 'Latn'."""
    cyr, lat = script_counts(text)
    if cyr == 0:
        return "Latn"
    if lat == 0:
        return "Cyrl"
    return "mixed"


def convert(text: str, script: str) -> str:
    """Prebaci tekst u traženo pismo ('Latn' ili 'Cyrl').

    Idempotentno: `convert(convert(t, s), s) == convert(t, s)`. Za `Cyrl` se ide
    preko latinice baš zato — tako mešan ulaz i ulaz koji je već u ciljnom pismu
    prolaze kroz isti put. Nema prečice „već je ćirilica, preskoči": ona bi na
    mešanom tekstu okinula za pogrešan deo i ostavila ga neprevedenog.
    """
    if script == "Cyrl":
        return to_cyrillic(to_latin(text))
    if script == "Latn":
        return to_latin(text)
    raise ValueError(f"Nepoznato pismo: {script!r} (očekivano 'Latn' ili 'Cyrl')")


def convert_all(texts: Iterable[str], script: str) -> list[str]:
    """`convert` nad celim skupom tekstova — jedan prolaz, jedno pismo."""
    return [convert(t, script) for t in texts]


# --------------------------------------------------------------------------
# Provera: da li je gotov tekst zaista u jednom pismu
# --------------------------------------------------------------------------

#: Jezici koji se pišu i ćirilicom i latinicom. Samo za njih ima smisla birati
#: pismo — i samo za njih se pismo proverava.
BISCRIPTAL_LANGUAGES: frozenset[str] = frozenset({"sr"})


def is_biscriptal(language: str | None) -> bool:
    return bool(language) and language in BISCRIPTAL_LANGUAGES


def foreign_letters(text: str, script: str) -> set[str]:
    """Slova iz drugog pisma koja je `convert` morao da prevede, a nisu.

    `q`, `w`, `x`, `y` NISU tu: nemaju ćirilični par, pa ih `to_cyrillic`
    namerno ostavlja (`Netflix` -> `Нетфликx`). Prijaviti njih značilo bi da
    svaki ispravan ćirilični titl sa stranim imenom pada kao „mešan".
    """
    if script == "Cyrl":
        return {ch for ch in text if ch in SRB_LATIN_LETTERS}
    if script == "Latn":
        return {ch for ch in text if ch in SRB_CYR_LETTERS}
    raise ValueError(f"Nepoznato pismo: {script!r} (očekivano 'Latn' ili 'Cyrl')")


def script_violations(
    items: Iterable[tuple[Any, str]], script: str
) -> list[tuple[Any, set[str]]]:
    """[(oznaka, strana slova)] za svaku stavku koja nije u traženom pismu.

    `items` su parovi (oznaka za poruku — id rečenice, redni broj cue-a — tekst).
    """
    out: list[tuple[Any, set[str]]] = []
    for label, text in items:
        foreign = foreign_letters(text, script)
        if foreign:
            out.append((label, foreign))
    return out
