"""Pravopisni razmaci ciljnog jezika i jedan oblik apostrofa, nad GOTOVIM
redovima titla.

Apostrof: `I'm` i `we’re` u istom fajlu -> svuda `’`, za svaki jezik.

Francuski traži razmak ispred `? ! : ;` i `»` i iza `«`. Razmak mora biti
nerazlomljiv (U+00A0): običan dozvoljava plejeru da prelomi red između reči i
znaka, pa `?` ostane sam na početku sledećeg reda.

Primenjuje se na redove, ne na tekst rečenice, jer tokenizer (`str.split`)
nerazlomljiv razmak vidi kao običan i odvaja znak u zaseban token. Prelamanje
ipak zna za te znakove unapred: `linebreak._Span` meri red provučen kroz
`apply` (`LanguageRules.typographic_width`), pa se prekoračenje ne pojavi tek
posle ove obrade. Izveštaj o prekoračenjima se u svakom slučaju računa nad redovima
POSLE ove obrade.
"""

from __future__ import annotations

import re

NBSP = " "

_SPACES = "   "

# Niz `?!` je jedna celina: razmak ide ispred prvog znaka, ne između njih.
_BEFORE_RE = re.compile(rf"(?<=\S)[{_SPACES}]*(\?[?!]*|![?!]*|[;:»])")
_AFTER_RE = re.compile(rf"«[{_SPACES}]*(?=\S)")


def _before(m: re.Match[str]) -> str:
    mark = m.group(1)
    if mark == ":":
        text, at, end = m.string, m.start(1), m.end(1)
        prev = text[at - 1] if at else ""
        nxt = text[end] if end < len(text) else ""
        # 10:30, 1:0, https:// — dvotačka koja nije interpunkcija
        if (prev.isdigit() and nxt.isdigit()) or text.startswith("//", end):
            return m.group(0)
    return NBSP + mark


def french(line: str) -> str:
    """Red francuskog titla sa nerazlomljivim razmacima. Idempotentno."""
    line = _BEFORE_RE.sub(_before, line)
    return _AFTER_RE.sub("«" + NBSP, line)


#: Jedan oblik apostrofa po fajlu. Tipografski, jer ga model češće vraća, pa
#: se tako menja manje znakova. Obe forme imaju po jedan znak — dužina reda se
#: ne menja.
APOSTROPHE = "’"

# Samo apostrof iza slova/cifre (we're, kids', l'amour). Prav navodnik na
# početku reči je češće otvoren polunavodnik nego apostrof — ne dira se.
_APOSTROPHE_RE = re.compile(r"(?<=\w)'")


def apostrophes(line: str) -> str:
    """`I'm` i `we’re` u istom fajlu -> svuda `’`."""
    return _APOSTROPHE_RE.sub(APOSTROPHE, line)


# ---- navodnici -----------------------------------------------------------

_QUOTES = '"“”„'
#: Za `linebreak`: sadrži li tekst nešto što ovaj modul menja po dužini.
QUOTE_CHARS = _QUOTES
#: Znak ispred krivog navodnika po kome se zna da navodnik OTVARA citat.
_OPENERS_BEFORE = " \t\n\u00a0([«"
_SPACE = "[ \u00a0]+"


def _quote_roles(text: str) -> list[tuple[int, bool]]:
    """(pozicija, otvara li) za svaki navodnik u tekstu cue-a.

    Prav `"` ne zna svoju stranu, pa se sparuje naizmenično kroz CEO cue —
    citat ume da počne u jednom redu a završi se u drugom. Kriv navodnik se
    čita po položaju, ne po obliku: `“` otvara u engleskom a zatvara u srpskom
    („…“), a model vraća oba.
    """
    roles: list[tuple[int, bool]] = []
    straight_open = False
    for i, ch in enumerate(text):
        if ch not in _QUOTES:
            continue
        if ch == '"':
            straight_open = not straight_open
            roles.append((i, straight_open))
        else:
            prev = text[i - 1] if i else " "
            roles.append((i, prev in _OPENERS_BEFORE))
    return roles


def _tighten(text: str, roles: dict[int, bool]) -> str:
    """Ukloni razmak sa unutrašnje strane navodnika (`" Comment`).

    Nastaje u prelamanju: tokenizer odvaja prav navodnik u zaseban token, a
    `join_words` ga ne lepi za reč iza njega.
    """
    drop: set[int] = set()
    for i, is_open in roles.items():
        step = 1 if is_open else -1
        k = i + step
        while 0 <= k < len(text) and text[k] in " \u00a0":
            drop.add(k)
            k += step
    return "".join(ch for i, ch in enumerate(text) if i not in drop)


#: Par navodnika (otvara, zatvara) u koji se pretvara PRAV `"` po jeziku.
#: Pravi navodnik nema stranu — tokenizer ga odvaja od reči, pa je prelamanje
#: umelo da ga preseče od citata (`"Ringe ringe raja" "` / `Ringe raja sa ""`).
#: Usmereni se lepe za reč i taj rez više ne postoji.
QUOTE_PAIRS: dict[str, tuple[str, str]] = {
    "en": ("“", "”"),
    "fr": ("«", "»"),
    "sr": ("„", "“"),
    "de": ("„", "“"),
}
DEFAULT_QUOTE_PAIR = ("“", "”")


def _replace_quotes(text: str, opening: str, closing: str, *, only_straight: bool) -> str:
    """Navodnici -> par jezika, priljubljeni uz citat."""
    if not any(q in text for q in _QUOTES):
        return text
    text = _tighten(text, dict(_quote_roles(text)))
    roles = dict(_quote_roles(text))  # pozicije posle skidanja razmaka
    out = []
    for i, ch in enumerate(text):
        if i in roles and (ch == '"' or not only_straight):
            out.append(opening if roles[i] else closing)
        else:
            out.append(ch)
    return "".join(out)


def directional_quotes(text: str, language: str | None) -> str:
    """Prav `"` -> usmeren navodnik jezika. Za tekst rečenice PRE prelamanja.

    Krivi navodnici ostaju kakvi jesu (srpski original sa `“ ”` se ne dira);
    menja se samo onaj koji ne zna svoju stranu.
    """
    lang = (language or "").split("-")[0].lower()
    return _replace_quotes(
        text, *QUOTE_PAIRS.get(lang, DEFAULT_QUOTE_PAIR), only_straight=True
    )


def straight_quotes(text: str) -> str:
    """Jezici bez para: oblik navodnika ostaje, samo se priljubi uz citat."""
    if not any(q in text for q in _QUOTES):
        return text
    return _tighten(text, dict(_quote_roles(text)))


def french_quotes(text: str) -> str:
    """`"…"`, `“…”`, `„…“` -> `«…»`; nerazlomljiv razmak unutra dodaje `french`."""
    return _replace_quotes(text, "«", "»", only_straight=False)


_ALWAYS_OPEN = "„«"
_ALWAYS_CLOSE = "”»"


def quotes_balanced(text: str) -> bool:
    """Svaki citat otvoren i zatvoren, i nijedan ne počinje zatvaranjem.

    Po obliku gde je oblik jednoznačan (`„ «` otvaraju, `” »` zatvaraju); `“`
    po položaju (otvara u engleskom, zatvara u srpskom); prav `"` naizmenično.
    """
    depth = 0
    straight_open = False
    for i, ch in enumerate(text):
        if ch in _ALWAYS_OPEN:
            depth += 1
        elif ch in _ALWAYS_CLOSE:
            depth -= 1
        elif ch == "“":
            prev = text[i - 1] if i else " "
            depth += 1 if prev in _OPENERS_BEFORE else -1
        elif ch == '"':
            straight_open = not straight_open
            depth += 1 if straight_open else -1
        else:
            continue
        if depth < 0:
            return False
    return depth == 0


def _per_cue(fix, lines: list[str]) -> list[str]:
    """Primeni `fix` na ceo cue odjednom (navodnici prelaze preko reda)."""
    return fix("\n".join(lines)).split("\n") if lines else lines


_BY_LANGUAGE = {"fr": french}


def apply(lines: list[str], language: str | None) -> list[str]:
    """Jedan cue: apostrofi i navodnici za sve jezike, pa pravopis jezika.

    Francuski: svi navodnici -> `« »`. Ostali jezici: samo prav `"` -> par
    jezika (`“ ”` za engleski); krivi ostaju kakve je autor stavio.
    """
    lang = (language or "").split("-")[0].lower()
    out = [apostrophes(l) for l in lines]
    if lang == "fr":
        quote_fix = french_quotes
    else:
        def quote_fix(text: str) -> str:
            return straight_quotes(directional_quotes(text, lang))
    out = _per_cue(quote_fix, out)
    fix = _BY_LANGUAGE.get(lang)
    return [fix(l) for l in out] if fix else out
