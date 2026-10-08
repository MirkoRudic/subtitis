"""Prevod postojećeg SRT-a: cue-ovi i vremena se ne diraju, menja se samo tekst.

Tok:

1. `assemble` — cue-ovi -> rečenice, po interpunkciji. Rečenica pamti koje
   cue-ove pokriva i koliki je udeo znakova imala u svakom (`CuePart`).
2. Prevod ide kao i inače, rečenicu po rečenicu, sa kontekstom.
3. `build` — prevedena rečenica se deli na onoliko delova koliko je original
   imao cue-ova (`linebreak.split_by_shares`, iste kazne kao pri prelamanju),
   delovi se sklapaju po cue-ovima i lome u redove (`linebreak.best_lines`).

Vremena u izlazu su vremena iz ulaza, bajt za bajt. Ništa u ovom modulu ne
računa novo vreme; prekoračenja dužine i CPS-a se PRIJAVLJUJU (`overflows`),
nikad ne popravljaju skraćivanjem.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

from . import linebreak
from . import translit
from . import typography
from .models import Cue, CuePart, CuesFile, Profile, Sentence, SourceCue, Word
from . import segment as segment_mod
from .segment import (
    NO_SPACE_AFTER,
    NO_SPACE_BEFORE,
    SENTENCE_END_CHARS,
    _is_boundary,
    _starts_upper,
    join_words,
    split_token,
    tokenize_text,
)

#: Rečenica bez završnog znaka se ne razvlači preko više od ovoliko cue-ova.
#: Bez granice bi SRT bez interpunkcije (tekst pesme, automatski titl) postao
#: jedna rečenica od stotinu cue-ova, a prevod jedne linije od 5000 znakova.
MAX_CUES_PER_SENTENCE = 6

#: Rečenica duža od ovoga se seče na granici cue-a sa NAJVEĆOM pauzom (pri
#: izjednačenju najbližoj sredini). Isti prag kao za ASR. Ulazni SRT ume da
#: bude nedosledno interpunktiran: bez ovoga je pet cue-ova bez tačke postalo
#: jedna „rečenica" od 15,96 s sa tri misli.
#:
#: Izmereno 2026-09-28 (Otvorena vrata 26, 305 cue-ova): 189 -> 195 rečenica, najduža
#: 20,39 -> 13,96 s. Najveća pauza je dala 2 tačna preseka i 1 loš
#: (`da vidimo da / u gradu`); zarez-prvo, kao u ASR putanji, 2 tačna i 2 loša.
SRT_MAX_SENTENCE_DURATION: float | None = segment_mod.MAX_SENTENCE_DURATION

#: Pauza između cue-ova od koje je granica rečenice bez obzira na interpunkciju.
#: NAMERNO ISKLJUČENO (None) — isto kao filter halucinacija: mehanizam se ne
#: uključuje bez dokaza da pomaže. Mehanizam postoji (`_split_by_timing`).
#:
#: Izmereno 2026-09-28 na istom poslu, uz trajanje od 15 s (primeri preseka
#: su izmišljeni, istog oblika kao stvarni):
#:   samo trajanje   195 rečenica
#:   + pauza 2,0 s   196 — jedini dodatni presek je POGREŠAN
#:                         (`Kad stignemo do jezera, / pravimo pauzu za ručak`)
#:   + pauza 1,5 s   197 — dodaje i `Lana, Teodora, / Ilija, Nevena` (usred nabrajanja)
#:   + pauza 1,0 s   198 — dodaje tačan `završimo, / Hvala svima` i pogrešan
#:                         `hoću da postanem / pravi planinar`
#: Jedini tačan presek iz prijave (`bor... / ...posle`, 3,27 s) hvata već
#: trajanje: to je najveća pauza u toj rečenici od 15,96 s.
SRT_MAX_GAP: float | None = None

#: Flag na cue-u čiji se kurziv posle prevoda nije mogao vratiti na isto mesto.
ITALIC_LOST_FLAG = "italic_lost"
#: Flag na cue-u koji je posle raspoređivanja ostao bez teksta.
EMPTY_FLAG = "empty_after_translation"

_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z]+)[^>]*>")
_ASS_RE = re.compile(r"\{\\[^}]*\}")
_CLOSING_QUOTES = "\"»”’)'"
_DIALOG_DASHES = ("-", "–", "—")


# ==========================================================================
# Oznake
# ==========================================================================


def strip_markup(line: str) -> str:
    """Red bez HTML oznaka i ASS blokova — ono što se čita i broji."""
    return _TAG_RE.sub("", _ASS_RE.sub("", line))


def outside_markup(line: str, fn: Callable[[str], str]) -> str:
    """Primeni `fn` samo na tekst između SRT/ASS oznaka; oznake ostaju iste."""
    out: list[str] = []
    pos = 0
    for m in re.finditer(rf"{_TAG_RE.pattern}|{_ASS_RE.pattern}", line):
        out.append(fn(line[pos : m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(fn(line[pos:]))
    return "".join(out)


def convert_markup(line: str, script: str) -> str:
    """`translit.convert` samo nad tekstom, ne nad oznakama.

    Bez ovoga ćirilica pretvori `<i>` u `<и>` i plejer ispiše oznaku kao tekst.
    """
    return outside_markup(line, lambda t: translit.convert(t, script))


#: Veličina slova u izlaznom SRT-u. Primenjuje se tek pri pisanju fajla;
#: rečenice (`sentences.*.json`) ostaju kakve jesu.
CASES = ("normal", "upper", "lower")

#: Slova čije bi `upper()`/`lower()` promenilo broj znakova (`ß` -> `SS`).
#: Profil broji znakove, pa veličina slova ne sme da ga menja.
_CASE_SAME_LENGTH = {"ß": "ẞ"}


def _change_case(text: str, case: str) -> str:
    out = []
    for ch in text:
        new = ch.upper() if case == "upper" else ch.lower()
        if len(new) != 1:
            new = _CASE_SAME_LENGTH.get(ch, ch) if case == "upper" else ch
        out.append(new)
    return "".join(out)


def apply_case(line: str, case: str) -> str:
    r"""VELIKA / mala slova samo nad tekstom, ne nad oznakama (`<i>`, `{\an8}`)."""
    if case == "normal":
        return line
    if case not in CASES:
        raise ValueError(f"nepoznata veličina slova: {case!r}")
    return outside_markup(line, lambda t: _change_case(t, case))


#: Unutar reči ostaju: apostrof u eliziji (`c'est`, `aujourd'hui`) i crtica
#: u složenici (`rendez-vous`) — deo pravopisa reči, ne znak rečenice.
_WORD_INNER = "'’-‐"

#: Između dve cifre ostaju i tačka, zarez i dve tačke: `3,5`, `1.000`, `15:30`.
#: Bez toga broj menja vrednost (`3,5` -> `35`) — to više nije izgled titla.
_NUMBER_INNER = ".,:"

#: Znaci kategorije „interpunkcija" koji nose značenje, a ne rečenicu.
_KEEP_SIGNS = "%&@#/"

_SPACES = re.compile(r"[ \t  ]{2,}")


def _drop_punct(text: str) -> str:
    out = []
    for k, ch in enumerate(text):
        if not unicodedata.category(ch).startswith("P") or ch in _KEEP_SIGNS:
            out.append(ch)
            continue
        prev = text[k - 1] if k else ""
        nxt = text[k + 1] if k + 1 < len(text) else ""
        if ch in _WORD_INNER and prev.isalnum() and nxt.isalnum():
            out.append(ch)
        elif ch in _NUMBER_INNER and prev.isdigit() and nxt.isdigit():
            out.append(ch)
    return _SPACES.sub(" ", "".join(out))


def drop_punctuation(line: str) -> str:
    r"""Izbaci interpunkciju između reči i na kraju reči; ostavi onu UNUTAR reči.

    Oznake (`<i>`, `{\an8}`) se ne diraju. Primenjuje se tek pri pisanju SRT-a,
    POSLE prelamanja: prelamanje bira rez po zarezima (kazna 0), pa bi
    interpunkcija skinuta ranije odnela upravo taj signal.
    """
    parts = re.split(rf"({_TAG_RE.pattern}|{_ASS_RE.pattern})", line)
    # re.split sa grupama u obrascu vraća i podgrupe — zadrži samo tekst i celu oznaku
    pieces: list[tuple[bool, str]] = []
    k = 0
    while k < len(parts):
        pieces.append((False, parts[k] or ""))
        if k + 1 < len(parts):
            pieces.append((True, parts[k + 1]))
        k += 1 + 1 + _TAG_RE.groups + _ASS_RE.groups
    texts = [i for i, (tag, _) in enumerate(pieces) if not tag]
    for i in texts:
        pieces[i] = (False, _drop_punct(pieces[i][1]))
    # razmak sa obe strane oznake postaje jedan; krajevi reda bez razmaka
    last_space = True
    for i in texts:
        t = pieces[i][1]
        if last_space:
            t = t.lstrip(" \t  ")
        if t:
            last_space = t[-1] in " \t  "
        pieces[i] = (False, t)
    for i in reversed(texts):
        t = pieces[i][1].rstrip(" \t  ")
        pieces[i] = (False, t)
        if t:
            break
    return "".join(t for _, t in pieces)


def output_lines(lines: list[str], case: str, drop_punct: bool) -> list[str]:
    """Izlazne transformacije cue-a, redom: veličina slova, pa interpunkcija.

    Transliteracija je pre njih (u `run_build`); sve tri komutiraju.
    Prekoračenja profila se broje PRE ovoga, nad redom sa interpunkcijom —
    izostavljena interpunkcija red skraćuje, pa poneko prijavljeno
    prekoračenje u gotovom fajlu ne postoji. To je bezbedan smer (prijavi se
    više, ne manje) i namerno se ne ispravlja.
    """
    out = [apply_case(l, case) for l in lines]
    if drop_punct:
        dropped = [drop_punctuation(l) for l in out]
        kept = [l for l in dropped if strip_markup(l).strip()]
        # cue sav od interpunkcije (`…`) ostaje kakav je — prazan cue nije titl
        out = kept or out
    return out


@dataclass
class _Tok:
    text: str
    cue: int
    line: int
    italic: bool
    #: token nema nijedno slovo ni cifru — ne ulazi u odluku o kurzivu
    punct: bool


@dataclass
class ImportStats:
    cues: int = 0
    sentences: int = 0
    forced_breaks: list[int] = field(default_factory=list)  # redni brojevi cue-ova
    #: cue-ovi posle kojih je rečenica presečena zbog pauze / trajanja
    gap_breaks: list[int] = field(default_factory=list)
    duration_breaks: list[int] = field(default_factory=list)
    italic_cues: int = 0
    partial_italic: list[int] = field(default_factory=list)
    dropped_tags: dict[str, int] = field(default_factory=dict)
    empty_cues: list[int] = field(default_factory=list)


def _line_tokens(
    raw: str, cue: int, line: int, italic: bool, stats: ImportStats
) -> tuple[list[_Tok], bool]:
    """Jedan red -> tokeni sa oznakom kurziva. Vraća i stanje kurziva na kraju reda."""
    raw = _ASS_RE.sub("", raw)
    chars: list[str] = []
    mask: list[bool] = []
    pos = 0
    for m in _TAG_RE.finditer(raw):
        for ch in raw[pos : m.start()]:
            chars.append(ch)
            mask.append(italic)
        name = m.group(2).lower()
        if name == "i":
            italic = m.group(1) != "/"
        else:
            stats.dropped_tags[name] = stats.dropped_tags.get(name, 0) + 1
        pos = m.end()
    for ch in raw[pos:]:
        chars.append(ch)
        mask.append(italic)
    plain = "".join(chars)

    toks: list[_Tok] = []
    for m in re.finditer(r"\S+", plain):
        offset = m.start()
        for piece in split_token(m.group(0)):
            at = plain.index(piece, offset)
            letters = [mask[at + k] for k, ch in enumerate(piece) if ch.isalnum()]
            offset = at + len(piece)
            # `...ćemo stići` — tri tačke su oznaka nastavka iz prethodnog
            # cue-a, ne deo reči. Ostavljene, zalepile bi se za prethodnu reč
            # (`da......ćemo`), jer tačka ne trpi razmak ispred sebe.
            stripped = piece.lstrip(".…")
            if stripped != piece and stripped[:1].isalnum():
                piece = stripped
            toks.append(
                _Tok(
                    text=piece, cue=cue, line=line,
                    italic=bool(letters) and all(letters),
                    punct=not letters,
                )
            )
    return toks, italic


def cue_tokens(cue: SourceCue, index: int, stats: ImportStats | None = None) -> list[_Tok]:
    """Svi tokeni jednog cue-a. Kurziv se prenosi preko reda, ne preko cue-a."""
    stats = stats or ImportStats()
    italic = False
    out: list[_Tok] = []
    for ln, raw in enumerate(cue.lines):
        toks, italic = _line_tokens(raw, index, ln, italic, stats)
        out.extend(toks)
    return out


def split_prefix(cue: SourceCue) -> SourceCue:
    """Vodeći `{\\an8}` iz prvog reda prebaci u `prefix`."""
    if not cue.lines:
        return cue
    m = re.match(rf"^\s*((?:{_ASS_RE.pattern})+)", cue.lines[0])
    if not m:
        return cue
    lines = [cue.lines[0][m.end() :]] + cue.lines[1:]
    return cue.model_copy(
        update={"prefix": m.group(1), "lines": [l for l in lines if l.strip()]}
    )


# ==========================================================================
# Sastavljanje rečenica
# ==========================================================================


def _as_words(toks: list[_Tok]) -> list[Word]:
    return [Word(i=k, s=0.0, e=0.0, w=t.text, p=1.0) for k, t in enumerate(toks)]


def _ends_sentence(words: list[Word], k: int) -> bool:
    """Da li token `k` zatvara rečenicu.

    Isto pravilo kao `segment._is_boundary` (skraćenice, redni brojevi), uz
    dve stvari kojih u ASR-u nema, a u ručno pravljenom titlu ima:
    završni navodnik posle tačke (`."`), i tri tačke kao nastavak u sledećem
    cue-u (`Mislio sam da...` / `...ćemo stići`).
    """
    tok = words[k].w.strip()
    core = tok.rstrip(_CLOSING_QUOTES) or tok
    if not core or not all(ch in SENTENCE_END_CHARS for ch in core):
        return False
    original = words[k]
    words[k] = original.model_copy(update={"w": core})
    try:
        if not _is_boundary(words, k):
            return False
    finally:
        words[k] = original
    is_ellipsis = core == "…" or (len(core) >= 2 and set(core) == {"."})
    if is_ellipsis and k + 1 < len(words):
        nxt = words[k + 1].w.lstrip(".…")
        if nxt and not _starts_upper(nxt):
            return False
    return True


def _split_by_timing(
    group: list[_Tok], cues: list[SourceCue], max_gap: float | None,
    max_duration: float | None, stats: ImportStats,
) -> list[list[_Tok]]:
    """Preseci rečenicu na granici cue-a kad interpunkcija ne kaže dovoljno.

    Ulazni SRT ume da bude nedosledno interpunktiran: pet cue-ova bez tačke,
    sa 3 s tišine između dva, spojeno je u jednu „rečenicu" od 16 s. Seče se
    samo na GRANICI cue-a — unutar cue-a nema vremena po rečima.
    """
    # granice cue-ova unutar grupe: indeks prvog tokena novog cue-a
    def boundaries(g: list[_Tok]) -> list[int]:
        return [k for k in range(1, len(g)) if g[k].cue != g[k - 1].cue]

    def gap_at(g: list[_Tok], k: int) -> float:
        return (cues[g[k].cue].start_ms - cues[g[k - 1].cue].end_ms) / 1000

    parts = [group]
    if max_gap is not None:
        parts = []
        cur_start = 0
        for k in boundaries(group):
            if gap_at(group, k) >= max_gap:
                parts.append(group[cur_start:k])
                stats.gap_breaks.append(group[k - 1].cue + 1)
                cur_start = k
        parts.append(group[cur_start:])

    if max_duration is None:
        return parts

    def span(g: list[_Tok]) -> float:
        return (cues[g[-1].cue].end_ms - cues[g[0].cue].start_ms) / 1000

    def split_long(g: list[_Tok]) -> list[list[_Tok]]:
        bs = boundaries(g)
        if span(g) <= max_duration or not bs:
            return [g]
        mid = (cues[g[0].cue].start_ms + cues[g[-1].cue].end_ms) / 2000
        # najveća pauza, pa pri izjednačenju najbliža sredini (kao `segment._split_long`)
        k = min(bs, key=lambda k: (-gap_at(g, k), abs(cues[g[k].cue].start_ms / 1000 - mid)))
        stats.duration_breaks.append(g[k - 1].cue + 1)
        return split_long(g[:k]) + split_long(g[k:])

    return [piece for p in parts for piece in split_long(p)]


def assemble(
    cues: list[SourceCue], *, normalize: Callable[[str], str] | None = None,
    max_gap: float | None = SRT_MAX_GAP,
    max_duration: float | None = SRT_MAX_SENTENCE_DURATION,
) -> tuple[list[Sentence], ImportStats]:
    """Cue-ovi -> rečenice sa `cue_parts`.

    `normalize` se primenjuje na svaki token PRE brojanja znakova (za srpski:
    ćirilica -> latinica), da udeli važe za tekst koji se zaista prevodi.
    `max_gap` i `max_duration` (sekunde) su granice rečenice mimo interpunkcije
    (`_split_by_timing`).
    """
    stats = ImportStats(cues=len(cues))
    toks: list[_Tok] = []
    for idx, cue in enumerate(cues):
        ct = cue_tokens(cue, idx, stats)
        if not ct:
            stats.empty_cues.append(idx + 1)
        if ct and all(t.italic for t in ct if not t.punct):
            stats.italic_cues += 1
        toks.extend(ct)
    if normalize:
        for t in toks:
            t.text = normalize(t.text)

    words = _as_words(toks)
    groups: list[list[_Tok]] = []
    cur: list[_Tok] = []
    for k, t in enumerate(toks):
        if cur:
            new_line = t.cue != cur[-1].cue or t.line != cur[-1].line
            # Crta na početku reda je nova replika — nova rečenica, bez obzira
            # na to čime se prethodni red završio.
            dialog = new_line and t.text in _DIALOG_DASHES
            span_cues = cur[-1].cue - cur[0].cue + 1
            too_long = t.cue != cur[-1].cue and span_cues >= MAX_CUES_PER_SENTENCE
            if dialog or too_long:
                if too_long:
                    stats.forced_breaks.append(cur[-1].cue + 1)
                groups.append(cur)
                cur = []
        cur.append(t)
        if _ends_sentence(words, k):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    groups = [
        piece for g in groups
        for piece in _split_by_timing(g, cues, max_gap, max_duration, stats)
    ]

    sentences: list[Sentence] = []
    for sid, group in enumerate(groups, start=1):
        parts: list[CuePart] = []
        by_cue: list[list[_Tok]] = []
        for t in group:
            if by_cue and by_cue[-1][0].cue == t.cue:
                by_cue[-1].append(t)
            else:
                by_cue.append([t])
        sizes = [len(join_words(_as_words(p))) for p in by_cue]
        total = sum(sizes) or 1
        for p, size in zip(by_cue, sizes):
            letters = [t.italic for t in p if not t.punct]
            if letters and all(letters):
                italic = "all"
            elif any(letters):
                italic = "partial"
                if p[0].cue + 1 not in stats.partial_italic:
                    stats.partial_italic.append(p[0].cue + 1)
            else:
                italic = ""
            parts.append(CuePart(
                cue=p[0].cue, chars=size, share=round(size / total, 4), italic=italic,
            ))
        first, last = cues[group[0].cue], cues[group[-1].cue]
        sentences.append(Sentence(
            id=sid,
            s=first.start_ms / 1000,
            e=last.end_ms / 1000,
            text=join_words(_as_words(group)),
            words=(0, -1),
            cue_parts=parts,
        ))

    # Prelom reda na granici rečenica unutar cue-a (dijalog) se čuva.
    for a, b in zip(sentences, sentences[1:]):
        last_a, first_b = a.cue_parts[-1], b.cue_parts[0]
        if last_a.cue == first_b.cue:
            end_tok = groups[a.id - 1][-1]
            start_tok = groups[b.id - 1][0]
            last_a.line_break = start_tok.line != end_tok.line

    stats.sentences = len(sentences)
    return sentences, stats


# ==========================================================================
# Raspoređivanje nazad u cue-ove
# ==========================================================================


@dataclass
class _Segment:
    words: list[Word]
    italic: bool
    line_break: bool
    sentence_id: int


@dataclass
class FixedBuild:
    cues: list[Cue]
    #: za svaki cue (po poziciji): id-jevi rečenica čiji je deo u njemu
    cue_sentences: list[list[int]]
    italic_lost: list[int] = field(default_factory=list)  # redni brojevi
    empty: list[int] = field(default_factory=list)
    unchanged: int = 0  # cue-ovi prepisani doslovno iz ulaza


def _render_line(words: list[Word], italic: list[bool]) -> str:
    """Tokeni -> red, uz `<i>…</i>` oko svakog neprekinutog niza kurziva."""
    out: list[str] = []
    open_ = False
    for k, w in enumerate(words):
        piece = w.w
        if k:
            # isto pravilo razmaka kao `segment.join_words`
            glue = piece[:1] in NO_SPACE_BEFORE or words[k - 1].w[-1:] in NO_SPACE_AFTER
            if not glue:
                if open_ and not italic[k]:
                    out.append("</i>")
                    open_ = False
                out.append(" ")
        if italic[k] and not open_:
            out.append("<i>")
            open_ = True
        elif not italic[k] and open_:
            out.append("</i>")
            open_ = False
        out.append(piece)
    if open_:
        out.append("</i>")
    return "".join(out)


def exact_ranges(words: list[Word], parts: list[CuePart]) -> list[tuple[int, int]] | None:
    """Rezovi tačno po `CuePart.chars`, ako tekst to dozvoljava; inače None.

    Za izvorni jezik čiji tekst nije menjan ovo vraća tačno originalnu podelu.
    Skorovanje bi tu moglo da pomeri reč preko granice cue-a (rez iza veznika
    je jeftiniji od reza iza obične reči), a neizmenjen ulaz ne sme da se menja.
    """
    out: list[tuple[int, int]] = []
    start = 0
    n = len(words)
    for idx, part in enumerate(parts):
        if idx == len(parts) - 1:
            end = n
        else:
            end = start + 1
            while end < n and len(join_words(words[start:end])) < part.chars:
                end += 1
        if len(join_words(words[start:end])) != part.chars:
            return None
        out.append((start, end))
        start = end
    return out


def build(
    source: CuesFile,
    sentences: list[Sentence],
    profile: Profile,
    *,
    language: str,
    script: str | None = None,
    exact: bool = False,
    glossary_terms: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> FixedBuild:
    """Rečenice (izvorne ili prevedene) -> cue-ovi sa vremenima iz `source`.

    `script` se primenjuje na tekst pre umetanja oznaka (vidi `convert_markup`).
    `exact=True` je za izvorni jezik: rečenica čija se dužina poklapa sa
    zapamćenim delovima deli se tačno po njima (`exact_ranges`). Cue čiji su
    tokeni isti kao u ulazu prepisuje se doslovno, sa izvornim redovima —
    neizmenjen izvorni jezik ne sme da izgubi ručno doterane prelome.
    """
    warn = log or (lambda m: None)
    rules = linebreak.rules_for(language, warn)
    terms = linebreak.term_tokens(glossary_terms or [])
    n_cues = len(source.cues)
    segments: list[list[_Segment]] = [[] for _ in range(n_cues)]

    for sent in sorted(sentences, key=lambda s: s.id):
        parts = [p for p in sent.cue_parts if 0 <= p.cue < n_cues]
        if not parts:
            warn(f"rečenica {sent.id} nema cue-ove — preskačem je")
            continue
        # Prav `"` postaje usmeren PRE tokenizacije: tokenizer ga odvaja od
        # reči, pa ga je rez između cue-ova odnosio u susedni cue. Izvorni
        # jezik (`exact`) se ne dira — mora da izađe doslovno.
        text = sent.text if exact else typography.directional_quotes(sent.text, language)
        toks = tokenize_text(text)
        words = [Word(i=k, s=0.0, e=0.0, w=t, p=1.0) for k, t in enumerate(toks)]
        ranges = exact_ranges(words, parts) if exact else None
        if ranges is None:
            ranges = linebreak.split_by_shares(
                words, [p.share for p in parts], profile, rules, terms
            )
        for part, (i, j) in zip(parts, ranges):
            segments[part.cue].append(_Segment(
                words=words[i:j],
                italic=part.italic == "all",
                line_break=part.line_break,
                sentence_id=sent.id,
            ))

    def to_script(text: str) -> str:
        return translit.convert(text, script) if script else text

    # Rečenice se drže u latinici (`import_srt`), pa se i ulaz poredi u latinici.
    def stored(text: str) -> str:
        return translit.convert(text, "Latn") if translit.is_biscriptal(language) else text

    same_language = language == source.language
    partial_italic = {
        p.cue for s in sentences for p in s.cue_parts if p.italic == "partial"
    }
    one_line = profile.model_copy(update={"max_lines": 1})
    result = FixedBuild(cues=[], cue_sentences=[])
    for idx, (src_cue, segs) in enumerate(zip(source.cues, segments)):
        number = idx + 1
        flags: list[str] = []
        new_tokens = [w.w for s in segs for w in s.words]
        orig_tokens = [stored(t.text) for t in cue_tokens(src_cue, idx)]

        # Samo za jezik ulaza: u prevodu cue ume slučajno da ostane isti (ime),
        # a izvorni redovi bi tada uneli pismo izvora u tuđ titl.
        if same_language and new_tokens and new_tokens == orig_tokens:
            lines = [
                convert_markup(l, script) if script else l for l in src_cue.lines
            ]
            result.unchanged += 1
        else:
            groups: list[list[_Segment]] = [[]]
            for s in segs:
                groups[-1].append(s)
                if s.line_break:
                    groups.append([])
            groups = [g for g in groups if any(s.words for s in g)]
            prof = profile if len(groups) <= 1 else one_line
            lines = []
            for g in groups:
                words = [w for s in g for w in s.words]
                italic = [s.italic for s in g for _ in s.words]
                words = [w.model_copy(update={"w": to_script(w.w)}) for w in words]
                for a, b in linebreak.best_lines(words, prof, rules, terms):
                    lines.append(_render_line(words[a:b], italic[a:b]))
            if not lines:
                flags.append(EMPTY_FLAG)
                result.empty.append(number)
            # Doslovno prepisan cue zadržava svoj kurziv; ovde je izgubljen.
            if idx in partial_italic:
                flags.append(ITALIC_LOST_FLAG)
                result.italic_lost.append(number)

        result.cues.append(Cue(
            index=number,
            s=src_cue.start_ms / 1000,
            e=src_cue.end_ms / 1000,
            lines=lines,
            flags=flags,
        ))
        result.cue_sentences.append(sorted({s.sentence_id for s in segs}))
    return result


def with_prefixes(cues: list[Cue], source: CuesFile) -> list[Cue]:
    """Vrati vodeće `{\\an8}` ispred prvog reda. Posle provere pisma, ne pre."""
    for cue, src in zip(cues, source.cues):
        if src.prefix:
            cue.lines = [src.prefix + (cue.lines[0] if cue.lines else "")] + cue.lines[1:]
    return cues


# ==========================================================================
# Profil iz ulaznog SRT-a
# ==========================================================================

#: Podrazumevani profil posla iz SRT-a: granice kakve poštuje sam original.
#: 95. percentil, ne maksimum — izmereno 2026-09-28 na prvom francuskom poslu:
#: maksimum (51 zn., 27.9 CPS) postavlja JEDAN red i JEDAN cue originala; kao
#: granica za prelamanje pogurao je 17% francuskih redova preko 42 znaka
#: (original: 12%), a kao granica za izveštaj sakrio 27 cue-ova bržih od 95%
#: originala. Percentil (47 zn., 21.4 CPS) daje 11% dugih redova.
ORIGINAL_PROFILE = "original"
#: Isto, ali granice su najduži red i najbrži cue originala.
ORIGINAL_MAX_PROFILE = "original-max"
SOURCE_PROFILES = (ORIGINAL_PROFILE, ORIGINAL_MAX_PROFILE)

SOURCE_PERCENTILE = 95


@dataclass
class CueMeasure:
    longest: int  # najduži red, bez oznaka
    cps: float
    lines: int


def measure_cue(lines: list[str], start: float, end: float) -> CueMeasure:
    """Ista mera za original i prevod — porede se istim lenjirom."""
    plain = [strip_markup(l) for l in lines]
    chars = len(" ".join(plain))
    dur = end - start
    cps = chars / dur if dur > 0 else (float("inf") if chars else 0.0)
    return CueMeasure(max((len(l) for l in plain), default=0), cps, len(plain))


def source_measures(source: CuesFile) -> list[CueMeasure]:
    return [
        measure_cue(c.lines, c.start_ms / 1000, c.end_ms / 1000) for c in source.cues
    ]


def _percentile(values: list[float], pct: float) -> float:
    """Najbliži rang: vrednost koju zaista ima neki red ili cue originala."""
    v = sorted(values)
    if not v:
        return 0.0
    return v[min(len(v) - 1, max(0, math.ceil(pct / 100 * len(v)) - 1))]


def source_profile(source: CuesFile, name: str = ORIGINAL_PROFILE) -> Profile:
    """Profil čije su granice izmerene na ulaznom SRT-u."""
    ms = [m for m in source_measures(source) if m.lines]
    line_lengths = [len(strip_markup(l)) for c in source.cues for l in c.lines]
    cps = [m.cps for m in ms if m.cps != float("inf")]
    if name == ORIGINAL_MAX_PROFILE:
        chars, speed = max(line_lengths, default=42), max(cps, default=20.0)
        note = "Iz ulaznog SRT-a: najduži red i najbrži cue originala."
    else:
        chars = _percentile(line_lengths, SOURCE_PERCENTILE) or 42
        speed = _percentile(cps, SOURCE_PERCENTILE) or 20.0
        note = (
            f"Iz ulaznog SRT-a: {SOURCE_PERCENTILE}. percentil dužine reda i "
            "brzine — granice koje poštuje sam original."
        )
    return Profile(
        name=name,
        note=note,
        max_chars_per_line=int(chars),
        # naviše na stotinku, da cue originala tačno na granici ne ispadne preko
        max_cps=math.ceil(speed * 100) / 100,
        max_lines=max((m.lines for m in ms), default=2),
    )


def source_profile_line(source: CuesFile) -> str:
    p = source_profile(source, ORIGINAL_PROFILE)
    m = source_profile(source, ORIGINAL_MAX_PROFILE)
    return (
        f"profil iz originala: {p.max_chars_per_line} zn./red, {p.max_cps:.1f} CPS, "
        f"do {p.max_lines} reda ({SOURCE_PERCENTILE}. percentil, profil "
        f"'{ORIGINAL_PROFILE}', podrazumevani); najduži red "
        f"{m.max_chars_per_line} zn., najbrži cue {m.max_cps:.1f} CPS "
        f"(profil '{ORIGINAL_MAX_PROFILE}')"
    )


# ==========================================================================
# Prekoračenja — prijavljuju se, ne popravljaju
# ==========================================================================


@dataclass
class Overflow:
    cue: int  # redni broj u SRT-u, od 1
    sentences: list[int]
    chars: int  # najduži red
    chars_over: int
    cps: float
    cps_over: float
    lines_over: int
    lines: int = 0
    #: relativno prekoračenje, da +9 znakova i +4 CPS budu uporedivi
    severity: float = 0.0
    #: isti cue u ulaznom SRT-u, ako je poznat
    source: CueMeasure | None = None

    @property
    def worse_than_source(self) -> bool:
        """Prevod je na probijenoj meri teži od istog cue-a u originalu.

        Ovo je brojka koja kaže gde skratiti: cue koji je i u originalu bio
        isto toliko dug ili brz je izbor autora, ne posledica prevoda.
        """
        src = self.source
        if src is None:
            return True
        return bool(
            (self.chars_over and self.chars > src.longest)
            or (self.cps_over and self.cps > src.cps + 1e-9)
            or (self.lines_over and self.lines > src.lines)
        )

    def describe(self) -> str:
        bits: list[str] = []
        src = self.source
        if self.chars_over > 0:
            bits.append(
                f"red {self.chars} zn. (+{self.chars_over}"
                + (f"; original {src.longest}" if src else "") + ")"
            )
        if self.cps_over > 0:
            bits.append(
                f"{self.cps:.1f} CPS (+{self.cps_over:.1f}"
                + (f"; original {src.cps:.1f}" if src else "") + ")"
            )
        if self.lines_over > 0:
            bits.append(f"+{self.lines_over} red(a)")
        return f"#{self.cue}: " + ", ".join(bits)

    def as_dict(self) -> dict:
        out = {
            "cue": self.cue, "sentences": self.sentences, "chars": self.chars,
            "chars_over": self.chars_over, "cps": round(self.cps, 2),
            "cps_over": round(self.cps_over, 2), "lines_over": self.lines_over,
            "worse_than_source": self.worse_than_source,
        }
        if self.source is not None:
            out["source_chars"] = self.source.longest
            out["source_cps"] = round(self.source.cps, 2)
        return out


def overflows(
    built: FixedBuild, profile: Profile, source: CuesFile | None = None,
) -> list[Overflow]:
    """Svi cue-ovi koji probijaju profil. Prvo oni gori od originala, pa
    unutar grupe najveće prekoračenje prvo."""
    src = source_measures(source) if source is not None else None
    out: list[Overflow] = []
    for idx, (cue, sids) in enumerate(zip(built.cues, built.cue_sentences)):
        m = measure_cue(cue.lines, cue.s, cue.e)
        o = Overflow(
            cue=cue.index,
            sentences=sids,
            chars=m.longest,
            chars_over=max(0, m.longest - profile.max_chars_per_line),
            cps=m.cps,
            cps_over=max(0.0, m.cps - profile.max_cps),
            lines_over=max(0, m.lines - profile.max_lines),
            lines=m.lines,
            source=src[idx] if src is not None and idx < len(src) else None,
        )
        if o.chars_over or o.cps_over or o.lines_over:
            o.severity = max(
                o.chars_over / profile.max_chars_per_line,
                o.cps_over / profile.max_cps,
                float(o.lines_over),
            )
            out.append(o)
    out.sort(key=lambda o: (not o.worse_than_source, -o.severity, o.cue))
    return out


#: Koliko najgorih cue-ova se ispisuje po imenu.
WORST_SHOWN = 5


def overflow_lines(found: list[Overflow], total: int, profile: Profile) -> list[str]:
    """Izveštaj za dnevnik: koliko, za koliko, koliko gore od originala, i
    najgorih pet sa rednim brojem."""
    if not found:
        return [
            f"prekoračenja: nijedno — svih {total} cue-ova je u granicama profila "
            f"{profile.name} ({profile.max_chars_per_line} zn./red, "
            f"{profile.max_cps:g} CPS)"
        ]
    by_chars = sum(1 for o in found if o.chars_over)
    by_cps = sum(1 for o in found if o.cps_over)
    by_lines = sum(1 for o in found if o.lines_over)
    head = (
        f"prekoračenja: {len(found)} od {total} cue-ova probija profil "
        f"{profile.name} — dužina reda {by_chars} "
        f"(>{profile.max_chars_per_line} zn., najviše +"
        f"{max(o.chars_over for o in found)}), CPS {by_cps} "
        f"(>{profile.max_cps:g}, najviše +{max(o.cps_over for o in found):.1f})"
    )
    if by_lines:
        head += f", broj redova {by_lines}"
    head += ". Ništa nije skraćeno."
    if all(o.source is None for o in found):
        return [head, "najgori: " + "; ".join(o.describe() for o in found[:WORST_SHOWN])]
    worse = [o for o in found if o.worse_than_source]
    out = [
        head,
        f"od toga GORE NEGO ISTI CUE U ORIGINALU: {len(worse)} — tu skraćuj; "
        f"kao u originalu (isto toliko dugo ili brzo): {len(found) - len(worse)}",
    ]
    if worse:
        out.append("najgori od njih: " + "; ".join(o.describe() for o in worse[:WORST_SHOWN]))
    return out
