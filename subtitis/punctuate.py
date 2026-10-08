"""Restauracija interpunkcije i velikih slova preko LLM-a.

Kad ASR ne vrati interpunkciju, `segment.py` nema na čemu da lomi, a
`linebreak.py` ostaje bez glavnog signala za tačku preloma. Zaštitna mreža u
`segment.merge_short` sprečava najgori ishod, ali ne vraća strukturu — ovo je
pravo rešenje.

Model sme da doda SAMO znakove i velika slova. Sve ostalo je kvar, i hvata se
programski: niz reči pre i posle mora biti identičan, poređen bez interpunkcije
i bez razlike u veličini slova. Model povremeno pokuša da „popravi" formulaciju
— to se odbacuje, ne prihvata.

Vremena reči se NE diraju. Postojeći tokeni zadržavaju `s`, `e` i `p`; znakovi
se ubacuju kao novi tokeni koji uzimaju sićušan odsečak sa kraja prethodne reči,
isto kao u `asr._split_word_times`.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, NamedTuple

from .asr import PUNCT_MAX_SECONDS
from . import config as cfg
from .config import SubtitisError
from .models import Word
from .segment import SENTENCE_END_CHARS, tokenize_text
from .translate import LANGUAGE_NAMES_TARGET, CallStats, OllamaClient

# ==========================================================================
# Konstante za štelovanje
# ==========================================================================

#: Reči po zahtevu. Veće = manje poziva, ali veća šansa da model omane niz.
CHUNK_WORDS = 60

#: Koliko puta se pokušava POJEDINAČNI čank pre nego što se prepolovi.
MAX_ATTEMPTS_PER_CHUNK = 2

#: Ispod ovoliko reči čank se više ne polovi — ostaje original.
#:
#: Namerno visoko. Za razliku od prevoda, gde je rečenica samostalna celina,
#: odlomak od tri reči nema šta da kaže o interpunkciji: model ga punktira kao
#: da je cela rečenica i dobiješ „Moi je m!". Bolje ostaviti bez znakova nego
#: dodati pogrešne.
MIN_CHUNK_WORDS = 20

# --- Povlačenje na poslednji znak ------------------------------------------
#
# Model svaki čank punktira kao zaokruženu celinu: zatvori ga tačkom i sledeći
# počne velikim slovom. Granica čanka time postaje granica rečenice, na mestu
# koje sa jezikom nema veze.
#
# Izmereno na francuskom testu 2026-08-12: od 6 granica čanka, 4 su dobile lom
# koji tamo ne pripada (oblika `dans .` / `Les jours`, `je vous .` / `remercie`,
# `avec nous .` / `Le matin`). Broj rečenica je skočio 21 -> 28.
#
# Rešenje ne traži poverenje u model: od odgovora se prihvata samo deo do
# POSLEDNJEG rečeničnog znaka iza koga još ima reči, a rep se vraća u sledeći
# poziv, gde ima svoj pravi nastavak. Veštačka tačka na kraju čanka se tako
# uvek odbaci — jer je uvek na samom kraju.

#: Rep se ne vraća ako je veći od ovog udela čanka.
#:
#: Bez ovoga bi čank bez ijednog unutrašnjeg znaka vratio sam sebe u sledeći
#: poziv, i to bi se vrtelo. Kad prag pukne, izlaz se prihvata kakav jeste —
#: veštačka granica je manje zlo od petlje.
MAX_RETREAT_SHARE = 0.5

#: Da li dodavanje dijakritike (`tres` -> `très`) računati kao izmenu reči.
#:
#: Strogo po pravilu „ne ispravljaj pravopis" — da, to jeste izmena. Ali ASR za
#: francuski ume da izostavi akcente, pa strogost obori skoro svaki čank.
#: Uključi ako meriš da je dobitak veći od rizika; podrazumevano je strogo.
ALLOW_DIACRITIC_CHANGES = False

#: Da li dozvoliti da model razdvoji jednu reč na više (`ya` -> `y a`).
#:
#: ASR ume da spoji dve reči u jednu; model to razdvaja i time je u pravu, ali
#: po strogoj proveri to jeste promena niza reči, pa ceo čank padne. Provera je
#: ista kao za eliziju, samo obrnuta: spojeni oblik mora se poklopiti sa
#: nadovezanim razdvojenim (`ya` == `y` + `a`). Nagađanja nema.
#:
#: UKLJUČENO 2026-08-12, po merenju: od 8 odbijenih čankova na francuskom testu,
#: 4 su padala samo zbog ovoga (`s'il ya` -> `s'il y a`). Razdvajanje je
#: pravopisna ispravka, ne izmena sadržaja — a provera je mehanička.
ALLOW_WORD_SPLITS = True

#: Da li dozvoliti izmenu nastavka (`généreuse` -> `généreuses`).
#:
#: Model ispravlja slaganje roda i broja. To je gramatika, a pravilo kaže da se
#: gramatika ne dira — ali je i najčešći razlog odbijanja na francuskom. Ovde
#: se dozvoljava SAMO razlika na kraju reči: zajednički prefiks od bar
#: `INFLECTION_MIN_STEM` znakova i najviše `INFLECTION_MAX_SUFFIX` znakova
#: razlike. `parle` -> `parlait` prolazi, `chien` -> `chat` ne.
#:
#: UKLJUČENO 2026-08-12: preostala 4 od 8 odbijenih čankova padala su samo zbog
#: ovoga (`généreuse` -> `généreuses`). Sa obe dozvole je prošlo 8/8.
#:
#: `ALLOW_DIACRITIC_CHANGES` ostaje isključen — isto merenje pokazuje da nije
#: bio potreban nijednom od tih čankova.
ALLOW_INFLECTION_CHANGES = True

INFLECTION_MIN_STEM = 3
INFLECTION_MAX_SUFFIX = 2

#: Znakovi koje model sme da doda. Sve van ovoga se briše pri primeni.
ALLOWED_PUNCTUATION = frozenset(".,!?…:;")

#: Flag na rečenici čiji deo nije uspeo da dobije interpunkciju.
PUNCTUATION_FAILED_FLAG = "punctuation_failed"


SYSTEM_PROMPT = """Ti si alat za restauraciju interpunkcije u {LANG_NAME} tekstu.

PRAVILA:
1. Vrati ISTI niz reči, istim redom. Nijednu reč ne menjaj, ne dodaj i ne izbaci.
2. Dodaj samo interpunkciju (. , ! ? … : ;) i velika slova.
3. Ne prevodi, ne parafraziraj, ne ispravljaj gramatiku ni pravopis.
   Ako je reč pogrešno prepoznata, ostavi je pogrešnu.
4. Ne dodaj objašnjenja, uvod ni zaključak. Odgovor je samo obrađeni tekst.
5. Ovo je transkript govora. Rečenice su često nepotpune — to je u redu."""


def build_system_prompt(language: str) -> str:
    return SYSTEM_PROMPT.format(
        LANG_NAME=LANGUAGE_NAMES_TARGET.get(language, language)
    )


# ==========================================================================
# Validacija — model sme da doda znakove, ne da menja reči
# ==========================================================================


def normalize_token(tok: str, allow_diacritics: bool | None = None) -> str:
    """Reč svedena na slova i cifre: `L'ordinateur,` -> `lordinateur`."""
    out = "".join(c for c in tok if c.isalnum()).casefold()
    if allow_diacritics is None:
        allow_diacritics = ALLOW_DIACRITIC_CHANGES
    if allow_diacritics:
        decomposed = unicodedata.normalize("NFKD", out)
        out = "".join(c for c in decomposed if not unicodedata.combining(c))
    return out


def word_sequence(text_or_tokens: str | list[str]) -> list[str]:
    toks = (
        tokenize_text(text_or_tokens)
        if isinstance(text_or_tokens, str)
        else list(text_or_tokens)
    )
    return [n for n in (normalize_token(t) for t in toks) if n]


@dataclass(frozen=True)
class Rejection:
    """Zašto je čank odbijen.

    `signature` je poenta: isti uzrok mora dati ISTI potpis bez obzira na
    veličinu čanka, da bi se prepoznalo da polovljenje ne pomaže. Zato je za
    manjak/višak reči potpis razlika (`count:+1`), a ne apsolutni brojevi.
    """

    kind: str       # "empty" | "count" | "word"
    signature: str
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True)
class Concession:
    """Izmena koju je propustila neka `ALLOW_*` dozvola.

    Beleži se i kad je dozvola upaljena — to je jedini način da se vidi da li
    model pod okriljem dozvole radi nešto ZNAČENJSKO umesto pravopisnog.
    """

    kind: str   # "split" | "inflection"
    before: str
    after: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.before!r} -> {self.after!r}"


class Aligned(NamedTuple):
    owner: list[int]
    concessions: list[Concession]


class Accepted(NamedTuple):
    tokens: list[str]
    owner: list[int]
    concessions: list[Concession]


def same_inflection(a: str, b: str) -> bool:
    """`généreuse`/`généreuses` — da. `chien`/`chat` — ne."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) < INFLECTION_MIN_STEM or not long.startswith(short):
        return False
    return len(long) - len(short) <= INFLECTION_MAX_SUFFIX


def align(
    want: list[str],
    got: list[str],
    *,
    allow_splits: bool,
    allow_inflection: bool,
) -> tuple[Aligned | None, Rejection | None]:
    """Poveži svaku dobijenu reč sa rednim brojem originalne.

    Vrati (`Aligned`, None) gde je `owner[j]` indeks originalne reči kojoj
    pripada `got[j]`, ili (None, razlog). Bez ijedne dozvole ovo je obično
    poređenje jedan-na-jedan i `concessions` je prazan.
    """
    owner: list[int] = []
    concessions: list[Concession] = []
    i = j = 0
    while i < len(want) and j < len(got):
        a, b = want[i], got[j]
        if a == b:
            owner.append(i)
            i += 1
            j += 1
            continue
        if allow_inflection and same_inflection(a, b):
            concessions.append(Concession("inflection", a, b))
            owner.append(i)
            i += 1
            j += 1
            continue

        # `ya` -> `y` + `a`: skupljaj dobijene reči dok se ne poklope sa jednom
        # originalnom. Prekida se čim prestane da bude prefiks — bez nagađanja.
        if allow_splits and b and a.startswith(b):
            merged, k = b, j + 1
            while k < len(got) and len(merged) < len(a) and a.startswith(merged + got[k]):
                merged += got[k]
                k += 1
            if merged == a and k > j + 1:
                concessions.append(Concession("split", a, " ".join(got[j:k])))
                owner.extend([i] * (k - j))
                i += 1
                j = k
                continue

        return None, Rejection(
            "word",
            f"word:{a}->{b}",
            f"reč {i + 1} promenjena: {a!r} -> {b!r}",
        )

    if i != len(want) or j != len(got):
        delta = len(got) - len(want)
        return None, Rejection(
            "count",
            f"count:{delta:+d}",
            f"broj reči: očekivano {len(want)}, dobijeno {len(got)}",
        )
    return Aligned(owner, concessions), None


def compare(
    want: list[str],
    got: list[str],
    *,
    allow_splits: bool = ALLOW_WORD_SPLITS,
    allow_inflection: bool = ALLOW_INFLECTION_CHANGES,
) -> tuple[Aligned | None, Rejection | None]:
    # Bez dozvole za razdvajanje, razlika u broju reči JESTE razlog odbijanja i
    # nema šta da se poravnava — poruka ostaje ista kao i pre.
    if not allow_splits and len(got) != len(want):
        delta = len(got) - len(want)
        return None, Rejection(
            "count",
            f"count:{delta:+d}",
            f"broj reči: očekivano {len(want)}, dobijeno {len(got)}",
        )
    return align(want, got, allow_splits=allow_splits, allow_inflection=allow_inflection)


def validate(
    response: str,
    original: list[str],
    *,
    allow_splits: bool = ALLOW_WORD_SPLITS,
    allow_inflection: bool = ALLOW_INFLECTION_CHANGES,
) -> tuple[Accepted | None, Rejection | None]:
    """Vrati (`Accepted`, None) ili (None, razlog)."""
    text = response.strip()
    if not text:
        return None, Rejection("empty", "empty", "prazan odgovor")

    produced = tokenize_text(text)
    got = word_sequence(produced)
    want = word_sequence(original)

    aligned, rej = compare(
        want, got, allow_splits=allow_splits, allow_inflection=allow_inflection
    )
    if aligned is None:
        return None, rej
    return Accepted(produced, aligned.owner, aligned.concessions), None


def would_pass(response: str, original: list[str]) -> dict[str, bool]:
    """Da li bi čank prošao sa svakom od dozvola. Za merenje, ne za odluku.

    Postoji da bi se `ALLOW_*` uključivalo po izmerenom, a ne po utisku.
    """
    out: dict[str, bool] = {}
    for name, kw in (
        ("ALLOW_WORD_SPLITS", {"allow_splits": True}),
        ("ALLOW_INFLECTION_CHANGES", {"allow_inflection": True}),
        ("oba", {"allow_splits": True, "allow_inflection": True}),
    ):
        ok, _ = validate(
            response, original,
            allow_splits=kw.get("allow_splits", False),
            allow_inflection=kw.get("allow_inflection", False),
        )
        out[name] = ok is not None
    return out


# ==========================================================================
# Povlačenje na poslednji znak
# ==========================================================================


def _is_sentence_end(tok: str) -> bool:
    t = tok.strip()
    return bool(t) and all(ch in SENTENCE_END_CHARS for ch in t)


def starts_new_sentence(out: list[Word]) -> bool:
    """Da li bi sledeći čank počeo na granici rečenice."""
    for w in reversed(out):
        if w.w.strip():
            return _is_sentence_end(w.w)
    return True  # ništa još nije upisano — to JESTE početak


def restore_lowercase(produced: list[str], originals: list[str]) -> list[str]:
    """Vrati malo slovo prvoj reči čanka koji ne počinje na granici rečenice.

    Povlačenje uklanja veštačku TAČKU sa kraja čanka, ali model i dalje piše
    prvu reč sledećeg čanka velikim slovom — pa se u sredini rečenice nađe
    `refaire Ce genre`.

    Menja se samo ako je original bio mali. `Paul`, `Serbie` i `France` su u
    `words.json` već veliki i ostaju veliki; ovo im ne može ništa.
    """
    if not originals:
        return produced
    for k, tok in enumerate(produced):
        if not any(c.isalnum() for c in tok):
            continue
        if tok[:1].isupper() and originals[0][:1].islower():
            out = list(produced)
            out[k] = tok[:1].lower() + tok[1:]
            return out
        break
    return produced


def retreat_point(
    produced: list[str], owner: list[int]
) -> tuple[int, int] | None:
    """Nađi poslednji rečenični znak iza koga još ima reči.

    Vrati `(broj tokena koji se prihvataju, broj ORIGINALNIH reči koje pokrivaju)`
    ili `None` kad povlačenja nema — nema znaka, ili je jedini znak na samom
    kraju, pa nema šta da se vrati.
    """
    total_words = sum(1 for t in produced if any(c.isalnum() for c in t))
    if total_words == 0:
        return None

    # Traži se poslednji znak IZA KOGA JOŠ IMA REČI. Znak na samom kraju je
    # upravo veštačka granica čanka i njega preskačemo — zato uslov
    # `seen < total_words`, a ne „poslednji znak".
    seen = 0
    best: tuple[int, int] | None = None
    for k, tok in enumerate(produced):
        if any(c.isalnum() for c in tok):
            seen += 1
            continue
        if seen and seen < total_words and _is_sentence_end(tok):
            best = (k + 1, seen)

    if best is None:
        return None
    tokens, words = best
    return tokens, owner[words - 1] + 1


# ==========================================================================
# Primena na tokene sa vremenima
# ==========================================================================


def _share_times(src: Word, tokens: list[str]) -> list[Word]:
    """Podeli vreme jedne reči na više tokena, srazmerno dužini.

    Treba samo kad je `ALLOW_WORD_SPLITS` uključen i model razdvoji `ya` u
    `y a`. Ista računica kao `asr._split_word_times`.
    """
    if len(tokens) == 1:
        return [src.model_copy(update={"w": tokens[0]})]
    span = max(src.e - src.s, 0.0)
    total = sum(len(t) for t in tokens) or 1
    out: list[Word] = []
    pos = src.s
    for t in tokens:
        share = span * len(t) / total
        out.append(src.model_copy(update={"w": t, "s": round(pos, 3), "e": round(pos + share, 3)}))
        pos += share
    last = out[-1]
    out[-1] = last.model_copy(update={"e": round(max(src.e, last.s), 3)})
    return out


def apply_tokens(
    words: list[Word], produced: list[str], owner: list[int] | None = None
) -> list[Word]:
    """Spoji originalne reči (sa vremenima) i interpunkciju iz odgovora.

    Postojeći interpunkcijski tokeni originala se odbacuju — restauracija ih
    zamenjuje. Reči zadržavaju `s`, `e` i `p`; menja im se samo ispis, da bi
    velika slova prošla.

    `owner` je poravnanje iz `validate`: za svaku dobijenu reč indeks originalne
    kojoj pripada. Bez njega se podrazumeva jedan-na-jedan.
    """
    originals = [w for w in words if any(c.isalnum() for c in w.w)]
    out: list[Word] = []
    idx = 0        # dokle smo stigli kroz ORIGINALNE reči
    seen = 0       # dokle smo stigli kroz DOBIJENE reči (bez znakova)
    group: list[str] = []      # dobijene reči koje pripadaju istoj originalnoj
    group_owner: int | None = None

    def flush() -> None:
        """Upiši nakupljenu grupu kao jednu originalnu reč (ili njene delove)."""
        nonlocal idx, group, group_owner
        if not group:
            return
        out.extend(_share_times(originals[idx], group))
        idx += 1
        group = []
        group_owner = None

    for tok in produced:
        if any(c.isalnum() for c in tok):
            mine = owner[seen] if owner is not None else seen
            seen += 1
            if group and mine != group_owner:
                flush()
            if not group and idx >= len(originals):
                break
            group.append(tok)
            group_owner = mine
            continue

        # Znak zatvara grupu: razdvojena reč nikad nema znak u sredini.
        flush()
        cleaned = "".join(c for c in tok if c in ALLOWED_PUNCTUATION)
        if not cleaned or not out:
            continue
        prev = out[-1]
        slice_len = min(PUNCT_MAX_SECONDS, max(0.0, prev.e - prev.s) * 0.25)
        start = round(max(prev.s, prev.e - slice_len), 3)
        out[-1] = prev.model_copy(update={"e": start})
        out.append(
            Word(i=0, s=start, e=round(prev.e, 3), w=cleaned, p=prev.p)
        )
    flush()

    if idx != len(originals):  # pragma: no cover — validacija ovo već hvata
        raise SubtitisError(
            f"restauracija: potrošeno {idx} od {len(originals)} reči"
        )
    return out


# ==========================================================================
# Glavni tok
# ==========================================================================


@dataclass
class PunctuateStats:
    words_in: int = 0
    words_out: int = 0
    chunks_ok: int = 0
    chunks_failed: int = 0
    marks_added: int = 0
    failed_ranges: list[tuple[int, int]] = field(default_factory=list)
    calls: CallStats = field(default_factory=CallStats)

    #: Ukupno trajanje i deo potrošen na odgovore koji su bačeni.
    seconds_total: float = 0.0
    seconds_wasted: float = 0.0
    halvings: int = 0
    #: Koliko puta je polovljenje preskočeno jer je greška bila ista kao u
    #: nadređenom čanku.
    halvings_skipped: int = 0

    #: Koliko je puta rep vraćen u sledeći poziv, i koliko reči ukupno.
    #: `retreats_forced` su slučajevi gde je zaštita presekla povlačenje.
    retreats: int = 0
    retreat_words: int = 0
    retreats_forced: int = 0

    #: Koliko bi odbijenih čankova prošlo sa svakom `ALLOW_*` dozvolom.
    #: Meri se uvek, i dok su dozvole isključene — da se odluka donese po broju.
    would_pass: dict[str, int] = field(default_factory=dict)

    #: Šta su upaljene dozvole stvarno propustile, po vrsti i sa primerima.
    #: Ovo je druga strana istog merenja: kad je dozvola upaljena, u logu mora
    #: da se vidi svaka izmena koju je propustila — da se uhvati ako neka nije
    #: pravopisna nego značenjska.
    concessions: dict[str, int] = field(default_factory=dict)
    concession_examples: list[str] = field(default_factory=list)

    #: Koliko je puta vraćeno malo slovo na prvu reč čanka.
    decapitalized: int = 0

    @property
    def wasted_share(self) -> float:
        return self.seconds_wasted / self.seconds_total if self.seconds_total else 0.0

    def timing_line(self) -> str:
        return (
            f"interpunkcija: {self.seconds_total:.0f}s ukupno, "
            f"{self.seconds_wasted:.0f}s ({100 * self.wasted_share:.0f}%) na "
            f"odbijene odgovore; {self.calls.requests} zahteva "
            f"({self.chunks_ok} prošlo, {self.chunks_failed} odbijeno)"
        )

    def concessions_line(self) -> str | None:
        """Šta su dozvole propustile. Pročitaj ovo — tu se vidi zloupotreba."""
        if not self.concessions:
            return None
        counts = ", ".join(f"{k} {n}" for k, n in sorted(self.concessions.items()))
        primeri = "; ".join(self.concession_examples[:8])
        vise = (
            f" (+{len(self.concession_examples) - 8} drugih)"
            if len(self.concession_examples) > 8
            else ""
        )
        return f"interpunkcija: dozvole propustile — {counts}: {primeri}{vise}"

    def retreat_line(self) -> str:
        forced = (
            f", {self.retreats_forced} put(a) presečeno zaštitom"
            if self.retreats_forced
            else ""
        )
        veliko = (
            f"; vraćeno malo slovo na {self.decapitalized} mesta"
            if self.decapitalized
            else ""
        )
        return (
            f"interpunkcija: povlačenja {self.retreats}, "
            f"vraćeno {self.retreat_words} reči u sledeći poziv{forced}{veliko}"
        )

    def would_pass_line(self) -> str | None:
        """Šta bi se dobilo popuštanjem pravila. Merenje, ne preporuka."""
        if not self.chunks_failed:
            return None
        parts = ", ".join(
            f"{name} {n}/{self.chunks_failed}"
            for name, n in self.would_pass.items()
            if n
        )
        if not parts:
            return (
                f"interpunkcija: nijedno popuštanje ne bi spasilo nijedan od "
                f"{self.chunks_failed} odbijenih čankova"
            )
        return f"interpunkcija: odbijenih čankova bi prošlo — {parts}"


def punctuate_words(
    words: list[Word],
    language: str,
    *,
    client: OllamaClient,
    chunk_words: int = CHUNK_WORDS,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Word], PunctuateStats]:
    """Vrati nove reči sa interpunkcijom i statistiku.

    Čank koji ne prođe validaciju se polovi; kad ni pojedinačno ne prođe,
    ostaje ORIGINAL. Posao se zbog toga nikad ne prekida.

    Polovljenje se PRESKAČE kad podčank padne sa istim razlogom kao nadređeni.
    Model koji dosledno razdvaja `ya` u `y a` uradiće to i u polovini, i u
    četvrtini — svaki dalji pokušaj je samo potrošeno vreme.

    Od uspelog čanka se prihvata samo deo do poslednjeg rečeničnog znaka; rep
    se vraća u sledeći poziv (v. `MAX_RETREAT_SHARE`). Zato sledeći čank ne
    počinje na `lo + chunk_words` nego tamo gde je prethodni stvarno stao.
    """
    warn = log or cfg.echo
    stats = PunctuateStats(words_in=len(words))
    if not words:
        return words, stats

    system = build_system_prompt(language)
    before = CallStats(**vars(client.stats))
    started = time.monotonic()

    # Radimo samo nad rečima; postojeću interpunkciju restauracija zamenjuje.
    source = [w for w in words if any(c.isalnum() for c in w.w)]
    if not source:
        return words, stats

    result: list[Word] = []
    #: Odakle je već počinjao neki čank. Rep koji bi nas vratio na već viđen
    #: početak znači da nema napretka — tada se izlaz prihvata kakav jeste.
    seen_starts: set[int] = set()

    def accept_with_retreat(
        lo: int, hi: int, span: list[Word], produced: list[str], owner: list[int]
    ) -> int:
        """Prihvati deo do poslednjeg znaka. Vrati broj potrošenih reči."""
        whole = hi - lo
        point = None if hi >= len(source) else retreat_point(produced, owner)
        # Poslednji čank se ne povlači: tamo je tačka na kraju prava, ne
        # veštačka — iza nje stvarno nema ničega.

        if point is not None:
            tokens, taken = point
            tail = whole - taken
            if taken <= 0 or (lo + taken) in seen_starts:
                warn(
                    f"  -> rep bi vratio na već viđen početak {lo + taken}; "
                    "prihvatam čank kakav jeste"
                )
                stats.retreats_forced += 1
                point = None
            elif tail > whole * MAX_RETREAT_SHARE:
                warn(
                    f"  -> poslednji znak je posle {taken}. reči od {whole}; "
                    f"rep od {tail} reči je preko {MAX_RETREAT_SHARE:.0%} čanka, "
                    "prihvatam čank kakav jeste (granica ostaje veštačka)"
                )
                stats.retreats_forced += 1
                point = None

        if point is None:
            result.extend(apply_tokens(span, produced, owner))
            return whole

        tokens, taken = point
        result.extend(apply_tokens(span[:taken], produced[:tokens], owner))
        stats.retreats += 1
        stats.retreat_words += whole - taken
        return taken

    def handle(lo: int, hi: int, attempt: int = 0, parent: Rejection | None = None) -> int:
        """Obradi opseg. Vrati koliko je reči STVARNO potrošeno."""
        span = source[lo:hi]
        if not span:
            return 0
        text = " ".join(w.w for w in span)
        originals = [w.w for w in span]
        call_started = time.monotonic()
        try:
            response = client.chat(system, text)
            accepted, rej = validate(response, originals)
        except SubtitisError:
            raise  # veza ili model — nema smisla polovati
        elapsed = time.monotonic() - call_started

        if accepted is not None:
            stats.chunks_ok += 1
            produced = accepted.tokens
            # Dozvole su upaljene, pa se beleži ŠTA su propustile — da se u logu
            # vidi ako model pod njihovim okriljem promeni značenje.
            for c in accepted.concessions:
                stats.concessions[c.kind] = stats.concessions.get(c.kind, 0) + 1
                if str(c) not in stats.concession_examples:
                    stats.concession_examples.append(str(c))
            # Čank koji ne počinje na granici rečenice ne sme da dobije veliko
            # slovo — inače usred rečenice ostane `refaire Ce genre`.
            if not starts_new_sentence(result):
                fixed = restore_lowercase(produced, originals)
                if fixed is not produced:
                    stats.decapitalized += 1
                    produced = fixed
            return accept_with_retreat(lo, hi, span, produced, accepted.owner)

        stats.chunks_failed += 1
        stats.seconds_wasted += elapsed
        for name, ok in would_pass(response, originals).items():
            if ok:
                stats.would_pass[name] = stats.would_pass.get(name, 0) + 1
        warn(f"interpunkcija: čank reči {lo}-{hi} odbijen ({rej})")

        if attempt + 1 < MAX_ATTEMPTS_PER_CHUNK:
            return handle(lo, hi, attempt + 1, parent)

        # Ista greška kao u nadređenom čanku znači da model to radi dosledno.
        # Polovljenje je već pokazalo da ne pomaže — ne troši još poziva.
        if parent is not None and rej is not None and rej.signature == parent.signature:
            stats.halvings_skipped += 1
            warn(
                f"  -> ista greška kao u nadređenom čanku ({rej.signature}); "
                "polovljenje ne pomaže"
            )
        elif hi - lo > MIN_CHUNK_WORDS:
            mid = (lo + hi) // 2
            stats.halvings += 1
            warn(f"  -> polovim na {lo}-{mid} i {mid}-{hi}")
            # Druga polovina počinje tamo gde je prva stvarno stala — i unutar
            # polovljenja se rep vraća, pa `mid` nije pouzdan.
            first = handle(lo, mid, 0, rej)
            return first + handle(lo + first, hi, 0, rej)

        warn(f"  -> ODUSTAJEM za reči {lo}-{hi}, ostaje original")
        stats.failed_ranges.append((lo, hi))
        result.extend(span)
        return hi - lo

    lo = 0
    while lo < len(source):
        seen_starts.add(lo)
        hi = min(lo + chunk_words, len(source))
        used = handle(lo, hi)
        if used <= 0:  # pragma: no cover — zaštite iznad ovo već sprečavaju
            warn(f"  -> nema napretka na {lo}; prihvatam ceo čank")
            result.extend(source[lo:hi])
            used = hi - lo
        lo += used

    stats.seconds_total = time.monotonic() - started

    # Stabilni indeksi: rečenice se prave posle ovoga, pa je prenumeracija bezbedna.
    out = [w.model_copy(update={"i": k}) for k, w in enumerate(result)]
    stats.words_out = len(out)
    stats.marks_added = sum(
        1 for w in out if w.w.strip() and not any(c.isalnum() for c in w.w)
    )
    stats.calls = CallStats(
        requests=client.stats.requests - before.requests,
        prompt_tokens=client.stats.prompt_tokens - before.prompt_tokens,
        output_tokens=client.stats.output_tokens - before.output_tokens,
        thinking_responses=client.stats.thinking_responses - before.thinking_responses,
        eval_ns=client.stats.eval_ns - before.eval_ns,
        prompt_eval_ns=client.stats.prompt_eval_ns - before.prompt_eval_ns,
        load_ns=client.stats.load_ns - before.load_ns,
    )
    return out, stats
