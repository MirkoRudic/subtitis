"""Merenje uzastopnih ponavljanja reči i fraza u tekstu.

Postoji zbog jednog pitanja: kad govornik zamuca ("parce que, parce que"),
da li model to ponavljanje PRENOSI u prevod, UKLANJA ga ili DODAJE svoje.
Zato se meri i na originalu i na prevodu — pojedinačan broj ne znači ništa,
odnos znači.

Udvajanje nije jedna pojava nego dve, i ne smeju se sabirati:

  POJAČAVANJE   "très, très bien" -> "veoma, veoma dobro"
                nosi značenje, treba da se prenese, i kad se prenese to je
                tačan prevod a ne kvar;
  OKLEVANJE     "parce que, parce que", "et et", "qui me qui me"
                ne nosi značenje i po pravilu 9 iz prevodnog prompta treba
                da otpadne.

Prvi posao (fr->sr, 27 rečenica) je pokazao zašto podela mora da postoji: od
5 udvajanja u originalu dva su bila "très très". Sabrana zajedno, brojka je
rekla „preneo 2 od 5"; razdvojena, ispada da je od tri OKLEVANJA preneo jedno,
a oba POJAČAVANJA — ispravno — zadržao. Dva različita zaključka iz istog posla.

Razvrstavanje je grubo i namerno takvo: po spisku reči, bez ikakve analize
rečenice. Prilog stepena ("très", "veoma") je skoro uvek pojačavanje; veznik,
predlog ili zamenica ("et", "que", "jer", "i", "koji") su skoro uvek oklevanje.
Sve van ta dva spiska ide u NERAZVRSTANO i tamo se i vidi — to je važnije nego
da se nagađanjem ugura u jednu od dve kategorije. Za jezik koji nema spisak
(sve osim fr/sr/en) sve ispada nerazvrstano, što je pošteno: mera tad ne zna.

Meri se SAMO unutar jedne rečenice, i samo uzastopno. Ponavljanje preko
granice rečenice je treća pojava i u titlovima je normalno ("Da. Da. Da."),
zbog čega je `translate.REPEAT_PENALTY` namerno 1.0 — kad bi se i to brojalo,
mera bi merila stil dijaloga umesto govora.

Ovaj modul ništa ne menja i ni na šta ne utiče. Samo broji.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Reč je niz slovno-brojevnih znakova, sa apostrofima unutra. Apostrof mora
#: da ostane u reči: bez njega se francusko "c'est c'est" raspada na
#: c/est/c/est i udvojena REČ se pogrešno prijavi kao udvojena FRAZA.
WORD_RE = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)

#: Koliko primera se pamti po vrsti.
MAX_EXAMPLES = 8

# ==========================================================================
# Vrste udvajanja
# ==========================================================================

EMPHASIS = "pojačavanje"
HESITATION = "oklevanje"
UNCLASSIFIED = "nerazvrstano"

#: Prilozi stepena. Njihovo udvajanje je skoro uvek pojačavanje i treba da
#: pređe u prevod. Spisak je kratak namerno — dvosmislene reči (fr "bien",
#: "tout", "plus"; sr "tako") su izostavljene jer bi tiho premeštale slučajeve
#: iz nerazvrstanog u pogrešnu kategoriju, a nerazvrstano se bar vidi.
INTENSIFIERS: dict[str, set[str]] = {
    "fr": {
        "très", "trop", "tellement", "vraiment", "beaucoup", "super", "hyper",
        "extrêmement", "franchement",
    },
    "sr": {
        "veoma", "vrlo", "mnogo", "jako", "baš", "previše", "sasvim",
        "stvarno", "strašno", "užasno", "toliko", "krajnje",
    },
    "en": {
        "very", "really", "so", "too", "extremely", "super", "totally",
        "absolutely",
    },
}

#: Veznici, predlozi, zamenice i pomoćni glagoli — reči bez sopstvenog težišta.
#: Njihovo udvajanje je skoro uvek govorno oklevanje.
FUNCTION_WORDS: dict[str, set[str]] = {
    "fr": {
        "et", "ou", "mais", "donc", "car", "or", "ni",
        "que", "qui", "quoi", "dont", "parce", "puisque", "quand", "comme",
        "de", "du", "des", "à", "au", "aux", "en", "dans", "sur", "sous",
        "pour", "par", "avec", "sans", "chez", "vers", "entre",
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
        "me", "te", "se", "lui", "leur", "y", "moi", "toi",
        "le", "la", "les", "un", "une", "ce", "cet", "cette", "ces", "ça",
        "ne", "pas", "est", "sont", "a", "ai",
        "c'est", "j'ai", "qu'il", "qu'elle", "n'est", "d'un", "d'une", "l'on",
    },
    "sr": {
        "i", "a", "ali", "ili", "jer", "da", "pa", "te", "no", "nego", "niti",
        "kad", "kada", "ako", "dok", "što", "jel",
        "koji", "koja", "koje", "kojeg", "kome", "koju",
        "ja", "ti", "on", "ona", "ono", "mi", "vi", "oni", "one",
        "me", "se", "ga", "je", "su", "sam", "si", "smo", "ste", "će", "bi",
        "to", "ta", "taj", "te", "ovo", "ovaj", "ova", "ono", "onaj",
        "u", "na", "za", "sa", "s", "o", "po", "od", "do", "iz", "kroz",
        "pri", "uz", "niz", "pred", "kod", "bez", "prema", "ne",
    },
    "en": {
        "and", "or", "but", "because", "so", "that", "which", "who", "whom",
        "when", "while", "if", "as", "than",
        "the", "a", "an", "of", "to", "in", "on", "at", "for", "with", "by",
        "from", "into", "about", "over", "under",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
        "us", "them", "my", "your", "his", "their",
        "is", "are", "was", "were", "do", "did", "does", "not", "no",
    },
}


def known_language(lang: str | None) -> bool:
    """Ima li ovaj jezik spiskove. Ako nema, sve ispada nerazvrstano."""
    key = (lang or "").strip().lower()
    return key in INTENSIFIERS and key in FUNCTION_WORDS


def classify(toks: list[str], lang: str | None) -> str:
    """Vrsta udvajanja, po rečima koje se ponavljaju.

    Pojačavanje ima prednost: dovoljno je da jedan prilog stepena bude u
    udvojenom komadu ("so so", "très très"). Oklevanje traži da SVE reči budu
    funkcijske — "parce que" jeste, "moj brat" nije, i drugo ostaje
    nerazvrstano umesto da se natera u kategoriju.
    """
    key = (lang or "").strip().lower()
    if not known_language(key):
        return UNCLASSIFIED
    if any(t in INTENSIFIERS[key] for t in toks):
        return EMPHASIS
    if all(t in FUNCTION_WORDS[key] for t in toks):
        return HESITATION
    return UNCLASSIFIED


def tokens(text: str) -> list[str]:
    """Reči malim slovima, bez interpunkcije. Zarez u „jer, jer" se ignoriše."""
    return [w.casefold() for w in WORD_RE.findall(text)]


# ==========================================================================
# Merenje
# ==========================================================================


@dataclass(frozen=True)
class Hit:
    """Jedno uhvaćeno udvajanje."""

    text: str
    kind: str
    #: True za udvojenu dvočlanu frazu, False za udvojenu reč.
    phrase: bool = False


@dataclass
class Repeats:
    """Koliko uzastopnih udvajanja ima u pregledanom tekstu i kakvih.

    Brojevi su izvedeni iz `hits` — nema zasebnih brojača koji mogu da se
    raziđu sa spiskom.
    """

    hits: list[Hit] = field(default_factory=list)
    #: Ukupno reči — bez ovoga se dva teksta različite dužine ne porede.
    total_words: int = 0
    #: Koliko tekstova (rečenica) je pregledano.
    texts: int = 0
    #: Koliko ih ima bar jedno udvajanje, i posebno bar jedno OKLEVANJE.
    texts_with_repeat: int = 0
    texts_with_hesitation: int = 0

    # ---- ukupno ----

    @property
    def total(self) -> int:
        return len(self.hits)

    @property
    def words(self) -> int:
        """Udvojenih pojedinačnih reči."""
        return sum(1 for h in self.hits if not h.phrase)

    @property
    def phrases(self) -> int:
        """Udvojenih dvočlanih fraza."""
        return sum(1 for h in self.hits if h.phrase)

    # ---- po vrsti ----

    def of_kind(self, kind: str) -> list[Hit]:
        return [h for h in self.hits if h.kind == kind]

    @property
    def hesitation(self) -> int:
        return len(self.of_kind(HESITATION))

    @property
    def emphasis(self) -> int:
        return len(self.of_kind(EMPHASIS))

    @property
    def unclassified(self) -> int:
        return len(self.of_kind(UNCLASSIFIED))

    @property
    def per_100_words(self) -> float:
        """Gustina OKLEVANJA. Pojačavanje nije kvar, pa se ovde ne broji."""
        return 100.0 * self.hesitation / self.total_words if self.total_words else 0.0

    def __add__(self, other: "Repeats") -> "Repeats":
        return Repeats(
            hits=self.hits + other.hits,
            total_words=self.total_words + other.total_words,
            texts=self.texts + other.texts,
            texts_with_repeat=self.texts_with_repeat + other.texts_with_repeat,
            texts_with_hesitation=(
                self.texts_with_hesitation + other.texts_with_hesitation
            ),
        )

    # ---- ispis ----

    def summary(self) -> str:
        """Jedan red. Oklevanje ide prvo jer je ono predmet pravila 9."""
        return (
            f"oklevanje {self.hesitation}, pojačavanje {self.emphasis}, "
            f"nerazvrstano {self.unclassified} "
            f"(u {self.texts_with_hesitation}/{self.texts} rečenica oklevanje, "
            f"{self.per_100_words:.2f}/100 reči)"
        )

    def examples_line(self, kind: str | None = None, limit: int = 3) -> str:
        """Primeri, po vrsti ili svi. Broj kaže koliko, primer kaže šta."""
        hits = self.of_kind(kind) if kind else self.hits
        shown = hits[:limit]
        tail = f" (+{len(hits) - len(shown)})" if len(hits) > len(shown) else ""
        if kind:
            return ", ".join(f'"{h.text}"' for h in shown) + tail
        return ", ".join(f'"{h.text}" [{h.kind}]' for h in shown) + tail


def _word_runs(toks: list[str], lang: str | None) -> list[Hit]:
    """Udvojene reči. Niz od n istih reči daje n-1 nalaza."""
    return [
        Hit(f"{toks[k]} {toks[k + 1]}", classify([toks[k]], lang), phrase=False)
        for k in range(len(toks) - 1)
        if toks[k] == toks[k + 1]
    ]


def _phrase_runs(toks: list[str], lang: str | None) -> list[Hit]:
    """Udvojene dvočlane fraze.

    Fraza od dve ISTE reči ("a a a a") se preskače — nju je već prebrojao
    `_word_runs`, pa bi se inače isto mucanje računalo dvaput.
    """
    out: list[Hit] = []
    k = 0
    while k + 3 < len(toks):
        a, b = toks[k], toks[k + 1]
        if a != b and (a, b) == (toks[k + 2], toks[k + 3]):
            out.append(Hit(f"{a} {b} {a} {b}", classify([a, b], lang), phrase=True))
            # Preskoči ceo uhvaćen komad: bez toga bi "x y x y x y" dalo tri
            # nalaza za dva ponavljanja, jer se prozori preklapaju.
            k += 4
            continue
        k += 1
    return out


def measure(text: str, lang: str | None = None) -> Repeats:
    """Izmeri jedan tekst (jednu rečenicu).

    `lang` je kod jezika TOG teksta ('fr', 'sr', 'en'). Bez njega se ništa ne
    razvrstava — mera i dalje broji, ali sve ide u nerazvrstano.
    """
    toks = tokens(text)
    hits = _word_runs(toks, lang) + _phrase_runs(toks, lang)
    return Repeats(
        hits=hits,
        total_words=len(toks),
        texts=1,
        texts_with_repeat=1 if hits else 0,
        texts_with_hesitation=1 if any(h.kind == HESITATION for h in hits) else 0,
    )


def measure_all(texts: list[str], lang: str | None = None) -> Repeats:
    """Izmeri više rečenica. Granica rečenice se NE prelazi."""
    total = Repeats()
    for t in texts:
        total = total + measure(t, lang)
    return total


def compare_line(
    src: Repeats,
    tgt: Repeats,
    src_label: str = "original",
    tgt_label: str = "prevod",
) -> str:
    """Red koji odgovara na pitanje: prenosi, uklanja ili dodaje.

    Sudi se po OKLEVANJU, ne po zbiru: preneto pojačavanje je tačan prevod i
    kad bi ulazilo u ocenu, model bi ispao lošiji upravo zato što je dobro
    preveo. Odnos ide na broju, ne na gustini — prevod je po pravilu kraći od
    originala, pa bi gustina prijavila „dodao" i kad model nije dodao ništa.
    """
    s, t = src.hesitation, tgt.hesitation
    extra = ""
    if src.emphasis or tgt.emphasis:
        extra = f"; pojačavanje {src.emphasis} -> {tgt.emphasis} (ne računa se)"
    if not s and not t:
        return (
            f"oklevanje: nijedno ni u {src_label} ni u {tgt_label}{extra}"
        )
    if not s:
        return f"oklevanje: {src_label} 0 -> {tgt_label} {t} (model ga DODAJE){extra}"
    share = t / s
    verdict = (
        "uklanja ga" if share <= 0.25
        else "uglavnom uklanja" if share < 0.75
        else "PRENOSI ga" if share <= 1.25
        else "dodaje svoje"
    )
    return (
        f"oklevanje: {src_label} {s} -> {tgt_label} {t} "
        f"({share:.2f}x, {verdict}){extra}"
    )
