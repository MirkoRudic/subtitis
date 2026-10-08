"""Da li je „prevod" zapravo original — i koliko cue-ova to pogađa.

Validacija odgovora (`translate.validate`) proverava samo redne brojeve. Model
ume da vrati izvorni tekst, ceo ili delimično prepravljen, pod ispravnim
brojem — viđeno 2026-09-25 na sr->fr: rečenica koja počinje francuskom reči
pa nastavlja srpski („Cela znači da…") prošla je kao prevod. Ovaj modul to hvata POSLE prevoda i prijavljuje glasno; tekst ne
menja.

Dve grube mere, svaka dovoljna sama:

* slova izvornog jezika kojih u ciljnom nema (`ć đ š ž č` u francuskom), u
  rečima sa MALIM početnim slovom — imena (`Petrović`, `Radionica otvorenih vrata`) su legitimna
  i bila su jedini lažni pogoci pre ovog izuzetka;
* poklapanje reči sa izvornom rečenicom >= `MAX_SIMILARITY`, osim kad su sve
  reči imena (`Vesna Kostić.`).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

#: Slova specifična za jezik. „Strano" je ono što izvor ima a cilj nema.
SPECIAL_LETTERS: dict[str, str] = {
    "sr": "čćđšž",
    "fr": "àâæçéèêëîïôœùûüÿ",
    "de": "äöüß",
    "es": "áéíñóúü",
    "it": "àèéìíîòóùú",
    "en": "",
}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_WORD = re.compile(r"[^\W\d_]+")

#: Udeo reči (bez imena) sa slovima izvornog jezika od kog je rečenica sumnjiva.
MIN_FOREIGN_SHARE = 0.05

#: Sličnost sa originalom (po rečima) od koje prevod nije prevod.
MAX_SIMILARITY = 0.8

TRANSLATION_FAILED_FLAG = "translation_failed"


def _lang(code: str | None) -> str:
    return (code or "").split("-")[0].lower()


def foreign_letters(src_lang: str, tgt_lang: str) -> set[str]:
    src = set(SPECIAL_LETTERS.get(_lang(src_lang), ""))
    tgt = set(SPECIAL_LETTERS.get(_lang(tgt_lang), ""))
    return src - tgt


def foreign_share(text: str, src_lang: str, tgt_lang: str) -> float:
    """Udeo reči sa malim početnim slovom koje sadrže slova izvornog jezika."""
    letters = foreign_letters(src_lang, tgt_lang)
    cyr_foreign = _lang(tgt_lang) != "sr"
    words = [w for w in _WORD.findall(text) if not w[0].isupper()]
    if not words:
        return 0.0
    bad = sum(
        1 for w in words
        if (set(w.lower()) & letters) or (cyr_foreign and _CYRILLIC.search(w))
    )
    return bad / len(words)


def similarity(src: str, tgt: str) -> float:
    a = [w.lower() for w in _WORD.findall(src)]
    b = [w.lower() for w in _WORD.findall(tgt)]
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(a=a, b=b, autojunk=False).ratio()


def _only_names(text: str) -> bool:
    words = _WORD.findall(text)
    return bool(words) and all(w[0].isupper() for w in words)


@dataclass
class Untranslated:
    sentence: int
    cues: list[int]  # od 1; prazno za posao bez cue-ova
    reason: str  # "translation_failed" | "slova izvora" | "isto kao original"

    def as_dict(self) -> dict[str, Any]:
        return {"sentence": self.sentence, "cues": self.cues, "reason": self.reason}


@dataclass
class Report:
    failed: list[Untranslated] = field(default_factory=list)
    #: nisu označene, a liče na original — model je vratio izvorni tekst
    suspect: list[Untranslated] = field(default_factory=list)

    @property
    def all(self) -> list[Untranslated]:
        return self.failed + self.suspect

    @property
    def cue_count(self) -> int:
        return len({c for u in self.all for c in u.cues})

    def as_dict(self) -> dict[str, Any]:
        return {
            "failed": [u.as_dict() for u in self.failed],
            "suspect": [u.as_dict() for u in self.suspect],
            "cues": self.cue_count,
        }


def check(
    source: list[Any], target: list[Any], src_lang: str, tgt_lang: str,
    cue_map: dict[int, list[int]] | None = None,
) -> Report:
    """Rečenice cilja koje su ostale (ili liče na) original.

    `cue_map` (id rečenice -> cue-ovi) je za posao bez `cue_parts`; inače se
    cue-ovi čitaju iz rečenice.
    """
    src_by_id = {s.id: s for s in source}
    report = Report()
    for s in target:
        cues = [p.cue + 1 for p in getattr(s, "cue_parts", [])] or (cue_map or {}).get(s.id, [])
        if TRANSLATION_FAILED_FLAG in s.flags:
            report.failed.append(Untranslated(s.id, cues, TRANSLATION_FAILED_FLAG))
            continue
        src = src_by_id.get(s.id)
        if src is None:
            continue
        if foreign_share(s.text, src_lang, tgt_lang) >= MIN_FOREIGN_SHARE:
            report.suspect.append(Untranslated(s.id, cues, "slova izvora"))
        elif not _only_names(s.text) and similarity(src.text, s.text) >= MAX_SIMILARITY:
            report.suspect.append(Untranslated(s.id, cues, "isto kao original"))
    return report


def plural(n: int, one: str, few: str, many: str) -> str:
    """Srpska množina: 1 rečenica, 2 rečenice, 5 rečenica, 11 rečenica, 21 rečenica."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def cue_ranges(cues: list[int]) -> str:
    """[1, 89, 168, 169, 170] -> `1, 89, 168–170`."""
    nums = sorted(set(cues))
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if i == j else f"{nums[i]}–{nums[j]}")
        i = j + 1
    return ", ".join(parts)


NOT_TRANSLATED = ("neprevedena", "neprevedene", "neprevedeno")
LOOKS_ORIGINAL = ("liči na original", "liče na original", "liči na original")


def _group(items: list[Untranslated], what: tuple[str, str, str]) -> str:
    n = len(items)
    head = f"{n} {plural(n, 'rečenica', 'rečenice', 'rečenica')} {plural(n, *what)}"
    cues = [c for u in items for c in u.cues]
    if cues:
        k = len(set(cues))
        head += f" — {k} {plural(k, 'cue', 'cue-a', 'cue-ova')} ({cue_ranges(cues)})"
    else:
        head += f" (rečenice {', '.join(str(u.sentence) for u in items)})"
    return head


def lines(report: Report, language: str) -> list[str]:
    """Glasne poruke za dnevnik. Prazno kad je sve prevedeno."""
    out: list[str] = []
    if report.failed:
        out.append(
            f"UPOZORENJE: {_group(report.failed, NOT_TRANSLATED)} na {language} — "
            "model je odustao, u titlu je ORIGINAL."
        )
    if report.suspect:
        out.append(
            f"UPOZORENJE: {_group(report.suspect, LOOKS_ORIGINAL)} na "
            f"{language} — model je vratio izvorni tekst pod ispravnim brojem. "
            "Proveri i prevedi ručno."
        )
    return out
