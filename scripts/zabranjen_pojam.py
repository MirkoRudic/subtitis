"""Unos za tests/zabranjeni_pojmovi.json — heš pojma, ne pojam.

    .venv\\Scripts\\python scripts\\zabranjen_pojam.py "Naziv organizacije"
    .venv\\Scripts\\python scripts\\zabranjen_pojam.py --osnova "prezim"

Bez `--osnova`: pojam se traži kao niz celih reči, u bilo kom pismu,
veličini slova, sa dijakritikom ili bez nje. Sa `--osnova`: kao početak
jedne reči (za padeže). Ispis ide u odgovarajući spisak u JSON-u; sam pojam
se nigde ne upisuje.

`reci()` je jedina normalizacija za obe provere javnog repoa (pojmovi i
četvorke iz izvornog materijala, scripts/izvorne_cetvorke.py) i za
tests/test_javni_repo.py.
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitis.translit import to_latin  # noqa: E402


def _bez_dijakritike(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.replace("đ", "d"))
    return "".join(c for c in s if not unicodedata.combining(c))


def reci(text: str) -> list[str]:
    """Latinica, mala slova, bez dijakritike (`đ` i `dj` → `d`), samo slova i cifre."""
    s = _bez_dijakritike(to_latin(text).casefold())
    return [r.replace("dj", "d") for r in re.findall(r"[^\W_]+", s)]


def hes(text: str) -> str:
    return hashlib.sha256(" ".join(reci(text)).encode("utf-8")).hexdigest()


def main() -> None:
    args = sys.argv[1:]
    osnova = "--osnova" in args
    for pojam in (a for a in args if a != "--osnova"):
        n = len(reci(pojam))
        if osnova:
            if n != 1:
                raise SystemExit(f"osnova mora biti jedna reč: {pojam!r}")
            print(f'osnova: {{"duzina": {len(reci(pojam)[0])}, "h": "{hes(pojam)}"}}')
        else:
            print(f'celo:   {{"reci": {n}, "h": "{hes(pojam)}"}}')


if __name__ == "__main__":
    main()
