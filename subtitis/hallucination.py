"""Reči bez zvuka iza sebe — izmišljene, a sigurne.

Whisper ume da vrati tekst koji nije izgovoren: na kraju snimka, preko muzike,
iz `hotwords` (glosar) ili iz titlova na kojima je učen („Sous-titrage ST'
501"). Verovatnoća to ne hvata — izmišljene reči umeju da imaju p 0,98–0,99.
Ono što ih izdaje je vreme: model ih vrati sve na istom trenutku, sa trajanjem
nula.

Pravilo je determinističko i bez modela: niz od `MIN_RUN_FOR_FLAG` ili više
uzastopnih reči kraćih od `ZERO_DURATION_S`. Interpunkcija je providna — ima
svoje tokene, često i nultog trajanja, a nije reč.

Izmereno 2026-10-01 (`merenja/krug5/`, dnevnik): šest urednih snimaka + Radionica draft 1,
francuska elizija spojena kao u produkciji. Niz ≥ 2: nijedna lažna uzbuna;
pogodci su Radionica draft 1 (glosar: `letnja radionica OKC Divčibare Srbija`),
snimak A (`titrage ST' 501`) i `na svijetu` iz `Top 10 najboljih uvijeka na
svijetu` (Radionica draft 1 bez glosara), koji niz ≥ 3 propušta. Ograničenje uzorka:
uredni snimci su francuski, bez `hotwords`.

Pravilo NE hvata izmišljen tekst sa razvučenim vremenima (Radionica draft 1: `Otvorena vrata`
preko džingla, `Otvorena` 2,34 s) — i uredan govor ima reči od 5 s, pa dužina
reči nije upotrebljiv drugi signal.
"""

from __future__ import annotations

from typing import Iterable

from .models import Sentence, Word

#: Ispod ovoga reč nema zvuk iza sebe.
ZERO_DURATION_S = 0.02

#: Koliko uzastopnih takvih reči čini niz. Pojedinačna nulta reč je normalna na
#: granici segmenta; 2 je izmereno bez lažnih uzbuna (vidi gore).
MIN_RUN_FOR_FLAG = 2

#: Oznaka rečenice. Zasebna od `low_confidence`: nisu ista pojava — izmišljene
#: reči imaju VISOKU verovatnoću.
HALLUCINATED_FLAG = "hallucinated"

#: Tekst uz rečenicu u editoru.
EDITOR_NOTE = (
    "Model je ove reči vratio bez zvuka iza njih. Gotovo sigurno su izmišljene "
    "— proveri pre nego što ih zadržiš."
)


def _is_punct(tok: str) -> bool:
    return not any(ch.isalnum() for ch in tok)


def zero_runs(
    words: Iterable[Word],
    *,
    zero: float = ZERO_DURATION_S,
    min_run: int = MIN_RUN_FOR_FLAG,
) -> list[list[Word]]:
    """Nizovi od `min_run`+ uzastopnih reči (interpunkcija providna) sa `e - s < zero`."""
    out: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if _is_punct(w.w):
            continue
        if w.e - w.s < zero:
            cur.append(w)
            continue
        if len(cur) >= min_run:
            out.append(cur)
        cur = []
    if len(cur) >= min_run:
        out.append(cur)
    return out


def word_key(w: Word) -> tuple[float, float, str]:
    """Identitet reči za bojenje posle ručne izmene: poravnanje zadrži `s`, `e`
    i tekst neizmenjene reči, a umetnutoj izmisli vreme — ne i isti tekst."""
    return (round(w.s, 3), round(w.e, 3), w.w.lower())


def flag_sentences(sentences: list[Sentence], runs: list[list[Word]]) -> list[int]:
    """Označi rečenice koje sadrže reč iz nekog niza. Vrati id-jeve označenih."""
    indices = {w.i for run in runs for w in run}
    flagged: list[int] = []
    for s in sentences:
        lo, hi = s.words
        if any(lo <= i <= hi for i in indices):
            if HALLUCINATED_FLAG not in s.flags:
                s.flags.append(HALLUCINATED_FLAG)
            flagged.append(s.id)
    return flagged


def run_line(run: list[Word]) -> str:
    return f"{run[0].s:.2f}–{run[-1].e:.2f} s: „{' '.join(w.w for w in run)}\""
