"""Krug 7, celina 1: ravnomerna podela rečenice u jednorednom profilu.

    .venv\Scripts\python scripts\krug7_ravnomerno.py ime > merenja\krug7\ime.log

Profil se u ovom krugu još zove `digital` (20 zn., jedan red). Za svaku
vrednost `PENALTY_CUE_SHARE_DEVIATION` (ako je kod ima) meri na svim ASR
poslovima isto što i `krug6_digital.py`, plus:
  - koliko je poslednjih komada rečenice jedna reč (rečenica sa više cue-ova);
  - cue-ove Letnje škole, za poređenje red po red.

Prelamanje je determinističko — jedan prolaz je merenje. Sirovi izlaz (tekst
pravog govora) ide u `merenja/krug7/`, u .gitignore.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug6_digital import jobs, layout_job  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402

OUT = ROOT / "merenja" / "krug7"  # u .gitignore
OUT.mkdir(parents=True, exist_ok=True)
SINTAGMA = 200
PRIMER = "Letnja"  # projekat čiji se cue-ovi ispisuju za poređenje


def measure(prof):
    floor = prof.min_chars_per_line
    t = dict(cues=0, under=0, under_long_sentence=0, cuts=0, sintagma=0,
             one_word_tail=0, short=0, cps=0)
    primer = []
    for d in jobs():
        rows, _, timed = layout_job(d, prof)
        # rečenica = uzastopni cue-ovi istog sentence_id
        by_sent = {}
        for r in rows:
            by_sent.setdefault(r["cue"].sentence_id, []).append(r)
        for group in by_sent.values():
            sent_len = group[0]["rest"]  # ostatak od prvog cue-a = cela rečenica
            for k, r in enumerate(group):
                text = " ".join(r["cue"].lines)
                if len(group) > 1 and len(text) < floor:
                    t["under"] += 1
                    if sent_len >= floor:
                        t["under_long_sentence"] += 1
            if len(group) > 1 and len(" ".join(group[-1]["cue"].lines).split()) == 1:
                t["one_word_tail"] += 1
        pens = [p for r in rows for p in r["pens"]]
        t["cues"] += len(rows)
        t["cuts"] += len(pens)
        t["sintagma"] += sum(1 for p in pens if p >= SINTAGMA)
        t["short"] += len(lb.too_short(timed, prof))
        t["cps"] += sum(
            1 for c in timed
            if c.e > c.s and sum(len(l) for l in c.lines) / (c.e - c.s) > prof.max_cps + 1e-9
        )
        if PRIMER in d.name:
            primer = [" ".join(c.lines) for c in timed]
    return t, primer


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "merenje"
    prof = cfg.load_profile("digital")
    weights = (0, 2) if hasattr(lb, "PENALTY_CUE_SHARE_DEVIATION") else (None,)
    print(f"{label}: profil {prof.name} ({prof.max_chars_per_line} zn., {prof.max_lines} red, "
          f"min {prof.min_chars_per_line}), {len(list(jobs()))} poslova\n")
    print("udeo | cue-ova | <10 zn. (rečenica >=10) | jedna reč na kraju | u sintagmi | <min trajanja | >20 CPS")
    primeri = {}
    for w in weights:
        if w is not None:
            lb.PENALTY_CUE_SHARE_DEVIATION = w
        t, primer = measure(prof)
        primeri[w] = primer
        print(f"{str(w):>4} | {t['cues']:>7} | {t['under']:>3} ({t['under_long_sentence']:>3}) | "
              f"{t['one_word_tail']:>3} | {t['sintagma']:>3} od {t['cuts']} ({t['sintagma'] / t['cuts']:.1%}) | "
              f"{t['short']:>3} | {t['cps']:>3} ({t['cps'] / t['cues']:.1%})")
    for w, m in primeri.items():
        print(f"\nLetnja škola, udeo {w}:")
        for k, line in enumerate(m, 1):
            print(f"  {k:>2} ({len(line):>2}) {line}")


if __name__ == "__main__":
    main()
