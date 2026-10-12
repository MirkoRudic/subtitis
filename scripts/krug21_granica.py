"""Krug 21: da li je 20 znakova u jednorednom profilu pretesno — merenje.

    .venv\\Scripts\\python scripts\\krug21_granica.py

Varijante profila `web` (prelamanje je isto kao `digital`): jedan red od 20
(sadašnje), 24, 28 i 32 znaka, i 28 u dva reda. Donja granica reda
(`min_chars_per_line`) skalira se srazmerno: 10 na 20 znakova. Sve ostalo
(`max_cps`, trajanja) ostaje kao u `web`.

Meri na svim srpskim ASR poslovima: cue-ove, rezove iza predloga, ispred
enklitike, iza i ispred veznika, cue-ove ispod donje granice, prekoračenja
CPS-a, prosečno i najduže trajanje cue-a, i koliko rečenica staje u JEDAN
cue bez ijednog preloma. Ništa se ne menja u kodu ni u poslovima.

Na terminal idu samo brojevi. Odlomak za poređenje izgleda (tekst stvarnog
govora) ide u `merenja/krug21/` — u .gitignore.
"""
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug20_veznici import VEZNICI, rezovi, ucitaj  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402

OUT = ROOT / "merenja" / "krug21"  # u .gitignore
#: (oznaka, znakova po redu, redova)
GRANICE = (("20", 20, 1), ("24", 24, 1), ("28", 28, 1), ("32", 32, 1), ("28×2", 28, 2))
#: Posao i rečenice za odlomak: isti u svim varijantama.
ODLOMAK = ("Sumice", 6)


def profil(chars, lines):
    web = cfg.load_profile("web")
    return web.model_copy(update={
        "name": f"web-{chars}x{lines}", "max_chars_per_line": chars, "max_lines": lines,
        "min_chars_per_line": round(web.min_chars_per_line * chars / web.max_chars_per_line),
    })


def meri(data, prof):
    rules = lb.rules_for("sr", lambda m: None)
    floor = prof.min_chars_per_line
    t = Counter()
    trajanja = []
    for name, sents, terms in data:
        cues_all = []
        for sid, sw in sents:
            cues = lb.layout_sentence(sw, prof, sid, [], rules, terms)
            t["rečenica"] += 1
            if len(cues) == 1 and len(cues[0].lines) == 1:
                t["cela u jednom cue-u"] += 1
            if len(cues) == 1:
                t["u jednom cue-u"] += 1
            for k in rezovi(sw, cues):
                left, right = lb._norm(sw[k].w), lb._norm(sw[k + 1].w)
                t["iza predloga"] += left in lb.SR_PREPOSITIONS
                t["ispred enklitike"] += right in lb.SR_CLITICS
                t["iza veznika"] += left in VEZNICI
                t["ispred veznika"] += right in VEZNICI
            if len(cues) > 1:
                t["ispod granice"] += sum(
                    1 for c in cues for line in c.lines if len(line) < floor)
            cues_all += cues
        timed = lb.apply_timing([c.model_copy(deep=True) for c in cues_all], prof)
        t["cue-ova"] += len(timed)
        t["preko CPS"] += sum(1 for c in timed if c.e > c.s
                              and sum(len(l) for l in c.lines) / (c.e - c.s) > prof.max_cps + 1e-9)
        trajanja += [c.e - c.s for c in timed]
    return t, statistics.mean(trajanja), max(trajanja)


def odlomak(data, prof):
    rules = lb.rules_for("sr", lambda m: None)
    for name, sents, terms in data:
        if ODLOMAK[0] in name:
            out = []
            for sid, sw in sents[: ODLOMAK[1]]:
                for c in lb.layout_sentence(sw, prof, sid, [], rules, terms):
                    out.append(f"{c.s:6.2f}–{c.e:6.2f}  " + "\n                ".join(c.lines))
                out.append("")
            return out
    return []


def main():
    data = ucitaj()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(data)} srpskih ASR poslova\n")
    print("granica | donja | cue-ova | cela rečenica u 1 red | u 1 cue (i sa 2 reda) | iza predloga | ispred enklitike | "
          "iza / ispred veznika | ispod donje | preko CPS | trajanje prosek / najduže")
    tekst = []
    for oznaka, chars, lines in GRANICE:
        prof = profil(chars, lines)
        t, avg, mx = meri(data, prof)
        print(f"{oznaka:>7} | {prof.min_chars_per_line:>5} | {t['cue-ova']:>7} | "
              f"{t['cela u jednom cue-u']:>3} od {t['rečenica']} ({t['cela u jednom cue-u'] / t['rečenica']:.0%}) | "
              f"{t['u jednom cue-u']:>3} | "
              f"{t['iza predloga']:>3} | {t['ispred enklitike']:>3} | {t['iza veznika']:>3} / {t['ispred veznika']:<3} | "
              f"{t['ispod granice']:>3} | {t['preko CPS']:>3} | {avg:.2f} / {mx:.2f} s")
        tekst += [f"===== {oznaka} ({chars} zn., {lines} red{'a' if lines > 1 else ''}, donja {prof.min_chars_per_line})", ""]
        tekst += odlomak(data, prof)
    (OUT / "odlomak.txt").write_text("\n".join(tekst), encoding="utf-8")


if __name__ == "__main__":
    main()
