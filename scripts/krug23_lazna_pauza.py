"""Krug 23: pauza iza reči na kojoj natpis ne sme da se završi.

    .venv\\Scripts\\python scripts\\krug23_lazna_pauza.py          # koliko ih ima
    .venv\\Scripts\\python scripts\\krug23_lazna_pauza.py meri     # mreža

Bez argumenta: koliko pauza (pripisanih granici, `pauses.boundary_pauses`)
pada iza predloga, enklitike, negacije, veznika ili `a`, odnosno ispred
enklitike — `linebreak.false_pause_at`. Sa `meri`: mreža kruga 19/22 u tri
načina (`linebreak.PAUSE_AT_FUNCTION_WORD`: keep, drop, move).

Merilo rezultata od kruga 23 (dnevnik, 2026-10-13): natpis sadrži pauzu ako
se tišina iz zvuka preklapa sa rasponom govora natpisa (početak prve do kraja
poslednje reči) bar za prag. Ne zavisi od pripisivanja tišine granici — a to
je baš ono što `move` menja. Stara mera (pripisana granica unutar natpisa)
ispisuje se kao dopunska.

Promene idu u `merenja/krug23/` — u .gitignore. Na terminal idu samo brojevi.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug19_pauze import granice_cue  # noqa: E402
from krug22_pauze import CUVARI, PRAGOVI, tacke, ucitaj  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402
from subtitis import pauses as pz  # noqa: E402

OUT = ROOT / "merenja" / "krug23"  # u .gitignore
NACINI = ("keep", "drop", "move")


def u_govoru(c, sil, thr):
    """Merilo: tišina se preklapa sa govorom natpisa bar `thr` sekundi."""
    return any(min(e, c.e) - max(s, c.s) >= thr for s, e in sil)


def klasa(sw, k, rules):
    w = lb._norm(sw[k].w)
    if w in lb.SR_PREPOSITIONS:
        return "predlog"
    if w in lb.SR_CLITICS:
        return "enklitika"
    if w in lb.SR_NEGATION:
        return "ne"
    if w in rules.conjunctions:
        return "veznik"
    if w in lb.SR_PAUSE_FALSE_EXTRA:
        return "a"
    return "ispred enklitike"


def koliko():
    data = ucitaj()
    rules = lb.rules_for("sr", lambda m: None)
    print(f"{len(data)} ASR poslova sa tišinama\n")
    print(" prag | pauza na granici | lažnih | predlog | enklitika | ne | veznik | a | ispred enklitike")
    for thr in PRAGOVI:
        n, kl = 0, Counter()
        for _, sents, sil, _ in data:
            for _, sw in sents:
                p = pz.boundary_pauses(sw, sil)
                for k, x in enumerate(p):
                    if x >= thr:
                        n += 1
                        if lb.false_pause_at(sw, k, rules):
                            kl[klasa(sw, k, rules)] += 1
        laz = sum(kl.values())
        print(f" {thr:.2f} | {n:>16} | {laz:>3} ({laz / n:5.1%}) | {kl['predlog']:>7} | {kl['enklitika']:>9} | "
              f"{kl['ne']:>2} | {kl['veznik']:>6} | {kl['a']:>1} | {kl['ispred enklitike']:>3}")
    # po rečima, na pragu 0,30
    reci = Counter()
    for _, sents, sil, _ in data:
        for _, sw in sents:
            for k, x in enumerate(pz.boundary_pauses(sw, sil)):
                if x >= 0.30 and lb.false_pause_at(sw, k, rules) == "left":
                    reci[lb._norm(sw[k].w)] += 1
    print("\nna pragu 0,30, iza reči:", ", ".join(f"{w} {c}" for w, c in reci.most_common()))


def meri(data, prof, thr, use_pauses):
    rules = lb.rules_for("sr", lambda m: None)
    floor = prof.min_chars_per_line or lb.MIN_CUE_CHARS
    t = dict(cues=0, pauza=0, granica=0, predlog=0, enklitika=0, ispod=0, cps=0, red=0)
    lay = {}
    for name, sents, sil, terms in data:
        cues_all = []
        for sid, sw in sents:
            cues = lb.layout_sentence(sw, prof, sid, [], rules, terms, sil if use_pauses else None)
            lay[(name, sid)] = [" / ".join(c.lines) for c in cues]
            p = pz.boundary_pauses(sw, sil)  # dopunska mera: sirove pripisane pauze
            for (a, b, line_cuts), c in zip(granice_cue(sw, cues), cues):
                if u_govoru(c, sil, thr):
                    t["pauza"] += 1
                if any(p[k] >= thr for k in range(a, b - 1)):
                    t["granica"] += 1
                if len(cues) > 1 and len(" ".join(c.lines)) < floor:
                    t["ispod"] += 1
                for k in line_cuts + ([b - 1] if b < len(sw) else []):
                    if lb._norm(sw[k].w) in lb.SR_PREPOSITIONS:
                        t["predlog"] += 1
                    if k + 1 < len(sw) and lb._norm(sw[k + 1].w) in rules.no_line_start:
                        t["enklitika"] += 1
            cues_all += cues
        timed = lb.apply_timing([c.model_copy(deep=True) for c in cues_all], prof)
        t["cues"] += len(timed)
        t["cps"] += sum(1 for c in timed if c.e > c.s
                        and sum(len(l) for l in c.lines) / (c.e - c.s) > prof.max_cps + 1e-9)
        t["red"] += sum(1 for c in timed for l in c.lines if len(l) > prof.max_chars_per_line)
    return t, lay


def postavi(thr, w, d, x, nacin="keep"):
    lb.PAUSE_MIN, lb.PENALTY_PAUSE_IN_CUE, lb.PAUSE_BREAK_DISCOUNT, lb.PENALTY_BREAK_WITHOUT_PAUSE = thr, w, d, x
    lb.PAUSE_AT_FUNCTION_WORD = nacin


def mreza():
    data = ucitaj()
    print(f"{len(data)} ASR poslova sa tišinama\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rezultat = {}
    for ime in ("tv", "web"):
        prof = cfg.load_profile(ime)
        baza = {}
        for thr in PRAGOVI:
            postavi(thr, 0, 0, 0)
            baza[thr] = meri(data, prof, thr, False)
        for nacin in NACINI:
            print(f"== {ime}, {nacin}")
            print(" prag  tež popust bez | cue-ova | u govoru (udeo)        | granica | predlog | enkl. | ispod | CPS | red | prom. | daleko")
            if nacin == "keep":
                for thr in PRAGOVI:
                    b = baza[thr][0]
                    print(f" {thr:.2f}  baza          | {b['cues']:>7} | {b['pauza']:>3} ({b['pauza'] / b['cues']:5.1%})          | "
                          f"{b['granica']:>7} | {b['predlog']:>7} | {b['enklitika']:>5} | {b['ispod']:>5} | {b['cps']:>3} | {b['red']:>3} |")
            for thr, w, d, x in tacke():
                postavi(thr, w, d, x, nacin)
                t, lay = meri(data, prof, thr, True)
                b, blay = baza[thr]
                ch = [k for k in lay if lay[k] != blay[k]]
                visak = {m: t[m] - b[m] for m in CUVARI if t[m] > b[m]}
                pada = t["pauza"] < b["pauza"]
                ok = pada and not visak
                daleko = "OK" if ok else (("pauza ne pada; " if not pada else "")
                                          + ", ".join(f"{m} +{v}" for m, v in visak.items()))
                u, ub = t["pauza"] / t["cues"], b["pauza"] / b["cues"]
                print(f" {thr:.2f} {w:>4} {d:>6} {x:>3} | {t['cues']:>7} | {t['pauza']:>3} ({u:5.1%}) {t['pauza'] - b['pauza']:+3} {u - ub:+6.1%} | "
                      f"{t['granica']:>7} | {t['predlog']:>7} | {t['enklitika']:>5} | {t['ispod']:>5} | {t['cps']:>3} | {t['red']:>3} | {len(ch):>5} | {daleko}")
                rezultat[f"{ime} {nacin} {thr} {w} {d} {x}"] = {
                    "mere": t, "baza": b, "ok": ok, "visak": visak,
                    "promene": {f"{n}#{sid}": {"pre": blay[(n, sid)], "posle": lay[(n, sid)]} for n, sid in ch},
                }
            print()
    postavi(0.30, 0, 0, 0)

    def pad(k):
        return rezultat[k]["baza"]["pauza"] - rezultat[k]["mere"]["pauza"]

    for ime in ("tv", "web"):
        for nacin in NACINI:
            pref = f"{ime} {nacin} "
            prosli = [k for k in rezultat if k.startswith(pref) and rezultat[k]["ok"]]
            line = f"{ime:>3} {nacin:<4}: prolazi {len(prosli):>2} od {len(tacke())}"
            if prosli:
                naj = max(prosli, key=lambda k: (pad(k), -float(k.split()[3])))
                line += f"; najveći pad: {naj.split(' ', 2)[2]} ({-pad(naj):+d}, promenjenih {len(rezultat[naj]['promene'])})"
            print(line)
    (OUT / "mreza.json").write_text(json.dumps(rezultat, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["meri"]:
        mreza()
    else:
        koliko()
