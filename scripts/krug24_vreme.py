"""Krug 24: vreme natpisa po tišini (`linebreak.snap_to_silences`).

    .venv\\Scripts\\python scripts\\krug24_vreme.py          # tabela
    .venv\\Scripts\\python scripts\\krug24_vreme.py deset    # + deset najvećih pomeranja

Prelom je današnji (pauze isključene); posle `apply_timing` se ivice natpisa
pomeraju na ivice tišine, uz dva minimuma praznog ekrana (0,20 i 0,40 s).
Tišine: `pauses.json`, a gde ga nema, iz `source.wav` u memoriji — u poslove
se ništa ne upisuje (kao krugovi 22 i 23).

Merilo „natpis sadrži pauzu" (dnevnik, krug 23): tišina se preklapa bar
pragom sa onim delom raspona govora natpisa koji je na ekranu.

Sa `deset` se ispisuje tekst natpisa — tekst stvarnog govora; ne ide u repo.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug22_pauze import ucitaj  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402

PRAG = lb.SNAP_SILENCE_MIN
MINIMUMI = (0.20, 0.40)


def preklop(a, b, s, e):
    return max(0.0, min(b, e) - max(a, s))


def mere(poslovi, prof):
    """poslovi: [(ime, cues, govor, sil)] — govor[k] = (s, e) govora cue-a k."""
    t = dict(cues=0, pauza=0, glava=0, rep=0, sredina=0, cps=0, preklop=0,
             mrtvo=0.0, nad_tisinom=0.0, kratkih_20=0, kratkih_40=0)
    for _, cues, govor, sil in poslovi:
        t["cues"] += len(cues)
        for c, (gs, ge) in zip(cues, govor):
            a, b = max(c.s, gs), min(c.e, ge)
            hit = [(s, e) for s, e in sil if preklop(a, b, s, e) >= PRAG]
            if hit:
                t["pauza"] += 1
                s, e = hit[0]
                t["glava" if s <= a + lb.SNAP_EDGE_SLACK else "rep" if e >= b - lb.SNAP_EDGE_SLACK else "sredina"] += 1
            t["nad_tisinom"] += sum(preklop(c.s, c.e, s, e) for s, e in sil)
            if c.e > c.s and sum(len(l) for l in c.lines) / (c.e - c.s) > prof.max_cps + 1e-9:
                t["cps"] += 1
        for x, y in zip(cues, cues[1:]):
            gap = y.s - x.e
            if gap < -1e-9:
                t["preklop"] += 1
            t["mrtvo"] += max(0.0, gap)
            t["kratkih_20"] += 0 < gap < 0.20 - 1e-9
            t["kratkih_40"] += 0 < gap < 0.40 - 1e-9
    return t


def postavi(data, prof):
    rules = lb.rules_for("sr", lambda m: None)
    out = []
    for name, sents, sil, terms in data:
        cues = []
        for sid, sw in sents:
            cues += lb.layout_sentence(sw, prof, sid, [], rules, terms)
        govor = [(c.s, c.e) for c in cues]
        cues = lb.apply_timing(cues, prof)
        out.append((name, cues, govor, sil))
    return out


def main(deset=False):
    data = ucitaj()
    print(f"{len(data)} ASR poslova sa tišinama, prag tišine {PRAG:.2f} s\n")
    for ime in ("tv", "web"):
        prof = cfg.load_profile(ime)
        pre = postavi(data, prof)
        b = mere(pre, prof)
        print(f"== {ime} (min. trajanje {prof.min_duration} s, {prof.max_cps:g} CPS)")
        red = lambda lab, m, extra="": print(  # noqa: E731
            f" {lab:<14}| {m['cues']:>4} | {m['pauza']:>3} ({m['pauza'] / m['cues']:5.1%}) "
            f"glava {m['glava']:>2} rep {m['rep']:>2} sredina {m['sredina']:>2} | CPS {m['cps']:>2} | "
            f"preklop {m['preklop']} | mrtvo {m['mrtvo']:7.2f} s | nad tišinom {m['nad_tisinom']:6.2f} s | "
            f"praznina <0,2: {m['kratkih_20']:>2} <0,4: {m['kratkih_40']:>2}{extra}")
        red("pre", b)
        for mb in MINIMUMI:
            posle = [(n, [c.model_copy(deep=True) for c in cs], g, s) for n, cs, g, s in pre]
            reps = [lb.snap_to_silences(cs, s, prof, min_blank=mb) for _, cs, _, s in posle]
            m = mere(posle, prof)
            pom = [d for r in reps for _, d in r.starts + r.ends]
            odb = {k: sum(r.rejected[k] for r in reps) for k in ("trajanje", "cps", "praznina")}
            red(f"posle, min {mb:.2f}", m)
            print(f"   početak pomeren {sum(len(r.starts) for r in reps)}, kraj skraćen {sum(len(r.ends) for r in reps)}"
                  f" | pomeraj prosek {sum(pom) / len(pom) if pom else 0:.2f} s, najveći {max(pom, default=0):.2f} s"
                  f" | odbijeno: trajanje {odb['trajanje']}, CPS {odb['cps']}, praznina {odb['praznina']}"
                  f" | Δ mrtvo {m['mrtvo'] - b['mrtvo']:+.2f} s, Δ nad tišinom {m['nad_tisinom'] - b['nad_tisinom']:+.2f} s")
            if deset and mb == MINIMUMI[0]:
                rows = []
                for (n, cs0, _, _), (_, cs1, _, _) in zip(pre, posle):
                    for k, (x, y) in enumerate(zip(cs0, cs1)):
                        d = (y.s - x.s) + (x.e - y.e)
                        if d > 0:
                            nxt = cs1[k + 1].s - y.e if k + 1 < len(cs1) else None
                            prv = y.s - cs1[k - 1].e if k else None
                            rows.append((d, x, y, prv, nxt, n))
                rows.sort(key=lambda r: -r[0])
                print("   stara -> nova vremena            | praznina pre | do sledećeg | tekst")
                for d, x, y, prv, nxt, n in rows[:10]:
                    f = lambda v: "—" if v is None else f"{v:.2f} s"  # noqa: E731
                    print(f"   {x.s:6.2f}–{x.e:6.2f} -> {y.s:6.2f}–{y.e:6.2f} | {f(prv):>12} | {f(nxt):>11} | "
                          f"{' / '.join(x.lines)}  [{n[-26:]}]")
        print()


if __name__ == "__main__":
    main(deset=sys.argv[1:] == ["deset"])
