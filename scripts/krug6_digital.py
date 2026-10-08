"""Krug 6, tačka 2: profil `digital` nad svim ASR poslovima na disku.

    .venv\Scripts\python scripts\krug6_digital.py > merenja\krug6\digital.log

Meri, za svaku vrednost `PENALTY_BELOW_MIN_CHARS` (0 = bez donje granice,
pa 15/30/60):
  - broj cue-ova;
  - koliko cue-ova ima manje od `min_chars_per_line` znakova (posebno oni na
    koje izuzetak ne važi — kraći su iako je ostatak rečenice bio dovoljan);
  - koliko reza unutar rečenice pada unutar sintagme: `break_pen` >= 200
    (pridev 200; predlog, enklitika, glosar 300);
  - prirodno trajanje cue-a (pre `apply_timing`) ispod 0,83 / 0,70 / 0,50 s,
    i koliko posle `apply_timing` ostaje ispod 0,50 s (nema mesta do sledećeg);
  - koliko cue-ova probija 20 CPS posle `apply_timing`, `digital` naspram `web`.

Prelamanje je determinističko (nema modela ni uzorkovanja), pa je jedan
prolaz merenje. Ništa se ne upisuje u poslove. Sirovi rezultati po cue-u (tekst
pravog govora) idu u `merenja/krug6/` — u .gitignore; brojke u `merenja/dnevnik.md`.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402
from subtitis.models import SentencesFile, WordsFile  # noqa: E402

OUT = ROOT / "merenja" / "krug6"  # u .gitignore
OUT.mkdir(parents=True, exist_ok=True)
SINTAGMA = 200
PENALTIES = (0, 15, 30, 60)


def jobs():
    for d in sorted(cfg.JOBS_DIR.iterdir()):
        if (d / "words.sr.json").exists() and (d / "sentences.sr.json").exists():
            yield d


def layout_job(d, profile):
    """Kao `build_cues` za izvorni jezik, ali uz granice cue-ova u rečima."""
    sf = SentencesFile.model_validate(cfg.read_json(d / "sentences.sr.json"))
    words = WordsFile.model_validate(cfg.read_json(d / "words.sr.json")).words
    terms = lb.term_tokens(cfg.glossary_terms(cfg.job_glossary(d.name), "sr"))
    rules = lb.rules_for("sr", lambda m: None)
    by_index = {w.i: w for w in words}
    rows = []  # (cue, rez-kazne unutar cue-a i na kraju, ostatak rečenice od početka)
    for sent in sf.sentences:
        lo, hi = sent.words
        sw = [by_index[k] for k in range(lo, hi + 1) if k in by_index]
        if not sw:
            continue
        sw, _ = lb.align_text_to_words(sent.text, sw)
        sw = [w for w in sw if w.w.strip()]
        if not sw:
            continue
        span = lb._Span(sw, profile, rules, terms)
        cues = lb.layout_sentence(sw, profile, sent.id, list(sent.flags), rules, terms)
        i = 0
        for c in cues:
            # granice reda u rečima: najkraći prefiks čiji je tekst taj red
            start = i
            cuts = []
            for line in c.lines:
                j = i + 1
                while j <= span.n and lb.join_words(sw[i:j]) != line:
                    j += 1
                if j > span.n:
                    raise RuntimeError(f"{d.name} #{sent.id}: red se ne poklapa: {line!r}")
                cuts.append(j - 1)
                i = j
            # rezovi: između redova i (ako rečenica ide dalje) na kraju cue-a
            pens = [span.break_pen[k] for k in cuts if k < span.n - 1]
            # ostatak rečenice od početka cue-a — za izuzetak donje granice
            rows.append({"cue": c, "pens": pens, "whole": len(cues) == 1,
                         "rest": span.length(start, span.n)})
    natural = [(r["cue"].e - r["cue"].s) for r in rows]
    timed = lb.apply_timing([r["cue"].model_copy(deep=True) for r in rows], profile)
    return rows, natural, timed


def cue_chars(c):
    return max(len(l) for l in c.lines)


def main():
    web = cfg.load_profile("web")
    base = cfg.load_profile("digital")
    floor = base.min_chars_per_line
    names = [d.name for d in jobs()]
    print(f"poslovi ({len(names)}): " + ", ".join(names))
    print(f"digital: {base.model_dump()}\n")

    raw = {}
    summary = {}
    for pen in PENALTIES:
        lb.PENALTY_BELOW_MIN_CHARS = pen
        prof = base.model_copy(update={"min_chars_per_line": floor if pen else 0})
        tot = dict(cues=0, under=0, under_unexempt=0, cuts=0, sintagma=0,
                   n083=0, n070=0, n050=0, timed050=0, cps=0)
        per_job = {}
        for d in jobs():
            rows, natural, timed = layout_job(d, prof)
            under = [r for r in rows if cue_chars(r["cue"]) < floor and not r["whole"]]
            unexempt = [r for r in under if r["rest"] >= floor]
            pens = [p for r in rows for p in r["pens"]]
            j = dict(
                cues=len(rows), under=len(under), under_unexempt=len(unexempt),
                cuts=len(pens), sintagma=sum(1 for p in pens if p >= SINTAGMA),
                n083=sum(1 for t in natural if t < 0.83),
                n070=sum(1 for t in natural if t < 0.70),
                n050=sum(1 for t in natural if t < 0.50),
                timed050=sum(1 for c in timed if c.e - c.s < 0.50 - 1e-9),
                cps=sum(1 for c in timed if c.e > c.s and lb_chars(c) / (c.e - c.s) > prof.max_cps + 1e-9),
            )
            per_job[d.name] = j
            for k in tot:
                tot[k] += j[k]
            raw.setdefault(f"digital-{pen}", {})[d.name] = [
                {"s": r["cue"].s, "e": r["cue"].e, "lines": r["cue"].lines,
                 "pens": r["pens"], "rest": r["rest"]} for r in rows
            ]
        summary[pen] = (tot, per_job)

    web_tot = dict(cues=0, cps=0, cuts=0, sintagma=0)
    for d in jobs():
        rows, _, timed = layout_job(d, web)
        pens = [p for r in rows for p in r["pens"]]
        web_tot["cues"] += len(rows)
        web_tot["cuts"] += len(pens)
        web_tot["sintagma"] += sum(1 for p in pens if p >= SINTAGMA)
        web_tot["cps"] += sum(
            1 for c in timed if c.e > c.s and lb_chars(c) / (c.e - c.s) > web.max_cps + 1e-9
        )

    print("PENALTY_BELOW_MIN_CHARS | cue-ova | <10 zn. (bez izuzetka) | rezova | u sintagmi | prirodno <0,83 / <0,70 / <0,50 s | posle timinga <0,50 | >20 CPS")
    for pen, (t, _) in summary.items():
        print(f"{pen:>23} | {t['cues']:>7} | {t['under']:>4} ({t['under_unexempt']:>3}) | {t['cuts']:>6} | "
              f"{t['sintagma']:>4} ({t['sintagma'] / max(t['cuts'], 1):.1%}) | "
              f"{t['n083']:>4} / {t['n070']:>4} / {t['n050']:>4} ({t['n050'] / t['cues']:.1%}) | "
              f"{t['timed050']:>4} | {t['cps']:>4} ({t['cps'] / t['cues']:.1%})")
    print(f"\nweb: {web_tot['cues']} cue-ova, >20 CPS: {web_tot['cps']} ({web_tot['cps'] / web_tot['cues']:.1%}), "
          f"rezova {web_tot['cuts']}, u sintagmi {web_tot['sintagma']} ({web_tot['sintagma'] / web_tot['cuts']:.1%})")

    print("\npo poslu (PENALTY_BELOW_MIN_CHARS=30): cue-ova, <10, u sintagmi, prirodno <0,50, >20 CPS")
    for name, j in summary[30][1].items():
        print(f"  {name}: {j['cues']}, {j['under']}, {j['sintagma']}, {j['n050']}, {j['cps']}")

    (OUT / "digital_cues.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")


def lb_chars(c):
    return sum(len(l) for l in c.lines)


if __name__ == "__main__":
    main()
