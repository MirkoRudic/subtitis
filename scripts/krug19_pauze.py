"""Krug 19: pauze između reči na svim ASR poslovima na disku.

    .venv\\Scripts\\python scripts\\krug19_pauze.py          # raspodela razmaka
    .venv\\Scripts\\python scripts\\krug19_pauze.py meri     # mreža konstanti

Pauza na granici dve reči = `s` sledeće reči − `e` prethodne, preskačući
tokene interpunkcije (zarez je poseban token od ~0,04 s). Ispisuje:
  - raspodelu pauza po pragovima, posebno na granici sa interpunkcijom i bez;
  - raspodelu „viška trajanja" reči: trajanje − medijana trajanja po znaku
    tog posla × broj znakova. Ako Whisper tišinu upija u reč umesto da je
    ostavi između reči, pauza se vidi tu, a ne u razmaku.

Sa `meri`: tišine iz zvuka (Silero VAD, `pauses.detect`) računaju se u
memoriji — u poslove se NE upisuje `pauses.json` — i za svaku tačku mreže
(prag, težina kazne za pauzu u cue-u, popust za rez na pauzi, kazna za rez
bez pauze) meri na sva tri profila: cue-ove sa pauzom iznad praga, rezove
iza predloga i ispred enklitike, cue-ove ispod donje granice, prekoračenja
CPS-a i dužine reda, i broj promenjenih rečenica. Rečenice koje su se
promenile (tekst stvarnog govora) idu u `merenja/krug19/` — u .gitignore.

Na terminal se ne ispisuje nijedna reč — samo brojevi.
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug6_digital import jobs  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis.linebreak import _is_punct  # noqa: E402
from subtitis.models import SentencesFile, WordsFile  # noqa: E402

PRAGOVI = (0.05, 0.12, 0.20, 0.30, 0.45, 0.70)


def granice(words):
    """(pauza, ima_interpunkciju) za svaku granicu između dve reči."""
    real = [k for k, w in enumerate(words) if not _is_punct(w.w)]
    for a, b in zip(real, real[1:]):
        yield words[b].s - words[a].e, b - a > 1


def main():
    unutar = {True: [], False: []}  # unutar rečenice: sa interpunkcijom / bez
    izmedju = []                    # granica između rečenica
    visak = []
    poslova = 0
    for d in jobs():
        poslova += 1
        words = WordsFile.model_validate(cfg.read_json(d / "words.sr.json")).words
        sf = SentencesFile.model_validate(cfg.read_json(d / "sentences.sr.json"))
        by_index = {w.i: w for w in words}
        prev_end = None
        for sent in sf.sentences:
            lo, hi = sent.words
            sw = [by_index[k] for k in range(lo, hi + 1) if k in by_index]
            if not sw:
                continue
            if prev_end is not None:
                izmedju.append(sw[0].s - prev_end)
            prev_end = sw[-1].e
            for g, p in granice(sw):
                unutar[p].append(g)
        reals = [w for w in words if not _is_punct(w.w) and len(w.w.strip()) > 0]
        per_char = statistics.median((w.e - w.s) / len(w.w.strip()) for w in reals)
        visak += [(w.e - w.s) - per_char * len(w.w.strip()) for w in reals]

    print(f"{poslova} ASR poslova\n")
    print("pauza između reči        | granica  | " + " | ".join(f">= {t:.2f}" for t in PRAGOVI) + " | tačno 0")
    for label, xs in (("unutar rečenice, bez zn.", unutar[False]),
                      ("unutar rečenice, uz zarez", unutar[True]),
                      ("između rečenica", izmedju)):
        cells = " | ".join(f"{sum(1 for x in xs if x >= t):>4} ({sum(1 for x in xs if x >= t) / len(xs):5.1%})"
                           for t in PRAGOVI)
        zero = sum(1 for x in xs if abs(x) < 1e-9)
        print(f"{label:<25}| {len(xs):>8} | {cells} | {zero} ({zero / len(xs):.1%})")
    print("\nvišak trajanja reči (s) | reči | " + " | ".join(f">= {t:.2f}" for t in PRAGOVI))
    cells = " | ".join(f"{sum(1 for x in visak if x >= t):>4} ({sum(1 for x in visak if x >= t) / len(visak):5.1%})"
                       for t in PRAGOVI)
    print(f"{'':<24}| {len(visak):>4} | {cells}")


# --- merenje mreže ---------------------------------------------------------

import json  # noqa: E402

from subtitis import linebreak as lb  # noqa: E402
from subtitis import pauses as pz  # noqa: E402

OUT = ROOT / "merenja" / "krug19"  # u .gitignore
PROFILI = ("tv", "web", "digital")


def ucitaj():
    """[(posao, rečenice kao liste reči, tišine ili None)]"""
    out = []
    for d in jobs():
        words = WordsFile.model_validate(cfg.read_json(d / "words.sr.json")).words
        sf = SentencesFile.model_validate(cfg.read_json(d / "sentences.sr.json"))
        by_index = {w.i: w for w in words}
        sents = []
        for sent in sf.sentences:
            lo, hi = sent.words
            sw = [by_index[k] for k in range(lo, hi + 1) if k in by_index]
            if not sw:
                continue
            sw, _ = lb.align_text_to_words(sent.text, sw)
            sw = [w for w in sw if w.w.strip()]
            if sw:
                sents.append((sent.id, sw))
        wav = d / "source.wav"
        sil = pz.detect(wav).silences if wav.exists() else None
        terms = lb.term_tokens(cfg.glossary_terms(cfg.job_glossary(d.name), "sr"))
        out.append((d.name, sents, sil, terms))
    return out


def granice_cue(sw, cues):
    """Za svaki cue: (prvi token, kraj, rezovi redova) u indeksima reči."""
    out, i = [], 0
    for c in cues:
        start, cuts = i, []
        for line in c.lines:
            j = i + 1
            while j <= len(sw) and lb.join_words(sw[i:j]) != line:
                j += 1
            cuts.append(j - 1)
            i = j
        out.append((start, i, cuts[:-1]))
    return out


def meri(data, prof, thr_ref, use_pauses):
    rules = lb.rules_for("sr", lambda m: None)
    floor = prof.min_chars_per_line or lb.MIN_CUE_CHARS
    t = dict(cues=0, pauza=0, predlog=0, enklitika=0, ispod=0, cps=0, red=0)
    lay = {}
    for name, sents, sil, terms in data:
        cues_all = []
        for sid, sw in sents:
            cues = lb.layout_sentence(sw, prof, sid, [], rules, terms,
                                      sil if use_pauses else None)
            lay[(name, sid)] = [" / ".join(c.lines) for c in cues]
            p = pz.boundary_pauses(sw, sil) if sil is not None else [0.0] * len(sw)
            bounds = granice_cue(sw, cues)
            for (a, b, line_cuts), c in zip(bounds, cues):
                if any(p[k] >= thr_ref for k in range(a, b - 1)):
                    t["pauza"] += 1
                if len(cues) > 1 and len(" ".join(c.lines)) < floor:
                    t["ispod"] += 1
                cuts = line_cuts + ([b - 1] if b < len(sw) else [])
                for k in cuts:
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


def postavi(thr, w, d, x):
    lb.PAUSE_MULTILINE = None  # krug 25: paket za dvoredni bi pregazio mrežu
    lb.PAUSE_MIN, lb.PENALTY_PAUSE_IN_CUE, lb.PAUSE_BREAK_DISCOUNT, lb.PENALTY_BREAK_WITHOUT_PAUSE = thr, w, d, x


def mreza():
    data = ucitaj()
    sa_zvukom = sum(1 for _, _, sil, _ in data if sil is not None)
    print(f"{len(data)} ASR poslova, {sa_zvukom} sa zvukom (ostali bez pauza, kao ranije)\n")
    OUT.mkdir(parents=True, exist_ok=True)
    tacke = [(thr, w, d, 0) for thr in (0.12, 0.20, 0.30, 0.45)
             for w in (0, 50, 100, 200) for d in (0, 25, 50) if (w, d) != (0, 0)]
    tacke += [(thr, 100, 25, x) for thr in (0.20, 0.30) for x in (10, 25)]
    rezultat = {}
    for ime in PROFILI:
        prof = cfg.load_profile(ime)
        print(f"== {ime}")
        print(" prag  tež popust bez | cue-ova | sa pauzom | predlog | enklitika | ispod | CPS | red | promenjenih reč.")
        baza = {}
        for thr in (0.12, 0.20, 0.30, 0.45):
            postavi(thr, 0, 0, 0)
            baza[thr] = meri(data, prof, thr, False)
            b = baza[thr][0]
            print(f" {thr:.2f}  baza          | {b['cues']:>7} | {b['pauza']:>9} | {b['predlog']:>7} | "
                  f"{b['enklitika']:>9} | {b['ispod']:>5} | {b['cps']:>3} | {b['red']:>3} |")
        for thr, w, d, x in tacke:
            postavi(thr, w, d, x)
            t, lay = meri(data, prof, thr, True)
            b, blay = baza[thr]
            ch = [k for k in lay if lay[k] != blay[k]]
            ok = all(t[m] <= b[m] for m in ("predlog", "enklitika", "ispod", "cps", "red")) and t["pauza"] < b["pauza"]
            print(f" {thr:.2f} {w:>4} {d:>6} {x:>3} | {t['cues']:>7} | {t['pauza']:>4} ({t['pauza'] - b['pauza']:+4}) | "
                  f"{t['predlog']:>3} ({t['predlog'] - b['predlog']:+3}) | {t['enklitika']:>4} ({t['enklitika'] - b['enklitika']:+3}) | "
                  f"{t['ispod']:>2} ({t['ispod'] - b['ispod']:+2}) | {t['cps']:>3} | {t['red']:>3} | {len(ch):>3}"
                  f"{'  OK' if ok else ''}")
            rezultat[f"{ime} {thr} {w} {d} {x}"] = {
                "mere": t, "baza": b,
                "promene": {f"{n}#{sid}": {"pre": blay[(n, sid)], "posle": lay[(n, sid)]} for n, sid in ch},
            }
        print()
    postavi(0.30, 0, 0, 0)
    (OUT / "mreza.json").write_text(json.dumps(rezultat, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:] == ["meri"]:
        mreza()
    else:
        main()
