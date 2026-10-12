"""Krug 20: na kojoj strani veznika srpski lomi — merenje na svim ASR poslovima.

    .venv\\Scripts\\python scripts\\krug20_veznici.py

Varijante srpske tabele (ostali jezici se ne diraju):
  after     — sadašnje stanje, rez iza veznika košta 10;
  before    — kao engleski i francuski, rez ispred veznika košta 10;
  before-da — kao before, a `da` nije veznik nego `no_line_end` (300);
  before-da-pa-te — isto i za `pa` i `te`.

Po profilu (`tv`, `web`, `digital`) meri: rezove iza i ispred veznika (po
vezniku), iza predloga, ispred enklitike, cue-ove ispod donje granice,
prekoračenja CPS-a i dužine reda. Rez je svaki prelom unutar rečenice — između
redova istog cue-a i između cue-ova.

Na terminal idu samo brojevi. Rečenice koje su se promenile (tekst stvarnog
govora) idu u `merenja/krug20/` — u .gitignore.
"""
import dataclasses
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug6_digital import jobs  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402
from subtitis.models import SentencesFile, WordsFile  # noqa: E402

OUT = ROOT / "merenja" / "krug20"  # u .gitignore
PROFILI = ("tv", "web", "digital")
VEZNICI = lb.SR_CONJUNCTIONS  # brojanje uvek po celom spisku iz specifikacije


def varijante():
    sr = lb.LANGUAGE_RULES["sr"]

    def bez(*reci):
        return dataclasses.replace(
            sr, conjunction_side="before",
            conjunctions=sr.conjunctions - set(reci),
            no_line_end=sr.no_line_end | set(reci),
        )

    return {
        "after": sr,
        "before": dataclasses.replace(sr, conjunction_side="before"),
        "before-da": bez("da"),
        "before-da-pa-te": bez("da", "pa", "te"),
    }


def ucitaj():
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
        terms = lb.term_tokens(cfg.glossary_terms(cfg.job_glossary(d.name), "sr"))
        out.append((d.name, sents, terms))
    return out


def rezovi(sw, cues):
    """Indeksi tokena posle kojih je rez (redovi i cue-ovi), bez kraja rečenice."""
    out, i = [], 0
    for c in cues:
        for line in c.lines:
            j = i + 1
            while j <= len(sw) and lb.join_words(sw[i:j]) != line:
                j += 1
            out.append(j - 1)
            i = j
    return [k for k in out if k < len(sw) - 1]


def meri(data, prof, rules):
    floor = prof.min_chars_per_line or lb.MIN_CUE_CHARS
    t = Counter()
    lay = {}
    for name, sents, terms in data:
        cues_all = []
        for sid, sw in sents:
            cues = lb.layout_sentence(sw, prof, sid, [], rules, terms)
            lay[(name, sid)] = cues
            for k in rezovi(sw, cues):
                left, right = lb._norm(sw[k].w), lb._norm(sw[k + 1].w)
                t["rezova"] += 1
                if left in VEZNICI:
                    t["iza veznika"] += 1
                    t[f"iza {left}"] += 1
                if right in VEZNICI:
                    t["ispred veznika"] += 1
                    t[f"ispred {right}"] += 1
                if left in lb.SR_PREPOSITIONS:
                    t["iza predloga"] += 1
                if right in lb.SR_CLITICS:
                    t["ispred enklitike"] += 1
            if len(cues) > 1:
                t["ispod granice"] += sum(1 for c in cues if len(" ".join(c.lines)) < floor)
            cues_all += cues
        timed = lb.apply_timing([c.model_copy(deep=True) for c in cues_all], prof)
        t["cue-ova"] += len(timed)
        t["preko CPS"] += sum(1 for c in timed if c.e > c.s
                              and sum(len(l) for l in c.lines) / (c.e - c.s) > prof.max_cps + 1e-9)
        t["red preko"] += sum(1 for c in timed for l in c.lines if len(l) > prof.max_chars_per_line)
    return t, lay


MERE = ("cue-ova", "rezova", "iza veznika", "ispred veznika", "iza predloga",
        "ispred enklitike", "ispod granice", "preko CPS", "red preko")
STROGE = ("iza predloga", "ispred enklitike", "ispod granice", "preko CPS", "red preko")


def main():
    data = ucitaj()
    print(f"{len(data)} srpskih ASR poslova, {sum(len(s) for _, s, _ in data)} rečenica\n")
    OUT.mkdir(parents=True, exist_ok=True)
    promene = {}
    for ime in PROFILI:
        prof = cfg.load_profile(ime)
        rez = {v: meri(data, prof, r) for v, r in varijante().items()}
        base, blay = rez["after"]
        print(f"== {ime}")
        print(f"{'varijanta':<16} | " + " | ".join(MERE) + " | promenjenih reč.")
        for v, (t, lay) in rez.items():
            ch = [k for k in lay if [c.lines for c in lay[k]] != [c.lines for c in blay[k]]]
            ok = all(t[m] <= base[m] for m in STROGE)
            print(f"{v:<16} | " + " | ".join(
                f"{t[m]:>3}" + (f" ({t[m] - base[m]:+d})" if v != "after" and t[m] != base[m] else "")
                for m in MERE) + f" | {len(ch)}" + ("" if v == "after" else ("  mere OK" if ok else "  MERA PORASLA")))
            if v != "after":
                promene[f"{ime} {v}"] = {
                    f"{n}#{sid}": {
                        "pre": [f"[{c.s:.2f}] " + " / ".join(c.lines) for c in blay[(n, sid)]],
                        "posle": [f"[{c.s:.2f}] " + " / ".join(c.lines) for c in lay[(n, sid)]],
                    } for n, sid in ch
                }
        print("  po vezniku (after):  " + ", ".join(
            f"{w}: iza {base[f'iza {w}']} / ispred {base[f'ispred {w}']}" for w in sorted(VEZNICI)))
        print()
    (OUT / "promene.json").write_text(json.dumps(promene, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
