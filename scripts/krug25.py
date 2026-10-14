"""Krug 25: rezerva pred govorom, udaljenost tišine od granice, spoj preloma i vremena.

    .venv\\Scripts\\python scripts\\krug25.py udaljenost
    .venv\\Scripts\\python scripts\\krug25.py rezerva
    .venv\\Scripts\\python scripts\\krug25.py spoj
    .venv\\Scripts\\python scripts\\krug25.py srt <job_id>    # tri SRT-a za slušanje

Skup: poslovi sa `pauses.json` (od kruga 25 osam, i raste). Prelom je
proizvodni: `tv` sa paketom pauze (`PAUSE_MULTILINE`), `web` bez; vreme po
tišini (`snap_to_silences`) sa `SNAP_MIN_BLANK`. Merilo: krug 23/24
(`krug24_vreme.mere`). Ponovljeni snimak istog govora se broji jednom
(dnevnik, krug 25) — u brojevima natpisa nije izuzet, u čitanju jeste.
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug24_vreme import mere  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402
from subtitis import pauses as pz  # noqa: E402
from subtitis.models import SentencesFile, WordsFile  # noqa: E402

REZERVE = (0.0, 0.08, 0.15)
PRAG = 0.30


def ucitaj():
    """Samo poslovi sa `pauses.json` — zvuk se ne čita."""
    out = []
    for d in sorted(cfg.JOBS_DIR.iterdir()):
        if not (d / "pauses.json").exists() or not (d / "words.sr.json").exists():
            continue
        sil = pz.load(d.name).silences
        words = WordsFile.model_validate(cfg.read_json(d / "words.sr.json")).words
        sf = SentencesFile.model_validate(cfg.read_json(d / "sentences.sr.json"))
        by = {w.i: w for w in words}
        sents = []
        for s in sf.sentences:
            sw = [by[k] for k in range(s.words[0], s.words[1] + 1) if k in by]
            if sw:
                sw, _ = lb.align_text_to_words(s.text, sw)
                sw = [w for w in sw if w.w.strip()]
                if sw:
                    sents.append((s.id, sw))
        terms = lb.term_tokens(cfg.glossary_terms(cfg.job_glossary(d.name), "sr"))
        out.append((d.name, sents, sil, terms))
    return out


def prelomi(data, prof, *, sa_pauzama=True):
    rules = lb.rules_for("sr", lambda m: None)
    out, lay = [], {}
    for name, sents, sil, terms in data:
        cues = []
        for sid, sw in sents:
            cs = lb.layout_sentence(sw, prof, sid, [], rules, terms, sil if sa_pauzama else None)
            lay[(name, sid)] = [" / ".join(c.lines) for c in cs]
            cues += cs
        govor = [(c.s, c.e) for c in cues]
        out.append((name, lb.apply_timing(cues, prof), govor, sil))
    return out, lay


def vreme(posao, prof, lead):
    posle = [(n, [c.model_copy(deep=True) for c in cs], g, s) for n, cs, g, s in posao]
    reps = [lb.snap_to_silences(cs, s, prof, min_blank=lb.SNAP_MIN_BLANK, lead=lead) for _, cs, _, s in posle]
    return posle, reps


def ispisi(lab, m, prof, extra=""):
    print(f" {lab:<26}| {m['cues']:>4} | {m['pauza']:>3} ({m['pauza'] / m['cues']:5.1%}) "
          f"p/k/s {m['glava']:>2}/{m['rep']:>2}/{m['sredina']:>2} | CPS {m['cps']:>2} | prekratkih {m.get('kratki', 0):>2}"
          f" | preklop {m['preklop']} | mrtvo {m['mrtvo']:6.2f} s{extra}")


def kratki(posao, prof):
    return sum(1 for _, cs, _, _ in posao for c in cs if c.e - c.s < prof.min_duration - 1e-9)


# --- 1. udaljenost -----------------------------------------------------------

def udaljenost():
    data = ucitaj()
    sl = lb.SNAP_EDGE_SLACK
    d_bins, zak = Counter(), Counter()
    n = 0
    for _, sents, sil, _ in data:
        for _, sw in sents:
            real = [w for w in sw if not lb._is_punct(w.w)]
            if len(real) < 2:
                continue
            granice = [(x.e, y.s) for x, y in zip(real, real[1:])]
            for a, b in sil:
                if b - a < PRAG or b <= real[0].s or a >= real[-1].e:
                    continue
                n += 1
                d = min(0.0 if (a <= g1 and b >= g0) else min(abs(a - g1), abs(b - g0)) for g0, g1 in granice)
                d_bins["0 (tišina dodiruje granicu)" if d == 0 else "≤ 0,10" if d <= 0.10 else
                       "≤ 0,20" if d <= 0.20 else "≤ 0,30" if d <= 0.30 else "> 0,30"] += 1
                # zakačivo: rez na nekoj granici pa vreme ivice skida tišinu
                pocetak = any(a <= g1 + sl and b - g1 >= PRAG for _, g1 in granice)
                kraj = any(b >= g0 - sl and g0 - a >= PRAG for g0, _ in granice)
                zak["početak sledećeg (tišina na čelu reči)" if pocetak else
                    "kraj prethodnog (tišina na repu reči)" if kraj else "ni jedno — duboko u reči"] += 1
    print(f"{len(data)} poslova sa pauses.json, tišina ≥ {PRAG:.2f} s unutar rečenice: {n}\n")
    print("udaljenost do najbliže granice reči:")
    for k in ("0 (tišina dodiruje granicu)", "≤ 0,10", "≤ 0,20", "≤ 0,30", "> 0,30"):
        print(f"  {k:<30} {d_bins[k]:>3} ({d_bins[k] / n:5.1%})")
    print("\nda li rez + vreme ivice mogu da je skinu (ivica tišine ≤ 0,10 s od granice, ≥ 0,30 s sa druge strane):")
    for k, v in zak.most_common():
        print(f"  {k:<42} {v:>3} ({v / n:5.1%})")


# --- 2. rezerva + stanje posle paketa ----------------------------------------

def rezerva():
    data = ucitaj()
    print(f"{len(data)} poslova sa pauses.json; SNAP_MIN_BLANK {lb.SNAP_MIN_BLANK}\n")
    for ime in ("tv", "web"):
        prof = cfg.load_profile(ime)
        print(f"== {ime} (paket pauze: {'da' if lb.pauses_enabled(prof) else 'ne'})")
        bez, _ = prelomi(data, prof, sa_pauzama=False)
        m = mere(bez, prof); m["kratki"] = kratki(bez, prof)
        ispisi("prelom bez pauza, bez vremena", m, prof)
        if lb.pauses_enabled(prof):
            sa, _ = prelomi(data, prof)
            m = mere(sa, prof); m["kratki"] = kratki(sa, prof)
            ispisi("paket, bez vremena", m, prof)
        else:
            sa = bez
        for lead in REZERVE:
            posle, reps = vreme(sa, prof, lead)
            m = mere(posle, prof); m["kratki"] = kratki(posle, prof)
            pom = [d for r in reps for _, d in r.starts + r.ends]
            odb = {k: sum(r.rejected[k] for r in reps) for k in ("trajanje", "cps", "praznina")}
            ispisi(f"+ vreme, rezerva {lead:.2f}", m, prof,
                   f" | poč. {sum(len(r.starts) for r in reps)} kraj {sum(len(r.ends) for r in reps)}"
                   f" | pomeraj Ø {sum(pom) / len(pom) if pom else 0:.2f} | odbijeno {odb}")
        print()


# --- 3. spoj preloma i vremena u jednorednom ---------------------------------

def spoj():
    data = ucitaj()
    prof = cfg.load_profile("web")
    print(f"{len(data)} poslova; `web` — popust za rez na pauzi + filter + zamenica, pa vreme (rezerva {lb.SNAP_LEAD})\n")
    bez, blay = prelomi(data, prof, sa_pauzama=False)
    bposle, _ = vreme(bez, prof, lb.SNAP_LEAD)
    b = mere(bposle, prof); b["kratki"] = kratki(bposle, prof)
    ispisi("današnje (samo vreme)", b, prof)
    stari = (lb.PAUSE_MIN, lb.PAUSE_BREAK_DISCOUNT, lb.PENALTY_PAUSE_IN_CUE, lb.PAUSE_AT_FUNCTION_WORD, lb.PENALTY_SPLIT_PRONOUN)
    for thr in (0.20, 0.30):
        for disc in (25, 50, 100):
            for w in (0, 50):
                lb.PAUSE_MIN, lb.PAUSE_BREAK_DISCOUNT, lb.PENALTY_PAUSE_IN_CUE = thr, disc, w
                lb.PAUSE_AT_FUNCTION_WORD, lb.PENALTY_SPLIT_PRONOUN = "discard", 25
                sa, lay = prelomi(data, prof)
                posle, _ = vreme(sa, prof, lb.SNAP_LEAD)
                m = mere(posle, prof); m["kratki"] = kratki(posle, prof)
                ch = sum(1 for k in lay if lay[k] != blay[k])
                ispisi(f"prag {thr:.2f} popust {disc:>3} kazna {w:>2}", m, prof,
                       f" | ΔCPS {m['cps'] - b['cps']:+d} | promenjenih reč. {ch}")
    lb.PAUSE_MIN, lb.PAUSE_BREAK_DISCOUNT, lb.PENALTY_PAUSE_IN_CUE, lb.PAUSE_AT_FUNCTION_WORD, lb.PENALTY_SPLIT_PRONOUN = stari


# --- 4. tri SRT-a za slušanje -------------------------------------------------

def srt(job_id, prof="digital", case="upper", drop=True):
    """Kao poslednje prelamanje posla (Video-2 draft 2: `digital`, VELIKA, bez
    interpunkcije) — tri fajla pored glavnog, glavni se ne dira."""
    from subtitis import pipeline

    out = cfg.job_dir(job_id) / "out"
    stari = lb.SNAP_LEAD
    try:
        for lead in REZERVE:
            lb.SNAP_LEAD = lead
            job = cfg.load_job(job_id)
            stem = Path(job.source_file).stem
            path = out / f"{stem}.sr-Latn_lead{round(lead * 100):03d}.srt"
            pipeline.run_build(job_id, profile=prof, case=case, drop_punct=drop, out=path)
            print(path)
    finally:
        lb.SNAP_LEAD = stari


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "udaljenost"
    {"udaljenost": udaljenost, "rezerva": rezerva, "spoj": spoj}.get(cmd, lambda: srt(sys.argv[2]))()
