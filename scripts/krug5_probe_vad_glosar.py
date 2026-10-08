"""Krug 5, tačka 3: jedna proba nad Radionica draft 1om, u svom procesu.

    .venv\Scripts\python scripts\krug5_probe_vad_glosar.py <ćelija> [<ponavljanje>]

Ćelije (VAD × glosar):
    K  VAD isključen, glosar pun     — podrazumevano u produkciji (kontrola)
    A  VAD „normalno", glosar pun
    B  VAD isključen, glosar prazan
    C  VAD „normalno", glosar prazan

Izlaz: merenja/krug5/proba_<ćelija>_<ponavljanje>.json (sve reči sa vremenima;
u .gitignore — sadržaj tuđeg snimka). Nalaz: `merenja/dnevnik.md`.

Ponavljanje postoji jer rezultat NIJE uvek isti: kad dekodiranje ne prođe
pragove, faster-whisper ponavlja sa temperaturom (nasumično). Proba A je u dva
pokretanja dala 11 reči čistog glosara, pa 60 reči pravog govora.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitis import asr, config as cfg, segment  # noqa: E402

CELL = sys.argv[1]
REP = sys.argv[2] if len(sys.argv) > 2 else "1"
WAV = str(ROOT / "jobs" / "20261001-175002-Radionica_draft-1" / "source.wav")
DUR = 32.28
gloss_terms = cfg.glossary_keys(cfg.load_glossary(), "sr")
use_vad, thr = asr.vad_preset("normalno" if CELL in ("A", "C") else "iskljuceno")
terms = [] if CELL in ("B", "C") else gloss_terms

t = time.perf_counter()
wf = asr.transcribe(WAV, job_id="probe", source_file=WAV, language="sr", duration=DUR,
                    glossary_terms=terms, vad=use_vad, vad_threshold=thr)
dt = time.perf_counter() - t
asr.unload_model()

words = wf.words
real = [w for w in words if any(ch.isalnum() for ch in w.w)]
zero = [w for w in real if w.e - w.s < 0.02]
gloss_words = {t.lower() for term in gloss_terms for t in term.replace(",", " ").split()}
from_gloss = [w for w in real if w.w.lower().strip(".,!?") in gloss_words]
sents = segment.segment(words)
cov, _ = segment.coverage(sents, DUR)
res = {
    "celija": CELL, "ponavljanje": REP, "vad": use_vad, "prag": thr,
    "glosar_termina": len(terms), "sekundi": round(dt, 1), "reci": len(real),
    "pokrivenost": round(cov, 1), "nultih": len(zero),
    "nulte": [f"{w.w}@{w.s:.2f}" for w in zero],
    "reci_iz_glosara": len(from_gloss),
    "poslednjih_15": " ".join(w.w for w in real[-15:]),
    "poslednjih_15_vremena": [f"{w.w} {w.s:.2f}-{w.e:.2f}" for w in real[-15:]],
    "sve_reci": [f"{w.w} {w.s:.2f}-{w.e:.2f} p={w.p:.2f}" for w in words],
}
out = ROOT / "merenja" / "krug5" / f"proba_{CELL}_{REP}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{CELL}#{REP}: reči {len(real)}, pokrivenost {cov:.1f}%, nultih {len(zero)}, "
      f"reči iz glosara {len(from_gloss)} | …{' '.join(w.w for w in real[-8:])}")
