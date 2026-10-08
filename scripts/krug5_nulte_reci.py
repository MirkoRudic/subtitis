"""Krug 5, tačka 1: pravilo „niz reči nultog trajanja" nad urednim snimcima.

    .venv\Scripts\python scripts\krug5_nulte_reci.py > merenja\krug5\zero_runs.log

Prepoznavanje ide kako ide u produkciji (VAD isključen, globalni glosar kao
hotwords, large-v3) nad svakim snimkom iz `import/`; `words` se keširaju u
`merenja/krug5/` (u .gitignore — sadržaj tuđih snimaka). Ništa se ne upisuje u
poslove. Nalaz: `merenja/dnevnik.md`.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from subtitis import asr, audio, config as cfg  # noqa: E402
from subtitis import hallucination as h  # noqa: E402
from subtitis.models import Word  # noqa: E402

S = ROOT / "merenja" / "krug5"  # u .gitignore
S.mkdir(parents=True, exist_ok=True)
IMPORT = ROOT / "import"


def report(name, raw_words, duration, lang):
    """Isto kao produkcija: francuska elizija spojena, pa `hallucination.zero_runs`.

    Bez spajanja elizije `c` + `'est` jedne reči broje se kao niz od dva tokena
    (snimak D, 83,76 s) — lažna uzbuna koje u produkciji nema.
    """
    words = [Word(**w) for w in raw_words]
    if asr.elision_language(lang):
        asr.merge_elisions(words, lang)
    real = [w for w in words if any(ch.isalnum() for ch in w.w)]
    longest = sorted(real, key=lambda w: w.e - w.s, reverse=True)[:3]
    print(f"\n== {name}: {duration:.1f} s, {len(real)} reči")
    for zero in (0.02, 0.05):
        for mr in (2, 3, 4):
            rs = h.zero_runs(words, zero=zero, min_run=mr)
            print(f"   prag {zero:.2f} s, niz >= {mr}: {len(rs)} niz(ova), "
                  f"{sum(len(r) for r in rs)} reči")
    for r in h.zero_runs(words):
        print(f"     NIZ {h.run_line(r)}")
    print("   najduže reči: " + ", ".join(f"{w.e - w.s:.2f} s" for w in longest))


def main():
    # 1) Radionica draft 1 — postojeći words.sr.json, bez ponovnog prepoznavanja
    b = json.load(open(ROOT / "jobs" / "20261001-175002-Radionica_draft-1" / "words.sr.json", encoding="utf-8"))
    report("Radionica draft 1 (postojeći posao)", b["words"], b["duration"], "sr")

    # 2) uredni snimci iz import/
    gloss = cfg.load_glossary()
    for src in sorted(IMPORT.glob("*.wav")):
        out_json = S / f"{src.stem}.words.json"
        if not out_json.exists():
            wav = S / f"{src.stem}.16k.wav"
            dur = audio.normalize(src, wav)
            model = asr.load_model(asr.DEFAULT_MODEL, asr.DEFAULT_DEVICE, asr.DEFAULT_COMPUTE_TYPE)
            from faster_whisper.audio import decode_audio
            lang, prob, _ = model.detect_language(decode_audio(str(wav))[: 30 * 16000])
            lang = "sr" if lang in ("hr", "bs", "sr") else lang
            t = time.perf_counter()
            wf = asr.transcribe(
                wav, job_id="probe", source_file=str(src), language=lang, duration=dur,
                glossary_terms=cfg.glossary_keys(gloss, lang), vad=False,
            )
            data = {"lang": lang, "lang_prob": round(float(prob), 2), "duration": dur,
                    "seconds": round(time.perf_counter() - t, 1),
                    "words": [w.model_dump() for w in wf.words]}
            json.dump(data, open(out_json, "w", encoding="utf-8"), ensure_ascii=False)
        data = json.load(open(out_json, encoding="utf-8"))
        report(f"{src.stem} ({data['lang']}, ASR {data['seconds']} s)", data["words"],
               data["duration"], data["lang"])
    asr.unload_model()


if __name__ == "__main__":
    main()
