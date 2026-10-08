"""Merenje ASR-a na više fajlova.

    .venv\\Scripts\\python scripts\\bench.py snimci\\*.mp3 --lang sr

Ispisuje: trajanje audija, vreme obrade, realtime faktor, broj reči, procenat
reči sa p < 0.6 i najduži niz uzastopnih (skoro) identičnih rečenica — jedini
pouzdan automatski detektor halucinacije. Model se učita jednom za sve fajlove.
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitis import asr, audio as audio_mod, config as cfg  # noqa: E402
from subtitis.config import SubtitisError  # noqa: E402
from subtitis.models import Sentence  # noqa: E402
from subtitis.segment import (  # noqa: E402
    LOW_COVERAGE_WARN,
    MIN_REPORTED_GAP,
    coverage,
    segment,
)

# --- Konstante za štelovanje ----------------------------------------------

#: Ispod ove ASR verovatnoće reč se broji kao sumnjiva.
LOW_CONFIDENCE = 0.6

#: Sličnost (difflib ratio) iznad koje su dve rečenice "gotovo identične".
REPEAT_SIMILARITY = 0.90

#: Duži niz od ovoga je petlja ponavljanja, ne stil govornika.
REPEAT_LOOP_THRESHOLD = 3

_NORM_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def expand(patterns: list[str]) -> list[Path]:
    """PowerShell ne širi globove kao bash — `D:\\snimci\\*.mp3` stiže kao
    bukvalan string, pa ga širimo sami."""
    out: list[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(pat)]
        if not hits:
            p = Path(pat)
            if p.exists():
                hits = [p]
            else:
                print(f"UPOZORENJE: ništa ne odgovara obrascu {pat!r}", file=sys.stderr)
        out.extend(sorted(hits))
    return out


# --- Detekcija petlji ponavljanja ------------------------------------------


def _norm(text: str) -> str:
    return _NORM_RE.sub("", text).lower().strip()


def longest_repeat_run(sentences: list[Sentence]) -> tuple[int, str]:
    """Najduži niz uzastopnih identičnih ili gotovo identičnih rečenica.

    Vraća (dužina niza, tekst koji se ponavlja). Preko `REPEAT_LOOP_THRESHOLD`
    to je petlja — Whisper se zaglavio i hrani se sopstvenim izlazom.
    """
    texts = [_norm(s.text) for s in sentences]
    best_run, best_text = (1, sentences[0].text) if sentences else (0, "")
    run, run_start = 1, 0
    for k in range(1, len(texts)):
        prev, cur = texts[k - 1], texts[k]
        similar = bool(cur) and (
            cur == prev or SequenceMatcher(None, prev, cur).ratio() >= REPEAT_SIMILARITY
        )
        run = run + 1 if similar else 1
        if not similar:
            run_start = k
        if run > best_run:
            best_run, best_text = run, sentences[run_start].text
    return best_run, best_text


def main() -> int:
    ap = argparse.ArgumentParser(description="Bench ASR-a na više fajlova.")
    ap.add_argument("files", nargs="+", help="Audio/video fajlovi ili glob obrasci.")
    ap.add_argument("--lang", default="sr")
    ap.add_argument("--model", default=asr.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument(
        "--no-vad", dest="vad", action="store_false",
        help="Isključi Silero VAD; za poređenje kad sumnjaš da jede govor.",
    )
    ap.add_argument("--out", default=None, help="Direktorijum za privremene WAV-ove.")
    args = ap.parse_args()

    try:
        audio_mod.check_environment()
    except SubtitisError as exc:
        print(f"\nGREŠKA: {exc}\n", file=sys.stderr)
        return 1

    files = expand(args.files)
    if not files:
        print("Nijedan fajl ne odgovara zadatim obrascima.", file=sys.stderr)
        return 1

    workdir = Path(args.out or (cfg.JOBS_DIR / "_bench"))
    workdir.mkdir(parents=True, exist_ok=True)

    gloss = cfg.load_glossary(None)
    terms = cfg.glossary_keys(gloss, args.lang)

    rows: list[tuple] = []
    total_audio = total_proc = 0.0

    for path in files:
        try:
            wav = workdir / f"{path.stem}.wav"
            dur = audio_mod.normalize(path, wav)

            t0 = time.perf_counter()
            wf = asr.transcribe(
                wav,
                job_id="_bench",
                source_file=str(path),
                language=args.lang,
                duration=dur,
                glossary_terms=terms,
                model_size=args.model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
                vad=args.vad,
            )
            proc = time.perf_counter() - t0
        except SubtitisError as exc:
            print(f"\nGREŠKA na {path.name}: {exc}\n", file=sys.stderr)
            continue

        n = len(wf.words)
        low = sum(1 for w in wf.words if w.p < LOW_CONFIDENCE)
        sents = segment(wf.words)
        run, run_text = longest_repeat_run(sents)
        cov, gaps = coverage(sents, dur)
        rows.append(
            (
                path.name,
                dur,
                proc,
                dur / proc if proc else 0.0,
                n,
                100.0 * low / n if n else 0.0,
                len(sents),
                run,
                run_text,
                cov,
                gaps,
                dict(wf.asr.rejections),
                wf.asr.detected_language,
                wf.asr.detected_probability,
                wf.asr.params.get("original_script"),
            )
        )
        total_audio += dur
        total_proc += proc

    asr.unload_model()

    hdr = (
        f"{'fajl':26s} {'audio':>8s} {'obrada':>8s} {'RTF':>6s} {'reči':>7s} "
        f"{'p<0.6':>7s} {'reč-ca':>7s} {'petlja':>7s} {'pokriv':>8s} {'odbač':>6s} "
        f"{'lang':>10s} {'pismo':>6s}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    loops: list[tuple[str, int, str]] = []
    holes: list[tuple[str, float, list[tuple[float, float]]]] = []
    rejected: list[tuple[str, dict]] = []
    for (
        name, dur, proc, rtf, n, pct, ns, run, run_text, cov, gaps, rej, lang, prob, script
    ) in rows:
        loop_mark = "!" if run > REPEAT_LOOP_THRESHOLD else " "
        cov_mark = "!" if cov < LOW_COVERAGE_WARN else " "
        dropped = sum(rej.get(k, 0) for k in asr.DROP_KEYS)
        print(
            f"{name[:26]:26s} {dur/60:7.1f}m {proc/60:7.1f}m {rtf:6.2f} {n:7d} "
            f"{pct:6.1f}% {ns:7d} {run:6d}{loop_mark} {cov:6.1f}%{cov_mark} {dropped:6d} "
            f"{str(lang):>5s}/{prob:.2f} {str(script):>6s}"
        )
        if run > REPEAT_LOOP_THRESHOLD:
            loops.append((name, run, run_text))
        if gaps:
            holes.append((name, dur, gaps))
        if rej:
            rejected.append((name, rej))
    print("-" * len(hdr))
    if total_proc:
        print(
            f"{'UKUPNO':28s} {total_audio/60:7.1f}m {total_proc/60:7.1f}m "
            f"{total_audio/total_proc:6.2f}"
        )

    print(f"\nVAD: {'uključen' if args.vad else 'ISKLJUČEN (--no-vad)'}")
    print("RTF > 1 znači brže od realnog vremena.")
    print(
        f"'petlja' = najduži niz uzastopnih (skoro) identičnih rečenica; "
        f"preko {REPEAT_LOOP_THRESHOLD} je halucinacija."
    )
    print(
        f"'pokriv' = procenat trajanja audija pokriven rečenicama; "
        f"ispod {LOW_COVERAGE_WARN:.0f}% je sumnjivo."
    )

    if holes:
        print(f"\nPRAZNINE > {MIN_REPORTED_GAP:.0f}s  (proveri ima li tu govora):")
        for name, dur, gaps in holes:
            total_gap = sum(b - a for a, b in gaps)
            print(
                f"  {name}  —  {len(gaps)} praznina, ukupno {total_gap:.1f}s "
                f"od {dur:.1f}s ({100.0 * total_gap / dur:.1f}%)"
            )
            for a, b in gaps:
                print(f"      {a:9.2f} -> {b:9.2f}   ({b - a:6.2f}s)")
        if args.vad:
            print("  Ako u prazninama ima govora, pojeo ih je VAD pre modela.")
            print("  Uporedi isti fajl sa --no-vad, ili spusti VAD_THRESHOLD u asr.py.")

    print("\nODBAČENI SEGMENTI (iz DEBUG loga faster-whisper-a):")
    if rejected:
        for name, rej in rejected:
            parts = [
                f"{key}={rej[key]}"
                for key, _, _ in asr.REJECTION_PATTERNS
                if rej.get(key)
            ]
            print(f"  {name}: {', '.join(parts)}")
    else:
        print("  nijedno zabeleženo")
    print(
        "  Kolona 'odbač' broji samo prozore odbačene po no_speech pragu — jedino\n"
        "  to faster-whisper loguje. `*_fallback` su ponovni pokušaji na višoj\n"
        "  temperaturi, ne gubitak audija.\n"
        "  FILTER HALUCINACIJA ODBACUJE SEGMENTE TIHO i nije obuhvaćen ovim brojem;\n"
        "  za njega koristi: scripts/asr_probe.py <fajl> --compare"
    )

    if loops:
        print("\nPETLJE PONAVLJANJA:")
        for name, run, run_text in loops:
            print(f"  {name}: {run}x  {run_text[:90]!r}")
        print(
            "  Proveri da je condition_on_previous_text=False i da temperature\n"
            "  fallback radi (asr.py, konstante na vrhu)."
        )
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
