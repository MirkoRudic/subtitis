"""Šta Whisper radi segment po segment — i koji ga prag obara.

    .venv\\Scripts\\python scripts\\asr_probe.py snimak.wav --lang sr
    .venv\\Scripts\\python scripts\\asr_probe.py snimak.wav --lang sr --window 79 90
    .venv\\Scripts\\python scripts\\asr_probe.py snimak.wav --lang sr --compare

Za svaki segment ispisuje vremena, `no_speech_prob`, `avg_logprob`,
`compression_ratio`, temperaturu na kojoj je prošao i početak teksta.

`--window START END` izdvaja i posebno prikazuje šta se dešava u tom intervalu.

`--compare` pušta više konfiguracija na istom fajlu i poredi dokle svaka stigne.
To je jedini pouzdan način da se utvrdi koji prag guta kraj snimka, jer filter
halucinacija odbacuje segmente TIHO (vidi napomenu ispod tabele).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitis import asr  # noqa: E402
from subtitis.audio import check_environment, normalize  # noqa: E402
from subtitis.config import JOBS_DIR, SubtitisError, glossary_keys, load_glossary  # noqa: E402
from subtitis.segment import MIN_REPORTED_GAP, coverage_of_spans  # noqa: E402

# --- Konstante za štelovanje ----------------------------------------------

#: Praznina između susednih segmenata koja se markira u tabeli.
SEGMENT_GAP_MARK = 3.0

#: Replika pragova iz faster-whisper `word_anomaly_score` — da se vidi koji bi
#: segment filter halucinacija proglasio sumnjivim. Mora da prati
#: faster_whisper/transcribe.py; ako se tamo promeni, promeni i ovde.
ANOMALY_LOW_PROBABILITY = 0.15
ANOMALY_SHORT_DURATION = 0.133
ANOMALY_LONG_DURATION = 2.0
ANOMALY_SCORE_LIMIT = 3.0

PUNCTUATION = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"


# --- Replika detektora anomalija -------------------------------------------


def word_anomaly_score(word) -> float:
    probability = getattr(word, "probability", 0.0) or 0.0
    duration = word.end - word.start
    score = 0.0
    if probability < ANOMALY_LOW_PROBABILITY:
        score += 1.0
    if duration < ANOMALY_SHORT_DURATION:
        score += (ANOMALY_SHORT_DURATION - duration) * 15
    if duration > ANOMALY_LONG_DURATION:
        score += duration - ANOMALY_LONG_DURATION
    return score


def segment_anomaly(segment) -> tuple[float, bool]:
    """(skor, da li bi ga filter halucinacija proglasio anomalijom)."""
    words = [w for w in (segment.words or []) if w.word.strip() not in PUNCTUATION]
    words = words[:8]
    if not words:
        return 0.0, False
    score = sum(word_anomaly_score(w) for w in words)
    return score, score >= ANOMALY_SCORE_LIMIT or score + 0.01 >= len(words)


# --- Jedan prolaz ----------------------------------------------------------


def run(wav: Path, args, *, overrides: dict | None = None):
    """Pusti transkripciju i vrati (segmenti, info, capture, iskorišćeni params)."""
    over = overrides or {}
    terms = glossary_keys(load_glossary(None), args.lang)
    prompt = ", ".join(terms) if terms else None

    params = {
        "language": args.lang,
        "word_timestamps": True,
        "vad_filter": over.get("vad", args.vad),
        "beam_size": args.beam_size,
        "initial_prompt": prompt,
        "hotwords": prompt,
        "condition_on_previous_text": asr.CONDITION_ON_PREVIOUS_TEXT,
        "compression_ratio_threshold": asr.COMPRESSION_RATIO_THRESHOLD,
        "log_prob_threshold": over.get("log_prob_threshold", args.log_prob_threshold),
        "no_speech_threshold": over.get("no_speech_threshold", args.no_speech_threshold),
        "temperature": list(asr.TEMPERATURE_FALLBACK),
        "hallucination_silence_threshold": over.get(
            "hallucination_silence_threshold", args.hallucination_silence_threshold
        ),
    }
    if params["vad_filter"]:
        params["vad_parameters"] = asr.vad_parameters()

    model = asr.load_model(args.model, args.device, args.compute_type)
    with asr.RejectionCapture() as capture:
        segments, info = model.transcribe(str(wav), **params)
        segments = list(segments)  # lenj generator; sve greške pucaju ovde
    return segments, info, capture, params


# --- Ispis -----------------------------------------------------------------


def print_segments(segments, duration: float, window: tuple[float, float] | None) -> None:
    hdr = (
        f"{'#':>4s} {'start':>9s} {'end':>9s} {'no_spch':>8s} {'avg_lp':>8s} "
        f"{'comp':>6s} {'temp':>5s} {'anom':>6s}  tekst"
    )
    print("\n" + hdr)
    print("-" * (len(hdr) + 30))

    prev_end = 0.0
    for seg in segments:
        gap = seg.start - prev_end
        if gap >= SEGMENT_GAP_MARK:
            print(f"     {'':>9s} {'':>9s}   ---- praznina {gap:.2f}s ----")
        score, anomaly = segment_anomaly(seg)
        in_win = (
            window is not None and seg.end > window[0] and seg.start < window[1]
        )
        mark = ">" if in_win else " "
        anom_mark = "!" if anomaly else " "
        text = seg.text.strip().replace("\n", " ")[:40]
        print(
            f"{mark}{seg.id:3d} {seg.start:9.2f} {seg.end:9.2f} {seg.no_speech_prob:8.3f} "
            f"{seg.avg_logprob:8.3f} {seg.compression_ratio:6.2f} {seg.temperature:5.1f} "
            f"{score:5.1f}{anom_mark} {text}"
        )
        prev_end = seg.end

    tail = duration - prev_end
    if tail >= SEGMENT_GAP_MARK:
        print(f"     {'':>9s} {'':>9s}   ---- praznina do kraja {tail:.2f}s ----")


def spans_of(segments) -> list[tuple[float, float]]:
    """Intervali za pokrivenost — po REČIMA kad ih ima.

    Segmentna vremena obuhvataju i tišinu unutar segmenta, pa daju lažno visoku
    pokrivenost. `transcribe` meri po rečima; ovde radimo isto da brojevi budu
    uporedivi.
    """
    spans: list[tuple[float, float]] = []
    for s in segments:
        if s.words:
            spans.extend((w.start, w.end) for w in s.words)
        else:
            spans.append((s.start, s.end))
    return spans


def summarize(segments, duration: float, capture) -> tuple[float, float]:
    cov, gaps = coverage_of_spans(spans_of(segments), duration, min_gap=MIN_REPORTED_GAP)
    last_end = max((s.end for s in segments), default=0.0)
    print(
        f"\nsegmenata: {len(segments)}   pokrivenost: {cov:.1f}%   "
        f"poslednji kraj: {last_end:.2f}s od {duration:.2f}s"
    )
    if gaps:
        print(f"praznine > {MIN_REPORTED_GAP:.0f}s:")
        for a, b in gaps:
            print(f"    {a:9.2f} -> {b:9.2f}   ({b - a:6.2f}s)")
    if capture.counts:
        print(f"odbacivanja iz loga: {capture.counts}")
        for m in capture.messages[:20]:
            print(f"    {m}")
    else:
        print("odbacivanja iz loga: nijedno")
    return cov, last_end


# --- Poređenje konfiguracija ----------------------------------------------

COMPARE_CONFIGS: list[tuple[str, dict]] = [
    ("kako je sada (asr.py)", {}),
    # Ovaj je ovde da se vidi KOLIKO odnosi, a ne kao kandidat za default.
    (
        "SA filterom halucinacija",
        {"hallucination_silence_threshold": asr.HALLUCINATION_SILENCE_SECONDS},
    ),
    ("bez no_speech praga", {"no_speech_threshold": None}),
    ("bez log_prob praga", {"log_prob_threshold": None}),
    (
        "bez oba praga",
        {"no_speech_threshold": None, "log_prob_threshold": None},
    ),
    ("bez VAD-a i bez oba praga", {
        "vad": False,
        "no_speech_threshold": None,
        "log_prob_threshold": None,
    }),
]


def compare(wav: Path, duration: float, args) -> None:
    print("\n" + "=" * 78)
    print("POREĐENJE KONFIGURACIJA")
    print("=" * 78)
    rows = []
    for name, over in COMPARE_CONFIGS:
        segments, _, capture, _ = run(wav, args, overrides=over)
        cov, _ = coverage_of_spans(spans_of(segments), duration, min_gap=MIN_REPORTED_GAP)
        last_end = max((s.end for s in segments), default=0.0)
        rows.append((name, len(segments), cov, last_end, capture.dropped))
        print(f"  {name:28s} -> poslednji kraj {last_end:7.2f}s, pokrivenost {cov:5.1f}%")

    hdr = f"{'konfiguracija':30s} {'segm':>5s} {'pokriv':>8s} {'posl. kraj':>11s} {'odbačeno':>9s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    base_last = rows[0][3]
    for name, nseg, cov, last_end, dropped in rows:
        delta = last_end - base_last
        mark = f"  (+{delta:.2f}s)" if delta > 0.5 else ""
        print(
            f"{name:30s} {nseg:5d} {cov:7.1f}% {last_end:10.2f}s {dropped:9d}{mark}"
        )
    print(
        "\nRed koji vrati znatno veći 'poslednji kraj' pokazuje koji prag guta kraj.\n"
        "'odbačeno' broji SAMO ono što faster-whisper loguje (no_speech prozori).\n"
        "Filter halucinacija ne loguje ništa — njega otkriva samo ovo poređenje.\n"
        "Zato je podrazumevano isključen (vidi HALLUCINATION_SILENCE_THRESHOLD)."
    )


# --- main ------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Dijagnostika Whisperovih segmenata.")
    ap.add_argument("audio")
    ap.add_argument("--lang", default="sr")
    ap.add_argument("--model", default=asr.DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    ap.add_argument("--beam-size", type=int, default=asr.DEFAULT_BEAM_SIZE)
    ap.add_argument("--no-vad", dest="vad", action="store_false")
    ap.add_argument(
        "--window", nargs=2, type=float, metavar=("START", "END"),
        help="Interval koji te zanima; segmenti u njemu se markiraju sa '>'.",
    )
    ap.add_argument("--no-speech-threshold", type=float, default=asr.NO_SPEECH_THRESHOLD)
    ap.add_argument("--log-prob-threshold", type=float, default=asr.LOG_PROB_THRESHOLD)
    ap.add_argument(
        "--hallucination-silence-threshold", type=float,
        default=asr.HALLUCINATION_SILENCE_THRESHOLD,
    )
    ap.add_argument(
        "--compare", action="store_true",
        help="Pusti više konfiguracija i uporedi dokle svaka stigne.",
    )
    args = ap.parse_args()

    try:
        check_environment()
        src = Path(args.audio)
        if not src.exists():
            raise SubtitisError(f"Nema fajla: {src}")

        workdir = JOBS_DIR / "_probe"
        workdir.mkdir(parents=True, exist_ok=True)
        wav = workdir / f"{src.stem}.wav"
        duration = normalize(src, wav)

        print(f"\n{src.name}   trajanje {duration:.2f}s   model {args.model}")
        print(
            f"pragovi: no_speech={args.no_speech_threshold} "
            f"log_prob={args.log_prob_threshold} "
            f"hallucination_silence={args.hallucination_silence_threshold} "
            f"vad={args.vad}"
        )

        segments, info, capture, params = run(wav, args)
        print(
            f"detektovan jezik: {info.language} ({info.language_probability:.2f})"
        )
        window = tuple(args.window) if args.window else None
        print_segments(segments, duration, window)
        summarize(segments, duration, capture)

        if window:
            inside = [s for s in segments if s.end > window[0] and s.start < window[1]]
            print(f"\nU intervalu {window[0]:.2f}–{window[1]:.2f}s: {len(inside)} segmenata")
            if not inside:
                print(
                    "  Nijedan segment. Whisper ga je odbacio ili preskočio.\n"
                    "  Pusti --compare da vidiš koja konfiguracija ga vraća."
                )
            for s in inside:
                score, anomaly = segment_anomaly(s)
                print(
                    f"  [{s.start:.2f}-{s.end:.2f}] no_speech={s.no_speech_prob:.3f} "
                    f"avg_lp={s.avg_logprob:.3f} comp={s.compression_ratio:.2f} "
                    f"temp={s.temperature:.1f} anomalija={score:.1f}"
                    f"{' (DA)' if anomaly else ''}"
                )
                print(f"      {s.text.strip()[:100]}")

        print(
            "\nNAPOMENA: faster-whisper NE izlaže odbačene segmente kroz API.\n"
            "Loguje samo no_speech odbacivanja i fallback-e (DEBUG, logger\n"
            "'faster_whisper'). Filter halucinacija briše segmente bez ijednog\n"
            "log poziva i ume da preskoči do kraja fajla — vidi --compare."
        )

        if args.compare:
            compare(wav, duration, args)

        asr.unload_model()
        return 0

    except SubtitisError as exc:
        print(f"\nGREŠKA: {exc}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
