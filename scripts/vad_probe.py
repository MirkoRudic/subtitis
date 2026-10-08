"""Šta VAD radi tvom snimku — bez pokretanja Whispera.

Pusti Silero VAD sa više pragova na istom fajlu i pokaži šta svaki od njih
zadržava. Traje sekunde, pa se prag može naštelovati bez čekanja transkripcije.

    .venv\\Scripts\\python scripts\\vad_probe.py snimak.mp3
    .venv\\Scripts\\python scripts\\vad_probe.py snimak.mp3 --check 54.18 61.60
    .venv\\Scripts\\python scripts\\vad_probe.py snimak.mp3 --thresholds 0.2,0.3,0.35,0.5

`--check START END` je za slučaj kad si uvom našao govor koji je nestao:
ispiše koliko je od tog intervala svaki prag zadržao.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitis import asr  # noqa: E402
from subtitis.audio import check_environment  # noqa: E402
from subtitis.config import SubtitisError  # noqa: E402
from subtitis.segment import MIN_REPORTED_GAP, coverage_of_spans  # noqa: E402

SAMPLE_RATE = 16_000

#: Pragovi koji se porede kad --thresholds nije zadat.
DEFAULT_THRESHOLDS = [0.20, 0.30, asr.VAD_THRESHOLD, 0.50]


def overlap(spans: list[tuple[float, float]], a: float, b: float) -> float:
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)


def main() -> int:
    ap = argparse.ArgumentParser(description="Provera Silero VAD-a bez Whispera.")
    ap.add_argument("audio", help="Audio ili video fajl.")
    ap.add_argument(
        "--thresholds", default=None,
        help=f"Zarezom razdvojeni pragovi. Podrazumevano: {DEFAULT_THRESHOLDS}",
    )
    ap.add_argument(
        "--check", nargs=2, type=float, metavar=("START", "END"), action="append",
        help="Interval za koji znaš da ima govora; može više puta.",
    )
    ap.add_argument("--min-gap", type=float, default=MIN_REPORTED_GAP)
    ap.add_argument("--quiet", action="store_true", help="Bez spiska odsečaka.")
    args = ap.parse_args()

    try:
        check_environment()
        from faster_whisper.audio import decode_audio
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except SubtitisError as exc:
        print(f"\nGREŠKA: {exc}\n", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"\nGREŠKA: faster-whisper nije instaliran ({exc}).\n", file=sys.stderr)
        return 1

    path = Path(args.audio)
    if not path.exists():
        print(f"\nGREŠKA: nema fajla {path}\n", file=sys.stderr)
        return 1

    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    thresholds = (
        [float(t) for t in args.thresholds.split(",")]
        if args.thresholds
        else list(DEFAULT_THRESHOLDS)
    )

    print(f"\n{path.name}   trajanje {duration:.2f}s")
    print(
        f"fiksno: min_silence_duration_ms={asr.VAD_MIN_SILENCE_DURATION_MS}, "
        f"speech_pad_ms={asr.VAD_SPEECH_PAD_MS}\n"
    )

    hdr = f"{'prag':>6s} {'zadržano':>10s} {'pokriv':>8s} {'odsečaka':>9s} {'praznina':>9s}"
    if args.check:
        for a, b in args.check:
            hdr += f" {f'{a:.1f}-{b:.1f}':>12s}"
    print(hdr)
    print("-" * len(hdr))

    per_threshold: dict[float, list[tuple[float, float]]] = {}
    for thr in thresholds:
        opts = VadOptions(
            threshold=thr,
            min_silence_duration_ms=asr.VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=asr.VAD_SPEECH_PAD_MS,
        )
        chunks = get_speech_timestamps(audio, opts)
        spans = [(c["start"] / SAMPLE_RATE, c["end"] / SAMPLE_RATE) for c in chunks]
        per_threshold[thr] = spans

        kept = sum(b - a for a, b in spans)
        cov, gaps = coverage_of_spans(spans, duration, min_gap=args.min_gap)
        mark = "  <- asr.py" if abs(thr - asr.VAD_THRESHOLD) < 1e-9 else ""
        row = f"{thr:6.2f} {kept:9.2f}s {cov:7.1f}% {len(spans):9d} {len(gaps):9d}"
        if args.check:
            for a, b in args.check:
                got = overlap(spans, a, b)
                row += f" {100.0 * got / max(b - a, 1e-9):11.0f}%"
        print(row + mark)

    print("")
    for thr in thresholds:
        spans = per_threshold[thr]
        _, gaps = coverage_of_spans(spans, duration, min_gap=args.min_gap)
        if not gaps:
            continue
        print(f"prag {thr:.2f} — praznine > {args.min_gap:.0f}s:")
        for a, b in gaps:
            print(f"    {a:9.2f} -> {b:9.2f}   ({b - a:6.2f}s)")

    if not args.quiet:
        thr = asr.VAD_THRESHOLD
        print(f"\nodsečci koje zadržava prag {thr:.2f} (onaj iz asr.py):")
        for a, b in per_threshold.get(thr, []):
            print(f"    {a:9.2f} -> {b:9.2f}   ({b - a:6.2f}s)")

    print(
        "\nKolone --check pokazuju koliki DEO tog intervala je VAD zadržao.\n"
        "Ako je tu govor a broj je nizak, spusti VAD_THRESHOLD u subtitis/asr.py.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
