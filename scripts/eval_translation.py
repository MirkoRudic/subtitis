"""Poređenje modela na ručno pripremljenim parovima.

    .venv\\Scripts\\python scripts\\eval_translation.py
    .venv\\Scripts\\python scripts\\eval_translation.py --model gemma3:12b
    .venv\\Scripts\\python scripts\\eval_translation.py --model qwen3:14b --model gemma3:12b

Učita `eval/pairs.json`, pusti rečenice kroz zadati model i ispiše
original / referencu / model, jedan par po redu, za RUČNU ocenu.

Bez BLEU — na 25 rečenica je beskorisan. Jedini automatski broj koji se ovde
prijavljuje je da li je model uopšte vratio ispravan broj linija; to je kvar
pajplajna, ne ocena kvaliteta.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitis import config as cfg  # noqa: E402
from subtitis import repeats as rep  # noqa: E402
from subtitis import translate as tr  # noqa: E402
from subtitis.config import SubtitisError  # noqa: E402

DEFAULT_PAIRS = Path(__file__).resolve().parents[1] / "eval" / "pairs.json"

#: Širina kolone pri ispisu. Duži tekst se prelama, ne seče.
WRAP = 88

#: Interni ključ po kom se prevod spaja sa parom. Postavlja ga `load_pairs`:
#: `id` iz fajla ako postoji, inače pozicija u nizu (od 1). Spajanje mora ići
#: SAMO preko njega — dva različita podrazumevanja su već napravila tihi kvar
#: gde je svaki red pokazivao "(nedostaje)".
KEY = "_key"


def load_pairs(data: dict) -> tuple[list[dict], str, str, str | None]:
    """Normalizuj `pairs.json`. Vrati (parovi, izvorni jezik, ciljni, varijanta).

    Svaki par dobija `_key`: `id` iz fajla ako ga ima, inače redni broj od 1.
    Sve što bi kasnije dalo tihe rupe u tabeli ovde puca odmah.
    """
    pairs = data.get("pairs", [])
    if not isinstance(pairs, list) or not pairs:
        raise SubtitisError(
            "nema nijedan par. Dodaj ih u niz 'pairs' "
            "(polja: src, ref; opciono id, why, asr_raw)."
        )

    # `src_lang`/`tgt_lang` su glavna imena; `src`/`tgt` su podržana zbog
    # starijih fajlova. Bez ovoga bi se jezik iz fajla tiho ignorisao.
    src_lang = data.get("src_lang") or data.get("src") or "sr"
    tgt_lang = data.get("tgt_lang") or data.get("tgt") or "en"
    variant = data.get("variant")

    out: list[dict] = []
    seen: dict[int, int] = {}
    for pos, p in enumerate(pairs, start=1):
        if not isinstance(p, dict) or not str(p.get("src", "")).strip():
            raise SubtitisError(f"par na poziciji {pos} nema neprazno polje 'src'.")
        raw = p.get("id")
        if raw is None:
            key = pos
        else:
            try:
                key = int(raw)
            except (TypeError, ValueError):
                raise SubtitisError(
                    f"par na poziciji {pos} ima 'id' koji nije ceo broj: {raw!r}."
                ) from None
        if key in seen:
            raise SubtitisError(
                f"par na poziciji {pos} ima isti id ({key}) kao par na poziciji "
                f"{seen[key]}. Redni brojevi moraju biti jedinstveni."
            )
        seen[key] = pos
        out.append({**p, KEY: key})
    return out, src_lang, tgt_lang, variant


def missing_keys(result: dict[int, str], pairs: list[dict]) -> list[int]:
    """Ključevi parova za koje prevod nije stigao."""
    return [p[KEY] for p in pairs if p[KEY] not in result]


def wrap(text: str, label: str, width: int = WRAP) -> list[str]:
    """Prvi red nosi oznaku, nastavci idu uvučeni pod nju."""
    pad = " " * len(label)
    body = width - len(label)
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > body:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    if not lines:
        lines = [""]
    return [(label if k == 0 else pad) + l for k, l in enumerate(lines)]


def repeat_verdicts(
    src_reps: list, tgt_reps: list, kind: str = rep.HESITATION
) -> tuple[int, int, int, int]:
    """Po REČENICI: (preneto, uklonjeno, dodato, ukupno rečenica sa udvajanjem).

    Zbir kroz ceo fajl ume da prevari: model može da ukloni jedno udvajanje i
    doda drugo na drugom mestu, a zbir ostane isti. Ova četiri broja se broje
    par po par i zato odgovaraju na pitanje koje je stvarno postavljeno.

    Gleda se JEDNA vrsta odjednom. Preneto pojačavanje i preneto oklevanje su
    suprotni ishodi — sabrani, ponište jedan drugog i broj ne znači ništa.
    """
    carried = removed = added = had = 0
    for s, t in zip(src_reps, tgt_reps):
        if s.of_kind(kind):
            had += 1
            if t.of_kind(kind):
                carried += 1
            else:
                removed += 1
        elif t.of_kind(kind):
            added += 1
    return carried, removed, added, had


def print_repeats(
    pairs: list[dict],
    results: dict[str, dict[int, str]],
    src_lang: str,
    tgt_lang: str,
) -> None:
    """Tabela uzastopnih udvajanja: original, referenca, svaki model.

    Ovo je čisto merenje. Ne postoji prag koji ovde nešto proglašava kvarom —
    postoji da bi se videlo da li model govorno OKLEVANJE prenosi, uklanja ili
    dodaje svoje, pre nego što se zbog toga dira prompt. Pojačavanje se prikazuje
    uz njega, ali se u presudu ne meša: ono treba da se prenese.
    """
    src_reps = [rep.measure(p["src"], src_lang) for p in pairs]
    ref_reps = [rep.measure(p.get("ref", ""), tgt_lang) for p in pairs]

    print("=" * WRAP)
    print("UZASTOPNA UDVAJANJA (ista reč ili ista dvočlana fraza, za redom)")
    print("-" * WRAP)
    print(
        f"{'':18s} {'oklev.':>7s} {'pojač.':>7s} {'nerazv.':>8s} "
        f"{'rečenica':>9s} {'/100 reči':>10s} {'vs orig':>8s}"
    )

    def line(label: str, reps: list, base: int | None = None) -> None:
        t = sum(reps, rep.Repeats())
        vs = f"{t.hesitation / base:7.2f}x" if base else " " * 8
        share = f"{t.texts_with_hesitation}/{t.texts}"
        print(
            f"{label:18s} {t.hesitation:7d} {t.emphasis:7d} {t.unclassified:8d} "
            f"{share:>9s} {t.per_100_words:10.2f} {vs}"
        )
        for kind in (rep.HESITATION, rep.EMPHASIS, rep.UNCLASSIFIED):
            if t.of_kind(kind):
                print(f"{'':20s}{kind}: {t.examples_line(kind)}")

    src_total = sum(src_reps, rep.Repeats()).hesitation
    line("ORIGINAL", src_reps)
    if any(p.get("ref") for p in pairs):
        line("REFERENCA", ref_reps, src_total)
    for model in results:
        model_reps = [rep.measure(results[model][p[KEY]], tgt_lang) for p in pairs]
        line(model, model_reps, src_total)
        for kind in (rep.HESITATION, rep.EMPHASIS):
            carried, removed, added, had = repeat_verdicts(src_reps, model_reps, kind)
            if had or added:
                print(
                    f"{'':20s}{kind} po rečenici: preneto {carried}/{had}, "
                    f"uklonjeno {removed}/{had}, dodato na čisto mesto {added}"
                )

    unknown = [l for l in (src_lang, tgt_lang) if not rep.known_language(l)]
    if unknown:
        print(
            f"\nPOZOR: nema spiska reči za {', '.join(sorted(set(unknown)))} — "
            "za taj jezik sve pada u 'nerazv.'\ni podela oklevanje/pojačavanje "
            "u ovoj tabeli ne važi."
        )

    print(
        "\nOva tabela ništa ne odlučuje — samo meri. Presuđuje se po OKLEVANJU:\n"
        "'preneto' je broj rečenica u kojima ga je bilo u originalu i ostalo i u\n"
        "prevodu. Ako je ono većina, pravilo 9 ne pokriva govorno oklevanje; ako\n"
        "nije, pojedinačan slučaj nije razlog da prompt raste. Preneto POJAČAVANJE\n"
        "je suprotno — ono je znak da je prevod tačan, i ne traži nikakvu izmenu.\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Ručna ocena prevoda po modelima.")
    ap.add_argument(
        "--model", action="append", default=None,
        help="Ollama model; može više puta radi poređenja.",
    )
    ap.add_argument("--pairs", default=str(DEFAULT_PAIRS))
    ap.add_argument("--url", default=cfg.DEFAULT_OLLAMA_URL)
    ap.add_argument(
        "--chunk-size", type=int, default=tr.CHUNK_SIZE,
        help="Rečenica po zahtevu; 1 izoluje svaku rečenicu od konteksta.",
    )
    ap.add_argument("--glossary", default=None)
    ap.add_argument(
        "--variant", default=None,
        help="Nadjačaj varijantu iz pairs.json ('en-GB'/'en-US'/'off').",
    )
    ap.add_argument(
        "--show-prompt", action="store_true",
        help="Ispiši sistemski prompt — provera da glosar i varijanta stvarno stižu.",
    )
    args = ap.parse_args()

    models = args.model or [cfg.DEFAULT_LLM_MODEL]

    try:
        data = cfg.read_json(Path(args.pairs))
        pairs, src_lang, tgt_lang, variant = load_pairs(data)
    except SubtitisError as exc:
        print(f"\nGREŠKA u {args.pairs}: {exc}\n", file=sys.stderr)
        return 1

    gloss = cfg.load_glossary(args.glossary)
    glossary_pairs = cfg.glossary_pairs(gloss, src_lang, tgt_lang)

    # Rečenice nose isti ključ kao parovi — spajanje ide samo preko njega.
    from subtitis.models import Sentence

    sentences = [
        Sentence(
            id=p[KEY],
            s=float(k),
            e=float(k) + 1.0,
            text=p["src"],
            words=(k, k),
        )
        for k, p in enumerate(pairs, start=1)
    ]

    if args.variant is not None:
        variant = None if args.variant.strip().lower() in ("off", "none", "-") else args.variant

    print(f"\n{args.pairs}: {len(pairs)} parova, {src_lang} -> {tgt_lang}", end="")
    print(f", varijanta {variant}" if variant else "")
    print(f"  glosar {src_lang}->{tgt_lang}: {len(glossary_pairs)} unosa")
    if not glossary_pairs:
        print("  UPOZORENJE: glosar je prazan za ovaj smer — prompt ide bez rečnika.")
    if variant and not tr.variant_rule(variant):
        print(
            f"  UPOZORENJE: varijanta '{variant}' nije poznata "
            f"({', '.join(sorted(tr.VARIANT_RULES))}) — pravilo o pravopisu se ne dodaje."
        )
    if args.show_prompt:
        print("\n--- SISTEMSKI PROMPT ---")
        print(tr.build_system_prompt(src_lang, tgt_lang, glossary_pairs, variant))
        print("--- kraj prompta ---")

    results: dict[str, dict[int, str]] = {}
    timings: dict[str, float] = {}
    problems: dict[str, list[int]] = {}
    broken: list[str] = []
    call_stats: dict[str, tr.CallStats] = {}
    residency: dict[str, dict | None] = {}

    for model in models:
        print(f"\n>>> {model} ...", flush=True)
        client = tr.OllamaClient(url=args.url, model=model, log=lambda m: print(f"    {m}"))
        try:
            client.check()
            t0 = time.perf_counter()
            translated, stats = tr.translate_sentences(
                sentences, src_lang, tgt_lang,
                client=client,
                glossary_pairs=glossary_pairs,
                variant=variant,
                chunk_size=args.chunk_size,
                log=lambda m: print(f"    {m}"),
            )
            timings[model] = time.perf_counter() - t0
            residency[model] = client.residency()  # pre oslobađanja iz VRAM-a
        except SubtitisError as exc:
            print(f"\nGREŠKA ({model}): {exc}\n", file=sys.stderr)
            continue
        finally:
            client.unload()
        call_stats[model] = stats.calls

        result = {s.id: s.text for s in translated}
        print(
            f"    {stats.calls.requests} zahteva, "
            f"čankova ok/pao {stats.chunks_ok}/{stats.chunks_failed}, "
            f"{timings.get(model, 0):.1f}s"
        )

        # Spajanje mora da se poklopi. Ako ne, tabela bi bila puna "(nedostaje)"
        # a prevod bi izgledao kao da je uspeo — tiho, i zato gore od pada.
        missing = missing_keys(result, pairs)
        extra = sorted(set(result) - {p[KEY] for p in pairs})
        if missing or extra:
            print(
                f"\nGREŠKA: spajanje prevoda sa parovima nije uspelo ({model}).\n"
                f"  parova: {len(pairs)}, prevedenih linija: {len(result)}\n"
                f"  nedostaju ključevi: {missing}\n"
                f"  višak ključeva: {extra}\n"
                "  Ovo je bug u eval skripti ili nekonzistentni 'id' u pairs.json,\n"
                "  ne rezultat prevoda. Model se izostavlja iz tabele.\n",
                file=sys.stderr,
            )
            broken.append(model)
            continue

        results[model] = result
        problems[model] = list(stats.failed_ids)

    if not results:
        print("\nNijedan model nije dao upotrebljiv rezultat.\n", file=sys.stderr)
        return 1

    print("\n" + "=" * WRAP)
    print("PAROVI ZA RUČNU OCENU")
    print("=" * WRAP)
    width = max([len(m) for m in results] + [len("REFERENCA")]) + 1

    def row(label: str, text: str, mark: str = " ", lang: str | None = None) -> None:
        # Udvajanja se obeležavaju na samom redu: o = oklevanje, p = pojačavanje.
        # Bez toga se u tabeli vidi samo zbir na dnu, a ne KOJI par ga je doneo
        # — a upravo to je pitanje.
        r = rep.measure(text, lang)
        tag = f" o{r.hesitation}p{r.emphasis}" if r.total else " " * 5
        head = f" {mark}{label:<{width}} {len(text):4d}{tag}  "
        print(*wrap(text, head), sep="\n")

    for p in pairs:
        pid = p[KEY]
        print(f"\n[{pid}]")
        row("ORIGINAL", p["src"], lang=src_lang)
        if p.get("asr_raw") and p["asr_raw"] != p["src"]:
            row("ASR", p["asr_raw"], lang=src_lang)
        row("REFERENCA", p.get("ref", "(nema reference)"), lang=tgt_lang)
        for model in results:
            mark = "!" if pid in problems.get(model, []) else " "
            row(model, results[model][pid], mark, lang=tgt_lang)
        if p.get("why"):
            print(*wrap(p["why"], "    -> "), sep="\n")

    src_chars = sum(len(p["src"]) for p in pairs)
    ref_chars = sum(len(p.get("ref", "")) for p in pairs)

    print("\n" + "=" * WRAP)
    print(
        f"{'':18s} {'znakova':>8s} {'vs ref':>7s} {'vs orig':>8s} "
        f"{'izlaz t/s':>10s} {'prompt t/s':>11s} {'vreme':>7s} {'neprev.':>8s}"
    )
    print("-" * WRAP)
    print(f"{'ORIGINAL':18s} {src_chars:8d}")
    if ref_chars:
        print(f"{'REFERENCA':18s} {ref_chars:8d} {'':>7s} {ref_chars / src_chars:7.2f}x")
    for model in results:
        total = sum(len(results[model][p[KEY]]) for p in pairs)
        st = call_stats[model]
        vs_ref = f"{total / ref_chars:6.2f}x" if ref_chars else " " * 7
        print(
            f"{model:18s} {total:8d} {vs_ref} {total / src_chars:7.2f}x "
            f"{st.output_tps:10.1f} {st.prompt_tps:11.1f} "
            f"{timings.get(model, 0):6.1f}s {len(problems.get(model, [])):8d}"
        )
    print(
        "\n'vs ref' preko 1.00 znači da je model duži od tvoje reference — diže CPS\n"
        "i pravi više cue-ova nego što treba. 'neprev.' je kvar pajplajna (model nije\n"
        "vratio ispravan broj linija), ne ocena kvaliteta.\n"
        "Tokeni u sekundi su mera brzine; 'vreme' sadrži i učitavanje modela u VRAM.\n"
    )

    print_repeats(pairs, results, src_lang, tgt_lang)

    for model, res in residency.items():
        if not res:
            continue
        gb = 1024 ** 3
        size, vram = res.get("size") or 0, res.get("size_vram") or 0
        pct = f"{100 * vram / size:.0f}%" if size else "?"
        flag = "  <- DEO U RAM-u" if size and vram < size else ""
        print(
            f"  {model:18s} size {size / gb:5.2f} GB   size_vram {vram / gb:5.2f} GB "
            f"({pct})   ctx {res.get('context_length')}{flag}"
        )
    print("")
    if broken:
        print(f"NEUSPELO SPAJANJE: {', '.join(broken)} — vidi grešku iznad.\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
