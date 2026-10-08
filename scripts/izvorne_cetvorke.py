"""tests/izvorne_cetvorke.json iz izvornog materijala — heševi, ne tekst.

    .venv\\Scripts\\python scripts\\izvorne_cetvorke.py D:\\VAJB\\subtitis-arhiva [dodatni.txt ...]

Izvorni materijal su stvarni snimci; u javnom repou ih nema i ne sme ih biti.
Skripta čita privatnu kopiju (repo sa `jobs/` i `merenja/`) i upisuje samo
heševe četvorki uzastopnih reči:

* `jobs/**/*.json`, `merenja/**/*.json` — polja `text`, `src`, `asr_raw`,
  `ref`, `lines` i nizovi reči (`words` sa poljem `w`);
* `jobs/**/*.srt` — tekst svakog bloka;
* `eval/pairs.json` iz poslednjeg commita privatnog repoa;
* dodatni fajlovi sa komandne linije — jedna izjava po redu (npr. doslovni
  citati iz starog dnevnika merenja).

Četvorka od samih funkcijskih reči (`i da se u`, `parce que parce que`) se
izostavlja: nije izjava, a javlja se svuda; isto i četvorka iz kratkog spiska
`OPSTE`. Normalizacija je `zabranjen_pojam.reci()`, ista kao u testu.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zabranjen_pojam import ROOT, reci  # noqa: E402

N = 4
#: Dužina heša u heks-ciframa: 64 bita, sudar sa ~10⁶ četvorki repoa ~10⁻⁸.
HEX = 16

FUNKCIJSKE = set("""
a ako al ali bi bih bila bile bili bilo bio bismo biste bit biti ce cemo cete
ci da dok do ga gde i ih il ili im iz izmedu ja je jer jesam jesmo jeste joj
ju k kad kada kako kao koja koje koji kojih kojim koju kroz li ma me medu mene
mi mu na nad nam nas ne nego neki nema ni nije nisam nismo niti nju njoj njih
njim njima njoj njega njemu o od oko on ona one oni ono ova ove ovaj ovo pa po
pod pored pre preko pri prema sa sam samo se si smo su sta su sve ta taj tako
te ti to tu u uz va vam vas vi za zar zbog
a about after all also an and any are as at be been but by can could did do
does for from had has have he her him his how i if in into is it its me my no
not of on or our she so than that the their them then there these they this
those to us was we were what when where which who will with would you your
a au aux avec c ce ces cet cette d dans de des du elle elles en est et etre il
ils j je l la le les leur leurs lui m ma mais me mes moi n ne nous on ou par
pas pour qu que qui s sa se ses son sont sur t ta te tes toi ton tu un une vos
votre vous y parce comme donc quand si puis alors ya
""".split())

#: Fraze iz izvornog materijala koje nisu izjave, a nisu ni od samih
#: funkcijskih reči. Svaka četvorka koja je cela unutar jedne od njih se
#: izostavlja. Spisak mora ostati kratak: svaki unos je rupa u proveri.
OPSTE = [
    "s obzirom na to da",  # unos glosara, javlja se u kodu, testovima i merilu
    "c'est-à-dire",
    # Whisper-ove izmišljotine na tišini — iz trening podataka, ne iz snimka;
    # opisane su u README-u i u hallucination.py
    "Top 10 najboljih uvijeka na svijetu",
    "Sous-titrage ST' 501",
]


ZASTO = [
    "Nijedna četvorka uzastopnih reči iz izvornog materijala (stvarnih snimaka) ne sme se pojaviti u javnom repou.",
    "Spisak pojmova (zabranjeni_pojmovi.json) hvata nazive, ali izjava nema naziv: 23 doslovne rečenice su prošle pored njega. "
    "Ova provera ne zavisi od toga da li je neko unapred nabrojao šta je poverljivo.",
    "Ovde su samo heševi: prvih 16 heks-cifara SHA-256 od četiri normalizovane reči spojene razmakom "
    "(scripts/zabranjen_pojam.py: reci() — latinica, mala slova, bez dijakritike, samo slova i cifre).",
    "Fajl pravi scripts/izvorne_cetvorke.py iz privatne kopije sa snimcima; ručno se ne menja. "
    "Kad test padne, rečenicu napiši iznova — zamena imena ostavlja izjavu netaknutu.",
]


def _tekstovi_iz_jsona(o, out: list[str]) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("text", "src", "asr_raw", "ref") and isinstance(v, str):
                out.append(v)
            elif k == "lines" and isinstance(v, list) and all(isinstance(x, str) for x in v):
                out.append(" ".join(v))
            elif k == "words" and isinstance(v, list) and v and isinstance(v[0], dict) and "w" in v[0]:
                out.append(" ".join(str(x.get("w", "")) for x in v))
            else:
                _tekstovi_iz_jsona(v, out)
    elif isinstance(o, list):
        for x in o:
            _tekstovi_iz_jsona(x, out)


def izjave(arhiva: Path, dodatni: list[Path]) -> list[str]:
    out: list[str] = []
    for p in sorted([*arhiva.glob("jobs/**/*.json"), *arhiva.glob("merenja/**/*.json")]):
        try:
            _tekstovi_iz_jsona(json.loads(p.read_text(encoding="utf-8")), out)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    for p in sorted(arhiva.glob("jobs/**/*.srt")):
        for blok in p.read_text(encoding="utf-8-sig").split("\n\n"):
            out.append(" ".join(blok.strip().splitlines()[2:]))
    pairs = subprocess.run(["git", "show", "HEAD:eval/pairs.json"], cwd=arhiva,
                           capture_output=True, check=True).stdout.decode("utf-8")
    for par in json.loads(pairs)["pairs"]:
        out += [par[k] for k in ("src", "asr_raw", "ref") if k in par]
    for p in dodatni:
        out += p.read_text(encoding="utf-8").splitlines()
    return [t for t in dict.fromkeys(out) if t.strip()]


def cetvorke(text: str, n: int = N) -> set[tuple[str, ...]]:
    r = reci(text)
    return {tuple(r[i:i + n]) for i in range(len(r) - n + 1)}


def hes(c: tuple[str, ...]) -> str:
    return hashlib.sha256(" ".join(c).encode("utf-8")).hexdigest()[:HEX]


def main() -> None:
    arhiva, *dodatni = (Path(a) for a in sys.argv[1:])
    tekst = izjave(arhiva, dodatni)
    sve = set().union(*(cetvorke(t) for t in tekst))
    opste = set().union(*(cetvorke(f) for f in OPSTE))
    izostavljene = {c for c in sve if all(r in FUNKCIJSKE for r in c) or c in opste}
    hesevi = sorted({hes(c) for c in sve - izostavljene})
    dst = ROOT / "tests" / "izvorne_cetvorke.json"
    dst.write_text(json.dumps({
        "_zasto": ZASTO,
        "n": N, "hex": HEX, "h": hesevi,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(tekst)} izjava, {len(sve)} četvorki, {len(izostavljene)} izostavljeno (samo "
          f"funkcijske reči ili OPSTE), {len(hesevi)} heševa -> {dst}")


if __name__ == "__main__":
    main()
