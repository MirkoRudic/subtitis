"""Krug 24: kazna za rez iza lične zamenice, i filter pauze u dvorednom prelomu.

    .venv\\Scripts\\python scripts\\krug24_zamenica.py

1. Kazna sama (`PENALTY_SPLIT_PRONOUN`), bez pauza: koliko rečenica menja u
   `tv` i `web` — to je opšta izmena preloma i čita se zasebno.
2. Tačka kruga 23 u `tv` (prag 0,12, kazna za pauzu u cue-u 100, popust 50,
   `PAUSE_AT_FUNCTION_WORD = "discard"`) uz istu kaznu za zamenicu: šta se
   menja prema bazi sa istom kaznom i bez pauza. Promene idu u
   `merenja/krug24/` (u .gitignore) za čitanje.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug22_pauze import ucitaj  # noqa: E402
from krug23_lazna_pauza import meri, postavi  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402

OUT = ROOT / "merenja" / "krug24"
KAZNE = (0, 25, 50, 100)
TACKA = (0.12, 100, 50, 0)


def main():
    data = ucitaj()
    print(f"{len(data)} ASR poslova sa tišinama\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rezultat = {}
    for ime in ("tv", "web"):
        prof = cfg.load_profile(ime)
        lb.PENALTY_SPLIT_PRONOUN = 0
        postavi(0.30, 0, 0, 0)
        _, lay0 = meri(data, prof, 0.30, False)
        print(f"== {ime}: kazna sama, bez pauza")
        for k in KAZNE[1:]:
            lb.PENALTY_SPLIT_PRONOUN = k
            t, lay = meri(data, prof, 0.30, False)
            ch = {f"{n}#{s}": {"pre": lay0[(n, s)], "posle": lay[(n, s)]} for n, s in lay if lay[(n, s)] != lay0[(n, s)]}
            print(f" kazna {k:>3}: promenjenih rečenica {len(ch)} | cue-ova {t['cues']} ispod {t['ispod']} CPS {t['cps']}")
            rezultat[f"{ime} sama {k}"] = ch
    prof = cfg.load_profile("tv")
    print("\n== tv, tačka kruga 23 + discard, prema bazi sa istom kaznom")
    for k in KAZNE:
        lb.PENALTY_SPLIT_PRONOUN = k
        postavi(TACKA[0], 0, 0, 0)
        b, blay = meri(data, prof, TACKA[0], False)
        postavi(*TACKA, "discard")
        t, lay = meri(data, prof, TACKA[0], True)
        ch = {f"{n}#{s}": {"pre": blay[(n, s)], "posle": lay[(n, s)]} for n, s in lay if lay[(n, s)] != blay[(n, s)]}
        zam = sum(1 for v in ch.values() for c in v["posle"][:-1]
                  if lb._norm(c.replace(" / ", " ").split()[-1]) in lb.SR_PERSONAL_PRONOUNS)
        print(f" kazna {k:>3}: promenjenih {len(ch)} | cue-ova {b['cues']} -> {t['cues']} | ispod {b['ispod']} -> {t['ispod']}"
              f" | CPS {b['cps']} -> {t['cps']} | natpis završen zamenicom (posle) {zam}")
        rezultat[f"tv tacka {k}"] = ch
    lb.PENALTY_SPLIT_PRONOUN = 0
    postavi(0.30, 0, 0, 0)
    (OUT / "zamenica.json").write_text(json.dumps(rezultat, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
