"""Krug 22: merenje kruga 19 ponovljeno na sadašnjim profilima.

    .venv\\Scripts\\python scripts\\krug22_pauze.py

Ista mreža i ista mera kao `krug19_pauze.py meri` (funkcije su odatle); od
kruga 19 promenila se širina `web`/`digital` (20/10 -> 28/14, krug 21).
Razlike u odnosu na krug 19:

  - mere se samo poslovi koji imaju tišine (`pauses.json` ili `source.wav`);
    poslovi bez zvuka nemaju pauza i samo razblažuju brojke — od 15 poslova
    kruga 19 osam je u međuvremenu izgubilo zvuk po roku;
  - uz apsolutni broj cue-ova sa pauzom ispisuje se i UDEO svih cue-ova;
  - prolaz se ocenjuje po obliku odvojeno (`tv` dvoredni, `web` jednoredni —
    `digital` se prelama isto kao `web`, što se proverava, pa se ne meri dvaput);
  - za svaku tačku se ispisuje koliko je daleko od prolaza: zbir porasta mera
    koje ne smeju da porastu;
  - kontrola: `web` na staroj širini 20/10, nad istim poslovima, samo baza —
    da se vidi šta je promenila širina, a šta skup poslova.

Promenjene rečenice (tekst stvarnog govora) idu u `merenja/krug22/` — u
.gitignore. Na terminal se ne ispisuje nijedna reč — samo brojevi.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from krug6_digital import jobs  # noqa: E402
from krug19_pauze import meri, postavi  # noqa: E402
from subtitis import config as cfg  # noqa: E402
from subtitis import linebreak as lb  # noqa: E402
from subtitis import pauses as pz  # noqa: E402
from subtitis.models import SentencesFile, WordsFile  # noqa: E402

OUT = ROOT / "merenja" / "krug22"  # u .gitignore
CUVARI = ("predlog", "enklitika", "ispod", "cps", "red")
PRAGOVI = (0.12, 0.20, 0.30, 0.45)


def ucitaj():
    """Kao `krug19_pauze.ucitaj`, ali samo poslovi sa tišinama; `pauses.json`
    ima prednost nad ponovnim računanjem iz zvuka."""
    out = []
    for d in jobs():
        p = pz.load(d.name)
        if p is not None:
            sil = p.silences
        elif (d / "source.wav").exists():
            sil = pz.detect(d / "source.wav").silences
        else:
            continue
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
        out.append((d.name, sents, sil, terms))
    return out


def tacke():
    t = [(thr, w, d, 0) for thr in PRAGOVI
         for w in (0, 50, 100, 200) for d in (0, 25, 50) if (w, d) != (0, 0)]
    t += [(thr, 100, 25, x) for thr in (0.20, 0.30) for x in (10, 25)]
    return t


def udeo(m):
    return m["pauza"] / m["cues"] if m["cues"] else 0.0


def main():
    data = ucitaj()
    nrec = sum(len(s) for _, s, _, _ in data)
    print(f"{len(data)} ASR poslova sa tišinama, {nrec} rečenica\n")
    OUT.mkdir(parents=True, exist_ok=True)

    profili = {ime: cfg.load_profile(ime) for ime in ("tv", "web", "digital")}
    stari_web = profili["web"].model_copy(update={"max_chars_per_line": 20, "min_chars_per_line": 10})

    # `digital` se prelama kao `web`: proveri pre nego što se izostavi.
    postavi(0.30, 0, 0, 0)
    _, lw = meri(data, profili["web"], 0.30, False)
    _, ld = meri(data, profili["digital"], 0.30, False)
    assert lw == ld, "digital se više ne prelama kao web — meri ga zasebno"
    print("digital == web u prelomu (provereno), meri se jednom\n")

    print("Baza bez pauza — udeo cue-ova sa pauzom iznad praga")
    print(" profil        | cue-ova | " + " | ".join(f">= {t:.2f}" for t in PRAGOVI))
    for ime, prof in (("tv", profili["tv"]), ("web 28/14", profili["web"]), ("web 20/10 (kontrola)", stari_web)):
        cells = []
        for thr in PRAGOVI:
            m, _ = meri(data, prof, thr, False)
            cells.append(f"{m['pauza']:>3} ({udeo(m):5.1%})")
        print(f" {ime:<14}| {m['cues']:>7} | " + " | ".join(cells))
    print()

    rezultat = {}
    for ime in ("tv", "web"):
        prof = profili[ime]
        print(f"== {ime}")
        print(" prag  tež popust bez | cue-ova | sa pauzom (udeo)        | predlog | enkl. | ispod | CPS | red | prom. | daleko")
        baza = {}
        for thr in PRAGOVI:
            postavi(thr, 0, 0, 0)
            baza[thr] = meri(data, prof, thr, False)
            b = baza[thr][0]
            print(f" {thr:.2f}  baza          | {b['cues']:>7} | {b['pauza']:>3} ({udeo(b):5.1%})            | "
                  f"{b['predlog']:>7} | {b['enklitika']:>5} | {b['ispod']:>5} | {b['cps']:>3} | {b['red']:>3} |")
        for thr, w, d, x in tacke():
            postavi(thr, w, d, x)
            t, lay = meri(data, prof, thr, True)
            b, blay = baza[thr]
            ch = [k for k in lay if lay[k] != blay[k]]
            visak = {m: t[m] - b[m] for m in CUVARI if t[m] > b[m]}
            pada = t["pauza"] < b["pauza"]
            ok = pada and not visak
            daleko = "OK" if ok else (
                ("" if pada else "pauza ne pada; ")
                + ", ".join(f"{m} +{v}" for m, v in visak.items()))
            print(f" {thr:.2f} {w:>4} {d:>6} {x:>3} | {t['cues']:>7} | {t['pauza']:>3} ({udeo(t):5.1%}) {t['pauza'] - b['pauza']:+3}"
                  f" {udeo(t) - udeo(b):+6.1%} | {t['predlog']:>7} | {t['enklitika']:>5} | {t['ispod']:>5} | "
                  f"{t['cps']:>3} | {t['red']:>3} | {len(ch):>5} | {daleko}")
            rezultat[f"{ime} {thr} {w} {d} {x}"] = {
                "mere": t, "baza": b, "ok": ok, "visak": visak,
                "promene": {f"{n}#{sid}": {"pre": blay[(n, sid)], "posle": lay[(n, sid)]} for n, sid in ch},
            }
        print()
    postavi(0.30, 0, 0, 0)

    # Izbor po kriterijumu 3, po obliku odvojeno i za oba zajedno.
    def pad(k):
        r = rezultat[k]
        return r["baza"]["pauza"] - r["mere"]["pauza"]

    for ime in ("tv", "web"):
        prosli = [k for k in rezultat if k.startswith(ime + " ") and rezultat[k]["ok"]]
        print(f"{ime}: prolazi {len(prosli)} od {len(tacke())} tačaka", end="")
        if prosli:
            naj = max(prosli, key=lambda k: (pad(k), -float(k.split()[2])))
            print(f"; najveći pad: {naj} ({-pad(naj):+d})")
        else:
            print()
    oba = [k.split(" ", 1)[1] for k in rezultat
           if k.startswith("tv ") and rezultat[k]["ok"] and rezultat["web " + k.split(" ", 1)[1]]["ok"]]
    print(f"oba oblika: prolazi {len(oba)} tačaka" + (f": {', '.join(oba)}" if oba else ""))
    (OUT / "mreza.json").write_text(json.dumps(rezultat, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
