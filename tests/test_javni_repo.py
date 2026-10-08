"""Javni repozitorijum: ništa iz stvarnog materijala ni u jednom praćenom fajlu.

Dve provere nad `git ls-files` (ne nad unapred nabrojanim direktorijumima — da
uhvate i fajl koji tek bude dodat):

* po spisku: nijedan zabranjen pojam (tests/zabranjeni_pojmovi.json) — nazivi
  organizacija, imena, imena projekata;
* po pravilu: nijedna četvorka uzastopnih reči iz izvornog materijala
  (tests/izvorne_cetvorke.json). Spisak hvata nazive, ali izjava nema naziv;
  ova provera ne zavisi od toga da li je neko unapred nabrojao šta je
  poverljivo.

Oba spiska su heširana (vidi `_zasto` u svakom); normalizacija je jedna,
`reci()` iz scripts/zabranjen_pojam.py.

Pre pretrage se uklanjaju ugrađeni `data:` blokovi (base64 slike u HTML-u
dokumenta): u njima se slučajno pojavljuju nizovi slova koji liče na pojmove,
a provera koja pada bez razloga je gora od provere koje nema.

PDF se proverava preko izvučenog teksta (`pypdf`, ili `pdftotext` iz PATH-a
ili iz Git for Windows). Kad alata nema, `test_pdf_je_proveren` to prijavljuje
imenom fajla — PDF se nikad ne preskače ćutke.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPISAK = json.loads((ROOT / "tests" / "zabranjeni_pojmovi.json").read_text(encoding="utf-8"))
CETVORKE = json.loads((ROOT / "tests" / "izvorne_cetvorke.json").read_text(encoding="utf-8"))
SPISKOVI = {"zabranjeni_pojmovi.json", "izvorne_cetvorke.json"}

_spec = importlib.util.spec_from_file_location("zabranjen_pojam", ROOT / "scripts" / "zabranjen_pojam.py")
_zp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_zp)
reci = _zp.reci


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def pracene_datoteke() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("nije git repozitorijum")
    return [ROOT / p for p in out.stdout.decode("utf-8").split("\0") if p]


_DATA = re.compile(r"data:[\w/+.-]*(?:;[\w=.-]+)*,[^\s\"')<>]*")


def bez_data(text: str) -> str:
    """Ugrađeni `data:` blokovi (base64) nisu tekst — vidi docstring modula."""
    return _DATA.sub(" ", text)


def _pdftotext() -> str | None:
    found = shutil.which("pdftotext")
    if found:
        return found
    git = shutil.which("git")  # Git for Windows nosi pdftotext u mingw64/bin
    if git:
        for base in Path(git).resolve().parents:
            cand = base / "mingw64" / "bin" / "pdftotext.exe"
            if cand.exists():
                return str(cand)
    return None


def pdf_tekst(p: Path) -> str | None:
    """Tekst PDF-a, ili None kad nema alata kojim bi se izvukao."""
    try:
        from pypdf import PdfReader
    except ImportError:
        pass
    else:
        return "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
    exe = _pdftotext()
    if exe is None:
        return None
    out = subprocess.run([exe, "-enc", "UTF-8", str(p), "-"], capture_output=True, check=True)
    return out.stdout.decode("utf-8")


NEPROVERENI_PDF: list[str] = []


def tekstovi() -> dict[str, str]:
    out = {}
    NEPROVERENI_PDF.clear()
    for p in pracene_datoteke():
        if p.name in SPISKOVI or not p.exists():
            continue
        rel = str(p.relative_to(ROOT))
        if p.suffix.lower() == ".pdf":
            t = pdf_tekst(p)
            if t is None:
                NEPROVERENI_PDF.append(rel)
            else:
                out[rel] = t
            continue
        try:
            out[rel] = bez_data(p.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            continue  # binarni fajl bez teksta: ikonica, fontovi
    return out


def nadjeni(text: str) -> list[str]:
    """Opisi pogodaka (bez samog pojma): `celo/3 @ reč 120`, `osnova @ reč 7`."""
    r = reci(text)
    celo = {}
    for e in SPISAK["celo"]:
        celo.setdefault(e["reci"], set()).add(e["h"])
    osnove = {}
    for e in SPISAK["osnova"]:
        osnove.setdefault(e["duzina"], set()).add(e["h"])
    out = []
    for n, hs in celo.items():
        for i in range(len(r) - n + 1):
            if _h(" ".join(r[i:i + n])) in hs:
                out.append(f"celo/{n} @ reč {i}")
    for i, w in enumerate(r):
        for d, hs in osnove.items():
            if len(w) >= d and _h(w[:d]) in hs:
                out.append(f"osnova @ reč {i}")
    return out


def izvorne(text: str) -> list[int]:
    """Položaji (redni broj reči) na kojima počinje četvorka iz izvornog materijala."""
    n, k, hs = CETVORKE["n"], CETVORKE["hex"], set(CETVORKE["h"])
    r = reci(text)
    return [i for i in range(len(r) - n + 1) if _h(" ".join(r[i:i + n]))[:k] in hs]


def test_spiskovi_nisu_prazni():
    assert len(SPISAK["celo"]) >= 25 and SPISAK["osnova"], "bez spiska test ne štiti ništa"
    assert len(CETVORKE["h"]) >= 1000 and CETVORKE["n"] == 4, "bez četvorki test ne štiti ništa"


def test_nijedan_zabranjen_pojam_u_pracenim_fajlovima():
    los = {rel: len(h) for rel, t in tekstovi().items() if (h := nadjeni(t))}
    assert not los, f"zabranjeni pojmovi: {los}"


def test_nijedna_cetvorka_iz_izvornog_materijala():
    los = {rel: izvorne(t) for rel, t in tekstovi().items()}
    los = {rel: f"{len(p)} (od reči {p[0]})" for rel, p in los.items() if p}
    assert not los, f"četvorke iz izvornog materijala — rečenicu napiši iznova: {los}"


def test_pdf_je_proveren():
    pdfs = [p for p in pracene_datoteke() if p.suffix.lower() == ".pdf"]
    tekst = tekstovi()
    if NEPROVERENI_PDF:
        pytest.skip(
            f"PDF NIJE PROVEREN: {', '.join(NEPROVERENI_PDF)} — nema alata za izvlačenje teksta. "
            "Ručno: `pip install pypdf` (ili Poppler `pdftotext` u PATH) pa ponovo pokreni "
            "tests/test_javni_repo.py; obe provere tada idu i preko PDF-a."
        )
    for p in pdfs:
        rel = str(p.relative_to(ROOT))
        assert len(tekst[rel]) > 1000, f"{rel}: izvučeno premalo teksta — PDF bez tekstualnog sloja se ne može proveriti"


def test_data_blokovi_se_ne_pretrazuju():
    """Pojam usred base64 slike nije pogodak; isti pojam van nje jeste."""
    global SPISAK
    stari = SPISAK
    SPISAK = {"celo": [{"reci": 1, "h": _h("kyx")}], "osnova": []}
    try:
        slika = '<img src="data:image/png;base64,iVBORw0KGgo+Kyx/AAAN">'
        assert not nadjeni(bez_data(slika))
        assert nadjeni(bez_data(slika + " Kyx"))
        assert "Kyx" in bez_data("url(data:font/woff2;base64,AAAA) Kyx")
    finally:
        SPISAK = stari


def test_pretraga_vidi_pismo_velicinu_dijakritiku_i_padez():
    """Na izmišljenom pojmu: pravila pretrage, ne sadržaj spiska."""
    global SPISAK
    stari = SPISAK
    SPISAK = {"celo": [{"reci": 2, "h": _h("izmisljena firma")}],
              "osnova": [{"duzina": 7, "h": _h("izmislj")}]}
    try:
        assert nadjeni("IZMIŠLJENA  Firma") and nadjeni("Измишљена фирма")
        assert nadjeni("izmisljena firma"), "bez dijakritike"
        assert nadjeni("u izmišljenoj kući"), "osnova hvata padež"
        assert not nadjeni("izmisl"), "kraće od osnove ne hvata"
    finally:
        SPISAK = stari


def test_cetvorke_ne_zavise_od_pisma_velicine_dijakritike_i_interpunkcije():
    """Na izmišljenoj rečenici: pravila pretrage, ne sadržaj spiska."""
    global CETVORKE
    stari = CETVORKE
    CETVORKE = {"n": 4, "hex": 16, "h": [_h(" ".join(reci("Đak je došao rano")))[:16]]}
    try:
        for t in ("Đak je došao rano.", "ĐAK JE — DOŠAO, rano", "Ђак је дошао рано", "a onda: djak je dosao rano!"):
            assert izvorne(t), t
        assert not izvorne("Đak je došao kasno."), "tri od četiri reči nisu dovoljne"
    finally:
        CETVORKE = stari
