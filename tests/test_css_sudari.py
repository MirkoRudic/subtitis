"""Krug 26: sudar imena klasa — bez brauzera.

Traka `<div class="flagbar info">` i ikonica `<span class="info">` delile su
ime `info`; `.info` je u `app.css` niže od `.flagbar`, pa je traka dobila
`display: inline-grid` i 14 × 14 px. Testovi su gledali tekst i nisu videli.

Provera: klase koje u `app.css` SAMOSTALNO (selektor je samo `.ime`)
postavljaju `display` su „komponente" — svaka tvrdi kako se element slaže.
Element sa dve takve klase je sudar: pobeđuje ona niže u fajlu, tiho.
Modifikator (`.flagbar.info`, `.btn.sm`) ne postavlja `display` sam, pa ga
provera ne broji.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "subtitis" / "web"
CSS = WEB / "static" / "app.css"

_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_CLASS_ATTR = re.compile(r'class="([^"]*)"')
_SIMPLE = re.compile(r"^\.([A-Za-z_][\w-]*)$")


def display_classes(css: str) -> dict[str, str]:
    """Ime klase -> vrednost `display` iz pravila čiji je selektor samo `.ime`.
    Pravila u `@media` se hvataju jednako (unutrašnji blok nema zagrada)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: dict[str, str] = {}
    for sel, body in _RULE.findall(css):
        m = re.search(r"(?:^|;)\s*display\s*:\s*([^;]+)", body)
        if not m:
            continue
        for s in sel.split(","):
            sm = _SIMPLE.match(s.strip())
            if sm:
                out[sm.group(1)] = m.group(1).strip()
    return out


def class_lists(text: str) -> list[list[str]]:
    """Sve `class="..."` vrednosti kao liste imena. Jinja izrazi (`{{ }}`) se
    brišu, a tekst unutar `{% if %}` ostaje — grane se računaju kao da mogu
    stajati zajedno (provera je stroža nego što treba, ne blaža)."""
    out = []
    for raw in _CLASS_ATTR.findall(text):
        raw = re.sub(r"\{\{.*?\}\}", " ", raw)
        raw = re.sub(r"\{%.*?%\}", " ", raw)
        out.append([c for c in raw.split() if re.match(r"^[A-Za-z_][\w-]*$", c)])
    return out


def collisions() -> list[tuple[str, list[str]]]:
    comp = display_classes(CSS.read_text(encoding="utf-8"))
    found = []
    for f in sorted((WEB / "templates").glob("*.html")) + [WEB / "app.py"]:
        for names in class_lists(f.read_text(encoding="utf-8")):
            hit = sorted({n for n in names if n in comp})
            if len(hit) >= 2:
                found.append((f.name, hit))
    return found


_PAIR = re.compile(r"^\.([A-Za-z_][\w-]*)\.([A-Za-z_][\w-]*)$")


def _props(body: str) -> set[str]:
    return {d.split(":", 1)[0].strip() for d in body.split(";") if ":" in d}


def modifier_leaks(css: str) -> dict[str, set[str]]:
    """Krug 27: modifikator čije ime ima i SAMOSTALNO pravilo prenosi na
    komponentu sve što to pravilo postavlja, a ni komponenta ni par ne
    postavljaju. `.err { white-space: pre-wrap }` je tako ušao u
    `.flagbar.err` i svaki prazan red šablona postao vidljiv (42 px iznad
    svake stavke spiska). Ključ je `.komponenta.modifikator`."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    single: dict[str, set[str]] = {}
    pairs: dict[tuple[str, str], set[str]] = {}
    for sel, body in _RULE.findall(css):
        for s in sel.split(","):
            s = s.strip()
            if m := _SIMPLE.match(s):
                single.setdefault(m.group(1), set()).update(_props(body))
            elif m := _PAIR.match(s):
                pairs.setdefault((m.group(1), m.group(2)), set()).update(_props(body))
    out = {}
    for (a, b), own in pairs.items():
        if b in single:
            leaked = single[b] - single.get(a, set()) - own
            if leaked:
                out[f".{a}.{b}"] = leaked
    return out


def test_modifikator_ne_prenosi_tudja_svojstva():
    leaks = modifier_leaks(CSS.read_text(encoding="utf-8"))
    assert not leaks, (
        f"modifikator ima i samostalno pravilo koje se preliva na komponentu: {leaks}. "
        "Svojstvo premesti na element kome pripada (npr. `pre.err`) ili daj drugo ime."
    )


def test_provera_vidi_curenje_iz_kruga_27():
    css = (".err { color: red; white-space: pre-wrap; }\n.flagbar { display: flex; color: gray; }\n"
           ".flagbar.err { background: pink; }\n.pill.ok { color: green; }")
    assert modifier_leaks(css) == {".flagbar.err": {"white-space"}}


def test_nijedan_element_ne_nosi_dve_komponente():
    sudari = collisions()
    assert not sudari, (
        "element nosi dve klase koje svaka postavlja `display` — pobeđuje niža u "
        f"app.css, tiho: {sudari}. Komponenta dobija ime koje ne deli sa modifikatorom."
    )


def test_provera_vidi_sudar_iz_kruga_26():
    """Da provera ne bi tiho prestala da radi (npr. promena formata CSS-a)."""
    comp = display_classes(".flagbar { display: flex; }\n.info {\n  display: inline-grid;\n}\n"
                           ".flagbar.info { color: red; }\n@media (x) { .kartica { display: none } }")
    assert comp == {"flagbar": "flex", "info": "inline-grid", "kartica": "none"}
    lists = class_lists('<div class="flagbar info">{% if x %}<span class="row {{ c }} {% if y %}info{% endif %}">')
    assert lists == [["flagbar", "info"], ["row", "info"]]


def test_ikonica_objasnjenja_ima_svoje_ime():
    css = CSS.read_text(encoding="utf-8")
    assert re.search(r"(^|\s|,)\.info\b", re.sub(r"\.flagbar\.info", "", css)) is None
    assert "why-ic" in display_classes(css)
