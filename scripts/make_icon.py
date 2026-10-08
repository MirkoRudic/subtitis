"""Napravi `subtitis.ico` — ikonicu za prečicu i sistemsku traku.

Crta se programski da bi bila reproducibilna i da u repou ne stoji binarni
fajl bez izvora. Boje se ČITAJU iz `app.css` (`:root`), ne prepisuju se: prepisane
vrednosti su se razišle sa temom čim je promenjena, a niko to nije primetio
jer ikonica sama za sebe izgleda uredno. Pokreni posle izmene boja:

    .venv\\Scripts\\python scripts\\make_icon.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "subtitis" / "web" / "static" / "subtitis.ico"
CSS = ROOT / "subtitis" / "web" / "static" / "app.css"

#: Deo ikonice -> promenljiva iz `:root` u `app.css`.
PARTS = {
    "fill": "--surface",
    "frame": "--accent",
    "line_long": "--fg",
    "line_short": "--faint",
}

_VAR = re.compile(r"(--[\w-]+)\s*:\s*#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\s*;")


def css_colors(css: Path = CSS) -> dict[str, tuple[int, int, int]]:
    """`:root` promenljive sa heks bojom -> RGB. Podržava `#RRGGBB` i `#RGB`.

    Čita samo prvi `:root { … }` blok — tamo je tema; kasnija pravila ne smeju
    da je tiho pregaze.
    """
    text = css.read_text(encoding="utf-8")
    m = re.search(r":root\s*\{(.*?)\}", text, re.S)
    if not m:
        raise SystemExit(f"{css}: nema :root bloka sa bojama.")
    out = {}
    for name, hexa in _VAR.findall(m.group(1)):
        if len(hexa) == 3:
            hexa = "".join(c * 2 for c in hexa)
        out[name] = tuple(bytes.fromhex(hexa))
    return out


def palette(css: Path = CSS) -> dict[str, tuple[int, int, int, int]]:
    """Boje delova ikonice. Promenljiva koja nedostaje obara skript — nikad
    rezervna vrednost: tiho zadržana stara boja je ono što je ovo i pokvarilo."""
    colors = css_colors(css)
    missing = [var for var in PARTS.values() if var not in colors]
    if missing:
        raise SystemExit(
            f"{css}: u :root nedostaje {', '.join(missing)} (heks boja) — "
            "ikonica se ne pravi sa izmišljenom bojom."
        )
    return {part: (*colors[var], 255) for part, var in PARTS.items()}

#: Veličine koje Windows traži: traka, Explorer, Alt+Tab.
SIZES = (16, 24, 32, 48, 64, 128, 256)


def draw(size: int, colors: dict[str, tuple[int, int, int, int]]) -> Image.Image:
    """Titl: okvir i dva reda teksta — čitljivo i na 16 px."""
    img = Image.new("RGBA", (size, size))  # providna pozadina
    d = ImageDraw.Draw(img)
    r = max(2, size // 8)
    pad = max(1, size // 16)
    d.rounded_rectangle([pad, pad, size - pad - 1, size - pad - 1],
                        radius=r, fill=colors["fill"], outline=colors["frame"],
                        width=max(1, size // 24))

    # Dva reda titla, drugi kraći i centriran — kao pravi prelom.
    h = max(1, size // 10)
    gap = max(1, size // 12)
    y = size - pad - h * 2 - gap - max(1, size // 10)
    inner = size - 2 * pad
    long_w = int(inner * 0.62)
    short_w = int(inner * 0.42)
    d.rounded_rectangle(
        [(size - long_w) // 2, y, (size + long_w) // 2, y + h],
        radius=h // 2, fill=colors["line_long"],
    )
    y2 = y + h + gap
    d.rounded_rectangle(
        [(size - short_w) // 2, y2, (size + short_w) // 2, y2 + h],
        radius=h // 2, fill=colors["line_short"],
    )
    return img


def build(out: Path = OUT, css: Path = CSS) -> Path:
    colors = palette(css)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Najveća ide kao osnova, ostale kao dodatne — svaka nacrtana za svoju
    # veličinu, jer smanjivanje 256 -> 16 zamuti tanak okvir.
    largest = draw(max(SIZES), colors)
    largest.save(
        out, format="ICO",
        sizes=[(s, s) for s in SIZES],
        append_images=[draw(s, colors) for s in SIZES if s != max(SIZES)],
    )
    return out


def main() -> None:
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT)
    print(f"napravljeno: {out}  ({out.stat().st_size} B, {len(SIZES)} veličina)")


if __name__ == "__main__":
    main()
