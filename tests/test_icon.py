"""Ikonica (`scripts/make_icon.py`): boje se izvode iz `app.css`, ne prepisuju se."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSS = ROOT / "subtitis" / "web" / "static" / "app.css"
SCRIPT = ROOT / "scripts" / "make_icon.py"
sys.path.insert(0, str(ROOT / "scripts"))

import make_icon  # noqa: E402


def test_u_skriptu_nema_upisanih_boja():
    """Prepisane vrednosti su zaostale iz stare teme (plava umesto --accent)."""
    import io
    import tokenize

    toks = tokenize.generate_tokens(io.StringIO(SCRIPT.read_text(encoding="utf-8")).readline)
    kod = " ".join(t.string for t in toks if t.type != tokenize.COMMENT)
    assert not re.findall(r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}", kod), "heks boja u skriptu"
    torke = re.findall(r"\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*(?:,\s*\d{1,3}\s*)?\)", kod)
    assert not torke, f"RGB konstanta u skriptu: {torke}"


def test_app_css_ima_sve_cetiri_boje():
    boje = make_icon.css_colors(CSS)
    for var in ("--surface", "--accent", "--fg", "--faint"):
        assert var in boje, f"app.css nema {var}"


def test_okvir_ikonice_je_akcenat_iz_app_css(tmp_path):
    """Crta se SAMO u privremeni koren — nikad preko subtitis.ico."""
    out = make_icon.build(tmp_path / "ikonica.ico", CSS)
    assert out.parent == tmp_path
    from PIL import Image

    accent = make_icon.css_colors(CSS)["--accent"]
    img = Image.open(out)
    img.size = (256, 256)  # najveća veličina iz .ico
    img = img.convert("RGBA")
    size = 256
    pad = size // 16
    ivica = img.getpixel((size // 2, pad + 2))[:3]  # sredina gornjeg okvira
    assert ivica == accent
    ispuna = img.getpixel((size // 2, size // 3))[:3]
    assert ispuna == make_icon.css_colors(CSS)["--surface"]


def test_nedostaje_promenljiva_obara_skript(tmp_path):
    css = tmp_path / "app.css"
    css.write_text(":root {\n  --surface: #191C21;\n  --fg: #E9EBEE;\n  --faint: #6E7682;\n}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="--accent"):
        make_icon.build(tmp_path / "x.ico", css)
    assert not (tmp_path / "x.ico").exists()


def test_skraceni_heks_oblik(tmp_path):
    css = tmp_path / "app.css"
    css.write_text(":root { --surface: #123; --accent: #45C6D6; --fg: #fff; --faint: #6E7682; }", encoding="utf-8")
    assert make_icon.css_colors(css)["--surface"] == (0x11, 0x22, 0x33)
