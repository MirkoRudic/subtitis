"""Testovi SRT serijalizacije."""

from __future__ import annotations

from subtitis.models import Cue
from subtitis.srt import bcp47, render, srt_name, timestamp, write_srt


def test_timestamp():
    assert timestamp(0) == "00:00:00,000"
    assert timestamp(1.24) == "00:00:01,240"
    assert timestamp(61.5) == "00:01:01,500"
    assert timestamp(3721.44) == "01:02:01,440"
    assert timestamp(-1) == "00:00:00,000"


def test_render_ima_crlf_i_prazan_red():
    cues = [Cue(s=1.24, e=4.1, lines=["Dobar dan,", "hvala što ste došli."])]
    out = render(cues)
    assert out.startswith("1\r\n00:00:01,240 --> 00:00:04,100\r\n")
    assert "Dobar dan,\r\nhvala što ste došli.\r\n" in out
    assert "\n" not in out.replace("\r\n", "")


def test_numeracija_krece_od_jedan():
    cues = [Cue(s=0, e=1, lines=["A"]), Cue(s=2, e=3, lines=["B"])]
    out = render(cues)
    assert out.split("\r\n")[0] == "1"
    assert "\r\n2\r\n" in out


def test_zapis_je_utf8_bez_bom_a(tmp_path):
    p = write_srt(tmp_path / "x.srt", [Cue(s=0, e=1, lines=["Šta ćeš?"])])
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert "Šta ćeš?" in raw.decode("utf-8")
    assert b"\r\n" in raw


def test_imenovanje():
    assert bcp47("sr", "Latn") == "sr-Latn"
    assert bcp47("sr", "Cyrl") == "sr-Cyrl"
    assert bcp47("en", "Latn") == "en"
    assert srt_name("intervju", "sr", "Cyrl") == "intervju.sr-Cyrl.srt"
    assert srt_name("intervju", "en") == "intervju.en.srt"
