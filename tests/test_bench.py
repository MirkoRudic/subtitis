"""Testovi detektora petlji ponavljanja iz scripts/bench.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from bench import REPEAT_LOOP_THRESHOLD, expand, longest_repeat_run  # noqa: E402

from subtitis.models import Sentence  # noqa: E402


def sents(*texts: str) -> list[Sentence]:
    return [
        Sentence(id=k, s=k, e=k + 1, text=t, words=(k, k))
        for k, t in enumerate(texts, start=1)
    ]


def test_bez_ponavljanja():
    run, _ = longest_repeat_run(sents("Dobar dan.", "Kako ste?", "Hvala, dobro."))
    assert run == 1


def test_identicna_ponavljanja():
    run, text = longest_repeat_run(
        sents("Uvod.", "Hvala na pažnji.", "Hvala na pažnji.", "Hvala na pažnji.", "Kraj.")
    )
    assert run == 3
    assert text == "Hvala na pažnji."


def test_petlja_preko_praga():
    run, _ = longest_repeat_run(sents("Uvod.", *(["Hvala na pažnji."] * 8)))
    assert run == 8
    assert run > REPEAT_LOOP_THRESHOLD


def test_gotovo_identicne_se_broje():
    run, _ = longest_repeat_run(
        sents(
            "Titlove je uradio Marko Petrovic",
            "Titlove je uradio Marko Petrovic.",
            "Titlove je uradio Marko Petrovič",
            "Nesto sasvim drugo.",
        )
    )
    assert run == 3


def test_interpunkcija_i_velika_slova_se_ignorisu():
    run, _ = longest_repeat_run(sents("Da, da, da!", "da da da"))
    assert run == 2


def test_slicne_ali_razlicite_recenice_nisu_petlja():
    run, _ = longest_repeat_run(
        sents("Otisli smo u grad.", "Otisli smo na more.", "Vratili smo se kasno.")
    )
    assert run == 1


def test_niz_se_prekida_pa_nastavlja():
    run, text = longest_repeat_run(sents("A.", "A.", "B.", "C.", "C.", "C."))
    assert run == 3
    assert text == "C."


def test_prazan_ulaz():
    assert longest_repeat_run([]) == (0, "")


def test_expand_siri_glob_sam(tmp_path):
    """PowerShell prosleđuje obrazac kao bukvalan string — expand ga mora širiti."""
    for name in ("a.mp3", "b.mp3", "c.wav"):
        (tmp_path / name).write_bytes(b"x")
    hits = expand([str(tmp_path / "*.mp3")])
    assert [p.name for p in hits] == ["a.mp3", "b.mp3"]


def test_expand_prihvata_i_obicnu_putanju(tmp_path):
    f = tmp_path / "snimak.mp3"
    f.write_bytes(b"x")
    assert expand([str(f)]) == [f]
