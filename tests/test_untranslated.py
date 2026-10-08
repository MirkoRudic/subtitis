"""„Prevod" koji je zapravo original.

Rečenice su izmišljene, a oblik grešaka je iz prvog francuskog posla
(2026-09-25): model je pod ispravnim rednim brojem vratio srpski, ceo ili
delimično, i to je prošlo validaciju.
"""

from __future__ import annotations

import pytest

from subtitis import pipeline, untranslated
from subtitis.config import SubtitisError
from subtitis.models import CuePart, Sentence


def sent(i: int, text: str, flags=(), cues=()) -> Sentence:
    return Sentence(
        id=i, s=0.0, e=1.0, text=text, words=(0, -1), flags=list(flags),
        cue_parts=[CuePart(cue=c - 1, chars=1, share=1.0) for c in cues],
    )


SR = {
    1: "Polako, polako!",
    64: "Mislim da ću stići ranije.",
    100: "To znači da je kabl za punjač u kutiji, ispod uputstva za montažu.",
    101: "Hvala svima koji su nam pozajmili opremu, bili ste veoma darežljivi!",
    59: "Vesna Kostić.",
    5: "Ja sam Bojan Šćekić.",
    12: "Ja sam Vesna, animatorka u Radionici otvorenih vrata.",
    22: "Zovem se Đorđe Ilić, imam 61 godinu.",
}
FR = {
    1: "Polako, polako!",                                  # ceo original
    64: "Mislim da ću ranije stići.",                      # original, reči premeštene
    100: "Cela znači da je kabl za punjač u kutiji, ispod uputstva za montažu.",  # delimičan
    101: "Hvala svima koji su nam pozajmili opremu, bili ste veoma darežljivi!",
    59: "Vesna Kostić.",                                   # same imena: nije greška
    5: "Moi, c'est Bojan Šćekić.",
    12: "Je suis Vesna, animatrice à la Radionica otvorenih vrata.",
    22: "Je m'appelle Đorđe Ilić, j'ai 61 ans.",
}


def test_hvata_original_pod_ispravnim_brojem():
    r = untranslated.check(
        [sent(i, t) for i, t in SR.items()], [sent(i, t) for i, t in FR.items()], "sr", "fr"
    )
    assert {u.sentence: u.reason for u in r.suspect} == {
        1: "isto kao original",
        64: "slova izvora",
        100: "slova izvora",
        101: "slova izvora",  # darežljivi
    }
    assert r.failed == []


def test_imena_nisu_original():
    """Jedini lažni pogoci pre izuzetka: imena sa ć/đ/š i rečenica od samih imena."""
    assert untranslated.foreign_share("Moi, c'est Bojan Petrović, à la Radionica otvorenih vrata.", "sr", "fr") == 0
    r = untranslated.check([sent(59, SR[59])], [sent(59, FR[59])], "sr", "fr")
    assert r.all == []


def test_cirilica_u_francuskom_je_strana():
    assert untranslated.foreign_share("Merci беаукоуп", "sr", "fr") > 0


def test_poruka_broji_cue_ove_ne_samo_recenice():
    r = untranslated.Report(failed=[
        untranslated.Untranslated(102, list(range(168, 178)), "translation_failed")
    ])
    (line,) = untranslated.lines(r, "fr")
    assert "1 rečenica neprevedena — 10 cue-ova (168–177) na fr" in line


@pytest.mark.parametrize("n,expected", [
    (1, "rečenica"), (2, "rečenice"), (4, "rečenice"), (5, "rečenica"),
    (11, "rečenica"), (12, "rečenica"), (21, "rečenica"), (22, "rečenice"),
])
def test_mnozina(n, expected):
    assert untranslated.plural(n, "rečenica", "rečenice", "rečenica") == expected


def test_opsezi_cue_ova():
    assert untranslated.cue_ranges([177, 1, 168, 169, 170, 89, 171]) == "1, 89, 168–171, 177"


# ==========================================================================
# Invarijanta: translation_failed nosi CEO original
# ==========================================================================


def test_pomesan_tekst_sa_oznakom_neuspeha_puca():
    src = [sent(102, "Uprkos kiši, snimanje je završeno na vreme.")]
    mixed = [sent(102, "Malgré la pluie, snimanje je završeno na vreme.", ["translation_failed"])]
    with pytest.raises(SubtitisError, match="procureo"):
        pipeline.assert_failed_is_original(mixed, src)
    pipeline.assert_failed_is_original(
        [sent(102, src[0].text, ["translation_failed"])], src
    )


def test_odustajanje_upisuje_ceo_original_posle_delimicnog_pokusaja():
    """Kao u logu: za 102 model prvo vrati deo prevoda pod drugim brojem, pa
    `1|` umesto `102|`. Ništa od toga ne sme da uđe u tekst."""
    from subtitis.translate import CallStats, translate_sentences

    class Client:
        stats = CallStats()

        def chat(self, system, user):
            block = user.split("ZA_PREVOD:\n", 1)[1].split("\n\nKONTEKST")[0]
            ids = [int(l.split("|")[0]) for l in block.splitlines() if l.strip()]
            if "Uprkos kiši" in block:
                # pola prevoda pod brojem kog u odlomku nema. (U dnevniku je
                # bilo `1|` umesto `102|`; otkad model vidi lokalne brojeve
                # 1..n, `1|` je za jednu rečenicu ispravan, pa ovde ide n+1.)
                return f"{len(ids) + 1}|Malgré la pluie, le tournage"
            return "\n".join(f"{i}|Phrase {i}." for i in ids)

    source = [sent(101, "Hvala."), sent(102, "Uprkos kiši, snimanje je završeno na vreme."), sent(103, "Kraj.")]
    out, stats = translate_sentences(source, "sr", "fr", client=Client(), log=lambda m: None)
    by_id = {s.id: s for s in out}
    assert stats.failed_ids == [102]
    assert by_id[102].text == "Uprkos kiši, snimanje je završeno na vreme."
    assert "translation_failed" in by_id[102].flags
    pipeline.assert_failed_is_original(out, source)
