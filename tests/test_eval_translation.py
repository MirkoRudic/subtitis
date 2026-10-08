"""Testovi učitavanja i spajanja u scripts/eval_translation.py.

Bag koji su ovi testovi trebalo da uhvate: `id` se podrazumevao kao pozicija u
jednoj petlji, a kao 0 u drugoj, pa je svaki red tabele pokazivao "(nedostaje)"
iako je prevod uspeo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_translation import KEY, load_pairs, missing_keys  # noqa: E402

from subtitis.config import SubtitisError  # noqa: E402


def data(pairs, **meta):
    return {"pairs": pairs, **meta}


# ==========================================================================
# Ključevi
# ==========================================================================


def test_bez_id_polja_kljuc_je_pozicija_od_jedan():
    """Slučaj iz stvarnog pairs.json — nijedan par nema `id`."""
    pairs, *_ = load_pairs(
        data([{"src": "Prva.", "ref": "First."}, {"src": "Druga.", "ref": "Second."}])
    )
    assert [p[KEY] for p in pairs] == [1, 2]


def test_bez_id_polja_spajanje_radi():
    pairs, *_ = load_pairs(data([{"src": f"R{k}."} for k in range(9)]))
    keys = [p[KEY] for p in pairs]
    assert keys == list(range(1, 10))
    # simuliraj prevod: isti ključevi
    result = {k: f"EN {k}" for k in keys}
    assert missing_keys(result, pairs) == []
    for p in pairs:
        assert result[p[KEY]] == f"EN {p[KEY]}"


def test_sa_id_poljem_kljuc_je_id():
    pairs, *_ = load_pairs(
        data([{"id": 12, "src": "Prva."}, {"id": 13, "src": "Druga."}])
    )
    assert [p[KEY] for p in pairs] == [12, 13]


def test_mesano_id_i_bez_id():
    pairs, *_ = load_pairs(
        data([{"id": 7, "src": "A."}, {"src": "B."}, {"id": 9, "src": "C."}])
    )
    assert [p[KEY] for p in pairs] == [7, 2, 9]


def test_id_kao_string_se_prihvata():
    pairs, *_ = load_pairs(data([{"id": "5", "src": "A."}]))
    assert pairs[0][KEY] == 5


def test_originalna_polja_ostaju():
    pairs, *_ = load_pairs(
        data([{"src": "A.", "ref": "B.", "why": "zato", "asr_raw": "A!"}])
    )
    p = pairs[0]
    assert p["src"] == "A." and p["ref"] == "B."
    assert p["why"] == "zato" and p["asr_raw"] == "A!"


# ==========================================================================
# Jezici i varijanta
# ==========================================================================


def test_src_lang_i_tgt_lang():
    _, src, tgt, variant = load_pairs(
        data([{"src": "A."}], src_lang="sr", tgt_lang="en", variant="en-GB")
    )
    assert (src, tgt, variant) == ("sr", "en", "en-GB")


def test_stara_imena_src_i_tgt_i_dalje_rade():
    _, src, tgt, _ = load_pairs(data([{"src": "A."}], src="sr", tgt="fr"))
    assert (src, tgt) == ("sr", "fr")


def test_podrazumevani_jezici():
    _, src, tgt, variant = load_pairs(data([{"src": "A."}]))
    assert (src, tgt, variant) == ("sr", "en", None)


# ==========================================================================
# Glasne greške umesto tihih rupa
# ==========================================================================


def test_prazan_niz_parova():
    with pytest.raises(SubtitisError, match="nema nijedan par"):
        load_pairs(data([]))


def test_nedostaje_pairs_kljuc():
    with pytest.raises(SubtitisError, match="nema nijedan par"):
        load_pairs({})


def test_par_bez_src_polja():
    with pytest.raises(SubtitisError, match="poziciji 2"):
        load_pairs(data([{"src": "A."}, {"ref": "samo referenca"}]))


def test_par_sa_praznim_src():
    with pytest.raises(SubtitisError, match="poziciji 1"):
        load_pairs(data([{"src": "   "}]))


def test_duplirani_id_puca():
    """Inače bi drugi par tiho pregazio prvi u mapi rezultata."""
    with pytest.raises(SubtitisError, match="isti id"):
        load_pairs(data([{"id": 3, "src": "A."}, {"id": 3, "src": "B."}]))


def test_id_koji_nije_broj_puca():
    with pytest.raises(SubtitisError, match="nije ceo broj"):
        load_pairs(data([{"id": "prvi", "src": "A."}]))


# ==========================================================================
# Detekcija neuspelog spajanja
# ==========================================================================


def test_missing_keys_prijavljuje_rupe():
    pairs, *_ = load_pairs(data([{"src": "A."}, {"src": "B."}, {"src": "C."}]))
    assert missing_keys({1: "x", 3: "z"}, pairs) == [2]


def test_missing_keys_prazan_rezultat():
    pairs, *_ = load_pairs(data([{"src": "A."}, {"src": "B."}]))
    assert missing_keys({}, pairs) == [1, 2]


def test_missing_keys_regresija_stari_bag():
    """Stari bag: ključevi 1..N, a spajanje je tražilo 0 za svaki par."""
    pairs, *_ = load_pairs(data([{"src": "A."}, {"src": "B."}]))
    result = {0: "sve pod nulom"}
    assert missing_keys(result, pairs) == [1, 2]


def test_pravi_pairs_json_se_ucitava():
    """Fajl iz repozitorijuma mora da prođe bez izuzetka."""
    from subtitis.config import read_json

    path = Path(__file__).resolve().parents[1] / "eval" / "pairs.json"
    if not path.exists():
        pytest.skip("eval/pairs.json ne postoji")
    pairs, src, tgt, _ = load_pairs(read_json(path))
    assert pairs
    assert [p[KEY] for p in pairs] == list(range(1, len(pairs) + 1))
    assert src and tgt
