"""Veličina slova u izlaznom SRT-u (krug 6, tačka 3).

Menja se samo izgled fajla: oznake ostaju, rečenice ostaju, broj znakova ostaje.
"""

from __future__ import annotations

import pytest

from subtitis import pipeline, translit
from subtitis.fixedcues import apply_case
from subtitis.models import SentencesFile


# --- Oznake --------------------------------------------------------------------


def test_cue_ceo_u_kurzivu():
    assert apply_case("<i>Kaže da ne može.</i>", "upper") == "<i>KAŽE DA NE MOŽE.</i>"
    assert apply_case("<i>Kaže da ne može.</i>", "lower") == "<i>kaže da ne može.</i>"


def test_cue_sa_delimicnim_kurzivom():
    assert apply_case("Rekao je <i>Zdravo</i> i otišao.", "upper") == "REKAO JE <i>ZDRAVO</i> I OTIŠAO."
    assert apply_case("<I>Glas</I> iz sobe", "lower") == "<I>glas</I> iz sobe", "ni veličina oznake se ne dira"
    assert apply_case(r"{\an8}Gore", "upper") == r"{\an8}GORE"


def test_normalno_ne_menja_nista():
    line = "Rekao je <i>Zdravo</i>, Ana."
    assert apply_case(line, "normal") is line


def test_nepoznata_velicina_se_odbija():
    with pytest.raises(ValueError):
        apply_case("x", "title")


# --- dž / lj / nj u obe azbuke ------------------------------------------------


def test_digrafi_u_latinici():
    assert apply_case("džep, ljubav, njiva", "upper") == "DŽEP, LJUBAV, NJIVA"
    assert apply_case("DŽEP, Ljubav, NJiva", "lower") == "džep, ljubav, njiva"


def test_slova_u_cirilici():
    assert apply_case("џеп, љубав, њива", "upper") == "ЏЕП, ЉУБАВ, ЊИВА"
    assert apply_case("Љубав", "lower") == "љубав"
    assert apply_case("љ", "upper") == "Љ"


def test_digraf_se_ne_raspada_u_cirilicu():
    """`DŽ` mora da postane `Џ`, ne `ДЖ`."""
    assert translit.convert(apply_case("džep ljubav njiva", "upper"), "Cyrl") == "ЏЕП ЉУБАВ ЊИВА"


@pytest.mark.parametrize("text", [
    "Džep, Ljubav i Njiva su u džepu.",
    "Џеп, Љубав и Њива су у џепу.",
    "NJEGOV DŽEP",
    "Rekao je <i>Ljubomir</i> da dođe.",
])
@pytest.mark.parametrize("script", ["Latn", "Cyrl"])
@pytest.mark.parametrize("case", ["upper", "lower"])
def test_redosled_transliteracija_i_velicina_je_svejedno(text, script, case):
    from subtitis.fixedcues import convert_markup

    prvo_pismo = apply_case(convert_markup(text, script), case)
    prvo_velicina = convert_markup(apply_case(text, case), script)
    assert prvo_pismo == prvo_velicina


@pytest.mark.parametrize("text", ["džep ljubav", "Straße", "<i>Ein Fuß</i>", "İstanbul"])
@pytest.mark.parametrize("case", ["upper", "lower"])
def test_broj_znakova_ostaje_isti(text, case):
    assert len(apply_case(text, case)) == len(text)


def test_nemacko_eszet_ostaje_jedno_slovo():
    assert apply_case("Straße", "upper") == "STRAẞE"


# --- Ceo tok: SRT se menja, rečenice ne ------------------------------------


@pytest.fixture()
def posao(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    src = tmp_path / "film.sr.srt"
    src.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n<i>Džep je pun.</i>\n\n"
        "2\n00:00:04,000 --> 00:00:06,000\nKaže <i>Ljubomir</i> da ne može.\n",
        encoding="utf-8",
    )
    pipeline.run_import_srt(src, "sr", job_id="slova")
    return cfg


@pytest.mark.parametrize("script", ["Latn", "Cyrl"])
def test_velika_slova_u_srt_u_oznake_i_recenice_netaknute(posao, tmp_path, script):
    cfg = posao
    jdir = cfg.job_dir("slova")
    before = (jdir / "sentences.sr.json").read_text(encoding="utf-8")
    normal = pipeline.run_build("slova", script=script, out=tmp_path / "n.srt")
    upper = pipeline.run_build("slova", script=script, case="upper", out=tmp_path / "u.srt")

    n_text = (tmp_path / "n.srt").read_text(encoding="utf-8-sig")
    u_text = (tmp_path / "u.srt").read_text(encoding="utf-8-sig")
    assert u_text == apply_case(n_text, "upper")  # vremena su cifre, `-->` nema slova
    assert "<i>" in u_text and "<I>" not in u_text
    assert ("ЏЕП" if script == "Cyrl" else "DŽEP") in u_text
    assert len(u_text) == len(n_text)
    # prekoračenja se broje isto — veličina ne menja broj znakova
    assert (upper.over_limit, upper.over_cps) == (normal.over_limit, normal.over_cps)
    assert (jdir / "sentences.sr.json").read_text(encoding="utf-8") == before
    build = [h for h in cfg.load_job("slova").history if h.phase == "build"][-1]
    assert build.detail["case"] == "upper"


def test_nepoznata_velicina_ne_prelama(posao):
    from subtitis.config import SubtitisError

    with pytest.raises(SubtitisError):
        pipeline.run_build("slova", case="naslovna")


def test_mala_slova_u_srt_u(posao, tmp_path):
    pipeline.run_build("slova", case="lower", out=tmp_path / "l.srt")
    text = (tmp_path / "l.srt").read_text(encoding="utf-8-sig")
    assert "<i>džep je pun.</i>" in text and "kaže <i>ljubomir</i>" in text
    sf = SentencesFile.model_validate(
        posao.read_json(posao.job_dir("slova") / "sentences.sr.json")
    )
    assert any("Ljubomir" in s.text for s in sf.sentences), "original ostaje netaknut"


# ==========================================================================
# Izostavljena interpunkcija (krug 8, profil `digital`)
# ==========================================================================

from subtitis.fixedcues import drop_punctuation, output_lines  # noqa: E402


def test_bez_interpunkcije_kurziv():
    assert drop_punctuation("<i>Kaže da ne može.</i>") == "<i>Kaže da ne može</i>"
    assert drop_punctuation("Rekao je <i>, zdravo</i> — i otišao.") == "Rekao je <i>zdravo</i> i otišao"


def test_bez_interpunkcije_pitanje_na_kraju():
    assert drop_punctuation("Šta ima?") == "Šta ima"
    assert drop_punctuation("Zaista?!…") == "Zaista"


def test_bez_interpunkcije_skracenica():
    assert drop_punctuation("Hleb, mleko, jaja itd.") == "Hleb mleko jaja itd"


def test_bez_interpunkcije_navodnici_svih_vrsta():
    assert drop_punctuation("On kaže: „Neću!”") == "On kaže Neću"
    assert drop_punctuation("\"Da\" i 'ne' i »možda«") == "Da i ne i možda"
    assert drop_punctuation("Il dit : « oui »") == "Il dit oui"
    assert drop_punctuation("(smeh) Dobro [muzika].") == "smeh Dobro muzika"


def test_francuski_apostrof_i_crtica_unutar_reci_ostaju():
    """Jedini jezik u kom se razlika vidi: `c'est` nije `cest`."""
    assert (
        drop_punctuation("C'est un rendez-vous à Saint-Malo, aujourd'hui.")
        == "C'est un rendez-vous à Saint-Malo aujourd'hui"
    )
    assert drop_punctuation("J’ai dit — non - c’est fini.") == "J’ai dit non c’est fini"


def test_oznake_se_ne_diraju():
    assert drop_punctuation(r"{\an8}- Da? - Ne.") == r"{\an8}Da Ne"
    assert drop_punctuation('<font color="#ff0">Stani!</font>') == '<font color="#ff0">Stani</font>'


def test_brojevi_ne_menjaju_vrednost():
    assert drop_punctuation("Košta 3,5 evra, u 15:30, sa 1.000 ljudi.") == "Košta 3,5 evra u 15:30 sa 1.000 ljudi"
    assert drop_punctuation("Rast je 50%.") == "Rast je 50%"


def test_cue_samo_od_interpunkcije_ostaje():
    assert output_lines(["…"], "normal", True) == ["…"]
    assert output_lines(["Da.", "—"], "normal", True) == ["Da"]


@pytest.mark.parametrize("text", [
    "Džep, Ljubav i Njiva su u „džepu”!",
    "Џеп, Љубав и Њива — у џепу?",
    "Rekao je <i>Ljubomir</i>: dođi.",
    "C'est un rendez-vous, n'est-ce pas?",
])
@pytest.mark.parametrize("script", ["Latn", "Cyrl"])
def test_redosled_pismo_velicina_interpunkcija_je_svejedno(text, script):
    from subtitis.fixedcues import convert_markup

    t, c, p = (lambda x: convert_markup(x, script)), (lambda x: apply_case(x, "upper")), drop_punctuation
    napred = p(c(t(text)))
    assert napred == t(c(p(text))) == c(p(t(text))) == p(t(c(text)))


def test_digital_postavlja_velika_slova_i_bez_interpunkcije(posao, tmp_path):
    """Podrazumevano, ne zaključano: izričit izbor ima prednost."""
    cfg = posao
    pipeline.run_build("slova", profile="digital", out=tmp_path / "d.srt")
    text = (tmp_path / "d.srt").read_text(encoding="utf-8-sig")
    assert "<i>DŽEP JE PUN</i>" in text and "KAŽE <i>LJUBOMIR</i> DA NE MOŽE" in text
    det = [h for h in cfg.load_job("slova").history if h.phase == "build"][-1].detail
    assert (det["case"], det["drop_punct"]) == ("upper", True)

    pipeline.run_build("slova", profile="digital", case="normal", drop_punct=False,
                       out=tmp_path / "r.srt")
    text = (tmp_path / "r.srt").read_text(encoding="utf-8-sig")
    assert "<i>Džep je pun.</i>" in text

    pipeline.run_build("slova", profile="web", out=tmp_path / "w.srt")
    assert "<i>Džep je pun.</i>" in (tmp_path / "w.srt").read_text(encoding="utf-8-sig")
