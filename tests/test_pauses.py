"""Pauze iz zvuka kao signal za prelom (krug 19). Rečenice su izmišljene."""

from __future__ import annotations

import json

import pytest
from conftest import make_words

from subtitis import linebreak as lb
from subtitis import pauses
from subtitis.models import Profile, Word

JEDAN_RED = Profile(name="t", max_chars_per_line=20, max_lines=1, min_chars_per_line=10,
                    max_cps=99, max_duration=99)
REC = "Sutra rano ujutru idemo kolima na more pa posle vozom kući."


def w(i, s, e, t):
    return Word(i=i, s=s, e=e, w=t, p=0.9)


def lines(words, silences=None, profile=JEDAN_RED):
    cues = lb.layout_sentence(words, profile, 1, [], lb.LANGUAGE_RULES["sr"], None, silences)
    return [" ".join(c.lines) for c in cues]


def tisina_posle(words, k, trajanje=0.4):
    t = words[k].e
    return [(t - trajanje / 2, t + trajanje / 2)]


# ==========================================================================
# Tišina -> granica reči
# ==========================================================================


def test_tisina_u_rasponu_reci_ide_na_najblizu_granicu():
    """Whisper upija tišinu u reč: `kada` traje 1 s, tišina je u njegovom kraju."""
    ws = [w(0, 0.0, 0.5, "onda"), w(1, 0.5, 1.5, "kada"), w(2, 1.5, 2.0, "pada")]
    assert pauses.boundary_pauses(ws, [(0.9, 1.45)]) == [0.0, 0.55, 0.0]


def test_tisina_usred_duge_reci_se_ne_pripisuje():
    ws = [w(0, 0.0, 0.3, "vrlo"), w(1, 0.3, 2.0, "najvažnija"), w(2, 2.0, 2.4, "tema")]
    assert pauses.boundary_pauses(ws, [(1.0, 1.2)]) == [0.0, 0.0, 0.0]


def test_tisina_pre_recenice_ne_ide_na_prvu_granicu():
    ws = [w(0, 1.0, 1.2, "a"), w(1, 1.2, 1.6, "onda"), w(2, 1.6, 2.0, "dalje")]
    assert pauses.boundary_pauses(ws, [(0.5, 0.95)]) == [0.0, 0.0, 0.0]


def test_pauza_iza_zareza_ide_na_zarez():
    """Rez ispred interpunkcije nije dozvoljen, pa pauza stoji na zarezu."""
    ws = [w(0, 0.0, 0.5, "dobro"), w(1, 0.5, 0.54, ","), w(2, 0.7, 1.0, "idemo")]
    assert pauses.boundary_pauses(ws, [(0.54, 0.7)]) == [0.0, 0.16, 0.0]


def test_bez_tisina_nema_pauza():
    ws = make_words("jedna dve tri")
    assert pauses.boundary_pauses(ws, []) == [0.0, 0.0, 0.0]


# ==========================================================================
# pauses.json: preživljava brisanje zvuka, nosi čime je napravljen
# ==========================================================================


@pytest.fixture()
def posao(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    cfg.job_dir("p").mkdir(parents=True)
    return cfg


def test_bez_zvuka_i_bez_fajla_radi_bez_pauza_i_kaze(posao):
    log = []
    assert pauses.ensure("p", log.append) is None
    assert "prelamanje bez pauza" in log[0]


def test_postojeci_fajl_se_cita_bez_zvuka(posao, monkeypatch):
    posao.write_json(pauses.path("p"), {"silences": [[1.0, 1.5]], "made_with": {"vad": "silero"}})
    monkeypatch.setattr(pauses, "detect", lambda wav: pytest.fail("ne sme ponovo da čita zvuk"))
    assert pauses.ensure("p").silences == [(1.0, 1.5)]


def test_iz_zvuka_se_upisuje_jednom(posao, monkeypatch):
    (posao.job_dir("p") / "source.wav").write_bytes(b"RIFF")
    pozivi = []

    def detect(wav):
        pozivi.append(wav)
        return pauses.PausesFile(made_with={"vad": "silero", "threshold": 0.5}, silences=[(2.0, 2.4)])

    monkeypatch.setattr(pauses, "detect", detect)
    log = []
    assert pauses.ensure("p", log.append).silences == [(2.0, 2.4)]
    assert pauses.ensure("p").silences == [(2.0, 2.4)]
    assert len(pozivi) == 1, "drugi put se čita fajl"
    zapis = json.loads(pauses.path("p").read_text(encoding="utf-8"))
    assert zapis["made_with"]["threshold"] == 0.5 and zapis["schema_version"] == 1
    assert "1 tišina iz" in log[0]


def test_greska_pri_citanju_zvuka_ne_obara(posao, monkeypatch):
    (posao.job_dir("p") / "source.wav").write_bytes(b"nije zvuk")

    def detect(wav):
        raise RuntimeError("loš fajl")

    monkeypatch.setattr(pauses, "detect", detect)
    log = []
    assert pauses.ensure("p", log.append) is None
    assert "bez pauza" in log[0] and not pauses.path("p").exists()


def test_osteceni_fajl_ne_obara(posao):
    pauses.path("p").write_text("{nije json", encoding="utf-8")
    log = []
    assert pauses.ensure("p", log.append) is None
    assert "ne može pročitati" in log[0]


def test_detect_belezi_vad_i_parametre(monkeypatch, tmp_path):
    import numpy as np
    from faster_whisper import audio, vad

    monkeypatch.setattr(audio, "decode_audio", lambda p, sampling_rate: np.zeros(16000 * 3, dtype=np.float32))
    monkeypatch.setattr(vad, "get_speech_timestamps", lambda a, o: [
        {"start": 0, "end": 8000}, {"start": 16000, "end": 24000}, {"start": 40000, "end": 48000},
    ])
    pf = pauses.detect(tmp_path / "x.wav")
    assert pf.silences == [(0.5, 1.0), (1.5, 2.5)]
    assert pf.duration == 3.0
    mw = pf.made_with
    assert mw["vad"] == "silero" and mw["faster_whisper"] and mw["model"].endswith(".onnx")
    assert mw["threshold"] == pauses.VAD_THRESHOLD
    assert mw["min_silence_duration_ms"] == pauses.VAD_MIN_SILENCE_MS


# ==========================================================================
# Prelamanje: isključeno ne menja ništa; uključeno ne nadjačava predlog
# ==========================================================================


def test_podrazumevano_iskljuceno():
    assert not lb.pauses_enabled()
    ws = make_words(REC)
    assert lines(ws, tisina_posle(ws, 4)) == lines(ws)


def test_popust_vuce_rez_na_pauzu(monkeypatch):
    monkeypatch.setattr(lb, "PAUSE_BREAK_DISCOUNT", 50)
    ws = make_words(REC)
    assert "idemo kolima na more" in lines(ws)            # bez pauze
    assert "idemo kolima" in lines(ws, tisina_posle(ws, 4))  # pauza posle `kolima`


def test_pauza_iza_predloga_ne_pravi_rez_iza_predloga(monkeypatch):
    """Govornik ume da udahne usred sintagme; `na / more` ostaje greška."""
    monkeypatch.setattr(lb, "PAUSE_BREAK_DISCOUNT", 50)
    monkeypatch.setattr(lb, "PENALTY_PAUSE_IN_CUE", 200)
    ws = make_words(REC)
    k = [x.w for x in ws].index("na")
    assert not any(l.endswith(" na") for l in lines(ws, tisina_posle(ws, k, 1.5)))


def test_kazna_za_pauzu_u_cue_u_ima_gornju_granicu(monkeypatch):
    monkeypatch.setattr(lb, "PENALTY_PAUSE_IN_CUE", 200)
    assert lb.pause_in_cue_penalty(lb.PAUSE_MIN - 0.01) == 0
    assert lb.pause_in_cue_penalty(lb.PAUSE_MIN + 0.1) == pytest.approx(20)
    assert lb.pause_in_cue_penalty(5.0) == lb.PENALTY_PAUSE_IN_CUE_MAX
    assert lb.PENALTY_PAUSE_IN_CUE_MAX < lb.PENALTY_SPLIT_PREPOSITION


def test_kazna_za_pauzu_u_cue_u_cepa_natpis(monkeypatch):
    monkeypatch.setattr(lb, "PENALTY_PAUSE_IN_CUE", 200)
    ws = make_words(REC)
    assert "idemo kolima na more" not in lines(ws, tisina_posle(ws, 4, 0.8))


def test_rez_bez_pauze_poskupljuje(monkeypatch):
    monkeypatch.setattr(lb, "PENALTY_BREAK_WITHOUT_PAUSE", 25)
    ws = make_words("grad selo reka")
    span = lb._Span(ws, JEDAN_RED, lb.LANGUAGE_RULES["sr"], None, [0.0, 0.5, 0.0])
    assert span.break_pen[0] == lb.PENALTY_BREAK_NOT_AT_CLAUSE + 25
    assert span.break_pen[1] == lb.PENALTY_BREAK_NOT_AT_CLAUSE


def test_prevod_ne_koristi_tisine(monkeypatch):
    """Izmišljena vremena prevoda nemaju pauze — tišine se ne smeju primeniti."""
    from subtitis.models import Sentence

    monkeypatch.setattr(lb, "PAUSE_BREAK_DISCOUNT", 50)
    sent = Sentence(id=1, s=0.0, e=4.0, text=REC, words=(0, -1))
    args = dict(language="sr", log=lambda m: None, interpolate_times=True)
    bez = lb.build_cues([sent.model_copy(deep=True)], [], JEDAN_RED, **args)
    sa = lb.build_cues([sent.model_copy(deep=True)], [], JEDAN_RED, silences=[(1.0, 1.5)], **args)
    assert [c.lines for c in sa] == [c.lines for c in bez]


# ==========================================================================
# Pipeline
# ==========================================================================


@pytest.fixture()
def asr_posao(tmp_path, monkeypatch):
    from subtitis import config as cfg
    from subtitis.models import AsrMeta, Job, Sentence, SentencesFile, WordsFile

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    jdir = cfg.job_dir("a")
    jdir.mkdir(parents=True)
    ws = make_words(REC)
    cfg.write_model(jdir / "words.sr.json", WordsFile(
        job_id="a", source_file="snimak.wav", duration=ws[-1].e, language="sr",
        asr=AsrMeta(model="test"), words=ws,
    ))
    cfg.write_model(jdir / "sentences.sr.json", SentencesFile(
        job_id="a", language="sr", derived_from="words.sr.json",
        sentences=[Sentence(id=1, s=0.0, e=ws[-1].e, text=REC, words=(0, len(ws) - 1))],
    ))
    cfg.save_job(Job(job_id="a", source_file="snimak.wav", language="sr", status="transcribed"))
    (jdir / "source.wav").write_bytes(b"RIFF")
    return cfg, ws


def test_iskljuceno_prelamanje_ne_cita_zvuk(asr_posao, monkeypatch, tmp_path):
    from subtitis import pipeline

    monkeypatch.setattr(pauses, "detect", lambda wav: pytest.fail("isključeno ne čita zvuk"))
    pipeline.run_build("a", out=tmp_path / "a.srt")
    assert not pauses.path("a").exists()


def test_ukljuceno_prelamanje_pravi_i_koristi_pauses_json(asr_posao, monkeypatch, tmp_path):
    from subtitis import pipeline

    cfg, ws = asr_posao
    monkeypatch.setattr(lb, "PAUSE_BREAK_DISCOUNT", 50)
    monkeypatch.setattr(pauses, "detect", lambda wav: pauses.PausesFile(
        made_with={"vad": "silero"}, silences=tisina_posle(ws, 4)))
    # Sopstveni uzak profil (kao `JEDAN_RED`), da test ne zavisi od širine `web`.
    (cfg.PROFILES_DIR / "uzak.json").write_text(json.dumps(
        JEDAN_RED.model_dump() | {"name": "uzak"}), encoding="utf-8")
    pipeline.run_build("a", profile="uzak", out=tmp_path / "a.srt")
    assert pauses.path("a").exists()
    srt = (tmp_path / "a.srt").read_text(encoding="utf-8")
    assert "idemo kolima\n" in srt
    build = [h for h in cfg.load_job("a").history if h.phase == "build"][-1]
    assert build.detail["pauses"] is True


def test_ukljuceno_bez_zvuka_radi_i_kaze(asr_posao, monkeypatch, tmp_path):
    from subtitis import pipeline

    cfg, _ = asr_posao
    (cfg.job_dir("a") / "source.wav").unlink()
    monkeypatch.setattr(lb, "PAUSE_BREAK_DISCOUNT", 50)
    log = []
    pipeline.run_build("a", out=tmp_path / "a.srt", on_log=log.append)
    assert any("prelamanje bez pauza" in m for m in log)
    build = [h for h in cfg.load_job("a").history if h.phase == "build"][-1]
    assert build.detail["pauses"] is False
