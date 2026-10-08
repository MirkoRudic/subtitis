"""Testovi hvatanja razloga odbacivanja i parsiranja pragova.

Ne diraju model — samo logiku oko njega.
"""

from __future__ import annotations

import logging

import pytest

from subtitis import asr
from subtitis.asr import DROP_KEYS, LOGGER_NAME, REJECTION_PATTERNS, RejectionCapture
from subtitis.cli import parse_threshold
from subtitis.config import SubtitisError


def emit(msg: str, *args) -> None:
    logging.getLogger(LOGGER_NAME).debug(msg, *args)


# --- RejectionCapture ------------------------------------------------------


def test_hvata_no_speech_odbacivanje():
    with RejectionCapture() as cap:
        emit("No speech threshold is met (%f > %f)", 0.91, 0.6)
    assert cap.counts == {"no_speech_dropped": 1}
    assert cap.dropped == 1


def test_fallback_se_ne_broji_kao_odbacivanje():
    with RejectionCapture() as cap:
        emit("Log probability threshold is not met with temperature %.1f (%f < %f)", 0.0, -1.4, -1.0)
        emit("Compression ratio threshold is not met with temperature %.1f (%f > %f)", 0.2, 2.9, 2.4)
    assert cap.counts == {"log_prob_fallback": 1, "compression_ratio_fallback": 1}
    assert cap.dropped == 0, "fallback je ponovni pokušaj, ne gubitak audija"


def test_broji_vise_pojava():
    with RejectionCapture() as cap:
        for _ in range(3):
            emit("No speech threshold is met (%f > %f)", 0.9, 0.6)
    assert cap.counts["no_speech_dropped"] == 3


def test_nepoznate_poruke_se_ignorisu():
    with RejectionCapture() as cap:
        emit("Processing segment at %s", "00:01")
        emit("Reset prompt. prompt_reset_on_temperature threshold is met")
    assert cap.counts == {}


def test_cuva_sirove_poruke():
    with RejectionCapture() as cap:
        emit("No speech threshold is met (%f > %f)", 0.91, 0.6)
    assert cap.messages == ["No speech threshold is met (0.910000 > 0.600000)"]


def test_vraca_logger_u_prethodno_stanje():
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    before_handlers = list(logger.handlers)
    with RejectionCapture():
        assert logger.level == logging.DEBUG
    assert logger.level == logging.WARNING
    assert logger.propagate is True
    assert logger.handlers == before_handlers


def test_cisti_se_i_kad_pukne_izuzetak():
    logger = logging.getLogger(LOGGER_NAME)
    before = list(logger.handlers)
    with pytest.raises(ValueError):
        with RejectionCapture():
            raise ValueError("puk")
    assert logger.handlers == before


def test_drop_keys_su_podskup_obrazaca():
    keys = {k for k, _, _ in REJECTION_PATTERNS}
    assert set(DROP_KEYS) <= keys
    assert DROP_KEYS == ("no_speech_dropped",)


# --- Podrazumevani pragovi -------------------------------------------------


def test_filter_halucinacija_je_iskljucen():
    """Regresija: sa 2.0 je gutao ceo rep snimka (pokrivenost 55.9% -> 83.8%).

    Ako ovaj test padne, neko je vratio filter u podrazumevane parametre.
    Ne vraćaj ga bez merenja pokrivenosti na dužem snimku sa tihim govorom.
    """
    assert asr.HALLUCINATION_SILENCE_THRESHOLD is None


def test_vrednost_za_izricito_ukljucivanje_postoji():
    assert asr.HALLUCINATION_SILENCE_SECONDS == 2.0


def test_odbrana_od_petlji_ostaje_na_mestu():
    """Filter halucinacija je uklonjen, ali prava odbrana od petlji mora ostati."""
    assert asr.CONDITION_ON_PREVIOUS_TEXT is False
    assert asr.COMPRESSION_RATIO_THRESHOLD == 2.4
    assert asr.TEMPERATURE_FALLBACK[0] == 0.0
    assert len(asr.TEMPERATURE_FALLBACK) > 1


# --- parse_threshold -------------------------------------------------------


def test_prazno_daje_podrazumevanu_vrednost():
    assert parse_threshold(None, 0.6, "--x") == 0.6


def test_off_iskljucuje():
    for v in ("off", "OFF", "none", "null", "-"):
        assert parse_threshold(v, 0.6, "--x") is None


def test_broj_se_parsira():
    assert parse_threshold("0.85", 0.6, "--x") == 0.85
    assert parse_threshold("-1.5", None, "--x") == -1.5


def test_smece_daje_jasnu_gresku():
    with pytest.raises(SubtitisError) as exc:
        parse_threshold("mnogo", 0.6, "--no-speech-threshold")
    assert "--no-speech-threshold" in str(exc.value)


# ==========================================================================
# Imenovani izbori VAD-a
# ==========================================================================


def test_normalno_je_podrazumevani_prag():
    on, thr = asr.vad_preset("normalno")
    assert on is True
    assert thr == asr.VAD_THRESHOLD


def test_tih_snimak_spusta_prag():
    on, thr = asr.vad_preset("tih")
    assert on is True
    assert thr == 0.20
    assert thr < asr.VAD_THRESHOLD, "tiši snimak traži manje agresivan VAD"


def test_iskljuceno_gasi_vad():
    on, thr = asr.vad_preset("iskljuceno")
    assert on is False
    assert thr is None


def test_nepoznat_izbor_pada_na_podrazumevani():
    assert asr.vad_preset("kako-god") == asr.vad_preset(asr.DEFAULT_VAD_PRESET)
    assert asr.vad_preset(None) == asr.vad_preset(asr.DEFAULT_VAD_PRESET)


def test_prag_stize_do_parametara_vad_a():
    assert asr.vad_parameters(0.20)["threshold"] == 0.20
    assert asr.vad_parameters()["threshold"] == asr.VAD_THRESHOLD
    # Ostali parametri se ne diraju izborom praga.
    assert asr.vad_parameters(0.20)["speech_pad_ms"] == asr.VAD_SPEECH_PAD_MS


# ==========================================================================
# Auto-detekcija jezika
#
# Model se ne pušta; podmeće se lažan, da se proveri šta mu se prosleđuje i
# šta se upisuje u `words.json`.
# ==========================================================================


class FakeWord:
    def __init__(self, word, start, end, probability=0.9):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class FakeSegment:
    def __init__(self, words):
        self.words = words
        self.end = words[-1].end


class FakeInfo:
    def __init__(self, language, probability=0.99):
        self.language = language
        self.language_probability = probability


class FakeModel:
    """Beleži parametre poziva i vraća unapred zadat rezultat."""

    def __init__(self, language="fr"):
        self.calls: list[dict] = []
        self.info = FakeInfo(language)

    def transcribe(self, path, **params):
        self.calls.append(params)
        segs = [FakeSegment([FakeWord("Bonjour", 0.0, 0.5), FakeWord("Paul", 0.5, 1.0)])]
        return segs, self.info


@pytest.fixture()
def fake_model(monkeypatch):
    m = FakeModel()
    monkeypatch.setattr(asr, "load_model", lambda *a, **k: m)
    return m


def run(fake_model, language, **kw):
    return asr.transcribe(
        "nebitno.wav", job_id="p", source_file="p.wav",
        language=language, duration=1.0, **kw,
    )


def test_zadat_jezik_ide_modelu_kakav_jeste(fake_model):
    wf = run(fake_model, "sr")
    assert fake_model.calls[0]["language"] == "sr"
    assert wf.language == "sr"


def test_auto_ne_prosledjuje_jezik_modelu(fake_model):
    """`language=None` je ono što faster-whisper razume kao auto-detekciju."""
    run(fake_model, "auto")
    assert fake_model.calls[0]["language"] is None


def test_auto_upisuje_detektovani_jezik(fake_model):
    wf = run(fake_model, "auto")
    assert wf.language == "fr", "words.json mora da nosi razrešen jezik, ne 'auto'"
    assert wf.asr.detected_language == "fr"


def test_prazan_jezik_je_i_dalje_greska(fake_model):
    """Razlika između „ne znam" i „neka model pogodi" mora da ostane."""
    with pytest.raises(SubtitisError):
        run(fake_model, "")


def test_auto_bez_detekcije_pada(monkeypatch):
    m = FakeModel(language=None)
    monkeypatch.setattr(asr, "load_model", lambda *a, **k: m)
    with pytest.raises(SubtitisError):
        run(m, "auto")


def test_izbor_vad_a_stize_do_modela(fake_model):
    run(fake_model, "sr", vad=True, vad_threshold=0.20)
    assert fake_model.calls[0]["vad_parameters"]["threshold"] == 0.20


def test_iskljucen_vad_ne_salje_parametre(fake_model):
    run(fake_model, "sr", vad=False)
    assert fake_model.calls[0]["vad_filter"] is False
    assert fake_model.calls[0]["vad_parameters"] is None
