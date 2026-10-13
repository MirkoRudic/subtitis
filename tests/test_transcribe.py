"""`run_transcribe` bez modela i bez kartice (krug 16).

Podmeće se SAMO ono što je spolja: priprema zvuka (`ffmpeg`), merenje govora
(Silero), prepoznavanje (Whisper) i LLM za interpunkciju (Ollama). Sve ostalo —
razrešenje jezika, elizija, ekavica, kapija interpunkcije, segmentacija,
upis datoteka, dnevnik — ide pravim kodom, a test gleda ISHODE.
"""

from __future__ import annotations

import pytest

from subtitis import asr, audio, pipeline
from subtitis import config as cfg
from subtitis.config import SubtitisError
from subtitis.hallucination import HALLUCINATED_FLAG
from subtitis.models import AsrMeta, SentencesFile, Word, WordsFile


def reci(tokens: list[str], *, p: float = 0.95, korak: float = 0.35, start: float = 0.0,
         nisko: set[str] = frozenset(), nulte: int = 0) -> list[Word]:
    """Reči sa vremenima. `nisko`: tokeni sa verovatnoćom 0,3. `nulte`: poslednjih
    N reči bez trajanja (izmišljene preko tišine, kao u Radionici draft 1)."""
    out, t = [], start
    for k, tok in enumerate(tokens):
        zero = k >= len(tokens) - nulte
        d = 0.0 if zero else korak
        out.append(Word(i=k, s=round(t, 3), e=round(t + d, 3), w=tok,
                        p=0.3 if tok in nisko else p))
        t += d if zero else korak + 0.05
    return out


@pytest.fixture()
def lazno(tmp_path, monkeypatch):
    """Spoljni svet: zvuk, Silero, Whisper, Ollama. Vraća podešavač ulaza."""
    stanje = {"words": [], "detected": None, "unloaded": 0, "progress": []}

    monkeypatch.setattr(audio, "check_environment", lambda: None)

    def normalize(src, wav):
        wav.write_bytes(b"RIFF")
        return 30.0

    monkeypatch.setattr(audio, "normalize", normalize)
    monkeypatch.setattr(asr, "measure_speech",
                        lambda wav, thr=None: asr.SpeechMeasure(duration=30.0, speech=24.0, segments=5))

    def transcribe(wav, *, job_id, source_file, language, duration, progress, **kw):
        progress(15.0, 30.0)
        progress(30.0, 30.0)
        det = stanje["detected"] or language
        return WordsFile(
            job_id=job_id, source_file=source_file, duration=duration,
            language=det if language == asr.AUTO_LANGUAGE else language,
            asr=AsrMeta(model="lažni", detected_language=det, detected_probability=0.91,
                        params={"original_script": "Latn"}),
            words=[w.model_copy() for w in stanje["words"]],
        )

    monkeypatch.setattr(asr, "transcribe", transcribe)
    monkeypatch.setattr(asr, "unload_model", lambda: stanje.__setitem__("unloaded", stanje["unloaded"] + 1))

    from subtitis import translate as tr

    class NedostupanLLM:
        model = "lažni-llm"

        def __init__(self, **kw):
            pass

        def check(self):
            raise SubtitisError("Ollama nije dostupna (test)")

        def unload(self):
            pass

    monkeypatch.setattr(tr, "OllamaClient", NedostupanLLM)

    src = tmp_path / "snimak.mp4"
    src.write_bytes(b"\0")
    stanje["src"] = src
    return stanje


def pokreni(stanje, language, **kw):
    logs: list[str] = []
    r = pipeline.run_transcribe(stanje["src"], language, job_id="p", on_log=logs.append,
                                on_progress=lambda f, x: stanje["progress"].append((f, x)), **kw)
    jdir = cfg.job_dir("p")
    log = (jdir / "log.txt").read_text(encoding="utf-8")
    return r, jdir, log, logs


# --------------------------------------------------------------------------
# 1. srpski, bez interpunkcije, ijekavica: ekavizacija; kapija interpunkcije okida
# --------------------------------------------------------------------------

SR_BEZ_ZNAKOVA = (
    "ovdje je bilo lijepo prije nego što su djeca otišla kući a onda smo sjeli "
    "i razgovarali dugo o svemu što nas je mučilo tih dana kad je bilo teško "
    "i nikome nije bilo lako ali smo ostali zajedno do kraja jer je to bilo važno"
).split()


def test_srpski_bez_znakova_ekavica_i_kapija_interpunkcije(lazno):
    lazno["words"] = reci(SR_BEZ_ZNAKOVA)
    r, jdir, log, logs = pokreni(lazno, "sr")

    # datoteke i status
    assert (jdir / "words.sr.json").exists() and (jdir / "sentences.sr.json").exists()
    assert (jdir / "source.wav").exists()
    job = cfg.load_job("p")
    assert job.status == "transcribed" and job.duration == 30.0
    assert [h.phase for h in job.history] == ["audio", "asr", "ekavica", "segment"]

    # ekavica: ijekavski oblici su nestali iz reči i rečenica
    wf = WordsFile.model_validate(cfg.read_json(jdir / "words.sr.json"))
    tekst = " ".join(w.w for w in wf.words)
    assert "ovde" in tekst and "lepo" in tekst and "pre" in tekst and "deca" in tekst
    assert "ovdje" not in tekst and "lijepo" not in tekst
    assert wf.ekavica["applied"] is True and wf.ekavica["changes"] == len(r.ekavica_changes) >= 4
    ekav = [l for l in log.splitlines() if l.split("] ", 1)[-1].startswith("ekavica ")]
    assert len(ekav) == len(r.ekavica_changes), "svaka izmena ekavice u dnevniku"

    # kapija interpunkcije je OKINULA (nema nijednog znaka kraja); LLM je
    # nedostupan, pa je to upozorenje, a transkript je ipak zapisan
    assert r.structure_before.needs_punctuation("sr")
    assert "UPOZORENJE: restauracija interpunkcije nije uspela: Ollama nije dostupna (test)" in log
    assert any("restauracija interpunkcije nije uspela" in m for m in logs)

    # rezultat i prijavljena pokrivenost
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    assert r.words == len(SR_BEZ_ZNAKOVA) and r.sentences == len(sf.sentences) >= 1
    assert not sf.edited_by_human and sf.derived_from == "words.sr.json"
    seg = next(h for h in job.history if h.phase == "segment").detail
    assert seg["coverage_pct"] == round(r.coverage_pct, 1) and 0 < r.coverage_pct <= 100
    assert seg["hallucinated_runs"] == []
    asr_det = next(h for h in job.history if h.phase == "asr").detail
    assert asr_det["words"] == len(SR_BEZ_ZNAKOVA) and asr_det["vad_discarded_share"] == 0.2
    # Whisper se oslobađa pre svega ostalog; napredak stiže do kraja
    assert lazno["unloaded"] == 1
    assert ("asr", 0.5) in lazno["progress"] and ("segment", 1.0) in lazno["progress"]


# --------------------------------------------------------------------------
# 2. francuski, razdvojene elizije, gusta interpunkcija: spajanje; kapija ne okida
# --------------------------------------------------------------------------

FR = [
    "C", "'est", "vrai", ".", "J'", "ai", "vu", "le", "chat", "hier", ".",
    "Il", "est", "là", "maintenant", ".", "Qu", "'il", "vienne", "demain", ".",
    "L'", "homme", "est", "parti", "très", "tôt", ".", "Nous", "sommes", "rentrés", ".",
]


def test_francuski_elizije_spojene_kapija_ne_okida(lazno):
    lazno["words"] = reci(FR)
    r, jdir, log, logs = pokreni(lazno, "fr")

    wf = WordsFile.model_validate(cfg.read_json(jdir / "words.fr.json"))
    tokeni = [w.w for w in wf.words]
    for spojeno in ("C'est", "J'ai", "Qu'il", "L'homme"):
        assert spojeno in tokeni, spojeno
    assert len(r.elisions) == 4 and "elizija: spojeno 4 parova tokena" in log
    # indeksi prenumerisani, bez rupa — rečenice se prave posle ovoga
    assert [w.i for w in wf.words] == list(range(len(wf.words)))

    # gusta interpunkcija: kapija NE okida, ekavica se ne primenjuje
    assert not r.structure_before.needs_punctuation("fr")
    assert "restauracija interpunkcije" not in log
    assert wf.ekavica == {"applied": False, "changes": 0}
    job = cfg.load_job("p")
    assert [h.phase for h in job.history] == ["audio", "asr", "elizija", "segment"]
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.fr.json"))
    assert len(sf.sentences) == 6
    assert sf.sentences[0].text == "C'est vrai."


# --------------------------------------------------------------------------
# 3. rep od pet reči bez trajanja: izmišljeno, odvojeno od nesigurnosti
# --------------------------------------------------------------------------

SR_REP = (
    "Dobar dan svima . Ovo je kratak razgovor o nečemu nesigurnom . "
    "Hvala svima na dolasku letnja radionica Divčibare Srbija ."
).split()


def test_rep_bez_zvuka_je_izmisljen_a_ne_nesiguran(lazno):
    # rep: poslednjih pet REČI bez trajanja (tačka na kraju je providna);
    # nesigurne: dve reči druge rečenice, sa visokom verovatnoćom svuda drugde
    words = reci(SR_REP[:-1], nisko={"kratak", "nesigurnom"}, nulte=5)
    words.append(Word(i=len(words), s=words[-1].e, e=words[-1].e, w=".", p=0.99))
    lazno["words"] = words
    r, jdir, log, logs = pokreni(lazno, "sr")

    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    by = {s.text: s for s in sf.sentences}
    izm = next(s for t, s in by.items() if "Divčibare" in t)
    nes = next(s for t, s in by.items() if "nesigurnom" in t)
    assert HALLUCINATED_FLAG in izm.flags and "low_confidence" not in izm.flags
    assert "low_confidence" in nes.flags and HALLUCINATED_FLAG not in nes.flags

    assert "UPOZORENJE: 1 niz(a) reči BEZ ZVUKA" in log
    assert any("BEZ ZVUKA" in m for m in logs)
    seg = next(h for h in cfg.load_job("p").history if h.phase == "segment").detail
    assert seg["hallucinated_sentences"] == [izm.id]
    assert len(r.hallucinated) == 1 and "Divčibare" in r.hallucinated[0]


# --------------------------------------------------------------------------
# grane oko glavnog toka
# --------------------------------------------------------------------------


def test_auto_jezik_razresava_srpski_i_upozorava(lazno):
    lazno["words"] = reci(SR_REP[:6])
    lazno["detected"] = "sr"
    r, jdir, log, logs = pokreni(lazno, asr.AUTO_LANGUAGE)
    assert r.language == "sr" and cfg.load_job("p").language == "sr"
    assert (jdir / "sentences.sr.json").exists()
    assert "jezik detektovan: sr (sigurnost 0.91)" in log
    assert "NAPOMENA: Detekcija srpski meša sa hrvatskim" in log
    # uz auto-detekciju glosar ne ide u prepoznavanje — jezik se zna tek posle
    assert "glosar projekta: 0 termina ide u prepoznavanje" in log


def test_ispravke_se_ne_gaze_bez_force(lazno):
    lazno["words"] = reci(SR_REP[:6])
    pokreni(lazno, "sr")
    jdir = cfg.job_dir("p")
    sf = SentencesFile.model_validate(cfg.read_json(jdir / "sentences.sr.json"))
    sf.edited_by_human = True
    cfg.write_model(jdir / "sentences.sr.json", sf)
    with pytest.raises(SubtitisError, match="ručno menjan"):
        pokreni(lazno, "sr")
    # sa force: novi prepis, stari sačuvan kao kopija
    _, _, log, _ = pokreni(lazno, "sr", force=True)
    assert "prethodne rečenice sačuvane u" in log
    assert list(jdir.glob("sentences.sr.json.*.bak"))


def test_prazan_rezultat_je_greska(lazno):
    lazno["words"] = []
    with pytest.raises(SubtitisError, match="nije vratio nijednu reč"):
        pokreni(lazno, "sr")


def test_ulaz_koji_ne_postoji(lazno, tmp_path):
    with pytest.raises(SubtitisError, match="Ulazna datoteka ne postoji"):
        pipeline.run_transcribe(tmp_path / "nema.mp4", "sr", job_id="p")


def test_tisine_se_cuvaju_dok_zvuk_postoji(lazno, monkeypatch):
    """Krug 19: `pauses.json` se pravi pri prepoznavanju — posle roka zvuka više nema."""
    from subtitis import pauses

    monkeypatch.setattr(pauses, "detect", lambda wav: pauses.PausesFile(
        made_with={"vad": "silero"}, silences=[(1.0, 1.4), (5.0, 5.6)]))
    lazno["words"] = reci(SR_BEZ_ZNAKOVA)
    r, jdir, log, logs = pokreni(lazno, "sr")
    assert pauses.load("p").silences == [(1.0, 1.4), (5.0, 5.6)]
    assert "pauze: 2 tišine iz source.wav u pauses.json." in log


def test_tisine_koje_se_ne_mogu_izmeriti_ne_obaraju_posao(lazno):
    """Lažni `source.wav` nije zvuk: posao ide dalje, bez `pauses.json`."""
    from subtitis import pauses

    lazno["words"] = reci(SR_BEZ_ZNAKOVA)
    r, jdir, log, logs = pokreni(lazno, "sr")
    assert cfg.load_job("p").status == "transcribed"
    assert pauses.load("p") is None and "prelamanje bez pauza" in log


# --------------------------------------------------------------------------
# Glosar u prepoznavanju (krug 23)
# --------------------------------------------------------------------------


def _spy_terms(lazno, monkeypatch):
    seen = {}
    orig = asr.transcribe

    def spy(wav, **kw):
        seen["terms"] = kw.get("glossary_terms")
        return orig(wav, **kw)

    monkeypatch.setattr(asr, "transcribe", spy)
    lazno["words"] = reci(SR_BEZ_ZNAKOVA)
    return seen


def test_u_prepoznavanje_idu_samo_termini_snimka(lazno, monkeypatch):
    """Video-4 draft 2: termini iz `sr->fr` drugog projekta izašli su preko
    tišine na kraju snimka. Prevodni parovi idu samo u prevod."""
    seen = _spy_terms(lazno, monkeypatch)
    cfg.write_json(cfg.job_glossary_path("p"), {
        "sr->fr": {"Kamp": "Camp"}, "sr->en": {"Kamp": "Camp"},
        cfg.RECORDING_TERMS_KEY.format(lang="sr"): {"Valjevo": "Valjevo"},
    })
    _, _, log, _ = pokreni(lazno, "sr")
    assert seen["terms"] == ["Valjevo"]
    assert "glosar projekta: 1 termina ide u prepoznavanje" in log


def test_bez_termina_snimka_nema_nagovestaja(lazno, monkeypatch):
    seen = _spy_terms(lazno, monkeypatch)
    cfg.write_json(cfg.job_glossary_path("p"), {"sr->fr": {"Kamp": "Camp"}})
    pokreni(lazno, "sr")
    assert seen["terms"] == []


@pytest.mark.parametrize("izbor, ocekivano", [("none", {}), ("default", {"sr->fr": {"Kamp": "Camp"}})])
def test_komandna_linija_postuje_podesavanje_glosara(lazno, monkeypatch, izbor, ocekivano):
    """Do kruga 23 CLI i uvoz SRT-a su uvek kopirali podrazumevani glosar."""
    _spy_terms(lazno, monkeypatch)
    cfg.save_glossary({"sr->fr": {"Kamp": "Camp"}})
    monkeypatch.setattr(cfg, "new_project_glossary", lambda: izbor)
    assert not cfg.job_glossary_path("p").exists()
    pokreni(lazno, "sr")
    assert cfg.job_glossary("p") == ocekivano
