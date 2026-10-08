"""Jedna datoteka — jedno pismo.

Reprodukuje kvar uhvaćen 2026-08-17 na prevodu fr -> sr: rečenice 1–9 ćirilica,
10 i dalje latinica, sa oštrim presekom. Devet je `CHUNK_SIZE` — model pravilo
„piši isključivo latinicom" poštuje po odgovoru, pa ga za prvi čank prekrši a za
ostale ispoštuje. Ništa posle toga nije normalizovalo tekst: `sentences.sr.json`
je nosio `script="Latn"` kao tvrdnju koju niko ne proverava.

Zato ovde stoje tri stvari, i sve tri su deo popravke:
  * transliteracija je JEDAN prolaz nad celim prevodom, posle svih čankova;
  * `convert` je idempotentan — ponovljena primena ne menja ništa;
  * gotova datoteka se proverava, i mešano pismo puca glasno umesto da prođe.
"""

from __future__ import annotations

import pytest

from subtitis import pipeline
from subtitis.config import SubtitisError
from subtitis.models import Job, Sentence, SentencesFile
from subtitis.translit import to_cyrillic

CHUNK = 9


@pytest.fixture()
def posao(tmp_path, monkeypatch):
    """Posao sa `sentences.fr.json` spreman za prevod na srpski."""
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "GLOSSARY_DIR", tmp_path / "glossary")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)

    job_id = "20260817-fr"
    jdir = cfg.job_dir(job_id)
    jdir.mkdir(parents=True, exist_ok=True)
    cfg.save_job(
        Job(job_id=job_id, source_file=str(jdir / "snimak.wav"), language="fr",
            status="transcribed", duration=60.0)
    )
    cfg.write_model(
        jdir / "sentences.fr.json",
        SentencesFile(
            job_id=job_id, language="fr", derived_from="words.fr.json",
            sentences=[
                Sentence(id=i, s=float(i), e=float(i) + 1.0,
                         text=f"Phrase {i}.", words=(i, i))
                for i in range(1, 21)
            ],
        ),
    )
    return cfg, job_id


class FakeClient:
    """Ollama koji vraća ćirilicu za prvi čank, latinicu za ostale.

    Tačno ono što je model uradio: pravilo o pismu poštovano po odgovoru, a ne
    kroz ceo posao.
    """

    model = "lazni"

    def __init__(self) -> None:
        from subtitis.translate import CallStats

        self.stats = CallStats()
        self.cankova = 0

    def check(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def chat(self, system: str, user: str) -> str:
        block = user.split("ZA_PREVOD:\n", 1)[1].split("\n\nKONTEKST")[0]
        lines = [l for l in block.splitlines() if l.strip()]
        self.cankova += 1
        self.stats.requests += 1
        cirilica = self.cankova == 1
        out = []
        for line in lines:
            # lokalni broj (1..n) nazad, pravi broj iz teksta „Phrase {i}."
            n, text = line.split("|", 1)
            i = text.split()[1].rstrip(".")
            tekst = f"Rečenica {i} o nadživelim konjugacijama."
            out.append(f"{n}|{to_cyrillic(tekst) if cirilica else tekst}")
        return "\n".join(out)


def prevedi(cfg, job_id, monkeypatch, client=None):
    from subtitis import translate as tr

    client = client or FakeClient()
    monkeypatch.setattr(tr, "OllamaClient", lambda **kw: client)
    hops = pipeline.run_translate(job_id, "sr", source="fr", chunk_size=CHUNK)
    return client, hops


def procitaj(cfg, job_id, lang="sr") -> SentencesFile:
    return SentencesFile.model_validate(
        cfg.read_json(cfg.job_dir(job_id) / f"sentences.{lang}.json")
    )


# --- reprodukcija ----------------------------------------------------------


def test_mesano_pismo_iz_modela_zavrsava_kao_jedno(posao, monkeypatch):
    """Kvar iz prijave: prvi čank ćirilica, ostatak latinica."""
    cfg, job_id = posao
    client, _ = prevedi(cfg, job_id, monkeypatch)
    assert client.cankova == 3, "20 rečenica, čank 9 -> tri odgovora"

    sf = procitaj(cfg, job_id)
    from subtitis.translit import detect_script

    assert detect_script(" ".join(s.text for s in sf.sentences)) == "Latn"
    # baš na spornoj granici: 9 i 10 su iz različitih odgovora modela
    assert sf.sentences[8].text == sf.sentences[9].text.replace("10", "9")
    assert sf.script == "Latn"


def test_presek_na_granici_odlomka_bi_pao_bez_normalizacije(posao, monkeypatch):
    """Bez jednog prolaza posle prevoda datoteka bi bila mešana — i mora pući.

    Normalizacija se ovde isključuje da bi provera na kraju ostala jedina
    odbrana; ona je ta koja pretvara tih kvar u glasan.
    """
    cfg, job_id = posao
    monkeypatch.setattr(pipeline.translit, "is_biscriptal", lambda lang: False)
    prevedi(cfg, job_id, monkeypatch)

    sf = procitaj(cfg, job_id)
    from subtitis.translit import detect_script

    assert detect_script(" ".join(s.text for s in sf.sentences)) == "mixed"
    with pytest.raises(SubtitisError, match="nije u pismu Latn"):
        pipeline.assert_single_script(
            [(s.id, s.text) for s in sf.sentences], "Latn", what="sentences.sr.json"
        )


def test_normalizacija_se_prijavljuje_u_dnevniku(posao, monkeypatch):
    cfg, job_id = posao
    redovi: list[str] = []
    from subtitis import translate as tr

    monkeypatch.setattr(tr, "OllamaClient", lambda **kw: FakeClient())
    pipeline.run_translate(
        job_id, "sr", source="fr", chunk_size=CHUNK, on_log=redovi.append
    )
    poruka = [r for r in redovi if "vratio drugo pismo" in r]
    assert poruka, redovi
    assert "9 od 20" in poruka[0]

    job = cfg.load_job(job_id)
    faza = [f for f in job.history if f.phase.startswith("translate:")][-1]
    assert faza.detail["script_normalized"] == 9


def test_ponovljeni_prevod_ne_menja_nista(posao, monkeypatch):
    """Idempotencija na nivou pajplajna: drugi prolaz daje isti tekst."""
    cfg, job_id = posao
    prevedi(cfg, job_id, monkeypatch)
    prvi = [s.text for s in procitaj(cfg, job_id).sentences]

    prevedi(cfg, job_id, monkeypatch, client=FakeClient())
    assert [s.text for s in procitaj(cfg, job_id).sentences] == prvi


# --- SRT -------------------------------------------------------------------


def upisi_mesano(cfg, job_id) -> None:
    """Prevod u kom su prve tri rečenice ćirilica, ostale latinica."""
    recenice = []
    for i in range(1, 7):
        tekst = f"Rečenica {i} o konjugaciji."
        recenice.append(
            Sentence(id=i, s=float(i), e=float(i) + 1.0,
                     text=to_cyrillic(tekst) if i <= 3 else tekst, words=(i, i))
        )
    cfg.write_model(
        cfg.job_dir(job_id) / "sentences.sr.json",
        SentencesFile(job_id=job_id, language="sr",
                      derived_from="sentences.fr.json", sentences=recenice),
    )


@pytest.mark.parametrize("pismo", ["Latn", "Cyrl"])
def test_srt_je_u_jednom_pismu_i_iz_mesanog_ulaza(posao, monkeypatch, pismo):
    """I `Latn` mora da normalizuje: ranije je konvertovao samo za `Cyrl`."""
    cfg, job_id = posao
    upisi_mesano(cfg, job_id)
    r = pipeline.run_build(job_id, script=pismo, language="sr")

    from subtitis.translit import detect_script

    tekst = r.path.read_text(encoding="utf-8")
    linije = [l for l in tekst.splitlines() if l and "-->" not in l and not l.isdigit()]
    assert detect_script(" ".join(linije)) == pismo
    assert r.cues > 0


def test_srt_puca_glasno_ako_ostane_mesan(posao, monkeypatch):
    """Ako konverzija ikad prestane da pokriva sve, gotov SRT se ne upisuje."""
    cfg, job_id = posao
    upisi_mesano(cfg, job_id)
    # `convert` koji „preskoči kad je već ćirilica" — tačno sumnja iz prijave.
    monkeypatch.setattr(
        pipeline.translit,
        "convert",
        lambda text, script: text if any(ch in "абвгдђежзијклмнопрстћуфхцчшљњџ"
                                         for ch in text.lower()) else text,
    )
    with pytest.raises(SubtitisError, match="nije u pismu"):
        pipeline.run_build(job_id, script="Latn", language="sr")

    srt = cfg.job_dir(job_id) / "out"
    assert not any(srt.glob("*.srt")), "pokvaren SRT ne sme da završi na disku"
