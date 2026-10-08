"""Testovi pronalaženja ffmpeg-a.

Povod: VS Code drži kopiju PATH-a od svog pokretanja, pa terminal iz njega ne
vidi ffmpeg instaliran u međuvremenu. Poruka „nije u PATH-u" je tačna a
beskorisna kad je program instaliran.
"""

from __future__ import annotations

import pytest

from subtitis import audio, config as cfg
from subtitis.config import SubtitisError


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    audio.reset_ffmpeg_cache()
    monkeypatch.delenv("SUBTITIS_FFMPEG", raising=False)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    yield
    audio.reset_ffmpeg_cache()


def fake_exe(tmp_path, name="ffmpeg.exe"):
    p = tmp_path / name
    p.write_bytes(b"MZ")
    return p


# --- redosled traženja ------------------------------------------------------


def test_konfiguracija_ima_prednost_nad_path(tmp_path, monkeypatch):
    mine = fake_exe(tmp_path, "moj.exe")
    monkeypatch.setenv("SUBTITIS_FFMPEG", str(mine))
    monkeypatch.setattr(audio.shutil, "which", lambda n: r"C:\iz\patha\ffmpeg.exe")
    assert audio.find_ffmpeg() == (str(mine), audio.ORIGIN_CONFIG)


def test_toml_se_cita(tmp_path, monkeypatch):
    mine = fake_exe(tmp_path)
    cfg.CONFIG_FILE.write_text(
        f'ffmpeg_path = "{str(mine).replace(chr(92), chr(92) * 2)}"\n', encoding="utf-8"
    )
    cfg.load_settings(reload=True)
    monkeypatch.setattr(audio.shutil, "which", lambda n: None)
    path, origin = audio.find_ffmpeg()
    assert path == str(mine) and origin == audio.ORIGIN_CONFIG


def test_promenljiva_okruzenja_pobedjuje_toml(tmp_path, monkeypatch):
    a, b = fake_exe(tmp_path, "a.exe"), fake_exe(tmp_path, "b.exe")
    cfg.CONFIG_FILE.write_text(
        f'ffmpeg_path = "{str(a).replace(chr(92), chr(92) * 2)}"\n', encoding="utf-8"
    )
    cfg.load_settings(reload=True)
    monkeypatch.setenv("SUBTITIS_FFMPEG", str(b))
    assert audio.find_ffmpeg()[0] == str(b)


def test_path_kad_nema_konfiguracije(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda n: r"C:\iz\patha\ffmpeg.exe")
    assert audio.find_ffmpeg() == (r"C:\iz\patha\ffmpeg.exe", audio.ORIGIN_PATH)


def test_poznata_lokacija_kad_nema_ni_u_pathu(tmp_path, monkeypatch):
    exe = fake_exe(tmp_path)
    monkeypatch.setattr(audio.shutil, "which", lambda n: None)
    monkeypatch.setattr(audio, "_from_known_locations", lambda: str(exe))
    assert audio.find_ffmpeg() == (str(exe), audio.ORIGIN_KNOWN)


# --- greške -----------------------------------------------------------------


def test_podeseni_ali_nepostojeci_puca_jasno(tmp_path, monkeypatch):
    monkeypatch.setenv("SUBTITIS_FFMPEG", str(tmp_path / "nema.exe"))
    with pytest.raises(SubtitisError, match="Podešeni ffmpeg ne postoji"):
        audio.find_ffmpeg()


def test_nigde_ga_nema(monkeypatch):
    monkeypatch.setattr(audio.shutil, "which", lambda n: None)
    monkeypatch.setattr(audio, "_from_known_locations", lambda: None)
    assert audio.find_ffmpeg() == (None, "")
    with pytest.raises(SubtitisError) as exc:
        audio.ffmpeg_path()
    msg = str(exc.value)
    assert "VS Code" in msg, "poruka mora da objasni zašto PATH nije dovoljan"
    assert "ffmpeg_path" in msg and "SUBTITIS_FFMPEG" in msg


def test_los_toml_puca_jasno(tmp_path):
    cfg.CONFIG_FILE.write_text("ovo nije = = toml\n", encoding="utf-8")
    with pytest.raises(SubtitisError, match="nije ispravan TOML"):
        cfg.load_settings(reload=True)


# --- ostalo -----------------------------------------------------------------


def test_kes_stedi_pretragu(monkeypatch):
    calls = []
    monkeypatch.setattr(audio.shutil, "which", lambda n: calls.append(n) or r"C:\f.exe")
    audio.find_ffmpeg()
    audio.find_ffmpeg()
    assert len(calls) == 1
    audio.reset_ffmpeg_cache()
    audio.find_ffmpeg()
    assert len(calls) == 2


def test_poznate_lokacije_pokrivaju_winget():
    assert any("WinGet" in p and "Gyan.FFmpeg" in p for p in audio.KNOWN_FFMPEG_GLOBS)


def test_prava_masina_nalazi_ffmpeg():
    """Na ovoj mašini ffmpeg jeste instaliran — mora ga naći bez obzira na PATH."""
    audio.reset_ffmpeg_cache()
    path, origin = audio.find_ffmpeg()
    if path is None:
        pytest.skip("ffmpeg nije instaliran na ovoj mašini")
    assert origin in (audio.ORIGIN_CONFIG, audio.ORIGIN_PATH, audio.ORIGIN_KNOWN)
