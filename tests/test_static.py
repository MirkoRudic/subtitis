"""`subtitis/web/static/` se servira i ide u javni repozitorijum (krug 28).

Dva sloja, svaki svoj posao (`krugovi/28-odgovor.md`): `.gitignore` sprečava
štetu — da `git add -A` ne odnese tuđi zvuk na javni repo — a ovaj test
sprečava zaborav: pada za SVAKI fajl u `static/` koji nije u gitu, i za one
koje `.gitignore` sakriva. Prolazi tek kad se fajl ukloni.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "subtitis" / "web" / "static"


def _tracked() -> set[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", "subtitis/web/static"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("nema gita — provera se radi u radnom stablu")
    return {p for p in out.split("\0") if p}


def test_static_nema_fajlova_van_gita():
    tracked = _tracked()
    extra = sorted(
        p.relative_to(ROOT).as_posix() for p in STATIC.rglob("*")
        if p.is_file() and p.relative_to(ROOT).as_posix() not in tracked
    )
    assert not extra, (
        "u subtitis/web/static/ stoje fajlovi koji nisu u gitu — folder se servira i ide u javni "
        f"repozitorijum. Ukloni ih kad posao sa njima završi: {extra}. Privremeni fajlovi idu u tmp/ "
        "u korenu aplikacije (prazni se pri pokretanju)."
    )


def test_tmp_se_prazni_pri_pokretanju(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "TMP_DIR", tmp_path / "tmp")
    assert cfg.empty_tmp() == 0 and cfg.TMP_DIR.is_dir()
    (cfg.TMP_DIR / "a.mp3").write_bytes(b"x")
    (cfg.TMP_DIR / "dir").mkdir()
    (cfg.TMP_DIR / "dir" / "b").write_text("y")
    assert cfg.empty_tmp() == 2 and list(cfg.TMP_DIR.iterdir()) == []


def test_server_prazni_tmp_pri_pokretanju(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from subtitis import config as cfg
    from subtitis.web import app as app_mod

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "TMP_DIR", tmp_path / "tmp")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp" / "ostalo.wav").write_bytes(b"x")
    with TestClient(app_mod.app):
        # ostaje samo PID ovog servera (krug 29)
        assert [p.name for p in (tmp_path / "tmp").iterdir()] == ["server.pid"]
