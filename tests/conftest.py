from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subtitis.models import Profile, Word  # noqa: E402
from subtitis.segment import tokenize_text  # noqa: E402


def make_words(
    text: str,
    *,
    start: float = 0.0,
    dur_per_char: float = 0.06,
    gap: float = 0.02,
    p: float = 0.95,
    start_index: int = 0,
) -> list[Word]:
    """Napravi listu `Word` iz teksta, sa ravnomernim vremenima."""
    out: list[Word] = []
    t = start
    for k, tok in enumerate(tokenize_text(text)):
        d = max(0.05, len(tok) * dur_per_char)
        out.append(Word(i=start_index + k, s=round(t, 3), e=round(t + d, 3), w=tok, p=p))
        t += d + gap
    return out


@pytest.fixture
def profil() -> Profile:
    return Profile(name="tv")


_PRIMER_GLOSARA: list[bytes | None] = []


def _primer_glosara() -> bytes | None:
    """`glossary/default.json` iz git indeksa — izmišljeni primer iz repoa.

    Na radnoj mašini na disku stoji stvarni glosar (skip-worktree, vidi
    tests/test_javni_repo.py); testovi ga ne smeju videti, jer bi tada
    zavisili od mašine. Čita se jednom po sesiji; bez gita ostaje disk.
    """
    if not _PRIMER_GLOSARA:
        import subprocess

        try:
            out = subprocess.run(
                ["git", "show", ":glossary/default.json"],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            out = None
        _PRIMER_GLOSARA.append(out)
    return _PRIMER_GLOSARA[0]


@pytest.fixture(autouse=True)
def _bez_pravih_poslova(tmp_path_factory, monkeypatch):
    """Nijedan test ne sme da dira pravi `jobs/`, `subtitis.toml`, glosar ni profile.

    Krug 8: test koji je samo pokrenuo dashboard (`lifespan`) migrirao je
    stvarne poslove na disku. Testovi koji žele svoj folder ga i dalje
    postavljaju sami; ovo je donja granica.
    """
    from subtitis import config as cfg

    import shutil

    root = tmp_path_factory.mktemp("izolovano")
    monkeypatch.setattr(cfg, "JOBS_DIR", root / "jobs")
    monkeypatch.setattr(cfg, "CONFIG_FILE", root / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    # Krug 11: test koji je snimio glosar prepisao je PRAVI glosar — ova
    # granica je pokrivala samo jobs/ i subtitis.toml. Glosar i profili se
    # kopiraju: čitanje vidi isto što i do sada, a upis ostaje u kopiji.
    for name in ("GLOSSARY_DIR", "PROFILES_DIR"):
        real = getattr(cfg, name)
        copy = root / real.name
        if real.exists():
            shutil.copytree(real, copy, ignore=shutil.ignore_patterns("*.bak"))
        monkeypatch.setattr(cfg, name, copy)
    primer = _primer_glosara()
    if primer is not None:
        (cfg.GLOSSARY_DIR / "default.json").write_bytes(primer)
