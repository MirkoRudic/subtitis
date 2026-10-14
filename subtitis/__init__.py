"""Subtitis — generisanje i prevod SRT titlova."""

from __future__ import annotations

import functools
import subprocess
from pathlib import Path
from typing import Any

#: Jedino mesto verzije: čita ga i `pyproject.toml` (dynamic), i potpis u
#: podnožju dashboarda, i `/api/identity`.
#: Prva dva broja prate projektni dokument (v1.5); `1.6.0` tek kad nastane
#: dokument v1.6. Krug rada nosi `ROUND`, ne treći broj.
__version__ = "1.5.0"

#: Krug rada u kome je napravljen ovaj kod — raste sa svakim krugom.
ROUND = 26

SCHEMA_VERSION = 1


@functools.lru_cache(maxsize=1)
def commit() -> str | None:
    """Kratki heš git commita iz kog radi kod; None van git stabla."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def version_string() -> str:
    """`Subtitis 1.5.0 · krug 25 · <heš>` — podnožje, `--version`, dnevnik."""
    parts = [f"Subtitis {__version__}", f"krug {ROUND}"]
    if commit():
        parts.append(commit())
    return " · ".join(parts)


def made_with() -> dict[str, Any]:
    """Čime je napravljen posao — u `job.json`, da se zna i posle nadogradnje."""
    return {"version": __version__, "round": ROUND, "commit": commit()}
