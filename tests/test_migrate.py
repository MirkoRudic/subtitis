"""Migracija imena profila u krugu 8 (`job.json` verzija 1 -> 2)."""

from __future__ import annotations

import json

import pytest

from subtitis import migrate


@pytest.fixture()
def disk(tmp_path, monkeypatch):
    from subtitis import config as cfg

    monkeypatch.setattr(cfg, "JOBS_DIR", tmp_path / "jobs")
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / "subtitis.toml")
    monkeypatch.setattr(cfg, "_settings_cache", None)
    return cfg


def stari_posao(cfg, job_id, profile, *, suggestions=(), version=1):
    """`job.json` kakav je bio pre kruga 8."""
    jdir = cfg.job_dir(job_id)
    jdir.mkdir(parents=True)
    detail = {"profile": profile, "script": "Latn", "language": "sr", "cues": 10}
    if suggestions:
        detail["profile_suggestions"] = [
            {"profile": s, "max_cps": 23.0, "max_chars": 47, "overflow": 1} for s in suggestions
        ]
    data = {
        "schema_version": version, "job_id": job_id, "source_file": "a.wav", "status": "done",
        "history": [
            {"phase": "transcribe", "status": "ok", "started_at": "x", "finished_at": "x", "duration_s": 1.0, "detail": {}},
            {"phase": "build", "status": "ok", "started_at": "x", "finished_at": "x", "duration_s": 1.0, "detail": detail},
        ],
    }
    (jdir / "job.json").write_text(json.dumps(data), encoding="utf-8")
    return jdir


def build_profile(cfg, job_id):
    data = json.loads((cfg.job_dir(job_id) / "job.json").read_text(encoding="utf-8"))
    return data["schema_version"], data["history"][1]["detail"]["profile"]


@pytest.mark.parametrize("staro,novo", [
    ("netflix", "tv"), ("tv", "tv"), ("web", "tv"), ("digital", "web"),
])
def test_svako_staro_ime(disk, staro, novo):
    cfg = disk
    jdir = stari_posao(cfg, "p", staro)
    m = migrate.migrate_job(jdir)
    assert m.status == "migrated"
    assert build_profile(cfg, "p") == (2, novo)
    log = (jdir / "log.txt").read_text(encoding="utf-8")
    assert "migracija profila na verziju 2" in log
    if staro != novo:
        assert f"{staro} -> {novo}" in log and "Razlog" in log


def test_novi_web_se_ne_migrira_ponovo(disk):
    """Najvažnije: `web` posle migracije znači jednoredni i ne sme da postane `tv`."""
    cfg = disk
    jdir = stari_posao(cfg, "p", "digital")
    migrate.migrate_job(jdir)
    assert build_profile(cfg, "p") == (2, "web")
    assert migrate.migrate_job(jdir).status == "current"
    assert build_profile(cfg, "p") == (2, "web")


def test_posao_na_novoj_verziji_se_ne_dira(disk):
    cfg = disk
    jdir = stari_posao(cfg, "p", "web", version=2)
    before = (jdir / "job.json").read_bytes()
    assert migrate.migrate_job(jdir).status == "current"
    assert (jdir / "job.json").read_bytes() == before
    assert not (jdir / "log.txt").exists()


def test_nepoznato_ime_se_ne_pogadja(disk):
    cfg = disk
    jdir = stari_posao(cfg, "p", "moj-profil")
    before = (jdir / "job.json").read_bytes()
    m = migrate.migrate_job(jdir)
    assert (m.status, m.unknown) == ("skipped", ["moj-profil"])
    assert (jdir / "job.json").read_bytes() == before
    assert "UPOZORENJE" in (jdir / "log.txt").read_text(encoding="utf-8")


def test_izvorni_profil_ostaje_a_predlozi_se_prevode(disk):
    cfg = disk
    jdir = stari_posao(cfg, "p", "original", suggestions=["web"])
    migrate.migrate_job(jdir)
    data = json.loads((jdir / "job.json").read_text(encoding="utf-8"))
    det = data["history"][1]["detail"]
    assert det["profile"] == "original"
    assert det["profile_suggestions"][0]["profile"] == "tv"


def test_predlog_skracivanja_nosi_novo_ime(disk):
    cfg = disk
    jdir = stari_posao(cfg, "p", "web")
    cfg.write_json(jdir / "shorten.fr.json", {"profile": "web", "items": []})
    migrate.migrate_job(jdir)
    assert cfg.read_json(jdir / "shorten.fr.json")["profile"] == "tv"


def test_ucitavanje_starog_posla_ga_migrira(disk):
    """CLI ne prolazi kroz start dashboarda — `load_job` migrira sam."""
    cfg = disk
    stari_posao(cfg, "p", "netflix")
    job = cfg.load_job("p")
    assert job.schema_version == 2
    assert job.history[1].detail["profile"] == "tv"


def test_migrate_all_prolazi_kroz_sve(disk):
    cfg = disk
    for k, name in enumerate(["netflix", "tv", "web", "digital"]):
        stari_posao(cfg, f"p{k}", name)
    stari_posao(cfg, "nov", "web", version=2)
    stari_posao(cfg, "cudan", "xyz")
    by = {m.job_id: m.status for m in migrate.migrate_all()}
    assert by == {"p0": "migrated", "p1": "migrated", "p2": "migrated", "p3": "migrated",
                  "nov": "current", "cudan": "skipped"}


# --- Podešavanja --------------------------------------------------------------


def test_podrazumevani_profil_u_podesavanjima(disk):
    cfg = disk
    cfg.CONFIG_FILE.write_text('default_profile = "web"\n', encoding="utf-8")
    assert migrate.migrate_settings() == ("web", "tv")
    assert cfg.default_profile() == "tv"
    assert cfg.load_settings(reload=True)["profiles_version"] == 2
    # drugi put: `web` bi sada bio nov, ali vrednost je `tv` i oznaka postoji
    assert migrate.migrate_settings() is None
    assert cfg.default_profile() == "tv"


def test_snimanje_podesavanja_upisuje_oznaku(disk):
    """Inače bi `web` snimljen posle kruga 8 sledeći start pretvorio u `tv`."""
    cfg = disk
    cfg.save_settings({"default_profile": "web"})
    assert migrate.migrate_settings() is None
    assert cfg.default_profile() == "web"


# --- Zadaci u redu -------------------------------------------------------------


def test_zadatak_u_redu_dobija_novo_ime(disk):
    from subtitis.web import db  # baza je u JOBS_DIR, pa je i ona privremena

    cfg = disk
    stari_posao(cfg, "p", "web")
    db.init_db()
    db.create_job("p", "a.wav", "/a.wav", "sr")
    db.enqueue("p", "build", {"language": "sr", "profile": "digital"})
    migrate.migrate_all(on_migrated=db.migrate_queued_profiles)
    assert json.loads(db.pending_task("p")["params"])["profile"] == "web"
    # drugi prolaz ne dira ni posao ni zadatak
    migrate.migrate_all(on_migrated=db.migrate_queued_profiles)
    assert json.loads(db.pending_task("p")["params"])["profile"] == "web"
