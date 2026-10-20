"""Krug 30: ostatak površine — „lokalno" nije isto što i „bezbedno" čim
postoji ruta koju neko drugi može da pozove.

* poreklo zahteva (CSRF): forma ili `fetch` sa tuđe stranice ne sme da menja
  stanje; HTMX i forme sa samog dashboarda moraju da prođu (pravi pregledač:
  vidi izveštaj kruga 30);
* sve što od pozivaoca ulazi u putanju: jezik, profil, ime fajla;
* vrednosti koje idu u `subtitis.toml`.
"""

from __future__ import annotations

import pytest
from test_web import make_job, web  # noqa: F401 — `web` je fixture

from subtitis import config as cfg_mod
from subtitis.web.app import cross_site_reason

SAME = "http://testserver"


# --------------------------------------------------------------------------
# poreklo zahteva
# --------------------------------------------------------------------------


@pytest.mark.parametrize("headers, ok", [
    ({}, True),                                                  # ne iz pregledača (MCP, curl)
    ({"origin": SAME, "host": "testserver"}, True),              # sam dashboard
    ({"sec-fetch-site": "same-origin", "origin": SAME, "host": "testserver"}, True),
    ({"sec-fetch-site": "none"}, True),                          # korisnik sam otkucao/obeleživač
    ({"origin": "http://napadac.example", "host": "testserver"}, False),
    ({"origin": "http://127.0.0.1:3000", "host": "127.0.0.1:8010"}, False),  # druga lokalna app
    ({"origin": "null", "host": "testserver"}, False),           # sandbox iframe, file://
    ({"sec-fetch-site": "cross-site"}, False),
    ({"sec-fetch-site": "same-site"}, False),                    # localhost:3000 je „same-site"
])
def test_poreklo_zahteva(headers, ok):
    assert (cross_site_reason("POST", headers) is None) is ok
    assert cross_site_reason("GET", headers) is None, "čitanje se ne proverava"


def test_tudja_stranica_ne_moze_da_obrise_posao(web):  # noqa: F811
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.post(f"/jobs/{jid}/delete", headers={"origin": "http://napadac.example"},
                    follow_redirects=False)
    assert r.status_code == 403 and "odbijen" in r.json()["detail"]
    r = client.post(f"/jobs/{jid}/delete", headers={"sec-fetch-site": "cross-site"}, follow_redirects=False)
    assert r.status_code == 403
    assert db.get_job(jid) is not None and cfg.job_dir(jid).exists()
    r = client.post(f"/jobs/{jid}/delete", headers={"origin": SAME, "sec-fetch-site": "same-origin"},
                    follow_redirects=False)
    assert r.status_code == 303 and db.get_job(jid) is None, "sam dashboard sme"


def test_tudja_stranica_ne_menja_podesavanja_ni_api(web):  # noqa: F811
    client, db, cfg = web
    jid = make_job(cfg, db)
    evil = {"origin": "http://napadac.example"}
    assert client.post("/settings", data={"default_profile": "tv", "default_target_language": "en",
                                          "llm_model": "x", "source_wav_ttl_days": "1"},
                       headers=evil).status_code == 403
    assert client.post(f"/api/jobs/{jid}/translate", json={"languages": ["en"]}, headers=evil).status_code == 403
    assert client.post("/api/shutdown", json={}, headers=evil).status_code == 403
    assert db.pending_task(jid) is None


# --------------------------------------------------------------------------
# putanje od pozivaoca
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["C:tajna.srt", "C:tajna.txt", "..srt", "a.txt", "x\\..\\y.srt", "nul:.srt"])
def test_download_samo_srt_iz_out(web, name):  # noqa: F811
    """Na Windows-u je `out / "C:tajna.txt"` putanja na disku C — van posla."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    r = client.get(f"/jobs/{jid}/download/{name}")
    assert r.status_code in (400, 404) and r.status_code != 200


def test_download_pravog_fajla_radi(web):  # noqa: F811
    client, db, cfg = web
    jid = make_job(cfg, db)
    out = cfg.job_dir(jid) / "out"
    out.mkdir(exist_ok=True)
    (out / "snimak 1 (final).sr-Latn.srt").write_text("1\n", encoding="utf-8")
    assert client.get(f"/jobs/{jid}/download/snimak 1 (final).sr-Latn.srt").status_code == 200


BAD_LANG = ["../../x", "sr/../..", "..", "sr.json", "C:sr"]


@pytest.mark.parametrize("bad", BAD_LANG)
def test_jezik_iz_zahteva_ne_ulazi_u_putanju(web, bad):  # noqa: F811
    client, db, cfg = web
    jid = make_job(cfg, db)
    calls = [
        client.get(f"/jobs/{jid}", params={"lang": bad}),
        client.get(f"/jobs/{jid}/versions", params={"lang": bad}),
        client.post(f"/jobs/{jid}/translate", data={"to": bad}),
        client.post(f"/jobs/{jid}/build", data={"language": bad}),
        client.post(f"/jobs/{jid}/rebuild", data={"language": bad}),
        client.post(f"/jobs/{jid}/shorten", data={"language": bad}),
        client.post(f"/jobs/{jid}/restore", json={"lang": bad, "name": "x"}),
        client.post(f"/jobs/{jid}/sentences", json={"lang": bad, "sentences": []}),
        client.post(f"/jobs/{jid}/shorten/discard", json={"lang": bad}),
    ]
    for r in calls:
        assert r.status_code == 400, (r.request.url, r.status_code, r.text[:120])
    assert db.pending_task(jid) is None


@pytest.mark.parametrize("bad", ["C:/Windows/win.ini", "../../profiles/tv", "tv.json", "TV"])
def test_profil_iz_zahteva_je_ime_ne_putanja(web, bad):  # noqa: F811
    """`load_profile` prima i putanju — za komandnu liniju, ne za zahtev."""
    client, db, cfg = web
    jid = make_job(cfg, db)
    for r in (
        client.post(f"/jobs/{jid}/translate", data={"to": "en", "profile": bad}),
        client.post(f"/jobs/{jid}/build", data={"language": "sr", "profile": bad}),
        client.post(f"/jobs/{jid}/rebuild", data={"language": "sr", "profile": bad}),
        client.post(f"/api/jobs/{jid}/rebuild", json={"profile": bad}),
        client.post(f"/api/jobs/{jid}/translate", json={"languages": ["en"], "profile": bad}),
    ):
        assert r.status_code == 400 and "Neispravno ime profila" in r.text, (r.request.url, r.text[:120])


def test_dobar_jezik_i_profil_i_dalje_rade(web):  # noqa: F811
    client, db, cfg = web
    jid = make_job(cfg, db)
    assert client.get(f"/jobs/{jid}", params={"lang": "sr"}).status_code == 200
    assert client.post(f"/jobs/{jid}/build", data={"language": "sr-Latn", "profile": "tv"},
                       follow_redirects=False).status_code == 303


# --------------------------------------------------------------------------
# subtitis.toml
# --------------------------------------------------------------------------


def test_ime_modela_ne_sme_da_dopise_kljuc(web):  # noqa: F811
    client, db, cfg = web
    base = {"default_profile": "tv", "default_target_language": "en", "source_wav_ttl_days": "1"}
    r = client.post("/settings", data={**base, "llm_model": "x\nport = 1"})
    assert r.status_code == 400 and "Neispravno ime modela" in r.text
    assert client.post("/settings", data={**base, "llm_model": "hf.co/korisnik/model:Q4_K_M"},
                       follow_redirects=False).status_code == 303


def test_toml_vrednost_sa_novim_redom_ostaje_jedna_vrednost():
    import tomllib

    v = cfg_mod._toml_value('a\nport = 1"b\\c\t')
    assert tomllib.loads("x = " + v) == {"x": 'a\nport = 1"b\\c\t'}
