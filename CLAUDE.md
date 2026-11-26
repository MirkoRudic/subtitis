Važe zajednička pravila Mirkovih aplikacija (`D:\VAJB\vajb-osnova\PRAVILA.md`), uz izuzetke ispod. Šta Subtitis radi iz ugla korisnika je u `SPEC.md`; tehnika, merenja i odluke u `README.md`, `document/` i `merenja/dnevnik.md`. Dokle smo stigli: `STANJE.md`.

# Subtitis

Folder `D:\VAJB\subtitis-javni` (git klon ovog javnog repoa; stvarni glosar van commita preko `skip-worktree`), port 8010, Python 3.11 u `.venv`, prečica `subtitis shortcut`.

## Pravila ovog repoa
- Javni repo: nikad stvarna imena, rečenice ili snimci. `tests/test_javni_repo.py` pada na zabranjene pojmove i četvorke reči iz stvarnih snimaka.
- Merenja pre pretpostavki; svaki krug ima red u `CHANGELOG.md`, verzija i krug u `subtitis/__init__.py`.

## Izuzeci od zajedničkih pravila
- T3 brojevi i UI su već srpski sa decimalnim zarezom; UI1–UI19 primenjeni samo gde ne menjaju izmereno ponašanje editora.
- UI12: tema je samo tamna; Podešavanja su stranica, a delovi app (vrednosti, glosar, interpunkcija) idu pre Dnevnika.
- L1: dnevnik (`data/logs/subtitis_*.log`, `subtitis/applog.py` iz zajedničkih modula) beleži start, ažuriranje i padove; obrada ima svoje izveštaje po poslu u `jobs/`.
- P8, V4: nema installer-a; javni repo je izvor, a kopija na laptopu se ažurira iz gita (`subtitis/appupdate.py`).
