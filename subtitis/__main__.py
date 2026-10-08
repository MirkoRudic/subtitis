# Pre svega ostalog: pod `pythonw.exe` (prečica na desktopu) nema ni stdout ni
# načina da se vidi pad. `bootstrap` podmeće struje i hvata izuzetke u datoteku
# — i to mora pre uvoza `cli`, da se zabeleži i puknuti uvoz.
from .bootstrap import bootstrap, crash_guard

bootstrap()

with crash_guard("uvoz i pokretanje"):
    from .cli import main

    main()
