# Istorija

Alat je razvijan u krugovima: svaki krug je jedan zadatak sa spiskom zahteva i
izveštajem. Ovo je sažetak istorije rada iz privatnog repozitorijuma, jedan
red po krugu; javni repozitorijum počinje jednim commitom nad očišćenim stanjem
(krug 17). Krugovi bez broja su radni dani između numerisanih krugova.

- **Faze 1–3** (2026-08-12) — ASR pajplajn, prevod, dashboard; 497 testova.
- **Krug 2** (2026-08-12) — kapija za punktuaciju (povlačenje, dozvole, malo slovo: 102 s → 60 s, 8 → 1 odbijen), elizija, srpski kao ciljni jezik; profil, opisi, pismo, dve kolone; desktop režim, poništavanje, prečice, autosnimanje, podešavanja; prečica nalazi Desktop preko Windows API-ja.
- **2026-08-17** — bootstrap preko `pythonw` bez stdout-a uz crash log; transliteracija u jednom prolazu nad celim prevodom, provera jedinstvenog pisma; 732 testa.
- **2026-08-19** — mera ponavljanja, minimalno trajanje po reči, `low_confidence` po udelu umesto `any()` (dobar snimak 41 → 4 %, loš 80 → 40 %), zarez u kapiji za restauraciju, podrazumevano bez detekcije govora, savet po odbačenom zvuku; 814 testova.
- **2026-09-25** — prevod postojećeg SRT-a sa očuvanim tajmingom, francuska tipografija, glasne neprevedene rečenice, termini glosara, veznici na engleskom i francuskom (5 → 2 granice iza veznika); 901 test.
- **2026-09-28** — detekcija izvornog jezika i neprevedenih rečenica, navodnici i apostrofi, profil iz ulaznog SRT-a, izveštaj gore-od-originala, port iz podešavanja sa proverom identiteta instance, sečenje rečenica dužih od 15 s, numerisanje unutar odlomka od 1 (16 → 0 odbijenih, −44 % vremena), skraćivanje prekoračenja uz obavezan pregled, pravilo o registru izmereno i odbačeno, uredno brisanje posla u toku; 964 testa.
- **Krug 4** (2026-10-01) — objašnjenja u interfejsu u dva sloja, stranica „Kako ovo radi"; projektni dokument; 984 testa.
- **Krug 5** (2026-10-01) — reči bez zvuka iza sebe, glosar po poslu, upozorenje o muzici; 999 testova.
- **Krug 6** (2026-10-07) — jedan profil za web sa pseudonimima, profil `digital` sa mekom donjom granicom reda i prijavom prekratkih cue-ova, veličina slova u izlaznom SRT-u i u formi za prevod.
- **Krug 7** (2026-10-07) — suženo izuzeće donje granice reda, negacija u tabelama preloma, ravnomeran udeo izmeren i isključen.
- **Krug 8** (2026-10-07) — profili tv / web / digital uz migraciju `job.json`, nov interfejs po mokapu (projekti, editor, izveštaj o obradi), potpis i verzija na jednom mestu, lokalni fontovi; 1.4.0.
- **Krug 9** (2026-10-07) — manje teksta na ekranu (ispod trake najviše jedan red), decimalni zarez svuda; 1.4.1.
- **Krug 10** (2026-10-07) — potvrda pre prelamanja, poruke bez reči iz koda, čišćenje interfejsa; 1.4.2.
- **Krug 11** (2026-10-07) — zamena zauzetog fajla, lista koja miruje kad ništa ne radi, brisanje sa potvrdom; 1.4.3.
- **Krug 12** (2026-10-07) — jedan prolaz kroz ceo interfejs po šest pravila, prelom ispod rečenice, „Pokušaj ponovo" za korak koji je pao; 1.4.4–1.4.5.
- **Krug 13** (2026-10-07) — editor bez dugmeta „Prelomi", stalna imena menija u traci editora, info ikonica u omotaču sa oznakom; 1.4.6–1.4.8.
- **Krug 14** (2026-10-07) — izlazak iz izmenjenog polja snima i prelama ceo titl, boje ikonice se čitaju iz `app.css`; 1.4.9.
- **Krug 15** (2026-10-07) — ikonica u kartici, merenje pokrivenosti testovima, uklonjen mrtav kod; 1.4.10.
- **Krug 16** (2026-10-07) — testovi za `run_transcribe` i pivot prevod (pokrivenost 84 → 87 %).
- **Krug 17** (2026-10-08) — javni repozitorijum: svi primeri zamenjeni izmišljenim, test koji pada na svaki pojam sa spiska zabranjenih (`tests/test_javni_repo.py`), ova istorija umesto starih commitova. Druga provera po pravilu umesto po spisku: nijedna četvorka uzastopnih reči iz izvornog materijala (krug 17b); licenca MIT.
- **Krug 19** (2026-10-12) — rad preseljen u javni folder (`.venv`, poslovi, stvarni glosar van commita preko `skip-worktree`); pauze iz zvuka (Silero VAD) u `pauses.json`; razmak između reči u `words.json` je nula u 825 od 830 granica; pauza kao signal za prelom izmerena i odbačena (4 bolje : 4 gore).
- **Krug 20** (2026-10-12) — strana veznika u srpskom izmerena: `before` i `before` sa `da` van veznika čitaju se bolje u `tv` (14 : 1), ali ne u `web`/`digital` (11 : 6) i podižu mere koje ne smeju da porastu; ostaje `after`. Napomena o `skip-worktree` glosaru u README.
- **Krug 21** (2026-10-12) — srpski u dvorednom prelomu lomi ispred veznika (`conjunction_side_multiline`), jednoredni ostaje iza; `web` i `digital` 28 znakova po redu (donja 14) umesto 20 — nestaju rezovi iza predloga i ispred enklitike, prekoračenja CPS-a 52 -> 18.
