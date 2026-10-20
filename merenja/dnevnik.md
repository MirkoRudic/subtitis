# Dnevnik merenja

Pamćenje projekta: nalazi, pravila merenja i odluke, sa izvorom.

- **Skripte** su u `scripts/` (u repozitorijumu) — pokreću se ponovo.
- **Sirovi podaci** (prepisi, `words` svih proba) su u `merenja/krug*/` i u
  `.gitignore`, iz istog razloga kao `jobs/`: to je sadržaj tuđih snimaka,
  a deo njih su izjave korisnika socijalnih usluga. Postoje samo lokalno, na
  mašini na kojoj je mereno; putanja je upisana uz svaki nalaz.
- U dnevniku smeju stajati citati **izmišljenog** teksta (ono što je model
  dodao), ali ne odlomci pravog govora iz snimaka.

---

## Pravila merenja

### Gde postupak nije determinističan, jedan prolaz nije merenje (2026-10-01)

Izveštaj je na osnovu jednog prolaza tvrdio da proba A (VAD + glosar) daje 11
reči. Pet ponavljanja daje 20–52. Taj broj nije bio merenje nego izvlačak.

To je drugačiji obrazac od dosadašnjih grešaka: ranije je hipoteza bila
pogrešna, ovde je hipoteza bila tačna, a uzorak nedovoljan.

Granica je izmerena: od četiri ćelije VAD × glosar, tri su u pet ponavljanja
dale identičan prepis bajt za bajt. Nedeterminizam se javlja samo u ćeliji sa
VAD-om i glosarom — verovatno kroz `TEMPERATURE_FALLBACK`, koji posle
neuspelog dekodiranja ponavlja nasumično. Pravilo: gde ima temperature (VAD +
`hotwords`, a i svaki LLM korak sa temperaturom > 0), ponavljati i javljati
raspon, ne jedan broj.

---

## 2026-10-01 — Pokrivenost vara kad model izmišlja

Posao `20261001-175002-Radionica_draft-1` (32 s, lokalni dijalekat, muzički
džingl na kraju). Isti fajl, ista podešavanja osim glosara:

| proba | pokrivenost | šta je u prepisu preko džingla |
|---|---|---|
| VAD isključen, glosar pun (kontrola) | **87,4%** | izmišljen tekst iz glosara |
| VAD isključen, glosar prazan | 61,5% | bez teksta iz glosara |

Viša pokrivenost je ovde bila **gori** znak: izmišljeni tekst „pokriva" džingl.
Uputstvo trenutno kaže da pokrivenost „vredi pogledati ako je ispod 80%" —
to važi samo kad model ne izmišlja. **Tekst se za sada ne menja**; ispraviti
kad bude poznato čime se pokrivenost dopunjuje (npr. udeo reči nultog
trajanja ili reči iz glosara preko negovora).

Izvor: `merenja/krug5/proba_K_*.json`, `merenja/krug5/proba_B_*.json`
(lokalno); skripta `scripts/krug5_probe_vad_glosar.py`.

## 2026-10-01 — VAD × glosar na Radionici draft 1: problem je interakcija

Četiri ćelije, svaka pet puta, svako pokretanje u zasebnom procesu:

| | glosar pun | glosar prazan |
|---|---|---|
| **VAD isključen** | K: 77 reči, 87,4%, 6 nultih, 19 reči glosara — 5/5 isto | B: 66 reči, 61,5%, 3 nulte, `Top 10 najboljih uvijeka na svijetu` preko džingla — 5/5 isto |
| **VAD „normalno"** | A: 20–52 reči, 55–67%, 0–10 nultih, 4–24 reči glosara — **svaki put drugačije** | C: 59 reči, 60,6%, 0 nultih, bez izmišljenog teksta — 5/5 isto |

(„Reči iz glosara" u B i C su `da`, `se`, `za`, `na` — reči iz termina kao
„s obzirom na to da", ne termini.)

- **Glosar je izvor halucinacije sa terminima** (K naspram B).
- **Nepredvidljiva je samo kombinacija VAD + `hotwords`** (A). Verovatno: VAD
  seče zvuk u kratke odsečke, svaki je nov prozor sa `hotwords`, model na
  kratkom odsečku odustane i vrati prompt.
- **Glosar nije jedini izvor izmišljanja**: B izmišlja `Top 10 najboljih…` bez
  ijednog termina. Popravka glosara uklanja jedan izvor, ne pojavu.
- **VAD nije bolji — samo uklanja priliku.** Pošteno poređenje bez glosara: B
  ima 66 reči, od kojih je šest izmišljeno (`Top 10 najboljih uvijeka na
  svijetu`), dakle oko 60 pravih; C ima 59 pravih. Po tačnosti govora su
  izjednačeni. VAD nije dodao tačnost: džingl nije pustio do modela.
- **Nijedna podrazumevana vrednost nije dobra za oba materijala.** Na snimku bez
  muzike te koristi nema, a gubitak govora ostaje — izmereno ranije na tihom
  snimku: 189 reči bez VAD-a prema 179 sa pragom 0,35. Zato je ovo **savet po
  materijalu**, ne nova podrazumevana vrednost: detekcija ostaje isključena, a
  upozorenje na stranici posla izričito pominje muziku (džingl, špica) kad
  negovora ima 25%+.

Izvor: `merenja/krug5/proba_{K,A,B,C}_{1..5}.json`,
`merenja/krug5/probe_ponavljanja.log` (lokalno); skripta
`scripts/krug5_probe_vad_glosar.py`.

## 2026-10-01 — Reči nultog trajanja: prag niza

Šest urednih francuskih snimaka (`import/`) + Radionica draft 1, pravilo „niz uzastopnih
reči sa `e - s < 0,02 s`", interpunkcija providna, francuska elizija spojena
kao u produkciji (`asr.merge_elisions`):

| snimak | niz ≥ 2 | niz ≥ 3 |
|---|---|---|
| Radionica draft 1 | 1 — `letnja radionica OKC Divčibare Srbija` (glosar) | 1 |
| snimak A | 1 — `titrage ST' 501` (Whisper-ova izmišljotina) | 1 |
| snimci B, C, D, E, F | 0 | 0 |

Bez spajanja elizije snimak E daje niz od dva tokena na 83,76 s: ponovljena
elidirana reč, sabijena u nulu, koja je u produkciji jedan token — dakle nije
niz. Prag 2 nema lažnu uzbunu na uzorku i hvata `na svijetu` iz `Top 10
najboljih uvijeka na svijetu` (proba B), koji prag 3 propušta. Ugrađeno:
`hallucination.py`, `MIN_RUN_FOR_FLAG = 2`.

Pravilo ne hvata izmišljen tekst sa razvučenim vremenima (Radionica draft 1: prvi prolaz
kroz glosar preko džingla, 2,34 s po reči), a dužina reči nije upotrebljiv
drugi signal — uredan govor na uzorku ima reči i od 5,3 s.

Ograničenje uzorka: svi uredni snimci su francuski i prošli su bez
`hotwords` (glosar nema francuskih ključeva). Uredan srpski snimak sa
glosarom na disku ne postoji.

Izvor: `merenja/krug5/*.words.json`, `merenja/krug5/zero_runs.log`
(lokalno); skripta `scripts/krug5_nulte_reci.py`.

## 2026-10-07 — Profil `digital`: donja granica reda, trajanje, CPS

Dvanaest ASR poslova na disku (Radionica draft 1, Letnja škola, Pita, Video-4..13; srpski
izvor, `sentences.sr.json` + `words.sr.json`), profil `digital` (20 zn., jedan
red, `min_chars_per_line` 10, 20 CPS, 0,50–4,0 s). Prelamanje je
determinističko, pa je jedan prolaz merenje. „U sintagmi" = rez sa
`break_pen` ≥ 200 (pridev 200; predlog, enklitika, glosar 300).

| `PENALTY_BELOW_MIN_CHARS` | cue-ova | < 10 zn. (od toga bez izuzetka) | rezova u sintagmi | prirodno < 0,83 / 0,70 / 0,50 s | > 20 CPS |
|---|---|---|---|---|---|
| 0 (bez granice) | 294 | 32 (20) | 28 od 232 (12,1%) | 76 / 49 / 32 | 55 (18,7%) |
| 15 | 306 | 36 (19) | 22 od 244 (9,0%) | 87 / 54 / 33 | 56 (18,3%) |
| 30 | 306 | 33 (15) | 22 od 244 (9,0%) | 85 / 52 / 29 | 52 (17,0%) |
| 60 | 304 | 28 (9) | 24 od 242 (9,9%) | 84 / 51 / 30 | 53 (17,4%) |

`web` na istim poslovima: 106 cue-ova, 6 preko 20 CPS (5,7%), 0 rezova u
sintagmi od 98.

CPS: razlika `digital`/`web` je trostruka, i nije u govoru nego u
`apply_timing`. Pre njega `digital` probija 20 CPS u 21 cue-u, posle u 52.
Cue-ovi `digital` su kratki (medijana 1,07 s, `web` 3,29 s) i u 222 od 294
susedna para nema prirodnog razmaka od `min_gap` 0,083 s, pa se prethodni
cue skraćuje za razmak — kod cue-a od jedne sekunde to je oko 8% trajanja.
Sa `min_gap` 0 `digital` probija u 26 cue-ova; `min_duration` 0,833 umesto
0,50 ne menja ništa (53).

Trajanje: 29 od 306 cue-ova (9,5%) je prirodno kraće od 0,50 s, a 24 ostaju
kraća i posle `apply_timing` (nema mesta ni napred ni nazad).

Izvor: `merenja/krug6/digital.log`, `merenja/krug6/digital_cues.json` (po
cue-u, tekst pravog govora — lokalno); skripta `scripts/krug6_digital.py`.

Odluke (2026-10-07):
- `PENALTY_BELOW_MIN_CHARS = 30`. Vrednost 60 kupuje šest kratkih redova manje
  za dva reza kroz sintagmu više, a u jednorednom formatu je taj rez teža
  greška: `u` i `kuću` se pojave u različitim trenucima. 30 je najbolje po tri
  merila od četiri.
- `min_gap` ostaje 0,083 s: razmak je signal da se natpis promenio, i u
  formatu koji se menja svake sekunde važniji je, ne manje važan.
- `max_cps` ostaje 20, iako je verovatno pogrešna mera za natpis od 20 znakova
  koji se hvata pogledom. Drugi broj nema izmerenu osnovu, a birati prag po
  broju prekoračenja je kružno. Odlučuje gledanje stvarnog posla uz sliku.
- Cue-ovi kraći od `min_duration` i posle `apply_timing` se prijavljuju u
  dnevniku posla i na stranici posla, sa rednim brojem i vremenom
  (`linebreak.too_short`) — kao prekoračenja profila.

## 2026-10-07 — Otvoreno pitanje: `ß` u VELIKIM slovima

`fixedcues.apply_case` pretvara `ß` u `ẞ` (U+1E9E), ne u `SS`.

- Za `ẞ`: broj znakova ostaje isti. Prekoračenja se broje PRE promene
  veličine slova, pa bi `SS` mogao tiho da probije dužinu reda koju je
  profil već proverio.
- Protiv `ẞ`: stariji fontovi to slovo nemaju i plejer ume da prikaže prazan
  kvadrat; uobičajeni nemački pravopis i dalje piše `SS`.

Nije rešavano: nemački nije radni jezik alata i nema ga ni u jednom poslu.
Kad se pojavi nemački materijal, odlučiti ovde — ako `SS`, prekoračenja treba
brojati POSLE promene veličine slova.

## 2026-10-07 — Krug 7: ravnomerna podela rečenice u jednorednom profilu

Profil se u ovom merenju zove **`digital`** (20 zn., jedan red, min 10 zn.,
0,50–4 s); u krugu 8 isti profil postaje `web`. Trinaest ASR poslova na disku
(dvanaest iz kruga 6 i Video-4_16x9, nov posao nad istim snimkom kao
Video-4_draft-1). Determinističko — jedan prolaz.

Kolone: cue-ova; cue-ovi ispod 10 znakova u rečenici sa više cue-ova (sve su
to rečenice od bar 10 znakova); rečenice čiji je poslednji cue jedna reč;
rezovi sa `break_pen` ≥ 200; cue-ovi kraći od `min_duration` i posle
`apply_timing`; cue-ovi preko 20 CPS.

| stanje | cue-ova | < 10 zn. | jedna reč na kraju | u sintagmi | < 0,50 s | > 20 CPS |
|---|---|---|---|---|---|---|
| pre (krug 6) | 333 | 36 | 21 | 26 od 267 | 24 | 54 |
| suženi izuzetak | 332 | 29 | 12 | 26 od 266 | 23 | 58 |
| + udeo, težina 2 | 331 | 29 | 12 | 26 od 265 | 23 | 59 |
| + udeo, težina 4 | 328 | 25 | 11 | 28 od 262 | 21 | 56 |
| + udeo, težina 8 | 326 | 24 | 11 | 30 od 260 | 20 | 56 |

Suženje izuzetka (kazne nema samo kad je CELA rečenica kraća od granice)
menja ishod: jednorečnih krajeva 21 → 12, kratkih cue-ova 36 → 29.

Udeo (`PENALTY_CUE_SHARE_DEVIATION`, skorovanje iz `split_by_shares`):
svaka veća težina skida po nekoliko kratkih cue-ova i dodaje rezove kroz
sintagmu. Na 4 se pojavljuju rezovi iza predloga u sredini rečenice (tri
nova, jedan od njih na Letnjoj školi), na 8 još dva. Težina 2 je jedina bez novog
reza iza predloga u odnosu na stanje bez udela (jedan nov rez iza `je`).

Izvor: `merenja/krug7/*.log` (lokalno, tekst pravog govora); skripta
`scripts/krug7_ravnomerno.py`.

**Ispravka formule (isti dan).** Prvi udeo je cilj računao kao
`ceil(dužina / max_chars_per_line)` delova, pa je broj delova postao cilj i
treći deo je bio kažnjen i kad je bolji. Sada se odstupanje meri prema
proseku delova koje je prelamanje stvarno izabralo (`_layout_even`: za svaki
broj delova k prosek `(dužina - (k - 1)) / k`); broj delova i dalje bira
`PENALTY_PER_CUE`. Gornja tabela za težine 2/4/8 važi za staru formulu.

| težina (nova formula) | cue-ova | < 10 zn. | jedna reč na kraju | u sintagmi | < 0,50 s | > 20 CPS | menja rečenica (od 59 sa više cue-ova) |
|---|---|---|---|---|---|---|---|
| 0 (isključeno) | 332 | 29 | 12 | 26 od 266 | 23 | 58 | — |
| 2 | 332 | 29 | 12 | 26 od 266 | 23 | 59 | 8 |
| 4 | 330 | 25 | 12 | 26 od 264 | 21 | 58 | 13 |
| 8 | 329 | 23 | 11 | 27 od 263 | 20 | 57 | 17 |

Rečenica od 40 znakova sa Letnje škole (molba sa spiskom stvari) ni na jednoj težini
ne izlazi u tri jednaka dela bez reza iza predloga. Uzrok nije udeo nego
`break_pen`: rez ispred enklitike `ćeš` košta 350, isto koliko i rez iza
predloga `sa` (`PENALTY_SPLIT_COMPOUND_VERB` = `PENALTY_SPLIT_PREPOSITION`
= 300, plus 50 za rez van granice klauze). Jeftina su samo dva mesta, pa
ravnoteža nema između čega da bira.

Izvor: `merenja/krug7/4_udeo_izabrani.log`, `5_razlike.log` (lokalno).

**Posle dodavanja negacije (isti dan).** `ne` je dodato u `no_line_end` za
srpski i francuski (`SR_NEGATION`, `FR_NEGATION`; ista kazna kao predlog),
da se udeo ne meri nad tabelom koja propušta negaciju. Isti skup, 13 poslova:

| težina | cue-ova | < 10 zn. | jedna reč na kraju | u sintagmi | < 0,50 s | > 20 CPS | menja rečenica (od 59) |
|---|---|---|---|---|---|---|---|
| 0 | 331 | 29 | 11 | 26 od 265 | 23 | 59 | — |
| 2 | 331 | 29 | 11 | 26 od 265 | 23 | 60 | 7 |

Rez iza negacije (`mi ne / idemo`) je nestao i na težini 0, pa ga udeo
više ne pravi. Od sedam promena na težini 2: tri bolje (pozdrav na početku
Letnje škole; rez pomeren tako da imenica ostane uz svoj predlog; rez ispred
predloške grupe umesto usred nje), dve gore (rez između glagola i njegovog
objekta; rez ispred enklitike `je`), dve neutralne. (Primeri u ovom unosu su
prepričani — rečenice iz snimaka ne stoje u repozitorijumu.)

Rez ispred enklitike `je` nije posledica nepotpune tabele: `je` je u srpskoj
tabeli i u `no_line_start` (enklitika) i u `no_line_end` (pomoćni glagol se
ne odvaja od participa, `je / došao`), pa i `… / je` i `je / …` koštaju 350
i udeo bira između jednako skupih rezova.

Izvor: `merenja/krug7/6_negacija.log`, `7_razlike_negacija.log` (lokalno).

**Odluka:** udeo ostaje isključen (`PENALTY_CUE_SHARE_DEVIATION = 0`), sa
mehanizmom u kodu za ponovno merenje. Cena: prvi cue Letnje škole ostaje
jednorečni pozdrav, lošiji nego na težini 2 — jedan primer ne
nadjačava skup (ista logika kao kod odbačenog pravila o registru).

## Otvoreno pitanje: `je` kao pomoćni glagol ispred reči koja nije particip

`je` je u srpskoj tabeli i enklitika (ne počinje red) i pomoćni glagol (ne
završava red), pa su oba reza oko njega skupa i reč ostaje u sredini reda.
To je namerno: `on / je došao` razdvaja enklitiku, `on je / došao` particip,
ispravno je samo `on je došao` u celini.

Pravilo o pomoćnom glagolu, međutim, okida i kad sledeća reč nije particip:
u `Na ovom mestu je zato važno da` rez `je / zato` ne razdvaja ništa, a
košta isto kao `je / došao`. Tačnije pravilo bi tražilo morfološku proveru
sledeće reči, što je nesrazmerno poslu. Razmotreno i svesno ostavljeno
(krug 7).

## 2026-10-07 — Krug 8: preimenovanje profila i migracija poslova

Od ovog unosa: `tv` = dosadašnji `web` (2 × 42), `web` = dosadašnji
`digital` (jednoredni), `digital` = jednoredni + VELIKA slova, bez
interpunkcije. Merenja iz krugova 6 i 7 nose stara imena.

Migracija `job.json` 1 -> 2 (`subtitis/migrate.py`) na 16 poslova na disku:
14 sa promenom, 2 bez imena profila (nijedno prelamanje), 0 preskočenih.
Promene po imenu (broj zapisa u istoriji; „predlog" = predlog blažeg profila
kod SRT posla):

| staro | novo | zapisa | poslova |
|---|---|---|---|
| `netflix` | `tv` | 9 | 9 (Radionica draft 1, Letnja škola, Pita, Video-4_draft, Video-5..9) |
| `web` | `tv` | 2 | 2 (Video-10, Video-4_16x9) |
| `web` (predlog) | `tv` | 7 | 3 (Otvorena vrata 26, sva tri) |
| `digital` | `web` | 6 | 3 (Letnja škola 4, Video-10, Video-4_16x9) |

Ime `tv` nije zatečeno ni u jednom poslu (stari profil `tv` je nestao u krugu
6). `subtitis.toml` nije imao `default_profile`; dobio je samo oznaku
`profiles_version = 2`.

**Propust u izolaciji testova.** Migracija nije pokrenuta komandom nego
testom: `test_desktop.py::test_pravi_server_odgovara_identitetom` diže pravi
dashboard (`lifespan`) bez privremenog `JOBS_DIR`, a `lifespan` od kruga 8
migrira poslove. Ishod je isti kao što bi dala komanda (svaka promena je u
dnevniku posla, sa razlogom), ali rezervna kopija pre migracije nije
napravljena na vreme — stanje pre se vidi samo iz tih redova dnevnika.
Zatvoreno: `tests/conftest.py` sada svakom testu daje privremeni `JOBS_DIR` i
`subtitis.toml` (autouse), a provereno je da ceo skup testova više ne menja
nijedan `job.json` na disku.

## Zamka: test nad stvarnim podacima

Najvredniji nalaz kruga 8. Migraciju poslova nije pokrenula komanda nego
test koji je samo podigao dashboard. Prošlo je dobro samo zato što je
migracija idempotentna i piše u dnevnik posla. Da je imala grešku, četrnaest
poslova bi bilo izgubljeno bez kopije.

**Pravilo: nijedan test ne sme da dodirne stvarni `jobs/` ni
`subtitis.toml`.** Sprovodi ga autouse fixture u `tests/conftest.py`; test
koji traži svoj folder postavlja ga sam, preko te donje granice. Svaki nov
kod koji radi pri startu (`lifespan`, `Worker.run`) mora se zamisliti kao
nešto što će test pokrenuti. Za Prilog A sveobuhvatnog prompta (v1.4): ovu
klasu greške će ponoviti svako ko ovo gradi.

## Sledeća stavka: kazna za cue kraći od najkraćeg trajanja

Letnja škola kroz `digital` (krug 8) počinje jednorečnim cue-om od 0,117 s — tri frejma,
natpis trepne i nestane. Prijavljen je kao prekratak, ali prijava ne pomaže
ako prelamanje ne ume da ga izbegne.

U krugu 7 je to pokušano udelom znakova, koji meri pogrešnu osovinu: taj pozdrav
nije loš zato što ima malo znakova nego zato što traje kratko, a trajanje ne
ulazi u izbor reza. Postoje kazna za kratak cue po znakovima i kazna za
brzinu čitanja, ali nijedna ne gleda da li cue stigne da se pročita.
Kazna za cue kraći od `min_duration` profila je verovatno ono što je
trebalo uvesti umesto udela. Radi se posle interfejsa (krug 8, celina 3),
da se prelamanje i prikaz ne mešaju u istom krugu.

**Ponovilo se u krugu 11, u drugom obliku.** Granica iz kruga 8 pokrivala je
`jobs/` i `subtitis.toml`, ali ne i glosar: nov test koji je proveravao upis
glosara prepisao je pravi `glossary/default.json`. Uhvaćeno odmah —
`save_glossary` pravi `.bak`, koji je bio identičan verziji u gitu, i vraćen
je. Granica sada pokriva SVAKI folder u koji alat piše: `jobs/`,
`subtitis.toml`, `glossary/`, `profiles/` (poslednja dva kao kopija, da čitanje
vidi isto). Pravilo se ne zadovoljava nabrajanjem fajlova koji su do sada
stradali, nego spiskom svega što alat ume da upiše — `config.py`, putanje
na vrhu.

## 2026-10-07 — Krug 11: `PermissionError` pri zameni `job.json`

Na Windowsu `os.replace` pada dok je ciljni fajl otvoren bilo gde. Projekat
je pao, a zavisni zadaci su preskočeni, iako sa zvukom i modelom nije bilo
ništa. Popravka: `config.replace_file` ponavlja zamenu posle 20, 40, 80, 160 i
320 ms; svaki upis preko postojećeg fajla (JSON, SRT, podešavanja, glosar) ide
kroz `write_text_atomic`. Stranica projekta je čitala `job.json` tri puta po
prikazu; sada jednom (keš vezan za vreme izmene). Lista projekata, suprotno
prvoj pretpostavci, `job.json` nije otvarala — čita bazu i veličine foldera —
ali je zamenom celog bloka na 3 s gubila brisanje (forma zamenjena usred
potvrde) i vraćala skrol na vrh; sada miruje kad ništa ne radi.

## 2026-10-07 — Krug 14: prelamanje na izlazak iz polja, trajanje

Izmenjen tekst rečenice se po izlasku iz polja snima (kao automatsko
snimanje: kopija najviše jednom na 5 minuta) i ceo titl se prelama kroz
`/rebuild`. Mereno na serveru, `/rebuild` od zahteva do odgovora, tri
prolaza po profilu:

| prepis | rečenica | cue-ova | medijana | najduže | odgovor |
|---|---|---|---|---|---|
| Otvorena vrata 26 (SRT, 25 min) | 195 | — | 114 ms | 274 ms | 65 kB |
| veštački, 60 min (`tv`) | 1683 | 1870 | 0,89 s | 0,94 s | 426 kB |
| veštački, 60 min (`web`) | 1683 | 3927 | 0,49 s | 0,52 s | 696 kB |

Veštački prepis: Letnja škola ponovljena do sat vremena sa pomerenim vremenima
(stvarnog srpskog prepisa te dužine na disku nema). Zahtev ide u pozadini i
ne blokira kucanje; vreme koje pregledač troši na zamenu ~1700 delova
strane nije mereno (nema pregledača u ovom okruženju). Skripta:
`scratchpad` sesije, nije u repozitorijumu — merenje se ponavlja sa
`/rebuild` nad kopijom projekta.

## 2026-10-12 — Razmak između reči u `words.json` je skoro uvek nula

Whisper (faster-whisper, `word_timestamps`, bez VAD-a) ne ostavlja tišinu
između reči: kraj jedne reči je početak sledeće, a tišina se upija u trajanje
susedne reči (`u` od 0,72 s, `kada` od 0,96 s). Izmereno na 15 ASR poslova
(`scripts/krug19_pauze.py`), pauza = `s` sledeće reči − `e` prethodne,
preskačući tokene interpunkcije:

| granica | ukupno | ≥ 0,05 s | ≥ 0,12 | ≥ 0,20 | ≥ 0,30 | ≥ 0,45 | tačno 0 |
|---|---|---|---|---|---|---|---|
| unutar rečenice, bez interpunkcije | 830 | 5 | 5 | 4 | 1 | 1 | **825** |
| unutar rečenice, uz zarez | 119 | 99 | 55 | 15 | 8 | 5 | 1 |
| između rečenica | 55 | 43 | 36 | 24 | 21 | 16 | 11 |

Razmak postoji samo uz interpunkciju — gde je rez ionako besplatan. Na
granici bez znaka, gde bi pauza jedina nosila informaciju, **825 od 830
razmaka je tačno 0**. Svaki mehanizam koji čita `words[k+1].s − words[k].e`
radi nad nulama:

- `segment._split_long` (`HARD_SPLIT_MIN_PAUSE` 0,7 s) — pri sečenju
  predugačke rečenice bez zareza pauzu praktično nikad ne vidi; to i komentar
  u kodu beleži („u govoru bez interpunkcije su sve pauze jednake"). Nije
  menjano u krugu 19.
- `fixedcues.SRT_MAX_GAP` (izmereno 2026-09-28, isključeno) **nije** pogođen:
  meren je na razmacima između cue-ova uvezenog SRT-a koji je napravio čovek,
  ne na razmacima Whisper-ovih reči. Njegov zaključak važi — ali za granicu
  rečenice u tuđem SRT-u, ne za pauzu u govoru.

Pauze postoje u zvuku: Silero VAD (isti koji `asr.py` koristi za probu) na
45 s snimka nalazi 17 tišina od 0,16 do 1,82 s, za 0,3 s obrade; leže
unutar Whisper-ovog raspona susedne reči, ne između reči.

## 2026-10-12 — Krug 19: pauza kao signal za prelom — kriterijum (zapisan pre merenja)

Izvor pauza: Silero VAD nad `source.wav`, jednom po poslu, u
`jobs/<id>/pauses.json` (preživljava brisanje zvuka). Pauza se pripisuje
granici dve reči koja leži u tišini ili joj je najbliža.

Mehanizmi: (1) kazna za cue koji sadrži pauzu iznad praga, srazmerna
najvećoj takvoj pauzi; (2) popust na `PENALTY_BREAK_NOT_AT_CLAUSE` za rez na
pauzi; (3) kazna za rez bez pauze — samo meren. Mreža: prag 0,12 / 0,20 /
0,30 / 0,45 s; težina (1) 50 / 100 / 200 po sekundi iznad praga, najviše 50;
popust (2) 0 / 25 / 50. Mereno na 15 ASR poslova, profili `tv`, `web`,
`digital`.

Kriterijum prihvatanja, unapred:

1. U svakom profilu, u odnosu na stanje bez pauza, **ne sme da poraste**:
   broj rezova iza predloga, broj rezova ispred enklitike, broj cue-ova ispod
   donje granice znakova (`min_chars_per_line`, a gde je nema `MIN_CUE_CHARS`;
   cela rečenica u jednom cue-u se ne broji), broj cue-ova preko `max_cps` i
   broj redova preko `max_chars_per_line`.
2. Broj cue-ova koji sadrže pauzu iznad praga **mora da padne** u svakom
   profilu.
3. Od konfiguracija koje ispunjavaju 1 i 2 bira se ona sa najvećim padom iz
   tačke 2; pri izjednačenju manje težine. Tek tada se čitaju sve rečenice
   koje su se promenile i svaka se ocenjuje kao bolja, gora ili neutralna.
   **Ako je odnos bolje : gore lošiji od 3 : 1, mehanizam se ne uvodi** i to
   se javlja — ne podešava se dalje dok brojka ne bude lepa.
4. Mehanizam (3) se uvodi samo ako uz (1)+(2) dodatno spusti broj cue-ova sa
   pauzom za bar 10 % i ne pokvari nijednu meru iz tačke 1.

## 2026-10-12 — Krug 19: rezultat — pauza se ne uvodi

Mereno `scripts/krug19_pauze.py meri` na 15 ASR poslova; 13 imaju zvuk, 2 su
ga izgubili po roku i rade bez pauza (to je i slučaj „nema ni `pauses.json`
ni zvuka"). Tišine su računate u memoriji — u poslove nije upisan nijedan
`pauses.json`. `web` i `digital` se prelamaju isto, pa su im brojke iste.

Stanje bez pauza i izabrana tačka (prag 0,30 s, samo popust 50); u zagradi
je razlika:

| profil | cue-ova | sa pauzom ≥ 0,30 | iza predloga | ispred enklitike | ispod granice | preko CPS | red preko |
|---|---|---|---|---|---|---|---|
| `tv` bez | 126 | 50 | 0 | 0 | 2 | 3 | 0 |
| `tv` sa | 131 | 49 (−1) | 0 | 0 | 2 | 3 | 0 |
| `web` bez | 371 | 47 | 3 | 2 | 33 | 52 | 0 |
| `web` sa | 372 | 42 (−5) | 3 | 2 | 33 | 52 | 0 |
| `digital` bez | 371 | 47 | 3 | 2 | 33 | 52 | 0 |
| `digital` sa | 372 | 42 (−5) | 3 | 2 | 33 | 52 | 0 |

**Kazna za pauzu u cue-u** (mehanizam 1, zamišljen kao glavni) ni u jednoj
od 36 tačaka ne prolazi kriterijum u sva tri profila: smanjuje broj cue-ova
sa pauzom najviše (prag 0,12, težina 200 uz popust 50: `tv` 71 -> 47), ali cepa natpise
na kraće, pa raste prekoračenje CPS-a (`tv` 3 -> 4–5) ili broj kratkih
cue-ova (`web` 33 -> 34–35). Kriterijum prolazi samo **popust** za rez na
pauzi (mehanizam 2), a najveći pad (ukupno −11) daje prag 0,30 uz popust 50.
**Kazna za rez bez pauze** (mehanizam 3) uz to dodaje jedan cue manje sa
pauzom i diže CPS u `tv` (3 -> 4) i kratke cue-ove u `web` (33 -> 34–37) —
ne prolazi.

Čitanje izabrane tačke: promenjeno 10 rečenica (6 u `tv`, 4 u `web`, iste u
`digital`). `tv`: 4 bolje (nestaju rezovi između glagola i objekta, kroz
imeničku sintagmu i između prideva i imenice preko granice natpisa), 1 gora
(rez kroz priloški izraz), 1 neutralna. `web`: 0 boljih, 3 gore (zamenica
otcepljena od svog glagola preko zareza, `kao / i` presečeno, jedna reč sama
u natpisu), 1 neutralna. Ukupno **4 bolje : 4 gore** — ispod 3 : 1, pa se
po kriterijumu **ne uvodi**, i ne podešava dalje.

Šta ostaje u kodu: `pauses.py` (tišine iz zvuka, `pauses.json` pri
prepoznavanju govora, jer posle brisanja zvuka tišina nema odakle) i
mehanizmi u `linebreak.py` sa težinama 0 (`pauses_enabled()` je False, pa
prelamanje zvuk ni ne čita). Zapažanje za sledeće merenje, ne zaključak:
u dvorednom `tv` je odnos bio 4 : 1, u jednorednom 0 : 3 — u jednom redu
od 20 znakova pauza retko pada tamo gde ima mesta za rez.

## 2026-10-12 — Krug 20: strana veznika u srpskom — kriterijum (zapisan pre merenja)

Povod: u krugu 19 je pauza u dvorednom `tv` bila 4 : 1, a u jednorednom
`web`/`digital` 0 : 3 — a loši prelomi zbog kojih je tražena (rez iza `da` u
složenom predikatu, rez iza `i` pred nastavkom misli) su iz `digital` i oba su
rez IZA veznika, koji
srpski naplaćuje samo 10 (`conjunction_side = "after"`, iz prve
specifikacije, nikad mereno). Engleski je na `before` prebačen merenjem
(2026-09-25).

Varijante, na 15 srpskih ASR poslova, profili `tv`, `web`, `digital`:

1. `after` — sadašnje stanje;
2. `before` — kao engleski i francuski;
3. `before`, a `da` izbačeno iz veznika i stavljeno u `no_line_end`
   (u `moramo da krenemo` je deo predikata, ponaša se kao enklitika);
4. kao 3, a uz `da` isto i `pa` i `te`.

Kriterijum prihvatanja, unapred — isti kao u krugu 19:

1. U svakom profilu, u odnosu na `after`, **ne sme da poraste**: broj rezova
   iza predloga, broj rezova ispred enklitike, broj cue-ova ispod donje
   granice znakova (cela rečenica u jednom cue-u se ne broji), broj cue-ova
   preko `max_cps` i broj redova preko `max_chars_per_line`. Rezovi iza i
   ispred veznika se prikazuju bez suda.
2. Promenjene rečenice se čitaju **po profilu odvojeno** (`tv` posebno,
   `web`/`digital` posebno — prelamaju se isto) i svaka se ocenjuje kao
   bolja, gora ili neutralna. Varijanta prolazi samo ako je **u obe grupe**
   odnos bolje : gore bar **3 : 1**. Broj promenjenih rezova ne meri
   kvalitet.
3. Od varijanti koje prođu bira se ona sa najboljim odnosom; pri
   izjednačenju ona koja menja manje. Ako ne prođe nijedna, ostaje `after` i
   merenje se upisuje — bez doterivanja.

## 2026-10-12 — Krug 20: rezultat — ostaje `after`

Mereno `scripts/krug20_veznici.py` na 15 srpskih ASR poslova (70 rečenica;
četiri posla su ponovljeni snimci istih izjava, pa su neke rečenice dvaput).
`web` i `digital` se prelamaju isto — brojke i promene su im identične.

Gde rez pada danas (`after`): u `tv` 33 od 122 reza su iza veznika, 20
ispred; u `web`/`digital` 41 od 301 iza, 40 ispred. Najčešći su `da` (iza 14
/ ispred 10 u `tv`; 12 / 20 u `web`) i `i` (14 / 2; 22 / 10). `pa` i `te`
skoro se ne javljaju (0–1), pa je varijanta 4 identična varijanti 3.

| profil | varijanta | cue-ova | iza veznika | ispred veznika | iza predloga | ispred enklitike | ispod granice | preko CPS | red preko | promenjenih reč. |
|---|---|---|---|---|---|---|---|---|---|---|
| `tv` | after | 126 | 33 | 20 | 0 | 0 | 2 | 3 | 0 | — |
| `tv` | before | 132 | 5 | 43 | 0 | 0 | **3** | **4** | 0 | 24 |
| `tv` | before, `da` van veznika | 132 | 5 | 38 | 0 | 0 | **3** | **4** | 0 | 22 |
| `web`/`digital` | after | 371 | 41 | 40 | 3 | 2 | 33 | 52 | 0 | — |
| `web`/`digital` | before | 370 | 9 | 62 | **4** | 2 | **36** | 50 | 0 | 24 |
| `web`/`digital` | before, `da` van veznika | 367 | 12 | 57 | **4** | **4** | **36** | 50 | 0 | 23 |

Svaka varijanta podiže bar jednu meru koja ne sme da poraste, u svakom
profilu. Čitanje, po profilu odvojeno (`before`; u zagradi bez ponovljenih
snimaka):

- `tv`: 16 bolje, 1 gora, 7 neutralnih (14 : 1). Rez ide ispred `da` i `i`
  (oblika `moramo / da krenemo`), a natpisi se zatvaraju na zarezu.
  Oba pogoršanja mera su granična: cue od 19 znakova (granica 20) i cue od
  20,5 CPS (granica 20), prvi u rečenici ocenjenoj kao bolja. Varijanta sa
  `da` van veznika čita se još čistije: zavisni deo rečenice staje ceo u natpis do zareza.
- `web`/`digital`: 13 bolje, 8 gore, 3 neutralne (11 : 6). Pored dobrih
  (oblika `moramo || da krenemo`, `sunce || i kiša`) javljaju se rez iza
  predloga, presečeni parovi `kao / i` i `nego / i`, zamenica otcepljena od
  glagola; sa `da` van veznika i rez ispred enklitike (oblika `koje || ćemo`).

Ni jedna varijanta ne ispunjava kriterijum (mere i 3 : 1 u obe grupe), pa
**ostaje `after`**, bez doterivanja. Zapažanje za sledeći put, ne zaključak:
u dvorednom `tv` strana veznika je jasno bolja ispred, a smetaju samo dve
granične mere; u jednorednom profilu od 20 znakova rez ispred veznika
često nema gde da stane — drugi primer iz povoda (veznik i nastavak, 23
znaka) ne može da stane u jedan red ni u jednoj varijanti.

## Obrazac: ono što pomaže u dvorednom prelomu šteti u jednorednom (2026-10-12)

Dva kruga zaredom isti nalaz, na istim poslovima:

| promena | `tv` (42 zn., 2 reda) | `web`/`digital` (20 zn., 1 red) |
|---|---|---|
| pauza iz zvuka kao jeftin rez (krug 19) | 4 : 1 | 0 : 3 |
| rez ispred veznika (krug 20) | 16 : 1 | 11 : 6 |

To nije slučajnost nego svojstvo oblika: u dvorednom titlu rez pada otprilike
jednom po rečenici i ima izbora gde; u redu od 20 znakova pada na svake dve
do tri reči i izbora nema — pravilo koje pomera rez nema kuda da ga pomeri
osim na lošije mesto. Posledica za merenja: odnos bolje : gore se uvek čita
po obliku odvojeno, a pravilo prelamanja se sme vezati za oblik.

I zapis o kriterijumu, treći put isti: u krugu 20 je `tv` pao na dve mere
koje su porasle za jedan — cue od 19 znakova pri granici 20 (u rečenici
ocenjenoj kao bolja) i 20,5 CPS pri granici 20 — uz čitanje 16 : 1.
„Nijedna mera ne sme da poraste" je bilo prestrogo: kriterijum se bira pre
merenja, ali se proverava posle njega, i popravka se ne odbacuje zato što
nije dostigla proizvoljan cilj. Granično pogoršanje koje je viđeno i
pročitano se zapisuje uz konstantu, ne obara rezultat.

## 2026-10-12 — Krug 21: da li je 20 znakova pretesno (merenje, bez izmene)

`scripts/krug21_granica.py` na 15 srpskih ASR poslova (70 rečenica); profil
`web` sa promenjenom širinom, donja granica srazmerna (10 na 20), ostalo kao
`web` (20 CPS, 0,5–4 s). Strana veznika je svuda `after`.

| granica | donja | cue-ova | rečenica u 1 redu | u 1 cue-u | iza predloga | ispred enklitike | iza / ispred veznika | ispod donje | preko CPS | trajanje prosek / najduže |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 10 | 371 | 7 (10 %) | 7 | 3 | 2 | 41 / 40 | 33 | 52 | 1,17 / 10,74 s |
| 24 | 12 | 300 | 12 (17 %) | 12 | 0 | 0 | 30 / 32 | 18 | 32 | 1,46 / 10,72 s |
| 28 | 14 | 261 | 14 (20 %) | 14 | 0 | 0 | 33 / 27 | 12 | 18 | 1,71 / 11,08 s |
| 32 | 16 | 225 | 17 (24 %) | 17 | 0 | 0 | 26 / 21 | 3 | 16 | 1,99 / 11,08 s |
| 28 × 2 reda | 14 | 183 | 14 (20 %) | 21 | 0 | 0 | 42 / 21 | 16 | 8 | 2,47 / 11,08 s |

Najduži cue (~11 s) je u svim varijantama jedna reč preko duge tišine — ne
može se preseći, nije posledica širine. Već na 24 znaka nestaju svi rezovi
iza predloga i ispred enklitike, a prekoračenja CPS-a padaju za trećinu; 28
ih prepolovi još jednom. Celih rečenica u jednom redu ima malo i na 32 (24 %)
— izgovorene rečenice su duge — pa širina ne uklanja prelom nego mu daje
izbor. Izbor vrednosti je odluka o izgledu i nije donet ovde.

## 2026-10-12 — Krug 21: uvedeno — `before` u dva reda, `web`/`digital` 28 znakova

Dve izmene, odobrene posle merenja iznad:

- `LanguageRules.conjunction_side_multiline` — srpski u profilu sa
  `max_lines >= 2` lomi ispred veznika; jednoredni ostaje iza. Veza je kroz
  broj redova, ne kroz ime profila.
- `web` i `digital`: 28 znakova po redu, donja granica 14 (bilo 20 / 10).

Stanje posle obe izmene zajedno, na 15 srpskih ASR poslova (`tv` nema donju
granicu, pa je tu „ispod donje" 0; po meri iz kruga 20, `MIN_CUE_CHARS`, je
2 -> 3):

| slučaj | cue-ova | rečenica u 1 redu | u 1 cue-u | iza predloga | ispred enklitike | iza / ispred veznika | ispod donje | preko CPS | trajanje prosek / najduže |
|---|---|---|---|---|---|---|---|---|---|
| `tv` pre (after) | 126 | 21 | 33 | 0 | 0 | 33 / 20 | — | 3 | 3,64 / 13,46 s |
| `tv` sada (before) | 132 | 21 | 31 | 0 | 0 | 5 / 43 | — | 4 | 3,47 / 13,46 s |
| `web` pre (20, after) | 371 | 7 | 7 | 3 | 2 | 41 / 40 | 33 | 52 | 1,17 / 10,74 s |
| `web` sada (28, after) | 261 | 14 | 14 | 0 | 0 | 33 / 27 | 12 | 18 | 1,71 / 11,08 s |
| 28 × 2, after (nije uvedeno) | 183 | 14 | 21 | 0 | 0 | 42 / 21 | 16 | 8 | 2,47 / 11,08 s |
| 28 × 2, before (nije uvedeno) | 183 | 14 | 22 | 0 | 0 | 6 / 49 | 13 | 7 | 2,47 / 11,08 s |

`web`/`digital` je glavna promena: rezovi iza predloga i ispred enklitike
nestaju, prekoračenja CPS-a padaju sa 52 na 18, kratki cue-ovi sa 33 na 12.
Dvoredni 28 × 2 sa `before` bi CPS spustio još (18 -> 7) i stavio više
rečenica u jedan natpis (14 -> 22), uz duže natpise (1,71 -> 2,47 s) — to je
drugi oblik natpisa, zaseban izbor, nije uveden.

## 2026-10-13 — Krug 22: pauza ponovo izmerena na 28 znakova — opet se ne uvodi

Povod: krug 19 je meren na `web`/`digital` od 20 znakova (donja 10); krug 21
ih je proširio na 28/14. Mreža i kriterijum kruga 19 bez izmene, prolaz i
čitanje po obliku odvojeno (pravilo posle kruga 20). `scripts/krug22_pauze.py`.

Skup poslova nije isti: od 15 poslova kruga 19 osam je u međuvremenu
izgubilo zvuk po roku (a nemaju `pauses.json`), pa nemaju pauza. Mereno je
11 poslova koji tišine imaju (7 iz kruga 19 + 4 nova), 51 rečenica.
`digital` se prelama kao `web` (provereno u skripti), meri se jednom.

Nova mera — udeo cue-ova sa pauzom iznad praga, bez pauza u prelomu, isti
poslovi:

| profil | cue-ova | ≥ 0,12 | ≥ 0,20 | ≥ 0,30 | ≥ 0,45 |
|---|---|---|---|---|---|
| `tv` | 102 | 66,7 % | 58,8 % | 47,1 % | 32,4 % |
| `web` 28/14 | 202 | 39,1 % | 35,6 % | 26,7 % | 16,3 % |
| `web` 20/10 (kontrola) | 294 | 25,5 % | 22,1 % | 15,3 % | 10,5 % |

Širina je udeo stvarno podigla: na pragu 0,30 s sa 15 na 27 % natpisa.

Mere (kriterijum 1 i 2, „ne sme da poraste" / „mora da padne"):

- `tv`: prolazi 42 od 48 tačaka, i kazna za pauzu u cue-u (mehanizam 1)
  — u krugu 19 nije prolazila nigde. Pada samo težina 200 (jedan cue
  ispod 20 znakova, uz popust 50 i jedan preko CPS-a). Najveći pad: prag
  0,12, težina 100, popust 50 — 68 -> 48 cue-ova sa pauzom (66,7 -> 41,7 %).
- `web`: prolazi 2 od 48 — prag 0,20 ili 0,30, težina 50, bez popusta —
  oba sa −2 cue-a (−1 procentni poen). Ostale tačke padaju: popust 50 daje
  dva reza iza predloga u svakoj tački; kazna od 100 i više diže kratke
  cue-ove. Najbliži promašaji obaraju se jednim cue-om ispod donje granice
  (prag 0,20 ili 0,30, težina 100 ili 200, bez popusta: −5 cue-ova sa
  pauzom, −2,6 poena).
- Mehanizam 3 (kazna za rez bez pauze) nigde ne dodaje 10 % pada — ne prolazi.

Čitanje (bolje : gore : neutralno; u zagradi bez ponovljenih snimaka):

| tačka | `tv` | `web`/`digital` |
|---|---|---|
| zajednička (0,20 / 50 / 0) — izbor po kriterijumu 3 | 6 : 1 : 3 (5 : 1 : 2) | 1 : 3 : 1 (1 : 2 : 1) |
| najveći pad po obliku (`tv` 0,12 / 100 / 50) | 6 : 11 : 7 | — |
| najbliži promašaj (`web` 0,30 / 100 / 0, `ispod` +1) | — | 1 : 5 : 2 (1 : 4 : 2) |

Ishod: **ne uvodi se.** Zajednička tačka pada na `web`; po obliku odvojeno
`tv` prolazi samo na blagoj tački (0,20 / 50), a tačka koju bira kriterijum 3
(najveći pad) čita se 6 : 11. `web` nije blizu ni po čitanju: najbliži
promašaj po merama čita se gore od tačke koja prolazi.

Gore je skoro svuda isti oblik: govornik zastaje POSLE veznika, pa pauza
vuče kraj natpisa iza `i`, `ali`, `a`, `pa`, `da` — tačno ono što je krug 21
uklonio u dvorednom prelomu. U `web` još: lista ili parentetički umetak
presečen tako da jedna reč ostane sama u natpisu, i stalni par razdvojen.
Bolje je razdvajanje paralelnih rečenica i uslovne rečenice sa posledicom u
jednom natpisu, i nestanak reza između reči i broja uz nju.

Zapažanje za sledeće merenje, ne zaključak: pauza na granici iza reči sa
koje se ne sme završiti natpis (veznik, predlog) nije signal za rez nego
oklevanje; tako filtrirana pauza nije merena.

## 2026-10-13 — Definicija: „natpis sadrži pauzu" (važi od kruga 23)

**Merilo rezultata:** natpis sadrži pauzu ako se tišina iz zvuka (Silero,
`pauses.json`) preklapa sa **rasponom govora natpisa** — od početka prve do
kraja poslednje reči — bar za prag. Raspon govora, ne raspon prikaza: kraj
prikaza `apply_timing` produžava u tišinu posle govora, a natpis koji stoji
posle izgovorenog nije greška preloma.

**Dopunska mera** (kako je mereno u krugovima 19 i 22): tišina se pripiše
najbližoj granici reči (`pauses.boundary_pauses`), pa se gleda da li je ta
granica unutar natpisa.

Zašto prva: (1) to gledalac vidi — natpis stoji, a govora nema; (2) ne
zavisi od pripisivanja tišine granici, a baš to pripisivanje mehanizmi
menjaju (`move` u krugu 23 premešta pauzu) — mera ne sme da bude deo onoga
što meri; (3) dopunska je pokazala napredak koji se na ekranu ne vidi (ispod).
Na snimku Video-4 draft 2 (profil `digital`): po merilu 9 od 15 stvarnih
natpisa, po dopunskoj 5.

## 2026-10-13 — Krug 23: pauza iza reči na kojoj natpis ne sme da se završi

`scripts/krug23_lazna_pauza.py`, isti 11 poslova i ista mreža kao krug 22.
`linebreak.PAUSE_AT_FUNCTION_WORD`: `keep` (do sada), `drop` (pauza se ne
računa ni u jednom mehanizmu), `move` (pripisuje se granici ispred te reči;
ispred enklitike se odbacuje). „Takva reč": `no_line_end` (predlozi,
enklitike, `ne`) + veznici iz `SR_CONJUNCTIONS` + `a`, ili enklitika iza
granice. Veznici NISU u `no_line_end` — strana veznika je posebno pravilo.

**Koliko ih ima** (pauza pripisana granici, pre mreže):

| prag | pauza na granici | lažnih | predlog | enklitika | `ne` | veznik | `a` | ispred enklitike |
|---|---|---|---|---|---|---|---|---|
| 0,12 | 117 | 52 (44 %) | 13 | 1 | 4 | 23 | 1 | 10 |
| 0,20 | 105 | 47 (45 %) | 12 | 1 | 4 | 21 | 1 | 8 |
| 0,30 | 81 | 28 (35 %) | 6 | 0 | 2 | 14 | 1 | 5 |
| 0,45 | 48 | 19 (40 %) | 3 | 0 | 2 | 10 | 1 | 3 |

Na 0,30 najčešće iza `i` (7) i `da` (4). Ima šta da se popravi.

**Glavni nalaz — pauza je u reči, ne između reči.** Od 100 tišina ≥ 0,30 s
unutar rečenice, 96 leži bar pola u trajanju neke reči (Whisper tišinu
upija u reč — krug 19: razmak nula na 825 od 830 granica). Rez je uvek na
granici reči, pa tišina posle svakog reza ostaje u jednom od dva natpisa.
Po merilu, nijedan mehanizam preloma je ne izbacuje sa ekrana: u `tv` broj
natpisa sa pauzom ne pada ni u jednoj od 144 tačke (raste sa brojem
natpisa; udeo pada najviše 5,2 poena), dok dopunska mera pada do −22 —
napredak kruga 22 u `tv` (68 -> 48) bio je napredak dopunske mere.

| oblik | način | prolazi brojke | najbolji pad po merilu | dopunska, najveći pad |
|---|---|---|---|---|
| `tv` | keep | 0 / 48 | nijedan (najmanje +1) | −22 |
| `tv` | drop | 0 / 48 | nijedan (najmanje +1) | −15 |
| `tv` | move | 0 / 48 | nijedan (najmanje +1) | −15 |
| `web` | keep | 3 / 48 | −2 (0,45 / 50 / 0) | −14 |
| `web` | drop | 5 / 48 | −2 (0,20 / 50 / 0) | −4 |
| `web` | move | 2 / 48 | −2 (0,45 / 50 / 0) | −4 |

**Čitanje** (bolje : gore : neutralno; u zagradi bez ponovljenih snimaka):

| tačka | `keep` | `drop` | `move` |
|---|---|---|---|
| `tv` 0,12 / 100 / 50 (krug 22: najveći pad) | 6 : 11 : 7 | 9 : 5 : 5 (8 : 4 : 3) | 9 : 5 : 7 (8 : 4 : 5) |
| `web`, tačka koju bira kriterijum | 0 : 1 : 1 | 1 : 3 : 1 (1 : 2 : 1) | 0 : 1 : 1 |

Mehanizam radi ono za šta je smišljen: krajeva natpisa na takvoj reči u
promenjenim rečenicama `tv` je 12 sa `keep`, 0 sa `drop`, 1 sa `move`. Gore
je posle toga druge vrste: subjekat odvojen od predikata preko granice
natpisa, zamenica odvojena od glagola, rez kroz zavisnu rečenicu. `move` i
`drop` se čitaju isto.

Ishod: **ne uvodi se** — merilo ne pada ni u jednom obliku, a čitanje `tv`
(9 : 5) ne dostiže 3 : 1. `PAUSE_AT_FUNCTION_WORD` ostaje `keep`, ostale
težine 0, kao posle kruga 19. Zapažanje, ne zaključak: dokle god reči upijaju
tišinu, ono što gledalac vidi menja samo vreme natpisa (kraj natpisa na
početku tišine, početak sledećeg na njenom kraju), ne prelom.

## Obrazac: mera koju mehanizam ne može da promeni nije kriterijum (2026-10-14)

Tišina leži unutar reči (krug 23: 96 od 100 tišina ≥ 0,30 s), a rez pada
između reči — nijedan prelom ne može da izbaci tišinu iz natpisa, po
konstrukciji. Krugovi 19, 22 i 23 su merili prelom protiv mere „natpis
sadrži pauzu", koju prelom ne može da promeni. Pre nego što se mera postavi,
proveri se da li mehanizam uopšte ima pristup onome što se meri.

Isto važi u krugu 24 i za „mrtvo vreme": kad se početak natpisa pomeri na
kraj tišine, prazan ekran poraste tačno onoliko koliko teksta nad tišinom
nestane (do 0,1 s po natpisu razlike, `SNAP_EDGE_SLACK`). To je ista
veličina gledana s druge strane, ne cena koja može da obori mehanizam.

## Obrazac: kriterijum koji obori popravku pet puta nije strog nego pogrešan (2026-10-14)

„Nijedna mera ne sme da poraste" je oborio popravku u krugovima 19, 20, 22
i 23 (dvaput), a čitanje nije bilo sporno ili je mera bila van dometa
mehanizma. Pravilo ima smisla samo za mere koje mehanizam dira. Mere koje
ne dira idu u izveštaj kao kontekst, ne kao uslov.

## 2026-10-14 — Krug 24: vreme natpisa po tišini

`linebreak.snap_to_silences`, `scripts/krug24_vreme.py`. Posle
`apply_timing`: tišina ≥ 0,30 s koja otvara natpis (počinje najviše 0,1 s
posle njegovog početka) pomera početak na svoj kraj; tišina koja ga zatvara
pomera kraj na svoj početak. Ivica se pomera cela ili nikako; odbija se ako
bi natpis pao ispod `min_duration`, prešao `max_cps`, ili ako bi prazan
ekran koji pravi to pomeranje bio kraći od minimuma.

Skup: 8 poslova sa zvukom (još 3 su izgubila `source.wav` po roku posle
kruga 23), tišine iz zvuka u memoriji. Prelom današnji (pauze isključene).
Oba minimuma praznine (0,20 i 0,40 s) daju ISTO: pomeren početak pravi
prazninu od bar praga tišine plus postojećeg razmaka (0,30 + 0,083), pa
minimum do 0,38 s ne može da odbije ništa.

| | `tv` pre | `tv` posle | `web` pre | `web` posle |
|---|---|---|---|---|
| natpisa sa pauzom (merilo kruga 23) | 41 / 72 (56,9 %) | 36 (50,0 %) | 53 / 139 (38,1 %) | 42 (30,2 %) |
| — pauza na početku / kraju / u sredini | 9 / 0 / 32 | 0 / 0 / 36 | 13 / 2 / 38 | 2 / 0 / 40 |
| pomeren početak / skraćen kraj | — | 9 / 0 | — | 11 / 2 |
| pomeraj prosek / najveći | — | 0,54 / 0,88 s | — | 0,60 / 0,88 s |
| odbijeno: trajanje / CPS / praznina | — | 0 / 0 / 0 | — | 0 / 2 / 0 |
| preko CPS | 1 | 1 | 5 | 5 |
| preklapanja | 0 | 0 | 0 | 0 |
| mrtvo vreme | 27,03 s | 31,89 s | 32,85 s | 40,60 s |
| tekst nad tišinom | 40,00 s | 35,43 s | 39,70 s | 32,31 s |

Whisper tišinu stavlja na POČETAK sledeće reči: tišina na kraju natpisa je
retka (0 i 2), na početku je 9 i 13 — i tu mehanizam skida sve što
ograničenja puštaju. Tišina u sredini natpisa (32 i 38) ostaje: ni prelom ni
vreme ivica do nje ne dopiru. Ograničenja drži; da li je prazan ekran za
vreme tišine bolji od teksta pre govora je odluka o izgledu — mehanizam je
u kodu isključen (`SNAP_MIN_BLANK = None`). Uključen, čita samo postojeći
`pauses.json`; bez njega se preskače i to piše u izveštaju posla.

## 2026-10-14 — Krug 24: kazna za zamenicu, filter pauze u dvorednom prelomu

Kazne za subjekat odvojen od predikata NEMA i ne može je biti bez sintakse:
subjekat je imenička grupa (oblika `cela ta priča / mora`), a alat ima
samo liste reči. Za ličnu zamenicu ispred glagola može: `PENALTY_SPLIT_PRONOUN`
za rez iza `ja ti on ona ono mi vi oni one`.

Sama, bez pauza, kazna 25 / 50 / 100 ne menja nijednu rečenicu ni u `tv` ni
u `web`. Uz tačku kruga 23 u `tv` (`PAUSE_MIN` 0,12, `PENALTY_PAUSE_IN_CUE`
100 do najviše 50, `PAUSE_BREAK_DISCOUNT` 50, `PENALTY_BREAK_WITHOUT_PAUSE`
0, `PAUSE_AT_FUNCTION_WORD` "discard") već 25 uklanja jedini natpis
završen zamenicom — ta rečenica se vraća na prelom bez pauza.

Čitanje na 8 poslova sa zvukom: **5 : 2 : 5**, bez ponovljenih snimaka
**5 : 1 : 3**. Preostalo „gore" (`nije / ni ideološki`) je treći tip, ne
jedan od dva — u krugu 23 pogrešno svrstan. Dva pogoršanja tipa subjekat /
predikat iz kruga 23 su u poslovima koji više nemaju zvuk, pa se ne mogu
izmeriti; ako se ponašaju isto, na skupu kruga 23 bi bilo 8 : 3 (bez
ponovljenih).

Ishod: **ne uvodi se.** Kao i u svim krugovima do sada, odluka je po broju
sa ponovljenim snimcima (5 : 2 = 2,5 : 1), a i procena na punom skupu kruga
23 je ispod 3 : 1. Bez ponovljenih bi prošlo (5 : 1) — to je pitanje kako
se broji, i nije rešavano ovde. `PENALTY_SPLIT_PRONOUN` ostaje 0, kao i
težine pauze.

Varijanta „pauza se premešta ispred veznika" je izbačena iz koda: čita se
isto kao odbacivanje (9 : 5), a ostavlja 1 kraj natpisa na vezniku prema 0.

## Obrazac: mera koja po konstrukciji raste kad mehanizam uspeva nije cena (2026-10-14)

Krug 24: kad se početak natpisa pomeri na kraj tišine, prazan ekran poraste
tačno onoliko koliko teksta nad tišinom nestane — isti broj s druge strane,
pa bi mehanizam po pravilu „mrtvo vreme ne sme da poraste" padao uvek.
Obrnut slučaj obrasca iz kruga 23 (mera koju mehanizam ne dira). Pre nego
što mera postane uslov, proveri se da li je nezavisna od mehanizma.

## Pravilo brojanja čitanja (od kruga 25)

**Ponovljeni snimak istog govora broji se jednom** — isti tekst
transkribovan dvaput nije dva nezavisna dokaza, ni za popravku ni za
pogoršanje. Daje se jedan merodavan broj; broj sa ponovljenima, ako se uopšte
navodi, ide u napomenu. Ponovljeni u ovom skupu: dva posla istog snimka
Video-1 draft 2, i Video-13 sa drugim snimkom istog govora.

Prevod ranijih brojeva: krug 24, filter + zamenica u `tv`: **5 : 1 : 3**
(sa ponovljenima bilo 5 : 2 : 5). Procena „8 : 3 na punom skupu kruga 23"
je već bila BEZ ponovljenih (8 : 4 : 3 iz kruga 23, minus popravljena
zamenica) — novo pravilo je ne menja, i ostaje ispod 3 : 1. Razlika 5 : 1
prema 8 : 3 nije u brojanju nego u skupu: dva pogoršanja tipa subjekat /
predikat su u poslovima koji su izgubili zvuk pre nego što je `pauses.json`
napravljen. Paket je uveden po izmerenom broju (5 : 1), odlukom.

## Veličina mernog skupa po krugu (2026-10-14)

Brojevi različitih krugova se porede samo na istom skupu.

| krug | poslova | od toga sa tišinama | napomena |
|---|---|---|---|
| 19 | 15 | 13 (iz zvuka) | 2 već bez zvuka |
| 20, 21 | 15 | — | tišine se ne koriste |
| 22, 23 | 11 | 11 (iz zvuka) | 8 poslova izgubilo zvuk po roku |
| 24 | 8 | 8 (iz zvuka) | još 3 izgubila zvuk |
| 25 | 8 | 8 (`pauses.json`) | `pauses.json` napravljen za svih 7 koji su još imali zvuk |

Od kruga 25 merenje čita samo `pauses.json` (bez zvuka), a svaki nov posao
ga dobija pri prepoznavanju — skup odsad raste. 11 starijih poslova ga
nikad neće imati.

## 2026-10-14 — Krug 25: rezerva pred govorom, spoj preloma i vremena, paket za `tv`

`scripts/krug25.py`, 8 poslova sa `pauses.json`. Uključeno: vreme po tišini
(`SNAP_MIN_BLANK` 0,40, `SNAP_LEAD` 0,15) i paket pauze za dvoredni prelom
(`PAUSE_MULTILINE`: `PAUSE_MIN` 0,12, `PENALTY_PAUSE_IN_CUE` 100 do najviše
50, `PAUSE_BREAK_DISCOUNT` 50, `PAUSE_AT_FUNCTION_WORD` "discard",
`PENALTY_SPLIT_PRONOUN` 25). `web` i `digital` netaknuti (test).

**Udaljenost tišine od granice reči** (65 tišina ≥ 0,30 s unutar rečenice):
0 (dodiruje granicu) 23 %, do 0,10 s 48 %, do 0,20 s 18 %, do 0,30 s 5 %,
preko 6 %. Rez na granici plus vreme ivice može da skine 69 % (rep
prethodne reči 38 %, čelo sledeće 31 %); 31 % je duboko u reči.

**Rezerva** (natpisa sa pauzom, posle vremena):

| | bez vremena | 0 | 0,08 | 0,15 |
|---|---|---|---|---|
| `tv` (sa paketom) | 41 / 78 | 29 (37,2 %) | 29 | 29 |
| `web` | 53 / 139 | 42 (30,2 %) | 41 | 41 (29,5 %) |

0,15 ne kvari ništa (u `web` je i jedan bolji: manje odbijenih zbog CPS-a,
jer pomeraj ostaje kraći) — uzeta je najveća. Izbor po slušanju je otvoren:
tri SRT-a istog posla, `..._lead000/008/015.srt`.

**Stanje `tv` posle paketa** (prema prelomu bez pauza, bez vremena):

| | cue-ova | sa pauzom | početak / kraj / sredina | preko CPS | prekratkih |
|---|---|---|---|---|---|
| prelom bez pauza | 72 | 41 (56,9 %) | 9 / 0 / 32 | 1 | 1 |
| paket | 78 | 41 (52,6 %) | 15 / 2 / 24 | 1 | 1 |
| paket + vreme (0,15) | 78 | **29 (37,2 %)** | 1 / 0 / 28 | 1 | 1 |

To je spoj: paket premešta rez na granicu kod tišine (sredina 32 -> 24,
početak 9 -> 15), vreme ga skida (početak 15 -> 1). Cena: 6 natpisa više,
CPS i prekratki isti.

**Spoj u `web`** (popust + filter + zamenica, pa vreme): prag 0,20 / 0,30 ×
popust 25 / 50 / 100 × kazna 0 / 50 daje najviše 41 -> 39, bez cene, 2–4
promenjene rečenice; čitanje (ponovljeni jednom) 1 : 1 : 1. Tek kazna 1000
uz kapu 300 spušta na 31, po ceni od natpisa ispod 14 znakova 6 -> 18 i
CPS +2. Ne uvodi se — u jednom redu rez nema kuda, kao u krugovima 19–23.

## Obrazac: generično ime klase se sudara tiho (2026-10-14, krug 26)

Ikonica objašnjenja (`<span class="info">`, kružić od 14 px, krug 12) i
traka obaveštenja (`<div class="flagbar info">`) delile su ime `info`. Obe
klase imaju istu težinu, `.info` je niže u `app.css`, pa je traka dobila
`display: inline-grid` i 14 × 14 px. Vidi se samo na strani koja oba
elementa prikazuje zajedno — prevod uvezenog SRT-a — pa je greška stajala
četrnaest krugova, i kroz krugove u kojima se CSS čitao. Testovi su gledali
tekst.

Pravilo: komponenta dobija ime koje ne može da deli sa modifikatorom
(`why-ic`, ne `info`). Provera bez brauzera (`tests/test_css_sudari.py`):
klase koje u `app.css` SAMOSTALNO postavljaju `display` su komponente;
element sa dve takve je sudar. Na kodu pre popravke: 37 komponenti, tačno
jedan sudar — ovaj. Klasa bez ijednog pravila u markupu je tri (`clean`,
`take`, `worker-state`), sve namerne (stanje, JS kuka, sestrinska klasa);
nijedna iz krugova 20–25.

## 2026-10-14 — Krug 27: trake upozorenja kao jedan red; `.err` je curio u traku

Svaka traka je jedan red: ikonica i boja (težina), šta i koliko, radnja ako
je ima, ikonica za detalje (`why-ic`). Spisak, „zašto" i uputstvo su u
prozorčiću. Broj se ne krije — da nešto nije u redu i koliko toga vidi se bez
klika. Ukupna visina traka na poslu prevoda uvezenog SRT-a (crvena + cijan),
Chrome 1280 × 900: **712 px -> 75 px** (crvena 669 -> 37, cijan 43 -> 37).

Prazan prostor u spisku (42 px po stavci, `ul` 379 px za tri stavke) je isti
obrazac kao `info` iz kruga 26, samo na drugom svojstvu: `.err { color;
white-space: pre-wrap }` je samostalno pravilo za crveni tekst i za
`<pre class="err">`, a `err` je i modifikator trake (`.flagbar.err`). Traka
je nasledila `pre-wrap`, pa je svaki prelom reda i uvlačenje šablona između
`<li>` postao vidljiv red. Provera iz kruga 26 to ne hvata — gleda samo
`display` na elementu sa dve komponente; ovde je jedna komponenta i jedan
modifikator, a svojstvo se nasleđuje. Proširenje (`modifier_leaks`): za
svaki par `.a.b` čiji `b` ima i samostalno pravilo, svojstva tog pravila
koja ni `.a` ni `.a.b` ne postavljaju cure u komponentu. Na kodu pre
popravke: tačno jedno curenje, `.flagbar.err` <- `white-space`. Popravka:
`pre-wrap` je samo na `pre.err`.

## 2026-10-20 — Krug 28: pokretač iz `.venv` vara proveru procesa (dva puta)

Na Windows-u je `.venv\Scripts\python(w).exe` samo pokretač: odmah pravi
osnovni interpreter kao dete i čeka ga. Zato (1) port servera drži proces
čiji je izvršni fajl VAN foldera aplikacije, a pokretač mu je roditelj —
provera „izvršni fajl u folderu aplikacije" za pravi server uvek odbija; i
(2) proces stavljen u Job objekat posle pokretanja nije onaj koji radi — dete
je već van Job-a. Prvi test preživljavanja je zato „prolazio" i u varijanti
bez izlaza iz Job-a. Pravilo: proveru procesa raditi na osnovnom
interpreteru (`sys._base_executable`) i potvrditi je (`IsProcessInJob`), a
identitet servera tražiti preko porta i `/api/identity`, ne preko putanje.

## 2026-10-20 — Krug 29: granica veze, PID fajl, i Python koji ne vidi HTTPS

- Veza kojom Claude dolazi do MCP procesa ističe posle ~60 s; svaki alat je
  sada ispod pola minuta i u najgorem slučaju, a duge radnje (`tests`,
  `script`) idu u pozadinu i prate se preko `zadatak(oznaka)`. `rebuild` na
  svih 87 prelamanja iz istorije: najduže 3,1 s.
- Rezerva gašenja ide preko `tmp/server.pid` (PID + vreme pokretanja iz
  Windows-a); provereno nad pravim serverom — upisani PID je vlasnik porta.
- Svaki test sa `TestClient`-om je praznio pravi `tmp/` (server ga prazni pri
  pokretanju) — i tako brisao izlaz `tests()` koji pokreće te iste testove.
- Na ovoj mašini sistemski Python 3.11 ne može da otvori nijednu HTTPS vezu:
  7 MUP sertifikata u Windows CA skladištu OpenSSL ne može da pročita, a
  `ssl.create_default_context()` učitava sve. Zato je pala automatska
  bezbednosna provera. Python 3.14 ih preživljava; `.venv` alati (httpx,
  pip) koriste `certifi` i ne osećaju ovo.

## Obrazac: „lokalno" nije isto što i „bezbedno" (2026-10-20, krug 30)

„Lokalno" nije isto što i „bezbedno" čim postoji ruta koju neko drugi može da
pozove. Pretpostavka koja je važila kad je kod pisan prestaje da važi tiho, i
ništa je ne podseti da je prestala. Dashboard je rastao dvadeset osam krugova
kao lokalni alat; čim je dobio API koji zove neko drugi, jedan commit je
nosio tri rupe — a pregled ostatka površine našao je još četiri i jednu
zajedničku (lažni zahtev sa tuđe stranice na svim rutama koje menjaju
stanje).

Dva izmerena primera iz kruga 30:

- Provera imena fajla po `..`, `/` i `\` liči na potpunu, a nije: na
  Windows-u je `out / "C:tajna.txt"` putanja `C:tajna.txt` — disk C, van
  posla. Ime se proverava po onome što sme da bude (ime `.srt` fajla), i
  razrešena putanja mora ostati u svom folderu.
- Proba zaštite od lažnog zahteva u pravom pregledaču meri nešto tek kad
  kontrola — isti napad sa isključenom zaštitom — stvarno uspe (obrisao je
  posao). Bez kontrole, „posao i dalje postoji" ne razlikuje zaštitu od napada
  koji se ni ne izvede.
