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
