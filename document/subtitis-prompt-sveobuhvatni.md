Radimo na internoj desktop aplikaciji **Subtitis**. Ona pravi SRT titlove iz
audio i video zapisa i prevodi ih, i ume da prevede već gotov SRT bez diranja
tajminga. Sve se izvršava lokalno — nijedan zapis ne napušta mašinu, jer je
materijal poverljiv.

Radi **fazno**. Završi fazu, stani, javi mi izmerene rezultate i čekaj
potvrdu. Ne prelazi na sledeću fazu sam.

## Pravilo rada koje važi od prve do poslednje faze

**Merenje pobeđuje pretpostavku.** U izgradnji ovog alata je najmanje osam
puta uverljivo objašnjenje kvara bilo pogrešno, a pravi uzrok drugde. Zato:

- Kad ti kažem šta je uzrok nečega, **prvo proveri pa onda popravljaj.** Ako
  nalaz ne stoji, reci pre nego što uradiš.
- Svaku promenu koja utiče na kvalitet izlaza **izmeri pre i posle** i ispiši
  brojeve. Ne „bolje je" nego „155 → 173 reči, pokrivenost 55,9% → 83,8%".
- Pre nego što predložiš prag ili konstantu, ispiši veličinu na osnovu koje
  bi se odlučivalo — i ostavi je u redovnom ispisu, da se prag kasnije podesi
  na osnovu podataka a ne pogađanja.
- Kriterijum postavi **pre** merenja i formuliši ga tako da meri kvalitet, a
  ne da li je mehanizam primećen. („Menja više od desetine slučajeva" meri da
  model primećuje pravilo, ne da je rezultat bolji.)
- **Gde ima nasumičnosti, jedan prolaz nije merenje.** Jedan prolaz je dao 11
  reči, pet ponavljanja 20–52. To nije drugačija hipoteza nego nedovoljan uzorak.
  Svuda gde postoji temperatura — popuštanje posle neuspelog dekodiranja, svaki
  poziv jezičkom modelu — ponavljaj i javljaj raspon, ne jedan broj.
- Ne odbacuj popravku zato što nije dostigla proizvoljan cilj. Ako je bolja po
  sva tri merila, uzima se.

## Minimum hardvera

Pročitaj ovo pre nego što počneš: ako mašina ne ispunjava prvi red tabele,
ništa od dalje opisanog ne radi, i bolje je da to kažeš odmah nego posle
preuzimanja petnaest gigabajta modela.

| Deo | Minimum | Zašto |
|---|---|---|
| **Grafička kartica** | **NVIDIA, 12 GB VRAM** (razvijano na 16 GB) | prepoznavanje ~5 GB, prevod ~8 GB; nikad istovremeno, ali svaki mora da stane |
| Prostor na disku | **15 GB** za modele + **115 MB po satu zapisa** | Whisper `large-v3` ~3 GB, `gemma3:12b` ~8 GB, ostalo biblioteke; radna kopija zvuka se briše po roku |
| Radna memorija | 16 GB, SSD | procesor nije usko grlo, sve teško ide na karticu |
| Operativni sistem | Windows 10 ili noviji | na Linuksu radi isto, osim zaobilaženja koja su ovde označena kao windows-specifična |

AMD kartica, Apple Silicon i rad samo na procesoru nisu podržani. Ako ti se
čini da bi neki od njih radio, **prvo izmeri**, pa javi — ne menjaj izbor
modela da bi stao u manju memoriju, jer je taj izbor donet merenjem kvaliteta
na srpskom.

## Softver

- **Python 3.11** (ne 3.12 — `ctranslate2` ume da pravi problem)
- Ollama lokalno na `http://localhost:11434`
- `ffmpeg` u PATH — proveri na startu i javi jasnu grešku ako nije
- `venv`, sve zavisnosti u `requirements.txt`, bez Dockera i bez Poetry
- Radni jezici: srpski (najčešći), engleski, francuski, nemački

Podrazumevaj ovaj redosled postavljanja i napravi ga tako da prolazi od nule:

```
1. Python 3.11 (ne 3.12), uz „Add to PATH"
2. ffmpeg, uz ffmpeg.exe u PATH
3. NVIDIA drajver (CUDA 12.x dolazi uz njega; CUDA Toolkit nije potreban)
4. Ollama, pa `ollama pull gemma3:12b`
5. git clone, `python -m venv .venv`, aktivacija, `pip install -r requirements.txt`
6. `subtitis doctor`   — javlja šta od navedenog nedostaje
7. `subtitis serve`    — ili prečica na radnoj površini
```

Whisper model se preuzima sam pri prvom prepoznavanju, pa prvi posao traje
dodatnih nekoliko minuta; napiši to u ispis, inače izgleda kao da je stalo.

Napravi i komandu `doctor` koja proverava sve spoljne zavisnosti (ffmpeg,
Ollama, CUDA, modeli, prostor na disku) i ispisuje šta nedostaje, sa tačnom
komandom za svaku stavku — koristiću je prva. Ona je i jedini dokument o
postavljanju koji neće zastariti, jer se izvršava.

## Osnovni princip arhitekture

**`words.json` je izvor istine. Sve ostalo je izvedeno i može se
regenerisati.**

Iz toga sledi:

- promena prelomnog profila → regeneriši SRT, **ne** pokrećeš ponovo
  prepoznavanje
- dodatni ciljni jezik → prevedi ponovo, **ne** pokrećeš ponovo prepoznavanje
- ručna korekcija → nikad se ne pregazi automatski

**Modeli se nikad ne drže u VRAM-u istovremeno.** Whisper large-v3 je ~5 GB,
jezički model ~8 GB. Pajplajn je ionako sekvencijalan, pa posle svake faze
oslobodi model: `keep_alive: 0` za Ollamu, `del` + `torch.cuda.empty_cache()`
za Whisper.

## Struktura projekta

```
subtitis/
  subtitis/
    __init__.py  __main__.py
    cli.py            # Typer CLI
    config.py         # subtitis.toml, port 8010
    bootstrap.py      # mora se izvršiti PRVI, vidi zamku o pythonw
    audio.py          # ffmpeg normalizacija
    asr.py            # faster-whisper
    segment.py        # reči -> rečenice
    punctuate.py      # uslovna restauracija interpunkcije
    ekavica.py        # kurirani rečnik, 4 sloja
    translit.py       # latinica <-> ćirilica
    linebreak.py      # prelamanje u cue-ove (DP sa kaznama)
    translate.py      # Ollama klijent + validacija
    srt.py            # čitanje i pisanje SRT-a
    fixedcues.py      # prevod SRT-a sa fiksnim tajmingom (faza 6)
    typography.py     # post-procesor po jeziku
    untranslated.py   # detekcija neprevedenog
    repeats.py        # detekcija ponavljanja
    models.py         # pydantic sheme
    pipeline.py       # orkestracija faza
    desktop.py        # pokretanje bez terminala
    web/
      app.py  db.py  worker.py
      templates/  static/
  profiles/ netflix.json tv.json web.json
  glossary/ default.json
  eval/ pairs.json
  scripts/ bench.py asr_probe.py vad_probe.py eval_translation.py
  tests/
  jobs/            # radni direktorijum, u .gitignore
  subtitis.bat
  subtitis.toml
  requirements.txt
```

Jedan posao na disku:

```
jobs/<job_id>/
  job.json          # status, parametri, istorija faza
  source.wav        # 16 kHz mono
  words.sr.json
  sentences.sr.json
  sentences.fr.json
  out/*.srt
  logs/job.log
```

### Sheme

`words.json`:

```json
{"schema_version": 1, "job_id": "...", "source_file": "...",
 "duration": 180.44, "language": "fr", "script": "Latn",
 "asr": {"model": "faster-whisper/large-v3", "params": {},
         "detected_language": "fr", "detected_probability": 1.0},
 "words": [{"i": 0, "s": 0.00, "e": 0.34, "w": "Je", "p": 0.99}]}
```

`sentences.<lang>.json`:

```json
{"schema_version": 1, "language": "fr", "script": "Latn",
 "derived_from": "words.fr.json", "edited_by_human": false,
 "sentences": [{"id": 1, "s": 0.00, "e": 3.06, "text": "...",
                "words": [0, 7], "speaker": null,
                "flags": ["low_confidence"]}]}
```

---

# FAZA 1 — pajplajn preko komandne linije, bez prevoda i bez interfejsa

Cilj: izmeriti koliko je prepoznavanje dobro na srpskom **pre** nego što se
išta gradi oko njega. Bez interfejsa, bez prevoda.

Redosled: `audio.py` → `asr.py` → `segment.py` → `linebreak.py` →
`translit.py` → `srt.py`, i CLI komande `doctor`, `transcribe`, `build`.

## Prepoznavanje govora

`faster-whisper`, model `large-v3`, `device="cuda"`,
`compute_type="float16"`. Izbor modela **fiksiraj u kodu** i ne izlaži ga na
korisničkom ekranu — manji modeli su na srpskom osetno lošiji, a rizik da se
izbor slučajno spusti bez objašnjenja veći je od koristi. Ostavi kao CLI
opciju.

Parametri, svi kao imenovane konstante na vrhu modula:

```python
WORD_TIMESTAMPS = True            # osnovni zahtev, bez ovoga nema prelamanja
CONDITION_ON_PREVIOUS_TEXT = False  # glavna odbrana od petlji ponavljanja
HALLUCINATION_SILENCE_THRESHOLD = None   # NE MENJATI, vidi zamku ispod
COMPRESSION_RATIO_THRESHOLD = 2.4
LOG_PROB_THRESHOLD = -1.0
NO_SPEECH_THRESHOLD = 0.6
TEMPERATURE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
DEFAULT_VAD_PRESET = "isključeno"        # vidi merenje ispod
VAD_PRESETS = {"normalno": 0.35, "tih snimak": 0.20, "isključeno": None}
MAX_VAD_DISCARD_SHARE = 0.10
```

Jezik se zadaje **eksplicitno**. Automatska detekcija srpski redovno
prepozna kao hrvatski i vrati ijekavicu. Ponudi i `auto`, ali prikaži šta je
detektovao i sa kojom sigurnošću.

Ključevi glosara ulaze u `initial_prompt` — znatno poboljšava pogađanje imena.

Posle svakog prepoznavanja ispiši **pokrivenost** (udeo trajanja zapisa
pokriven rečenicama) i nabroj praznine duže od 3 s sa tačnim vremenima. To je
jedina zaštita od tihog gubitka sadržaja.

> **ZAMKA 1 — `hallucination_silence_threshold` guta rep zapisa.**
> Kada ta grana segment proglasi sumnjivim, u implementaciji izvrši
> `seek = content_frames` — preskok **do kraja datoteke** — i tiho obriše sav
> preostali sadržaj, bez ijednog reda u dnevniku. Izmereno: pokrivenost
> 55,9% → 83,8%, reči 155 → 173 čim je isključen. Ostavi ga na `None`, izloži
> kao izričitu opciju sa upozorenjem, i **napiši test koji pada ako ga neko
> vrati u podrazumevane parametre**, sa objašnjenjem u samom testu.

> **ZAMKA 2 — filter tišine (VAD) je podrazumevano isključen, suprotno
> očekivanju.** Tri prolaza istog tihog zapisa: prag 0,35 → 179 reči i 85,6%
> pokrivenosti; 0,20 → 180 i 87,9%; **isključeno → 189 i 88,9%**, uz osetno
> tačniji tekst. Na tihom zapisu filter seče delove govora koje model inače
> rekonstruiše iz konteksta. Odbrana od izmišljenog teksta ostaje na
> `condition_on_previous_text=False` i pragu kompresije.
> Izloži VAD kao **tri imenovana izbora**, ne kao broj — prag korisniku ne
> znači ništa dok ne pusti ceo prepis.
> Kad je filter uključen a odbacio bi više od desetine zvuka, predloži
> poređenje sa isključenim. Mera je **udeo odbačenog zvuka**, ne pokrivenost:
> najveća ikad izmerena pokrivenost je 88,9%, pa je kao prag beskorisna.

## Segmentacija

Reči → rečenice po interpunkciji, uz zaštitu od skraćenica (`itd.`, `npr.`,
`dr.`, redni brojevi).

```python
MIN_SENTENCE_WORDS = 3
MIN_SENTENCE_DURATION = 1.0
MIN_DURATION_PER_WORD = 0.15
MAX_SENTENCE_DURATION = 15.0      # važi i za SRT režim iz faze 6
LOW_CONFIDENCE_THRESHOLD = 0.6
LOW_CONFIDENCE_SENTENCE_SHARE = 0.20
MIN_WORDS_FOR_SHARE_RULE = 5
```

Rečenica duža od `MAX_SENTENCE_DURATION` lomi se na najvećoj pauzi, a pri
izjednačenju na onoj **najbližoj sredini**.

> **ZAMKA 3 — `max()` nad listom torki pravi jednorečne rečenice.**
> Na francuskom zapisu je svaka reč postala zasebna rečenica (0,2–0,4 s).
> Objašnjenje koje se nameće — „nema interpunkcije pa nema na čemu lomiti" —
> je pogrešno. Pravi uzrok: `max()` nad listom parova `(pauza, indeks)`; kada
> su sve pauze jednake, a u tečnom govoru bez interpunkcije jesu, poređenje
> pada na drugi član i bira **najveći indeks**. Rez uvek na poslednjoj reči,
> pa rekurzija ljušti po jednu reč sa kraja do kraja zapisa. Eksplicitno
> zadaj ključ i tie-break ka sredini.

> **ZAMKA 4 — `low_confidence` na osnovu `any()` pali skoro sve.**
> Prvo rešenje je palilo oznaku ako **bilo koja** reč ima verovatnoću ispod
> praga: 41% rečenica na dobrom zvuku, 80% na lošem. Uzrok nije prag po reči
> nego `any()`. Pređi na **udeo** reči ispod praga (0,20), uz minimum reči da
> pravilo ne pali na kratkim rečenicama. Posle: 4% na dobrom, 40% na lošem
> zvuku. Obojene reči su pretežno kratke funkcijske reči na kojima je model
> nesiguran i kad ih tačno prepozna — prave greške su drugde.

## Reči bez zvuka iza sebe

Postoji klasa izmišljenog teksta koju verovatnoća **ne vidi uopšte**. Na jednom
zapisu je poslednjih devet reči imalo trajanje tačno nula, sve na istom trenutku,
a verovatnoće `0.87`, `0.98`, `0.99` — model je siguran u ono što je izmislio.

```python
ZERO_DURATION_S = 0.02
MIN_RUN_FOR_FLAG = 2    # niz od dve ili više uzastopnih; interpunkcija se ne broji
```

Prijavi takav niz sa vremenima i tekstom, označi rečenicu oznakom **odvojenom** od
oznake nesigurnosti, i u editoru je oboji drugom bojom. To su dve različite
pojave: jedna znači da model nije siguran, druga da je siguran u nešto čega nema.

**Pravilo mora da se primeni posle spajanja elizija**, ne nad sirovim izlazom —
inače francusko `c'` `est` u ponovljenom `c'est c'est-à-dire` daje lažnu uzbunu i
navede te da podigneš prag.

## Prelamanje u cue-ove

Ovo je srce alata i jedini netrivijalan deo. **Deterministički** — bez
jezičkog modela. Generiši kandidate za tačku preloma, dodeli kaznene poene,
izaberi kombinaciju sa najmanjim zbirom (dinamičko programiranje).

Vremena cue-a izvode se iz **stvarnih vremena prve i poslednje reči** u njemu.
Trajanje se ne deli srazmerno broju znakova — prava vremena postoje.

| Situacija | Kazna |
|---|---|
| red duži od dozvoljenog | 1000 / znak |
| razdvaja predlog od imenice (`u / kuću`) | 300 |
| razdvaja delove složenog glagola (`bi / došao`) | 300 |
| enklitika na kraju ili početku reda | 300 |
| razdvaja pridev od imenice | 200 |
| jedna reč sama u redu | 400 |
| brzina čitanja preko dozvoljene | 100 / jedinica |
| prelom nije na granici klauze | 50 |
| prelom iza veznika | 10 |
| prelom posle zareza | 0 |
| neravnoteža redova | 2 / znak |
| prekratak cue | 20 / znak |
| svaki novi cue | 20 |

```python
PENALTY_SHARE_DEVIATION = 4     # koristi se u fazi 6
PENALTY_PER_CUE = 20
MIN_ALIGNED_TOKEN_SHARE = 0.6
```

> **ZAMKA 5 — tabele funkcijskih reči moraju biti po jeziku.** Srpska
> enklitika ne sme ni da počne red, dok engleski i francuski pomoćni glagoli
> slobodno počinju red (`he / is going`). U istom skupu bi engleski dobio
> zabranu koja u tom jeziku ne postoji. Ako jezik nije u tabelama, primeni
> samo opšta pravila i **upiši upozorenje** — ne lomi tiho bilo gde.

> **ZAMKA 6 — francuska elizija nije spisak od četiri reči.** Rešenje je
> opštije: pravilo hvata svaki token koji se završava apostrofom, jer je
> elizija fonološka pojava.

Profili kao JSON fajlovi:

| Profil | oblik | brzina (CPS) | trajanje cue-a | izlaz |
|---|---|---|---|---|
| **tv** | 2 reda × 42 znaka | 20 | 0,83 – 7,0 s | kako jeste |
| **web** | 1 red, 10–20 znakova | 20 | 0,50 – 4,0 s | kako jeste |
| **digital** | isto kao `web` | 20 | 0,50 – 4,0 s | VELIKA SLOVA, bez interpunkcije |

Ne pravi tri varijante istog (42 / 37 / 47 znakova u dva reda) — izbor među njima
ne nosi odluku. Razlika mora biti u **obliku**.

Za jednoredni profil isključi tri kazne koje tamo mere nešto čega nema: jedna reč
sama u redu, neravnoteža dva reda, prekratak cue po znakovima. Kazna za enklitiku
**ostaje** i tamo je vidljivija nego inače, jer se dva dela sintagme pojavljuju u
različitim trenucima. Umesto isključenih uvedi **donju granicu dužine reda** kao
meku kaznu (30 po znaku ispod 10), sa izuzetkom **samo kad je cela rečenica kraća
od granice** — ne kad je kratak ostatak posle reza, inače svaki poslednji komad
rečenice prolazi bez kazne i dobijaš natpis od tri znaka.

Promena profila ne pokreće nikakvu ponovnu obradu — samo regeneriše SRT iz
sačuvanih rečenica. Prikaži kojim je profilom napravljen postojeći fajl.

## Transliteracija

Smer ćirilica → latinica je trivijalna zamena. Obrnuti smer je zamka.

> **ZAMKA 7 — digrafi `lj`, `nj`, `dž` nisu uvek digrafi.**
> `nadživeti` → `надживети` (ne `наџивети`), `injekcija` → `инјекција`,
> `konjugacija` → `конјугација`, `podžanr` → `поджанр`, `Tanjug` → `Танјуг`.
> Zato je transliteracija deterministična funkcija sa listom izuzetaka u kodu
> (prefiksi `nad-`, `pod-`, `od-`, `pred-`, internacionalizmi sa `nj`/`inj`),
> a **ne biblioteka**. Svaki gornji primer dobija svoj test.
> Ovaj korak nikada ne izvršava jezički model — mora biti stopostotno
> predvidiv.

## Format izlaza

Standardni SRT: redni broj, vremenski opseg, tekst, prazan red. Kodiranje
**UTF-8 bez BOM-a**, prelomi redova **CRLF**. Obe stavke su izvor tihih
problema u plejerima ako se pogreše. Imenovanje po oznaci jezika, sa pismom
gde je relevantno; profil **ne** ulazi u ime fajla.

## Na kraju faze 1 — stani

Pokreni prepoznavanje na stvarnom zapisu i javi mi: broj reči, broj rečenica,
pokrivenost, praznine preko 3 s, odnos prema stvarnom vremenu, i izgled prvih
dvadeset cue-ova. Čekam da pročitam pre nego što idemo dalje.

---

# FAZA 2 — prevod

Ollama, podrazumevani model **`gemma3:12b`**. Izbor je donet merenjem na
skupu od devet parova rečenica iz stvarnog materijala sa ručno napisanim
referentnim prevodom; ocenjivano ručno u tri stepena, jer automatske mere na
toliko malom uzorku ne znače ništa. `gemma3:12b` (7,8 GB, 37 tok/s, 1,03×
dužine reference) pobedio je `qwen3:14b` (9,7 GB, 35 tok/s, 1,06×) na
kvalitetu, brzini i memoriji. Napravi `eval/pairs.json` i skript za
ocenjivanje da se izbor može ponoviti.

```python
num_ctx = 8192
temperature = 0.2
top_p = 0.9
repeat_penalty = 1.0
think = False
CHUNK_SIZE = 9        # rečenica po zahtevu
```

## Tri tehnička uslova — bez njih sistem tiho ne radi

1. **Isključeno rezonovanje.** Model podrazumevano „razmišlja" pre odgovora,
   što za prevod znači sedmostruko sporiju obradu bez koristi. Proveri
   **dvostruko**: novije verzije vraćaju rezonovanje u zasebnom polju, ne u
   oznakama.
2. **Izričita veličina konteksta.** Podrazumevanih 4096 tokena servis **tiho
   odseca**, bez greške i bez traga. Posle prvog poziva pročitaj vrednost
   nazad iz servisa i uporedi.
3. **Provera nepoznatih polja.** Servis tiho prihvata izmišljena polja —
   greška u imenu opcije ne ostavlja trag. Čitaj stvarno primenjene vrednosti
   nazad, ne pretpostavljaj ih.

## Prevodi se rečenica, ne cue

Prevođenje pojedinačnih cue-ova je najčešća greška u ovakvim sistemima: cue
je proizvoljan isečak rečenice i izolovano gubi rod, padež i vezu sa
kontekstom. Prevodi **rečenice, sa dve rečenice konteksta sa svake strane**
koje se šalju ali ne prevode. Prelamanje se posle radi **ponovo**, nad ciljnim
jezikom i njegovim pravilima.

Prevod je zasebna faza koju pokreće izričit klik, **nikad automatski** —
prevođenje neispravljenog teksta prevodi i greške, pa se onda ispravljaju dve
datoteke umesto jedne.

## Zaštita od pomeranja

Najozbiljniji kvar nije loš prevod nego **pomeren prevod**: pošalješ deset
numerisanih rečenica, model vrati devet, ceo ostatak fajla se pomeri za jedan
red — a tekst izgleda savršeno. Zato traži numerisan izlaz u tačnom formatu i
proveri broj programski. Neuspeh → prepolovi odlomak → dve rečenice → jedna
po jedna. Posle dva neuspeha na pojedinačnoj rečenici upiši original sa
oznakom i idi dalje. Jedna rečenica nikad ne obara ceo posao.

Dopuna iz merenja: ako se ista greška ponovi u manjem odlomku kao u većem,
prekini polovljenje odmah — očigledno ne pomaže.

> **ZAMKA 20 — glosar curi u prepis tamo gde nema govora.**
> Ključevi glosara idu u `initial_prompt`, ali da bi preživeli ceo zapis moraju i u
> `hotwords`, koji ulazi u **svaki** prozor od 30 s. Gde model nema šta da čuje —
> muzika, džingl, špica — vraća baš te reči. Mereno na zapisu od 32 s sa džinglom:
> sa glosarom 77 reči i pet termina preko džingla doslovno
> (`Otvorena vrata, letnja radionica, OKC, Divčibare, Srbija`), sa praznim
> glosarom 66 reči i nijedan termin.
> **Glosar mora biti vezan za posao, ne globalan**, sa kopijom pri otvaranju i
> praznim glosarom kao ravnopravnim izborom. Globalan glosar znači da materijal
> jedne organizacije curi u snimke druge.

> **ZAMKA 21 — filter tišine i glosar zajedno daju nasumičan prepis.**
> Četiri kombinacije, svaka pet puta: bez glosara prepis je ponovljiv i sa filterom
> i bez njega; **sa oba uključena izlazi 20–52 reči, svaki put drugačije.** Filter
> seče zapis u kratke odsečke, u svaki ulazi glosar, i model na njima odustaje pa
> vraća prompt. Filter sam po sebi radi uredno. Ne meri ih odvojeno — interakcija
> se vidi samo u tabeli 2×2.

> **ZAMKA 8 — brojevi koje model vidi moraju počinjati od 1.**
> Dok su se odlomku šiljali stvarni identifikatori rečenica, model je ispred
> njih dodavao **svoje** nabrajanje: `1. 28|And then we packed the tents…`. Provera nije
> prepoznavala cifru i odbacivala ceo odlomak — 14 od 16 odbijanja na jednom
> poslu. Šalji lokalne brojeve `1..n` i drži mapiranje u kodu. Izmereno:
> **16 odbijenih odlomaka → 0, zahteva 54 → 22, vreme 326 → 181 s.**

## Kada prevod nije prevod

Provera broja linija ne hvata slučaj u kom model vrati **izvorni tekst** pod
ispravnim rednim brojem — a u gotovom titlu to izgleda kao da je sve prošlo.
Zato posle prevoda radi zasebna provera: rečenica je sumnjiva ako sadrži slova
kojih u ciljnom jeziku nema (`ć đ š ž č` za francuski) ili je po rečima 80%
ista kao original. **Vlastita imena izuzmi** — bez tog izuzetka `Jelena Đurđević`
i nazivi organizacija imaju veći udeo takvih slova od pravih promašaja.
Izveštaj navodi **broj cue-ova**, ne rečenica: jedna neprevedena rečenica ume
da pokriva deset cue-ova.

## Glosar

Tabela termina po smeru prevoda (`sr->en`, `sr->fr`). Koristi se na dva mesta:
ključevi u `initial_prompt` za prepoznavanje, celi parovi u prevodni upit.

Izmereno: bez glosara `Radionica otvorenih vrata` → „Open Doors Workshop",
`animator` → „animator", `sporazumevati se` → „to reach an agreement" (pravno
značenje umesto svakodnevnog); sa glosarom sva tri tačno, i greška se vraća
čim se unos ukloni.

U glosar se upisuju i termini koje **ne treba menjati** (`"Đurđević": "Đurđević"`),
da ih model ne transliteruje ili „popravi".

## Pivot preko trećeg jezika

Za parove sa slabom direktnom spregom (srpski → francuski) implementiraj i put
preko engleskog, kao podešavanje. Ne odlučuj unapred — oba puta postoje, a
merenje odlučuje. Glosar za drugi skok izvedi iz postojećih smerova; ako
ostane prazan, upiši upozorenje.

## Na kraju faze 2 — stani

Javi mi rezultat ocenjivanja nad `eval/pairs.json` za oba modela i prevod
istog zapisa sa glosarom i bez njega.

---

# FAZA 3 — veb interfejs

FastAPI + SQLite + Jinja2 + HTMX + SSE. **Bez Reacta i bez koraka izgradnje** —
za alat koji koristi jedan čovek lanac alata za izgradnju je veći teret od
koristi.

Ekrani: lista poslova i nov posao; stranica posla sa fazama i napretkom;
editor rečenica; podešavanja i glosar.

Worker je **nit unutar servera**, ne zaseban proces — korisnik ne treba da zna
da postoje dva procesa. Zadrži zasebnu komandu za razvoj (`--no-worker` da se
nit isključi). Nit ne sme da zadržava gašenje: petlja čeka na **događaj**, ne
na istek vremena, i najviše dve sekunde pri gašenju.

Indikator „obrada aktivna" u zaglavlju hrani otkucaj iz **zasebne** niti na 5
sekundi. Da ga kuca glavna petlja, indikator bi pocrveneo usred svakog dužeg
prepoznavanja. Prelaz u crveno posle 20 s bez otkucaja.

SSE za napredak. U editoru: bojenje po sigurnosti, poništavanje, automatsko
snimanje, isečak zvuka po rečenici (dugme ▶), prečice.

## Funkcionalna pravila

- **Izvedeni fajlovi se uvek mogu regenerisati** — promena profila ili pisma
  ne pokreće prepoznavanje ni prevod.
- **Ručna ispravka se nikad ne pregazi.** Fajl sa oznakom ljudske izmene ne
  prepisuje se automatski; ponovno prepoznavanje vraća odbijanje i traži
  izričitu potvrdu, uz rezervnu kopiju. Provera se izvodi **posle**
  prepoznavanja a ne pre — radije jedan uzaludan prolaz nego pregažene
  ispravke.
- **Ispravljen tekst zadržava prava vremena.** Poravnaj izmenjeni tekst sa
  originalnim tokenima; nepromenjena mesta zadržavaju stvarna vremena,
  interpoliraju se samo dodati tokeni. Ako udeo poravnatih tokena padne ispod
  `MIN_ALIGNED_TOKEN_SHARE`, označi degradirano poravnanje i upiši upozorenje
  — tiho loše poravnanje se otkriva tek u plejeru.
- **Neuspeh prekida zavisne zadatke** — ne izvršavaj prelamanje nad
  nepostojećim rečenicama i ne prijavljuj „gotovo". Preskočeni zadaci se vide
  sa razlogom.
- **Prazan rezultat je greška**, ne uspeh sa praznim titlom. Jedan prazan
  zapis je prošao ceo tok i proizveo titl od nula bajtova označen kao gotov.
- **Posao se može obrisati i dok radi.** Radna kopija zvuka ne može se
  obrisati dok posao ima zadatak; sam posao može — brisanje otkazuje zadatak,
  a obrada staje na prvoj prirodnoj tački, između delova posla, nikad usred
  poziva modelu. Prekid **ne sme** biti ista klasa greške kao ostale greške
  pajplajna: te se na nekim mestima hvataju da bi se nastavilo, pa bi prekid
  bio progutan. Posle prekida worker ne pada, nego prelazi na sledeći zadatak
  i **još jednom** briše direktorijum — zadatak ume da upiše dnevnik posle
  brisanja, a na Windowsu se fajl koji je držao otvorenim ne može obrisati dok
  ga ne pusti.
- **Pisma i jezici se proveravaju na serveru.** Izbor pisma je dostupan samo
  kad je ciljni jezik dvopismen; zahtev za ćirilicom nad jezikom koji je nema
  odbija se sa jasnom porukom u interfejsu, ne sistemskom greškom.

> **ZAMKA 9 — kodni raspored konzole ruši mrežne zahteve.** Dnevnik je išao
> direktno na standardni izlaz, a Windows konzola koristi cp1252 koji ne
> poznaje `ć š đ ž` — dakle skoro svaku poruku na srpskom. Svaki takav ispis
> rušio je **ceo HTTP zahtev**, usred snimanja korekture, sa porukom koja nema
> veze sa uzrokom. Pravilo: **fajl dnevnika je uvek u punom UTF-8; ispis u
> konzolu pada na zamenu znakova, pa na tiho odustajanje.**

> **ZAMKA 22 — na Windowsu zamena datoteke pada dok je ona otvorena.**
> Za razliku od Linuksa, `os.replace` odbija ako je ciljna datoteka otvorena bilo
> gde — i obara ceo posao usred obrade, sa preskočenim zavisnim zadacima. Napravi
> jednu funkciju koja ponavlja zamenu posle 20, 40, 80, 160 i 320 ms, i **pusti sve
> upise preko postojeće datoteke kroz nju**: zapis o poslu, rečenice, titlove,
> podešavanja, glosar. Uz to proveri da nijedan prikaz ne čita istu datoteku više
> puta. Pošto ponavljanje smanjuje verovatnoću ali je ne uklanja, dodaj i **dugme
> za ponavljanje koraka koji je pao**, zajedno sa svim što je zbog njega
> preskočeno.

> **ZAMKA 23 — osvežavanje liste na nekoliko sekundi kvari tri stvari odjednom.**
> Zamena celog bloka vraća skrol na vrh dok korisnik čita, guta brisanje (forma
> zamenjena usred potvrde, pa projekat ostaje) i drži stalan pritisak na datoteke.
> Osvežavaj **samo kad bar jedan projekat ima zadatak**, i tada menjaj samo oznake
> statusa, ne ceo blok. Lista koja se sama menja dok ništa ne radi je laž o stanju.

> **ZAMKA 10 — Python ne pretovaruje module.** Pri merenju pragova je
> izgledalo da nova vrednost „ne stiže do obrade". Uzrok nije bio u
> prosleđivanju nego u tome što je worker držao stari modul u memoriji.
> **Restartuj proces pre svakog merenja.**

- Svaki korak beleženja greške zaštiti **zasebno** — nađeno je da i sam upis
  greške može baciti izuzetak i oboriti celu petlju obrade.
- Nalaženje spoljnih alata: podešavanje → PATH sistema → poznate lokacije
  instalacije. Razvojno okruženje drži kopiju PATH-a od trenutka pokretanja i
  ne vidi kasnije instalacije, pa je `ffmpeg` bio „nedostupan" iako je bio
  uredno instaliran.

## Na kraju faze 3 — stani

---

# FAZA 4 — jezičke normalizacije

Tri normalizacije nad `words.json`, **pre** segmentacije, tako da sve što
sledi radi sa čistim ulazom. Svaka beleži broj izmena i svaku pojedinačnu
izmenu u dnevnik.

## Pismo

Model za srpski vraća čas ćirilicu, čas latinicu, ponekad pomešano unutar
istog fajla. Svedi izlaz odmah na latinicu i radi interno isključivo njome —
tokenizacija je bolja, a modeli za prevod su na latinici pouzdaniji. Ćirilica
se dobija transliteracijom na samom kraju.

## Ekavizacija

> **ZAMKA 11 — regularni izraz za ijekavicu uništava reči.** Pravilo
> „ije/je → e" bi pokvarilo `jesen`, `jedan`, `jezik`, `prijem`, `sjaj`.
> Rešenje je **kurirani rečnik u četiri sloja**, gde se sloj zaštite proverava
> prvi:
>
> | Sloj | Značenje | Primer |
> |---|---|---|
> | zaštićeno | nikad se ne dira | `jesen`, `jezik`, `prijem`, `Mjanmar` |
> | tačan oblik | cela reč | `prije` → `pre` |
> | prefiks | reč počinje osnovom | `djec` → `dec` |
> | bilo gde | osnova unutar reči | `vidje` → `vide`, hvata i `predvidjeti` |
>
> Pri učitavanju modula proveri tabele na **međusobne sudare** — provera je
> već uhvatila slučaj gde bi zaštićena imenica `želja` blokirala glagol
> `željeti`.

## Spajanje elizija

Na francuskom model vraća razdvojene oblike: `j 'aimerais`, `d 'autres`,
`c 'est`, `aujourd 'hui`. Posledica: prelamanje vidi `j` i `'aimerais` kao dve
reči, pravilo o eliziji nikad ne okine, i cue počinje sa `'espère`.

Pokrij kratke osnove po pravilu i duže po spisku (`qu`, `jusqu`, `lorsqu`,
`aujourd`, `puisqu`, `quoiqu`, `quelqu`, `presqu`). Nova reč uzima početak
prve i kraj druge, a **manju od dve verovatnoća** — da spajanje ne sakrije
nesigurnost koju editor treba da oboji. Izmereno: 33 spajanja, 428 → 395
reči, nijedan token više ne počinje apostrofom.

## Restauracija interpunkcije

Zarezi su glavni signal za tačku preloma unutar rečenice. Tekst bez njih daje
loše prelome bez obzira koliko je algoritam dobar. Model za prepoznavanje
punktira nepouzdano — na jednom zapisu je uredno postavio znake u prvih pet
rečenica pa prestao do kraja.

Uslovni korak šalje tekst jezičkom modelu sa zahtevom da vrati **identičan
niz reči**, dodajući samo interpunkciju i velika slova.

### Kapija

```python
PUNCTUATION_DENSITY_THRESHOLD = {"default": 4.0}   # znakova kraja / 100 reči
COMMA_DENSITY_THRESHOLD = {"default": 8.0}         # zareza / 100 reči
MIN_WORDS_FOR_DIAGNOSIS = 25
```

Pokreni restauraciju samo ako je **i** gustina znakova kraja ispod praga **i**
gustina zareza ispod svog praga.

> **ZAMKA 12 — kapija samo po znacima kraja lažno okida.** Prvo rešenje je
> odlučivalo samo po znacima kraja. Jedan zapis je imao 3,4 znaka kraja na sto
> reči, ali **17,6 zareza** — sasvim dovoljno za prelamanje. Restauracija je
> svejedno pokrenuta: 61 s, od toga 33 bačene, pet od osam odlomaka odbijeno,
> i dva sumnjiva ustupka (`curic → curi` i `curic → curici` u istom prolazu).
> Zarezi su ono što prelamanje stvarno koristi, pa oni ulaze u odluku.

Odluka po **gustini**, ne po prosečnoj dužini rečenice: „Dobar dan. Hvala."
ima prosek 1,5 reči po rečenici i savršenu interpunkciju.

Pragove izloži kao **rečnik po jeziku** — francuski i engleski imaju kraće
rečenice od srpskog i prirodno višu gustinu, a sa dva merenja se ne može
izvesti jedan broj koji važi svuda. Ispiši gustinu posle **svakog**
prepoznavanja, bez obzira da li kapija okine, **i koja je od dve odlučila**.

### Validacija odgovora

Poredi odgovor sa originalom reč po reč, bez interpunkcije i bez razlike u
veličini slova. Izmena sadržaja obara **ceo** odgovor. Dve klase odstupanja
dozvoli, jer su pravopisne a ne značenjske:

```python
ALLOW_WORD_SPLITS = True        # ya -> y a; spojeni delovi moraju dati original
ALLOW_INFLECTION_CHANGES = True # prefiks >= 3 znaka, razlika <= 2 znaka
ALLOW_DIACRITIC_CHANGES = False # tres -> très ostaje zabranjeno
```

Svaki dozvoljeni ustupak beleži se sa konkretnim rečima.

### Povlačenje na poslednji znak

> **ZAMKA 13 — model punktira svaki odlomak kao zaokruženu celinu.** Zatvori
> ga tačkom, a sledeći počne velikim slovom — posledica su lažni prelomi
> rečenica na granicama odlomaka: `...avant la pluie.` / `Nous avons rangé le matériel...`
> gde je original bio jedna rečenica. Moja prva hipoteza („odlomak koji je
> odustao") bila je pogrešna.
>
> Rešenje je deterministično i ne traži poverenje u model: posle svakog
> odlomka nađi **poslednji rečenični znak koji nije na samom kraju**, prihvati
> izlaz do tog mesta, a rep vrati u sledeći poziv gde ima pravi nastavak. Uz
> to vrati veliko slovo prve reči na malo — ali samo ako je original bio mali,
> da vlastita imena ostanu netaknuta. Zaštita: ako bi rep bio veći od polovine
> odlomka (`MAX_RETREAT_SHARE = 0.5`), prihvati izlaz kakav jeste uz
> upozorenje, da se izbegne beskonačno vraćanje.
>
> Izmereno pre/posle: 102 s → 60 s, 8 → 1 odbijen odlomak, 45% → 10% bačenog
> vremena, tri od četiri lažna preloma nestala. Četvrti je ostao tačno tamo
> gde je zaštita presekla povlačenje — cena zaštite je vidljiva i prihvaćena.

```python
MIN_CHUNK_WORDS = 20
```

## Na kraju faze 4 — stani

---

# FAZA 5 — upotrebljivost i pokretanje bez terminala

`subtitis.bat` u korenu koji diže server i otvara pregledač. Dvoklik i radi.
Ikonica na radnoj površini preko `pythonw`.

**Ikonicu crtaj skriptom, i boje čitaj iz `app.css` — ne prepisuj ih.**
Skript nacrta `.ico` sa svim veličinama koje Windows traži (16 do 256 px,
svaka crtana za svoju veličinu, jer smanjivanje 256 → 16 zamuti tanak okvir).
Boje uzima iz `:root` promenljivih teme; ako neka nedostaje, **pada sa jasnom
porukom** umesto da se vrati na rezervnu vrednost. Prepisana boja se raziđe sa
temom pri prvoj promeni i niko to ne primeti, jer ikonica izgleda uredno sama
za sebe. Istu datoteku serviraj i kao ikonicu u kartici pregledača, na
`/favicon.ico` i kroz `<link rel="icon">` sa verzijom u putanji — bez toga
pregledač dobija 404 i prikazuje svoju podrazumevanu sliku, što takođe niko
ne prijavljuje.

> **ZAMKA 14 — `pythonw` nema standardni izlaz, i puca pre prvog sopstvenog
> reda koda.** U režimu bez terminala `sys.stdout` je `None`, a
> `uvicorn.Config` zove `sys.stdout.isatty()` — izuzetak iz tuđeg koda, pre
> nego što se izvrši ijedan tvoj red. Rešenje: `bootstrap.py` koji se
> izvršava **prvi**, podmeće izlaz u fajl i upisuje svaki neuhvaćen izuzetak
> sa punim tragom. Bez toga prečica na radnoj površini ne radi, a nigde se ne
> vidi zašto.

> **ZAMKA 15 — provera „jedna instanca" ne sme gledati samo otvoren port.**
> Port 8000 je bila zauzela tuđa aplikacija, pa je Subtitis mislio da je već
> pokrenut i otvarao pregledač na tuđem sajtu. Dodaj `/api/identity` i proveri
> **identitet**, ne otvoren port. Podrazumevani port pomeri na 8010.

Čišćenje diska:

| Mera | Vrednost |
|---|---|
| ograničenje otpremanja | 2 GB (nedovršeno se briše) |
| rok brisanja radne kopije zvuka | 7 dana |
| čišćenje | pri pokretanju **i** na svakih 6 sati |

Čišćenje mora biti periodično a ne samo pri pokretanju — aplikacija ume da
radi nedeljama bez restarta, pa bi „pri pokretanju" značilo nikad. Prikaži
zauzeće po poslu i ukupno. Zasebno dugme za brisanje samo radne kopije zvuka
nije potrebno — rok ga ionako briše, a ceo posao se briše jednim dugmetom.

Rezervne kopije pre svake izmene rečenica, sa vremenskom oznakom **i
brojačem** — puna sekunda je preniska rezolucija, dva snimanja u istoj
sekundi su davala isto ime i drugo je tiho pregazilo prvo. Sortiraj po
vremenu i brojaču, ne po imenu.

Tri sloja zaštite u editoru: poništavanje u toku rada, povratak na poslednje
snimljeno stanje, lista svih ranijih verzija sa vremenom i brojem rečenica.
Povratak prvo snima tekuće stanje, pa i on može da se poništi.

`.gitignore` napiši **bez inline komentara** — `import/ # komentar` ne radi i
audio je ušao u repo. Izuzmi `jobs/`, `import/`, `venv/`, modele. Zaključaj
verzije zavisnosti (`requirements.lock.txt`) u trenutku kada je kombinacija
potvrđeno radila: sprega između `faster-whisper` i CUDA biblioteka je osetljiva
na nadogradnje.

## Na kraju faze 5 — stani

---

# FAZA 6 — prevod postojećeg SRT-a sa očuvanim tajmingom

Drugi način upotrebe: umesto zvuka, ulaz je gotov titl. Povod je konkretan —
titl na jeziku originala često nastane izvan alata, ručno doteran u montaži,
sa vremenima podešenim uz sliku. Kada za taj titl zatreba prevod, sve što je u
njemu urađeno mora ostati netaknuto.

**Tvrda garancija: broj cue-ova i sva vremena u izlazu identični su ulazu.**
Ne menja se ništa osim teksta. Provera se izvodi **pre upisa** — ako se
tajming razlikuje, titl se ne zapisuje. Napiši i test koji poredi ulazni i
izlazni fajl red po red.

## Tri koraka

1. **Sastavljanje.** Rečenice se sklapaju iz cue-ova po interpunkciji.
   Rečenica se često prostire kroz više cue-ova, a jedan cue ume da sadrži
   kraj jedne i početak druge.
2. **Prevod.** Prevode se cele rečenice, sa kontekstom, kao i inače.
3. **Raspoređivanje nazad.** Prevedena rečenica se deli na onoliko delova
   koliko je original imao cue-ova, sa istim udelom znakova po delu.

Tačke deljenja bira **isto skorovanje koje radi prelamanje** — iste kazne za
funkcijske reči, iste tabele po jeziku. Razlika je samo u tome što su granice
zadate a ne predmet optimizacije, pa dodaj kaznu za odstupanje od ciljnog
udela (`PENALTY_SHARE_DEVIATION = 4`).

`MAX_SENTENCE_DURATION = 15.0` važi i ovde: jedan nedosledno interpunktiran
ulaz je dao rečenicu od 15,96 s koja je sadržala tri odvojene misli. Pauze
između cue-ova kao dopunski signal granice su **merene i odbačene** (pragovi
1,0 / 1,5 / 2,0 s): preko onoga što trajanje već hvata dodavale su tačno jedan
presek, i taj je bio pogrešan. Mehanizam ostavi u kodu, isključen, sa upisanim
merenjem (`SRT_MAX_GAP = None`).

## Profil se izvlači iz ulaza, ne nameće se

> **ZAMKA 16 — merenje prevoda standardnim profilom daje besmislene brojke.**
> Prva verzija je prijavljivala 47 prekoračenja od 204 cue-a. Merenje na samom
> ulazu je pokazalo zašto: ručno doteran original ima najduži red 51 znak (26
> redova od 219 preko 42) i najveći CPS 27,9 (18 cue-ova od 204 preko 20).
> Autorov sopstveni titl, koji je prihvatio i isporučio, probija standardni
> profil na 44 mesta. Mereno tim profilom, prevod se poredi sa normom koju ni
> original ne poštuje.
>
> Uvedi **profil izveden iz ulaznog fajla** kao podrazumevani za ovaj režim.
> Vrednosti sa **95. percentila, ne sa maksimuma** — maksimum pretvara jedan
> izuzetan red u normu:
>
> | Profil | redova preko 42 zn. | prekoračenja | od toga gore od originala |
> |---|---|---|---|
> | original (ulaz) | 12% | — | — |
> | netflix | 0% | 47 | 44 |
> | maksimum (51 / 27,9) | 17% | 8 | 8 |
> | **95. percentil (47 / 21,4)** | **11%** | **35** | **33** |

Izveštaj o prekoračenjima meri ono što stvarno treba: **koliko je cue-ova teže
za čitanje nego isti cue u originalu.** Uz svaki prijavljeni cue ispiši i
vrednost koju je imao pre prevoda.

Na dva posla izmereno isto: **sva prekoračenja su brzina čitanja, nijedno
dužina reda.** To nije slučajnost nego posledica postavke — vremena su fiksna,
a ciljni jezik je duži od srpskog za 15 do 20 procenata. Prelamanje to ne može
rešiti; ono samo raspoređuje tekst, ne skraćuje ga.

## Skraćivanje prekoračenja

Drugi prolaz kroz model, koji pokreće **dugme i nikad automatski**. Šalju se
samo rečenice čiji su cue-ovi gori od originala, uz izvorni tekst i granicu u
znakovima. Predlog se prihvata samo ako je stvarno kraći, ako je brzina
čitanja pala i ako nije izgubio termin iz glosara.

> **ZAMKA 17 — model seče više nego što se traži, i tu se gubi značenje.**
> Za potrebnih 1–2 znaka ume da skrati i 40. Tri formulacije zahteva su
> merene: „kraće za najmanje N" (2,73× traženog), opseg (2,36×), **apsolutna
> granica „najviše X znakova" (2,15×, uzeta)**. Čitanjem svih prihvaćenih
> predloga: oko tri četvrtine je čisto (izbačene poštapalice i ponavljanja),
> oko desetine menja značenje, ostatak gubi sadržaj — na primer šalu koja je
> bila poenta rečenice. Zato **ništa se ne upisuje bez pregleda**, a predlozi
> skraćeni znatno više od traženog stoje bez kvačice i moraju se uključiti
> ručno.

Izmereno: francuski 47 pokušano / 36 prihvaćeno / 38 od 59 izašlo iz
prekoračenja / prosečno 27,6 znakova (26%); engleski 29 / 24 / 23 od 33 /
22,6 znakova (29%).

## Tipografija po jeziku

Izlaz prolazi kroz post-procesor specifičan za ciljni jezik. Francuski traži
nerazlomljivi razmak pre `? ! : ;` i unutar `« »` — bez toga titl izgleda
pogrešno svakom francuskom čitaocu, kao što bi nama izgledalo `reč ,`.
Razmaci se računaju u dužinu reda **pre** prelamanja, da ne probiju profil
naknadno.

> **ZAMKA 18 — navodnici se raspadaju pri raspoređivanju, ne u modelu.**
> Prevod je izašao sa `"Ringe ringe raja" "` u jednom cue-u i `Ringe raja sa ""` u
> sledećem. Model je vratio uredan tekst. Tokenizer odvaja pravi navodnik od
> reči i ne zna otvara li ili zatvara citat, pa je rez smeo da padne između
> navodnika i reči. Popravka: **pre raspoređivanja** svaki pravi navodnik
> postaje usmereni oblik ciljnog jezika, a oni se lepe za reč i ne mogu se
> razdvojiti.

## Šta ovaj režim ne može — reci korisniku, ne pogađaj

- Nema slušanja ni bojenja po sigurnosti (nema zvuka ni verovatnoća po
  rečima). Crvene su samo rečenice čiji cue probija profil i gori je nego u
  originalu.
- **Kurziv preko dela rečenice se gubi.** Posle prevoda se ne može pouzdano
  vratiti na isto mesto, pa cue **prijavi** umesto da pogađaš. Cue koji je ceo
  u kurzivu ostaje u kurzivu.
- **Nedosledna interpunkcija u ulazu pravi greške.** Cue `Ja sam Marko` bez
  tačke, praćen cue-om `Imam 12 godina`, daje rečenicu u kojoj model čita
  `Imam` kao prezime: `Je suis Marko Imam`. Rešenje je u ulazu, ne u
  alatu — ali prijavi sumnju.

---

# FAZA 7 — objašnjenja u interfejsu

Alat koji si napravio biće jasan tebi, jer si svaku odluku sam doneo. Nekome ko ga
prvi put otvori nijedna od njih nema vidljiv razlog.

Dodaj objašnjenja u **dva sloja**, nikad kao zid teksta:

- **sloj 1** — jedan red ispod oznake polja, uvek vidljiv, kaže *šta da izaberem*
- **sloj 2** — `zašto ovo`, razvija se na klik, zatvoreno po default-u, sadrži razlog
  i izmerene brojeve

Jedna komponenta za oba sloja, bez JavaScripta (`<details>` i `<summary>` su
dovoljni). Duže celine — upotreba editora, garancija tajminga u fazi 6, ograničenja
tog režima — stoje kao trake koje se razvijaju iznad sadržaja na koji se odnose.
Dodaj i zasebnu stranicu *Kako ovo radi* sa oba toka u koracima i tabelom čestih
problema.

Kad ovo radiš, **izbaci stare jednoredne napomene** razbacane po ekranima — nova
objašnjenja ih pokrivaju potpunije i na jednom mestu. Ostavi samo oznake koje se
odnose na konkretan red.

> **ZAMKA 19 — tekst objašnjenja pisan prema dokumentaciji ne slaže se sa kodom.**
> Od dvadeset pripremljenih tekstova, devet je odstupalo od stvarnog stanja: dugme
> koje je opisano a nikad napravljeno, pogrešno najveće trajanje profila, boja
> oznake opisana kao narandžasta iako je žuta, polja smeštena na pogrešan ekran,
> tvrdnja o potvrdi koja postoji samo u komandnoj liniji. **Svaki broj i svaku
> tvrdnju u objašnjenju proveri u izvoru pre ugradnje**, i gradi tabele iz stvarnih
> datoteka (profili, podešavanja) umesto da ih prepisuješ — tako ne mogu zastariti.
> Broj koji ne možeš proveriti iz koda nemoj prepisati: izmeri ga ili ga izostavi.

---

# FAZA 8 — interfejs iz temelja

Prvi interfejs će biti tabela polja bez hijerarhije i to je u redu dok ga koristiš
sam. Kad ga prvi put pogleda neko drugi, prepravka traje dugo ako se radi mesto po
mesto. **Postavi četiri pravila odmah i napiši test za svako:**

1. **Objašnjenje je ikonica i prozorčić, nikad tekst na stranici.** Nikakve trake
   koje se razvijaju u toku sadržaja — one guraju sve naniže i gube ti mesto.
   Prozorčić lebdi, zatvara se klikom van, ponovnim klikom i `Esc`, najviše jedan
   otvoren. Test pada ako negde ostane traka sa objašnjenjem.
2. **Ispod jedne trake najviše jedan red teksta.** Kad svako polje nosi svoje
   objašnjenje, ispod trake od četiri polja stoje četiri pasusa i ništa se ne čita.
3. **Radnja ne pomera pogled.** Isti skrol, isto fokusirano polje, iste otvorene
   trake posle svake radnje.
4. **Dugme kaže šta radi**, nikad „OK". Potvrde kroz sopstveni prozor; nativni nosi
   adresu servera u naslovu i dugmad na engleskom.

> **ZAMKA 25 — pravilo se vraća dok se prijavljuje kao mesto.** Ista stavka —
> objašnjenje kao traka umesto ikonice — popravljana je pet puta zaredom, jer je
> svaki put pokazana na jednom ekranu. Prestala je tek kad je formulisana kao
> pravilo sa obavezom da se prođe kroz **svaki** šablon i sa testom koji pada ako
> se vrati. Isti test je odmah uhvatio dva mesta koja niko nije prijavio.

## Prelom se mora videti u editoru

Ovo je najskuplje nesporazumljenje u celom projektu i rešava se prikazom, ne
promenom modela.

Editor uređuje **rečenice**, a profil, pismo i veličina slova menjaju **titl**.
Tekst u editoru ostaje isti, pa izgleda da izbor ne radi — prijavljeno tri puta
kao kvar, a svaki put je funkcija radila ispravno.

**Ispod svake rečenice prikaži cue-ove koji su iz nje nastali**, tačno kako stoje u
datoteci: prelom redova, pismo, veličina slova, interpunkcija, vreme svakog cue-a.
Monospace, sitnije, prigušeno. Cue koji probija profil ili je prekratak obeleži
bojom i tu. Prekidač za isključivanje, jer na dugom prepisu udvostručuje visinu
liste.

## Nijedna izmena ne čeka na dugme

Prelamanje je determinističko, ne poziva model i traje deo sekunde (na satu prepisa
od 1683 rečenice oko 0,9 s). Nema razloga da ga pokreće dugme:

- promena bilo kog izbora prelama odmah, uz odlaganje od 300 ms da dve brze
  promene daju jedno prelamanje;
- izlazak iz polja sa izmenjenim tekstom snima i prelama;
- osvežava se **samo deo sa prelomom**, ne cela stranica — inače se vidi kako se
  vraća stari pa posle sekunde dolazi novi;
- stanje snimanja nosi ikonica, ne rečenica na stranici.

Kad to uradiš, potvrda „titl ponovo prelomljen" postaje suvišna — promena se vidi u
samom prelomu. Zadrži je samo za grešku.

---

# Testovi

Ciljaj na blizu hiljadu testova i **najmanje 85% pokrivenosti naredbi**, sa
težištem na modulima gde je greška tiha: prelamanje, transliteracija,
segmentacija, ekavizacija, spajanje elizija, validacija odgovora modela,
očuvanje tajminga u fazi 6.

> **ZAMKA 26 — broj testova nije pokrivenost, i razlika ume da bude ogromna.**
> Pri 1140 testova pokrivenost je prvi put izmerena: glavna faza pajplajna —
> funkcija koja se najviše menjala — bila je na **1%**. Nijedan test nije
> prolazio kroz nju, jer traži model i karticu, pa je svaka izmena u njoj
> mogla da prođe nezapaženo do prvog stvarnog posla.
> **Meri pokrivenost od prve faze**, ne na kraju, i prijavljuj je po modulu.

Za fazu koju nijedan test ne dodiruje, recept je isti i ne traži menjanje
koda: napiši test koji **podmeće samo spoljne stvari** — pripremu zvuka,
merenje govora, prepoznavanje i jezički model — a sve ostalo pušta kroz pravi
kod i proverava **ishode**, ne pozive: nastale datoteke, broj reči i rečenica,
oznake, faze i redove u dnevniku. Pripremi više ulaza, da svaki bira drugu
granu: jedan koji pali ekavizaciju i kapiju interpunkcije, jedan sa
razdvojenim elizijama i gustom interpunkcijom gde kapija ne okida, i jedan sa
repom reči nultog trajanja. Test koji samo gleda da je funkcija pozvana
prolazi i kad je rezultat pogrešan — takav ne piši.

Ne deli dugačku funkciju da bi je testirao. Dužina nije rizik; nepokrivenost
jeste. Deljenje bi premestilo isti sadržaj u druge fajlove i tražilo objekat
stanja kroz korake, a pokrivenost bi ostala ista.

Nekoliko testova su **namenske brane** koje pamte skupo plaćene nalaze.
Najvažniji proverava da `hallucination_silence_threshold` nije vraćen u
podrazumevane parametre, sa objašnjenjem u samom testu da je taj parametar
gutao rep zapisa i koštao 28% pokrivenosti.

> **ZAMKA 24 — zaštita nabrajanjem propušta ono što još nije napisano.**
> Test koji je samo podigao server pokrenuo je migraciju nad **stvarnim** podacima,
> pre nego što je napravljena rezervna kopija. Zaštita koja je iz toga nastala
> nabrajala je ono što je stradalo — direktorijum poslova i podešavanja — pa se tri
> kruga kasnije isti kvar ponovio na glosaru.
> **Okreni pravilo naopako:** svaki test dobija privremen koren projekta, a pisanje
> bilo gde izvan njega je greška, ne izuzetak. Jedan `conftest.py` koji to podmeće
> svakom testu bez izuzetka, i provera koja pada ako je išta u stvarnom
> repozitorijumu dirnuto tokom celog skupa. Tada ne moraš da pamtiš koji je fajl
> osetljiv. Svaki nov kod koji radi pri pokretanju zamisli kao nešto što će test
> pokrenuti.

Kuriozitet koji je pojeo pola sata: generator testova je upisao **stvarni
upravljački bajt** umesto oznake granice reči u obrazac za pretragu, pa su
tri testa padala bez vidljivog razloga. Nađeno prikazom nevidljivih znakova.

---

# Kako da mi javljaš stanje

- Posle svake faze: šta je izmereno, konkretni brojevi, šta je ostalo
  otvoreno.
- Kada nađeš da moja pretpostavka ne stoji: **reci pre nego što uradiš**, sa
  merenjem koje to pokazuje.
- Komanda za zaustavljanje procesa mora biti **filtrirana na ovaj projekat** —
  komanda koja gasi sve Python procese mi je jednom prekinula drugi projekat:

```powershell
Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
  Where-Object { $_.CommandLine -like '*subtitis*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Počni od faze 1.
