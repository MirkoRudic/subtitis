# Subtitis

Subtitis pravi SRT titlove iz audio i video snimaka i prevodi ih: govor
prepoznaje faster-whisper (`large-v3`), redove prelama po pravilima za
srpski, engleski i francuski, a prevodi lokalni jezički model kroz Ollama.
Namenjen je malim timovima i organizacijama koje titluju sopstvene snimke
(izjave, radionice, kratke dokumentarce) i rade na Windows-u. Minimum
hardvera: **NVIDIA grafička kartica sa 12 GB VRAM-a** — Whisper `large-v3` i
prevodni model (npr. `qwen3:14b`, ~10 GB) ne staju na manju; bez nje alat
nije upotrebljiv.

Svi primeri u kodu, testovima i dokumentaciji su **izmišljeni** — imena,
organizacije, projekti i rečenice. Alat radi isključivo lokalno: zvuk, prepisi
i prevodi ne napuštaju računar. Da tako i ostane, proverava `tests/test_javni_repo.py`:
pada na svaki pojam sa spiska zabranjenih i na svaku četvorku uzastopnih reči
iz stvarnih snimaka na kojima je alat razvijan (oba spiska su samo heševi).
Spisak četvorki dopunjuje `scripts/izvorne_cetvorke.py` iz `jobs/` radnog foldera (van gita); pokreni ga posle svakog novog posla — stari heševi ostaju.

Licenca: MIT, vidi `LICENSE`.

---

Interna alatka: SRT titlovi iz audio fajlova, pa prevod.

**Status: Faze 1–3 (ASR pajplajn, prevod, dashboard).**

Pokretanje za nekog ko ne otvara terminal: dvoklik na `subtitis.bat`.

---

## Osnovni princip

`words.<lang>.json` je izvor istine. Sve ostalo je izvedeno i može se regenerisati.

- Promena prelomnog profila → `subtitis build`, **ne** ponovni ASR
- Dodatni ciljni jezik → prevod, **ne** ponovni ASR
- Ručna korekcija → pajplajn je nikad ne gazi automatski (`edited_by_human`)

ASR i LLM nikad ne stoje u VRAM-u istovremeno. Whisper se oslobađa čim
transkripcija završi (`asr.unload_model()`).

---

## Instalacija

Traži se **Python 3.11** (ne 3.12+ — `ctranslate2` nema wheel-ove) i **ffmpeg u PATH**.

```powershell
winget install Python.Python.3.11
winget install Gyan.FFmpeg          # pa OTVORI NOVI TERMINAL, PATH se osvežava

py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m subtitis doctor
```

`requirements.lock.txt` je zamrznuta kombinacija koja je provereno radila
(ctranslate2 + `nvidia-*` paketi su osetljivi jedni na druge). Kad nešto pukne
posle update-a, vrati se na njega: `pip install -r requirements.lock.txt`.

`doctor` mora da ispiše putanju do ffmpeg-a i `faster-whisper : ok`.

### ffmpeg se traži i van PATH-a

VS Code (i svaki dugo otvoren program) drži **kopiju PATH-a od trenutka svog
pokretanja**, pa terminal otvoren iz njega ne vidi ffmpeg instaliran u
međuvremenu. Poruka „nije u PATH-u" je tada tačna a beskorisna — program jeste
instaliran.

Redosled traženja: **konfiguracija → PATH → poznate lokacije**
(`%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\**\bin`, chocolatey,
scoop, `C:\ffmpeg`). Konfiguracija je `subtitis.toml` u korenu ili
`SUBTITIS_FFMPEG`:

```toml
ffmpeg_path = "C:\\Users\\...\\Gyan.FFmpeg...\\bin\\ffmpeg.exe"
```

`doctor` prijavljuje odakle je ffmpeg došao i, ako je nađen van PATH-a,
ispisuje gotov red za `subtitis.toml`:

```
ffmpeg          : C:\...\ffmpeg-9.0-full_build\bin\ffmpeg.exe   [poznata lokacija]
  Nađen VAN PATH-a. … upiši putanju u D:\VAJB\subtitis\subtitis.toml:
      ffmpeg_path = "C:\\...\\bin\\ffmpeg.exe"
```

> **cuBLAS/cuDNN na Windows-u.** `ctranslate2` učitava `cublas64_12.dll` običnim
> `LoadLibrary`, pa `os.add_dll_directory` nije dovoljno — `asr.py` sam gura
> `site-packages/nvidia/*/bin` u PATH procesa. Ako ipak dobiješ
> `Library cublas64_12.dll is not found`, proveri da su instalirani
> `nvidia-cublas-cu12` i `nvidia-cudnn-cu12`.

---

## Upotreba

```powershell
.venv\Scripts\python -m subtitis transcribe snimak.mp3 --lang sr
.venv\Scripts\python -m subtitis build <job_id> --profile tv --script Latn
.venv\Scripts\python -m subtitis build <job_id> --profile tv --script Cyrl
.venv\Scripts\python -m subtitis translate <job_id> --to en
.venv\Scripts\python -m subtitis build <job_id> --profile tv --lang en
.venv\Scripts\python -m subtitis info <job_id>
```

**Claude desktop (MCP):** Claude u chatu sam pokreće dashboard, poslove, testove i merne skripte i čita prepis — bez ručnog koraka.
Uključuje se jednom, `.venv\Scripts\python -m subtitis mcp-install` (upiše unos u konfiguraciju Claude aplikacije; `mcp-uninstall` ga skida), pa restart Claude aplikacije.

Korisne zastavice za `transcribe`: `--job-id`, `--model`, `--device cpu`,
`--compute-type int8`, `--beam-size`, `--glossary`, `--force`, `--no-vad`.

`transcribe` na kraju ispisuje **pokrivenost** — procenat trajanja audija
pokriven rečenicama — i spisak praznina dužih od 3 s sa tačnim vremenima. Ako u
praznini ima govora, VAD ga je pojeo pre modela; uporedi isti fajl sa `--no-vad`.

Bench na više fajlova (model se učita jednom):

```powershell
.venv\Scripts\python scripts\bench.py snimci\*.mp3 --lang sr
```

Ispisuje trajanje audija, vreme obrade, RTF, broj reči, procenat reči sa
`p < 0.6` i kolonu **`petlja`** — najduži niz uzastopnih identičnih ili gotovo
identičnih rečenica. Preko 3 je halucinacija, ne stil govornika; takvi redovi
se označavaju sa `!` i ispisuju u zasebnom bloku sa primerom teksta.

Ima i kolonu **`pokriv`** plus, ispod tabele, blok `PRAZNINE > 3s` sa tačnim
vremenima po fajlu. `--no-vad` radi i ovde, za poređenje na celom skupu.

`bench.py` sam širi globove (`glob.glob` + fallback na običnu putanju) — PowerShell
ih ne širi kao bash, `D:\snimci\*.mp3` stiže kao bukvalan string.

Štelovanje VAD-a bez čekanja Whispera (traje sekunde):

```powershell
.venv\Scripts\python scripts\vad_probe.py snimak.mp3
.venv\Scripts\python scripts\vad_probe.py snimak.mp3 --check 54.18 61.60
.venv\Scripts\python scripts\vad_probe.py snimak.mp3 --thresholds 0.2,0.3,0.35,0.5
```

`--check START END` je za interval za koji si uvom utvrdio da ima govora:
kolona pokazuje koliki deo tog intervala je svaki prag zadržao.

Ako VAD nije krivac, red je na Whispera:

```powershell
.venv\Scripts\python scripts\asr_probe.py snimak.mp3 --lang sr --window 79 90
.venv\Scripts\python scripts\asr_probe.py snimak.mp3 --lang sr --compare
```

Prvi ispisuje svaki segment sa `no_speech_prob`, `avg_logprob`,
`compression_ratio`, temperaturom na kojoj je prošao, skorom anomalije i
početkom teksta. Drugi pušta šest konfiguracija na istom fajlu i poredi dokle
svaka stigne — jedini način da se otkrije filter halucinacija (vidi ispod).

---

## Struktura posla

```
jobs/<job_id>/
  job.json            # status, parametri, istorija faza sa trajanjima
  log.txt             # svaka faza, model, retry pokušaji
  source.wav          # 16 kHz mono PCM s16le
  pauses.json         # tišine iz zvuka (Silero VAD); ostaje i kad se source.wav obriše
  words.sr.json       # IZVOR ISTINE
  sentences.sr.json
  out/<ime>.sr-Latn.srt
```

`pauses.json` (krug 19) nastaje pri prepoznavanju govora: razmak između reči
u `words.json` za pauzu ne služi — Whisper tišinu upija u susednu reč, pa je na
granici bez interpunkcije tačno 0 u 825 od 830 slučajeva. Fajl nosi čime je
napravljen (`made_with`: verzija, model i parametri VAD-a). Prelamanje ga zasad
ne koristi — izmereno i odbačeno (`linebreak.PAUSE_BREAK_DISCOUNT` i susedne
konstante, `merenja/dnevnik.md`); stariji poslovi ga nemaju i rade kao ranije.

`sentences.*.json` ima `edited_by_human`. Kad je `true`, `transcribe` odbija da
prepiše fajl bez `--force`, a i tada pravi `.bak` kopiju.

Ručna ispravka teksta rečenice preživi `build`: `linebreak.align_text_to_words`
poravna izmenjeni tekst sa originalnim tokenima i **zadrži prava ASR vremena**
za sve što nije menjano. Interpolira se samo za tokene koje je čovek dodao.

Poravnanje ne sme tiho da degradira. Kad udeo tokena koji su zadržali prava ASR
vremena padne ispod `MIN_ALIGNED_TOKEN_SHARE` (0.6), rečenica dobija flag
`alignment_degraded`, upozorenje ide u `log.txt`, a `build` ga ispisuje žuto sa
spiskom ID-jeva. Flag se upisuje nazad u `sentences.*.json` — to je dijagnostika,
tekst rečenica se ne dira.

`read_json` čita kao `utf-8-sig`: ove fajlove otvaraš ručno, a Notepad i
PowerShell umeju da ostave BOM.

---

## Moduli

| fajl | šta radi |
|---|---|
| `audio.py` | ffmpeg → WAV 16 kHz mono s16le, vraća trajanje |
| `asr.py` | faster-whisper large-v3, `word_timestamps`, `vad_filter`, normalizacija pisma |
| `segment.py` | reči → rečenice; skraćenice, redni brojevi, lomljenje preko 15 s |
| `linebreak.py` | rečenica → cue-ovi; DP nad kaznenim poenima, bez LLM-a |
| `translate.py` | Ollama klijent, prompt, validacija, pad na manji čank |
| `ekavica.py` | ijekavica → ekavica, kurirani rečnik oblika |
| `translit.py` | latinica ↔ ćirilica, sa izuzecima za `lj`/`nj`/`dž` |
| `srt.py` | SRT: UTF-8 bez BOM-a, CRLF, BCP-47 imenovanje |
| `models.py` | pydantic sheme svih artefakata |
| `config.py` | putanje, profili, glosar, log, merenje faza |

### Prelamanje

Vremena cue-a se izvode iz `s` prve i `e` poslednje reči — trajanje se nikad ne
deli proporcionalno broju znakova.

Svi kazneni poeni su imenovane konstante na vrhu `linebreak.py`. Pored tabele iz
specifikacije dodate su tri koje su se pokazale kao nužne (svaka je označena u
kodu):

- `PENALTY_LINE_IMBALANCE` — tabela kažnjava samo „gornji red duži od donjeg”,
  pa bi bez ovoga „Danas ćemo / pričati o jednoj važnoj temi” bilo jeftinije od
  uravnoteženog preloma.
- `PENALTY_SHORT_CUE` + `MIN_CUE_CHARS` — bez ovoga DP otcepi „It was” u
  zaseban cue da bi ostatak stao u dva reda. Ne važi kad je cue cela rečenica,
  pa „Da.” i dalje prolazi.
- `PENALTY_OVER_MAX_DURATION` — inače nema razloga da lomi duge pauze.

### Zaštita od petlji ponavljanja

`large-v3` na dužim snimcima ulazi u petlje i halucinira tekst preko tišine.
Krivac je `condition_on_previous_text=True` (Whisperov default): model svoj izlaz
koristi kao kontekst, pa kad jednom sklizne, ostaje zaglavljen do kraja fajla.
Konstante su na vrhu [asr.py](subtitis/asr.py):

| konstanta | vrednost |
|---|---|
| `CONDITION_ON_PREVIOUS_TEXT` | `False` |
| `COMPRESSION_RATIO_THRESHOLD` | `2.4` |
| `LOG_PROB_THRESHOLD` | `-1.0` |
| `NO_SPEECH_THRESHOLD` | `0.6` |
| `TEMPERATURE_FALLBACK` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` |
| `HALLUCINATION_SILENCE_THRESHOLD` | `None` — **isključeno namerno**, vidi ispod |

### Šta Whisper odbacuje i šta se od toga vidi

faster-whisper **ne izlaže odbačene segmente kroz API** — ni povratnom
vrednošću, ni na `Segment` objektima. Deo toga loguje na DEBUG (logger
`faster_whisper`), i to `asr.RejectionCapture` hvata i upisuje u
`words.json` pod `asr.rejections`:

| ključ | šta znači | gubi audio? |
|---|---|---|
| `no_speech_dropped` | ceo prozor od 30 s preskočen po `no_speech_threshold` | **da** |
| `compression_ratio_fallback` | ponovni pokušaj na višoj temperaturi | ne |
| `log_prob_fallback` | ponovni pokušaj na višoj temperaturi | ne |

### Filter halucinacija je isključen — namerno

`HALLUCINATION_SILENCE_THRESHOLD = None`. **Ne uključuj ga bez merenja pokrivenosti.**

Sa vrednošću 2.0 je na testnom snimku gutao ceo rep fajla:

| | pokrivenost | reči | rečenice staju na |
|---|---|---|---|
| `2.0` | 55.9% | 155 | 79.74s od 90s |
| `None` | **83.8%** | **173** | rep se vratio |

Uzrok je u `faster_whisper/transcribe.py`, grana `if silence_before and
silence_after:` — radi `current_segments[si:] = []`, a kad je sporni segment
blizu kraja sadržaja i `seek = content_frames`, dakle **preskok do kraja fajla**.
Segment se proglasi sumnjivim ako mu prvih 8 reči skupe skor ≥ 3 **ili**
skor + 0.01 ≥ broj reči (reč sa verovatnoćom < 0.15, kraća od 133 ms ili duža
od 2 s). Kratak i tih odgovor to ispuni bez muke.

Ta grana **ne loguje ništa**, pa se gubitak ne vidi ni u `asr.rejections` —
otkriva ga samo pad pokrivenosti ili `scripts/asr_probe.py <fajl> --compare`.

Protiv petlji ponavljanja glavna odbrana je `CONDITION_ON_PREVIOUS_TEXT=False`,
uz `COMPRESSION_RATIO_THRESHOLD` i `TEMPERATURE_FALLBACK`; ovaj filter za to
nije potreban. Ako ti ipak zatreba na drugom materijalu:
`subtitis transcribe … --hallucination-filter` (koristi
`HALLUCINATION_SILENCE_SECONDS = 2.0`) — ispis će te upozoriti da proveriš
pokrivenost.

Pragovi se mogu menjati bez diranja koda:

```powershell
subtitis transcribe snimak.mp3 --lang sr --no-speech-threshold off
subtitis transcribe snimak.mp3 --lang sr --log-prob-threshold -2.0
subtitis transcribe snimak.mp3 --lang sr --hallucination-filter   # opt-in
```

### VAD

Silero VAD sa podrazumevanim `threshold=0.5` jede tiši govor, a sve što odbaci
nikad ne stigne do modela — gubitak se ne vidi ni u jednoj drugoj metrici osim
pokrivenosti. Konstante su na vrhu [asr.py](subtitis/asr.py):

| konstanta | vrednost | silero default |
|---|---|---|
| `VAD_THRESHOLD` | `0.35` | 0.5 |
| `VAD_MIN_SILENCE_DURATION_MS` | `700` | 2000 |
| `VAD_SPEECH_PAD_MS` | `400` | 400 |

`--no-vad` isključuje VAD u potpunosti: ništa ne može biti odbačeno pre modela,
po cenu veće sklonosti halucinacijama na tišini. `--vad-threshold` menja prag za
jedan posao.

**Na dashboardu su tri imenovana izbora, ne broj.** Vrednost `0.35` korisniku ne
znači ništa dok ne pusti ceo prepis i ne prebroji šta fali; naziv je odluka koja
se može doneti unapred:

| izbor | prag | kad |
|---|---|---|
| normalno | `0.35` | snimci sa jasnim govorom |
| tih snimak | `0.20` | zadržava tiši govor, propušta više šuma |
| isključeno | — | ništa se ne odbacuje pre modela; sporije |

Preslikavanje je u `asr.VAD_PRESETS`; dashboard šalje naziv, ne broj.

> Niži `min_silence_duration_ms` sam po sebi seče odsečke ranije, ali uz
> `speech_pad_ms=400` sa obe strane pauze do ~800 ms ne gube ništa. Dobitak
> dolazi od nižeg praga.

### Glosar i `hotwords`

> **Glosar ide i u `hotwords`, ne samo u `initial_prompt`.** Uz
> `condition_on_previous_text=False` prompt se resetuje posle svakog segmenta, pa
> `initial_prompt` važi samo za prvi prozor od 30 s. `hotwords` se ubacuje u svaki
> prozor (`faster_whisper/transcribe.py`, `get_prompt`), pa glosar preživi ceo fajl.

Svi ASR parametri se upisuju u `words.json` pod `asr.params` — vidi se čime je
tačno napravljen svaki fajl.

**Cena toga:** gde model nema šta da čuje (muzika, tišina), vrati baš ono što mu
je dato. Snimak sa džinglom (Radionica draft 1, 2026-10-01) dobio je poslednjih 7 s prepisa
iz glosara prethodnog projekta. Zato **svaki posao ima svoj glosar**
(`jobs/<id>/glossary.json`): na otpremanju kopija podrazumevanog ili prazan, plus
termini tog snimka (samo za prepoznavanje, ključ `<jezik>` umesto `sr->en`).
Prepoznavanje, prevod, prelamanje i skraćivanje čitaju glosar posla; stari
poslovi bez tog fajla čitaju podrazumevani. Merenje (VAD × glosar, po pet
ponavljanja) je u `merenja/dnevnik.md`: izvor je glosar; VAD + `hotwords` daje
nepredvidljiv prepis (od 11 do 52 reči istog fajla). VAD sam nije bolji — po
tačnosti govora izjednačen je sa isključenim, samo ne pušta džingl do modela,
a na tihom snimku bez muzike gubi govor. Zato ostaje isključen, uz savet po
materijalu.

`glossary/default.json` u repou je izmišljen primer. Radna mašina preko njega drži stvarni glosar,
označen sa `git update-index --skip-worktree glossary/default.json` — git ne vidi izmenu i ne može da je
komituje. Vraća se sa `git update-index --no-skip-worktree glossary/default.json`.

### Reči bez zvuka iza sebe

Izmišljene reči umeju da imaju verovatnoću 0,98–0,99, pa ih `low_confidence` ne
vidi. Izdaje ih vreme: model ih vrati na istom trenutku, sa trajanjem nula.
Niz od 2+ uzastopnih reči kraćih od 0,02 s (interpunkcija providna,
`hallucination.py`) upisuje se u dnevnik sa vremenima i tekstom, rečenica dobija
oznaku `hallucinated` (zasebnu od `low_confidence`), a u editoru su te reči
crvene, uz napomenu. Izmereno na šest urednih snimaka + Radionici draft 1: nijedna lažna
uzbuna, pogodci su glosar (Radionica draft 1), `titrage ST' 501` (snimak A) i `na svijetu` iz
`Top 10 najboljih uvijeka na svijetu`. Pravilo **ne** hvata izmišljen tekst sa
razvučenim vremenima (Radionica draft 1: `Otvorena vrata` preko džingla, 2,34 s po reči).

Kad je detekcija govora isključena a negovora ima 25%+ snimka, stranica posla
to kaže i izričito pominje muziku, džingl i špicu — čest slučaj u montiranom
materijalu (ranije samo red u `log.txt`, koji niko nije video).

Merenja: nalazi u `merenja/dnevnik.md`, skripte u `scripts/krug*_*.py`, sirovi
podaci u `merenja/krug*/` — van repozitorijuma, jer su sadržaj tuđih snimaka.

## Prevod (Faza 2)

```powershell
subtitis translate <job_id> --to en
subtitis translate <job_id> --to fr --pivot en      # sr -> en -> fr
subtitis build <job_id> --profile tv --lang en
```

Bez novih zavisnosti — klijent je `urllib` iz standardne biblioteke.

### Tri stvari bez kojih tiho ne radi

Sve tri su izmerene na ovoj mašini, ne pretpostavljene.

**1. `think: false`.** Qwen3 je hibridni model i po defaultu razmišlja.
Izmereno: **124.8 s naspram 17.0 s** za isti zadatak, 7.3×. U Ollami 0.32
rezonovanje ne ide u `<think>` tagove nego u zasebno polje `message.thinking`,
pa se proveravaju **oba**; ako se ipak pojavi, ide glasno upozorenje i broji se
u `thinking_responses`.

**2. `num_ctx: 8192`.** Ollamin default je 4096 i tiho odseca. Gore od toga —
Ollama **tiho prihvata i nepoznata polja** (podmetnuo sam `izmisljeno_polje`,
prošlo bez greške), pa se greška u imenu opcije ne vidi nigde. Zato posle prvog
poziva čitamo `/api/ps` i poredimo stvarni `context_length` sa traženim:

```
num_ctx potvrđen: 8192
```

Ako se ne poklapa, ide upozorenje sa uputstvom (`ollama stop <model>`), jer
Ollama zadrži kontekst sa kojim je model prvi put učitan.

**3. Validacija broja linija.** Pošalješ 10 numerisanih rečenica, model vrati 9
(spojio dve) ili 11 (razbio jednu) — i ceo ostatak fajla se pomeri. `validate()`
poredi **skup rednih brojeva**, ne samo količinu. Da ovo nije prazan strah: pri
prvom probnom pozivu sa nedovoljno strogim sistemskim promptom model je vratio
`2|` za obe linije.

**Linije se numerišu lokalno, 1..n**, a ne pravim id-jem rečenice; mapiranje
nazad drži kod. Sa trocifrenim id-jevima (odlomak 100..108) model je vraćao
1..9 — prenumerisao je sam — i svaki takav odgovor je odbijen i polovljen.
Provera broja linija ostaje ista: ona hvata spajanje i razdvajanje, što je
pravi rizik.

Na neuspeh se čank polovi (9 → 4/5 → 2 → 1). Posle
`MAX_ATTEMPTS_PER_SENTENCE` pokušaja na pojedinačnoj rečenici upisuje se
original sa `translation_failed` flagom i posao ide dalje. **Ceo posao se zbog
jedne rečenice nikad ne prekida.** Greška veze ili modela se ne polovi — puca
odmah i jasno.

Nema drugih pragova ni filtera. Ništa se ne odbacuje po „kvalitetu"; odbija
se samo strukturno neispravan odgovor: skup rednih brojeva se ne poklapa sa
poslatim, ili rečenica ima neuravnotežene navodnike a original ih nema.

### Prompt i glosar

Čankovi od `CHUNK_SIZE` rečenica, sa po `CONTEXT_BEFORE`/`CONTEXT_AFTER`
rečenica konteksta sa svake strane. Kontekst se šalje **nenumerisan**, pa ga
model ne može pomešati sa gradivom za prevod.

Isti glosar se koristi dvaput: ključevi idu u Whisperov `initial_prompt` i
`hotwords`, a ceo par u prevodni prompt. Unos tipa `"Đurđević": "Đurđević"` nije
greška — tako se beleže termini koje **ne treba** menjati.

Prazan glosar za dati smer daje upozorenje u `log.txt` umesto da tiho prođe.
`--show-prompt` u eval skripti ispisuje ceo sistemski prompt, pa se vidi šta je
model stvarno dobio.

**Glosar merljivo utiče.** A/B na 9 parova, isti model i seed:

| termin | bez glosara | sa glosarom |
|---|---|---|
| Radionica otvorenih vrata | Workshop of Open Doors | **Open Door Workshop** |
| animator | caregiver | **activity leader** |
| sporazumevati se | to reach an agreement | **understanding each other** |

### Pravila 11–13 i varijanta

Pravilo 12 nosi primere svakodnevnih reči **po ciljnom jeziku**
(`PLAIN_WORD_EXAMPLES`); engleski primeri u francuskom promptu bi bili šum.

**Mereno i odbačeno: pravilo o registru** (2026-09-28). Pravilo „ovo je
izgovoreni jezik, ne pisani tekst; biraj svakodnevne izraze; idiom prevodi
idiomom, nikad doslovno" — dodato kao 13, pa uklonjeno. Isti posao (Otvorena vrata 26),
isti seed, ponovljivost proverena (dva prolaza bez izmene: 0 od 195 razlika):

| | rečenica sa promenjenim rečima | ocena promena (fr, ručno) |
|---|---|---|
| fr | 30 od 195 (15%) | ~11 bolje, ~11 neutralno, ~9 gore |
| en | 37 od 195 (19%) | — |

Bolje: `Bonjour` → `Salut`, `Ceci est un appareil photographique` → `C'est un appareil photo`.
Gore: `huit paliers` → `huit pali` (pokvarena reč), `N'est-ce pas ?` →
`Pas ça ?` (greška već prijavljivana), `merveilleux enfants` → `très beaux
enfants`. Idiom zbog kog je pravilo traženo („vetar u leđa") ostao je kalk
(doslovan prevod obe reči); engleski je sa dobrog `a massive boost` prešao na
doslovniji `a big, big push`.

Pouka: „menja više od desetine rečenica" meri da li model **primećuje**
pravilo, ne da li je prevod **bolji** — odluka ide po ručnoj oceni promena.
Pravilo 12 („govorni jezik, svakodnevne reči") ostaje. Ne pokušavati ponovo
bez nove ideje (drugi model, primeri idioma u glosaru, drugi prolaz samo za
idiome).

Pravilo 13 se dodaje samo ako je varijanta zadata i poznata
(`VARIANT_RULES`: `en-GB`, `en-US`). Nepoznata varijanta ne dodaje ništa i
prijavljuje se u eval skripti.

Varijanta dolazi iz `variant` u `pairs.json` (eval) ili iz
`config.TARGET_VARIANT` (posao), a `--variant` je nadjačava; `off` je isključuje.

### Pivot

`config.PIVOT_VIA` je **namerno prazan**. Za `sr->fr` je dvostruki prolaz preko
engleskog često bolji jer je `en<->fr` mnogo bolje pokriveno, ali to treba
izmeriti na svom materijalu, a ne pretpostaviti. Oba puta rade:

```powershell
subtitis translate <job_id> --to fr              # kako kaže PIVOT_VIA
subtitis translate <job_id> --to fr --pivot en   # nateraj pivot
subtitis translate <job_id> --to fr --pivot off  # nateraj direktno
```

Kod pivota se međukorak (`sentences.en.json`) čuva na disku — inspektabilan je
i ne mora da se ponavlja.

**Glosar na drugom skoku.** Za `sr -> en -> fr` drugi skok traži `en->fr` unose,
kojih u glosaru po pravilu nema — pa bi druga polovina puta išla bez rečnika.
`config.hop_glossary` ih **izvodi** iz `sr->en` i `sr->fr` gde postoje oba:

```
sr->en: "animator" -> "activity leader"
sr->fr: "animator" -> "éducateur spécialisé"
   =>   en->fr: "activity leader" -> "éducateur spécialisé"
```

Eksplicitni `en->fr` unosi imaju prednost nad izvedenima. Ako posle svega
rečnik za taj skok ostane prazan, ide žuto upozorenje sa objašnjenjem šta se
gubi i kako da se popravi — ne tiho.

### Vremena prevedenih titlova

Prevod nema vremena po rečima — zna se samo početak i kraj rečenice. Zato
`build` za prevedeni jezik raspoređuje vremena unutar rečenice **srazmerno
broju znakova** (`linebreak.synthesize_words`), i to izričito kaže u ispisu.
Granice rečenica ostaju prave, iz ASR-a. Za izvorni jezik se ovo nikad ne
koristi — tamo postoje prava vremena i koriste se.

### Prevod postojećeg SRT-a (tajming nepromenljiv)

Za ručno doteran titl kome treba isti titl na drugom jeziku: dashboard,
„Prevod postojećeg SRT-a". Cue-ovi i vremena izlaza su **identični** ulazu —
menja se samo tekst. Posao nema zvuka ni `words.json`; nastaje odmah kao
`transcribed`, a prevod ide u red jer je ciljni jezik izabran na otpremanju.

```
jobs/<job_id>/
  cues.json           # ulazni cue-ovi: vremena u ms, redovi sa <i>, {\an8}
  sentences.sr.json   # derived_from: cues.json; svaka rečenica ima cue_parts
  out/<ime>.fr.srt    # `film.sr.srt` -> `film.fr.srt`
```

1. **Rečenice iz cue-ova** (`fixedcues.assemble`), po interpunkciji — ne cue po
   cue. Rečenica pamti koje cue-ove pokriva i udeo znakova u svakom
   (`CuePart`). Crta na početku reda otvara novu repliku; `...` + malo slovo je
   nastavak; bez završnog znaka se seče posle `MAX_CUES_PER_SENTENCE` (6).
   Rečenica duža od 15 s (`SRT_MAX_SENTENCE_DURATION`, isto kao ASR) seče se na
   granici cue-a sa najvećom pauzom — ulaz ume da bude nedosledno
   interpunktiran (pet cue-ova bez tačke, 15,96 s, tri misli). Pauza kao
   samostalan signal (`SRT_MAX_GAP`) postoji, ali je **isključena**: izmereno
   2026-09-28, uz trajanje je dodavala samo pogrešne preseke (brojke su uz
   konstantu).
2. **Prevod** kao i inače, sa kontekstom i glosarom, uz jedno pravilo više
   (`FIT_LENGTH_RULE`): prevod ne sme biti duži od originala. Samo gornja
   granica — rizik je jednosmeran, pa vuče isto kao pravilo 11. Srpski izvor se samo
   prebacuje u latinicu; ekavica i interpunkcija se ne pokreću.
3. **Nazad u cue-ove** (`linebreak.split_by_shares`): prevedena rečenica se
   deli na onoliko delova koliko je original imao cue-ova, srazmerno udelima.
   Rez se kažnjava istim `_Span.break_pen` kao prelom (predlog, član,
   enklitika, pridev), a deo odstupanjem od ciljne dužine
   (`PENALTY_SHARE_DEVIATION`). Redovi unutar cue-a: isti `_best_lines`.
   Prelom reda na granici rečenica (dijalog) se čuva.
4. **Prekoračenja se prijavljuju, ne popravljaju.** Posle prelamanja u dnevnik
   ide koliko cue-ova probija `max_chars_per_line`/`max_cps` i za koliko, sa
   najgorih pet po rednom broju; editor te rečenice boji crveno (`Alt+N`).

**Profil se meri na samom ulazu.** Ručno doteran original ume da probija
spoljni standard (prvi francuski posao: 26 od 219 redova preko 42 znaka, 18
cue-ova preko 20 CPS), pa bi profil od 42/20 merio prevod aršinom koji ni original ne
poštuje. Podrazumevani profil posla iz SRT-a je zato `original`: 95. percentil
dužine reda i CPS-a ulaza, najveći broj redova po cue-u. Posle uvoza se ispisuje:

```
profil iz originala: 47 zn./red, 21.4 CPS, do 2 reda (95. percentil, …); najduži red 51 zn., najbrži cue 27.9 CPS (profil 'original-max')
```

Percentil, a ne maksimum — izmereno na istom poslu: maksimum postavljaju jedan
red (51) i jedan cue (27,9; sledeći je 25,5). Kao granica za prelamanje gurnuo
je 17% francuskih redova preko 42 znaka (original: 12%, percentil: 11%), a kao
granica za izveštaj sakrio bi 27 cue-ova bržih od 95% originala. Maksimum je na
izbor kao `original-max`; `web` ostaje za nametanje strožeg standarda.

Izveštaj uz ukupan broj prekoračenja daje i onaj koji kaže gde skratiti —
koliko je cue-ova **gore nego isti cue u originalu** (na probijenoj meri):

```
prekoračenja: 35 od 204 cue-ova probija profil original — …
od toga GORE NEGO ISTI CUE U ORIGINALU: 33 — tu skraćuj; kao u originalu (…): 2
najgori od njih: #131: 36.9 CPS (+15.4; original 23.1); …
```

U editoru su crveni samo ti cue-ovi; prekoračenje koje je imao i original
piše se sivo.

Kurziv se vraća kad je ceo deo rečenice u cue-u bio u `<i>`; kurziv na delu
rečenice se posle prevoda ne može smestiti na isto mesto i prijavljuje se
(`italic_lost`). Druge oznake (`<b>`, `<font>`) se skidaju uz upozorenje.
Neizmenjen izvorni jezik izlazi doslovno, sa ručnim prelomima.

`run_build` pre upisa poredi tajming izlaza sa `cues.json` i puca ako se
razlikuje; `test_tajming_izlaza_je_identican_ulazu` to proverava poređenjem
ulaznog i izlaznog fajla.

### Skraćivanje prekoračenja (drugi prolaz, samo na dugme)

Prekoračenja u prevodu SRT-a su po pravilu CPS: tajming je fiksan, a ciljni
jezik duži, pa pomaže samo kraći tekst. Dugme „Skrati prekoračenja" (vidi se
kad postoje cue-ovi gori od originala) šalje modelu **samo** rečenice iz tih
cue-ova, uz original i zahtev u znakovima: N = višak preko `max_cps × trajanje`,
podeljen na rečenice u cue-u; modelu ide apsolutna granica „najviše X znakova" (X = dužina − N).

Validacija je ista kao za prevod (lokalni brojevi, broj linija, navodnici).
Predlog se prihvata samo ako je **stvarno kraći**, ako je **najveći CPS**
cue-ova te rečenice pao (provereno ponovnim prelamanjem) i ako nije izgubio
**termin iz glosara**. Rezultat je `shorten.<lang>.json` — **ništa se ne
upisuje**: editor pokazuje original, sadašnji i predloženi tekst uporedo, sa
precrtanim izbačenim rečima, i upisuje samo izabrane (`shortened` flag, `.bak`
pre upisa, rečenica menjana u međuvremenu se preskače).

Izmereno 2026-09-28 (Otvorena vrata 26, gemma3:12b), sa apsolutnom granicom:

| | pokušano | prihvaćeno | cue-ova izlazi iz prekoračenja | prosečno skraćenje |
|---|---|---|---|---|
| fr | 47 | 38 | 39 od 59 | 25,6 zn. (24%) |
| en | 29 | 25 | 24 od 33 | 21,6 zn. (27%) |

Ručni pregled svih prihvaćenih: većina čuva značenje, ali model **seče više
nego što treba** (medijana 2,15–2,25× traženog i uz najbolju formulaciju), i tu se
gubi sadržaj ili menja značenje (`What's behind the hills?` → `What's out there?`;
`plus grandes` → `plus belles`). Zato predlog skraćen preko opsega (`overcut`)
u editoru **nije izabran** — tu izmenu čovek mora sam da uključi.

Isprobane formulacije zahteva (isti posao, medijana stvarnog / traženog
skraćenja kod prihvaćenih, fr / en):

| zahtev u promptu | medijana | prihvaćeno | cue-ova izlazi |
|---|---|---|---|
| „kraće za najmanje N" | 2,73× / 3,44× | 40 / 27 | 46 / 28 |
| opseg N…N+slack | 2,36× / 2,93× | 36 / 24 | 38 / 23 |
| apsolutno „najviše X znakova" (**u upotrebi**) | 2,15× / 2,25× | 38 / 25 | 39 / 24 |

U upotrebi je apsolutna granica: bolja je po sva tri merila. Uz nju su iz
prompta izbačeni reč „skrati" i trenutna dužina, a blok je preimenovan u
`TITLOVI` — sve što model vuče na relativno sečenje.

**Zaključak: formulacija prompta je iscrpljena.** Tri varijante su izmerene na
istom poslu; najbolja i dalje seče preko 2× traženog. Dalje se ne meri bez nove
ideje — drugi model, ili drugačiji pristup nego jedan prolaz (npr. više kraćih
koraka ili izbor između više predloga). Do tada gubitak hvata pregled u
editoru, sa prekomerno skraćenim predlozima neizabranim.

### Francuski razmaci

Svaki francuski SRT (običan prevod i prevod postojećeg SRT-a) dobija
nerazlomljiv razmak (U+00A0) ispred `? ! : ;` i `»` i iza `«`
(`typography.py`). Običan razmak bi plejeru dozvolio da prelomi red između reči
i znaka. Obrada ide nad gotovim redovima, jer tokenizer i U+00A0 vidi kao
razmak; prelamanje ipak unapred računa taj znak u dužinu reda
(`LanguageRules.nbsp_before/after`), a izveštaj o prekoračenjima se računa nad
konačnim tekstom. Red se nikad ne završava otvorenim navodnikom ili zagradom.

Prav `"` ne zna svoju stranu, pa ga tokenizer odvaja od reči i rez između
cue-ova ume da ga odnese u susedni cue (engleski posao 2026-09-28, cue 66–71:
`"Ringe ringe raja" "` / `Ringe raja sa ""` — model je vratio uravnotežen tekst).
Zato prevod PRE prelamanja dobija usmerene navodnike jezika
(`typography.directional_quotes`: engleski `“ ”`, francuski `« »`, srpski
`„ “`), koji se lepe za reč. Krivi navodnici koje je autor stavio ostaju.

Prevod rečenice sa neuravnoteženim navodnicima (otvoren citat, ili počinje
zatvaranjem) odbija se kao i odgovor sa pogrešnim brojem linija: ponavlja se,
a posle dva neuspeha ostaje original sa `translation_failed`. Samo kad je
original uravnotežen — rečenica presečena usred citata to ne može biti.

Navodnici u francuskom (`"…"`, `“…”`, `„…“`) izlaze kao `« … »` sa
nerazlomljivim razmakom unutra; u ostalim jezicima oblik ostaje, samo se skida
razmak sa unutrašnje strane (`" Comment` → `"Comment`). Prelamanje meri red
provučen kroz ovaj post-procesor (`typographic_width`), pa dužina reda važi za
tekst kakav ide u fajl.

Isti post-procesor za **svaki** jezik svodi apostrof na jedan oblik, `’`
(`I'm` i `we’re` u istom fajlu). Dira samo `'` iza slova ili cifre; navodnici
ostaju.

### Termini iz glosara se ne seku

Rez unutar višečlanog termina glosara (`Open Door / for Children`) košta
`PENALTY_SPLIT_GLOSSARY_TERM` (300, kao predlog) — i između cue-ova i između
redova, u oba režima. Termini jezika su ključevi `jezik->*` i prevodi
`*->jezik` (`config.glossary_terms`). Prekoračenje profila (1000 po znaku) i
dalje ima prednost: termin koji ne staje u red biće presečen.

### Neprevedene rečenice

Validacija odgovora proverava samo redne brojeve, pa model ume da vrati
**izvorni tekst pod ispravnim brojem** — prvi francuski posao (2026-09-25) je
tako dobio srpski u 10 cue-ova, a dnevnik je javio „neprevedeno 1". Zato se
posle prevoda i posle svakog prelamanja (`untranslated.py`) prijavljuju dve
grupe, i to u **cue-ovima**, ne samo rečenicama:

```
UPOZORENJE: 1 rečenica neprevedena — 2 cue-a (176–177) na fr — model je odustao, …
UPOZORENJE: 4 rečenice liče na original — 10 cue-ova (1, 89, 168–175) na fr — …
```

„Liči na original": reči sa malim početnim slovom koje sadrže slova izvora kojih
cilj nema (`ć đ š ž č` u francuskom; imena su izuzeta — bila su jedini lažni
pogoci), ili sličnost sa originalom ≥ 0,8 po rečima (osim rečenice od samih
imena). Obe grupe su u crvenom bloku na vrhu stranice posla i crvene u editoru
(`Alt+N`); ručna izmena skida oznaku. Tekst se ne menja.

Pre upisa prevoda važi invarijanta: rečenica sa `translation_failed` nosi
**ceo** original, bez ijedne reči iz pokušaja — inače posao puca
(`assert_failed_is_original`).

Kad su SVA prekoračenja CPS (nijedno po dužini), izveštaj dodaje predlog za
profil sa višim `max_cps`, sa brojem izračunatim punim prelamanjem po tom
profilu (`predlog: … profil web … bi ih smanjio sa 28 na 13`).

### Ocena prevoda

```powershell
.venv\Scripts\python scripts\eval_translation.py --model qwen3:14b --model gemma3:12b
```

Učita `eval/pairs.json` (tvoje rečenice + tvoj ručni prevod) i ispiše
original / referencu / model za ručnu ocenu. **Bez BLEU** — na 25 rečenica je
beskorisan. Jedini automatski broj je „neprevedeno", što je kvar pajplajna, ne
ocena kvaliteta.

Shema `pairs.json`:

```json
{
  "src_lang": "sr", "tgt_lang": "en", "variant": "en-GB",
  "pairs": [
    { "src": "…", "ref": "…", "asr_raw": "…", "why": "…" }
  ]
}
```

`src` i `ref` su obavezni. `id` je **opcion** — ako ga nema, ključ je pozicija u
nizu od 1. Spajanje prevoda sa parovima ide **isključivo** preko tog ključa
(`KEY` u skripti); dva različita podrazumevanja su već napravila tihi kvar gde
je svaki red pokazivao `(nedostaje)` iako je prevod uspeo.

Sve što bi dalo tihe rupe u tabeli puca odmah i sa izlaznim kodom 1: prazan
niz, par bez `src`, duplirani `id`, `id` koji nije broj, i neslaganje broja
prevedenih linija sa brojem parova.

`asr_raw` (sirovi ASR oblik) se prikazuje u ispisu. `variant` ide u prompt kao
pravilo 13 (vidi gore).

Kolona sa brojem znakova stoji uz svaki red, a u zbirnoj tabeli su ukupni
brojevi i odnosi:

```
                    znakova  vs ref  vs orig  izlaz t/s  prompt t/s   vreme  neprev.
ORIGINAL                926
REFERENCA               931            1.01x
qwen3:14b               983   1.06x    1.06x       34.0      1094.4   20.0s        0
gemma3:12b              959   1.03x    1.04x       36.9      1229.0   20.7s        0

  qwen3:14b          size  9.75 GB   size_vram  9.75 GB (100%)   ctx 8192
  gemma3:12b         size  7.77 GB   size_vram  7.77 GB (100%)   ctx 8192
```

Brzina se meri **tokenima u sekundi**, ne ukupnim vremenom — ukupno sadrži i
učitavanje modela u VRAM, koje na ovoj mašini traje 8–10 s hladno i pod 2 s
toplo, pa bi zasenilo razliku između modela.

`size_vram` manji od `size` znači da je deo modela u RAM-u i da generisanje radi
delom na CPU-u. Klijent to prijavljuje sam, jer se drugačije vidi samo po
brzini.

### Interpunkcija

Interpunkcija je važnija nego što izgleda: pored segmentacije, ona je **glavni
signal za tačku preloma** u `linebreak.py`. Bez nje su i prelomi loši.

**Dijagnostika.** Posle transkripcije se prijavljuje prosečna dužina rečenice,
broj znakova kraja po 100 reči, broj zareza i udeo velikih početnih slova:

```
struktura    : 4.2 reči/rečenici, 0 znakova kraja (0.0/100 reči), 0 zareza,
               velikih početnih 0%
UPOZORENJE: ASR nije vratio nijedan znak interpunkcije…
```

**Tri** kvara se prepoznaju odvojeno:

| kvar | mera | prag |
|---|---|---|
| nema znakova | `sentence_end_marks == 0` | — |
| znak posle svake reči | prosek reči po rečenici | `MIN_AVG_WORDS_PER_SENTENCE` |
| punktirao pa prestao | znakova kraja na 100 reči | `PUNCTUATION_DENSITY_THRESHOLD` |

Drugi je isto toliko beskoristan kao prvi. O strukturi se ne sudi ispod
`MIN_WORDS_FOR_DIAGNOSIS` reči — „Dobar dan. Hvala." ima prosek 1,5 i savršenu
interpunkciju.

Treći je dodat 2026-08-12, posle francuskog testa koji su prve dve provere
propustile: Whisper je punktirao prvih pet rečenica pa prestao. Ostalo je 10
znakova kraja na 428 reči i rečenica od 214 znakova bez ijednog zareza — a
prosek rečenice je bio 20,4, dakle „zdravo" po staroj meri. Kapija je zato na
**gustini**, ne na prosečnoj dužini rečenice.

Prag je **rečnik po jeziku**, ne jedna konstanta:

```python
PUNCTUATION_DENSITY_THRESHOLD = {"default": 4.0}   # segment.py
```

Francuski i engleski imaju kraće rečenice od srpskog, pa im je prirodna gustina
viša. Zasad postoje dva merenja — premalo da se iz njih izvede broj koji važi
svuda. Zato se posle **svake** transkripcije ispisuje merenje, bez obzira na
odluku:

```
interpunkcija: 2.3 znaka kraja/100 reči, 0.9 zareza/100 reči (prag 4.0) -> restauracija
interpunkcija: 5.5 znaka kraja/100 reči, 8.9 zareza/100 reči (prag 4.0) -> u redu
```

Kad se merenja skupe, podešava se ključ po jeziku.

**Zašto se gustina zareza meri a ne odlučuje po njoj.** Za `linebreak.py` su
zarezi važniji od tačaka — oni su glavni signal za tačku preloma *unutar*
rečenice, i tekst sa 10 tačaka i nula zareza je gori od teksta sa 5 tačaka i 30
zareza. Ali prag za zareze nemamo od čega da izvedemo; drugi prag na osnovu nula
merenja bio bi nagađanje. Zato se broj meri i ispisuje, a kriterijum ostaje na
znakovima kraja dok merenja ne kažu drugačije.

**Zaštitna mreža.** `segment.merge_short` spaja rečenice ispod
`MIN_SENTENCE_WORDS` (3) ili `MIN_SENTENCE_DURATION` (1,0 s) sa sledećom, osim
ako se završavaju jasnom interpunkcijom — a to izuzeće pada kad je znakova
previše da bi bili verodostojni. Radi i pre i posle lomljenja dugih rečenica.

**Restauracija (`punctuate.py`).** Model sme da doda samo znakove i velika
slova. Niz reči pre i posle mora biti identičan, poređen bez interpunkcije i
bez razlike u veličini slova; svaka izmena reči se odbacuje. Na neuspeh: ponovi,
pa prepolovi, pa **ostavi original** — posao se nikad ne prekida. Vremena reči
se ne diraju; znak uzima sićušan odsečak sa kraja prethodne reči.

```powershell
subtitis transcribe snimak.mp3 --lang fr --punctuate
subtitis transcribe snimak.mp3 --lang fr --no-punctuate
```

Podrazumevano se pali sam kad dijagnostika pokaže da treba
(`config.AUTO_PUNCTUATE`, `auto_punctuate` u `subtitis.toml`).

`MIN_CHUNK_WORDS = 20` je namerno visok: za razliku od prevoda, odlomak od tri
reči nema šta da kaže o interpunkciji — model ga punktira kao celu rečenicu i
dobiješ `Moi je m!`. Bolje bez znakova nego pogrešni.

#### Povlačenje na poslednji znak

Model svaki čank punktira kao **zaokruženu celinu**: zatvori ga tačkom i sledeći
počne velikim slovom. Granica čanka time postaje granica rečenice, na mestu koje
sa jezikom nema veze.

Izmereno 2026-08-12 na francuskom testu: od 6 granica čanka, **4** su dobile lom
koji tamo ne pripada, ovog oblika (primeri izmišljeni): `avec nous .` / `Le
matin`, `un accord !` / `Pour la suite`, `je vous .` / `remercie`, `partir dans …` /
`Les jours`.

> Prvo objašnjenje je bilo da lom pravi čank koji je odustao. Merenje ga je
> oborilo: granica uz odustali čank bila je **jedina čista** od šest. Odustali
> čank ne dobija znakove, pa ne može ni da slomi rečenicu — odustajanje je lom
> *sprečilo*, ne izazvalo.

Rešenje ne traži poverenje u model: od odgovora se prihvata samo deo do
**poslednjeg rečeničnog znaka iza koga još ima reči**, a rep se vraća u sledeći
poziv, gde ima svoj pravi nastavak. Veštačka tačka na kraju čanka se tako uvek
odbaci — jer je uvek na samom kraju. Poslednji čank se ne povlači: tamo je tačka
prava.

Zato sledeći čank ne počinje na `lo + CHUNK_WORDS` nego tamo gde je prethodni
stvarno stao; `handle` vraća broj potrošenih reči.

**Dve zaštite od beskonačnog repa:**

* rep veći od `MAX_RETREAT_SHARE` (50%) čanka se ne vraća — izlaz se prihvata
  kakav jeste, uz upozorenje;
* rep koji bi vratio na već viđen početak se ne vraća.

Rezultat na istom snimku: 5 povlačenja (60 reči vraćeno), 2 puta presečeno
zaštitom. Tri od četiri lažna loma nestala; četvrti je ostao upravo tamo gde je
zaštita presekla povlačenje.

**Polovljenje staje kad ne pomaže.** Svaki neuspeh nosi `Rejection` sa
**potpisom** koji je isti bez obzira na veličinu čanka: manjak/višak reči je
`count:+1`, izmenjena reč je `word:généreuse->généreuses`. Kad podčank padne sa
istim potpisom kao nadređeni, model to očigledno radi dosledno i dalje
polovljenje je samo potrošeno vreme — odustaje se odmah.

Izmereno 2026-08-12: model dosledno razdvaja `ya` u `y a`, pa je ista greška
išla 60 → 61, 30 → 31, 15 → 16. Šest poziva da bi se saznalo ono što je prvi
rekao.

#### Malo slovo na prvoj reči čanka

Povlačenje skida veštačku **tačku** sa kraja čanka, ali model i dalje piše prvu
reč sledećeg čanka velikim slovom. Kad čank ne počinje na granici rečenice — a
to je uvek posle **odustalog** čanka, koji ne dobija nijedan znak — usred
rečenice ostane `revoir Ce genre` i `ce Genre de salle`.

`restore_lowercase` vraća malo slovo prvoj reči takvog čanka, ali **samo ako je
original bio mali**. `Paul`, `Serbie` i `France` su u `words.json` već veliki i
ostaju veliki.

#### Tri dozvole

| dozvola | šta pušta | primer | stanje |
|---|---|---|---|
| `ALLOW_WORD_SPLITS` | razdvajanje jedne reči na više | `ya` → `y a` | **uključeno** |
| `ALLOW_INFLECTION_CHANGES` | izmenu nastavka | `généreuse` → `généreuses` | **uključeno** |
| `ALLOW_DIACRITIC_CHANGES` | akcente | `tres` → `très` | isključeno |

Prve dve su uključene 2026-08-12, po merenju: od 8 odbijenih čankova, 4 su
padala samo zbog razdvajanja i 4 samo zbog nastavka — zajedno 8/8. Obe su
pravopisne ispravke, ne izmene sadržaja. Dijakritika je ostala isključena jer
isto merenje pokazuje da nijednom od tih čankova nije bila potrebna.

**Dozvola ne znači da se izmena ne vidi.** Svaki ustupak se beleži i ispisuje,
sa konkretnim rečima — to je jedini način da se uhvati ako model pod okriljem
dozvole promeni značenje umesto pravopisa:

```
interpunkcija: dozvole propustile — inflection 1, split 1: split: 'ya' -> 'y a'; inflection: 'généreuse' -> 'généreuses'
```

Nijedna nije nagađanje. Razdvajanje se prihvata samo kad se nadovezani delovi
**tačno** poklope sa spojenim oblikom — ista provera kao za eliziju, samo
obrnuta. Nastavak se prihvata samo uz zajednički prefiks od bar
`INFLECTION_MIN_STEM` (3) znakova i najviše `INFLECTION_MAX_SUFFIX` (2) znakova
razlike: `parle` → `parlait` prolazi, `chien` → `chat` ne.

**Koliko bi svaka spasila meri se i dok su isključene.** Na svaki odbijeni čank
se poređenje ponovi sa svakom dozvolom i rezultat ide u `PunctuateStats.would_pass`:

```
interpunkcija: odbijenih čankova bi prošlo — ALLOW_WORD_SPLITS 4/8, oba 8/8, ALLOW_INFLECTION_CHANGES 4/8
```

Tek kad brojevi kažu da se isplati, dozvola se pali. Ranije merenje za
dijakritiku: strogo 0/14 čankova prošlo, tolerantno 1/8.

Merenje ostaje upaljeno i posle odluke, pa se vidi i kad **nijedno** popuštanje
ne bi pomoglo — što je tačan ishod za čank u kome model prepravi `qui` u
`filme`. To nije pravopis nego sadržaj i mora da padne.

Kad je razdvajanje dozvoljeno i primenjeno, nova reč **ne izmišlja vreme** —
deli vreme originalne reči srazmerno dužini, isto kao `asr._split_word_times`.

#### Koliko košta

Restauracija je najskuplji korak posle ASR-a, pa se meri i ispisuje:

```
interpunkcija: 103s ukupno, 62s (60%) na odbijene odgovore; 20 zahteva (10 prošlo, 10 odbijeno)
```

„Bačeno" je zbir trajanja poziva čiji je odgovor odbačen. Bez tog broja se ne
vidi da li skupoća dolazi od posla ili od ponavljanja.

### Francuska elizija

Whisper za francuski redovno vraća eliziju kao **dva tokena**, sa apostrofom uz
drugi: `j` + `'aimerais`, `aujourd` + `'hui`. Ispisano sa razmacima to je
„j 'aimerais" — razmak pre apostrofa umesto posle.

Šteta nije kozmetička:

* `linebreak.py` vidi `j` i `'aimerais` kao dve reči, pa pravilo o eliziji ne
  radi i cue ume da počne sa `'espère`, odvojenim od svog `j`;
* `punctuate.validate` odbaci ceo čank, jer model normalizuje `j 'aimerais` u
  `j'aimerais`, a to je promena niza reči.

`asr.merge_elisions` spaja par odmah posle transkripcije, **pre svega što čita
reči**. Po istoj disciplini kao `ekavica.py`: kurirani spisak, bez širokog
pravila. Osnove do `MAX_ELIDED_STEM` (2) slova idu po pravilu (`j'`, `d'`, `l'`,
`c'`, `n'`, `s'`, `t'`, `m'`), duže moraju biti u `ELIDED_STEMS` (`qu`, `jusqu`,
`lorsqu`, `aujourd`, `puisqu`, `quoiqu`, `quelqu`, `presqu`) — inače bi svaki
navodnik posle reči progutao sledeću reč.

Spojena reč uzima `s` prve i `e` druge; verovatnoća je **manja** od dve, da
spajanje ne sakrije nesigurnost ASR-a u editoru. Radi samo za `fr`
(`ELISION_LANGUAGES`). Na testnom snimku: 33 spajanja, 428 → 395 reči.

Postojeći `words.json` se ne popravlja retroaktivno — za stari posao treba
ponoviti transkripciju.

### Ekavica

Whisper provuče ijekavicu i uz `language="sr"` — `posjeduju`, `ovdje`,
`vrijeme`. `ekavica.py` to ispravlja **nad `words.json`, pre segmentacije**, pa
su `sentences.json`, SRT i sve nizvodno čisti.

Nema pravila „ije/je → e". Ono uništava `jesen`, `jedan`, `jezik`, `jeftino`,
`prijem`, `sjajno`, `Jelena`. Umesto toga stoji kurirani rečnik u tri sloja,
po istoj disciplini kao izuzeci u `translit.py`:

| sloj | značenje | primer i zašto |
|---|---|---|
| `PROTECTED_*` | nikad se ne dira, proverava se prvo | `Mjanmar` — `mj` nije ijekavizam |
| `EXACT` | cela reč | `prije`→`pre`; kao prefiks bi pokvario `prijem` |
| `PREFIX` | reč počinje osnovom | `djec`→`dec`; kao ANY bi pokvario `odjeci` |
| `ANY` | osnova bilo gde | `vidje`→`vide`; hvata i `predvidjeti`, `uvidjeti` |

Ostali slučajevi zbog kojih podela postoji: `želje` (imenica) vs `željeti`,
`dijeta` vs `dijete`, `umio se` vs `razumio`, `radio`/`studio` vs `dio`.

Prekidač je `force_ekavica` — podrazumevano uključen za `sr`
(`config.FORCE_EKAVICA_LANGUAGES`), isključuje se sa `--no-force-ekavica`.
Svaka izmena ide u `log.txt` sa indeksom reči i vremenom, a ukupan broj se
ispisuje posle transkripcije:

```
  ekavica      : 6 izmena
      [1] 0.20s  'posjeduju' -> 'poseduju'
      [3] 1.02s  'ovdje' -> 'ovde'
      [5] 1.42s  'Ovdje' -> 'Ovde'
```

`words.json` beleži `ekavica: {applied, changes}`.

### Funkcijske reči po jeziku

Kazne za prelom su vezane za **tabelu jezika**, koja se bira po polju `language`
iz `sentences.<lang>.json`. Bez toga bi se na engleskom i francuskom
primenjivale srpske liste, pa bi prelom padao iza `the`.

| jezik | `no_line_end` (ne sme da završi red, 300) | `no_line_start` (ne sme da počne red, 300) |
|---|---|---|
| `sr` | predlozi + enklitike | enklitike (`je`, `su`, `bi`, `ću`…) |
| `en` | članovi `a/an/the`, determinatori (`my…their`, `this…those`), predlozi, pomoćni glagoli, `to` ispred infinitiva | — |
| `fr` | članovi, determinatori (`mon…leurs`, `ce/cet/cette/ces`), predlozi, pomoćni glagoli, elidirani oblici | — |

Francuski dodatno: **svaki** token koji se završava apostrofom (`l'`, `d'`,
`qu'`, `n'`, `s'`, `j'`…) drži sledeću reč uz sebe.

Enklitike su jedine koje ne smeju ni da **počnu** red — enklitika po definiciji
nema svoj naglasak. Engleski i francuski pomoćni glagoli slobodno počinju red
(`he / is going`), pa su samo u `no_line_end`.

Heuristika za pridev (kazna 200) postoji samo za srpski; `en`/`fr` imaju praznu
listu nastavaka.

Veznici za `en`/`fr` **nisu bili zadati** — dodate su kratke liste. Pravilo
„prelom na vezniku = 10" važi na **različitoj strani** (`conjunction_side`):
srpski lomi iza veznika (`i / onda`), engleski i francuski **ispred**
(`old / and when`); veznik ispred drugog veznika ne sme sam da završi red.
Izmereno 2026-09-25 na prvom SRT poslu (211 cue-ova, sr→en): sa pravilom „iza"
39 od 160 granica je padalo iza veznika, posle promene 1; prosečno odstupanje
od udela cue-a nije poraslo (3,5 → 3,4 znaka). `that` nije među
determinatorima nego među veznicima: na istom poslu je od 10 pojava samo jedna
bila determinator, šest veznik. Na spisku su i `which/who/where` (en) i
`qui/dont/lorsque/parce que` (fr); `parce / que` se ne razdvaja.

**Pouka šira od ovog pravila:** kazne prepisane iz jednog jezika u drugi ne
važe automatski. „Prelom iza veznika" je pisan za srpski i na engleskom je vukao
rez na pogrešnu stranu. Isto se već desilo sa enklitikama (samo srpske ne smeju
da počnu red; engleski pomoćni glagoli smeju) i sa francuskom elizijom (`l'` se
ne odvaja od reči, čega u drugim jezicima nema). **Sledeći jezik koji se doda
počinje proverom koje kazne u njemu uopšte imaju smisla** — pa merenjem na
pravom poslu — a ne kopiranjem tabele najbližeg jezika.

**Nepoznat jezik** ne lomi tiho bilo gde — primenjuju se samo opšta pravila
(dužina, balans, CPS, trajanje) uz upozorenje u `log.txt`:

```
nema jezičkih pravila za 'de' — prelamam samo po opštim pravilima
(dužina, balans, CPS). Funkcijske reči se neće poštovati.
Poznati jezici: en, fr, sr.
```

Nova tabela se dodaje kao unos u `LANGUAGE_RULES` u `linebreak.py`.

### Transliteracija

Latinica → ćirilica je jedini netrivijalan smer: `lj`, `nj`, `dž` nisu uvek
digrafi. Lista izuzetaka je **u kodu** (`SPLIT_STEMS_ANY`, `SPLIT_STEMS_START`,
`SPLIT_PREFIXES`), da bude verzionisana. `cyrtranslit` se ne koristi — pravi
tačno ove greške.

```
nadživeti   → надживети      odžak   → оџак      (kontraprimer)
injekcija   → инјекција      konjak  → коњак
konjugacija → конјугација    kanjon  → кањон
podžanr     → поджанр        inje    → иње
tanjug      → танјуг         hodža   → хоџа
```

Generalno pravilo za prefiks `van-` **namerno ne postoji** — pokvarilo bi ime
`Vanja`. Umesto toga stoje eksplicitne osnove (`vanjezič`, …).

#### Jedna datoteka, jedno pismo

Unutar pajplajna je sve latinica (`pipeline.STORAGE_SCRIPT`): `words.*.json` i
`sentences.*.json` se drže u `Latn`, a ćirilica postoji **samo u gotovom SRT-u**.
Tako se ni glosar, ni poređenja, ni prelamanje nikad ne oslanjaju na pismo.

Tri pravila koja to drže:

1. **Transliteracija je jedan prolaz nad celim tekstom**, tek kad je ceo korak
   gotov — posle svih čankova prevoda, odnosno nad svim cue-ovima pri
   prelamanju. Nikad po odlomku.
2. **`convert` je idempotentan** i nema prečicu „već je u ciljnom pismu,
   preskoči". Za `Cyrl` se ide preko latinice baš zato: mešan ulaz i ulaz koji
   je već ćirilica prolaze kroz isti put. Prečica bi na mešanom tekstu okinula
   za pogrešan deo i ostavila ga neprevedenog.
3. **Gotov tekst se proverava** (`pipeline.assert_single_script`) i pre nego što
   datoteka nastane. Mešano pismo puca kao `SubtitisError` sa spiskom rečenica i
   spornim slovima; ne upisuje se ni `sentences.*.json` ni SRT.

Provera zna za `q`, `w`, `x`, `y`: oni nemaju ćirilični par i ostaju i u
ispravnom ćiriličnom titlu (`Netflix` → `Нетфлиx`), pa se ne prijavljuju.

> **Uhvaćeno 2026-08-17**, prevod fr → sr: rečenice 1–9 ćirilica, 10 i dalje
> latinica, presek oštar. Devet je `CHUNK_SIZE` — model pravilo „piši isključivo
> latinicom" iz prompta poštuje **po odgovoru**, pa ga za jedan čank prekrši a za
> ostale ispoštuje. Ništa posle prevoda nije normalizovalo tekst; `script="Latn"`
> u zaglavlju datoteke je bio tvrdnja koju niko nije proveravao. Prelamanje je
> pritom konvertovalo **samo** za `Cyrl`, pa je `Latn` SRT prosleđivao mešavinu
> dalje. `tests/test_script_consistency.py` drži sva tri pravila.

---

## Poznato ponašanje: lažna imena iz spojenih cue-ova

Ulazni SRT bez tačke na kraju cue-a spaja se sa sledećim u istu rečenicu. Kad
sledeći cue počinje velikim slovom, model to čita kao nastavak imena:

```
cue 82  Ja sam Bojan
cue 83  Imam dva psa
->      „Ja sam Bojan Imam dva psa …"  ->  "I'm Bojan Imam, I have two dogs …"
```

Nije kvar u sastavljanju — nema interpunkcije po kojoj bi se znalo da je
„Imam" nova rečenica, a veliko slovo posle imena nije pouzdan znak (imena u
nabrajanju: „Lana, Teodora, / Ilija, Nevena"). Posledica je nedosledne
interpunkcije ulaza i ispravlja se u njemu: tačka iza „Bojan". Primećeno
2026-09-28 (Otvorena vrata 26, cue 82–83), isto u engleskom i francuskom.

## Poznato ponašanje: cue-ovi umesto dva reda

Prelom cue-a plaća `break_pen` i `PENALTY_PER_CUE`, ali **izbegne kazne za
balans** (`PENALTY_LINE_IMBALANCE`, `PENALTY_TOP_LONGER`), koje važe samo unutar
cue-a. Zato DP ume da izabere dva jednoredna cue-a umesto jednog dvorednog.

Izmereno na testnom snimku (9 rečenica, profil netflix):

| | cue-ova | jednorednih | prosek znakova |
|---|---|---|---|
| `sr` sa tabelom | 21 | 14 | 43.1 |
| `sr` bez tabele | 17 | 8 | 53.5 |
| `en` sa tabelom | 22 | 15 | 44.6 |
| `en` bez tabele | 16 | 4 | 61.7 |

Ovo nije posledica jezičkih tabela nego njihove interakcije sa balansom —
srpski se ponaša isto. Konkretno, za rečenicu od 74 znaka: jedan cue sa dva
reda košta 106 poena, dva jednoredna cue-a 90.

Popravka je jedna konstanta: `PENALTY_PER_CUE` sa 20 na ~50 čini da svaki
razumno uravnotežen dvoredni cue (razlika redova ≤ 10 znakova) pobedi dva
jednoredna. **Nije primenjeno** — menja i već pregledani srpski izlaz.

---

## Testovi

```powershell
.venv\Scripts\python -m pytest -q
```

Pokrivaju `translit.py`, `segment.py`, `linebreak.py` i `srt.py`.

---

## Dashboard (Faza 3)

FastAPI + SQLite + Jinja2 + HTMX. Bez Reacta, bez build koraka —
`htmx.min.js` je vendorovan u `subtitis/web/static/`.

**Pokretanje: prečica na desktopu.** Napravi se jednom:

```powershell
.venv\Scripts\python -m subtitis shortcut
```

Prečica cilja `pythonw.exe`, pa **nema terminala** — umesto njega je ikonica u
sistemskoj traci sa „Otvori" i „Izađi". Ikonica se pravi iz izvora
(`scripts/make_icon.py`), u sedam veličina koje Windows traži.

**Port.** Podrazumevano **8010** (8000 drže mnoge razvojne alatke). Menja se
bez izmene koda, ključem u `subtitis.toml` ili promenljivom `SUBTITIS_PORT`:

```toml
port = 8010
```

Prečica ne nosi port — čita ga iz podešavanja, pa izmena važi i za nju.

**Jedna instanca.** Ako Subtitis već radi, prečica (i `serve`) ne diže drugi
proces nego samo otvori pregledač — dva procesa nad istom bazom značila bi dva
workera i dva modela u VRAM-u istovremeno. Provera **pita port ko je**
(`/api/identity`), ne samo da li je otvoren: ranije je tuđa aplikacija na
portu ispadala „Subtitis već radi" i pregledač se otvarao na tuđoj stranici.
Starija verzija Subtitisa, bez `/api/identity`, prepoznaje se po naslovu
početne strane — da se uz nju ne digne druga instanca.

Kad port drži **tuđa** aplikacija, ide se naviše do prvog porta na kome je
Subtitis (tad se samo otvori) ili do prvog slobodnog, i to se ispisuje:

```
Subtitis se diže na http://127.0.0.1:8011 — port 8010 drži druga aplikacija, pa je izabran 8011.
```

Redosled je bitan: sledeće pokretanje nađe instancu na 8011, ne diže treću.

`pystray` i `Pillow` su neobavezni: bez njih sve radi isto, samo bez ikonice u
traci.

`subtitis.bat` ostaje za pokretanje sa vidljivim terminalom. Isto iz komandne
linije:

```powershell
.venv\Scripts\python -m subtitis desktop               # bez terminala, sa trakom
.venv\Scripts\python -m subtitis serve                 # web + obrada, :8010
.venv\Scripts\python -m subtitis serve --no-worker     # samo web (za razvoj)
.venv\Scripts\python -m subtitis worker                # obrada zasebno
.venv\Scripts\python -m subtitis shortcut              # prečica na desktopu
```

**Obrada je nit u `serve` procesu, ne drugi proces.** Korisnik ne treba da zna
da postoji red zadataka. Ono što se time NE menja:

* red i dalje ide preko SQLite-a, jedan zadatak odjednom — dva modela se nikad
  ne nađu u VRAM-u istovremeno, jer `pipeline` oslobađa Whisper pre LLM-a bez
  obzira ko ga je pozvao;
* nit je `daemon`, a petlja čeka na `Event` umesto `sleep`, pa gašenje servera
  ne visi ni kad je `--poll` velik;
* zadaci zatečeni kao `running` posle pada vraćaju se u red pri sledećem
  pokretanju.

`subtitis worker` ostaje kao zasebna komanda za razvoj. Ako se pokrene uz
`serve` bez `--no-worker`, dva workera će deliti isti red — `claim_next_task` je
atomičan, pa se zadatak neće izvršiti dvaput, ali dva modela mogu da se nađu u
VRAM-u istovremeno. Za razvoj u dva prozora koristi `serve --no-worker`.

### Indikator obrade

U zaglavlju stoji zelena tačka „obrada aktivna" ili crvena „obrada ne radi —
N zadataka čeka", uz ukupno zauzeće `jobs/`.

Postoji zato što je prvi susret sa dashboardom bio: upload prođe, ništa se ne
dešava, i nigde ne piše zašto. Worker upisuje otkucaj u tabelu `worker_beat`
na `HEARTBEAT_SECONDS` (5 s), **iz zasebne niti** — `tick()` blokira koliko
traje transkripcija, pa bi otkucaj iz glavne petlje pocrveneo usred posla.
Otkucaj stariji od `HEARTBEAT_STALE_SECONDS` (20 s) znači da obrade nema.
Pri urednom gašenju se briše, da indikator pocrveni odmah a ne za 20 s.

**Worker ostaje da radi.** Prazan red ga ne obara, jedan pokvaren posao ga ne
obara (ni beleženje neuspeha ne sme da baci izuzetak), a **jedan** prekid ga
samo naoruža:

```
Primljen prekid (SIGINT). Worker OSTAJE da radi.
  Za zaustavljanje ponovi u roku od 5s.
```

Razlog je što Windows ume da isporuči konzolni događaj celoj grupi procesa, pa
je jedan `SIGINT` slab dokaz namere. `SIGTERM` se na Windows-u i ne registruje —
tamo se isporučuje kao `TerminateProcess`, koji handler nikad ne vidi.

### Zadatak čiji je posao pao se ne izvršava

Bez toga se posle neuspele transkripcije pusti prelamanje nad nepostojećim
rečenicama i prijavi „gotovo". Sada:

```
[transcribe] pao GREŠKA: Ulazna datoteka ne postoji: D:\ne\postoji.wav
    preskočeno zadataka u redu: 2

ZADACI:  transcribe  error    Ulazna datoteka ne postoji…
         build       skipped  prethodni korak je pao
         translate   skipped  prethodni korak je pao
POSLOVI: pao         error    Ulazna datoteka ne postoji…
```

Prelamanje koje ide **uz** uspešnu transkripciju ostaje neobavezno (glavni
posao je zapisan), ali se neuspeh prijavljuje kao greška u dnevniku, ne kao
usputna beleška. Samostalni `build` zadatak, koji ništa drugo ne radi, **mora**
da padne ako nema nad čim da radi.

Dashboard prikazuje crveni okvir sa porukom, spiskom palih i preskočenih
zadataka i brojem preskočenih.

### Tok je eksplicitan, nikad lančan

```
uploaded -> transcribing -> transcribed -> [PAUZA ZA PREGLED] -> translating -> done
```

Posle transkripcije worker **staje**. Original SRT je za preuzimanje odmah — ne
čeka prevod. Prevod pokreće čovek posebnim dugmetom, tek pošto pregleda original.
Test `test_upload_stavlja_samo_transkripciju_u_red` proverava da u redu posle
otpremanja stoji samo `transcribe`.

### Ciljni jezik i pismo

Ciljni jezici su u `app.TARGET_LANGUAGES`; izvorni jezik posla se izbacuje iz
ponude (prevod `sr → sr` inače pukne u pajplajnu).

Izbor pisma (`Latn`/`Cyrl`) se prikazuje **samo za dvopismene ciljne jezike**
(`BISCRIPTAL_LANGUAGES`, zasad `sr`). Ćirilica nad francuskim tekstom nije
opcija nego kvar, pa se ne nudi — i odbija se na ulazu (HTTP 400) za slučaj da
neko pošalje obrazac ručno.

> **Zašto `[hidden] { display: none !important }` stoji u `app.css`.**
> `[hidden]` iz pregledača je specifičnosti (0,1,0) i **gubi** od
> `.row-form label { display: flex }` (0,1,1). Bez tog pravila je polje za pismo
> ostajalo vidljivo i za francuski, pa se `Cyrl` stvarno slao. Test
> `test_izbor_pisma_je_sakriven_css_om_a_ne_samo_atributom` čuva to pravilo.

### Greška iz obrasca je poruka, ne pad

Podrazumevani FastAPI rukovalac vraća goli JSON, pa je korisnik koji pošalje
obrazac dobijao crn ekran sa `{"detail": ...}`. Sada `HTTPException` renderuje
`error.html` kad zahtev traži HTML; poziv iz JavaScripta i dalje dobija JSON —
razlikuje se po `Accept` zaglavlju.

### Editor

Glavni ekran. Red = **rečenica**, ne cue.

To je jedina mogućnost koja se slaže sa ostatkom: `edited_by_human` živi na
`sentences.json`, a cue-ovi su izvedeni iz profila. Da se uređuju cue-ovi, svaka
promena prelomnog profila bi obrisala ispravke — što je suprotno principu da se
SRT regeneriše bez ponovnog ASR-a. Red prikazuje vreme rečenice; prelamanje na
cue-ove ostaje posao `linebreak.py`.

- `contenteditable` po redu, snimanje postavlja `edited_by_human: true` i pravi `.bak`
- dugme ▶ uz svaki red pušta **taj isečak** audija (`/jobs/{id}/clip`, ffmpeg
  iseca na zahtev — ne šalje se ceo fajl)
- reči sa `p < 0.6` obojene žuto, sa tačnom verovatnoćom u `title`
- posle snimanja se automatski stavlja `build` u red, pa se SRT osveži
- ručno menjan fajl blokira ponovnu transkripciju (HTTP 409)

Bojenje se odlučuje po **poreklu** rečenica, ne po prisustvu `words.json`:
prevedene rečenice nemaju verovatnoće po rečima i ne boje se.

#### Poništavanje i verzije

Tri sloja, od najbržeg ka najtrajnijem:

1. **Ctrl+Z** — sopstveni stek u pregledaču, do `UNDO_LIMIT` (20) izmena.
   Ugrađeni `contenteditable` undo se ne koristi jer radi **po elementu**: kad
   pređeš u drugi red, Ctrl+Z vrati izmenu u pogrešnom redu. Ovde je jedinica
   izmene RED, a kucanje u istom redu bez prekida od 1,2 s je jedna izmena —
   inače bi se vraćalo slovo po slovo.
2. **„vrati na prethodno snimljeno"** — najnovija `.bak` kopija.
3. **Lista verzija** sa vremenom i brojem rečenica.

Povratak **prvo snima tekuće stanje**, pa i on može da se poništi.

> `cfg.backup` je dobio brojač: oznaka ima rezoluciju od sekunde, pa su dva
> snimanja u istoj sekundi davala isto ime i drugo je tiho pregazilo prvo.
> Brojač ide na kraj oznake, tako da prvih 15 znakova ostaje `%Y%m%d-%H%M%S` i
> za stare i za nove kopije; sortiranje ide po (vreme, brojač), ne po imenu —
> `-2` bi po imenu ispalo ispred `.`.

#### Automatsko snimanje

Na `AUTOSAVE_SECONDS` (30) kad ima neupisanih izmena, plus kad kartica ode u
pozadinu, plus `sendBeacon` pri zatvaranju (u tom trenutku `fetch` više ne
stigne). Automatsko snimanje se razlikuje od ručnog u tri stvari:

* ne pravi `.bak` češće od `MIN_BACKUP_INTERVAL_SECONDS` (5 min);
* ne stavlja prelamanje u red — SRT se regeneriše tek na ručno snimanje;
* **prazna rečenica se preskače**, ne odbija. Usred kucanja je prazan red
  normalno stanje; ručno snimanje ga i dalje odbija kao grešku.

#### Prečice na tastaturi

| taster | šta radi |
|---|---|
| `Tab` / `Shift+Tab` | sledeći / prethodni red |
| `Ctrl+Space` | pusti isečak tekućeg reda |
| `Space` | isto, **ali samo kad kursor nije u tekstu** |
| `Alt+N` / `Alt+P` | sledeća / prethodna rečenica sa nesigurnim rečima |
| `Ctrl+S` | snimi |
| `Ctrl+Z` | poništi izmenu |

Gola razmaknica ne može bezuslovno da pušta zvuk jer u `contenteditable` mora
da kuca razmak — zato `Ctrl+Space` radi uvek, a sama razmaknica samo van teksta.

#### Original i prevod uporedo

Kad je otvoren prevod, u istoj vrsti stoji i original — bez toga se pri
ispravljanju mora pamtiti šta je pisalo. Vezuju se po `id`, koji prevod
zadržava iz izvornog fajla.

Izvor se čita iz `derived_from`, **ne** iz jezika posla: kod pivota
(`sr → en → de`) original za `de` je `en`, a ne `sr`.

Uređuje se **samo prevod**. Ispravke se čuvaju u `sentences.<otvoreni>.json`,
pa bi izmena u koloni originala bila tiho izgubljena — zato ta kolona nije
`contenteditable`.

**Žuto važi samo za kolonu originala.** Prevod nema verovatnoće po rečima, pa
bi bojenje na njemu bilo izmišljeno. Kolona se može sakriti kvačicom, za uske
ekrane.

### Profil prelamanja

Svi profili pišu u **isti** fajl (`ime.fr.srt`) — ime je po BCP-47 i ne nosi
profil. Zato stranica ispisuje čime je postojeći SRT napravljen, a meni se
preselektuje po tome, ne po `DEFAULT_PROFILE`:

```
Napravljeno profilom web, pismo Latn, 46 cue-ova.
```

Podatak dolazi iz `job.json` (`history` -> poslednja uspešna faza `build`), što
je jedini zapis o tome šta je stvarno primenjeno.

**Profili od kruga 8.**

| profil | prelom | izlaz | namena |
|---|---|---|---|
| `tv` | 2 reda × 42 zn., 20 CPS, 0,83–7 s | normalno | klasičan titl (podrazumevani) |
| `web` | 1 red, 14–28 zn., 20 CPS, 0,50–4 s | normalno | vertikalni video |
| `digital` | isto kao `web` | VELIKA slova, bez interpunkcije | društvene mreže |

`tv` je dosadašnji `web` (a pre kruga 6 `netflix`), `web` je dosadašnji
`digital`. Pošto je ime `web` promenilo značenje, stari poslovi se ne čitaju
preko pseudonima nego se jednom prepisuju (`subtitis/migrate.py`, `job.json`
verzija 2): `netflix`, `tv` i `web` -> `tv`, `digital` -> `web`. Migracija ide
pri startu dashboarda i workera, pri učitavanju starog posla, i kao
`subtitis migrate`; svaka promena se upisuje u dnevnik posla. Posao sa
nepoznatim imenom se ne dira i dobija upozorenje.

Jednoredni prelom nosi meku donju granicu reda (`min_chars_per_line` 10,
`PENALTY_BELOW_MIN_CHARS`); kazna za usamljenu reč i neravnotežu redova ne
važi kad je `max_lines` 1. Cue-ovi koji ni posle produžavanja ne stignu do
`min_duration` prijavljuju se u dnevniku i na stranici posla. Merenja i
odluke: `merenja/dnevnik.md` (krugovi 6 i 7, profil se tada zvao `digital`).

`digital` samo POSTAVLJA veličinu slova i izostavljanje interpunkcije kad se
izabere (`Profile.case`, `Profile.drop_punctuation`); oba se menjaju ručno.
Interpunkcija se skida pri pisanju SRT-a, posle prelamanja — prelamanje bira
rez po zarezima. Ostaju apostrof i crtica UNUTAR reči (`c'est`,
`rendez-vous`) i znaci između cifara (`3,5`, `15:30`). Prekoračenja se broje
pre toga, pa poneko prijavljeno u gotovom fajlu ne postoji (bezbedan smer).

**Veličina slova** (`--case normal|upper|lower`, na stranici uz izbor
pisma) primenjuje se tek pri pisanju SRT-a (`fixedcues.apply_case`), posle
transliteracije i brojanja prekoračenja; `sentences.*.json` se ne dira. Menja
se samo tekst van oznaka (`<i>`, `{\an8}`), a broj znakova ostaje isti —
`ß` postaje `ẞ`, ne `SS`, jer profil broji znakove. Redosled sa
transliteracijom je svejedno (`DŽ` -> `Џ`, test `test_case.py`).

Uz izbor idu brojevi i opis (`Profile.note`, u samom `profiles/*.json`, pa ne
može da se raziđe sa vrednostima), i **primer preloma iz tog posla** — ista
rečenica prelomljena po svakom profilu. Uzima se najduža rečenica, jer se na
kratkoj profili ne razlikuju.

### Šta bojenje NE znači

Iz Faze 1 znamo da `low_confidence` ima visoku preciznost ali nepotpun odziv —
pet od devet rečenica označeno, a greške su bile i u neoznačenima. Zato editor
nosi izričito upozorenje umesto da neoznačene redove prikaže kao uredne:

> Žuto = ASR prijavljuje nesigurnost za tu reč. Odsustvo žutog *nije* potvrda
> tačnosti. Boja služi da ubrza pregled, ne da dozvoli preskakanje.

Nema zelenih kvačica, nema „provereno" oznaka, nema brojača preostalih
označenih redova — sve troje bi sugerisalo da je pregled završen kad nestanu
žute reči.

### Šta gde stoji

Baza (`jobs/subtitis.db`) drži **samo** status, red zadataka i događaje.
Sadržaj poslova ostaje na disku u `jobs/<id>/`, pa brisanje baze gubi istoriju,
ne posao. Brisanje posla iz liste briše i red u bazi i direktorijum posla.

### Podešavanja

Ekran `/settings` piše u `subtitis.toml`: podrazumevani profil, ciljni jezik
prevoda, model za LLM, TTL za brisanje WAV-ova i da li se interpunkcija pali
sama. Uz to uređuje glosar (`glossary/default.json`), koji ide u ASR kao
`hotwords` i u prevodni prompt — svaka izmena pravi `.bak`.

Dve stvari koje ovo čine korisnim, a ne ukrasnim:

* **vrednosti se stvarno čitaju.** Svaka ima pristupnu funkciju
  (`cfg.default_profile()`, `cfg.llm_model()`, `cfg.source_wav_ttl_days()`), a
  ne golu konstantu — konstanta bi bila pročitana jednom pri uvozu i izmena
  kroz ekran ne bi imala efekta do ponovnog pokretanja;
* **promenljive okruženja i dalje pobeđuju**, i ekran to kaže kad je tako
  (`SUBTITIS_LLM_MODEL=… pobeđuje`), da se ne menja vrednost koja se ne primenjuje.

Pisac TOML-a je namerno prost i pokriva samo skalarne vrednosti iz
`EDITABLE_SETTINGS`. **Komentari u ručno pisanom `subtitis.toml` se gube** čim
se ekran upotrebi.

### Sažetak posla

Na stranici posla stoji „Šta je urađeno": model, broj reči, rečenica,
pokrivenost, elizija, i za interpunkciju — koliko znakova, koliko čankova je
prošlo, koliko je vremena bačeno i koje je ustupke dozvola propustila. Sve iz
`job.json`, da se ne traži po `log.txt`.

### Šta se bira na ekranu, a šta ne

**Whisper model se ne bira.** Fiksiran je na `large-v3`. Slučajno spuštanje na
`medium` daje osetno lošiji srpski prepis, a ništa na ekranu to ne bi objasnilo
— rizik je veći od koristi. Ostaje kao CLI opcija (`transcribe --model`).

**Izvorni jezik ima `auto`.** Detektovani jezik i sigurnost se prikazuju na
stranici posla, i kad se ne poklapaju sa zadatim (`jezik: hr (80%) — zadato je
sr`) — to je često jedini trag zašto prepis izgleda ijekavski.

Za srpski i dalje **zadaj `sr`**: auto-detekcija ga meša sa hrvatskim i vraća
ijekavicu. Uz `auto` glosar ne može da se prosledi kao `hotwords` (termini se
biraju po jeziku, a jezik se zna tek posle prolaza), a i provera
`edited_by_human` se pomera na posle ASR-a — cena je jedan uzaludan prolaz u
retkom slučaju, umesto pregaženih ispravki.

### Čišćenje diska

`source.wav` je ~115 MB po satu audija i potreban je samo dok traje korektura —
dugme ▶ iz njega seče isečke.

* upload je ograničen na `MAX_UPLOAD_BYTES` (2 GB, ~17 h zvuka); nedovršen
  upload se briše, ne ostaje na disku;
* lista poslova pokazuje zauzeće po poslu i koliko od toga je radni WAV;
* **brisanje posla briše i direktorijum.** Ranije su datoteke ostajale, pa je
  „obriši" ostavljalo 40 MB po poslu i nije postojao nijedan način da se toga
  rešiš osim ručno kroz Explorer. Potvrda kaže koliko se briše;
* **brisanje otkazuje i zadatak** — jedini način da se ukloni obrada pokrenuta
  greškom (zasebno „prekini zadatak" ne postoji). Potvrda to kaže. Zadatak u
  redu se briše odmah, zajedno sa poslom. Zadatak u toku staje na prvoj
  prirodnoj tački (`config.checkpoint()`): pre svakog poziva LLM-u — prevod,
  interpunkcija, skraćivanje — i posle svakog ASR segmenta; nikad usred poziva
  modelu. Worker to beleži kao uredno „prekinut — posao je obrisan", ne kao
  grešku, i ide na sledeći zadatak; isto i kad posao pukne pre te tačke jer su
  mu fajlovi nestali. Posle zadatka worker još jednom briše direktorijum i
  događaje posla, jer zadatak ume da upiše nešto posle brisanja (a na
  Windows-u otvoren `source.wav` ne može da se obriše odmah) — bez ostataka.
  Whisper se tada oslobađa iz VRAM-a, Ollama sama;
* `source.wav` se briše **sam** posle `SOURCE_WAV_TTL_DAYS` (7) dana. Worker to
  radi pri pokretanju i na svakih `PURGE_EVERY_SECONDS` (6 h) — server ume da
  radi nedeljama, pa bi čišćenje samo na pokretanju značilo nikad. Svako
  brisanje ide u `log.txt` tog posla i u dnevnik na ekranu. `0` isključuje;
* zaglavlje pokazuje ukupno zauzeće `jobs/`.

Automatski se briše **samo radni WAV** — ono što se može ponovo napraviti iz
originalnog fajla. `words.json`, `sentences.json` i SRT-ovi se ne diraju nikad;
njih briše samo čovek, brisanjem posla.

### Ispis na konzolu ne sme da obori posao

`cfg.echo` postoji zato što `print` na Windows-u nije bezbedan: konzola je
podrazumevano **cp1252**, pa `print("vraćeno")` baca `UnicodeEncodeError` — i to
je rušilo ceo HTTP zahtev usred snimanja, sa 500.

Uhvaćeno 2026-08-12 pri probi povratka na verziju: `restore` je vraćao
`Internal Server Error`, a kriva je bila jedna reč sa `ć` u log poruci.

Pravilo: **log fajl je UTF-8 i uvek tačan; konzola je pomoćni ispis i sme da
izgubi kvačicu, ali ne sme da obori rad.** Svi ispisi u paketu idu kroz
`cfg.echo`, koji pada nazad na zamenu znakova, pa na tiho odustajanje.

### Pod `pythonw` izlaza nema uopšte

Prečica pokreće `pythonw.exe`, gde `sys.stdout` i `sys.stderr` nisu pogrešno
kodirani nego **ne postoje** — `None` su. Svaki ispis tada baca `AttributeError`,
i to ne samo naš: `uvicorn` pri pravljenju formatera loga zove
`sys.stdout.isatty()`, pa je dvoklik na prečicu padao pre nego što se ijedan red
ispiše. Popravka po pozivnom mestu ne pomaže kad izlaz dira tuđi kod, pa
`bootstrap.ensure_streams()` na ulaznoj tački podmeće struje jednom za sve.

Zamena nije prazna kanta nego `jobs/console.log`: pod `pythonw` je to jedini
način da se vidi šta je server ispisao.

Uz to, `bootstrap` hvata svaki neuhvaćen izuzetak — iz glavne niti i iz radnih —
u **`jobs/startup-error.txt`**, sa punim traceback-om, komandom i interpreterom.
Bez toga je pad pod `pythonw` potpuno nevidljiv: nema ni prozora ni poruke,
prečica prosto ne uradi ništa. **Kad dvoklik ne uradi ništa, prvo tu gledaj.**

Testovi ovo ne uhvate sami od sebe, jer se izvršavaju pod običnim interpreterom
sa urednim `stdout`; u `tests/test_bootstrap.py` se struje gase namerno, a cela
ulazna tačka se pokreće u zasebnom procesu sa `sys.stdout = sys.stderr = None`.

### Zajedničko jezgro

`pipeline.py` sadrži `run_transcribe`, `run_translate` i `run_build` bez ijednog
ispisa. CLI i worker zovu iste funkcije — da su imali svoje verzije, razišle bi
se prvom promenom, a razlika između terminala i dashboarda je greška koja se
vidi tek na gotovom titlu.
