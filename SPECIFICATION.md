# Subtitis — specifikacija

Šta Subtitis radi iz ugla korisnika: **kad uradim X → desi se Y**. Svaka izmena ponašanja prvo se upiše ovde, pa u kod.
Detaljan opis rada, merenja i odluka je u `document/` (projektni dokument) i `README.md`; istorija u `CHANGELOG.md`.

## Svrha
SRT titlovi iz audio i video snimaka i njihov prevod, lokalno na Windows-u (NVIDIA 12 GB VRAM). Zvuk, prepisi i prevodi ne napuštaju računar.

## 1. Pokretanje
- Dvoklik na prečicu Subtitis → server i obrada u istom procesu, prozor bez adresne trake, ikonica pored sata (Otvori / Izađi).
- Drugi klik → otvara se postojeći prozor, druga instanca se ne diže.
- Port 8010; ako ga drži tuđa aplikacija, ide se na sledeći slobodan i to se kaže.

## 2. Posao
- Ubacim snimak → prepoznavanje govora (faster-whisper large-v3) → original SRT odmah za preuzimanje.
- Posle prepoznavanja **pauza za pregled**: ništa se ne dešava samo od sebe. Prevod pokrećem ja, pošto pregledam original.
- Ubacim gotov SRT → prevod uz isti broj i ista vremena natpisa.
- Ručna izmena u editoru se nikad ne gazi automatski.
- Profil preloma (tv / web / digital) menja samo prelom, bez ponovnog prepoznavanja.

## 3. Podešavanja (redom)
- Podrazumevane vrednosti, glosar novog projekta, interpunkcija, glosar.
- **Dnevnik**: veličina, „Prikaži dnevnik" (filter, samo problemi), „Obriši starije od 30 dana".
- **O programu**: šta radi, „Napravio Mirko Rudić sa Claude Opus 5.5 (Anthropic)", licenca MIT; „Više" → motori i licence.
- **Verzija**: potpis verzije, ispod stanje poslednje provere; „Proveri", a kad ima nova „Ažuriraj odmah" (restart sačeka da obrada završi, prozor se sam osveži).
- **Deinstaliraj**: uklanja prečicu sa Desktopa i gasi Subtitis; folder i poslovi ostaju.

## 4. Ažuriranje (kopija koja je git klon)
- Provera na startu i na sat. Nova verzija se instalira sama kad nijedan prozor Subtitisa nije otvoren i ništa se ne obrađuje; otvoren prozor se ne gasi zbog ažuriranja.

## 5. Veza sa drugim alatima
- Ghostcut poziva Subtitis za prepis i titlove (`tools/subtitis_bridge.py` u Ghostcut-u).
- Claude desktop preko MCP servera (`subtitis mcp`).

## Namerno ne radi
- Ništa u oblaku; nema naloga.
- Svetla tema (samo tamna).
- Automatski prevod posle prepoznavanja (uvek pauza za pregled).

## Otvoreno (nije provereno na pravom računaru posle 1.6.0)
Prozor bez adresne trake, tiho ažuriranje, Deinstaliraj.
