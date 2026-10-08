"""Prevod rečenica preko Ollame.

Tri stvari moraju biti tačne, inače pajplajn tiho pravi smeće:

1. `think: false` — Qwen3 je hibridni model i po defaultu razmišlja. Izmereno na
   ovoj mašini: 124.8 s sa razmišljanjem naspram 17.0 s bez, za isti zadatak.
   Rezonovanje u Ollami 0.32 ne ide u `<think>` tagove nego u zasebno polje
   `message.thinking` — proveravamo oba.
2. `num_ctx: 8192` — Ollamin default je 4096 i tiho odseca sve preko toga.
   Gore od toga: Ollama tiho prihvata i nepoznata polja, pa se greška u imenu
   opcije ne vidi. Zato posle prvog poziva čitamo `/api/ps` i poredimo stvarni
   `context_length` sa traženim.
3. Validacija broja linija — pošalješ 10 numerisanih rečenica, model vrati 9
   (spojio dve) ili 11 (razbio jednu). Ako to ne uhvatiš, ceo ostatak fajla se
   pomeri i titl više ne prati sliku.

Nema drugih pragova ni filtera. Ništa se ne odbacuje po „kvalitetu" — odbija se
samo strukturno neispravan odgovor: skup rednih brojeva se ne poklapa sa
poslatim, ili rečenica ima neuravnotežene navodnike a original ih nema
(`typography.quotes_balanced`; otvoren citat bi se prelio u susedne cue-ove).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from . import config as cfg
from .config import SubtitisError
from .models import Sentence
from .typography import quotes_balanced

# ==========================================================================
# Konstante za štelovanje
# ==========================================================================

DEFAULT_MODEL = "qwen3:14b"
DEFAULT_URL = "http://localhost:11434"

#: Broj rečenica po zahtevu. Veće = brže, ali veća šansa da model omane broj linija.
CHUNK_SIZE = 9

#: Rečenice konteksta sa svake strane; šalju se, ali se NE prevode.
CONTEXT_BEFORE = 2
CONTEXT_AFTER = 2

#: Ollamin default je 4096 i tiho seče. Ovo mora biti eksplicitno.
NUM_CTX = 8192

TEMPERATURE = 0.2
TOP_P = 0.9

#: NE DIRAJ. Preko 1.0 model izbegava ponavljanja koja su u titlovima normalna
#: ("Da. Da. Da."), pa počne da izmišlja sinonime.
REPEAT_PENALTY = 1.0

#: Fiksan seed da bi dva pokretanja bila uporediva.
SEED = 42

#: Koliko sekundi se čeka jedan odgovor.
REQUEST_TIMEOUT_S = 900

#: Koliko puta se pokušava POJEDINAČNA rečenica pre nego što se odustane.
#: Posle toga ide original sa flagom; ceo posao se nikad ne prekida.
MAX_ATTEMPTS_PER_SENTENCE = 2

#: Koliko dugo Ollama drži model u VRAM-u posle poziva.
KEEP_ALIVE = "5m"

#: Flag na rečenici koju nismo uspeli da prevedemo.
TRANSLATION_FAILED_FLAG = "translation_failed"

#: Separator između rednog broja i teksta.
SEP = "|"

LANGUAGE_NAMES: dict[str, str] = {
    "sr": "srpskog",
    "en": "engleskog",
    "fr": "francuskog",
    "de": "nemačkog",
    "es": "španskog",
    "it": "italijanskog",
    "ru": "ruskog",
}

LANGUAGE_NAMES_TARGET: dict[str, str] = {
    "sr": "srpski",
    "en": "engleski",
    "fr": "francuski",
    "de": "nemački",
    "es": "španski",
    "it": "italijanski",
    "ru": "ruski",
}


def _lang_name(code: str, target: bool = False) -> str:
    table = LANGUAGE_NAMES_TARGET if target else LANGUAGE_NAMES
    return table.get(code, code)


# ==========================================================================
# Validacija — bez ovoga pajplajn tiho pravi smeće
# ==========================================================================


def validate(response: str, expected_ids: list[int]) -> tuple[dict | None, str | None]:
    """Vrati (mapa id->tekst, None) ili (None, razlog)."""
    lines = [l for l in response.strip().split("\n") if l.strip()]
    got = {}
    for line in lines:
        if SEP not in line:
            return None, "linija bez separatora"
        num, text = line.split(SEP, 1)
        if not num.strip().isdigit():
            return None, "broj nije cifra"
        got[int(num)] = text.strip()
    if set(got.keys()) != set(expected_ids):
        return None, f"očekivano {expected_ids}, dobijeno {sorted(got)}"
    if any(not t for t in got.values()):
        return None, "prazna linija"
    return got, None


# ==========================================================================
# Prompt
# ==========================================================================

#: Primeri svakodnevnih reči uz pravilo 12, po ciljnom jeziku.
#: Za jezik bez unosa ide samo uputstvo bez primera — engleski primeri u
#: francuskom promptu su čist šum i vuku model ka pogrešnom jeziku.
PLAIN_WORD_EXAMPLES: dict[str, str] = {
    "en": (
        '"because" a ne "considering that", "second-to-last" a ne\n'
        '    "penultimate", "took part" a ne "participated", "got to"\n'
        '    a ne "had the opportunity to".'
    ),
}

#: Pravilo o pravopisnoj varijanti ciljnog jezika. Dodaje se samo ako je
#: varijanta zadata i poznata; nepoznata varijanta ne dodaje ništa.
VARIANT_RULES: dict[str, str] = {
    "en-gb": "Piši britanskim pravopisom: -ise, -our, centre, organisation.",
    "en-us": "Piši američkim pravopisom: -ize, -or, center, organization.",
}


SYSTEM_PROMPT = """Ti si profesionalni prevodilac titlova. Prevodiš sa {SRC_NAME} na {TGT_NAME}.

PRAVILA:
1. Prevedi SAMO linije u bloku ZA_PREVOD. Blok KONTEKST služi
   isključivo za razumevanje i ne prevodi se.
2. Vrati tačno onoliko linija koliko ih ima u bloku ZA_PREVOD,
   sa istim rednim brojevima i istim formatom "broj|tekst".
3. Nikada ne spajaj dve linije u jednu niti razdvajaj jednu na dve.
   Ako je linija nepotpuna rečenica, prevedi je kao nepotpunu.
4. Ne dodaj objašnjenja, napomene, uvod ni zaključak.
   Prvi znak tvog odgovora je cifra.
5. Zadrži registar originala.
6. Zadrži oslovljavanje (ti/vi) dosledno kroz ceo tekst.
7. Imena osoba, firmi i proizvoda ne prevodi osim ako su u REČNIKU.
8. Brojeve, datume i mere prilagodi normi ciljnog jezika.
9. Ovo je govorni jezik. Poštapalice ("ovaj", "znači", "pa")
   izostavi ako ne nose značenje.
10. Piši isključivo latinicom.
11. Titl mora biti kratak. Ako postoji kraći način da se kaže isto,
    izaberi njega. Prevod ne sme biti duži od originala.
12. Ovo je govorni jezik, ne pisani. Biraj svakodnevne reči{EXAMPLES}
{VARIANT_RULE}
REČNIK (obavezno poštuj):
{GLOSSARY}"""


def format_glossary(pairs: dict[str, str]) -> str:
    if not pairs:
        return "(nema unosa)"
    return "\n".join(f"{src} -> {tgt}" for src, tgt in pairs.items())


def variant_rule(variant: str | None) -> str:
    """Pravilo 13, samo za poznatu varijantu. Inače prazno."""
    if not variant:
        return ""
    rule = VARIANT_RULES.get(variant.strip().lower())
    return f"13. {rule}\n" if rule else ""


#: Pravilo koje važi SAMO kad se prevodi postojeći SRT: prevod se upisuje u
#: cue-ove sa zadatim vremenima, pa svaki znak preko originala ide u CPS.
#: Namerno samo gornja granica: rizik je jednosmeran (francuski je 15–20% duži
#: od srpskog), kraći prevod ovde ne škodi, pa pravilo vuče isto kao pravilo 11.
FIT_LENGTH_RULE = (
    "Prevod se upisuje u postojeće titlove sa zadatim vremenima,\n"
    "    zato ne sme biti duži od originala."
)


def build_system_prompt(
    src_lang: str,
    tgt_lang: str,
    pairs: dict[str, str],
    variant: str | None = None,
    *,
    fit_length: bool = False,
) -> str:
    examples = PLAIN_WORD_EXAMPLES.get(tgt_lang)
    extra = variant_rule(variant)
    if fit_length:
        extra += f"{14 if extra else 13}. {FIT_LENGTH_RULE}\n"
    return SYSTEM_PROMPT.format(
        SRC_NAME=_lang_name(src_lang),
        TGT_NAME=_lang_name(tgt_lang, target=True),
        EXAMPLES=f":\n    {examples}" if examples else ".",
        VARIANT_RULE=extra,
        GLOSSARY=format_glossary(pairs),
    )


def build_user_prompt(
    before: list[str], items: list[tuple[int, str]], after: list[str]
) -> str:
    parts: list[str] = []
    if before:
        parts.append("KONTEKST (pre):\n" + "\n".join(before))
    parts.append(
        "ZA_PREVOD:\n" + "\n".join(f"{i}{SEP}{text}" for i, text in items)
    )
    if after:
        parts.append("KONTEKST (posle):\n" + "\n".join(after))
    return "\n\n".join(parts)


# ==========================================================================
# Ollama klijent
# ==========================================================================


@dataclass
class CallStats:
    requests: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_responses: int = 0
    #: Trajanja koja Ollama vraća, u nanosekundama. Ukupno vreme zahteva nije
    #: uporediva mera — sadrži i učitavanje modela u VRAM.
    eval_ns: int = 0
    prompt_eval_ns: int = 0
    load_ns: int = 0

    @property
    def output_tps(self) -> float:
        """Tokena u sekundi pri generisanju."""
        return self.output_tokens / (self.eval_ns / 1e9) if self.eval_ns else 0.0

    @property
    def prompt_tps(self) -> float:
        """Tokena u sekundi pri obradi prompta."""
        return self.prompt_tokens / (self.prompt_eval_ns / 1e9) if self.prompt_eval_ns else 0.0


class OllamaClient:
    """Minimalan klijent nad `/api/chat`. Bez novih zavisnosti (stdlib urllib)."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        model: str = DEFAULT_MODEL,
        *,
        num_ctx: int = NUM_CTX,
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        repeat_penalty: float = REPEAT_PENALTY,
        seed: int = SEED,
        timeout: float = REQUEST_TIMEOUT_S,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.options = {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "seed": seed,
        }
        self.timeout = timeout
        self.log = log or cfg.echo
        self.stats = CallStats()
        self._context_verified = False

    # ---- HTTP ----

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise SubtitisError(
                f"Ollama je vratila HTTP {exc.code} na {path}.\nOdgovor: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise SubtitisError(
                f"Ollama ne odgovara na {self.url} ({exc.reason}).\n"
                "  Proveri da servis radi:  ollama ps\n"
                "  Ako ne radi:             ollama serve"
            ) from exc
        except TimeoutError as exc:
            raise SubtitisError(
                f"Ollama nije odgovorila u {self.timeout:.0f}s. Model je verovatno "
                "još uvek učitavan u VRAM, ili je čank prevelik."
            ) from exc

    def _get(self, path: str) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(f"{self.url}{path}", timeout=30) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SubtitisError(f"Ollama ne odgovara na {self.url}{path}: {exc}") from exc

    # ---- provere ----

    def check(self) -> None:
        """Servis radi i model postoji. Zovi pre posla, ne usred njega."""
        tags = self._get("/api/tags")
        names = [m.get("name", "") for m in tags.get("models", [])]
        if self.model not in names:
            raise SubtitisError(
                f"Model '{self.model}' nije instaliran u Ollami.\n"
                f"  Dostupni: {', '.join(names) or '(nijedan)'}\n"
                f"  Instaliraj:  ollama pull {self.model}"
            )

    def residency(self) -> dict[str, Any] | None:
        """`/api/ps` unos za naš model: `size`, `size_vram`, `context_length`.

        Kad je `size_vram` manji od `size`, deo modela je u RAM-u i generisanje
        je višestruko sporije. To se ne vidi ni iz čega drugog osim iz brzine.
        """
        try:
            ps = self._get("/api/ps")
        except SubtitisError:
            return None
        for m in ps.get("models", []):
            if m.get("name") == self.model:
                return m
        return None

    def _verify_context(self) -> None:
        """Da li je `num_ctx` zaista primljen i da li model ceo stao u VRAM.

        Ollama tiho prihvata nepoznata polja, pa greška u imenu opcije prolazi
        bez traga — a posledica je tiho odsecanje prompta na 4096.
        """
        if self._context_verified:
            return
        self._context_verified = True
        m = self.residency()
        if m is None:
            return  # provera je korisna, ali nije razlog da posao padne

        actual = m.get("context_length")
        if actual is not None:
            if int(actual) < self.num_ctx:
                self.log(
                    f"UPOZORENJE: tražen num_ctx={self.num_ctx}, a Ollama drži model "
                    f"sa context_length={actual}. Duži promptovi će biti TIHO odsečeni. "
                    f"Proveri OLLAMA_CONTEXT_LENGTH i da li je model već bio učitan "
                    f"sa manjim kontekstom (`ollama stop {self.model}` pa ponovo)."
                )
            else:
                self.log(f"num_ctx potvrđen: {actual}")

        size, vram = m.get("size"), m.get("size_vram")
        if size and vram is not None:
            gb = 1024 ** 3
            if vram < size:
                self.log(
                    f"UPOZORENJE: model nije ceo u VRAM-u — {vram / gb:.1f} GB od "
                    f"{size / gb:.1f} GB ({100 * vram / size:.0f}%). Ostatak radi na CPU-u "
                    f"i generisanje je višestruko sporije. Smanji num_ctx ili uzmi manji model."
                )
            else:
                self.log(f"ceo u VRAM-u: {vram / gb:.1f} GB")

    # ---- poziv ----

    def chat(self, system: str, user: str) -> str:
        # Prirodna tačka prekida: pre poziva, nikad usred njega. Pokriva prevod,
        # restauraciju interpunkcije i skraćivanje — sve idu kroz ovaj poziv.
        cfg.checkpoint()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,  # bez ovoga je 7x sporije, bez ikakve koristi
            "options": self.options,
            "keep_alive": KEEP_ALIVE,
        }
        data = self._post("/api/chat", payload)
        message = data.get("message", {})
        content = message.get("content", "") or ""

        self.stats.requests += 1
        self.stats.prompt_tokens += int(data.get("prompt_eval_count") or 0)
        self.stats.output_tokens += int(data.get("eval_count") or 0)
        self.stats.eval_ns += int(data.get("eval_duration") or 0)
        self.stats.prompt_eval_ns += int(data.get("prompt_eval_duration") or 0)
        self.stats.load_ns += int(data.get("load_duration") or 0)

        if message.get("thinking") or "<think>" in content:
            self.stats.thinking_responses += 1
            if self.stats.thinking_responses == 1:
                self.log(
                    "UPOZORENJE: model je ipak razmišljao uprkos think=false. "
                    "Prevod će raditi, ali višestruko sporije. Proveri verziju Ollame "
                    "(`ollama --version`; polje `think` traži 0.9+)."
                )
            content = strip_thinking(content)

        self._verify_context()
        return content

    def unload(self) -> None:
        """Izbaci model iz VRAM-a. ASR i LLM nikad ne stoje zajedno."""
        try:
            self._post("/api/chat", {"model": self.model, "messages": [], "keep_alive": 0})
        except SubtitisError as exc:
            self.log(f"nije uspelo oslobađanje modela iz VRAM-a: {exc}")


def strip_thinking(text: str) -> str:
    """Ukloni `<think>...</think>` blok ako se ipak pojavi u sadržaju."""
    while "<think>" in text:
        start = text.index("<think>")
        end = text.find("</think>")
        if end < 0:
            return text[:start].strip()
        text = text[:start] + text[end + len("</think>") :]
    return text.strip()


# ==========================================================================
# Prevod
# ==========================================================================


@dataclass
class TranslationStats:
    sentences: int = 0
    chunks_ok: int = 0
    chunks_failed: int = 0
    failed_ids: list[int] = field(default_factory=list)
    #: rečenice čiji je prevod odbijen zbog neuravnoteženih navodnika
    quotes_rejected: int = 0
    calls: CallStats = field(default_factory=CallStats)


def translate_sentences(
    sentences: list[Sentence],
    src_lang: str,
    tgt_lang: str,
    *,
    client: OllamaClient,
    glossary_pairs: dict[str, str] | None = None,
    variant: str | None = None,
    fit_length: bool = False,
    chunk_size: int = CHUNK_SIZE,
    log: Callable[[str], None] | None = None,
) -> tuple[list[Sentence], TranslationStats]:
    """Prevedi rečenice. Vrati nove `Sentence` objekte i statistiku.

    Vremena, `id` i raspon reči se ne diraju — menja se samo `text`. Rečenica
    koju nismo uspeli da prevedemo zadržava original i dobija
    `translation_failed` flag; ceo posao se zbog nje nikad ne prekida.
    """
    warn = log or cfg.echo
    system = build_system_prompt(
        src_lang, tgt_lang, glossary_pairs or {}, variant, fit_length=fit_length
    )
    stats = TranslationStats(sentences=len(sentences))
    out: dict[int, str] = {}
    # Klijent skuplja ukupan zbir kroz sve pozive; nas zanima samo ovaj prolaz
    # (kod pivota se inače drugi skok prikazuje sa brojevima prvog).
    before = CallStats(**vars(client.stats))

    def context_before(lo: int) -> list[str]:
        return [s.text for s in sentences[max(0, lo - CONTEXT_BEFORE) : lo]]

    def context_after(hi: int) -> list[str]:
        return [s.text for s in sentences[hi : hi + CONTEXT_AFTER]]

    def try_once(lo: int, hi: int) -> tuple[dict[int, str] | None, str | None]:
        # Model vidi linije 1..n, a ne prave id-jeve: sa trocifrenim brojevima
        # (100..108) je vraćao 1..9 — prenumerisao je sam. Mapiranje nazad je
        # ovde. Provera broja linija ostaje: ona hvata spajanje i razdvajanje.
        real_ids = [sentences[k].id for k in range(lo, hi)]
        items = [(n, sentences[k].text) for n, k in enumerate(range(lo, hi), start=1)]
        user = build_user_prompt(context_before(lo), items, context_after(hi))
        response = client.chat(system, user)
        got, err = validate(response, [n for n, _ in items])
        if got is None:
            return None, err
        return {real_ids[n - 1]: text for n, text in got.items()}, None

    index = {s.id: k for k, s in enumerate(sentences)}

    def quotes_ok(sid: int, text: str) -> bool:
        """Navodnici uravnoteženi — ako su bili i u originalu. Rečenica
        presečena usred citata (trajanje, 6 cue-ova) nema uravnotežen prevod."""
        src = sentences[index[sid]].text
        return not quotes_balanced(src) or quotes_balanced(text)

    def handle(lo: int, hi: int, attempt: int = 0) -> None:
        ids = [sentences[k].id for k in range(lo, hi)]
        try:
            got, err = try_once(lo, hi)
        except SubtitisError:
            raise  # veza/model — nema smisla polovati, neka pukne jasno
        if got is not None:
            bad = [sid for sid in ids if not quotes_ok(sid, got[sid])]
            out.update({sid: t for sid, t in got.items() if sid not in bad})
            if not bad:
                stats.chunks_ok += 1
                return
            stats.quotes_rejected += len(bad)
            if hi - lo > 1:
                # Ostatak čanka je ispravan; ponavlja se samo pokvarena rečenica.
                stats.chunks_ok += 1
                warn(f"čank {ids}: neuravnoteženi navodnici u rečenici {bad}")
                for sid in bad:
                    k = index[sid]
                    warn(f"  -> ponavljam rečenicu {sid} (2. pokušaj)")
                    handle(k, k + 1, attempt + 1)
                return
            got, err = None, "neuravnoteženi navodnici"

        stats.chunks_failed += 1
        warn(f"čank {ids}: neispravan odgovor ({err})")

        if hi - lo > 1:
            mid = (lo + hi) // 2
            warn(f"  -> polovim na {ids[: mid - lo]} i {ids[mid - lo :]}")
            handle(lo, mid)
            handle(mid, hi)
            return

        if attempt + 1 < MAX_ATTEMPTS_PER_SENTENCE:
            warn(f"  -> ponavljam rečenicu {ids[0]} ({attempt + 2}. pokušaj)")
            handle(lo, hi, attempt + 1)
            return

        sent = sentences[lo]
        warn(f"  -> ODUSTAJEM od rečenice {sent.id}, upisujem original")
        out[sent.id] = sent.text
        stats.failed_ids.append(sent.id)

    for lo in range(0, len(sentences), chunk_size):
        handle(lo, min(lo + chunk_size, len(sentences)))

    stats.calls = CallStats(
        requests=client.stats.requests - before.requests,
        prompt_tokens=client.stats.prompt_tokens - before.prompt_tokens,
        output_tokens=client.stats.output_tokens - before.output_tokens,
        thinking_responses=client.stats.thinking_responses - before.thinking_responses,
        eval_ns=client.stats.eval_ns - before.eval_ns,
        prompt_eval_ns=client.stats.prompt_eval_ns - before.prompt_eval_ns,
        load_ns=client.stats.load_ns - before.load_ns,
    )
    translated = [
        s.model_copy(
            update={
                "text": out.get(s.id, s.text),
                "flags": (
                    list(s.flags) + [TRANSLATION_FAILED_FLAG]
                    if s.id in stats.failed_ids
                    else list(s.flags)
                ),
            }
        )
        for s in sentences
    ]
    return translated, stats


# ==========================================================================
# Skraćivanje prekoračenja — drugi prolaz, samo na zahtev
# ==========================================================================

#: Apsolutna granica („najviše X znakova"), ne relativno smanjenje. Izmereno
#: 2026-09-28 (Otvorena vrata 26, medijana stvarnog / traženog skraćenja, fr / en):
#: „kraće za najmanje N" 2,73× / 3,44×; opseg N..N+slack 2,36× / 2,93×;
#: „najviše X" 2,15× / 2,25×, uz najviše prihvaćenih i najviše cue-ova koji
#: izlaze iz prekoračenja. Reč „skrati" i trenutna dužina su izbačene iz
#: prompta jer vuku na relativno sečenje.
SHORTEN_SYSTEM_PROMPT = """Ti si urednik titlova na {TGT_NAME} jeziku. Svaki titl iz bloka TITLOVI mora da stane u dužinu koja je za njega zadata u bloku DUŽINE.

PRAVILA:
1. Linija sme imati NAJVIŠE onoliko znakova koliko piše za nju u bloku DUŽINE.
   To je granica, ne cilj: ostavi što više od postojećeg teksta. Menjaj samo
   onoliko koliko je potrebno da stane.
2. Značenje se ne sme izgubiti. Zadrži ko, šta, kome, kada, negaciju, brojeve
   i imena. Prvo izbaci ponavljanja, poštapalice i suvišne reči; biraj kraće
   izraze za isto.
3. Blok ORIGINAL služi samo za proveru značenja i ne prevodi se.
4. Vrati tačno onoliko linija koliko ih ima u bloku TITLOVI, sa istim rednim
   brojevima i formatom "broj|tekst". Ne spajaj i ne razdvajaj linije.
5. Ne dodaj objašnjenja. Prvi znak tvog odgovora je cifra.
6. Piši na {TGT_NAME} jeziku, latinicom. Navodnike zadrži kao u liniji.

REČNIK (termini se ne menjaju):
{GLOSSARY}"""


@dataclass
class ShortenItem:
    id: int
    text: str  # postojeći prevod
    source: str  # original, samo za proveru značenja
    need: int  # za koliko znakova najmanje mora biti kraće


@dataclass
class ShortenStats:
    items: int = 0
    chunks_ok: int = 0
    chunks_failed: int = 0
    failed_ids: list[int] = field(default_factory=list)
    calls: CallStats = field(default_factory=CallStats)


def shorten_slack(need: int) -> int:
    """Tolerancija preko traženog skraćenja, posle koje je predlog „prekomerno
    skraćen" (`overcut`) i u editoru nije izabran. Izmereno 2026-09-28: model
    je za traženih 1–2 znaka skraćivao i po 40–54, i tu je nastajao gotovo
    sav gubitak sadržaja. Ne ide u prompt — prompt nosi apsolutnu granicu."""
    return max(5, need // 2)


def build_shorten_prompt(items: list[tuple[int, ShortenItem]], src_lang: str) -> str:
    return "\n\n".join([
        f"ORIGINAL ({src_lang}):\n" + "\n".join(f"{n}: {it.source}" for n, it in items),
        "DUŽINE:\n" + "\n".join(
            f"{n}: najviše {len(it.text) - it.need} znakova" for n, it in items
        ),
        "TITLOVI:\n" + "\n".join(f"{n}{SEP}{it.text}" for n, it in items),
    ])


def shorten_sentences(
    items: list[ShortenItem],
    src_lang: str,
    tgt_lang: str,
    *,
    client: OllamaClient,
    glossary_pairs: dict[str, str] | None = None,
    chunk_size: int = CHUNK_SIZE,
    log: Callable[[str], None] | None = None,
) -> tuple[dict[int, str], ShortenStats]:
    """Predlozi kraćih verzija. Vraća samo one koje je model ispravno vratio.

    Ista validacija kao prevod: lokalni brojevi 1..n, skup brojeva, navodnici.
    Da li je predlog stvarno kraći i da li je CPS pao — to proverava pozivalac,
    posle prelamanja; ovde se ništa ne prihvata po „kvalitetu".
    """
    warn = log or cfg.echo
    system = SHORTEN_SYSTEM_PROMPT.format(
        TGT_NAME=_lang_name(tgt_lang, target=True),
        GLOSSARY=format_glossary(glossary_pairs or {}),
    )
    stats = ShortenStats(items=len(items))
    out: dict[int, str] = {}
    before = CallStats(**vars(client.stats))

    def handle(lo: int, hi: int, attempt: int = 0) -> None:
        chunk = items[lo:hi]
        local = list(enumerate(chunk, start=1))
        response = client.chat(system, build_shorten_prompt(local, src_lang))
        got, err = validate(response, [n for n, _ in local])
        if got is not None:
            bad = [n for n, it in local if quotes_balanced(it.text) and not quotes_balanced(got[n])]
            if not bad:
                out.update({it.id: got[n] for n, it in local})
                stats.chunks_ok += 1
                return
            err = "neuravnoteženi navodnici"
        stats.chunks_failed += 1
        ids = [it.id for it in chunk]
        warn(f"skraćivanje {ids}: neispravan odgovor ({err})")
        if hi - lo > 1:
            mid = (lo + hi) // 2
            handle(lo, mid)
            handle(mid, hi)
            return
        if attempt + 1 < MAX_ATTEMPTS_PER_SENTENCE:
            handle(lo, hi, attempt + 1)
            return
        warn(f"  -> odustajem od skraćivanja rečenice {ids[0]}, ostaje postojeći prevod")
        stats.failed_ids.append(ids[0])

    for lo in range(0, len(items), chunk_size):
        handle(lo, min(lo + chunk_size, len(items)))

    stats.calls = CallStats(
        requests=client.stats.requests - before.requests,
        prompt_tokens=client.stats.prompt_tokens - before.prompt_tokens,
        output_tokens=client.stats.output_tokens - before.output_tokens,
        thinking_responses=client.stats.thinking_responses - before.thinking_responses,
        eval_ns=client.stats.eval_ns - before.eval_ns,
        prompt_eval_ns=client.stats.prompt_eval_ns - before.prompt_eval_ns,
        load_ns=client.stats.load_ns - before.load_ns,
    )
    return out, stats
