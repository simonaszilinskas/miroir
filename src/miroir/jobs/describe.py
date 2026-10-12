"""Step H1: ask a vision model what a query photo shows, and keep its answer (experiments.md).

The model sees one query image and answers in French, as JSON: what it is, its likely name,
where it is, period, author, any text it can read, and how sure it is. `miroir.eval.vlm`
turns the answer into a text search over the records.

Rules, as for embeddings (`miroir.jobs.embed`): only images of `publishable` records go out
(`PublishableImages`), a 100-image probe measures the cost before a run, every cent goes to
the project ledger, and each answer is saved as it arrives so a stopped run resumes without
paying twice. Answers live in `Settings.vlm_answers_path(slug)`, one JSON line per image.
"""

from __future__ import annotations

import base64
import logging
import random
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import batched
from pathlib import Path
from typing import cast

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from miroir.config import SEED, Settings
from miroir.eval.run import publishable_only
from miroir.eval.sets import load_queries, read_gallery
from miroir.images.fetch import LONG_SIDE, Fetched, ImageFetcher
from miroir.jobs.openrouter import JSON, ChatAnswer, OpenRouterClient, OpenRouterError
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.spend import (
    PROBE_IMAGES,
    Budget,
    BudgetError,
    Probe,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    load_probe,
    save_probe,
)
from miroir.parallel import bounded_map

logger = logging.getLogger(__name__)

#: Parallel requests on the standard endpoint.
REQUEST_CONCURRENCY = 8
#: Requests per Batch API call: a stopped run loses at most one batch.
BATCH_SIZE = 50
MAX_TOKENS = 700
#: A run stops if it costs this many times its projection.
OVERRUN_FACTOR = 1.5
#: A run stops after this many requests fail in a row (out of credit, model gone).
MAX_FAILURES_IN_A_ROW = 5

PROMPT = """Tu es expert du patrimoine culturel français : monuments historiques, objets de \
musée, archives photographiques. Regarde cette image et réponds uniquement par un objet JSON, \
en français, avec ces clés :
- "type" : ce que montre l'image (type d'objet ou d'édifice, ou partie d'édifice : église, \
portail, retable, tableau, monnaie, vase…)
- "nom" : le nom probable de l'édifice, de l'œuvre ou de l'objet si tu le reconnais, sinon ""
- "commune", "departement", "region" : où il se trouve ou est conservé, si tu peux le dire, \
sinon ""
- "periode_style" : époque, siècle ou style probable
- "auteur" : artiste, architecte ou atelier probable, sinon ""
- "texte_visible" : tout texte lisible dans l'image (inscriptions, plaques, étiquettes, \
légendes, signatures), recopié tel quel, sinon ""
- "description" : une phrase sur le sujet représenté et les matériaux
- "confiance" : nombre de 0 à 1, ta confiance dans le nom et le lieu
N'invente rien : laisse vide ce que tu ne sais pas."""


@dataclass(frozen=True, slots=True)
class DescribeOption:
    option: str
    """Probe and ledger label, e.g. "H1-qwen3-vl-30b-a3b"."""
    model: str
    """OpenRouter model id."""
    slug: str
    batch: bool
    """Use the half-price Batch API (`model:batch` must exist)."""
    extra: Mapping[str, JSON]
    """Request fields this model needs beyond messages and max_tokens."""


OPTIONS: dict[str, DescribeOption] = {
    "qwen3-vl-30b-a3b": DescribeOption(
        "H1-qwen3-vl-30b-a3b",
        "qwen/qwen3-vl-30b-a3b-instruct",
        "qwen3-vl-30b-a3b",
        batch=False,  # no :batch variant on OpenRouter (2026-10-12)
        extra={"temperature": 0.0, "response_format": {"type": "json_object"}},
    ),
    "gemini-3.5-flash-lite": DescribeOption(
        "H1-gemini-3.5-flash-lite",
        "google/gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
        batch=True,
        # Reasoning cannot be turned off on this model; "minimal" is its default and cheapest.
        extra={"reasoning": {"effort": "minimal"}, "response_format": {"type": "json_object"}},
    ),
}


@dataclass(frozen=True, slots=True)
class Answer:
    """One model answer about one image, as saved."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    image_id: str
    model: str
    text: str
    """The model's raw answer; parse it with `parse_description`."""
    cost_usd: float
    """For a batch, the batch's cost shared equally among its answers."""
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    kind: SpendKind
    at: datetime


@dataclass(frozen=True, slots=True)
class Description:
    """What the model said, field by field. Empty strings where it did not say."""

    kind: str = ""
    name: str = ""
    commune: str = ""
    departement: str = ""
    region: str = ""
    period: str = ""
    author: str = ""
    visible_text: str = ""
    description: str = ""
    confidence: float | None = None


_ANSWER = TypeAdapter(Answer)
_OBJECT = TypeAdapter(dict[str, object])
_FIELDS = {
    "type": "kind",
    "nom": "name",
    "commune": "commune",
    "departement": "departement",
    "region": "region",
    "periode_style": "period",
    "auteur": "author",
    "texte_visible": "visible_text",
    "description": "description",
}
_EMPTY = {"", "inconnu", "inconnue", "non", "aucun", "aucune", "none", "null", "n/a", "-"}


def parse_description(text: str) -> Description | None:
    """The JSON object in a model answer, or None if there is none (fences and prose allowed)."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None
    try:
        raw = _OBJECT.validate_json(match.group(0))
    except ValidationError:
        return None
    fields: dict[str, str] = {}
    for key, value in raw.items():
        name = _FIELDS.get(_fold_key(key))
        if name is not None:
            fields[name] = _as_text(value)
    confidence = _as_confidence(raw.get("confiance", raw.get("confidence")))
    return Description(**fields, confidence=confidence)


def _fold_key(key: str) -> str:
    return key.strip().lower().replace("é", "e").replace(" ", "_").replace("/", "_")


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list | dict):
        items = cast("list[object]", value) if isinstance(value, list) else []
        if isinstance(value, dict):
            items = list(cast("dict[str, object]", value).values())
        return "; ".join(t for t in (_as_text(v) for v in items) if t)
    text = str(value).strip()
    return "" if text.lower() in _EMPTY else text


def _as_confidence(value: object) -> float | None:
    try:
        number = float(str(value).replace(",", ".").rstrip("%"))
    except ValueError:
        return None
    if number > 1:
        number /= 100
    return min(max(number, 0.0), 1.0)


def read_answers(path: Path) -> dict[str, Answer]:
    """Saved answers by image id; the last line wins if an image was asked twice."""
    if not path.exists():
        return {}
    with path.open("rb") as lines:
        answers = [_ANSWER.validate_json(line) for line in lines if line.strip()]
    return {a.image_id: a for a in answers}


def query_image_ids(settings: Settings, set_ids: Sequence[str], gallery: str) -> list[str]:
    """Query images of the sets' tune halves that may go out: publishable and in the gallery.

    The same queries every option is scored on (technical spec §7, closed models).
    """
    in_gallery = read_gallery(settings, gallery)
    ids: list[str] = []
    for set_id in set_ids:
        kept = publishable_only(
            settings, load_queries(settings, set_id, "tune"), in_gallery=in_gallery
        )
        ids.extend(q.query_image_id for q in kept)
    return list(dict.fromkeys(ids))


@dataclass(frozen=True, slots=True)
class Asked:
    answered: int
    cost_usd: float
    skips: Mapping[str, int]


class Describer:
    """Asks one model about POP images, saves each answer, logs every cent it spends."""

    def __init__(
        self,
        settings: Settings,
        option: DescribeOption,
        client: OpenRouterClient,
        fetcher: ImageFetcher,
        *,
        request_concurrency: int = REQUEST_CONCURRENCY,
        batch_size: int = BATCH_SIZE,
        batch_poll_s: float = 30.0,
    ) -> None:
        self.settings = settings
        self.option = option
        self._client = client
        self._fetcher = fetcher
        self._concurrency = request_concurrency
        self._batch_size = batch_size
        self._poll_s = batch_poll_s
        self.ledger = SpendLedger(settings.spend_ledger)
        self.budget = Budget.from_settings(settings)
        self._store = PublishableImages.load(settings)
        self.answers_path = settings.vlm_answers_path(option.slug)

    def answers(self) -> dict[str, Answer]:
        return read_answers(self.answers_path)

    def check_publishable(self, image_ids: Sequence[str]) -> None:
        self._store.check(image_ids, self.option.model)

    def key_usage(self) -> float | None:
        try:
            return self._client.key_spend().usage_usd
        except (OpenRouterError, httpx.HTTPError):
            logger.warning("could not read the key's spend from OpenRouter; using the ledger")
            return None

    def ask(self, image_ids: Sequence[str], *, kind: SpendKind, limit_usd: float) -> Asked:
        """Ask about every image not answered yet. Stops with BudgetError past `limit_usd`."""
        self.check_publishable(image_ids)
        done = self.answers()
        todo = [self._store.images[i] for i in image_ids if i not in done]
        skips: Counter[str] = Counter()
        fetched = (f for f in self._fetcher.fetch_all(todo) if _keep(f, skips))
        asking = self._ask_batches if self.option.batch else self._ask_one_by_one
        spent = 0.0
        answered = 0
        for answers in asking(fetched, kind, skips):
            cost = sum(a.cost_usd for a in answers)
            self.ledger.record(
                kind,
                option=self.option.option,
                run=f"{self.option.option}-{kind.value}",
                model=self.option.model,
                cost_usd=cost,
                items=len(answers),
            )
            self._save(answers)
            spent += cost
            answered += len(answers)
            if spent > limit_usd:
                msg = f"{self.option.option}: spent ${spent:.4f}, over its ${limit_usd:.4f} limit"
                raise BudgetError(msg)
            if answered % 100 < len(answers):
                logger.info(
                    "%s: %d/%d answered, $%.4f", self.option.option, answered, len(todo), spent
                )
        if skips:
            logger.warning("%s: left out %s", self.option.option, dict(skips))
        return Asked(answered, spent, dict(skips))

    def _ask_one_by_one(
        self, fetched: Iterator[Fetched], kind: SpendKind, skips: Counter[str]
    ) -> Iterator[list[Answer]]:
        failures = 0
        for image_id, result in bounded_map(self._chat_one, fetched, workers=self._concurrency):
            if isinstance(result, OpenRouterError):
                failures += 1
                skips["model_error"] += 1
                logger.warning("%s: %s", image_id, result)
                if failures >= MAX_FAILURES_IN_A_ROW:
                    msg = f"{failures} requests failed in a row; stopped (last: {result})"
                    raise OpenRouterError(msg) from result
                continue
            failures = 0
            yield [self._answer(image_id, result, kind, result.cost_usd)]

    def _chat_one(self, fetched: Fetched) -> tuple[str, ChatAnswer | OpenRouterError]:
        try:
            return fetched.image_id, self._client.chat(self.option.model, self._body(fetched))
        except OpenRouterError as err:
            return fetched.image_id, err

    def _ask_batches(
        self, fetched: Iterator[Fetched], kind: SpendKind, skips: Counter[str]
    ) -> Iterator[list[Answer]]:
        model = self._client.batch_model(self.option.model)
        if model is None:
            msg = f"{self.option.model} has no :batch variant on OpenRouter"
            raise OpenRouterError(msg)
        for chunk in batched(fetched, self._batch_size):
            requests = {f.image_id: self._body(f) for f in chunk}
            answers, outcome = self._client.chat_batch(model, requests, poll_s=self._poll_s)
            skips["model_error"] += len(chunk) - len(answers)
            for cid, error in outcome.errors.items():
                logger.warning("%s: %s", cid, error[:300])
            if not answers:
                if outcome.cost_usd:
                    self.ledger.record(
                        kind,
                        option=self.option.option,
                        run=f"{self.option.option}-{kind.value}",
                        model=model,
                        cost_usd=outcome.cost_usd,
                        items=0,
                    )
                msg = f"batch {outcome.batch_id} {outcome.status}: no answer came back"
                raise OpenRouterError(msg)
            share = outcome.cost_usd / len(answers)
            yield [self._answer(i, a, kind, share) for i, a in answers.items()]

    def _body(self, fetched: Fetched) -> dict[str, JSON]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(fetched.jpeg or b"").decode()
        content: list[JSON] = [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        return {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": MAX_TOKENS,
            **self.option.extra,
        }

    def _answer(self, image_id: str, chat: ChatAnswer, kind: SpendKind, cost: float) -> Answer:
        return Answer(
            image_id=image_id,
            model=self.option.model,
            text=chat.text,
            cost_usd=cost,
            prompt_tokens=chat.prompt_tokens,
            completion_tokens=chat.completion_tokens,
            finish_reason=chat.finish_reason,
            kind=kind,
            at=datetime.now(UTC),
        )

    def _save(self, answers: Sequence[Answer]) -> None:
        self.answers_path.parent.mkdir(parents=True, exist_ok=True)
        with self.answers_path.open("ab") as out:
            out.write(b"".join(_ANSWER.dump_json(a) + b"\n" for a in answers))
            out.flush()


# --- entry points ---------------------------------------------------------------------------


def probe(describer: Describer, image_ids: Sequence[str], label: str) -> Probe:
    """Ask about 100 of the images and store the cost per image. Its answers are kept."""
    option = describer.option
    describer.check_publishable(image_ids)  # fail before paying
    limit = check_probe(describer.budget, describer.ledger, describer.key_usage())
    sample = random.Random(SEED).sample(list(image_ids), min(PROBE_IMAGES, len(image_ids)))
    describer.ask(sample, kind=SpendKind.PROBE, limit_usd=limit)
    answers = describer.answers()
    paid = [answers[i] for i in sample if i in answers]
    if not paid:
        msg = f"probe for {option.option}: no answer came back"
        raise RuntimeError(msg)
    result = Probe(
        option=option.option,
        model=option.model,
        resolution=LONG_SIDE,
        gallery=label,
        images=len(paid),
        cost_usd=sum(a.cost_usd for a in paid),
        created=date.today(),  # noqa: DTZ011 (a calendar date for people)
    )
    save_probe(describer.settings.probe_path(option.option), result)
    logger.info("probe %s: %d images, $%.5f", option.option, result.images, result.cost_usd)
    return result


def run(describer: Describer, image_ids: Sequence[str], *, max_usd: float) -> Asked:
    """Ask about every image not answered yet. Needs a probe; refuses a run past `max_usd`."""
    option = describer.option
    describer.check_publishable(image_ids)  # fail before paying
    done = describer.answers()
    todo = [i for i in image_ids if i not in done]
    projected = check_run(
        describer.budget,
        describer.ledger,
        describer.key_usage(),
        load_probe(describer.settings.probe_path(option.option)),
        option=option.option,
        model=option.model,
        resolution=LONG_SIDE,
        count=len(todo),
    )
    if projected > max_usd:
        msg = (
            f"{option.option}: {len(todo)} images projected at ${projected:.2f}, "
            f"over ${max_usd:.2f}"
        )
        raise BudgetError(msg)
    limit = min(projected * OVERRUN_FACTOR + 0.05, max_usd)
    logger.info(
        "%s: %d images to ask (%d answered before), about $%.3f (stops past $%.3f)",
        option.option,
        len(todo),
        len(image_ids) - len(todo),
        projected,
        limit,
    )
    return describer.ask(todo, kind=SpendKind.RUN, limit_usd=limit)


def _keep(fetched: Fetched, skips: Counter[str]) -> bool:
    if fetched.skip is None:
        return True
    skips[f"fetch_{fetched.skip.value}"] += 1
    return False
