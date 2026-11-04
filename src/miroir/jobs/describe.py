"""Step H1: ask a vision model what a query photo shows, and keep its answer (experiments.md).

The model sees one query image and answers in French, as JSON: what it is, its likely name,
where it is, period, author, any text it can read, and how sure it is. `miroir.eval.vlm`
turns the answer into a text search over the records.

Rules, as for embeddings (`miroir.jobs.embed`): only images of `publishable` records go out
(`PublishableImages`), a 100-image probe measures the cost before a run, every cent goes to
the project ledger, and each answer is saved as it arrives so a stopped run resumes without
paying twice. Answers live in `Settings.vlm_answers_path(slug)`, one JSON line per image.
Each request holds its worst case against the budget while in flight (`RunSpend`). A batch is
written to `.pending` when submitted; once it comes back, its cost is logged and its answers
saved, and only then is it dropped from `.pending`, so a crash never loses a paid batch.
Commons query photos (openly licensed) go through the standard endpoint only: the Batch API
takes URLs, and these are fetched from Wikimedia on the Mac.
"""

from __future__ import annotations

import base64
import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from functools import partial
from itertools import batched, chain
from pathlib import Path
from typing import Protocol, cast

import httpx
from pydantic import ConfigDict, TypeAdapter, ValidationError

from miroir.commons.images import split_commons
from miroir.config import SEED, Settings
from miroir.eval.run import publishable_only
from miroir.eval.sets import load_queries, read_gallery
from miroir.images.fetch import LONG_SIDE, Fetched, ImageFetcher
from miroir.jobs.durable import append_lines, read_lines, rewrite_lines
from miroir.jobs.openrouter import JSON, ChatAnswer, OpenRouterClient, OpenRouterError
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.resumable import kept
from miroir.jobs.spend import (
    PROBE_IMAGES,
    Bill,
    Budget,
    BudgetError,
    Probe,
    RunSpend,
    SpendKind,
    SpendLedger,
    check_probe,
    check_run,
    load_probe,
    run_limit,
    save_probe,
)
from miroir.parallel import bounded_map
from miroir.records.model import Image

logger = logging.getLogger(__name__)

#: Parallel requests on the standard endpoint.
REQUEST_CONCURRENCY = 8
#: Requests per Batch API call. A batch can take several minutes whatever its size.
BATCH_SIZE = 250
MAX_TOKENS = 700
#: A run stops if it costs this many times its projection.
OVERRUN_FACTOR = 1.5
#: `Probe.resolution` of a batch run: the model gets the image as POP serves it (a URL).
FULL_SIZE = 0
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
    """Use the half-price Batch API (`model:batch` must exist). It sees images at another
    size and price than the standard endpoint, so the two never share a probe
    (`standard`)."""
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


class SuppliesImages(Protocol):
    """Commons query photos by id (`miroir.commons.images.CommonsImages`)."""

    def fetch_all(self, image_ids: Iterable[str]) -> Iterator[Fetched]: ...


def standard(option: DescribeOption) -> DescribeOption:
    """A batch option sent to the standard endpoint instead (`--no-batch`): full price, images
    at `LONG_SIDE`. Its own label, so it has its own probe and ledger lines; same answers file."""
    if not option.batch:
        return option
    return replace(option, option=f"{option.option}-standard", batch=False)


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
    """Saved answers by image id; the last line wins if an image was asked twice. A last line
    cut short by a crash is skipped (that image is asked again)."""
    answers = [_ANSWER.validate_json(line) for line in read_lines(path)]
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


@dataclass(frozen=True, slots=True)
class _Got:
    """Answers that came back, paid and logged, and the batch they came from (if any)."""

    answers: list[Answer]
    batch_id: str | None = None


class Describer:
    """Asks one model about POP images, saves each answer, logs every cent it spends."""

    def __init__(
        self,
        settings: Settings,
        option: DescribeOption,
        client: OpenRouterClient,
        fetcher: ImageFetcher,
        *,
        commons: SuppliesImages | None = None,
        request_concurrency: int = REQUEST_CONCURRENCY,
        batch_size: int = BATCH_SIZE,
        batch_poll_s: float = 30.0,
        prompts: Mapping[str, str] | None = None,
    ) -> None:
        """`prompts` gives some images a prompt of their own (T7's writer); the rest get
        `PROMPT`. `commons` fetches Commons query photos (T2, T2b, T8c)."""
        self.settings = settings
        self.option = option
        self._client = client
        self._fetcher = fetcher
        self._commons = commons
        self._concurrency = request_concurrency
        self._batch_size = batch_size
        self._poll_s = batch_poll_s
        self._prompts = prompts or {}
        self.ledger = SpendLedger(settings.spend_ledger)
        self.budget = Budget.from_settings(settings)
        self._store = PublishableImages.load(settings)
        self.answers_path = settings.vlm_answers_path(option.slug)

    @property
    def resolution(self) -> int:
        """The size the model sees: `LONG_SIDE`, or the image as POP serves it for a batch,
        which takes URLs (`FULL_SIZE`)."""
        return FULL_SIZE if self.option.batch else LONG_SIDE

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

    def spend(self, kind: SpendKind, *, limit_usd: float, per_item_usd: float) -> RunSpend:
        """The run's paid requests, each held against the budget, stopped past `limit_usd`."""
        return RunSpend(
            self.ledger,
            self.budget,
            self.key_usage(),
            limit_usd=limit_usd,
            per_item_usd=per_item_usd,
            kind=kind,
            option=self.option.option,
            run=f"{self.option.option}-{kind.value}",
            model=self.option.model,
        )

    def ask(self, image_ids: Sequence[str], spend: RunSpend) -> Asked:
        """Ask about every image not answered yet. Stops with BudgetError past the limit."""
        self.check_publishable(image_ids)
        done = self.answers()
        pending = self._pending()
        waiting = {i for ids in pending.values() for i in ids}
        pop, commons = split_commons(i for i in image_ids if i not in done and i not in waiting)
        if commons and (self.option.batch or self._commons is None):
            msg = (
                f"{len(commons)} Commons photo(s): they go to the standard endpoint "
                "(--no-batch) with a Commons fetcher"
            )
            raise ValueError(msg)
        todo = [self._store.images[i] for i in pop]
        skips: Counter[str] = Counter()
        kind = spend.kind
        answered = 0

        def save(got: _Got) -> None:
            nonlocal answered
            self._save(got.answers)  # answers first: the batch is dropped once they are kept
            if got.batch_id is not None:
                self._drop_pending(got.batch_id)
            answered += len(got.answers)

        if self.option.batch:
            asking = self._ask_batches(pending, todo, spend, kind, skips)
        else:
            fetched = chain(
                self._fetcher.fetch_all(todo),
                self._commons.fetch_all(commons) if self._commons else (),
            )
            asking = self._ask_one_by_one(
                (f for f in fetched if kept(f, skips)), spend, kind, skips, save
            )
        for got in asking:
            save(got)
            if answered % 100 < len(got.answers):
                logger.info(
                    "%s: %d/%d answered, $%.4f",
                    self.option.option,
                    answered,
                    len(todo) + len(commons),
                    spend.spent_usd,
                )
        if skips:
            logger.warning("%s: left out %s", self.option.option, dict(skips))
        return Asked(answered, spend.spent_usd, dict(skips))

    def _ask_one_by_one(
        self,
        fetched: Iterator[Fetched],
        spend: RunSpend,
        kind: SpendKind,
        skips: Counter[str],
        save: Callable[[_Got], None],
    ) -> Iterator[_Got]:
        failures = 0

        def salvage(got: tuple[str, ChatAnswer | OpenRouterError]) -> None:
            if not isinstance(got[1], OpenRouterError):
                save(_Got([self._answer(got[0], got[1], kind, got[1].cost_usd)]))

        chat = partial(self._chat_one, spend)
        for image_id, result in bounded_map(
            chat, fetched, workers=self._concurrency, salvage=salvage
        ):
            if isinstance(result, OpenRouterError):
                failures += 1
                skips["model_error"] += 1
                logger.warning("%s: %s", image_id, result)
                if failures >= MAX_FAILURES_IN_A_ROW:
                    msg = f"{failures} requests failed in a row; stopped (last: {result})"
                    raise OpenRouterError(msg) from result
                continue
            failures = 0
            yield _Got([self._answer(image_id, result, kind, result.cost_usd)])

    def _chat_one(
        self, spend: RunSpend, fetched: Fetched
    ) -> tuple[str, ChatAnswer | OpenRouterError]:
        data_url = "data:image/jpeg;base64," + base64.b64encode(fetched.jpeg or b"").decode()
        body = self._body(fetched.image_id, data_url)
        try:
            with spend.call(1) as bill:
                answer = self._client.chat(self.option.model, body)
                bill.cost_usd = answer.cost_usd
        except OpenRouterError as err:  # logged by the hold: estimated if it may have billed
            return fetched.image_id, err
        return fetched.image_id, answer

    def _ask_batches(
        self,
        pending: Mapping[str, Sequence[str]],
        images: Sequence[Image],
        spend: RunSpend,
        kind: SpendKind,
        skips: Counter[str],
    ) -> Iterator[_Got]:
        """Collect batches a stopped run left, then submit the rest, one batch at a time.

        The Batch API takes public image URLs only, not data URLs (checked 2026-10-12), so
        the image goes as its asset bucket URL. Not the IIIF service's: Google's fetcher timed
        out on 99 of 100 IIIF URLs, since IIIF resizes each image on demand.

        Each batch is held against the budget from its submission to its answer, and logged
        before it is handed over; the caller saves its answers, then drops it from `.pending`.
        """
        for batch_id, ids in pending.items():
            with spend.call(len(ids)) as bill:
                got = self._collect(batch_id, len(ids), kind, skips, bill)
            yield got
        model = self._client.batch_model(self.option.model)
        if model is None:
            msg = f"{self.option.model} has no :batch variant on OpenRouter"
            raise OpenRouterError(msg)
        for chunk in batched(images, self._batch_size):
            requests = {i.image_id: self._body(i.image_id, i.url) for i in chunk}
            with spend.call(len(chunk)) as bill:
                batch_id = self._client.submit_batch("/v1/chat/completions", model, requests)
                self._add_pending(batch_id, [i.image_id for i in chunk])
                got = self._collect(batch_id, len(chunk), kind, skips, bill)
            yield got

    def _collect(
        self, batch_id: str, size: int, kind: SpendKind, skips: Counter[str], bill: Bill
    ) -> _Got:
        answers, outcome = self._client.wait_chat_batch(batch_id, poll_s=self._poll_s)
        bill.cost_usd = outcome.cost_usd
        skips["model_error"] += size - len(answers)
        for cid, error in outcome.errors.items():
            logger.warning("%s: %s", cid, error[:300])
        if not answers:
            self._drop_pending(batch_id)  # finished, nothing to keep; its cost is logged
            msg = f"batch {outcome.batch_id} {outcome.status}: no answer came back"
            raise OpenRouterError(msg)
        share = outcome.cost_usd / len(answers)
        return _Got([self._answer(i, a, kind, share) for i, a in answers.items()], batch_id)

    # Batches submitted but not collected yet: "<batch id>\t<image id>,<image id>..." per line.

    @property
    def _pending_path(self) -> Path:
        return self.answers_path.with_suffix(".pending")

    def _pending(self) -> dict[str, list[str]]:
        lines = (line.decode() for line in read_lines(self._pending_path))
        return {b: ids.split(",") for b, ids in (line.split("\t") for line in lines)}

    def _add_pending(self, batch_id: str, image_ids: Sequence[str]) -> None:
        append_lines(self._pending_path, [f"{batch_id}\t{','.join(image_ids)}".encode()])

    def _drop_pending(self, batch_id: str) -> None:
        kept_batches = {b: ids for b, ids in self._pending().items() if b != batch_id}
        lines = (f"{b}\t{','.join(ids)}".encode() for b, ids in kept_batches.items())
        rewrite_lines(self._pending_path, lines)

    def _body(self, image_id: str, image_url: str) -> dict[str, JSON]:
        content: list[JSON] = [
            {"type": "text", "text": self._prompts.get(image_id, PROMPT)},
            {"type": "image_url", "image_url": {"url": image_url}},
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
        append_lines(self.answers_path, (_ANSWER.dump_json(a) for a in answers))


# --- entry points ---------------------------------------------------------------------------


def probe(describer: Describer, image_ids: Sequence[str], label: str) -> Probe:
    """Ask about 100 of the images and store the cost per image. Its answers are kept."""
    option = describer.option
    describer.check_publishable(image_ids)  # fail before paying
    limit = check_probe(describer.budget, describer.ledger, describer.key_usage())
    sample = random.Random(SEED).sample(list(image_ids), min(PROBE_IMAGES, len(image_ids)))
    spend = describer.spend(SpendKind.PROBE, limit_usd=limit, per_item_usd=limit / len(sample))
    describer.ask(sample, spend)
    answers = describer.answers()
    paid = [answers[i] for i in sample if i in answers]
    if not paid:
        msg = f"probe for {option.option}: no answer came back"
        raise RuntimeError(msg)
    result = Probe(
        option=option.option,
        model=option.model,
        resolution=describer.resolution,
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
    key_usage = describer.key_usage()
    projected = check_run(
        describer.budget,
        describer.ledger,
        key_usage,
        load_probe(describer.settings.probe_path(option.option)),
        option=option.option,
        model=option.model,
        resolution=describer.resolution,
        count=len(todo),
    )
    if projected > max_usd:
        msg = (
            f"{option.option}: {len(todo)} images projected at ${projected:.2f}, "
            f"over ${max_usd:.2f}"
        )
        raise BudgetError(msg)
    limit = run_limit(
        describer.budget,
        describer.ledger,
        key_usage,
        projected,
        factor=OVERRUN_FACTOR,
        margin_usd=0.05,
        max_usd=max_usd,
    )
    logger.info(
        "%s: %d images to ask (%d answered before), about $%.3f (stops past $%.3f)",
        option.option,
        len(todo),
        len(image_ids) - len(todo),
        projected,
        limit,
    )
    per_image = projected / max(len(todo), 1)
    spend = describer.spend(SpendKind.RUN, limit_usd=limit, per_item_usd=per_image)
    return describer.ask(todo, spend)
