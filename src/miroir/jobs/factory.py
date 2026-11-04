"""Wire OpenRouter runners from settings: API key, HTTP clients, image caches."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager

import httpx

from miroir.commons.client import WikimediaClient, open_client
from miroir.commons.images import CommonsImages, commons_photos, open_commons_images
from miroir.config import WIKIMEDIA_USER_AGENT, Settings
from miroir.images.fetch import CONCURRENCY, ImageFetcher, local_cache
from miroir.jobs.agent_calls import AgentKit, CallCache
from miroir.jobs.describe import OPTIONS as DESCRIBE_OPTIONS
from miroir.jobs.describe import DescribeOption, Describer, standard
from miroir.jobs.embed import OPTIONS, OpenRouterEmbedder
from miroir.jobs.openrouter import OpenRouterClient
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.spend import Budget, RunSpend, SpendKind, SpendLedger
from miroir.web.chat_model import OpenRouterModel


@contextmanager
def open_embedder(settings: Settings, option: str) -> Generator[OpenRouterEmbedder]:
    if option not in OPTIONS:
        msg = f"unknown OpenRouter option {option!r}; known: {', '.join(OPTIONS)}"
        raise ValueError(msg)
    if settings.openrouter_api_key is None:
        msg = "OPENROUTER_API_KEY is not set (see .env.example)"
        raise RuntimeError(msg)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    client = OpenRouterClient(settings.openrouter_api_key)
    cache = local_cache(settings)
    with (
        httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http,
        open_commons_images(settings) as commons,
    ):
        try:
            yield OpenRouterEmbedder(
                settings, OPTIONS[option], client, ImageFetcher(images_http, cache), commons=commons
            )
        finally:
            client.close()


@contextmanager
def open_describer(settings: Settings, model: str, *, batch: bool = True) -> Generator[Describer]:
    """A `Describer` for one vision model of `miroir.jobs.describe.OPTIONS` (step H1).

    `batch=False` sends to the standard endpoint even when the model has a batch variant.
    """
    if model not in DESCRIBE_OPTIONS:
        msg = f"unknown vision model {model!r}; known: {', '.join(DESCRIBE_OPTIONS)}"
        raise ValueError(msg)
    option = DESCRIBE_OPTIONS[model]
    with open_vision(settings, option if batch else standard(option)) as describer:
        yield describer


@contextmanager
def open_vision(
    settings: Settings, option: DescribeOption, prompts: Mapping[str, str] | None = None
) -> Generator[Describer]:
    """A `Describer` for any vision option, with per-image prompts if given (T7's writer)."""
    if settings.openrouter_api_key is None:
        msg = "OPENROUTER_API_KEY is not set (see .env.example)"
        raise RuntimeError(msg)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    client = OpenRouterClient(settings.openrouter_api_key)
    cache = local_cache(settings)
    with (
        httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http,
        open_commons_images(settings) as commons,
    ):
        try:
            yield Describer(
                settings,
                option,
                client,
                ImageFetcher(images_http, cache),
                commons=commons,
                prompts=prompts,
            )
        finally:
            client.close()


#: The ledger option of the offline agent's calls, and its expected cost per call (H4's S3
#: and S4 paid about $0.0015 a call; the site's calls reason a little and send more).
AGENT_OPTION = "H5-offline"
AGENT_SLUG = "mistral-large-4"
EXPECTED_CALL_USD = 0.004


@contextmanager
def open_agent_kit(
    settings: Settings, *, run: str, limit_usd: float, key_usage_usd: float | None
) -> Generator[AgentKit]:
    """What `miroir.eval.h4` needs to run the site's agent offline: the site's model through
    OpenRouter, its saved answers, the run's spend (held and settled in the project ledger),
    and the images. Query photos come from the image cache or Commons; thumbnails by URL, as
    the site fetches them."""
    if settings.openrouter_api_key is None:
        msg = "OPENROUTER_API_KEY is not set (see .env.example)"
        raise RuntimeError(msg)
    model = OpenRouterModel(settings.openrouter_api_key)
    spend = RunSpend(
        SpendLedger(settings.spend_ledger),
        Budget.from_settings(settings),
        key_usage_usd,
        limit_usd=limit_usd,
        per_item_usd=EXPECTED_CALL_USD,
        kind=SpendKind.RUN,
        option=AGENT_OPTION,
        run=run,
        model=model.model,
    )
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    cache = local_cache(settings)
    store = PublishableImages.load(settings)
    photos = commons_photos(settings)
    with (
        httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http,
        httpx.Client(
            timeout=30.0, follow_redirects=True, headers={"User-Agent": WIKIMEDIA_USER_AGENT}
        ) as thumbs_http,
        open_client() as commons_http,
    ):
        fetcher = ImageFetcher(images_http, cache)
        commons = CommonsImages(photos, WikimediaClient(commons_http, settings.commons_cache_dir))
        lock = threading.Lock()

        def photo(image_id: str) -> bytes:
            if image_id in store.images:
                fetched = fetcher.fetch(store.images[image_id])
            else:
                with lock:  # the Commons client keeps Wikimedia's pace, one request at a time
                    fetched = commons.fetch(image_id)
            if fetched.jpeg is None:
                msg = f"{image_id}: could not fetch the photo ({fetched.skip})"
                raise RuntimeError(msg)
            return fetched.jpeg

        async def thumb(url: str) -> bytes:
            def get() -> bytes:
                response = thumbs_http.get(url)
                response.raise_for_status()
                return response.content

            return await asyncio.to_thread(get)

        yield AgentKit(
            model=model,
            cache=CallCache(settings.h4_calls_path(f"{AGENT_OPTION}-{AGENT_SLUG}")),
            spend=spend,
            photo=photo,
            fetch=thumb,
        )
