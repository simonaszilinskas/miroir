"""Wire OpenRouter runners from settings: API key, HTTP clients, image caches."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import replace

import httpx

from miroir.commons.client import WikimediaClient, open_client
from miroir.commons.images import CommonsImages, commons_photos, open_commons_images
from miroir.config import Settings
from miroir.images.fetch import CONCURRENCY, ImageCache, ImageFetcher
from miroir.jobs.describe import OPTIONS as DESCRIBE_OPTIONS
from miroir.jobs.describe import DescribeOption, Describer
from miroir.jobs.embed import OPTIONS, OpenRouterEmbedder
from miroir.jobs.h4_agent import OPTIONS as AGENT_OPTIONS
from miroir.jobs.h4_agent import CallCache, Caller, ImageSource
from miroir.jobs.openrouter import OpenRouterClient
from miroir.jobs.privacy import PublishableImages
from miroir.jobs.spend import SpendLedger
from miroir.records.store import read_commons


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
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1024 * 1024)
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
    with open_vision(settings, option if batch else replace(option, batch=False)) as describer:
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
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1024 * 1024)
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http:
        try:
            yield Describer(
                settings, option, client, ImageFetcher(images_http, cache), prompts=prompts
            )
        finally:
            client.close()


@contextmanager
def open_agent_caller(settings: Settings, model: str, *, limit_usd: float) -> Generator[Caller]:
    """A `Caller` for step H4: one vision model, its saved replies, publishable images only."""
    if model not in AGENT_OPTIONS:
        msg = f"unknown H4 model {model!r}; known: {', '.join(AGENT_OPTIONS)}"
        raise ValueError(msg)
    if settings.openrouter_api_key is None:
        msg = "OPENROUTER_API_KEY is not set (see .env.example)"
        raise RuntimeError(msg)
    option = AGENT_OPTIONS[model]
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    client = OpenRouterClient(settings.openrouter_api_key)
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1024 * 1024)
    photos = commons_photos(settings)
    for path in settings.commons_gallery_parquets():  # F2 gallery photos, shown as thumbnails
        photos.update((p.image_id, p) for p in read_commons(path))
    with (
        httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http,
        open_client() as commons_http,
    ):
        images = ImageSource(
            PublishableImages.load(settings),
            ImageFetcher(images_http, cache),
            CommonsImages(photos, WikimediaClient(commons_http, settings.commons_cache_dir)),
            option.model,
        )
        try:
            yield Caller(
                option,
                client,
                images,
                CallCache(settings.h4_calls_path(option.slug)),
                SpendLedger(settings.spend_ledger),
                limit_usd=limit_usd,
            )
        finally:
            client.close()
