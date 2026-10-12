"""Wire OpenRouter runners from settings: API key, HTTP clients, image cache."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import httpx

from miroir.config import Settings
from miroir.images.fetch import CONCURRENCY, ImageCache, ImageFetcher
from miroir.jobs.describe import OPTIONS as DESCRIBE_OPTIONS
from miroir.jobs.describe import Describer
from miroir.jobs.embed import OPTIONS, OpenRouterEmbedder
from miroir.jobs.openrouter import OpenRouterClient


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
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http:
        try:
            yield OpenRouterEmbedder(
                settings, OPTIONS[option], client, ImageFetcher(images_http, cache)
            )
        finally:
            client.close()


@contextmanager
def open_describer(settings: Settings, model: str) -> Generator[Describer]:
    """A `Describer` for one vision model of `miroir.jobs.describe.OPTIONS` (step H1)."""
    if model not in DESCRIBE_OPTIONS:
        msg = f"unknown vision model {model!r}; known: {', '.join(DESCRIBE_OPTIONS)}"
        raise ValueError(msg)
    if settings.openrouter_api_key is None:
        msg = "OPENROUTER_API_KEY is not set (see .env.example)"
        raise RuntimeError(msg)
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    client = OpenRouterClient(settings.openrouter_api_key)
    cache = ImageCache(settings.image_cache_dir, settings.image_cache_mb * 1024 * 1024)
    with httpx.Client(timeout=60.0, limits=limits, follow_redirects=True) as images_http:
        try:
            yield Describer(
                settings, DESCRIBE_OPTIONS[model], client, ImageFetcher(images_http, cache)
            )
        finally:
            client.close()
