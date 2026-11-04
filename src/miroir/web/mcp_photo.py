"""Photos an MCP client points at (`miroir.web.mcp`): a public https link, fetched safely.

The calling model (ChatGPT, Claude) gives a link, not bytes, so the server downloads it. Only
`https` on the default port; every address the name resolves to must be public (no loopback,
private, link-local or reserved range), and the request goes to the address that was checked,
with the name kept for TLS (`sni_hostname`), so a second lookup cannot point elsewhere.
Redirects are followed by hand, a few at most, each checked the same way. The download stops
at `MAX_PHOTO_BYTES` or after `FETCH_TIMEOUT_S`; Pillow must read it, at `MAX_PIXELS` at most.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from PIL import Image as PILImage

from miroir.config import WIKIMEDIA_USER_AGENT
from miroir.web.app import MAX_PHOTO_BYTES

logger = logging.getLogger(__name__)

#: The whole download, redirects included.
FETCH_TIMEOUT_S = 20.0
MAX_REDIRECTS = 3
#: Largest photo Pillow is asked to decode (a 40-megapixel camera photo passes).
MAX_PIXELS = 50_000_000

#: A host name and port to the addresses it resolves to.
Resolver = Callable[[str, int], Awaitable[list[str]]]


class PhotoError(ValueError):
    """A photo that cannot be used; the message is for the calling model."""


async def system_resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def public_address(address: str) -> bool:
    """True for an address on the public internet (an IPv4 inside IPv6 is judged as IPv4)."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


class PhotoFetcher:
    """Fetches a photo from a public https link. `transport` and `resolve` are for tests."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolve: Resolver = system_resolve,
    ) -> None:
        self._transport = transport
        self._resolve = resolve

    async def fetch(self, url: str) -> bytes:
        try:
            async with asyncio.timeout(FETCH_TIMEOUT_S):
                return await self._follow(url)
        except TimeoutError as err:
            msg = f"the image link did not answer within {FETCH_TIMEOUT_S:.0f} s"
            raise PhotoError(msg) from err
        except httpx.HTTPError as err:
            logger.info("photo link failed: %s", type(err).__name__)
            msg = "the image link could not be downloaded"
            raise PhotoError(msg) from err

    async def _follow(self, url: str) -> bytes:
        async with httpx.AsyncClient(
            transport=self._transport, follow_redirects=False, timeout=FETCH_TIMEOUT_S
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                host, target = await self._pinned(url)
                request = client.build_request(
                    "GET",
                    target,
                    headers={"Host": host, "User-Agent": WIKIMEDIA_USER_AGENT, "Accept": "image/*"},
                    extensions={"sni_hostname": host},
                )
                response = await client.send(request, stream=True)
                try:
                    if response.is_redirect and "location" in response.headers:
                        url = urljoin(url, response.headers["location"])
                        continue
                    if response.status_code != 200:
                        msg = f"the image link answered HTTP {response.status_code}"
                        raise PhotoError(msg)
                    return await _body(response)
                finally:
                    await response.aclose()
        msg = f"the image link redirects more than {MAX_REDIRECTS} times"
        raise PhotoError(msg)

    async def _pinned(self, url: str) -> tuple[str, str]:
        """The host name, and the URL with the host replaced by its checked public address."""
        parts = urlsplit(url.strip())
        host = parts.hostname
        if parts.scheme != "https" or not host:
            msg = "image_url must be an https:// link"
            raise PhotoError(msg)
        if parts.username or parts.password or parts.port not in (None, 443):
            msg = "image_url may not carry a user name, a password or a port"
            raise PhotoError(msg)
        try:
            addresses = await self._resolve(host, 443)
        except OSError as err:
            msg = f"the image host {host} is not known"
            raise PhotoError(msg) from err
        if not addresses or not all(public_address(a) for a in addresses):
            msg = "image_url must point to a public internet address"
            raise PhotoError(msg)
        address = addresses[0]
        netloc = f"[{address}]" if ":" in address else address
        return host, urlunsplit(("https", netloc, parts.path or "/", parts.query, ""))


async def _body(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_PHOTO_BYTES:
        raise PhotoError(_too_big())
    out = bytearray()
    async for chunk in response.aiter_bytes():
        out += chunk
        if len(out) > MAX_PHOTO_BYTES:
            raise PhotoError(_too_big())
    return bytes(out)


def _too_big() -> str:
    return f"the image is larger than {MAX_PHOTO_BYTES // (1024 * 1024)} MB"


def decode_base64(data: str) -> bytes:
    """A photo sent inline, as base64 or a data: URL."""
    if data.startswith("data:"):
        data = data.split(",", 1)[-1]
    if len(data) > MAX_PHOTO_BYTES * 4 // 3 + 4:
        raise PhotoError(_too_big())
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as err:
        msg = "image_base64 is not valid base64"
        raise PhotoError(msg) from err


def checked_photo(data: bytes) -> bytes:
    """The bytes, once Pillow reads them as an image of at most `MAX_PIXELS` (not decoded)."""
    try:
        with PILImage.open(io.BytesIO(data)) as opened:
            width, height = opened.size
    except (OSError, ValueError, PILImage.DecompressionBombError) as err:
        msg = "the file is not an image MIROIR can read (JPEG, PNG, WebP…)"
        raise PhotoError(msg) from err
    if width * height > MAX_PIXELS:
        msg = f"the image has more than {MAX_PIXELS // 1_000_000} million pixels"
        raise PhotoError(msg)
    return data
