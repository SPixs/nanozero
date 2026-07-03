"""ASGI middleware: transparently inflate gzip-encoded request bodies.

FastAPI/Starlette only compresses *responses* (GZipMiddleware); it never inflates
*request* bodies. Self-play workers POST submits with ``Content-Encoding: gzip`` to
slash the WiFi bandwidth on DevSrv — the dense 119-plane input (~30 KB) + dense policy
target (~18.7 KB) per position compress ~3-5x. This middleware inflates the body
before the route handler parses JSON.

Backward-compatible by design: a request *without* ``Content-Encoding: gzip`` passes
through untouched, so the jobserver accepts both compressed (new) and uncompressed
(old) workers during a gradual fleet rollout — no flag day.
"""

from __future__ import annotations

import os
import zlib
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

# B.2 (A2-D1) — plafond anti zip-bomb. `gzip.decompress` non borné laissait un POST gzip
# de quelques Ko inflater en Go (ratio ~1000×) → OOM. On borne À LA FOIS le body compressé
# drainé ET la sortie dé-gzippée (décompression en streaming). Défaut 64 Mo : très au-dessus
# du plus gros submit fleet légitime (~34 Mo, partie de 512 plies en JSON base64) → zéro faux
# rejet ; borne la RAM par requête. Surchargeable via NANOZERO_JOBSERVER_MAX_BODY_BYTES.
MAX_REQUEST_BODY_BYTES = int(
    os.environ.get("NANOZERO_JOBSERVER_MAX_BODY_BYTES", str(64 * 1024 * 1024))
)


class _BodyTooLargeError(Exception):
    """Body compressé ou dé-gzippé au-delà du plafond → 413."""


def _safe_gunzip(data: bytes, max_out: int) -> bytes:
    """Décompresse un gzip en bornant la SORTIE à ``max_out`` octets.

    Lève ``_BodyTooLargeError`` si la sortie dé-gzippée dépasse ``max_out`` (zip-bomb),
    ``zlib.error`` si ``data`` n'est pas un gzip valide (l'appelant retombe sur le brut).
    """
    d = zlib.decompressobj(wbits=31)  # 31 = en-tête gzip
    out = d.decompress(data, max_out + 1)
    # unconsumed_tail non vide ⇒ borne atteinte avant de tout traiter ⇒ trop gros.
    if len(out) > max_out or d.unconsumed_tail:
        raise _BodyTooLargeError
    out += d.flush()
    if len(out) > max_out:
        raise _BodyTooLargeError
    return out


async def _respond_413(send: Send) -> None:
    """Réponse ASGI 413 Payload Too Large (corps non traité)."""
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [(b"content-type", b"text/plain; charset=utf-8")],
        }
    )
    await send({"type": "http.response.body", "body": b"Request body too large"})


class GzipRequestMiddleware:
    """Inflate ``Content-Encoding: gzip`` request bodies before downstream parsing."""

    def __init__(self, app: Callable[..., Awaitable[None]]) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = scope.get("headers", [])
        if not any(k.lower() == b"content-encoding" and v.lower() == b"gzip" for k, v in headers):
            await self.app(scope, receive, send)
            return

        # Drain the (compressed) body — borné : un body compressé > cap est déjà hostile.
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body.extend(message.get("body", b""))
            if len(body) > MAX_REQUEST_BODY_BYTES:
                await _respond_413(send)
                return
            if not message.get("more_body", False):
                break

        try:
            inflated = _safe_gunzip(bytes(body), MAX_REQUEST_BODY_BYTES)
        except _BodyTooLargeError:
            await _respond_413(send)
            return
        except (OSError, EOFError, zlib.error):
            inflated = bytes(body)  # header lied / not gzip — pass raw, let the route 4xx.

        # Strip content-encoding, fix content-length for the inflated body.
        new_headers = [
            (k, v) for (k, v) in headers if k.lower() not in (b"content-encoding", b"content-length")
        ]
        new_headers.append((b"content-length", str(len(inflated)).encode("latin-1")))
        new_scope = dict(scope)
        new_scope["headers"] = new_headers

        delivered = False

        async def patched_receive() -> Message:
            nonlocal delivered
            if delivered:
                return {"type": "http.disconnect"}
            delivered = True
            return {"type": "http.request", "body": inflated, "more_body": False}

        await self.app(new_scope, patched_receive, send)
