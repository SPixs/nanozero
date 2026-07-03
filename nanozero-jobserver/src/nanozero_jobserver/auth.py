"""API key authentication for protected endpoints.

Pattern: simple shared-secret X-API-Key header verified against the value in
ServerConfig.api_key. Empty config.api_key means dev mode — auth disabled
(any caller allowed). Always-on for non-empty config, regardless of source IP.

This is appropriate for the "trusted small" profile (b) per ADR-014. For public
volunteer grids (profile c-d) we'd need per-worker keys + rate limiting + crypto.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """FastAPI dependency: verify X-API-Key header against config.api_key.

    Args:
        request: injected by FastAPI; used to read app.state.config.
        x_api_key: value of the X-API-Key request header (None if absent).

    Returns:
        The verified API key string (empty string in dev mode).

    Raises:
        HTTPException 401: missing or mismatched key when auth is enabled.
    """
    config = request.app.state.config
    if not config.api_key:
        return ""  # dev mode — auth disabled
    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )
    return x_api_key
