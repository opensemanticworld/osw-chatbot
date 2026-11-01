"""An LLM proxy for the in-browser sandbox.

The JupyterLite sandbox is a static site: whatever it ships is readable by anyone who can
read the wiki, so it can never hold an API key. This sits between it and the model
provider - it holds the key, and it decides whether the caller is allowed to use it by
verifying the signed token the wiki mints for that user.

Deliberately separate from the chat backend even though both live in this repository.
That one is `panel serve --num-procs 4`, a Bokeh websocket application with per-user
session state; this is a stateless streaming proxy. Bolting a streaming HTTP endpoint onto
the Bokeh process would conflate two unrelated lifecycles and inherit a process model it
has no use for.

Equally deliberately **not** a MediaWiki REST endpoint. Streaming a completion through
PHP-FPM holds a worker for the whole response; a handful of concurrent chats would starve
the wiki itself, which is a site-wide outage rather than a degraded sandbox.

## What it checks

Every request must carry `Authorization: Bearer <token>`, where the token is what
`action=chatbottoken` issues - HMAC-signed by the wiki with the shared secret, carrying
the username, an expiry and the action list. `osw_chatbot.auth.decode_token` verifies it,
so this process needs no connection to the wiki and no database.

Two consequences worth stating:

* **A request with no valid token gets nothing.** When `CHATBOT_SHARED_SECRET` is unset
  the wiki is not minting tokens either, and rather than silently running open this
  refuses to start. That is the opposite of the chat backend's behaviour, which falls back
  to anonymous - but the chat backend cannot spend money on a vendor API, and this can.
* **The action list is enforced.** A token issued to a user the wiki has not granted the
  sandbox to does not work here either, so enabling the action is the single switch.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from osw_chatbot.auth import decode_token

#: Where completions actually go. The same Foundry endpoint the chat backend uses, so a
#: deployment configures one provider rather than two.
UPSTREAM_BASE_URL = os.getenv(
    "LLM_PROXY_UPSTREAM_URL", os.getenv("OPENAI_BASE_URL", "")
).rstrip("/")
UPSTREAM_API_KEY = os.getenv("LLM_PROXY_UPSTREAM_KEY", os.getenv("OPENAI_API_KEY", ""))

#: The action a user needs for the sandbox. Mirrors $wgChatbotActions, which is what the
#: wiki signs into the token.
REQUIRED_ACTION = os.getenv("LLM_PROXY_REQUIRED_ACTION", "code-sandbox")

#: Origins allowed to call this. The sandbox is same-origin with its wiki, so in a normal
#: deployment this is only needed when the proxy is reached cross-origin.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("LLM_PROXY_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]

#: Completions can take minutes with a long context. The default httpx timeout of 5s
#: would cut nearly all of them off.
UPSTREAM_TIMEOUT = httpx.Timeout(
    connect=10.0, read=float(os.getenv("LLM_PROXY_READ_TIMEOUT", "600")), write=60.0, pool=10.0
)

app = FastAPI(title="osw llm proxy")

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        # Never "*" here, unlike the structured-output wrapper: this endpoint spends
        # money against a vendor account, so the set of callers is named explicitly.
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )


def _caller(authorization: str | None) -> dict[str, Any]:
    """The verified token payload, or 401/403."""
    if not os.getenv("CHATBOT_SHARED_SECRET", ""):
        # Refusing rather than running open. An unauthenticated proxy in front of a paid
        # API is a bill waiting to happen, and a quiet fallback would hide the
        # misconfiguration until it arrived.
        raise HTTPException(
            status_code=503,
            detail="CHATBOT_SHARED_SECRET is not set, so no token can be verified",
        )
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    payload = decode_token(authorization.split(" ", 1)[1].strip())
    if payload is None:
        raise HTTPException(status_code=401, detail="token is not valid or has expired")

    actions = payload.get("a")
    if REQUIRED_ACTION and REQUIRED_ACTION not in (actions or []):
        raise HTTPException(
            status_code=403,
            detail=f"this wiki has not enabled '{REQUIRED_ACTION}' for you",
        )
    return payload


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Unauthenticated on purpose: it reports configuration, never anything about a user."""
    return {
        "ok": bool(UPSTREAM_BASE_URL and UPSTREAM_API_KEY),
        "upstream_configured": bool(UPSTREAM_BASE_URL),
        "key_configured": bool(UPSTREAM_API_KEY),
        "auth_configured": bool(os.getenv("CHATBOT_SHARED_SECRET", "")),
        "required_action": REQUIRED_ACTION,
    }


@app.api_route("/v1/{path:path}", methods=["POST", "GET"])
async def passthrough(
    path: str, request: Request, authorization: str | None = Header(default=None)
) -> Any:
    """Forward an OpenAI-compatible call upstream, with our key rather than the caller's.

    A passthrough rather than a typed endpoint: the client is the AI SDK's
    OpenAI-compatible provider, and pinning a request schema here would mean editing this
    file every time it starts sending a new field. What is *not* passed through is the
    caller's Authorization header - it is replaced, which is the entire point.
    """
    payload = _caller(authorization)
    if not UPSTREAM_BASE_URL or not UPSTREAM_API_KEY:
        raise HTTPException(status_code=503, detail="no upstream model is configured")

    body = await request.body()
    wants_stream = False
    if body:
        try:
            wants_stream = bool(json.loads(body).get("stream"))
        except ValueError:
            wants_stream = False

    url = f"{UPSTREAM_BASE_URL}/{path}"
    headers = {
        "Authorization": f"Bearer {UPSTREAM_API_KEY}",
        "Content-Type": request.headers.get("content-type", "application/json"),
        # Who this is on behalf of, for the provider's own logs. Not trusted by anything
        # downstream - it is derived from a token we just verified, not from the client.
        "X-OSW-User": str(payload.get("u", "")),
    }

    client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    upstream = client.build_request(request.method, url, content=body, headers=headers)

    if not wants_stream:
        try:
            response = await client.send(upstream)
        finally:
            await client.aclose()
        return StreamingResponse(
            iter([response.content]),
            status_code=response.status_code,
            media_type=response.headers.get("content-type", "application/json"),
        )

    # Streaming: the response is handed back chunk by chunk, so the user sees tokens as
    # they arrive rather than waiting for the whole completion. The client is closed when
    # the generator finishes, not before - closing it here would cancel the stream.
    response = await client.send(upstream, stream=True)

    async def relay():
        try:
            async for chunk in response.aiter_raw():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "text/event-stream"),
    )
