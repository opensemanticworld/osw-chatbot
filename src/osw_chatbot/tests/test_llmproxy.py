"""The LLM proxy: who gets through, and what reaches the model.

The thing worth testing here is not the forwarding - that is a handful of lines - but the
refusals. This endpoint spends money against a vendor account on behalf of whoever calls
it, so every way of *not* being allowed to is asserted explicitly.
"""

from __future__ import annotations

import base64
import importlib
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

SECRET = "test-shared-secret"
UPSTREAM = "https://upstream.invalid/v1"
UPSTREAM_KEY = "upstream-key"


@pytest.fixture
def proxy(monkeypatch):
    """The app, configured, with a stub upstream that echoes what it was sent."""
    monkeypatch.setenv("CHATBOT_SHARED_SECRET", SECRET)
    monkeypatch.setenv("LLM_PROXY_UPSTREAM_URL", UPSTREAM)
    monkeypatch.setenv("LLM_PROXY_UPSTREAM_KEY", UPSTREAM_KEY)
    monkeypatch.setenv("LLM_PROXY_REQUIRED_ACTION", "code-sandbox")
    monkeypatch.setenv("LLM_PROXY_AUTH_HEADER", "authorization")
    monkeypatch.setenv("LLM_PROXY_AUTH_SCHEME", "Bearer")

    # Reloaded in dependency order, because each of these reads its configuration once
    # at import. config holds the secret; auth binds it by value with a `from ... import`,
    # so reloading auth alone would re-bind whatever config still had - the value left by
    # whichever test file imported it first, which made this file pass alone and fail
    # alongside test_session.
    import osw_chatbot.auth as auth_module
    import osw_chatbot.config as config_module
    import osw_chatbot.llmproxy.app as module

    importlib.reload(config_module)
    importlib.reload(auth_module)
    module = importlib.reload(module)

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.content.decode() or ""
        if json.loads(seen["body"] or "{}").get("stream"):
            # An async generator, not a bytes body: a Response built from bytes is
            # already consumed, so aiter_raw() on it raises StreamConsumed and the test
            # would fail for a reason the real upstream never produces.
            async def chunks():
                yield b'data: {"delta":"one"}\n\n'
                yield b'data: {"delta":"two"}\n\n'
                yield b"data: [DONE]\n\n"

            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=chunks()
            )
        return httpx.Response(200, json={"ok": True, "echo": seen["body"]})

    real_client = httpx.AsyncClient

    def stubbed(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(module.httpx, "AsyncClient", stubbed)
    return TestClient(module.app), seen


def mint(actions=("code-sandbox",), user="alice", ttl=3600, secret=SECRET) -> str:
    from osw_chatbot.auth import sign

    payload = {
        "u": user,
        "w": "wiki",
        "iat": time.time(),
        "exp": time.time() + ttl,
        "n": "nonce",
        "a": list(actions),
        "ca": [],
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"{encoded}.{sign(encoded, secret)}"


def auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_no_token_is_refused(proxy):
    client, _ = proxy
    assert client.post("/v1/chat/completions", json={}).status_code == 401


def test_garbage_token_is_refused(proxy):
    client, _ = proxy
    response = client.post("/v1/chat/completions", json={}, headers=auth("nonsense"))
    assert response.status_code == 401


def test_token_signed_with_another_secret_is_refused(proxy):
    """The signature is the whole guarantee - a well-formed payload is not enough."""
    client, _ = proxy
    forged = mint(secret="not-the-shared-secret")
    assert client.post("/v1/chat/completions", json={}, headers=auth(forged)).status_code == 401


def test_expired_token_is_refused(proxy):
    client, _ = proxy
    assert client.post(
        "/v1/chat/completions", json={}, headers=auth(mint(ttl=-10))
    ).status_code == 401


def test_token_without_the_action_is_refused(proxy):
    """Enabling the sandbox is one switch on the wiki, and it is honoured here too."""
    client, _ = proxy
    response = client.post(
        "/v1/chat/completions", json={}, headers=auth(mint(actions=("websearch",)))
    )
    assert response.status_code == 403


def test_valid_token_reaches_the_model_with_our_key(proxy):
    client, seen = proxy
    response = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": []},
        headers=auth(mint()),
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    # The whole path survives. Anthropic lives at <base>/v1/messages, so a proxy that
    # swallowed the /v1 would 404 against the vendor for reasons invisible from here.
    assert seen["url"] == f"{UPSTREAM}/v1/chat/completions"
    # The caller's credential is replaced, never forwarded. This is the point of the
    # whole service: the browser never holds the key.
    assert seen["headers"]["authorization"] == f"Bearer {UPSTREAM_KEY}"
    assert seen["headers"]["x-osw-user"] == "alice"


def test_streaming_is_relayed(proxy):
    client, _ = proxy
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "m", "messages": [], "stream": True},
        headers=auth(mint()),
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        body = b"".join(response.iter_bytes()).decode()
    assert '"delta":"one"' in body and "[DONE]" in body


def test_without_a_shared_secret_it_refuses_rather_than_running_open(monkeypatch):
    """An unauthenticated proxy in front of a paid API is a bill waiting to happen."""
    monkeypatch.setenv("CHATBOT_SHARED_SECRET", "")
    monkeypatch.setenv("LLM_PROXY_UPSTREAM_URL", UPSTREAM)
    monkeypatch.setenv("LLM_PROXY_UPSTREAM_KEY", UPSTREAM_KEY)
    import osw_chatbot.auth as auth_module
    import osw_chatbot.llmproxy.app as module

    importlib.reload(auth_module)
    module = importlib.reload(module)
    client = TestClient(module.app)
    assert client.post("/v1/chat/completions", json={}, headers=auth(mint())).status_code == 503


def test_anthropic_style_token_in_x_api_key(proxy):
    """The AI SDK's Anthropic provider sends the key as x-api-key, not a bearer token."""
    client, seen = proxy
    response = client.post(
        "/v1/messages",
        json={"model": "claude", "messages": []},
        headers={"x-api-key": mint()},
    )
    assert response.status_code == 200
    assert seen["url"] == f"{UPSTREAM}/v1/messages"


def test_the_callers_credential_never_reaches_the_vendor(proxy):
    """Both headers a caller might authenticate with are replaced, not forwarded."""
    client, seen = proxy
    token = mint()
    client.post("/v1/messages", json={}, headers={"x-api-key": token})
    assert token not in json.dumps(seen["headers"])
    assert seen["headers"]["authorization"] == f"Bearer {UPSTREAM_KEY}"


def test_anthropic_upstream_shape(monkeypatch):
    """With LLM_PROVIDER=claude the upstream is authenticated by x-api-key, no scheme."""
    monkeypatch.setenv("CHATBOT_SHARED_SECRET", SECRET)
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://foundry.invalid/anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("LLM_PROXY_UPSTREAM_URL", raising=False)
    monkeypatch.delenv("LLM_PROXY_UPSTREAM_KEY", raising=False)
    monkeypatch.delenv("LLM_PROXY_AUTH_HEADER", raising=False)
    monkeypatch.delenv("LLM_PROXY_AUTH_SCHEME", raising=False)

    import osw_chatbot.auth as auth_module
    import osw_chatbot.llmproxy.app as module

    importlib.reload(auth_module)
    module = importlib.reload(module)

    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"ok": True})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        module.httpx,
        "AsyncClient",
        lambda *a, **k: real_client(*a, **{**k, "transport": httpx.MockTransport(handler)}),
    )

    client = TestClient(module.app)
    response = client.post(
        "/v1/messages",
        json={"model": "claude-sonnet-5", "messages": []},
        headers={"x-api-key": mint(), "anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200
    assert seen["url"] == "https://foundry.invalid/anthropic/v1/messages"
    # No "Bearer " prefix: Anthropic takes the raw key.
    assert seen["headers"]["x-api-key"] == "anthropic-key"
    # Client headers the vendor needs are forwarded; the caller's credential is not.
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
