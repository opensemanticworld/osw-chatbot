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

    # Re-imported rather than reloaded in place: the module reads its configuration at
    # import time, so a test that only set the environment would exercise whatever the
    # previous test left behind.
    import osw_chatbot.auth as auth_module
    import osw_chatbot.llmproxy.app as module

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
    assert seen["url"] == f"{UPSTREAM}/chat/completions"
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
