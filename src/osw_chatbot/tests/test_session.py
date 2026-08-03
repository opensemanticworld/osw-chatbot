"""Regression tests for per-session isolation (issue #1).

Run with:  uv run pytest src/osw_chatbot/tests/test_session.py
"""

import asyncio
import base64
import json
import os
import time

import pytest

# The app modules build an LLM client at import time; stub the config so the
# tests do not need a real deployment. No request is ever sent.
os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.invalid/openai/v1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_MODEL", "gpt-5-mini")
# setdefault is not enough: a deployment .env usually defines the key with an
# empty value, and an empty secret disables token checking altogether - which
# would make every rejection test below pass for the wrong reason.
if not os.environ.get("CHATBOT_SHARED_SECRET"):
    os.environ["CHATBOT_SHARED_SECRET"] = "test-secret"

from osw_chatbot import auth  # noqa: E402

EXPECTED_TOOLS = [
    "multiply",
    "where_am_i",
    "highlight_html_element",
    "redirect",
    "full_text_search",
    "create_category_instance",
    "get_page_content",
    "smw_ask_query",
    "get_file_content",
]


def _mint(payload, secret="test-secret"):
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return encoded + "." + auth.sign(encoded, secret)


## token


def test_valid_token_yields_the_user():
    now = int(time.time())
    token = _mint({"u": "Alice", "w": "wiki", "iat": now, "exp": now + 600})
    assert auth.verify_token(token) == "Alice"


@pytest.mark.parametrize(
    "payload,secret",
    [
        ({"u": "Alice", "iat": 0, "exp": 1}, "test-secret"),  # expired
        ({"u": "Alice", "iat": int(time.time()), "exp": int(time.time()) + 60}, "other"),
        ({"w": "wiki", "iat": int(time.time()), "exp": int(time.time()) + 60}, "test-secret"),
    ],
)
def test_bad_tokens_are_rejected(payload, secret):
    assert auth.verify_token(_mint(payload, secret)) is None


def test_garbage_token_is_rejected():
    assert auth.verify_token("not-a-token") is None
    assert auth.verify_token("") is None


## history


def test_history_survives_a_round_trip():
    hist = pytest.importorskip("osw_chatbot.chat.history")
    from langchain_core.messages import AIMessage, HumanMessage

    messages = [HumanMessage(content="hello"), AIMessage(content="hi there")]
    restored = hist.loads(hist.dumps(messages))
    assert [m.content for m in restored] == ["hello", "hi there"]


def test_unreadable_history_degrades_to_empty():
    hist = pytest.importorskip("osw_chatbot.chat.history")
    assert hist.loads("{not json") == []
    assert hist.loads(json.dumps({"v": 999, "messages": []})) == []
    assert hist.loads(None) == []


def test_history_is_bounded():
    hist = pytest.importorskip("osw_chatbot.chat.history")
    from langchain_core.messages import HumanMessage

    from osw_chatbot.config import HISTORY_MAX_BYTES, HISTORY_MAX_MESSAGES

    messages = [HumanMessage(content=f"m{i}") for i in range(HISTORY_MAX_MESSAGES * 3)]
    assert len(hist.loads(hist.dumps(messages))) <= HISTORY_MAX_MESSAGES
    assert len(hist.dumps([HumanMessage(content="x" * 200000)])) <= HISTORY_MAX_BYTES


## sessions


@pytest.fixture()
def session():
    agent = pytest.importorskip("osw_chatbot.toolcalling.agent")
    instance = agent.ChatSession()
    yield instance
    instance.close()


def test_the_llm_sees_the_same_tools_as_before(session):
    names = [t.name for t in session.tools]
    assert names[: len(EXPECTED_TOOLS)] == EXPECTED_TOOLS
    for tool in session.tools:
        assert tool.description, f"{tool.name} lost its docstring"

    by_name = {t.name: t for t in session.tools}
    assert set(by_name["get_page_content"].args) == {"titles", "include_html"}
    assert set(by_name["multiply"].args) == {"a", "b"}


def test_two_sessions_share_nothing(session):
    from osw_chatbot.toolcalling.agent import ChatSession

    other = ChatSession()
    try:
        assert session.frontend is not other.frontend
        assert session.chat_history is not other.chat_history
        assert session.agent_executor is not other.agent_executor
        # The client-side tools must be bound to their own session's widget.
        # The websearch tools are deliberately excluded: they are still module
        # level (see the follow-up note in the README) but they never touch the
        # browser bridge, so they cannot cross-deliver between users.
        mine = {id(t) for t in session.tools[: len(EXPECTED_TOOLS)]}
        theirs = {id(t) for t in other.tools[: len(EXPECTED_TOOLS)]}
        assert mine.isdisjoint(theirs)
    finally:
        other.close()


def test_a_tool_call_resolves_only_on_its_own_id(session):
    async def scenario():
        pending = asyncio.ensure_future(
            session.call_client_side_tool("where_am_i", [], timeout=5)
        )
        await asyncio.sleep(0)
        call_id = session.frontend.function_call["id"]

        # a reply for a different call must not resolve this one
        session.frontend.function_called = {"id": "someone-else", "result": {"url": "b"}}
        await asyncio.sleep(0.05)
        assert not pending.done()

        session.frontend.function_called = {"id": call_id, "result": {"url": "a"}}
        return await pending

    assert asyncio.run(scenario()) == {"url": "a"}


def test_where_am_i_accepts_an_anonymous_user():
    """mw.user.getName() is null for anonymous wiki readers."""
    from osw_chatbot.toolcalling.agent import WebPage

    page = WebPage(
        url="https://wiki.example/x",
        title="X",
        content="<html></html>",
        user={"id": None, "name": "Anonymouse"},
    )
    assert page.user == {"id": None, "name": "Anonymouse"}
    assert WebPage(url="u", title="t", content="c").user is None


def test_client_tool_errors_are_routed_to_the_model(session):
    """Only ToolException reaches handle_tool_error - anything else escapes."""
    from langchain_core.tools import ToolException

    from osw_chatbot.toolcalling.agent import ClientToolError

    assert issubclass(ClientToolError, ToolException)
    for tool in session.tools[: len(EXPECTED_TOOLS)]:
        assert callable(tool.handle_tool_error), tool.name


def test_a_silent_browser_raises_instead_of_returning_none(session):
    from osw_chatbot.toolcalling.agent import ClientToolError

    async def scenario():
        with pytest.raises(ClientToolError):
            await session.call_client_side_tool("where_am_i", [], timeout=0.1)

    asyncio.run(scenario())
