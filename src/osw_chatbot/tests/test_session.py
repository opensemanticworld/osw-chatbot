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

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from osw_chatbot import auth  # noqa: E402

EXPECTED_TOOLS = [
    "multiply",
    "where_am_i",
    "highlight_html_element",
    "redirect",
    "full_text_search",
    "find_page_from_topic",
    "create_category_instance",
    "get_category_schema",
    "get_page_content",
    "smw_ask_query",
    "get_file_content",
    "edit_page",
    "delete_page",
    "upload_file",
    "load_skill",
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


## write tools, actions and skills


def test_edit_page_documents_the_slot_data_model(session):
    """The docstring is the only place the model learns what a slot takes."""
    tool = {t.name: t for t in session.tools}["edit_page"]
    assert set(tool.args) == {"title", "slots", "summary"}
    doc = tool.description
    for slot in ("jsondata", "jsonschema", "main", "header", "footer", "schema_template"):
        assert slot in doc, f"{slot} missing from the edit_page description"
    assert "Markdown" in doc  # the mistake it has to avoid
    assert "User:" in doc  # the user-scope rule for scripts


def test_a_refusal_from_the_browser_is_raised(session):
    """Write tools report refusals as data; the model must not read that as success."""
    from osw_chatbot.toolcalling.agent import ClientToolError

    async def scenario():
        pending = asyncio.ensure_future(
            session.call_client_side_tool("edit_page", ["Item:X", {}], timeout=5)
        )
        await asyncio.sleep(0)
        call_id = session.frontend.function_call["id"]
        session.frontend.function_called = {
            "id": call_id,
            "result": {"ok": False, "code": "permissiondenied", "message": "nope"},
        }
        with pytest.raises(ClientToolError, match="nope"):
            await pending

    asyncio.run(scenario())


def test_websearch_is_gated_by_the_signed_actions():
    agent = pytest.importorskip("osw_chatbot.toolcalling.agent")

    # the backend-side tools, as opposed to the wiki's own full_text_search
    websearch = {"web_search_tool", "simple_browse_web_page_tool"}

    without = agent.ChatSession(actions=[])
    with_search = agent.ChatSession(actions=["websearch"])
    try:
        assert websearch.isdisjoint({t.name for t in without.tools})
        assert websearch <= {t.name for t in with_search.tools}
    finally:
        without.close()
        with_search.close()


def test_token_carries_the_action_lists():
    from osw_chatbot.auth import token_actions

    now = int(time.time())
    token = _mint({"u": "Alice", "iat": now, "exp": now + 600,
                   "a": ["websearch", "edit"], "ca": ["script"]})
    assert token_actions(token) == (["websearch", "edit"], ["script"])
    # no / invalid token: fall back to search only, never to write access
    assert token_actions("garbage") == (["websearch"], [])


def test_skills_are_indexed_and_loadable(session):
    from osw_chatbot.skills import skill_index

    assert {"mediawiki-component", "osl-page-editing"} <= set(session.skills)
    index = skill_index(session.skills)
    for name in session.skills:
        assert name in index

    body = session.skills["osl-page-editing"].body
    # the two things this skill exists to stop / enable
    assert "'''bold'''" in body and "**bold**" in body
    assert "format=datatables" in body

    # the component skill has to carry the data APIs an app actually needs,
    # otherwise the model invents endpoints
    # the entity-first rule and the slot trap the model actually fell into
    assert "Never write structured data to `main`" in body
    # reference fields: the model kept writing labels into autocomplete slots
    assert "format\": \"autocomplete\"" in body or "autocomplete" in body
    assert "subclassof_range" in body and "uuid without hyphens" in body
    assert "find_page_from_topic" in body and "Wiki File" in body

    component = session.skills["mediawiki-component"].body
    for needle in ("action: 'ask'", "rvslots", "mwjson.api.updatePage", "postProcessPage"):
        assert needle in component, f"{needle} missing from mediawiki-component"
    # Writing must go through mwjson, which fills the header/footer slots an
    # entity needs - a raw action=edit leaves the page rendering empty.
    assert "postWithToken" not in component


def test_load_skill_rejects_an_unknown_name(session):
    tool = {t.name: t for t in session.tools}["load_skill"]

    async def scenario():
        assert "wikitext" in (await tool.ainvoke({"name": "osl-page-editing"})).lower()
        # handle_tool_error turns the refusal into an observation rather than
        # letting it escape as a traceback - the model is told, and recovers.
        answer = await tool.ainvoke({"name": "nope"})
        assert "no skill" in answer and "osl-page-editing" in answer

    asyncio.run(scenario())


## attachments, multimodal turns and progress


def test_an_image_attachment_reaches_the_model_as_an_image_block(session):
    """A string template would stringify the list - the model would get a repr."""
    data_url = "data:image/jpeg;base64,/9j/4AAQSkZJRg=="
    content = session._content(
        "what is this?",
        [{"id": "att-1", "name": "device.jpg", "mime": "image/jpeg",
          "size": 1234, "preview": data_url}],
    )
    assert isinstance(content, list)
    images = [b for b in content if b.get("type") == "image_url"]
    assert len(images) == 1 and images[0]["image_url"]["url"] == data_url
    text = next(b for b in content if b.get("type") == "text")["text"]
    assert "att-1" in text and "device.jpg" in text   # so it can call upload_file

    # and the prompt alone stays a plain string when nothing is attached
    assert session._content("hello", []) == "hello"


def test_an_attachment_never_lands_in_the_stored_history(session):
    from osw_chatbot.chat import history as hist

    session.chat_history = [
        HumanMessage(content="what is this?"),
        AIMessage(content="a multimeter"),
    ]
    assert "data:image" not in hist.dumps(session.chat_history)


def test_upload_is_gated_like_the_other_writes(session):
    tool = {t.name: t for t in session.tools}["upload_file"]
    assert set(tool.args) == {"attachment_id", "label"}
    assert "File:" in tool.description


def test_edit_page_insists_on_entities_and_the_jsondata_slot(session):
    doc = {t.name: t for t in session.tools}["edit_page"].description
    assert "instance" in doc and "find_page_from_topic" in doc
    assert "User:" in doc and ".js" in doc          # the exceptions
    assert "Never put structured data into 'main'" in doc
    assert "autocomplete" in doc and "not free text" in doc


def test_progress_summaries_say_something_useful():
    """A summary that reports only a byte count tells the user nothing."""
    from osw_chatbot.toolcalling.progress import MAX_DETAIL, describe

    page_query = {"query": {"pages": {"1": {
        "title": "Item:OSW123",
        "revisions": [{"slots": {"main": {}, "jsondata": {}, "header": {}}}],
    }}}}
    got = describe("get_page_content", page_query)
    assert "Item:OSW123" in got and "jsondata" in got

    schema = {"id": "x", "allOf": [
        {"properties": {"label": {}, "description": {}}},
        {"properties": {"image": {}, "manufacturer": {}}},
    ], "required": ["label"]}
    got = describe("get_category_schema", schema)
    assert "4 properties" in got and "label" in got and "required" in got

    assert "2 results" in describe(
        "smw_ask_query", {"query": {"results": {"Item:A": {}, "Item:B": {}}}})
    assert "2 matches" in describe(
        "find_page_from_topic", [{"label": "Device"}, {"label": "Sensor"}])
    assert "Item:OSW999" in describe(
        "edit_page", {"ok": True, "title": "Item:OSW999", "slots": ["jsondata"]})

    # where_am_i hands back a pydantic model, not a dict
    from osw_chatbot.toolcalling.agent import WebPage

    got = describe("where_am_i", WebPage(
        url="https://w/x", title="Harness page", content="<html/>",
        user={"id": "Alice", "name": "Alice"}))
    assert "Harness page" in got and "Alice" in got and "<html" not in got

    # refusals keep showing why
    assert "no permission" in describe(
        "edit_page", {"ok": False, "code": "permissiondenied", "message": "no permission"})

    # and nothing ever grows unbounded or pastes markup in
    long = describe("get_file_content", "<html>" + "x" * 5000)
    assert len(long) <= MAX_DETAIL and "<html" not in long


def test_progress_labels_stay_short_and_leak_nothing(session):
    from osw_chatbot.toolcalling.progress import LABELS, MAX_ARG, summarize

    assert summarize({"titles": ["Item:OSW123"], "include_html": True}) == "Item:OSW123"
    assert summarize({"query": "x" * 500}).endswith("…")
    assert len(summarize({"query": "x" * 500})) <= MAX_ARG
    # every registered tool should have a human label
    for name in EXPECTED_TOOLS:
        if name not in ("multiply",):
            assert name in LABELS, f"no progress label for {name}"


def test_a_silent_browser_raises_instead_of_returning_none(session):
    from osw_chatbot.toolcalling.agent import ClientToolError

    async def scenario():
        with pytest.raises(ClientToolError):
            await session.call_client_side_tool("where_am_i", [], timeout=0.1)

    asyncio.run(scenario())
