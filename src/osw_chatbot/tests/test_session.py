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
    "new_osw_id",
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


## the browser bridge
#
# Both directions carry the whole current set, so these helpers mimic what the
# widget's JS does: read the outstanding calls, append to the cumulative event
# list.


def browser_calls(session):
    """The calls the browser can currently see, in the order they were made."""
    return list(session.frontend.function_calls)


def browser_replies(session, call_id, result=None, progress=False, token="t"):
    # The widget numbers monotonically per render and prefixes a per-render
    # token, so two renders cannot produce colliding sequence ids.
    seq = f"{token}:{len(session.frontend.function_events) + 1}"
    event = {
        "type": "function_call_progress" if progress else "function_call_result",
        "id": call_id,
        "seq": seq,
    }
    if not progress:
        event["result"] = result
    session.frontend.function_events = list(session.frontend.function_events) + [event]


def test_a_tool_call_resolves_only_on_its_own_id(session):
    async def scenario():
        pending = asyncio.ensure_future(
            session.call_client_side_tool("where_am_i", [], timeout=5)
        )
        await asyncio.sleep(0.05)
        call_id = browser_calls(session)[0]["id"]

        # a reply for a different call must not resolve this one
        browser_replies(session, "someone-else", {"url": "b"})
        await asyncio.sleep(0.05)
        assert not pending.done()

        browser_replies(session, call_id, {"url": "a"})
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
        await asyncio.sleep(0.05)
        call_id = browser_calls(session)[0]["id"]
        browser_replies(
            session, call_id,
            {"ok": False, "code": "permissiondenied", "message": "nope"},
        )
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


def test_token_carries_the_wiki_origin():
    from osw_chatbot.auth import token_wiki_url

    now = int(time.time())

    def mint(server):
        return _mint({"u": "Alice", "iat": now, "exp": now + 600, "s": server})

    assert token_wiki_url(mint("https://batterie-digital.fraunhofer.de")) == (
        "https://batterie-digital.fraunhofer.de"
    )
    # a trailing slash would double up when a link is built from it
    assert token_wiki_url(mint("https://wiki.example.org/")) == "https://wiki.example.org"
    # the claim is echoed into the system prompt, so anything that is not an
    # http(s) origin is dropped rather than passed through
    for bad in ("javascript:alert(1)", "wiki.example.org", "", 42, None):
        assert token_wiki_url(mint(bad)) is None
    # a wiki too old to send the claim leaves where_am_i as the fallback
    assert token_wiki_url(_mint({"u": "Alice", "iat": now, "exp": now + 600})) is None
    assert token_wiki_url("garbage") is None


def test_the_wiki_origin_reaches_the_system_prompt():
    """The whole point of the claim: the model is told, not left to guess."""
    from osw_chatbot.toolcalling.agent import ChatSession

    session = ChatSession(wiki_url="https://batterie-digital.fraunhofer.de")
    try:
        extra = session._prompt_extra()
        assert "https://batterie-digital.fraunhofer.de" in extra
    finally:
        session.close()

    bare = ChatSession()
    try:
        assert "fraunhofer" not in bare._prompt_extra()
    finally:
        bare.close()


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
        "title": "Item:OSW0024d7c0a0d64642bf51f5facfe85d42",
        "revisions": [{"slots": {
            "main": {}, "header": {},
            "jsondata": {"*": json.dumps({"label": [{"text": "Fluke 87V"}]})},
        }}],
    }}}}
    got = describe("get_page_content", page_query)
    # label when it is known, and the id shortened rather than 32 hex digits
    assert got.startswith("Fluke 87V (Item:OSW0024d7c0…)"), got
    assert "jsondata" in got

    # no label available - just the id
    bare = describe("get_page_content", {"query": {"pages": {"1": {
        "title": "Item:OSW0024d7c0a0d64642bf51f5facfe85d42", "revisions": [{"slots": {}}]}}}})
    assert bare == "Item:OSW0024d7c0…", bare

    schema = {"id": "x", "allOf": [
        {"properties": {"label": {}, "description": {}}},
        {"properties": {"image": {}, "manufacturer": {}}},
    ], "required": ["label"]}
    got = describe("get_category_schema", schema)
    assert "4 properties" in got and "label" in got and "required" in got

    got = describe("smw_ask_query", {"query": {"results": {
        "Item:OSWaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa": {"displaytitle": "Multimeter"},
        "Item:OSWbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb": {},
    }}})
    assert "2 results" in got and "Multimeter (Item:OSWaaaaaaaa…)" in got
    assert "Item:OSWbbbbbbbb…" in got      # unlabelled falls back to the id

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


def test_concurrent_reads_are_delivered_together(session):
    """The model routinely returns several reads in one turn.

    They must all reach the browser - the single param slot used to drop every
    call but the last - and none of them should wait on another.
    """
    async def scenario():
        first = asyncio.ensure_future(
            session.call_client_side_tool("get_page_content", [["A"]], timeout=5))
        second = asyncio.ensure_future(
            session.call_client_side_tool("get_page_content", [["B"]], timeout=5))
        await asyncio.sleep(0.05)

        calls = browser_calls(session)
        assert len(calls) == 2, calls
        assert [c["args"] for c in calls] == [[["A"]], [["B"]]]

        # answered out of order, because a browser may well do that
        browser_replies(session, calls[1]["id"], "second")
        browser_replies(session, calls[0]["id"], "first")
        assert await first == "first"
        assert await second == "second"
        # and the outstanding set empties again
        assert browser_calls(session) == []

    asyncio.run(scenario())


def test_a_dialog_tool_blocks_other_dialogs_but_not_reads(session):
    """One OOUI window manager, one editor - but any number of reads."""
    async def scenario():
        write = asyncio.ensure_future(
            session.call_client_side_tool("edit_page", ["Item:X", {}], timeout=5))
        await asyncio.sleep(0.05)
        other_write = asyncio.ensure_future(
            session.call_client_side_tool("delete_page", ["Item:Y"], timeout=5))
        read = asyncio.ensure_future(
            session.call_client_side_tool("get_page_content", [["A"]], timeout=5))
        await asyncio.sleep(0.05)

        names = [c["name"] for c in browser_calls(session)]
        assert "edit_page" in names
        assert "get_page_content" in names, "a read must not wait for a dialog"
        assert "delete_page" not in names, "a second dialog must wait"

        for call in browser_calls(session):
            browser_replies(session, call["id"], "ok")
        assert await write == "ok"
        assert await read == "ok"
        await asyncio.sleep(0.05)
        browser_replies(session, browser_calls(session)[0]["id"], "ok")
        assert await other_write == "ok"

    asyncio.run(scenario())


def test_a_confirmation_dialog_stops_the_clock(session):
    """The timeout measures the browser, not how long the user reads a dialog.

    Without this the backend gave up mid-dialog, told the model the write had
    failed, and the model wrote the page a second time.
    """
    async def scenario():
        pending = asyncio.ensure_future(
            session.call_client_side_tool("edit_page", ["Item:X", {}], timeout=0.2))
        await asyncio.sleep(0.05)
        call_id = browser_calls(session)[0]["id"]
        browser_replies(session, call_id, progress=True)

        # well past the original budget, still waiting
        await asyncio.sleep(0.5)
        assert not pending.done()

        browser_replies(session, call_id, "saved")
        assert await pending == "saved"

    asyncio.run(scenario())


def test_a_timeout_does_not_let_the_model_claim_success(session):
    """A timed-out write has an unknown outcome, not a failed one."""
    from osw_chatbot.toolcalling.agent import ClientToolError

    async def scenario():
        with pytest.raises(ClientToolError) as excinfo:
            await session.call_client_side_tool("edit_page", ["Item:X", {}], timeout=0.1)
        message = str(excinfo.value)
        assert "may still have completed" in message
        assert "do not report success" in message

    asyncio.run(scenario())


def test_a_silent_browser_raises_instead_of_returning_none(session):
    from osw_chatbot.toolcalling.agent import ClientToolError

    async def scenario():
        with pytest.raises(ClientToolError):
            await session.call_client_side_tool("where_am_i", [], timeout=0.1)

    asyncio.run(scenario())


def test_writing_a_page_requires_reading_the_skill_first(session):
    """Guidance that is merely available keeps getting skipped.

    Generated components hand-rolled `action=editslots` and invented property
    names while the skill saying not to sat one tool call away.
    """
    from osw_chatbot.toolcalling.agent import ClientToolError, required_skill

    assert required_skill("User:Alice/Widget.js") == "mediawiki-component"
    assert required_skill("Category:OSW123") == "osl-page-editing"
    assert required_skill("Item:OSW123") == "osl-page-editing"
    assert required_skill("Main Page") is None

    tool = {t.name: t for t in session.tools}["edit_page"]

    async def scenario():
        with pytest.raises(ClientToolError, match="mediawiki-component"):
            await tool.coroutine(title="User:Alice/Widget.js", slots={"main": "x"})

        # ...and reading it lifts the gate
        load = {t.name: t for t in session.tools}["load_skill"]
        await load.coroutine(name="mediawiki-component")
        pending = asyncio.ensure_future(
            tool.coroutine(title="User:Alice/Widget.js", slots={"main": "x"}))
        await asyncio.sleep(0.05)
        calls = browser_calls(session)
        assert [c["name"] for c in calls] == ["edit_page"]
        browser_replies(session, calls[0]["id"], {"ok": True})
        assert await pending == {"ok": True}

    asyncio.run(scenario())


def test_the_component_skill_covers_uploads_and_schema_first(session):
    """The two failures seen in review: bare File: pages and blank entities."""
    body = session.skills["mediawiki-component"].body
    assert "getFilePage" in body
    assert "OSW11a53cdfbdc24524bf8ac435cbf65d9d" in body    # Wiki File category
    assert "crypto.randomUUID()" in body
    assert "jsonschema" in body
    assert "osl-page-editing" in body                       # cross reference


def test_a_rerendered_widget_does_not_strand_every_call(session):
    """A fresh render restarts its counter; the backend remembers the old ones.

    Without a per-render token every reply after a re-render looks like a
    duplicate and the browser's answers are discarded in silence.
    """
    async def scenario():
        first = asyncio.ensure_future(
            session.call_client_side_tool("get_page_content", [["A"]], timeout=5))
        await asyncio.sleep(0.05)
        browser_replies(session, browser_calls(session)[0]["id"], "a", token="old")
        assert await first == "a"

        # the widget re-renders: numbering starts over under a new token
        session.frontend.function_events = []
        second = asyncio.ensure_future(
            session.call_client_side_tool("get_page_content", [["B"]], timeout=5))
        await asyncio.sleep(0.05)
        browser_replies(session, browser_calls(session)[0]["id"], "b", token="new")
        assert await second == "b"

    asyncio.run(scenario())


def test_keepalives_cannot_extend_a_call_forever(session):
    """The browser knows the call id, so the ceiling must not be negotiable."""
    async def scenario():
        pending = asyncio.ensure_future(
            session.call_client_side_tool("edit_page", ["Item:X", {}], timeout=1))
        await asyncio.sleep(0.05)
        call_id = browser_calls(session)[0]["id"]
        ceiling = session._ceilings[call_id]

        for _ in range(5):
            browser_replies(session, call_id, progress=True)
            assert session._deadlines[call_id] <= ceiling

        browser_replies(session, call_id, "done")
        assert await pending == "done"

    asyncio.run(scenario())


def test_edit_page_accepts_slots_as_a_json_string(session):
    """A ValidationError here is not a ToolException, so it killed the turn."""
    from osw_chatbot.toolcalling.agent import ClientToolError

    tool = {t.name: t for t in session.tools}["edit_page"]
    session.loaded_skills.add("osl-page-editing")

    async def scenario():
        pending = asyncio.ensure_future(tool.coroutine(
            title="Item:OSW1", slots='{"jsondata": {"label": "x"}}'))
        await asyncio.sleep(0.05)
        call = browser_calls(session)[0]
        assert call["args"][1] == {"jsondata": {"label": "x"}}, "not parsed"
        browser_replies(session, call["id"], {"ok": True})
        assert await pending == {"ok": True}

        with pytest.raises(ClientToolError, match="valid JSON"):
            await tool.coroutine(title="Item:OSW1", slots="not json")

    asyncio.run(scenario())


## prefilling the form editor


def test_reference_fields_are_resolved_or_emptied(session):
    """The prefill pass writes names and label objects where a title belongs.

    `keywords` is an array of autocomplete strings ranged on the Keyword
    category, so {"text": "Materials science"} is not a value it can hold.
    """
    from osw_chatbot.toolcalling import prefill

    schema = {
        "properties": {
            "label": {"type": "array", "items": {"type": "object"}},
            "keywords": {
                "type": "array",
                "items": {"type": "string", "format": "autocomplete",
                          "range": "Category:OSW09f6cdd54bc54de786eafced5f675cbe"},
            },
        }
    }
    data = {
        "label": [{"text": "A sample", "lang": "en"}],
        "keywords": [
            {"text": "Materials science", "lang": "en"},     # object, not a title
            "Item:OSWaa8d29404288446a9f3ec7afa4e2a512",      # already valid
            "Something nobody has defined",                  # unresolvable
        ],
    }

    found = list(prefill.find_references(data, schema))
    # only the keyword items are references - the label objects are not
    assert len(found) == 3
    assert all(path.startswith("keywords.") for _, _, path, _, _ in found)

    async def scenario():
        async def fake_ask(name, args, timeout=None):
            assert name == "smw_ask_query"
            if "materialsscience" in args[0]:
                return {"query": {"results": {
                    "Item:OSW1111111111111111111111111111aaaa": {}}}}
            return {"query": {"results": {}}}

        session.call_client_side_tool = fake_ask
        cleaned, unresolved = await session._resolve_references(data, schema)

        assert cleaned["keywords"] == [
            "Item:OSW1111111111111111111111111111aaaa",   # looked up by label
            "Item:OSWaa8d29404288446a9f3ec7afa4e2a512",   # left alone
        ]
        assert len(unresolved) == 1
        assert "Something nobody has defined" in unresolved[0]
        assert "Category:OSW09f6cdd54bc54de786eafced5f675cbe" in unresolved[0]
        # the label is untouched - it is not a reference field
        assert cleaned["label"] == [{"text": "A sample", "lang": "en"}]

    asyncio.run(scenario())


def test_an_ambiguous_reference_is_not_guessed(session):
    async def scenario():
        async def two_hits(name, args, timeout=None):
            return {"query": {"results": {
                "Item:OSW1111111111111111111111111111aaaa": {},
                "Item:OSW2222222222222222222222222222bbbb": {}}}}

        session.call_client_side_tool = two_hits
        schema = {"properties": {"status": {
            "format": "autocomplete", "range": "Category:OSWstatus"}}}
        cleaned, unresolved = await session._resolve_references(
            {"status": "in progress"}, schema)
        assert "status" not in cleaned
        assert len(unresolved) == 1

    asyncio.run(scenario())


## OSW-IDs


def test_models_do_not_get_to_invent_osw_ids(session):
    """Observed on the wiki: aa11bb22cc3344..., bb22cc33dd4455..., a1b2c3d4...

    Those are a template being filled in, not random, so two wikis coining the
    same keyword produce the same id. Local statistics cannot tell them apart
    (`a1b2c3d4e5f6...` scores like a real uuid4), so the model does not choose.
    """
    from osw_chatbot.toolcalling.agent import (
        ClientToolError, osw_id_of, uuid_from_osw_id)

    assert osw_id_of("Item:OSWaa11bb22cc3344d5a6b7c8d9e0f1a2b3") == (
        "aa11bb22cc3344d5a6b7c8d9e0f1a2b3")
    assert osw_id_of("Category:OSW" + "a" * 32) == "a" * 32
    assert osw_id_of("User:Alice/Widget.js") is None
    assert osw_id_of("Main Page") is None
    assert uuid_from_osw_id("aa11bb22cc3344d5a6b7c8d9e0f1a2b3") == (
        "aa11bb22-cc33-44d5-a6b7-c8d9e0f1a2b3")

    tools = {t.name: t for t in session.tools}
    session.loaded_skills.add("osl-page-editing")

    async def scenario():
        # a made-up id for a page that does not exist is refused
        async def missing(name, args, timeout=None):
            assert name == "get_page_content"
            return {"query": {"pages": {"-1": {"missing": ""}}}}

        session.call_client_side_tool = missing
        with pytest.raises(ClientToolError, match="new_osw_id"):
            await tools["edit_page"].coroutine(
                title="Item:OSWaa11bb22cc3344d5a6b7c8d9e0f1a2b3",
                slots={"jsondata": {"label": [{"text": "Organigramm"}]}})

        # an id this session minted goes straight through
        minted = await tools["new_osw_id"].coroutine()
        assert minted.startswith("OSW") and len(minted) == 35
        assert minted[3:] in session.minted_ids

        sent = {}

        async def capture(name, args, timeout=None):
            sent["args"] = args
            return {"ok": True}

        session.call_client_side_tool = capture
        await tools["edit_page"].coroutine(
            title=f"Item:{minted}",
            slots={"jsondata": {"label": [{"text": "Organigramm"}]}})
        # ...and the uuid in the data is made to agree with the title
        assert sent["args"][1]["jsondata"]["uuid"] == uuid_from_osw_id(minted[3:])

    asyncio.run(scenario())


def test_an_existing_page_can_still_be_edited(session):
    """The gate must not block edits to pages the model did not create."""
    tools = {t.name: t for t in session.tools}
    session.loaded_skills.add("osl-page-editing")

    async def scenario():
        async def exists(name, args, timeout=None):
            if name == "get_page_content":
                return {"query": {"pages": {"42": {"title": args[0][0]}}}}
            return {"ok": True}

        session.call_client_side_tool = exists
        assert await tools["edit_page"].coroutine(
            title="Item:OSW452ec0273916478099c4716395e1bc18",
            slots={"main": ""}) == {"ok": True}

    asyncio.run(scenario())


def test_the_opening_messages_are_configurable():
    """A deployment may reword, translate or drop either opener."""
    import importlib, os
    import osw_chatbot.config as config

    try:
        os.environ["CHATBOT_GREETING"] = "Wie kann ich helfen?"
        os.environ["CHATBOT_INTRO"] = ""
        importlib.reload(config)
        assert config.CHATBOT_GREETING == "Wie kann ich helfen?"
        assert config.CHATBOT_INTRO == ""      # empty hides it
    finally:
        del os.environ["CHATBOT_GREETING"]
        del os.environ["CHATBOT_INTRO"]
        importlib.reload(config)

    assert config.CHATBOT_GREETING == "what's on your mind?"
    assert "can be wrong" in config.CHATBOT_INTRO


def test_handled_replies_are_acknowledged_so_the_browser_can_forget_them(session):
    """The cumulative list must not keep carrying old results.

    A get_page_content result is a whole page and a file result is a data url.
    Re-serialising those on every later event froze the chat UI.
    """
    async def scenario():
        pending = asyncio.ensure_future(
            session.call_client_side_tool("get_page_content", [["A"]], timeout=5))
        await asyncio.sleep(0.05)
        call = browser_calls(session)[0]
        browser_replies(session, call["id"], "x" * 50000)
        assert await pending == "x" * 50000

        # the backend confirms exactly what it consumed
        assert session.frontend.processed_events, "nothing acknowledged"
        assert all(s in session._seen_events for s in session.frontend.processed_events)

        # replaying an acknowledged event is a no-op, not a second resolve
        before = len(session._seen_events)
        session.frontend.function_events = list(session.frontend.function_events)
        await asyncio.sleep(0.05)
        assert len(session._seen_events) == before

    asyncio.run(scenario())


def test_malformed_tool_arguments_reach_the_model_not_the_user(session):
    """Pydantic rejects bad arguments before the tool body runs.

    That ValidationError is not a ToolException, so handle_tool_error never saw
    it and the turn died with a traceback in the chat - twice: a nested `param`
    object the model flattened, and `slots` passed as a JSON string.
    """
    from osw_chatbot.toolcalling.agent import _tool_input_observation

    for tool in session.tools:
        assert tool.handle_validation_error, f"{tool.name} has no validation handler"
        assert tool.handle_tool_error, f"{tool.name} has no error handler"

    async def scenario():
        tool = {t.name: t for t in session.tools}["get_page_content"]
        # wrong shape entirely - must come back as an observation, not a raise
        observation = await tool.arun({"nonexistent_argument": 1})
        assert "did not match this tool's schema" in observation

    asyncio.run(scenario())
    assert "call it again" in _tool_input_observation(ValueError("x"))


def test_the_websearch_tools_take_flat_arguments(session):
    """A single nested `param` object is the shape models get wrong."""
    from osw_chatbot.websearch.interative_websearch import (
        simple_browse_web_page_tool, web_search_tool)

    assert set(web_search_tool.args) == {"query", "max_results"}
    assert set(simple_browse_web_page_tool.args) == {"url"}
    assert "param" not in web_search_tool.args


def test_full_text_search_reports_results_and_emptiness(session):
    """It returns structured hits now, not scraped Special:Search html.

    The scrape missed uploaded-file text entirely - it used the default search
    profile, which covers content namespaces only, so a phrase inside an
    indexed PDF matched nothing - and it returned null when nothing matched,
    which the backend raised at the model as a tool failure.
    """
    from osw_chatbot.toolcalling.progress import describe

    tool = {t.name: t for t in session.tools}["full_text_search"]
    doc = tool.description
    assert "PDF" in doc
    assert "not an error" in doc

    pdf_hit = {"totalhits": 1, "results": [
        {"title": "File:Manual.pdf", "snippet": "1.7 Referenzlogo 33"}]}
    assert describe("full_text_search", pdf_hit) == "1 result"
    assert describe("full_text_search", {"totalhits": 0, "results": []}) == "no results"
    assert describe("full_text_search", {"totalhits": 57, "results": [
        {"title": f"P{i}", "snippet": ""} for i in range(10)]}) == "10 results of 57"


def test_document_text_is_rendered_with_page_numbers(session):
    """Citations need a page, and the file itself must never cross the wire.

    A 16 MB pdf becomes ~21.5 MB as a base64 data url, past bokeh's 20 MB
    websocket limit - the reply was dropped and the call hung until timeout.
    """
    from osw_chatbot.toolcalling.agent import _render_file_text

    indexed = {
        "title": "File:Manual.pdf", "page_count": 279,
        "matched_pages": [2, 5, 33],
        "pages": [{"page": 2, "url": "https://wiki.example/w/index.php?title=File%3AManual.pdf&page=2",
                   "text": "1.7 Referenzlogo"},
                  {"page": 5, "url": "https://wiki.example/w/index.php?title=File%3AManual.pdf&page=5",
                   "text": "Referenzlogo rules"}],
        "truncated": True,
    }
    out = _render_file_text(indexed, "Referenzlogo")
    assert "File:Manual.pdf (279 pages)" in out
    assert "Pages containing 'Referenzlogo': 2, 5, 33" in out
    assert "--- page 2 |" in out and "--- page 5 |" in out
    # the citation link must travel with the page it belongs to
    assert "title=File%3AManual.pdf&page=2" in out
    assert "truncated" in out

    # a format without page breaks must not invite an invented page number
    blob = _render_file_text(
        {"title": "File:Sheet.xlsx", "pages": None, "text": "some text",
         "chars": 9, "truncated": False})
    assert "no page breaks available" in blob
    assert "--- page" not in blob, "no page marker when there are no pages"

    assert _render_file_text({"title": "x", "pages": None, "text": ""}) == ""

    tool = {t.name: t for t in session.tools}["get_file_content"]
    assert set(tool.args) == {"file_title", "query"}
    assert "url that renders exactly that page" in tool.description


def test_one_huge_tool_result_cannot_fail_the_turn():
    """A 2M-token prompt is a 400 from the vendor, i.e. no answer at all.

    The scratchpad is never trimmed, so the cap has to be on the way out of the
    tool. Truncating degrades the answer instead of losing it.
    """
    import asyncio

    from osw_chatbot.config import MAX_TOOL_RESULT_CHARS
    from osw_chatbot.toolcalling.agent import _bounded, _truncate

    assert _truncate("short") == "short"
    big = _truncate("x" * (MAX_TOOL_RESULT_CHARS + 5000))
    assert len(big) < MAX_TOOL_RESULT_CHARS + 500
    assert "truncated" in big

    class FakeTool:
        def __init__(self, result):
            async def coroutine(*a, **k):
                return result

            self.coroutine = coroutine

    huge = FakeTool("y" * (MAX_TOOL_RESULT_CHARS * 3))
    _bounded(huge)
    out = asyncio.run(huge.coroutine())
    assert len(out) < MAX_TOOL_RESULT_CHARS + 500
    assert "truncated" in out

    # Structured results are left alone: slicing a dict repr or a base64 data
    # url would corrupt it rather than shorten it.
    structured = FakeTool({"pages": [1, 2, 3]})
    _bounded(structured)
    assert asyncio.run(structured.coroutine()) == {"pages": [1, 2, 3]}

    # A tool with no coroutine must survive wrapping untouched.
    class SyncOnly:
        coroutine = None

    assert _bounded(SyncOnly()) is not None


def test_replayed_history_has_no_dangling_tool_calls():
    """A tool_use with no tool_result is malformed, and it made the assistant lie.

    Dropping ToolMessages while keeping tool_calls on the AIMessage that made them left
    the model's own "Done" as the only evidence anything happened - so it stopped calling
    tools and asserted success instead, worsening as a session grew.
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    from osw_chatbot.chat.history import dumps, loads

    msgs = [
        HumanMessage(content="edit the page"),
        AIMessage(
            content="I will edit it",
            tool_calls=[
                {
                    "name": "edit_page",
                    "args": {"title": "Item:X", "slots": {"main": "x" * 5000}},
                    "id": "c1",
                    "type": "tool_call",
                }
            ],
        ),
        ToolMessage(content="ok, revision 42 saved", tool_call_id="c1"),
        AIMessage(content="Done - I have updated the page."),
    ]
    back = loads(dumps(msgs))

    assert not [c for m in back for c in (getattr(m, "tool_calls", None) or [])]

    # The record has to survive, or the model still cannot tell what it already did.
    transcript = "\n".join(str(m.content) for m in back)
    assert "edit_page" in transcript
    assert "revision 42" in transcript
    assert "Item:X" in transcript

    # ...without the payload that made tool messages worth dropping.
    assert "x" * 200 not in transcript


def test_trimming_a_long_message_keeps_its_other_fields():
    """type(message)(content=...) dropped tool_calls, so behaviour depended on length."""
    from langchain_core.messages import AIMessage

    from osw_chatbot.chat.history import trim_history

    long_ai = AIMessage(content="y" * 50_000, id="keep-me")
    (trimmed,) = trim_history([long_ai], max_messages=5, max_chars=100)
    assert trimmed.id == "keep-me"
    assert len(trimmed.content) < 200



def test_writing_header_or_footer_is_refused():
    """A prompt rule was not enough; these slots are rejected at the tool.

    header/footer carry the default entity invokes and nothing else - overwriting them
    makes the entity render blank on every page that uses them.
    """
    import asyncio

    from osw_chatbot.toolcalling.agent import ChatSession, ClientToolError

    session = ChatSession(actions=["websearch", "edit"])
    try:
        # The skill gate fires before any slot checking, which is the right order - the
        # skill is where the rule is taught. Satisfy it so the guard itself is reachable.
        session.loaded_skills.add("osl-page-editing")
        edit = next(t for t in session.tools if t.name == "edit_page")
        for slot in ("header", "footer"):
            with pytest.raises(ClientToolError) as caught:
                asyncio.run(edit.coroutine(title="Item:OSW1", slots={slot: "x"}))
            message = str(caught.value)
            assert slot in message
            # The refusal has to say where the content actually goes, or the model
            # retries the same write.
            assert "header_template" in message and "main" in message
    finally:
        session.close()


def test_a_loaded_skill_is_not_sent_twice():
    """The body is already upstream in the conversation; resending spends it again."""
    import asyncio

    from osw_chatbot.toolcalling.agent import ChatSession

    session = ChatSession()
    try:
        load = next(t for t in session.tools if t.name == "load_skill")
        name = next(iter(session.skills))
        first = asyncio.run(load.coroutine(name=name))
        second = asyncio.run(load.coroutine(name=name))
        assert len(first) > 200, "the first call must return the real body"
        assert "already loaded" in second
        assert len(second) < len(first)
    finally:
        session.close()


def test_a_page_written_this_session_is_not_reread():
    """The transcript carries no record of the write, so the model asks again."""
    import asyncio

    from osw_chatbot.toolcalling.agent import ChatSession

    session = ChatSession()
    try:
        session.written_pages["Item:OSW1"] = {"main": "hello"}
        read = next(t for t in session.tools if t.name == "get_page_content")
        out = asyncio.run(read.coroutine(titles=["Item:OSW1"], include_html=False))
        assert out["Item:OSW1"]["slots"] == {"main": "hello"}
        assert "session" in out["Item:OSW1"]["note"]
    finally:
        session.close()


def test_a_long_history_is_trimmed_to_fit_the_prompt():
    """Over the limit the turn fails outright, so trimming beats not trimming."""
    from langchain_core.messages import AIMessage, HumanMessage

    from osw_chatbot.config import MAX_PROMPT_CHARS
    from osw_chatbot.toolcalling.agent import ChatSession

    session = ChatSession()
    try:
        big = "z" * 50_000
        session.chat_history = [
            m
            for _ in range(40)
            for m in (HumanMessage(content=big), AIMessage(content=big))
        ]
        assert sum(len(m.content) for m in session.chat_history) > MAX_PROMPT_CHARS

        dropped = session._fit_prompt("a short prompt")
        assert dropped is True
        assert sum(len(m.content) for m in session.chat_history) <= MAX_PROMPT_CHARS
        # Turns are dropped in pairs, so a human message never loses its reply.
        assert len(session.chat_history) % 2 == 0

        # A short history is left entirely alone.
        session.chat_history = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert session._fit_prompt("x") is False
        assert len(session.chat_history) == 2
    finally:
        session.close()


def test_a_mistyped_osw_id_is_caught_before_it_becomes_a_second_page():
    """IDs drift between edits; creating the near-miss leaves two half-written pages."""
    import asyncio

    from osw_chatbot.toolcalling.agent import (
        ChatSession,
        ClientToolError,
        nearest_known_id,
    )

    good = "a" * 32
    assert nearest_known_id("a" * 31 + "b", {good}) == good  # one character off
    assert nearest_known_id(good, {good}) is None  # identical is not a near miss
    assert nearest_known_id("b" * 32, {good}) is None  # a different entity
    assert nearest_known_id("a" * 31, {good}) is None  # different length

    session = ChatSession(actions=["websearch", "edit"])
    try:
        session.loaded_skills.add("osl-page-editing")
        session.minted_ids.add(good)
        edit = next(t for t in session.tools if t.name == "edit_page")
        typo = "Item:OSW" + ("a" * 31 + "b").upper()
        with pytest.raises(ClientToolError) as caught:
            asyncio.run(edit.coroutine(title=typo, slots={"main": "x"}))
        # Naming the right title matters: without it the model guesses again.
        assert good.upper() in str(caught.value)
    finally:
        session.close()
