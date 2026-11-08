import asyncio
import copy
import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Set, Union

import langchain_core.tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from osw_chatbot.chat.chat_panel_component import ChatFrontendWidget
from osw_chatbot.config import (
    CLIENT_TOOL_TIMEOUT,
    MAX_INLINE_FILE_BYTES,
    MAX_PROMPT_CHARS,
    SKILL_CATEGORY,
    SKILL_DEFAULT_DATA,
    SKILL_DISCOVERY_QUERY,
    MAX_TOOL_RESULT_CHARS,
    CONFIRM_TIMEOUT,
    PARENT_ORIGINS,
)
from osw_chatbot.chat.history import trim_history
from osw_chatbot.llm import llm
from osw_chatbot.skills import load_skills, skill_index
from osw_chatbot.toolcalling import prefill
from osw_chatbot.structured_output.llm import get_llm_response
from osw_chatbot.structured_output.util import multimedia_data_url_to_text
from osw_chatbot.websearch.interative_websearch import invoke as websearch_invoke
from osw_chatbot.websearch.interative_websearch import tools as websearch_tools


# Client tools that drive the singleton OOUI window manager behind every
# confirmation dialog. Two dialogs at once queue invisibly, so these take the
# browser lock; everything else - the reads - runs concurrently.
# `create_category_instance` is here for the same reason, though note the
# browser returns as soon as the editor is open rather than awaiting it, so
# the lock does not actually keep two editors apart.
# How much extracted document text to hand the model at once.
FILE_TEXT_MAX_CHARS = int(os.getenv("FILE_TEXT_MAX_CHARS", "20000"))

EXCLUSIVE_CLIENT_TOOLS = {
    "edit_page",
    "delete_page",
    "upload_file",
    "redirect",
    "create_category_instance",
    "confirm_action",
    # Each opens or focuses the sandbox tab, and run_in_sandbox puts a confirmation in
    # front of the user. Two of those racing would leave them answering one dialog while
    # another is being drawn over it.
    "open_sandbox",
    "create_notebook",
    "run_in_sandbox",
}


# Which skill has to be read before writing a given page. The guidance has
# existed for a while and was still ignored - generated components kept
# hand-rolling `action=editslots` and inventing property names - so the write
# tool asks for it rather than trusting the system prompt.
# Namespaces whose pages are entities identified by an OSW-ID.
ENTITY_NAMESPACES = ("item", "category", "property")
OSW_TITLE = re.compile(r"^([A-Za-z][A-Za-z_ ]*):OSW([0-9a-fA-F]{32})$")


def osw_id_of(title: str) -> Optional[str]:
    """The OSW-ID in an entity page title, lowercased, or None."""
    match = OSW_TITLE.match((title or "").strip().lstrip(":"))
    if not match or match.group(1).lower() not in ENTITY_NAMESPACES:
        return None
    return match.group(2).lower()


def nearest_known_id(osw_id: str, known) -> Optional[str]:
    """A known OSW-ID that `osw_id` is a near-miss of, or None.

    OSW-IDs are 32 hex characters with no redundancy, so a model copying one across turns
    transposes or drops a character and produces something that is still well-formed and
    points at nothing. Same length and a handful of differing positions is a typo; more
    than that is a different entity and must not be silently rewritten.
    """
    if not osw_id:
        return None
    for candidate in known:
        if candidate == osw_id or len(candidate) != len(osw_id):
            continue
        differing = sum(1 for a, b in zip(candidate, osw_id) if a != b)
        if 0 < differing <= 3:
            return candidate
    return None


def uuid_from_osw_id(osw_id: str) -> str:
    """OSW-IDs are a uuid with the hyphens taken out; put them back."""
    return str(uuid.UUID(osw_id))


# Category:Skill - instances are assistant skills authored on the wiki. A subclass of
# Category:Tutorial, so querying Tutorial would return fifty pages that are not skills.


#: Slots the wiki owns. 'header' and 'footer' always carry the default entity invokes and
#: nothing else; writing them replaces the rendering machinery on every page that uses
#: them. Content for one page belongs in 'main'; content for every instance of a category
#: belongs in 'header_template' / 'footer_template'.
MANAGED_SLOTS = frozenset({"header", "footer"})


def _first_text(value) -> str:
    """OSW stores labels and descriptions as [{text, lang}]. Prefer English, take any."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        english = next(
            (e for e in value if isinstance(e, dict) and e.get("lang") == "en"), None
        )
        chosen = english if english is not None else value[0]
        if isinstance(chosen, dict):
            return chosen.get("text", "")
    return ""


def required_skill(title: str) -> Optional[str]:
    # MediaWiki normalises ":category:foo" and "Category:Foo" to one page, so
    # match the way it does rather than on the exact string the model sent.
    title = (title or "").strip().lstrip(":")
    if title.lower().endswith((".js", ".css")):
        return "mediawiki-component"
    namespace = title.split(":", 1)[0].strip().lower()
    if namespace in ("category", "item", "property", "jsonschema"):
        return "osl-page-editing"
    return None


class ClientToolError(ToolException):
    """The users browser did not answer a client-side tool call.

    Deriving from ToolException matters: LangChain only routes ToolException
    through `handle_tool_error`, anything else propagates out of ainvoke and
    ends up as a traceback in the chat.
    """


class ToolRefusal(ClientToolError):
    """A tool deliberately declined - a gate, a guard, a malformed argument.

    Distinct from ClientToolError because the two need opposite advice. A browser that
    did not answer is worth telling the user about and not retrying; a refusal says what
    to do instead - load this skill, use that slot, use this title - and retrying after
    doing it is the point.
    """


def as_text(output) -> str:
    """Flatten an LLM response to plain text.

    OpenAI returns a string, Anthropic a list of content blocks - without this
    the chat would render `[{'text': 'hi', 'type': 'text', 'index': 0}]`.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, dict):
                if block.get("type", "text") == "text":
                    parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if output is None else str(output)


def _truncate(text: str, limit: int = MAX_TOOL_RESULT_CHARS) -> str:
    """Cap one contribution to the prompt, saying so rather than silently cutting."""
    if len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n\n[truncated: {len(text)} characters, showing {limit}. "
        "Narrow the query and call again for the part you need.]"
    )


def _bounded(tool):
    """Stop one oversized observation from failing the whole turn.

    The scratchpad is not trimmed - see MAX_TOOL_RESULT_CHARS in config. Only
    string results are capped: the tools that can be huge return text, while the
    structured ones are small, and blindly slicing a base64 data url or a dict
    repr would corrupt it instead of shortening it.
    """
    original = getattr(tool, "coroutine", None)
    if original is None:
        return tool

    async def bounded(*args, **kwargs):
        result = await original(*args, **kwargs)
        return _truncate(result) if isinstance(result, str) else result

    bounded.__name__ = getattr(original, "__name__", "bounded")
    tool.coroutine = bounded
    return tool


def _tool_error_observation(error: Exception) -> str:
    """Feed a failed client-side tool back to the model as an observation.

    A sentence the model can act on - retry, or tell the user what went wrong. Returning
    None here reaches `WebPage(**None)` and surfaces as a traceback in the chat.
    """
    if isinstance(error, ToolRefusal):
        # A refusal carries its own instruction; the browser advice below would
        # contradict it.
        return f"Error: {error}"
    return (
        f"Error: {error}. The users browser could not be reached - do not retry "
        "more than once, tell the user instead."
    )


def _render_file_text(indexed: dict, query: str = "") -> str:
    """Lay out extracted document text so the model can cite a page.

    Page numbers only exist for formats the wiki splits per page - PDFs via
    PdfHandler. For anything else there is one undifferentiated blob, and the
    header says so rather than inviting an invented page number.
    """
    title = indexed.get("title") or "the file"
    pages = indexed.get("pages")

    if pages:
        head = [f"Extracted text of {title}"]
        if indexed.get("page_count"):
            head[0] += f" ({indexed['page_count']} pages)"
        matched = indexed.get("matched_pages")
        if query and matched:
            head.append(
                f"Pages containing {query!r}: "
                + ", ".join(str(p) for p in matched[:20])
                + ("..." if len(matched) > 20 else "")
            )
        elif query:
            head.append(f"No page contains {query!r}; showing the start instead.")
        body = "\n\n".join(
            # the url renders that single page, so a citation can point at it
            f"--- page {page['page']}"
            + (f" | {page['url']}" if page.get("url") else "")
            + f" ---\n{page['text']}"
            for page in pages
        )
        if indexed.get("truncated"):
            body += "\n\n[truncated - ask again with a narrower query for more]"
        return "\n".join(head) + "\n\n" + body

    text = indexed.get("text")
    if not text:
        return ""
    header = f"Extracted text of {title} (no page breaks available)"
    if indexed.get("truncated"):
        text += (
            f"\n\n[truncated - showing {len(text)} of "
            f"{indexed.get('chars')} characters]"
        )
    return f"{header}\n\n{text}"


def _tool_input_observation(error: Exception) -> str:
    """Feed a malformed tool call back to the model instead of crashing.

    Pydantic raises ValidationError while parsing the arguments, before the
    tool body runs. It is not a ToolException, so handle_tool_error never sees
    it and the whole turn dies with a traceback in the chat - twice now: a
    nested `param` object the model flattened, and `slots` sent as a JSON
    string. Telling the model what the arguments should look like lets it
    correct itself.
    """
    return (
        f"Error: the arguments did not match this tool's schema. {error}. "
        "Look at the tool's parameters and call it again with the right ones."
    )


class WebPage(BaseModel):
    url: str
    """The full url of the web page."""
    title: str
    """The title of the web page."""
    content: str
    """The html content of the page."""
    user: Optional[Dict[str, Optional[str]]] = None
    """The user object with id and name of the current user.

    The values are optional: for anonymous wiki readers `mw.user.getName()`
    returns null, so `id` arrives as None.
    """


SYSTEM_TEXT = (
    "You are a helpful assistant called 'EVE' that guides a user on a wiki-like website based on OpenSemanticLab. If the user wants to create something first try to find the category page for the given topic/keyword the user metions. Then create the instance. "
    "If the user asks something about the current page or the website in general, analyse the situation with the 'where_am_i' tool and guide him afterwards."
    "If the user does not provide a context assume that he's referring to the current page or the website as a whole."
    "If you provide links, format them directly as html anchor elements in your response, always with attribute target=\"_blank\". In particular link OSW-IDs, e.g. Item:OSW123... with <wiki-url>/wiki/Item:OSW123... and the page label as link text."
    "Never guess or assume the wiki domain: take it from the 'url' the 'where_am_i' tool returned in this conversation, and call that tool first if you have not already. Different deployments use different domains."
    "For mailto links use semicolons to separate multiple email addresses."
    "You also have web search tools. Use them whenever external or up to date information would help to answer, and do so directly instead of asking the user for permission first. The topic does not have to be related to this wiki - answer general questions too, guiding the user on the wiki is your main job, not your only one."
    "Edit pages only when the user requests it. Otherwise ask for confirmation first."
    "This wiki stores things as structured entities, so create a device, event, person, project or sample as an instance of its category (find it with 'find_page_from_topic', read it with 'get_category_schema'), never as a plain wikitext page. The exceptions are the user's own 'User:<name>/...' subpages, '.js' and '.css' pages, and the main page."
    "Never invent an OSW-ID. Every new entity gets its title from the 'new_osw_id' tool, used exactly as returned - OSW-IDs must be random and globally unique, and a made up one collides with other wikis or points at nothing."
    "Report what actually happened: only call a page created, saved or deleted when the tool returned success. If a tool reported an error or a timeout, say exactly that - a timed out write may or may not have gone through, so re-read the page and tell the user what you found instead of retrying blindly or claiming it worked."
)


def build_prompt(extra: str = "") -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEXT + (("\n\n" + extra) if extra else "")),
            ("placeholder", "{chat_history}"),
            # A placeholder, not ("human", "{input}"): a string template
            # stringifies a content list, so an attached image would reach the
            # model as a python repr instead of an image block.
            ("placeholder", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )


SYSTEM_PROMPT = build_prompt()


class ChatSession:
    """Everything that belongs to *one* browser session.

    Previously the frontend widget, the tools, the agent executor and the chat
    history were module level, i.e. one set per process. Because Panel pushes a
    param change to every Document a component is rendered in, a single
    `function_call` was broadcast to every connected browser: user A's redirect
    ran in user B's tab, and B's page content could answer A's `where_am_i`.
    Creating all of it per session is what fixes that.
    """

    def __init__(
        self,
        allowed_origins=None,
        tool_timeout=None,
        user=None,
        actions=None,
        confirm_actions=None,
        wiki_url=None,
    ):
        self.user = user  # verified wiki username, or None when unauthenticated
        # Origin of the wiki that signed the token, when it sent one. Goes into
        # the system prompt so the model never has to guess it; see
        # auth.token_wiki_url for why guessing was the previous behaviour.
        self.wiki_url = wiki_url
        # Actions the wiki signed into the token. Only the ones that run *here*
        # are enforced here - the write actions are gated in the browser, which
        # is where the user's MediaWiki session is.
        self.actions = list(actions if actions is not None else ["websearch"])
        self.confirm_actions = list(confirm_actions or [])
        self.frontend = ChatFrontendWidget(
            allowed_origins=list(allowed_origins or PARENT_ORIGINS)
        )
        self._timeout = tool_timeout or CLIENT_TOOL_TIMEOUT
        self._pending: Dict[str, asyncio.Future] = {}
        # The calls the browser has not answered yet. Published as a whole (see
        # ChatFrontendWidget) so that two tool calls started in the same tick
        # cannot overwrite each other the way a single param slot did.
        self._outstanding: Dict[str, dict] = {}
        # call_id -> loop time at which we give up. A progress message from the
        # browser pushes it out; see _on_function_events.
        self._deadlines: Dict[str, float] = {}
        # Hard upper bound per call, so keep-alives cannot extend forever.
        self._ceilings: Dict[str, float] = {}
        self._seen_events: Set[str] = set()
        # Only the tools that open a dialog or an editor are serialised - the
        # browser can run any number of reads at once, and a confirmation the
        # user leaves sitting must not block them.
        self._browser = asyncio.Lock()
        self._closed = False
        self.frontend.param.watch(self._on_function_events, "function_events")

        self.skills = load_skills()
        self.loaded_skills: Set[str] = set()
        # title -> slots this session successfully wrote. The model re-reads pages it has
        # just written because the history carries no record of the write; serving that
        # read from here costs nothing and answers it honestly. Only successful writes are
        # recorded, and `include_html` still goes to the wiki, so a confirming read that
        # actually needs the rendered page is never short-circuited.
        self.written_pages: Dict[str, Any] = {}
        # SKILL_DEFAULT_DATA with references this wiki lacks removed; see
        # _usable_skill_defaults. None until the first skill write probes for them.
        self._skill_defaults: Optional[Dict[str, Any]] = None
        # category -> its superclass chain. The hierarchy does not move during a
        # conversation, and the gate would otherwise resolve it on every write.
        self._superclass_cache: Dict[str, List[str]] = {}
        # Wiki-authored skills, fetched through the browser the first time something
        # asks. None means 'not looked yet'; {} means 'looked, this wiki has none'.
        self._wiki_skills: Optional[Dict[str, Dict[str, str]]] = None
        # OSW-IDs this session handed out. Models do not generate random
        # uuids - they fill in a pattern (aa11bb22cc3344..., a1b2c3d4...),
        # which collides the moment two wikis coin the same entity.
        self.minted_ids: Set[str] = set()
        self.chat_history: List[BaseMessage] = []
        self.tools = self._build_tools()
        if "websearch" in self.actions:
            self.tools += [self._gated_websearch_tool(t) for t in websearch_tools]
        # The websearch tools join after _build_tools, so the handlers are attached
        # here, to the whole list. Without them a malformed call kills the turn.
        for tool in self.tools:
            tool.handle_tool_error = _tool_error_observation
            tool.handle_validation_error = _tool_input_observation
            _bounded(tool)
        self.agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(
                llm, self.tools, build_prompt(self._prompt_extra())
            ),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    def _prompt_extra(self) -> str:
        """Per-session additions to the system prompt.

        The wiki origin goes first and in imperative form. SYSTEM_TEXT already
        forbids guessing the domain, but that rule points at a tool the model
        has to remember to call; stating the answer outright removes the step
        it was skipping.
        """
        parts = []
        if self.wiki_url:
            parts.append(
                f"This conversation is hosted by the wiki at {self.wiki_url} - "
                "use exactly that origin for every link you write, and do not "
                "call 'where_am_i' merely to learn the domain. Any other host "
                "in a link is wrong, including one you remember from elsewhere: "
                "you have no memory of previous conversations."
            )
        index = skill_index(self.skills)
        if index:
            parts.append(index)
        return "\n\n".join(parts)

    def _gated_websearch_tool(self, tool):
        """Ask the browser before searching, when the wiki wants that.

        Web search happens on the server, so the consent dialog has to be
        requested from the browser explicitly - unlike the write tools, which
        confirm themselves. Returning the tool unchanged when no confirmation
        is configured keeps the common path free of a round trip.
        """
        if "websearch" not in self.confirm_actions:
            return tool

        session = self
        original = tool.coroutine

        async def confirmed(*args, **kwargs):
            allowed = await session.call_client_side_tool(
                "confirm_action", ["websearch", {"query": str(args or kwargs)[:200]}]
            )
            if not allowed:
                raise ToolRefusal(
                    "the user declined the web search - answer without it and say so"
                )
            return await original(*args, **kwargs)

        gated = copy.copy(tool)
        gated.coroutine = confirmed
        return gated

    ## client-side tool call bridge

    def _on_function_events(self, event):
        """Handle the browser's replies: results, and keep-alives for dialogs.

        The list is cumulative, so an update that coalesces with the next one
        still carries everything we have not processed; `_seen_events` makes
        replaying the overlap harmless.
        """
        handled = []
        for data in event.new or []:
            seq = data.get("seq")
            if seq in self._seen_events:
                continue
            self._seen_events.add(seq)
            handled.append(seq)
            call_id = data.get("id")
            if data.get("type") == "function_call_progress":
                # The browser is alive and waiting on the user. Give the call a
                # fresh, generous budget rather than timing out mid-dialog.
                waiting = self._pending.get(call_id)
                if waiting is not None and call_id in self._deadlines:
                    # The loop's clock is monotonic and safe to read from the
                    # watcher's thread; only the waiter acts on the new value.
                    self._deadlines[call_id] = min(
                        waiting.get_loop().time() + CONFIRM_TIMEOUT,
                        self._ceilings.get(call_id, float("inf")),
                    )
                continue
            future = self._pending.pop(call_id, None)
            if future is not None and not future.done():
                # The watcher may fire off another thread - always hop back to the
                # loop the future was created on.
                future.get_loop().call_soon_threadsafe(future.set_result, data.get("result"))
        if handled:
            # An update already in flight can still carry an acked event, so
            # keep recognising them for a while - just not forever.
            if len(self._seen_events) > 1000:
                self._seen_events = set(list(self._seen_events)[-500:])
            # Let the browser drop these. A result holds a whole page or a data
            # url, and the list is republished on every later event.
            self.frontend.processed_events = handled

    def _publish_calls(self):
        # json round trip: a fresh object so param detects the change, and it
        # fails loudly here rather than during serialization.
        self.frontend.function_calls = json.loads(
            json.dumps(list(self._outstanding.values()))
        )

    async def call_client_side_tool(self, name, args, timeout=None):
        """Ask this session's browser to run `name` and wait for its answer.

        Correlation is by uuid and by an awaited Future. Calls run concurrently
        unless the tool opens UI (`EXCLUSIVE_CLIENT_TOOLS`), because the wiki's
        dialog and editor are singletons but its read APIs are not.
        """
        if self._closed:
            raise ClientToolError("the chat session was closed")

        if name not in EXCLUSIVE_CLIENT_TOOLS:
            return await self._dispatch(name, args, timeout)

        # A user who walks away from a confirmation dialog holds this lock for
        # the whole confirmation budget, so queueing behind it is bounded too -
        # otherwise the turn just hangs with nothing to tell the model.
        try:
            await asyncio.wait_for(
                self._browser.acquire(), CONFIRM_TIMEOUT + (timeout or self._timeout)
            )
        except asyncio.TimeoutError:
            raise ClientToolError(
                f"'{name}' was not started: another action is still waiting for "
                "the user to answer a confirmation dialog. Tell the user that, "
                "and do not report this as done."
            ) from None
        try:
            return await self._dispatch(name, args, timeout)
        finally:
            self._browser.release()

    async def _dispatch(self, name, args, timeout):
        if self._closed:
            raise ClientToolError("the chat session was closed")

        loop = asyncio.get_running_loop()
        call_id = str(uuid.uuid4())
        future = loop.create_future()
        self._pending[call_id] = future
        self._outstanding[call_id] = {
            "type": "function_call", "id": call_id, "name": name, "args": args
        }
        budget = timeout or self._timeout
        started = loop.time()
        self._deadlines[call_id] = started + budget
        # However many keep-alives arrive, the call cannot outlive this. The
        # browser is told the id, so without a ceiling a looping or hostile
        # parent page could pin the call - and the lock - for the whole session.
        self._ceilings[call_id] = started + budget + CONFIRM_TIMEOUT
        try:
            self._publish_calls()
            while True:
                # .get: close() may have cleared the deadlines between the wait
                # expiring and this read; a KeyError here would bypass
                # handle_tool_error and surface as a raw traceback.
                remaining = self._deadlines.get(call_id, 0.0) - loop.time()
                if remaining <= 0:
                    waited = round(loop.time() - started)
                    raise ClientToolError(
                        f"the browser did not answer '{name}' within {waited}s. "
                        "The action may still have completed - do not retry it "
                        "blindly and do not report success. Re-read the page "
                        "and tell the user what you actually found."
                    ) from None
                try:
                    # shield: the wait expiring must not cancel the call itself,
                    # since a progress message may have extended its deadline.
                    result = await asyncio.wait_for(
                        asyncio.shield(future), remaining
                    )
                    break
                except asyncio.TimeoutError:
                    continue   # re-read the deadline, which may have moved
        finally:
            self._pending.pop(call_id, None)
            self._deadlines.pop(call_id, None)
            self._ceilings.pop(call_id, None)
            if self._outstanding.pop(call_id, None) is not None and not self._closed:
                self._publish_calls()

        if result is None:
            raise ToolRefusal(f"'{name}' returned no result")
        # Convention for the write tools: they report refusals as data rather
        # than throwing, so the browser can phrase the reason. Raise here so the
        # model is told instead of reading {'ok': false} as a success.
        if isinstance(result, dict) and result.get("ok") is False:
            raise ToolRefusal(
                result.get("message") or f"'{name}' failed ({result.get('code')})"
            )
        return result

    ## tools

    def _build_tools(self):
        # Closure over the session instead of a module global. The docstrings
        # below are the tool descriptions the LLM sees - keep them verbatim.
        session = self

        @langchain_core.tools.tool
        async def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""

            return await session.call_client_side_tool("multiply", [a, b])

        @langchain_core.tools.tool
        async def where_am_i() -> WebPage:
            """Returns the current user (id and name), the window.location url, the document.title and body html of the users browser client."""

            response = await session.call_client_side_tool("where_am_i", [])
            if not isinstance(response, dict):
                raise ToolRefusal("where_am_i: unexpected response from the browser")
            try:
                return WebPage(**response)
            except ValidationError as e:
                # The browser is not a trusted source - never let a surprising
                # payload surface as a raw traceback in the chat.
                raise ToolRefusal(f"where_am_i: malformed response: {e}") from e

        @langchain_core.tools.tool
        async def highlight_html_element(x_path) -> str:
            """Hihglights the html element with the given x_path. Returns 'success' if the element was found and highlighted, else 'failure'."""

            return await session.call_client_side_tool("highlight_html_element", [x_path])

        @langchain_core.tools.tool
        async def redirect(page: str) -> str:
            """Redicts the user to the given page title or url. A page title must contain the namespace (e.g. 'Category:' or 'Item:'). Returns 'accepted' if the redirect was successful, else 'rejected'."""

            return await session.call_client_side_tool("redirect", [page])

        @langchain_core.tools.tool
        async def full_text_search(query) -> str:
            """Finds pages by running a full text search over everything indexed: page titles, wikitext, and the extracted text of uploaded files such as PDFs. Searches all namespaces, so a phrase that appears only inside an uploaded document is found too.

            Returns {'totalhits': n, 'results': [{'title', 'snippet'}]}, where the snippet is the matching passage as plain text. An empty 'results' list means nothing matched - that is an answer, not an error. Pass a title to 'get_page_content' to read the whole page.
            """
            return await session.call_client_side_tool("full_text_search", [query])

        @langchain_core.tools.tool
        async def find_page_from_topic(topic) -> List[Dict[str, str]]:
            """Finds a page for a given topic for searching titles were the topic is contained in the label.
            Returns a list of results with title, description and type
            """
            return await session.call_client_side_tool("find_page_from_topic", [topic])

        @langchain_core.tools.tool
        async def get_page_content(titles: List[str], include_html: bool) -> dict:
            """Gets the content of one or multiple pages by their title. A page title must contain the namespace (e.g. 'Category:' or 'Item:').
            The structured content and wikitext source is also returned.
            If include_html is true the html content of the page also returned. This is more expensive but may also contain aggregated content from other pages,
            e.g. if the page contains a overview list.
            """
            # Pages written in this session are answered from what was written. The model
            # re-reads them because the transcript carries no record of its own writes;
            # fetching again costs a round trip to say what we already know. Anything
            # needing the *rendered* page still goes to the wiki.
            if not include_html:
                known = [t for t in titles if t in session.written_pages]
                if known and len(known) == len(titles):
                    return {
                        t: {
                            "slots": session.written_pages[t],
                            "note": (
                                "written by you earlier in this session; the wiki was not "
                                "queried again. Pass include_html=true to re-read it."
                            ),
                        }
                        for t in titles
                    }

            return await session.call_client_side_tool(
                "get_page_content", [titles, include_html]
            )

        @langchain_core.tools.tool
        async def create_category_instance(
            category_page: str, instance_description: Optional[str] = None
        ) -> str:
            """Opens the form editor for a new instance of a category, prefilled from the given description, for the user to review and save.

            Prefer 'edit_page' instead whenever you can fill the data yourself - for an entity whose only meaningful field is a label, such as a keyword, the editor is needless work for the user. Use this tool only when the category has required fields you cannot infer and the user should fill them in.

            It returns as soon as the editor is open - the user then has as long as they like to check the fields, and the page only exists once they save it. Never report it as created.

            Only one editor can be open at a time, so open a single one and tell the user what to do with it. If several instances are needed, write them with 'edit_page' instead, or do them one after another as the user saves each."""

            default_data = None
            unresolved: List[str] = []
            if instance_description is not None:
                try:
                    org_prompt = instance_description
                    prompt = org_prompt
                    prompt += "\nCreate only attributes that are defined in the schema. If you are not sure about an attributes, leave it empty.\n\n"
                    prompt += (
                        "A property with \"format\": \"autocomplete\" (usually carrying a "
                        "\"range\", \"subclassof_range\" or an autocomplete \"query\") holds a "
                        "page title of the form '<Namespace>:OSW<uuid without hyphens>', never "
                        "a name, a phrase or an object. If you do not know the exact page title, "
                        "leave the property out entirely.\n\n"
                    )
                    force_websearch = False
                    if force_websearch:
                        try:
                            prompt += "\nUse the following addtional information\n\n"
                            res = await websearch_invoke(
                                "Search in the web for addition information that could help to resolve the following request:\n"
                                + org_prompt
                            )
                            prompt += res["output"]
                        except Exception as e:
                            print(e)
                    jsonschema = await session.call_client_side_tool(
                        "get_category_schema", [category_page]
                    )
                    # sync (requests) - off the event loop, it would stall every session
                    default_data_res = await asyncio.to_thread(
                        get_llm_response, prompt, jsonschema, None, None, False
                    )
                    default_data = default_data_res["result"]
                    print("DESCRIPTION", instance_description, "DATA", default_data)
                    default_data, unresolved = await session._resolve_references(
                        default_data, jsonschema
                    )
                except Exception as e:
                    print(e)
            opened = await session.call_client_side_tool(
                "create_category_instance", [category_page, default_data]
            )
            if unresolved:
                # Say it here rather than leaving the user to notice the gaps:
                # these fields reference pages, and the referenced pages do not
                # exist yet or could not be matched.
                return (
                    f"{opened}\nThe editor is open, but these reference fields were "
                    "left empty because no matching page was found: "
                    + "; ".join(unresolved)
                    + ". Tell the user, and offer to create the missing entities "
                    "(with 'edit_page', as instances of the category the field "
                    "expects) before they save. Never type a name into such a field."
                )
            return opened

        @langchain_core.tools.tool
        async def smw_ask_query(query) -> dict:
            """Runs a Semantic Mediawiki Ask API Query, e.g. to get all pages in a category with their properties.
            Param `query` is only the query string with the select / filter conditions, no other parameters.
            To get existing SMW properties consult the categories schema with the get_category_schema tool.
            """
            return await session.call_client_side_tool("smw_ask_query", [query])

        @langchain_core.tools.tool
        async def get_file_content(file_title, query: str = "") -> str:
            """Gets the content of a wiki file by its title (File:OSW...). Can handle images and documents.
            For documents such as PDFs this returns the text the wiki already extracted, page by page. Each page comes with the url that renders exactly that page, so cite a quote by linking to it - never cite a page number without the link, and never build such a url by hand. Pass 'query' with the term you are looking for and only the pages containing it come back - do that for a long document rather than reading it from the start. Long results are truncated and say so.
            Important: Use this function only for file in the current wiki domain, not for external urls retrieved via web search!
            For file links from web search use the web search tool to get the content.
            """
            # The indexed text first. Fetching the file itself meant base64 over
            # the websocket - a 16 MB pdf becomes ~21.5 MB, past bokeh's 20 MB
            # message limit, so the reply was dropped and the call hung until it
            # timed out. The wiki already has the text; ask for that.
            try:
                indexed = await session.call_client_side_tool(
                    "get_file_text", [file_title, FILE_TEXT_MAX_CHARS, query or ""]
                )
            except ClientToolError:
                indexed = None
            if isinstance(indexed, dict):
                rendered = _render_file_text(indexed, query)
                if rendered:
                    return rendered

            result = await session.call_client_side_tool(
                "get_file_data_url", [file_title, MAX_INLINE_FILE_BYTES]
            )

            # result is a data_url, e.g. data:/plain;base64,....

            # for images, generate a description
            if result.startswith("data:image/"):
                text = await multimedia_data_url_to_text(result)

            else:
                # use textract to extract the text content
                from osw_chatbot.structured_output.util import data_url_to_text

                # textract is blocking - keep it off the shared event loop
                text = await asyncio.to_thread(data_url_to_text, file_title, result)

            return text

        @langchain_core.tools.tool
        async def upload_file(attachment_id: str, label: str = "") -> dict:
            """Uploads a file the user attached to this chat into the wiki and returns its page title, e.g. 'File:OSW1234....png'. Use that title as the value of an entity's image or file property. The attachment id is given in the message that carried the attachment. Only upload when the user wants the file stored in the wiki - reading or describing an attachment needs no upload."""

            return await session.call_client_side_tool("upload_file", [attachment_id, label])

        @langchain_core.tools.tool
        async def get_category_schema(category_page: str) -> dict:
            """Returns the JSON schema of a category, i.e. which properties an instance of it can have and which are required. Read this before creating an instance so the data you write is valid and you use the right property names (the property holding a picture, for example, differs between categories)."""

            return await session.call_client_side_tool("get_category_schema", [category_page])

        @langchain_core.tools.tool
        async def edit_page(
            title: str, slots: Union[Dict[str, Any], str], summary: str = ""
        ) -> dict:
            """Creates or edits a wiki page. Only the given slots are changed, everything else on the page stays as it is. A page title must contain the namespace (e.g. 'Category:', 'Item:' or 'User:').

            Anything that is a thing - a device, an event, a person, a project, a sample, a method - is created as an *instance of its category*, not as a plain wikitext page: find the category with 'find_page_from_topic', read it with 'get_category_schema', then write an 'Item:OSW<uuid without dashes>' page whose 'jsondata' slot is {'type': ['Category:OSW...'], 'uuid': '<uuid>', 'label': [{'text': ..., 'lang': ...}], ...}.
            The only pages that are plain wikitext or source are the user's own 'User:<name>/...' subpages, '.js'/'.css' pages, and the main page.

            Properties with "format": "autocomplete" (usually with a "range", "subclassof_range" or an autocomplete "query") are references, not free text: their value is a page title like 'Item:OSW<uuid without hyphens>', and it must satisfy that range or query. Resolve it to a real page first - never write a label or an invented id into such a field.

            Never put structured data into 'main'. On Item: and Property: pages all structured content goes into the 'jsondata' slot and 'main' stays empty wikitext - JSON written to 'main' shows up as a wall of raw text, leaves the entity without data and makes it invisible to queries.

            `slots` maps a slot name to its content. Choose by what the content is *for*:
            - content of this page itself -> 'main'. Wikitext only. Exceptions: on pages ending in .js or .css it is JavaScript or CSS source, and in the legacy 'JsonSchema:' namespace it is a JSON object.
            - content that should appear on *every instance* of a category -> 'header_template' (rendered above the page) or 'footer_template' (below). Wikitext only.
            - structured data -> 'jsondata', a JSON object (not a string). 'jsonschema' likewise.
            - 'schema_template' and 'data_template': template source (Handlebars/wikitext).
            - 'header' and 'footer': NEVER write these. They always carry the default entity invokes and nothing else; the call is rejected if you include them.

            Write wikitext, never Markdown - read the 'osl-page-editing' skill first if you are unsure.
            Scripts (.js/.css) may only be written inside the current user's own space, i.e. 'User:<name>/...'.
            Returns {'ok': true, 'title', 'created'} or raises with the reason the wiki refused.
            """
            needed = required_skill(title)
            if needed in session.skills and needed not in session.loaded_skills:
                raise ToolRefusal(
                    f"read the '{needed}' skill before writing this page: call "
                    f"load_skill('{needed}'), follow it, then make this edit again"
                )
            # Models regularly hand the whole slot map over as a JSON string.
            # Accepting it is cheaper than a failed turn: a pydantic
            # ValidationError is not a ToolException, so it bypasses
            # handle_tool_error and takes the conversation down with a traceback.
            if isinstance(slots, str):
                try:
                    slots = json.loads(slots)
                except ValueError as exc:
                    raise ToolRefusal(
                        "'slots' must be an object mapping slot name to content, "
                        f"and the string given is not valid JSON ({exc})"
                    ) from None
            if not isinstance(slots, dict):
                raise ToolRefusal(
                    "'slots' must be an object mapping slot name to content, "
                    f"got {type(slots).__name__}"
                )
            # Instantiating a category means following that category's conventions, and
            # the wiki is where they are written down. Checked before the write, not
            # after, because the point is to shape what gets written.
            for category in (slots.get("jsondata") or {}).get("type") or []:
                if not isinstance(category, str):
                    continue
                for required in await session.skills_for_category(category):
                    if required not in session.loaded_skills:
                        raise ToolRefusal(
                            f"'{category}' has a skill on this wiki that has to be read "
                            f"first: call load_skill('{required}'), follow it, then make "
                            "this edit again"
                        )

            managed = sorted(MANAGED_SLOTS.intersection(slots))
            if managed:
                # A prompt rule was not enough - the model writes these anyway, and a
                # replaced header/footer breaks the entity rendering on every instance.
                raise ToolRefusal(
                    f"{', '.join(repr(s) for s in managed)} cannot be written: header and "
                    "footer always carry the default invokes and nothing else. Content for "
                    "this page goes in 'main'; content that should appear on every "
                    "instance of a category goes in 'header_template' (above) or "
                    "'footer_template' (below); structured data goes in 'jsondata'."
                )
            osw_id = osw_id_of(title)
            if osw_id is not None and osw_id not in session.minted_ids:
                # A near-miss of an id this session already handled is a transcription
                # slip, not a new entity: creating it would leave two half-written pages
                # and the user looking at whichever one the model mentions last.
                known = set(session.minted_ids) | {
                    i
                    for i in (osw_id_of(t) for t in session.written_pages)
                    if i is not None
                }
                near = nearest_known_id(osw_id, known)
                if near is not None:
                    raise ToolRefusal(
                        f"'{title}' differs by a few characters from OSW{near.upper()}, "
                        "which this session already worked on. Use that title exactly as "
                        "written, or call 'new_osw_id' if you really mean a new entity."
                    )
                # Either this is a real page being edited, or the model made the
                # id up. Only the wiki can tell us which.
                if not await session._page_exists(title):
                    raise ToolRefusal(
                        f"'{title}' does not exist and its OSW-ID was not issued "
                        "here. Do not invent OSW-IDs - call 'new_osw_id' and use "
                        "exactly what it returns as the title of the new entity."
                    )
            if osw_id is not None and isinstance(slots.get("jsondata"), dict):
                # The uuid in the data and the id in the title are the same
                # thing; a mismatch leaves the entity unfindable.
                slots["jsondata"]["uuid"] = uuid_from_osw_id(osw_id)
            result = await session.call_client_side_tool(
                "edit_page", [title, slots, summary]
            )
            # Only on success, and merged rather than replaced: edit_page changes just the
            # slots it is given, so a later read must still see the ones written earlier.
            session.written_pages.setdefault(title, {}).update(slots)
            return result

        @langchain_core.tools.tool
        async def delete_page(title: str, reason: str = "") -> dict:
            """Deletes a wiki page. A page title must contain the namespace (e.g. 'Item:'). Returns {'ok': true} or raises with the reason the wiki refused. Ask the user before deleting anything."""

            return await session.call_client_side_tool("delete_page", [title, reason])

        @langchain_core.tools.tool
        async def new_osw_id() -> str:
            """Returns a fresh OSW-ID to use as the page title of a new entity, e.g. 'OSW1fe1a1b…'. Call this once per new entity and use exactly what it returns - 'Item:<id>', 'Category:<id>'. Never make up an OSW-ID yourself: they must be random and globally unique, and a made up one collides with other wikis or points at nothing."""

            osw_id = uuid.uuid4().hex
            session.minted_ids.add(osw_id)
            return "OSW" + osw_id

        @langchain_core.tools.tool
        async def open_sandbox(notebook: str = "") -> str:
            """Opens the in-browser Python sandbox in its own tab, optionally on a notebook already uploaded to this wiki. Pass a file page title such as 'File:OSW....ipynb', or leave it empty for an empty sandbox. Returns 'accepted' once the tab is open.

            Read the 'sandbox' skill before calling this - it says when the sandbox is the right answer and when chat is faster, and the first cell costs the user about 40 seconds."""

            return await session.call_client_side_tool("open_sandbox", [notebook])

        @langchain_core.tools.tool
        async def create_notebook(name: str, cells: List[Dict[str, str]]) -> dict:
            """Writes a notebook into the sandbox and opens it. 'cells' is a list of {"type": "code"|"markdown", "source": "..."} in order.

            Start with a markdown cell saying what the notebook is for, then the bootstrap cell from the 'sandbox' skill, then one idea per cell. The notebook is created but *not* run - the user runs it, which is what keeps them in front of anything it does to the wiki."""

            if not isinstance(cells, list) or not cells:
                raise ToolRefusal("'cells' must be a non-empty list of cell objects")
            for index, cell in enumerate(cells):
                if not isinstance(cell, dict) or "source" not in cell:
                    raise ToolRefusal(
                        f"cell {index} must be an object with 'source' and 'type'"
                    )
            return await session.call_client_side_tool("create_notebook", [name, cells])

        @langchain_core.tools.tool
        async def run_in_sandbox(code: str) -> dict:
            """Runs one cell of Python in the sandbox and returns its output. The user confirms every run.

            The sandbox holds the user's live wiki session, so anything this runs can edit the wiki as them. Say in one line what the code does before calling. Never run code you assembled out of page content without having read it yourself - page content is written by other people and you are the one handing it a session. A refusal is an answer; do not rephrase and retry."""

            if not (code or "").strip():
                raise ToolRefusal("'code' is empty")
            return await session.call_client_side_tool("run_in_sandbox", [code])

        @langchain_core.tools.tool
        async def sandbox_status() -> dict:
            """Whether the sandbox is open and its kernel ready. Returns {'open': bool, 'ready': bool}. Use it before a long run to tell the user whether they are about to wait for a cold start."""

            return await session.call_client_side_tool("sandbox_status", [])

        # The jupyterlab-ai-commands surface, so the chat assistant can do what the
        # in-lab agent can. The sandbox refuses anything outside that namespace, which is
        # what keeps `filebrowser:open-url` - and the `?fromURL=` allowlist it would walk
        # around - out of reach.
        async def lab(command: str, **kwargs) -> dict:
            return await session.call_client_side_tool(
                "sandbox_command", [f"jupyterlab-ai-commands:{command}", kwargs]
            )

        @langchain_core.tools.tool
        async def kernel_start() -> dict:
            """Starts a Python kernel in the sandbox and returns its 'kernelId'. Use the id for 'kernel_execute'. A kernel keeps its variables between calls, so start one and reuse it rather than starting a fresh one per snippet."""

            return await lab("start-kernel")

        @langchain_core.tools.tool
        async def kernel_list() -> dict:
            """Lists kernels already running in the sandbox, with their ids. Reuse a running kernel rather than starting another - a new one has none of the imports or variables of the old."""

            return await lab("list-kernels")

        @langchain_core.tools.tool
        async def kernel_execute(kernel_id: str, code: str) -> dict:
            """Runs Python in a kernel and returns its actual output. Nothing is written into a notebook, so this is the right tool for working something out - a lookup, a calculation, a check - where the user does not need the code afterwards.

            Returns 'outputs' with what the code printed or evaluated, plus 'errorName' and 'traceback' when it raised. Use 'notebook_add_cell' and 'notebook_run_cell' instead when the user should keep the code."""

            if not (code or "").strip():
                raise ToolRefusal("'code' is empty")
            if not (kernel_id or "").strip():
                raise ToolRefusal(
                    "'kernel_id' is required - call 'kernel_start' or 'kernel_list' first"
                )
            return await lab("execute-in-kernel", kernelId=kernel_id, code=code)

        @langchain_core.tools.tool
        async def kernel_shutdown(kernel_id: str) -> dict:
            """Shuts a kernel down and frees its memory. The user loses every variable in it, so only when they are finished."""

            return await lab("shutdown-kernel", kernelId=kernel_id)

        @langchain_core.tools.tool
        async def notebook_info(notebook_path: str = "") -> dict:
            """What notebook is open in the sandbox: its path, its cells and which cell is active. Call it before adding or running a cell so you act on the notebook you mean."""

            return await lab("get-notebook-info", notebookPath=notebook_path)

        @langchain_core.tools.tool
        async def notebook_add_cell(
            notebook_path: str, content: str, cell_type: str = "code"
        ) -> dict:
            """Appends one cell to a notebook. 'cell_type' is 'code' or 'markdown'. The cell is added but not run - 'notebook_run_cell' runs it."""

            if cell_type not in ("code", "markdown"):
                raise ToolRefusal("'cell_type' must be 'code' or 'markdown'")
            return await lab(
                "add-cell",
                notebookPath=notebook_path,
                content=content,
                cellType=cell_type,
            )

        @langchain_core.tools.tool
        async def notebook_run_cell(notebook_path: str, cell_id: str) -> dict:
            """Runs one cell of a notebook and returns its output, so the user keeps the code and you see the result. Get 'cell_id' from 'notebook_info' - running the wrong cell is worse than running none."""

            return await lab("run-cell", notebookPath=notebook_path, cellId=cell_id)

        @langchain_core.tools.tool
        async def notebook_delete_cell(notebook_path: str, cell_id: str) -> dict:
            """Deletes one cell from a notebook. Ask the user before removing anything they wrote."""

            return await lab("delete-cell", notebookPath=notebook_path, cellId=cell_id)

        @langchain_core.tools.tool
        async def notebook_save(notebook_path: str) -> dict:
            """Saves the notebook to the sandbox's own storage. That storage is the browser's, so it does not survive clearing site data - offer to upload the notebook to the wiki for anything worth keeping."""

            return await lab("save-notebook", notebookPath=notebook_path)

        @langchain_core.tools.tool
        async def load_skill(name: str) -> str:
            """Returns detailed instructions for a task, which you must follow. Call this *before* writing any wiki page and before generating any component or script - not after. It carries the conventions this wiki actually enforces (slots, schemas, which APIs to call), and code written without it is reliably wrong in ways that are invisible until a page renders blank. Reading it is cheap; redoing the work is not.

            Skills that ship with the assistant are listed in the system prompt. This wiki may also hold its own, written by its editors - use 'discover_skills' to find those, then load them by name here too."""

            if name in session.loaded_skills:
                # Returning the body again would spend the whole skill a second time for
                # nothing. The instructions are already upstream in this conversation.
                return (
                    f"'{name}' is already loaded earlier in this session - its "
                    "instructions are above and still apply. Follow them; do not ask for "
                    "it again."
                )

            skill = session.skills.get(name)
            if skill is not None:
                session.loaded_skills.add(name)
                return skill.body

            # Not a bundled skill - try the wiki. Done here rather than by loading every
            # wiki skill up front because the browser bridge is not live when the session
            # is constructed, which is when the bundled ones are read and the system
            # prompt is built.
            body = await session.load_wiki_skill(name)
            if body is None:
                available = ", ".join(session.skills) or "none"
                raise ToolRefusal(
                    f"no skill named {name!r}. Bundled: {available}. "
                    "Call 'discover_skills' for the ones this wiki holds."
                )
            session.loaded_skills.add(name)
            return body

        @langchain_core.tools.tool
        async def discover_skills(query: str = "") -> List[Dict[str, str]]:
            """Lists the skills available on this wiki, written by its editors. Returns a list of {name, description, title}; pass a name to 'load_skill' to read one.

            Use it when a task looks wiki-specific and nothing in the system prompt covers it: local conventions, a house style for a category, a procedure this site follows. 'query' narrows by substring over names and descriptions; leave it empty to see everything."""

            found = await session.list_wiki_skills(query)
            if found:
                return found

            # A wiki that holds no skills yet should still get the ones shipped here,
            # rather than an empty list that reads as "this wiki has no conventions".
            needle = (query or "").strip().lower()
            return [
                {"name": s.name, "description": s.description, "title": ""}
                for s in session.skills.values()
                if not needle
                or needle in s.name.lower()
                or needle in s.description.lower()
            ]

        @langchain_core.tools.tool
        async def save_skill(
            name: str, description: str, instructions: str, title: str = ""
        ) -> dict:
            """Writes a skill to the wiki as an instance of Category:Skill, so it is available to this assistant and to the notebook agent from now on. Opens a confirmation first - it is a wiki edit, made in the user's name.

            'name' is a slug the skill is called by, lower case with hyphens, e.g. 'tensile-test-reporting'. 'description' is the single line every assistant sees for this skill on every turn, so say *when* to reach for it, not what it contains. 'instructions' is the body, in markdown. Pass 'title' to update an existing skill page; leave it empty to create one.

            Write a skill when the user has taught you something durable about this wiki - a convention, a procedure, a shape of query that keeps coming up. Do not write one to remember a single answer."""

            if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name or ""):
                raise ToolRefusal(
                    f"{name!r} is not a slug - use lower case words joined by hyphens, "
                    "e.g. 'tensile-test-reporting'"
                )
            if not description.strip() or not instructions.strip():
                raise ToolRefusal("description and instructions are both required")
            return await session.save_wiki_skill(
                name=name,
                description=description.strip(),
                instructions=instructions.strip(),
                title=title.strip() or None,
            )

        tools = [
            multiply,
            where_am_i,
            highlight_html_element,
            redirect,
            full_text_search,
            # the "user said 'device', find me the category" lookup - the
            # extension has always implemented it, it was simply never
            # registered, which left the model guessing at category titles
            find_page_from_topic,
            create_category_instance,
            get_category_schema,
            get_page_content,
            smw_ask_query,
            get_file_content,
            edit_page,
            delete_page,
            upload_file,
            new_osw_id,
            discover_skills,
            save_skill,
        ]
        # One switch for the whole sandbox, not per tool: an assistant that can write a
        # notebook but not open it, or open one but not run it, is worse than one with no
        # sandbox at all. The wiki signs the action list into the token, so this cannot be
        # turned on from here.
        if "code-sandbox" in session.actions:
            tools += [
                open_sandbox, create_notebook, run_in_sandbox, sandbox_status,
                kernel_start, kernel_list, kernel_execute, kernel_shutdown,
                notebook_info, notebook_add_cell, notebook_run_cell,
                notebook_delete_cell, notebook_save,
            ]
        # load_skill is registered whenever *any* source of skills exists. The wiki's own
        # are discovered at call time, so the bundled ones being empty does not mean there
        # is nothing to load.
        tools.append(load_skill)
        for tool in tools:
            tool.handle_tool_error = _tool_error_observation
            tool.handle_validation_error = _tool_input_observation
            _bounded(tool)
        return tools

    async def _resolve_references(self, data, schema):
        """Turn reference values into real page titles, or drop them.

        The prefill pass writes names and label objects into fields whose range
        says 'a page title'. Look each one up in the range it must satisfy; an
        unresolved value is removed, because a wrong title is indistinguishable
        from a right one once it is in the form.
        """
        from osw_chatbot.structured_output.util import merge_all_of

        if not isinstance(data, (dict, list)):
            return data, []

        try:
            merged = merge_all_of(schema)
        except Exception:
            merged = schema

        unresolved: List[str] = []
        for container, key, path, value, constraint in list(
            prefill.find_references(data, merged)
        ):
            if prefill.looks_like_reference(value):
                continue
            text = prefill.label_text(value)
            if not text:
                prefill.drop(container, key)
                continue
            title = await self._lookup_reference(text, constraint)
            if title:
                container[key] = title
            else:
                prefill.drop(container, key)
                unresolved.append(
                    f"{path} = {text!r} (needs {prefill.describe_constraint(constraint)})"
                )
        return prefill.prune_empty(data), unresolved

    async def _wiki_skill_pages(self, refresh=False):
        """Every Category:Skill page on this wiki, as {name: {...}}.

        Read through the browser, not from here: this process holds no wiki credential by
        design, and on a wiki that denies anonymous reads - the dev wiki does - a
        server-side fetch would see nothing. Going through the user's session also means
        a skill they may not read is one they do not get told about.

        Cached per session. The bridge is only live once the widget is mounted, which is
        why this cannot happen in __init__ where the bundled skills are read.
        """
        if self._wiki_skills is not None and not refresh:
            return self._wiki_skills

        try:
            listing = await self.call_client_side_tool(
                "smw_ask_query", [f"{SKILL_DISCOVERY_QUERY}|limit=500"]
            )
        except ClientToolError:
            # No skills is a perfectly good answer; a failed lookup must not end the turn.
            self._wiki_skills = {}
            return self._wiki_skills

        titles = list(((listing or {}).get("query", {}).get("results") or {}).keys())
        skills = {}
        # Batched: the browser does one api.php round trip per call, and a wiki with
        # thirty skills should not cost thirty of them.
        for start in range(0, len(titles), 50):
            batch = titles[start : start + 50]
            try:
                data = await self.call_client_side_tool(
                    "get_page_content", [batch, False]
                )
            except ClientToolError:
                continue
            pages = ((data or {}).get("query", {}) or {}).get("pages") or {}
            for page in pages.values():
                slots = (page.get("revisions") or [{}])[0].get("slots") or {}
                raw = (slots.get("jsondata") or {}).get("*")
                if not raw:
                    continue
                try:
                    jsondata = json.loads(raw)
                except ValueError:
                    continue
                applies = jsondata.get("applies_to") or []
                if applies and "osw-chatbot" not in applies:
                    continue
                name = jsondata.get("skill_name") or jsondata.get("name")
                if not name:
                    continue
                skills[name] = {
                    "name": name,
                    "title": page.get("title", ""),
                    "description": _first_text(jsondata.get("description"))
                    or _first_text(jsondata.get("label")),
                    "body": (slots.get("main") or {}).get("*") or "",
                    # Target concepts this skill applies to - Categories, mostly. Read
                    # straight off the page rather than queried: none of the skill
                    # category's fields are exported as SMW properties, so there is
                    # nothing to ask for, and the jsondata is already parsed here.
                    "related_to": [
                        r for r in (jsondata.get("related_to") or []) if isinstance(r, str)
                    ],
                    # Prerequisite chain. Real content already uses it: "OSW Python
                    # Package" requires "JSON-SCHEMA Tutorial".
                    "required_predecessors": [
                        r
                        for r in (jsondata.get("required_predecessor") or [])
                        if isinstance(r, str)
                    ],
                }
        self._wiki_skills = skills
        return skills

    async def _superclasses(self, category):
        """`category` and everything it is a subclass of, nearest first.

        Cached per session: a gate that resolved this on every write would cost a round
        trip per edit, and the hierarchy does not move during a conversation.
        """
        if category in self._superclass_cache:
            return self._superclass_cache[category]

        chain, seen, current = [], set(), category
        while current and current not in seen:
            seen.add(current)
            chain.append(current)
            # Someone already walked from here upwards. Reuse that tail rather than
            # re-fetching it: hierarchies converge, so the second category in a session
            # usually meets the first one's chain within a level or two.
            known = self._superclass_cache.get(current)
            if known is not None and len(chain) > 1:
                chain = chain[:-1] + known
                break
            try:
                data = await self.call_client_side_tool(
                    "get_page_content", [[current], False]
                )
            except ClientToolError:
                break
            pages = ((data or {}).get("query", {}) or {}).get("pages") or {}
            parent = None
            for page in pages.values():
                slots = (page.get("revisions") or [{}])[0].get("slots") or {}
                raw = (slots.get("jsondata") or {}).get("*")
                if not raw:
                    continue
                try:
                    parents = json.loads(raw).get("subclass_of") or []
                except ValueError:
                    continue
                parent = next((p for p in parents if isinstance(p, str)), None)
            current = parent
        # Cache every suffix, not only the category asked for. Walking Tutorial also
        # answers Article, CreativeWork, Item and Entity for free, which is most of
        # the cost of the next lookup.
        for i, step in enumerate(chain):
            self._superclass_cache.setdefault(step, chain[i:])
        return chain

    async def skills_for_category(self, category):
        """Skill names that must be loaded before instantiating `category`.

        A skill on a supercategory covers its subcategories - a convention for "Device"
        applies to every kind of device - so the whole chain is considered. Prerequisites
        are pulled in transitively, nearest requirement last so the chain reads in the
        order it should be followed.
        """
        skills = await self._wiki_skill_pages()
        if not any(s.get("related_to") for s in skills.values()):
            # Nothing is gated on this wiki, so the chain walk - five sequential round
            # trips through the browser - would cost that to learn nothing.
            return []

        by_title = {s["title"]: s for s in skills.values() if s.get("title")}
        chain = await self._superclasses(category)

        matched, seen = [], set()

        def pull(skill):
            # Depth first through required_predecessor, guarding against a cycle: the
            # property is authored by hand and nothing on the wiki prevents one.
            if skill["name"] in seen:
                return
            seen.add(skill["name"])
            for predecessor in skill.get("required_predecessors") or []:
                earlier = by_title.get(predecessor)
                if earlier is not None:
                    pull(earlier)
            matched.append(skill["name"])

        for target in chain:
            for skill in skills.values():
                if target in (skill.get("related_to") or []):
                    pull(skill)
        return matched

    async def list_wiki_skills(self, query=""):
        """Name, description and title of each wiki skill, optionally filtered."""
        skills = await self._wiki_skill_pages()
        needle = (query or "").strip().lower()
        return [
            {"name": s["name"], "description": s["description"], "title": s["title"]}
            for s in skills.values()
            if not needle
            or needle in s["name"].lower()
            or needle in s["description"].lower()
        ]

    async def load_wiki_skill(self, name):
        """The body of a wiki skill, or None if this wiki has no such skill."""
        skills = await self._wiki_skill_pages()
        skill = skills.get(name)
        return skill["body"] if skill else None

    async def save_wiki_skill(self, name, description, instructions, title=None):
        """Create or update a Category:Skill page, through the usual write path.

        Routed through `edit_page` rather than a new browser tool so it inherits
        everything that already guards a write: the confirmation dialog, the action gate,
        and the fact that it happens in the user's browser under their rights.
        """
        existing = (await self._wiki_skill_pages()).get(name)
        if title is None:
            title = existing["title"] if existing else f"Item:OSW{uuid.uuid4().hex}"
        if title.startswith("Item:OSW"):
            self.minted_ids.add(title.split("OSW", 1)[1].lower())

        if not await self._page_exists(SKILL_CATEGORY):
            # Writing the skill anyway would type it against a category this wiki does not
            # have: the page renders, nothing finds it, and nobody notices until someone
            # asks why the assistant has no skills.
            raise ToolRefusal(
                f"Category for skills not installed: {SKILL_CATEGORY} does not exist on "
                "this wiki. Install the opensemantic base vocabulary, or point "
                "SKILL_CATEGORY at the category this wiki uses."
            )

        jsondata = {
            "type": [SKILL_CATEGORY],
            "skill_name": name,
            "label": [{"text": name, "lang": "en"}],
            "description": [{"text": description, "lang": "en"}],
        }
        osw_id = osw_id_of(title)
        if osw_id is not None:
            jsondata["uuid"] = uuid_from_osw_id(osw_id)

        # The defaults go to the browser rather than being merged here: mwjson.util
        # .mergeDeep is the wiki's own merge - arrays concatenated and uniqued - and
        # reimplementing it in python would be a second copy to keep in step.
        result = await self.call_client_side_tool(
            "edit_page",
            [
                title,
                {"jsondata": jsondata, "main": instructions},
                f"Skill: {name}",
                await self._usable_skill_defaults(),
            ],
        )
        # The cache would otherwise hide the skill that was just written for the rest of
        # the conversation, which is exactly when it is most likely to be used.
        await self._wiki_skill_pages(refresh=True)
        return {"ok": True, "title": title, "name": name, "result": result}

    async def _usable_skill_defaults(self):
        """SKILL_DEFAULT_DATA with references this wiki does not have removed.

        The defaults name pages - the AI Skill keyword, the developers article - that exist
        on an OSW wiki with the standard vocabulary installed and nowhere else. Writing
        them regardless would leave dangling references that queries silently skip, so each
        is probed once and dropped if absent. Cached: this runs on every skill write.
        """
        if self._skill_defaults is not None:
            return self._skill_defaults

        usable = {}
        for key, value in SKILL_DEFAULT_DATA.items():
            if isinstance(value, list):
                kept = []
                for ref in value:
                    if not isinstance(ref, str) or osw_id_of(ref) is None:
                        kept.append(ref)  # not a page reference, nothing to check
                    elif await self._page_exists(ref):
                        kept.append(ref)
                if kept:
                    usable[key] = kept
            else:
                usable[key] = value
        self._skill_defaults = usable
        return usable

    async def _page_exists(self, title):
        """Does this wiki page exist? Used before trusting a model-chosen id."""
        try:
            data = await self.call_client_side_tool(
                "get_page_content", [[title], False]
            )
        except ClientToolError:
            # Do not block a write because the probe failed.
            return True
        pages = (data or {}).get("query", {}).get("pages")
        if not isinstance(pages, dict):
            return True
        return not any("missing" in page for page in pages.values())

    async def _lookup_reference(self, text, constraint):
        """The page in `constraint`'s range whose label is `text`, if unambiguous."""
        category = constraint.get("range") or constraint.get("subclassof_range")
        if not category:
            return None
        normalized = prefill.normalize_label(text)
        if not normalized:
            return None
        query = (
            f"[[{category}]][[HasNormalizedLabel::{normalized}]]"
            "|?HasNormalizedLabel|limit=2"
        )
        try:
            result = await self.call_client_side_tool("smw_ask_query", [query])
        except ClientToolError:
            return None
        results = (result or {}).get("query", {}).get("results", {})
        titles = [t for t in results if prefill.looks_like_reference(t)]
        # Exactly one match or nothing: picking from several would be a guess.
        return titles[0] if len(titles) == 1 else None

    ## conversation

    @staticmethod
    def _content(prompt, attachments):
        """Build the human turn, with any attachment inlined as content blocks."""
        if not attachments:
            return prompt
        blocks = []
        notes = []
        for att in attachments:
            name, mime = att.get("name", "file"), att.get("mime") or "unknown type"
            if att.get("error"):
                notes.append(f"[attachment {name} could not be read: {att['error']}]")
                continue
            notes.append(
                f"[attachment id={att.get('id')} name={name} type={mime} "
                f"size={att.get('size')} - use upload_file with this id to put it "
                "into the wiki]"
            )
            if att.get("preview"):
                blocks.append(
                    {"type": "image_url", "image_url": {"url": att["preview"]}}
                )
            elif att.get("text"):
                # ATTACHMENT_MAX_BYTES allows 20 MB, and this went into the prompt
                # whole - the other half of the same hole the tool results had.
                notes.append(f"Content of {name}:\n{_truncate(att['text'])}")
        return [{"type": "text", "text": "\n".join([prompt, *notes])}, *blocks]

    def _fit_prompt(self, content) -> bool:
        """Drop oldest turns until the prompt plausibly fits. True if anything went.

        Nothing counted the whole prompt before - only one observation at a time
        (MAX_TOOL_RESULT_CHARS), which does not bound their sum. Over the model's limit the
        request is refused outright rather than degrading, so a cheap estimate is worth
        spending to stay under it.

        The estimate is characters/4, not a real tokenizer: tokenizing the whole history
        every turn costs more than the imprecision does, and the budget is set well below
        the hard limit so the slack absorbs the error.
        """
        budget = MAX_PROMPT_CHARS
        fixed = len(SYSTEM_TEXT) + len(as_text(content))
        dropped = False
        while self.chat_history:
            used = fixed + sum(len(as_text(m.content)) for m in self.chat_history)
            if used <= budget:
                break
            # Two at a time: a turn is a human message and its reply, and leaving half
            # of one behind reads as the model ignoring what was just said.
            self.chat_history = self.chat_history[2:]
            dropped = True
        return dropped

    async def invoke(self, prompt, attachments=None, callbacks=None):
        self.chat_history = trim_history(self.chat_history)
        content = self._content(prompt, attachments)
        trimmed = self._fit_prompt(content)
        res = await self.agent_executor.ainvoke(
            {
                "input": [HumanMessage(content=content)],
                "chat_history": self.chat_history,
            },
            config={"callbacks": list(callbacks)} if callbacks else None,
        )
        # Anthropic returns a list of content blocks where OpenAI returns a
        # string; the chat UI and the stored history both want plain text.
        res["output"] = as_text(res["output"])
        if trimmed:
            # Say it rather than quietly forgetting: the user is the only one who can
            # tell whether what was dropped mattered.
            res["output"] += (
                "\n\n*(This conversation got long, so the earliest messages were dropped "
                "to stay within the model's context. Repeat anything from the start that "
                "still matters.)*"
            )
        self.chat_history.extend(
            [
                HumanMessage(content=prompt),
                AIMessage(content=res["output"]),
            ]
        )
        return res

    def clear(self):
        self.chat_history.clear()

    def close(self):
        self._closed = True
        for future in self._pending.values():
            if not future.done():
                future.get_loop().call_soon_threadsafe(
                    future.set_exception, ClientToolError("the chat session was closed")
                )
        self._pending.clear()
        self._outstanding.clear()
        self._deadlines.clear()
        self._ceilings.clear()
        self._seen_events.clear()
        self.chat_history.clear()
