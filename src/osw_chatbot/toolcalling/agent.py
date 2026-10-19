import asyncio
import copy
import json
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
EXCLUSIVE_CLIENT_TOOLS = {
    "edit_page",
    "delete_page",
    "upload_file",
    "redirect",
    "create_category_instance",
    "confirm_action",
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


def uuid_from_osw_id(osw_id: str) -> str:
    """OSW-IDs are a uuid with the hyphens taken out; put them back."""
    return str(uuid.UUID(osw_id))


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


def _tool_error_observation(error: Exception) -> str:
    """Feed a failed client-side tool back to the model as an observation.

    Previously a timeout returned None, which then blew up in `WebPage(**None)`
    and surfaced as a raw traceback in the chat. Handing the model a sentence
    instead lets it retry or tell the user what went wrong.
    """
    return (
        f"Error: {error}. The users browser could not be reached - do not retry "
        "more than once, tell the user instead."
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
    ):
        self.user = user  # verified wiki username, or None when unauthenticated
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
        # OSW-IDs this session handed out. Models do not generate random
        # uuids - they fill in a pattern (aa11bb22cc3344..., a1b2c3d4...),
        # which collides the moment two wikis coin the same entity.
        self.minted_ids: Set[str] = set()
        self.chat_history: List[BaseMessage] = []
        self.tools = self._build_tools()
        if "websearch" in self.actions:
            self.tools += [self._gated_websearch_tool(t) for t in websearch_tools]
        self.agent_executor = AgentExecutor(
            agent=create_tool_calling_agent(
                llm, self.tools, build_prompt(skill_index(self.skills))
            ),
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

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
                raise ClientToolError(
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
            raise ClientToolError(f"'{name}' returned no result")
        # Convention for the write tools: they report refusals as data rather
        # than throwing, so the browser can phrase the reason. Raise here so the
        # model is told instead of reading {'ok': false} as a success.
        if isinstance(result, dict) and result.get("ok") is False:
            raise ClientToolError(
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
                raise ClientToolError("where_am_i: unexpected response from the browser")
            try:
                return WebPage(**response)
            except ValidationError as e:
                # The browser is not a trusted source - never let a surprising
                # payload surface as a raw traceback in the chat.
                raise ClientToolError(f"where_am_i: malformed response: {e}") from e

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
            """Finds a pages by running a full text search in the indexed content (title, description, content). Returns a html list.
            Anchor links contain the page title (title attribute) that can be used to load the page with the get_page_content tool.
            Anchor links also contain the human label as inner text.
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
        async def get_file_content(file_title) -> str:
            """Gets the content of a wiki file by its title (File:OSW...). Can handle images and documents.
            Important: Use this function only for file in the current wiki domain, not for external urls retrieved via web search!
            For file links from web search use the web search tool to get the content.
            """
            result = await session.call_client_side_tool("get_file_data_url", [file_title])

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

            `slots` maps a slot name to its content:
            - 'jsondata' and 'jsonschema': a JSON object (not a string).
            - 'main': wikitext. Exceptions: on pages ending in .js or .css it is JavaScript or CSS source, and in the legacy 'JsonSchema:' namespace it is a JSON object.
            - 'header' and 'footer': wikitext, normally '{{#invoke:Entity|header}}' and '{{#invoke:Entity|footer}}'. These are set automatically when you write 'jsondata', so you rarely pass them.
            - 'schema_template', 'data_template', 'header_template', 'footer_template': template source (Handlebars/wikitext).

            Write wikitext, never Markdown - read the 'osl-page-editing' skill first if you are unsure.
            Scripts (.js/.css) may only be written inside the current user's own space, i.e. 'User:<name>/...'.
            Returns {'ok': true, 'title', 'created'} or raises with the reason the wiki refused.
            """
            needed = required_skill(title)
            if needed in session.skills and needed not in session.loaded_skills:
                raise ClientToolError(
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
                    raise ClientToolError(
                        "'slots' must be an object mapping slot name to content, "
                        f"and the string given is not valid JSON ({exc})"
                    ) from None
            if not isinstance(slots, dict):
                raise ClientToolError(
                    "'slots' must be an object mapping slot name to content, "
                    f"got {type(slots).__name__}"
                )
            osw_id = osw_id_of(title)
            if osw_id is not None and osw_id not in session.minted_ids:
                # Either this is a real page being edited, or the model made the
                # id up. Only the wiki can tell us which.
                if not await session._page_exists(title):
                    raise ClientToolError(
                        f"'{title}' does not exist and its OSW-ID was not issued "
                        "here. Do not invent OSW-IDs - call 'new_osw_id' and use "
                        "exactly what it returns as the title of the new entity."
                    )
            if osw_id is not None and isinstance(slots.get("jsondata"), dict):
                # The uuid in the data and the id in the title are the same
                # thing; a mismatch leaves the entity unfindable.
                slots["jsondata"]["uuid"] = uuid_from_osw_id(osw_id)
            return await session.call_client_side_tool("edit_page", [title, slots, summary])

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
        async def load_skill(name: str) -> str:
            """Returns detailed instructions for a task, which you must follow. Call this *before* writing any wiki page and before generating any component or script - not after. It carries the conventions this wiki actually enforces (slots, schemas, which APIs to call), and code written without it is reliably wrong in ways that are invisible until a page renders blank. Reading it is cheap; redoing the work is not."""

            skill = session.skills.get(name)
            if skill is None:
                available = ", ".join(session.skills) or "none"
                raise ClientToolError(f"no skill named {name!r}. Available: {available}")
            session.loaded_skills.add(name)
            return skill.body

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
        ]
        if session.skills:
            tools.append(load_skill)
        for tool in tools:
            tool.handle_tool_error = _tool_error_observation
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
                notes.append(f"Content of {name}:\n{att['text']}")
        return [{"type": "text", "text": "\n".join([prompt, *notes])}, *blocks]

    async def invoke(self, prompt, attachments=None, callbacks=None):
        self.chat_history = trim_history(self.chat_history)
        content = self._content(prompt, attachments)
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
