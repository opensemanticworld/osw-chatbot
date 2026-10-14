import asyncio
import copy
import json
import uuid
from typing import Any, Dict, List, Optional

import langchain_core.tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import ToolException
from pydantic import BaseModel, ValidationError

from osw_chatbot.chat.chat_panel_component import ChatFrontendWidget
from osw_chatbot.config import CLIENT_TOOL_TIMEOUT, PARENT_ORIGINS
from osw_chatbot.chat.history import trim_history
from osw_chatbot.llm import llm
from osw_chatbot.skills import load_skills, skill_index
from osw_chatbot.structured_output.llm import get_llm_response
from osw_chatbot.structured_output.util import multimedia_data_url_to_text
from osw_chatbot.websearch.interative_websearch import invoke as websearch_invoke
from osw_chatbot.websearch.interative_websearch import tools as websearch_tools


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
        self._closed = False
        self.frontend.param.watch(self._on_function_called, "function_called")

        self.skills = load_skills()
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

    def _on_function_called(self, event):
        data = event.new or {}
        future = self._pending.pop(data.get("id"), None)
        if future is not None and not future.done():
            # The watcher may fire off another thread - always hop back to the
            # loop the future was created on.
            future.get_loop().call_soon_threadsafe(future.set_result, data.get("result"))

    async def call_client_side_tool(self, name, args, timeout=None):
        """Ask this session's browser to run `name` and wait for its answer.

        Correlation is by uuid instead of the previous randint(0, 1000), and by
        an awaited Future instead of a 1 Hz poll of a single shared slot.
        """
        if self._closed:
            raise ClientToolError("the chat session was closed")

        call_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        # json round trip: a fresh object so param detects the change, and it
        # fails loudly here rather than during serialization to the browser.
        self.frontend.function_call = json.loads(
            json.dumps({"type": "function_call", "id": call_id, "name": name, "args": args})
        )
        try:
            result = await asyncio.wait_for(future, timeout or self._timeout)
        except asyncio.TimeoutError:
            raise ClientToolError(
                f"the browser did not answer '{name}' within {timeout or self._timeout}s"
            ) from None
        finally:
            self._pending.pop(call_id, None)

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
            """Opens an editor to create an instance for the given category page. A description of the instance can be provided that supports to fill out the fields. Returns 'success' if the editor was opened, else 'failure'."""

            default_data = None
            if instance_description is not None:
                try:
                    org_prompt = instance_description
                    prompt = org_prompt
                    prompt += "\nCreate only attributes that are defined in the schema. If you are not sure about an attributes, leave it empty.\n\n"
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
                except Exception as e:
                    print(e)
            return await session.call_client_side_tool(
                "create_category_instance", [category_page, default_data]
            )

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
        async def edit_page(title: str, slots: Dict[str, Any], summary: str = "") -> dict:
            """Creates or edits a wiki page. Only the given slots are changed, everything else on the page stays as it is. A page title must contain the namespace (e.g. 'Category:', 'Item:' or 'User:').

            Anything that is a thing - a device, an event, a person, a project, a sample, a method - is created as an *instance of its category*, not as a plain wikitext page: find the category with 'find_page_from_topic', read it with 'get_category_schema', then write an 'Item:OSW<uuid without dashes>' page whose 'jsondata' slot is {'type': ['Category:OSW...'], 'uuid': '<uuid>', 'label': [{'text': ..., 'lang': ...}], ...}.
            The only pages that are plain wikitext or source are the user's own 'User:<name>/...' subpages, '.js'/'.css' pages, and the main page.

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
            return await session.call_client_side_tool("edit_page", [title, slots, summary])

        @langchain_core.tools.tool
        async def delete_page(title: str, reason: str = "") -> dict:
            """Deletes a wiki page. A page title must contain the namespace (e.g. 'Item:'). Returns {'ok': true} or raises with the reason the wiki refused. Ask the user before deleting anything."""

            return await session.call_client_side_tool("delete_page", [title, reason])

        @langchain_core.tools.tool
        async def load_skill(name: str) -> str:
            """Returns detailed instructions for a task. Call this before starting any task that matches one of the skills listed in your instructions, and follow what it returns."""

            skill = session.skills.get(name)
            if skill is None:
                available = ", ".join(session.skills) or "none"
                raise ClientToolError(f"no skill named {name!r}. Available: {available}")
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
        ]
        if session.skills:
            tools.append(load_skill)
        for tool in tools:
            tool.handle_tool_error = _tool_error_observation
        return tools

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
        self.chat_history.clear()
