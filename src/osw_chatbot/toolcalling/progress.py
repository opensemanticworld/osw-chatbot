"""Show what the assistant is doing while it does it.

A turn that searches, reads three pages and writes one looks exactly like a
turn that does nothing: a spinner. This maps LangChain's tool callbacks onto
Panel chat steps, one compact line per call.

Arguments are deliberately summarised rather than shown: `where_am_i` carries
the whole DOM and `get_page_content` returns kilobytes, none of which belongs
in the transcript.
"""

from langchain_core.callbacks import AsyncCallbackHandler

MAX_ARG = 60

# tool name -> (pending, done). {arg} is the summarised key argument.
LABELS = {
    "where_am_i": ("Looking at the current page", "Looked at the current page"),
    "full_text_search": ("Searching the wiki for {arg}", "Searched the wiki for {arg}"),
    "find_page_from_topic": ("Looking up the category for {arg}", "Looked up {arg}"),
    "get_page_content": ("Reading {arg}", "Read {arg}"),
    "get_category_schema": ("Reading the schema of {arg}", "Read the schema of {arg}"),
    "smw_ask_query": ("Querying the wiki", "Queried the wiki"),
    "get_file_content": ("Reading the file {arg}", "Read the file {arg}"),
    "edit_page": ("Writing {arg}", "Wrote {arg}"),
    "delete_page": ("Deleting {arg}", "Deleted {arg}"),
    "upload_file": ("Uploading {arg}", "Uploaded {arg}"),
    "create_category_instance": ("Opening the editor for {arg}", "Opened the editor"),
    "redirect": ("Taking you to {arg}", "Opened {arg}"),
    "highlight_html_element": ("Highlighting an element", "Highlighted an element"),
    "web_search_tool": ("Searching the web for {arg}", "Searched the web for {arg}"),
    "simple_browse_web_page_tool": ("Reading {arg}", "Read {arg}"),
    "load_skill": ("Reading the {arg} instructions", "Read the {arg} instructions"),
}


def summarize(raw) -> str:
    """One short, readable fragment from whatever the tool was called with."""
    if isinstance(raw, dict):
        for key in ("title", "titles", "query", "page", "name", "topic",
                    "category_page", "file_title", "x_path", "url"):
            if raw.get(key):
                raw = raw[key]
                break
        else:
            raw = next(iter(raw.values()), "") if raw else ""
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else ""
    text = str(raw).strip().replace("\n", " ")
    if len(text) > MAX_ARG:
        text = text[: MAX_ARG - 1] + "…"
    return text


# Panel renders a step as a Card (.chat-steps) with a .step-header row, an
# avatar column and a .step-title. The default is a full card with a 10px
# inset - far too heavy for a status line in a ~210px popup. Flatten it to a
# muted single line. Sizes are absolute because the stylesheet reaches every
# nested component, so a relative font-size would compound.
STEP_CSS = """
.chat-steps {
  margin-inline: 0 !important; border: none !important;
  background: transparent !important; box-shadow: none !important;
  padding: 0 !important;
}
.step-avatar-container { display: none !important; }
.step-header {
  padding: 1px 0 !important; min-height: 0 !important; gap: 5px !important;
  align-items: center !important;
}
.step-title, .step-title p {
  color: #9aa0a6 !important; font-size: 12px !important;
  margin: 0 !important; line-height: 1.5 !important;
}
"""


class ToolProgressHandler(AsyncCallbackHandler):
    """Renders one collapsed chat step per tool call."""

    def __init__(self, instance):
        self.instance = instance   # the pn.chat.ChatInterface of this turn
        self._steps = {}

    def _titles(self, name, raw):
        pending, done = LABELS.get(
            name, ("Running " + name, "Ran " + name)
        )
        arg = summarize(raw)
        if not arg:
            pending, done = pending.replace(" {arg}", ""), done.replace(" {arg}", "")
        return pending.format(arg=arg), done.format(arg=arg)

    async def on_tool_start(self, serialized, input_str, *, run_id=None, **kwargs):
        name = (serialized or {}).get("name") or "tool"
        pending, done = self._titles(name, kwargs.get("inputs") or input_str)
        try:
            step = self.instance.add_step(
                pending_title=pending, success_title=done,
                title=pending, collapsed=True,
                # default_layout="card" wraps the steps in a Card titled
                # "Steps", a full-width header that dwarfs the lines inside it;
                # an empty user drops the duplicate "Assistant" label, since
                # the answer right below carries one already.
                default_layout="column", user="",
                stylesheets=[STEP_CSS],
                margin=0,
            )
        except Exception as e:  # noqa: BLE001 - progress must never break a turn
            print(f"osw-chatbot: could not add a progress step ({e})")
            return
        self._steps[run_id] = step

    async def on_tool_end(self, output, *, run_id=None, **kwargs):
        step = self._steps.pop(run_id, None)
        if step is None:
            return
        # A refused write comes back as an observation rather than an error,
        # so look for it here too - otherwise a failure renders as success.
        text = str(output)
        failed = text.startswith("Error:") or "do not have permission" in text
        self._fin(step, "failed" if failed else "success")

    async def on_tool_error(self, error, *, run_id=None, **kwargs):
        step = self._steps.pop(run_id, None)
        if step is not None:
            self._fin(step, "failed")

    @staticmethod
    def _fin(step, status):
        try:
            step.status = status
        except Exception:  # noqa: BLE001
            pass
