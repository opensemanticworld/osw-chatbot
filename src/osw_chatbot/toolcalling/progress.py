"""Show what the assistant is doing while it does it.

A turn that searches, reads three pages and writes one looks exactly like a
turn that does nothing: a spinner. This maps LangChain's tool callbacks onto
Panel chat steps, one compact line per call.

Arguments are deliberately summarised rather than shown: `where_am_i` carries
the whole DOM and `get_page_content` returns kilobytes, none of which belongs
in the transcript.
"""

import json

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
/* .step-avatar-container holds the pending/running/success/failed badge -
   hiding it also hides whether the call worked. Keep it, just small. */
.step-avatar-container {
  min-width: 0 !important; width: auto !important; margin: 0 4px 0 0 !important;
}
.step-avatar-container img, .step-avatar-container svg { width: 13px; height: 13px; }
.step-header {
  padding: 1px 0 !important; min-height: 0 !important; gap: 4px !important;
  align-items: center !important;
}
/* bokeh sizes the title in fixed pixels, so once the badge column takes its
   width back the title collapses to nothing - let it fill the row instead */
.step-title {
  flex: 1 1 auto !important; width: auto !important; min-width: 0 !important;
}
.step-title, .step-title p {
  color: #9aa0a6 !important; font-size: 12px !important;
  margin: 0 !important; line-height: 1.5 !important;
}
"""


MAX_DETAIL = 180


def _clip(items, limit=4):
    items = [str(i) for i in items if i]
    shown = ", ".join(items[:limit])
    rest = len(items) - limit
    return shown + (f" +{rest} more" if rest > 0 else "")


def short_id(title):
    """'Item:OSW0024d7c0a0d6...' - an OSW-ID in full is 32 hex digits of noise."""
    title = str(title or "")
    head, sep, rest = title.rpartition("OSW")
    if sep and len(rest) > 10:
        return f"{head}OSW{rest[:8]}…"
    return title


def named(title, label=None):
    """`label (OSW123…)` when the label is known, plain id when it is not."""
    short = short_id(title)
    label = str(label or "").strip()
    return f"{label} ({short})" if label and label != title else short


def _json_label(page):
    """The entity label out of a page's jsondata slot, if it has one."""
    revisions = (page or {}).get("revisions") or [{}]
    slot = ((revisions[0] or {}).get("slots") or {}).get("jsondata") or {}
    raw = slot.get("*")
    if not raw:
        return None
    data = json.loads(raw) if isinstance(raw, str) else raw
    labels = (data or {}).get("label") or []
    return labels[0].get("text") if labels and isinstance(labels[0], dict) else None


def _page_titles(data):
    pages = ((data.get("query") or {}).get("pages") or {}) if isinstance(data, dict) else {}
    return [p.get("title") for p in pages.values() if isinstance(p, dict)]


def _schema_properties(schema, seen=None):
    """Property names of a category schema, which hides them under allOf."""
    if not isinstance(schema, dict) or (seen is not None and id(schema) in seen):
        return []
    seen = seen or set()
    seen.add(id(schema))
    names = list(schema.get("properties") or {})
    for part in schema.get("allOf") or []:
        names += _schema_properties(part, seen)
    return names


def describe(name, output) -> str:
    """What a tool returned, in one useful line.

    Per tool, because the generic answer is worthless: the interesting part of
    a page query is which pages came back, and of a schema which properties it
    has - not that the payload was 10451 characters long.
    """
    try:
        detail = _describe(name, output)
    except Exception:  # noqa: BLE001 - a summary must never break a turn
        detail = ""
    if not detail:
        detail = "done"
    return detail[: MAX_DETAIL - 1] + "…" if len(detail) > MAX_DETAIL else detail


def _describe(name, output):
    # the write tools report refusals as data
    if isinstance(output, dict) and output.get("ok") is False:
        return str(output.get("message") or output.get("code") or "failed")

    if name == "where_am_i":
        title = getattr(output, "title", None) or (output or {}).get("title")
        user = getattr(output, "user", None) or (output or {}).get("user") or {}
        who = (user or {}).get("name") or (user or {}).get("id")
        return f"{title}" + (f", as {who}" if who else "")

    if name == "get_page_content":
        pages = list(((output.get("query") or {}).get("pages") or {}).values())
        if not pages:
            return "no page"
        shown, slots = [], []
        for page in pages:
            shown.append(named(page.get("title"), _json_label(page)))
            if not slots:
                revisions = page.get("revisions") or [{}]
                slots = list((revisions[0] or {}).get("slots") or {})
        return _clip(shown, 3) + (f" - slots: {_clip(slots, 6)}" if slots else "")

    if name == "get_category_schema":
        props = _schema_properties(output)
        required = (output or {}).get("required") or []
        out = f"{len(props)} properties: {_clip(props, 6)}"
        return out + (f" - required: {_clip(required, 4)}" if required else "")

    if name == "smw_ask_query":
        results = ((output or {}).get("query") or {}).get("results") or {}
        shown = [
            named(title, (entry or {}).get("displaytitle"))
            for title, entry in list(results.items())[:4]
        ]
        return f"{len(results)} result{'s' if len(results) != 1 else ''}: " + _clip(shown)

    if name == "find_page_from_topic" and isinstance(output, list):
        shown = [
            named(r.get("title"), r.get("label") or r.get("displaytitle"))
            for r in output if isinstance(r, dict)
        ]
        return f"{len(output)} match{'es' if len(output) != 1 else ''}: " + _clip(shown)

    if name in ("edit_page", "delete_page", "upload_file") and isinstance(output, dict):
        # upload_file hands back the original filename as the label
        shown = named(output.get("title") or "", output.get("name"))
        extra = output.get("slots") or []
        return shown + (f" - slots: {_clip(extra, 6)}" if extra else "")

    if name == "full_text_search" and isinstance(output, str):
        hits = output.count("<li")
        return f"{hits} result{'s' if hits != 1 else ''}" if hits else "no results"

    if name == "load_skill" and isinstance(output, str):
        return f"{len(output)} characters of instructions"

    # web search returns a pydantic result with a list of hits
    results = getattr(output, "results", None)
    if isinstance(results, list):
        return f"{len(results)} result{'s' if len(results) != 1 else ''}: " + _clip(
            [getattr(r, "title", None) or getattr(r, "url", "") for r in results]
        )

    if isinstance(output, (list, tuple)):
        return f"{len(output)} item{'s' if len(output) != 1 else ''}"
    if isinstance(output, dict):
        return _clip(list(output), 6)

    text = " ".join(str(output).replace("<", "‹").split())
    return text


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
                # One per status: the title switches with `status`, and a
                # status whose title is unset renders an empty header.
                default_title=pending, pending_title=pending,
                running_title=pending, success_title=done,
                failed_title=done + " - failed",
                # Collapsed by default so the answer stays the thing you read,
                # but expandable - the body carries what the call returned.
                collapsed=True, collapsible=True,
                # default_layout="card" wraps the steps in a Card titled
                # "Steps", a full-width header that dwarfs the lines inside it;
                # an empty user drops the duplicate "Assistant" label, since
                # the answer right below carries one already.
                default_layout="column", user="",
                stylesheets=[STEP_CSS],
                margin=0,
            )
            step.status = "running"
        except Exception as e:  # noqa: BLE001 - progress must never break a turn
            print(f"osw-chatbot: could not add a progress step ({e})")
            return
        # the name is only known here, but the summary needs it at the end
        self._steps[run_id] = (step, name)

    async def on_tool_end(self, output, *, run_id=None, **kwargs):
        step, name = self._steps.pop(run_id, (None, ""))
        if step is None:
            return
        # A refused write comes back as an observation rather than an error,
        # so look for it here too - otherwise a failure renders as success.
        text = str(output)
        failed = text.startswith("Error:") or "do not have permission" in text
        self._fin(step, "failed" if failed else "success", describe(name, output))

    async def on_tool_error(self, error, *, run_id=None, **kwargs):
        step, _ = self._steps.pop(run_id, (None, ""))
        if step is not None:
            self._fin(step, "failed", str(error)[:MAX_DETAIL])

    @staticmethod
    def _fin(step, status, detail=""):
        try:
            if detail:
                step.stream(detail)
            step.status = status
        except Exception:  # noqa: BLE001
            pass
