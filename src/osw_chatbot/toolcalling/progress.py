"""Show what the assistant is doing while it does it.

A turn that searches, reads three pages and writes one looks exactly like a
turn that does nothing: a spinner. This maps LangChain's tool callbacks onto
Panel chat steps, one compact line per call.

The step *title* carries a summary rather than the call itself: `where_am_i`
carries the whole DOM and `get_page_content` returns kilobytes, none of which
belongs in a one-line header. Expanding a step shows the summary, then the
arguments and the result as JSON, clipped - enough to see what was actually
asked and answered without the transcript becoming the payload.
"""

import json

from langchain_core.callbacks import AsyncCallbackHandler

from osw_chatbot.config import OPAQUE_MIME_PREFIXES

MAX_ARG = 60

# tool name -> (pending, done). {arg} is the summarised key argument.
LABELS = {
    # The jupyterlite surface. Without an entry a step renders as a bare
    # "Running notebook_add_cell", which tells the user nothing about what is happening
    # to their sandbox.
    "kernel_start": ("Starting a Python kernel", "Started a Python kernel"),
    "kernel_list": ("Looking for running kernels", "Looked for running kernels"),
    "kernel_execute": ("Running Python in the sandbox", "Ran Python in the sandbox"),
    "kernel_shutdown": ("Shutting the kernel down", "Shut the kernel down"),
    "notebook_info": ("Looking at the open notebook", "Looked at the open notebook"),
    "notebook_add_cell": ("Adding a cell to the notebook", "Added a cell to the notebook"),
    "notebook_run_cell": ("Running a notebook cell", "Ran a notebook cell"),
    "notebook_delete_cell": ("Deleting a notebook cell", "Deleted a notebook cell"),
    "notebook_save": ("Saving the notebook", "Saved the notebook"),
    "save_notebook_to_wiki": ("Uploading the notebook to the wiki",
                             "Uploaded the notebook to the wiki"),
    "where_am_i": ("Looking at the current page", "Looked at the current page"),
    "full_text_search": ("Searching the wiki for {arg}", "Searched the wiki for {arg}"),
    "find_page_from_topic": ("Looking up the category for {arg}", "Looked up {arg}"),
    "get_page_content": ("Reading {arg}", "Read {arg}"),
    "get_category_schema": ("Reading the schema of {arg}", "Read the schema of {arg}"),
    "smw_ask_query": ("Querying the wiki", "Queried the wiki"),
    "new_osw_id": ("Reserving an ID", "Reserved an ID"),
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
    "discover_skills": ("Looking for skills on this wiki", "Looked for skills"),
    "save_skill": ("Saving the {arg} skill", "Saved the {arg} skill"),
    "open_sandbox": ("Opening the sandbox", "Opened the sandbox"),
    "create_notebook": ("Writing the notebook {arg}", "Wrote the notebook {arg}"),
    # No {arg}: the argument is a block of code, and a clipped first line of it reads as
    # noise in a step title.
    "run_in_sandbox": ("Running code in the sandbox", "Ran code in the sandbox"),
    "sandbox_status": ("Checking the sandbox", "Checked the sandbox"),
}


def summarize(raw) -> str:
    """One short, readable fragment from whatever the tool was called with."""
    if isinstance(raw, dict):
        for key in ("title", "titles", "query", "page", "name", "topic",
                    "category_page", "file_title", "x_path", "url", "notebook"):
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

#: Per block, in the expanded body. Room for a real call without the step becoming a
#: wall of text in a ~210px panel - and `where_am_i` would otherwise paste a whole DOM.
MAX_JSON = 700


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


def _cell_output(output) -> str:
    """What a cell actually produced: printed text, a result, or the error.

    The most useful thing a step can show and the one that was missing - expanding
    "Ran a notebook cell" said "done", when the user wanted the number it printed or
    the exception it raised.
    """
    if not isinstance(output, dict):
        return ""
    if output.get("error"):
        return str(output["error"])

    parts, opaque = [], []
    for item in output.get("outputs") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("output_type")
        if kind == "error":
            name = str(item.get("ename") or "Error").strip("<>'\" class")
            return f"{name}: {item.get('evalue') or ''}".strip()
        if kind == "stream":
            parts.append(str(item.get("text") or ""))
        else:
            data = item.get("data") or {}
            opaque += [
                str(k) for k in data if str(k).startswith(OPAQUE_MIME_PREFIXES)
            ]
            if data.get("text/plain"):
                parts.append(str(data["text/plain"]))

    text = " ".join(" ".join(parts).split())
    if opaque:
        # "figure" only when it really is one: a cell can emit a pdf or audio, and
        # calling that a figure is the kind of small lie that costs trust in the rest.
        if all(m.startswith("image/") for m in opaque):
            made = f"{len(opaque)} figure{'s' if len(opaque) > 1 else ''} drawn"
        else:
            made = ", ".join(sorted(set(opaque))) + " produced"
        return f"{made} - {text}" if text else made
    if text:
        return text
    if output.get("success") is False or output.get("status") == "error":
        return str(output.get("message") or "failed")
    return "no output"


def _file_result(output) -> str:
    """A wiki file, named the way a reader sees it rather than by its OSW-ID alone."""
    if not isinstance(output, dict):
        return ""
    title = output.get("title") or ""
    label = output.get("label") or ""
    size = output.get("size")
    bits = [named(title, label)] if title else []
    if output.get("path"):
        bits.append(f"as {output['path']}")
    if isinstance(size, int):
        bits.append(f"{size // 1024} KB" if size >= 1024 else f"{size} bytes")
    return " ".join(bits)


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
    # "None" is what a per-tool case produces when the field it reads is absent - a
    # shape change upstream, not something to show a user.
    if not detail or str(detail).strip() in {"None", "none"}:
        detail = "done"
    return detail[: MAX_DETAIL - 1] + "…" if len(detail) > MAX_DETAIL else detail


def _describe(name, output):
    # the write tools report refusals as data; the lab bridge uses a bare `error`
    if isinstance(output, dict) and (output.get("ok") is False or "error" in output):
        return str(
            output.get("message") or output.get("error") or output.get("code") or "failed"
        )

    # Anything that runs code: show what it produced.
    if name in {"notebook_run_cell", "kernel_execute"}:
        return _cell_output(output)

    if name in {"save_notebook_to_wiki", "push_file_to_osw", "upload_file"}:
        return _file_result(output) or "uploaded"

    if name == "pull_file_from_osw":
        out = output if isinstance(output, dict) else {}
        return _file_result(out) or "copied into the sandbox"

    if name == "edit_page":
        out = output if isinstance(output, dict) else {}
        title = named(out.get("title") or "", out.get("label"))
        what = "created" if out.get("created") else "updated"
        slots = ", ".join(out.get("slots") or [])
        return f"{what} {title}" + (f" - {slots}" if slots else "")

    if name == "delete_page":
        out = output if isinstance(output, dict) else {}
        return f"deleted {short_id(out.get('title') or '')}".strip()

    if name == "notebook_info":
        out = output if isinstance(output, dict) else {}
        cells = out.get("cellCount")
        path = out.get("notebookPath") or out.get("notebookName") or ""
        return f"{path} - {cells} cells" if cells is not None else str(path or "")

    if name == "notebook_add_cell":
        out = output if isinstance(output, dict) else {}
        return str(out.get("message") or out.get("cellId") or "added")

    if name in {"kernel_list", "kernel_start"}:
        out = output if isinstance(output, dict) else {}
        kernels = out.get("kernels") or []
        if kernels:
            # A kernel id is a bare uuid - short_id only trims OSW-prefixed titles, so
            # the whole 36 characters would land in a 210px step line.
            return _clip([str(k.get("id", ""))[:8] + "…" for k in kernels])
        return str(out.get("kernelId") or out.get("message") or "no kernel")

    if name == "discover_commands":
        out = output if isinstance(output, dict) else {}
        return f"{out.get('count', 0)} things the sandbox can do"

    if name == "execute_command":
        out = output if isinstance(output, dict) else {}
        return _cell_output(out) if out.get("outputs") else str(
            out.get("message") or "done"
        )

    if name == "new_osw_id":
        return short_id(str(output or ""))

    if name == "discover_skills":
        if not output:
            return "this wiki has no skills of its own"
        names = [s.get("name", "") for s in output if isinstance(s, dict)]
        return f"{len(names)} skill{'s' if len(names) != 1 else ''}: " + _clip(names)

    if name == "save_skill":
        return str((output or {}).get("title") or "saved")

    if name == "sandbox_status":
        if not isinstance(output, dict):
            return ""
        if not output.get("open"):
            return "not open"
        return "ready" if output.get("ready") else "still starting"

    if name == "run_in_sandbox":
        if not isinstance(output, dict):
            return ""
        if output.get("error"):
            return str(output["error"])
        # The cell's own output, which is what the user wants to see in the step - not
        # the shape of the envelope it arrived in.
        return " ".join(str(output.get("output") or "").split()) or "no output"

    if name == "create_notebook":
        return str((output or {}).get("path") or (output or {}).get("name") or "created")

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

    if name == "full_text_search" and isinstance(output, dict):
        # counted <li> back when this returned scraped Special:Search html
        found = output.get("results") or []
        total = output.get("totalhits", len(found))
        if not found:
            return "no results"
        shown = f"{len(found)} result{'s' if len(found) != 1 else ''}"
        return f"{shown} of {total}" if total > len(found) else shown

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


def as_json(value) -> str:
    """A tool's arguments or result, readable and clipped.

    `default=str` because a result can hold anything a tool chose to return, and a step
    that raised TypeError while rendering would lose the call it was describing.
    """
    if isinstance(value, str):
        text = value.strip()
        # Tools are called with a JSON string as often as with a dict; showing it parsed
        # is the difference between a readable block and one long line.
        try:
            text = json.dumps(json.loads(text), indent=1, ensure_ascii=False)
        except ValueError:
            pass
    else:
        try:
            text = json.dumps(value, indent=1, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = text.strip()
    return text[: MAX_JSON - 1] + "…" if len(text) > MAX_JSON else text


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
        # the name is only known here, and the end needs both it and the arguments
        self._steps[run_id] = (step, name, kwargs.get("inputs") or input_str)

    async def on_tool_end(self, output, *, run_id=None, **kwargs):
        step, name, args = self._steps.pop(run_id, (None, "", None))
        if step is None:
            return
        # A refused write comes back as an observation rather than an error,
        # so look for it here too - otherwise a failure renders as success.
        #
        # Three shapes mean failure, because three layers report it differently: the
        # write tools return `ok: False`, the lab bridge returns a bare `{error: ...}`,
        # and a refusal arrives as text. Checking only the first two left a failed
        # upload wearing a green tick while the model was correctly telling the user it
        # had failed.
        text = str(output)
        failed = (
            text.startswith("Error:")
            or "do not have permission" in text
            or (
                isinstance(output, dict)
                and (output.get("ok") is False or "error" in output)
            )
        )
        self._fin(
            step,
            "failed" if failed else "success",
            describe(name, output),
            args,
            output,
        )

    async def on_tool_error(self, error, *, run_id=None, **kwargs):
        step, _, args = self._steps.pop(run_id, (None, "", None))
        if step is not None:
            self._fin(step, "failed", str(error)[:MAX_DETAIL], args, None)

    @staticmethod
    def _fin(step, status, detail="", args=None, output=None):
        try:
            body = [detail] if detail else []
            # Fenced, so page content that happens to contain markup is shown rather
            # than rendered - a result is data here, not something to interpret.
            if args not in (None, "", {}):
                body.append(f"```json\nargs: {as_json(args)}\n```")
            if output is not None:
                body.append(f"```json\nresult: {as_json(output)}\n```")
            if body:
                step.stream("\n\n".join(body))
            step.status = status
        except Exception:  # noqa: BLE001
            pass
