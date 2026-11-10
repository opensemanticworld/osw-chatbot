"""Serialize the chat history so it can be parked in the browser.

Tool observations are not stored verbatim: `where_am_i` alone ships up to 100 kB of page
HTML and `get_file_content` base64 data urls, none of which belong in a 5 MB localStorage
budget shared with MediaWiki itself. Each tool call is kept as a one-line record of what
ran and what came back, folded into the AIMessage that made it.

Two things that record has to get right. The Anthropic Messages API requires every
`tool_use` to be answered by a `tool_result`, so no `tool_calls` may survive into a
replayed turn. And the model needs evidence of its own actions: without it the only trace
of an edit is the assistant's claim to have made one, which it will believe and act on.
"""

import json
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from osw_chatbot.config import (
    HISTORY_MAX_BYTES,
    HISTORY_MAX_CHARS_PER_MESSAGE,
    HISTORY_MAX_MESSAGES,
)

SCHEMA_VERSION = 1

_PERSISTED_TYPES = (HumanMessage, AIMessage)

#: How much of one tool call's record survives. Enough to identify the call and its
#: outcome, far short of the payload that made these worth dropping in the first place.
_TOOL_RECORD_MAX_CHARS = 200

#: Prefix for the synthetic record, so it reads as transcript rather than as the
#: assistant's prose, and so a future reader can find and strip it.
_TOOL_RECORD_PREFIX = "[tools used]"


def _as_text(content):
    """Content may be a string or a list of blocks; this flattens to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return "" if content is None else str(content)


def _clip(value, limit=_TOOL_RECORD_MAX_CHARS):
    text = _as_text(value) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _tool_record(call, result):
    """One line describing a tool call and what it returned."""
    name = call.get("name") or "tool"
    args = call.get("args") or {}
    # Three arguments is enough to tell two calls of the same tool apart - which page,
    # which slot - without dragging the payload back in.
    shown = ", ".join(f"{k}={_clip(v, 60)!r}" for k, v in list(args.items())[:3])
    outcome = _clip(result) if result is not None else "no result recorded"
    return f"- {name}({shown}) -> {outcome}"


def for_persistence(messages):
    """Human/AI turns, with tool calls folded into the message that made them.

    Returns messages safe to replay: no `tool_calls` survive, so nothing dangles, and the
    evidence of what ran is carried as text on the AIMessage that ran it.
    """
    results = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            results[getattr(message, "tool_call_id", None)] = message.content

    out = []
    for message in messages:
        if not isinstance(message, _PERSISTED_TYPES):
            continue
        calls = getattr(message, "tool_calls", None)
        if isinstance(message, AIMessage) and calls:
            records = [_tool_record(c, results.get(c.get("id"))) for c in calls]
            body = _as_text(message.content).strip()
            joined = "\n".join([_TOOL_RECORD_PREFIX, *records])
            message = AIMessage(content=f"{body}\n{joined}".strip())
        out.append(message)
    return out


def trim_history(messages, max_messages=None, max_chars=None):
    """Bound the history in both message count and per-message size."""
    max_messages = HISTORY_MAX_MESSAGES if max_messages is None else max_messages
    max_chars = HISTORY_MAX_CHARS_PER_MESSAGE if max_chars is None else max_chars

    trimmed = []
    for message in messages[-max_messages:]:
        content = message.content
        if isinstance(content, str) and len(content) > max_chars:
            # model_copy, not type(message)(content=...): reconstructing would keep
            # only the content and silently drop every other field.
            message = message.model_copy(
                update={"content": content[:max_chars] + " [truncated]"}
            )
        trimmed.append(message)
    return trimmed


def dumps(messages):
    """Return a JSON string for `messages`, dropping oldest turns until it fits."""
    persistable = for_persistence(messages)
    payload = {
        "v": SCHEMA_VERSION,
        "updated": int(time.time()),
        "messages": messages_to_dict(trim_history(persistable)),
    }
    blob = json.dumps(payload)
    while len(blob) > HISTORY_MAX_BYTES and payload["messages"]:
        payload["messages"] = payload["messages"][2:]  # drop the oldest turn
        blob = json.dumps(payload)
    return blob


def loads(raw):
    """Parse what `dumps` produced. Never raises - a bad blob means no history."""
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        if payload.get("v") != SCHEMA_VERSION:
            return []
        messages = messages_from_dict(payload["messages"])
    except Exception as e:  # noqa: BLE001 - corrupt storage must not break the app
        print(f"osw-chatbot: discarding unreadable history ({e})")
        return []
    return [m for m in messages if isinstance(m, _PERSISTED_TYPES)]
