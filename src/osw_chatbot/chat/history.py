"""Serialize the chat history so it can be parked in the browser.

Only Human/AI turns are stored, never tool observations: `where_am_i` alone
ships up to 100 kB of page HTML and `get_file_content` base64 data urls, none
of which belong in a 5 MB localStorage budget shared with MediaWiki itself.
"""

import json
import time

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
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


def trim_history(messages, max_messages=None, max_chars=None):
    """Bound the history in both message count and per-message size."""
    max_messages = HISTORY_MAX_MESSAGES if max_messages is None else max_messages
    max_chars = HISTORY_MAX_CHARS_PER_MESSAGE if max_chars is None else max_chars

    trimmed = []
    for message in messages[-max_messages:]:
        content = message.content
        if isinstance(content, str) and len(content) > max_chars:
            message = type(message)(content=content[:max_chars] + " [truncated]")
        trimmed.append(message)
    return trimmed


def dumps(messages):
    """Return a JSON string for `messages`, dropping oldest turns until it fits."""
    persistable = [m for m in messages if isinstance(m, _PERSISTED_TYPES)]
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
