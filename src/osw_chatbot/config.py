"""Central place for reading the environment.

Imported by the app modules so that env parsing (and its defaults) live in one
file instead of being scattered over ``os.environ[...]`` lookups.
"""

import os


def _flag(name, default="false"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _origins(name, default="*"):
    """Parse a comma separated list of origins, e.g. 'https://a.org,https://b.org'.

    Returns ``["*"]`` to mean "accept any parent origin".
    """
    value = os.getenv(name, default).strip()
    if value in ("", "*"):
        return ["*"]
    return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]


## Panel <-> MediaWiki bridge

# Origins of the wiki(s) that are allowed to embed the chatbot iframe and to
# exchange postMessages with it. This is NOT the same as BOKEH_ALLOW_WS_ORIGIN:
# that one is the *chatbot* host in host[:port] form used for the websocket
# check, while PARENT_ORIGIN is the scheme-qualified *wiki* origin exactly as
# `event.origin` reports it in the browser.
PARENT_ORIGINS = _origins("PARENT_ORIGIN")

# Seconds to wait for the users browser to answer a client-side tool call.
CLIENT_TOOL_TIMEOUT = float(os.getenv("CLIENT_TOOL_TIMEOUT", "30"))

## Chat history persistence

HISTORY_ENABLED = _flag("HISTORY_ENABLED", "true")
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "40"))
HISTORY_MAX_BYTES = int(os.getenv("HISTORY_MAX_BYTES", "100000"))
# Truncate a single message before storing / replaying it.
HISTORY_MAX_CHARS_PER_MESSAGE = int(os.getenv("HISTORY_MAX_CHARS_PER_MESSAGE", "8000"))

## Authentication

# Shared with the MediaWiki Chatbot extension ($wgChatbotSecret). When empty,
# token validation is disabled and the app stays open - which is what keeps a
# new backend working against an older, token-less version of the extension.
CHATBOT_SHARED_SECRET = os.getenv("CHATBOT_SHARED_SECRET", "").strip()
# Reject tokens older than this many seconds even if they carry a later `exp`.
TOKEN_MAX_AGE = int(os.getenv("CHATBOT_TOKEN_MAX_AGE", "86400"))

## LLM provider

# "openai"  -> OpenAI-compatible surface of Microsoft Foundry (/openai/v1)
# "claude"  -> Anthropic Messages surface of Microsoft Foundry (/anthropic)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
