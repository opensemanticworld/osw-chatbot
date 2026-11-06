"""Central place for reading the environment.

Imported by the app modules so that env parsing (and its defaults) live in one
file instead of being scattered over ``os.environ[...]`` lookups.
"""

import json
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
CLIENT_TOOL_TIMEOUT = float(os.getenv("CLIENT_TOOL_TIMEOUT", "60"))

# Once the browser reports that it is showing a confirmation dialog, the clock
# is measuring how long the *user* takes to read it, not whether the browser is
# still alive. Timing out there told the model a write had failed while it was
# in fact about to succeed, so it retried and wrote twice.
CONFIRM_TIMEOUT = float(os.getenv("CONFIRM_TIMEOUT", "600"))

## Chat UI

# Shown once at the top of a new conversation. Configurable so a deployment
# can reword or translate it.
CHATBOT_INTRO = os.getenv(
    "CHATBOT_INTRO",
    "I am an AI assistant - my answers can be wrong or incomplete, so please "
    "check anything important. I act with your own wiki account, and I can "
    "create or change pages when you ask me to.",
).strip()

# The conversational opener, shown under the intro on a new chat. Empty
# hides it, leaving the intro alone.
CHATBOT_GREETING = os.getenv("CHATBOT_GREETING", "what's on your mind?").strip()

# Largest file the browser may inline as a base64 data url. Base64 adds a
# third, and bokeh drops a websocket message over its limit (20 MB by default)
# without telling anyone - the call simply never comes back. Keep this below
# three quarters of that; raise both together, with --websocket-max-message-size
# on the panel serve command, if a genuinely large image has to be read.
MAX_INLINE_FILE_BYTES = int(os.getenv("MAX_INLINE_FILE_BYTES", str(12 * 1024 * 1024)))

# Largest attachment the browser will stage, in bytes.
ATTACHMENT_MAX_BYTES = int(os.getenv("ATTACHMENT_MAX_BYTES", str(20 * 1024 * 1024)))

## Chat history persistence

HISTORY_ENABLED = _flag("HISTORY_ENABLED", "true")
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "40"))
HISTORY_MAX_BYTES = int(os.getenv("HISTORY_MAX_BYTES", "100000"))
# Truncate a single message before storing / replaying it.
HISTORY_MAX_CHARS_PER_MESSAGE = int(os.getenv("HISTORY_MAX_CHARS_PER_MESSAGE", "8000"))

# Largest contribution a single tool observation, or one attachment's extracted
# text, may make to the prompt.
#
# chat_history is trimmed on every turn by the three limits above, but the agent
# scratchpad is not trimmed at all: whatever a tool returns goes to the model
# whole, inside the same request. One oversized result therefore does not degrade
# the answer, it fails the turn - the vendor rejects the request outright with
# "prompt is too long: 2006895 tokens > 1000000 maximum", which surfaces as a
# traceback rather than as an answer. Attachments have the same hole: _content
# appends att['text'] unbounded, and ATTACHMENT_MAX_BYTES allows 20 MB.
#
# 120k characters is roughly 30k tokens, so several large observations still fit
# in a 1M window alongside the history and the system prompt. Raise it only
# together with the model's context limit.
MAX_TOOL_RESULT_CHARS = int(os.getenv("MAX_TOOL_RESULT_CHARS", "120000"))

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

# Largest prompt we will assemble, in characters, across system text + history + the
# current input. Roughly four characters per token, so ~150k tokens: well under the
# model's 1M, because the agent scratchpad grows inside the turn after this check and
# cannot be measured here.
#
# MAX_TOOL_RESULT_CHARS bounds one observation; this bounds the total. Without it a long
# session simply fails - "prompt is too long: 2006895 tokens > 1000000 maximum" is a 400
# from the vendor, i.e. no answer rather than a worse one.
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", str(600_000)))

## Skills

# Skills are ordinary wiki pages, found by configuration rather than by a category we
# invented. Everything here is a default a deployment can retarget.

#: Category a skill page must be an instance of. Article is deliberately broad - Tutorial
#: and LLM-Skill both sit under it - because the keyword below is what actually makes a
#: page a skill. `save_skill` refuses to write if this is missing from the target wiki.
SKILL_CATEGORY = os.getenv(
    "SKILL_CATEGORY", "Category:OSW92cc6b1a2e6b4bb7bad470dfdcfdaf26"
)

#: The "AI Skill" keyword, an instance of Category:Keyword.
AI_SKILL_KEYWORD = os.getenv(
    "AI_SKILL_KEYWORD", "Item:OSW1dd1c2ee927343ca9097a45a0af21678"
)

#: Merged into the jsondata of a saved skill, filling only what the assistant did not
#: supply - it may add further keywords or further part_of targets. Any OSW-page id in
#: here that does not exist on the target wiki is dropped before the merge, so a wiki
#: without the keyword still gets a valid write instead of a dangling reference.
#: `part_of` places skills under "OpenSemanticLab for developers", itself part of the
#: OpenSemanticLab Manual.
SKILL_DEFAULT_DATA = json.loads(
    os.getenv(
        "SKILL_DEFAULT_DATA",
        json.dumps(
            {
                "keywords": [AI_SKILL_KEYWORD],
                "part_of": ["Item:OSWe0cbdc9c1e0e4a53b21bd83e6c6d99ad"],
            }
        ),
    )
)

#: How skills are found. The sandbox agent has its own setting sharing this default, so a
#: deployment can narrow one without disturbing the other; keep the default here so the
#: two cannot drift.
SKILL_DISCOVERY_QUERY = os.getenv(
    "SKILL_DISCOVERY_QUERY", f"[[HasKeyword::{AI_SKILL_KEYWORD}]]"
)
