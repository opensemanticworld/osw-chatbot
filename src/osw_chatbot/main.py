
from pathlib import Path
from dotenv import load_dotenv
import panel as pn

# going to do load_dotenv() here
# as OLLAMA_HOST needs to be in the environment
# before the imports below
env_path = Path(__file__).parent.parent.parent / ".env"
env_loaded = load_dotenv(env_path, verbose=True)
if not env_loaded:
    print(f"No .env file found at {env_path}, using environment variables.")

from langchain_core.messages import HumanMessage

from osw_chatbot.auth import verify_token
from osw_chatbot.chat import history as hist
from osw_chatbot.chat.client_store import ClientStoreWidget
from osw_chatbot.config import HISTORY_ENABLED, PARENT_ORIGINS
from osw_chatbot.toolcalling.agent import ChatSession

pn.extension()

pn.config.theme = 'dark'

GREETING = "what's on your mind?"

# Embedded in the wiki the chat is only ~210px wide, and the avatar column eats
# 60px of that - enough to wrap the text to one character per line. Panel 1.7.4
# accepts ChatMessage's `show_avatar=False` but still renders the avatar, so it
# has to go via CSS. The stylesheet is passed through `message_params` so it
# lands inside each message's shadow root.
# Identity is carried by the user name instead (as panelini does it).
USER_NAME = "🧑 User"
ASSISTANT_NAME = "🤖 Assistant"

COMPACT_MESSAGE_CSS = """
.avatar, .left { display: none !important; }
/* :host is the message row itself, which otherwise keeps a 10px margin on
   each side; .center is the bubble, which does not fill its column. Together
   they left ~30px unused on the right against 15px on the left. */
:host { margin-left: 0 !important; margin-right: 0 !important; }
.right, .center { width: 100% !important; max-width: 100% !important; }
/* the bubble sizes to its content, so short lines still wrapped while ~60px
   sat unused next to them */
.message { width: 100% !important; max-width: 100% !important; }
"""


def current_user():
    """Verified wiki username from the ?token= query arg, or None.

    The auth module has already rejected invalid tokens by the time the app is
    built; this only re-reads the name. Never use the `user` field returned by
    the where_am_i tool for this - that one is asserted by the client.
    """
    try:
        args = pn.state.session_args or {}
        raw = args.get("token")
    except Exception:  # noqa: BLE001 - no session (e.g. imported for tests)
        return None
    if not raw:
        return None
    token = raw[0].decode() if isinstance(raw[0], bytes) else raw[0]
    return verify_token(token)


def build_app():
    session = ChatSession(allowed_origins=PARENT_ORIGINS, user=current_user())
    store = ClientStoreWidget(
        store_key="history",
        allowed_origins=PARENT_ORIGINS,
        visible=False,
        height=0,
        margin=0,
    )

    async def get_response(contents, user, instance):
        print(contents)
        response = await session.invoke(contents)
        print(response)
        persist()
        return response["output"]

    chat_bot = pn.chat.ChatInterface(
        callback=get_response,
        user=USER_NAME,
        callback_user=ASSISTANT_NAME,
        show_send=True,
        show_rerun=False,
        show_undo=False,
        # clears the widget, this session's memory and the stored history
        show_clear=True,
        show_button_name=False,
        sizing_mode="stretch_width",
        callback_exception="verbose",
        # show_avatar / show_timestamp / show_reaction_icons are ChatMessage
        # params, not ChatInterface ones - passed directly they are silently
        # swallowed. They belong in message_params.
        message_params={
            "show_avatar": False,
            "show_timestamp": False,
            "show_reaction_icons": False,
            "stylesheets": [COMPACT_MESSAGE_CSS],
        },
    )

    def persist():
        if HISTORY_ENABLED:
            store.write = hist.dumps(session.chat_history) if session.chat_history else ""

    def on_messages_changed(event):
        # The clear button empties the widget - drop the model's memory and the
        # stored copy with it, otherwise "clear" only hides the conversation.
        if len(event.new) == 0:
            session.clear()
            store.write = ""

    restored = False

    def restore(*_):
        nonlocal restored
        if restored:
            return
        restored = True

        messages = hist.loads(store.value) if HISTORY_ENABLED else []
        # Restore the agent's memory too, not just what is on screen.
        session.chat_history = list(messages)
        for message in messages:
            chat_bot.send(
                message.content,
                user=USER_NAME if isinstance(message, HumanMessage) else ASSISTANT_NAME,
                respond=False,
            )
        if not messages:
            chat_bot.send(GREETING, user=ASSISTANT_NAME, respond=False)
        # Attach after the replay so it is not mistaken for a user edit.
        chat_bot.param.watch(on_messages_changed, "objects")

    # The widget tells us when it has resolved storage - including the blocked
    # and empty cases. The periodic callback is only a safety net for when the
    # widget never renders at all.
    store.param.watch(restore, "ready")
    try:
        pn.state.add_periodic_callback(restore, period=2000, count=1)
    except Exception as e:  # noqa: BLE001 - no server session (pn.serve dev path)
        print(f"osw-chatbot: no periodic callback available ({e})")

    try:
        pn.state.on_session_destroyed(lambda ctx: session.close())
    except Exception as e:  # noqa: BLE001
        print(f"osw-chatbot: no session teardown hook available ({e})")

    return pn.Column(chat_bot, session.frontend, store)


if __name__ == "__main__":
    pn.serve(build_app, port=52670)

else:
    # Run with `panel serve main.py`
    build_app().servable()
