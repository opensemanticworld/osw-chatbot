"""Bokeh/Panel auth module verifying the token minted by the MediaWiki extension.

Wire it up with::

    panel serve ... --auth-module=src/osw_chatbot/auth.py \
                    --cookie-secret=$BOKEH_SECRET_KEY --session-ids signed

The extension calls ``action=chatbottoken`` and appends the result to the
iframe src as ``?token=...``. The token is::

    base64url(json{u, w, iat, exp, n}) + "." + base64url(hmac_sha256(payload, secret))

signed with the secret shared as ``$wgChatbotSecret`` (wiki) /
``CHATBOT_SHARED_SECRET`` (here). When that secret is empty the module lets
everybody in, so a new backend still works against an older, token-less
version of the extension.

Note the chatbot is a *third-party* iframe: the cookie set below is a
third-party cookie and Safari (and Firefox in strict mode) will drop it. That
is why the websocket upgrade is additionally accepted on the strength of a
server-signed bokeh session id, which can only exist if the document request
that created it passed the token check.
"""

import base64
import hashlib
import hmac
import json
import time

from osw_chatbot.config import CHATBOT_SHARED_SECRET, TOKEN_MAX_AGE

COOKIE_NAME = "osw_chatbot_user"
login_url = "/login"


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign(payload_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode(), payload_b64.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode().rstrip("=")


def verify_token(token, secret=None, now=None):
    """Return the username carried by `token`, or None if it is not valid."""
    secret = CHATBOT_SHARED_SECRET if secret is None else secret
    if not secret or not token:
        return None
    try:
        payload_b64, signature = token.split(".", 1)
    except ValueError:
        return None

    if not hmac.compare_digest(sign(payload_b64, secret), signature):
        return None

    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:  # noqa: BLE001 - any malformed payload is simply invalid
        return None

    now = time.time() if now is None else now
    if float(payload.get("exp", 0)) < now:
        return None
    # Independent upper bound, so a wiki misconfigured with a far-future `exp`
    # cannot mint effectively eternal tokens.
    issued = float(payload.get("iat", 0))
    if issued and now - issued > TOKEN_MAX_AGE:
        return None

    user = payload.get("u")
    return user if isinstance(user, str) and user else None


def _session_id_is_signed(handler) -> bool:
    """True if the request carries a bokeh session id this server signed."""
    session_id = handler.get_argument("bokeh-session-id", None)
    if not session_id:
        return False
    try:
        from bokeh.util.token import check_session_id_signature

        secret = getattr(handler.application, "secret_key", None) or getattr(
            handler.application.settings.get("bokeh_server", None), "secret_key", None
        )
        return bool(check_session_id_signature(session_id, signed=True, secret_key=secret))
    except Exception:  # noqa: BLE001 - bokeh internals differ between versions
        return False


def _remember(handler, user):
    try:
        secure = handler.request.protocol == "https"
        handler.set_secure_cookie(
            COOKIE_NAME,
            user,
            expires_days=1,
            httponly=True,
            # third-party context: the cookie is only ever sent from an iframe
            samesite="None" if secure else "Lax",
            secure=secure,
        )
    except Exception as e:  # noqa: BLE001 - websocket handlers cannot set cookies
        print(f"osw-chatbot: could not persist auth cookie ({e})")


def get_user(handler):
    if not CHATBOT_SHARED_SECRET:
        return "anonymous"

    token = handler.get_argument("token", None)
    if token:
        user = verify_token(token)
        if user:
            _remember(handler, user)
            return user
        print("osw-chatbot: rejected an invalid or expired token")
        return None

    try:
        cookie = handler.get_secure_cookie(COOKIE_NAME, max_age_days=1)
    except Exception:  # noqa: BLE001
        cookie = None
    if cookie:
        return cookie.decode("utf-8", "replace")

    if _session_id_is_signed(handler):
        return "session"

    return None


try:
    from tornado.web import RequestHandler

    class LoginHandler(RequestHandler):
        def get(self):
            self.set_status(403)
            self.write(
                "<html><body style='font-family:sans-serif;padding:2rem'>"
                "<h3>Not authorized</h3>"
                "<p>Please open the assistant from within the wiki.</p>"
                "</body></html>"
            )

    login_handler = LoginHandler
except ImportError:  # pragma: no cover - tornado ships with bokeh
    pass
