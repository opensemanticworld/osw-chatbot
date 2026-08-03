"""Serve the integration harness and mint tokens for it.

    uv run python src/osw_chatbot/tests/harness/serve.py [--port 8099]

Then open http://localhost:8099/ and point the "backend" field at the running
chatbot (http://localhost:52670/main for `panel serve`, or the public url).

If CHATBOT_SHARED_SECRET is set, /token?user=Alice returns a valid token to
paste into a pane - the same thing action=chatbottoken does on the wiki.
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).parent


def mint(user, secret, ttl=3600, wiki="harnesswiki"):
    now = int(time.time())
    payload = {"u": user, "w": wiki, "iat": now, "exp": now + ttl, "n": os.urandom(8).hex()}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    return encoded + "." + signature


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/token":
            secret = os.getenv("CHATBOT_SHARED_SECRET", "").strip()
            user = (parse_qs(parsed.query).get("user") or ["Alice"])[0]
            body = (
                json.dumps({"token": mint(user, secret)})
                if secret
                else json.dumps({"error": "CHATBOT_SHARED_SECRET is not set"})
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/":
            self.path = "/parent.html"
        return super().do_GET()

    def log_message(self, fmt, *args):  # quieter
        if "/token" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    args = ap.parse_args()

    env = Path(__file__).resolve().parents[4] / ".env"
    if env.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env)
        except ImportError:
            pass

    server = ThreadingHTTPServer(
        ("0.0.0.0", args.port), partial(Handler, directory=str(HERE))
    )
    print(f"harness on http://localhost:{args.port}/  (Ctrl-C to stop)")
    print(
        "token minting: "
        + ("enabled" if os.getenv("CHATBOT_SHARED_SECRET", "").strip() else "disabled (no secret)")
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
