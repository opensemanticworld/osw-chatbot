"""Browser-level integration test: drives the harness with Playwright.

This is the half that unit tests cannot reach - the postMessage handshake,
per-session routing of client-side tool calls, and history restore across a
reload. It loads the harness page (two iframes, two fake wiki users) and
asserts on what each pane actually received.

    # inside the running container, backend on :81, harness on :8099
    docker compose --profile test up -d osw-chatbot-harness
    docker compose exec osw-chatbot uv run python \
        src/osw_chatbot/tests/harness/integration_test.py \
        --harness http://osw-chatbot-harness:8099/ --backend http://localhost:81/main

Checks that need no LLM call run by default. Pass --with-llm to also send a
real prompt (costs tokens, takes ~30s).
"""

import argparse
import contextlib
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PANE_A, PANE_B = 0, 1

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))
    return ok


def pane_log(page, index):
    return page.locator(".pane").nth(index).locator(".log").inner_text()


def chat_frame(page, index):
    return page.locator(".pane").nth(index).frame_locator("iframe")


def send_prompt(page, index, text, timeout):
    """Type into the Panel ChatInterface inside a pane's iframe."""
    frame = chat_frame(page, index)
    box = frame.locator("input.bk-input, textarea.bk-input").first
    box.wait_for(state="visible", timeout=timeout)
    box.click()
    box.fill(text)
    box.press("Enter")


def wait_for_log(page, index, needle, timeout):
    page.wait_for_function(
        """([i, needle]) => document.querySelectorAll('.pane')[i]
               .querySelector('.log').textContent.includes(needle)""",
        arg=[index, needle],
        timeout=timeout,
    )


@contextlib.contextmanager
def backend_server(port, env_extra=None):
    """Start a throwaway `panel serve` that accepts our websocket origin.

    The deployed container pins BOKEH_ALLOW_WS_ORIGIN to the public hostname,
    so a browser loading the app from localhost gets a 403 on the websocket
    upgrade. Rather than loosening the running deployment, the test brings up
    its own server.
    """
    # prefer the panel next to the interpreter running us (the venv)
    panel_bin = Path(sys.executable).parent / "panel"
    cmd = [
        str(panel_bin) if panel_bin.exists() else "panel",
        "serve", "src/osw_chatbot/main.py",
        "--port", str(port), "--address", "127.0.0.1",
        "--allow-websocket-origin", f"localhost:{port}",
        "--allow-websocket-origin", f"127.0.0.1:{port}",
    ]
    # BOKEH_ALLOW_WS_ORIGIN takes precedence over --allow-websocket-origin, and
    # the deployment sets it to the public hostname - override it for the child.
    env = dict(os.environ, BOKEH_ALLOW_WS_ORIGIN=f"localhost:{port},127.0.0.1:{port}")
    env.update(env_extra or {})
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    url = f"http://localhost:{port}/main"
    try:
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError(f"panel serve exited with {proc.returncode}")
            try:
                urllib.request.urlopen(url, timeout=2).read()
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            raise RuntimeError("panel serve did not come up")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="http://localhost:8099/")
    ap.add_argument("--backend", default=None,
                    help="existing backend url; omit to start a throwaway server")
    ap.add_argument("--backend-port", type=int, default=52680)
    ap.add_argument("--with-llm", action="store_true")
    ap.add_argument("--timeout", type=int, default=60000)
    ap.add_argument("--headed", action="store_true")
    args = ap.parse_args()

    with contextlib.ExitStack() as stack:
        if args.backend is None:
            args.backend = stack.enter_context(backend_server(args.backend_port))
            print(f"started backend at {args.backend}")
        return run(args)


def run(args):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        page = browser.new_page()
        page.on("console", lambda m: m.type == "error" and print(f"        [console] {m.text}"))
        page.goto(args.harness, timeout=args.timeout)

        page.fill("#backend", args.backend)
        page.click("#reload")

        ## 1. both panes come up, and each one only ever hears from its own iframe

        try:
            for pane in (PANE_A, PANE_B):
                wait_for_log(page, pane, "storage_get", args.timeout)
            ok = True
            detail = ""
        except Exception as e:
            ok, detail = False, str(e)[:200]
        check("both panes load and each asks the parent for its history", ok, detail)

        log_a, log_b = pane_log(page, PANE_A), pane_log(page, PANE_B)
        check(
            "each pane got exactly one storage_get (no cross-delivery)",
            log_a.count("storage_get") == 1 and log_b.count("storage_get") == 1,
            f"A={log_a.count('storage_get')} B={log_b.count('storage_get')}",
        )
        check(
            "no message was dropped as unroutable",
            "dropped 0" in page.locator("#counts").inner_text(),
            page.locator("#counts").inner_text(),
        )

        ## 2. a reply nobody asked for must be ignored

        page.click("#forge")
        page.wait_for_timeout(2000)
        still_alive = chat_frame(page, PANE_A).locator("input.bk-input, textarea.bk-input").first.is_visible()
        check("forged function_call_result with an unknown id is ignored", still_alive)

        ## 3. the message column is not squeezed by the avatar gutter
        ##
        ## Embedded in the wiki the chat is ~210px wide. ChatMessage's
        ## show_avatar=False is accepted but ignored by Panel 1.7.4, so the
        ## avatar is hidden with CSS - which is exactly the kind of fix that
        ## silently regresses on a Panel upgrade.

        frame = chat_frame(page, PANE_A)
        iframe_w = page.locator(".pane").nth(PANE_A).locator("iframe").bounding_box()["width"]
        gutter = frame.locator(".left")
        gutter_w = 0.0
        if gutter.count():
            box = gutter.first.bounding_box()
            gutter_w = box["width"] if box else 0.0
        check(
            "the avatar gutter takes no width",
            gutter_w == 0,
            f"avatar column is {gutter_w:.0f}px of {iframe_w:.0f}px",
        )

        right = frame.locator(".right")
        if right.count():
            box = right.first.bounding_box()
            share = (box["width"] if box else 0) / iframe_w
            check(
                "the message column uses most of the width",
                share > 0.7,
                f"message column is {share * 100:.0f}% of the iframe",
            )

        # The bubble sizes to its content by default, which left a wide gap on
        # the right and wrapped short lines unnecessarily.
        bubble = frame.locator(".message")
        if bubble.count():
            box = bubble.first.bounding_box()
            gap = iframe_w - (box["x"] + box["width"]) if box else iframe_w
            check(
                "the message bubble leaves no wide gap on the right",
                gap < 0.15 * iframe_w,
                f"gap right of the bubble is {gap:.0f}px of {iframe_w:.0f}px",
            )

        ## 4. the stored history is per wiki user, because the parent - not the
        ##    backend - builds the storage key

        page.evaluate(
            """() => {
                localStorage.setItem('osw-chatbot:harnesswiki:Alice:history',
                    JSON.stringify({v:1, updated:1, messages:[
                      {type:'human', data:{content:'ALICE ONLY', additional_kwargs:{},
                       response_metadata:{}, type:'human'}}]}));
                localStorage.removeItem('osw-chatbot:harnesswiki:Bob:history');
            }"""
        )
        for pane in (PANE_A, PANE_B):
            page.locator(".pane").nth(pane).locator("button.reload").click()
        try:
            page.wait_for_function(
                """() => [...document.querySelectorAll('.pane')].every(p =>
                     (p.querySelector('.log').textContent.match(/storage_get/g) || []).length >= 2)""",
                timeout=args.timeout,
            )
            ok, detail = True, ""
        except Exception as e:
            ok, detail = False, str(e)[:200]
        check("both panes re-read storage after switching users", ok, detail)

        a_lines = [l for l in pane_log(page, PANE_A).splitlines() if "storage_get" in l]
        b_lines = [l for l in pane_log(page, PANE_B).splitlines() if "storage_get" in l]
        check(
            "Alice's pane gets Alice's history",
            bool(a_lines) and "bytes" in a_lines[-1],
            f"last: {a_lines[-1] if a_lines else '(none)'}",
        )
        check(
            "Bob's pane does not see Alice's history",
            bool(b_lines) and "(empty)" in b_lines[-1],
            f"last: {b_lines[-1] if b_lines else '(none)'}",
        )

        # leave storage clean for the LLM phase below
        page.evaluate(
            """() => Object.keys(localStorage)
                 .filter(k => k.startsWith('osw-chatbot:'))
                 .forEach(k => localStorage.removeItem(k))"""
        )
        for pane in (PANE_A, PANE_B):
            page.locator(".pane").nth(pane).locator("button.reload").click()
        page.wait_for_timeout(4000)

        ## 5. a tool call reaches only the pane that triggered it

        if args.with_llm:
            try:
                send_prompt(page, PANE_A, "Where am I? Use the where_am_i tool.", args.timeout)
                wait_for_log(page, PANE_A, "function_call where_am_i", args.timeout)
                ok, detail = True, ""
            except Exception as e:
                ok, detail = False, str(e)[:200]
            check("pane A's prompt triggers where_am_i in pane A", ok, detail)

            log_b = pane_log(page, PANE_B)
            check(
                "pane B never saw pane A's tool call",
                "function_call" not in log_b,
                f"pane B log: {log_b[-300:]}",
            )

            ## 4. history survives a reload of that pane

            try:
                wait_for_log(page, PANE_A, "storage_set", args.timeout)
                stored = True
            except Exception:
                stored = False
            check("the answer was persisted to the parent's localStorage", stored)

            if stored:
                # A fresh session asks the parent for its history again; that
                # second storage_get is how we know the reload really happened
                # and we are no longer looking at the previous document.
                before = pane_log(page, PANE_A).count("storage_get")
                page.locator(".pane").nth(PANE_A).locator("button.reload").click()
                try:
                    page.wait_for_function(
                        """([i, n]) => (document.querySelectorAll('.pane')[i]
                               .querySelector('.log').textContent.match(/storage_get/g) || []).length > n""",
                        arg=[PANE_A, before],
                        timeout=args.timeout,
                    )
                    reloaded, detail = True, ""
                except Exception as e:
                    reloaded, detail = False, str(e)[:200]
                if not check("the reloaded pane starts a new session", reloaded, detail):
                    browser.close()
                    return summarize()

                try:
                    frame = chat_frame(page, PANE_A)
                    frame.locator("input.bk-input, textarea.bk-input").first.wait_for(
                        state="visible", timeout=args.timeout
                    )
                    # Panel renders into shadow DOM, so body.inner_text() is
                    # empty - get_by_text pierces open shadow roots.
                    frame.get_by_text("Where am I", exact=False).first.wait_for(
                        state="attached", timeout=args.timeout
                    )
                    ok, detail = True, ""
                except Exception as e:
                    ok, detail = False, str(e)[:200]
                check("the conversation is restored after reloading the pane", ok, detail)

                greeting = chat_frame(page, PANE_A).get_by_text("what's on your mind").count()
                check(
                    "no duplicate greeting on a restored conversation",
                    greeting == 0,
                    f"greeting count = {greeting}",
                )
        else:
            print("SKIP  LLM-dependent checks (pass --with-llm to run them)")

        ## 6. PARENT_ORIGIN is enforced: a backend configured for a different
        ##    parent must not talk to this page at all

        with backend_server(
            args.backend_port + 1,
            env_extra={"PARENT_ORIGIN": "https://not-the-harness.example"},
        ) as locked:
            page.fill("#backend", locked)
            before = pane_log(page, PANE_A).count("storage_get")
            page.locator(".pane").nth(PANE_A).locator("button.reload").click()

            # Positive control: the app must actually be up and rendered,
            # otherwise "no messages" would pass for the wrong reason.
            try:
                chat_frame(page, PANE_A).locator(
                    "input.bk-input, textarea.bk-input"
                ).first.wait_for(state="visible", timeout=args.timeout)
                rendered, detail = True, ""
            except Exception as e:
                rendered, detail = False, str(e)[:200]
            check("the origin-pinned backend still renders (control)", rendered, detail)

            page.wait_for_timeout(8000)
            after = pane_log(page, PANE_A).count("storage_get")
            check(
                "a backend pinned to another PARENT_ORIGIN stays silent",
                rendered and after == before,
                f"storage_get before={before} after={after} "
                "(a message got through - origin pinning is not effective)",
            )

        browser.close()

    return summarize()


def summarize():
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
