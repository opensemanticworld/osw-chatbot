import param
import panel as pn
from panel.custom import AnyWidgetComponent

pn.extension()


# Shared helper for every component that talks to the embedding wiki page.
#
# The chatbot runs in a cross-origin iframe, so `postMessage` is the only
# channel. Without validation any page that can get a handle on this window can
# inject a `function_call_result` (or read our outgoing messages, since a
# wildcard targetOrigin sends to whoever happens to be the parent).
#
# Two independent checks:
#   * `event.source !== window.parent` - blocks sibling frames and openers.
#     Free, and effective even when allowed_origins is ["*"].
#   * `event.origin` against allowed_origins - the actual origin lock.
# The first verified origin is pinned and used as the targetOrigin for all
# outgoing messages, so we stop broadcasting with "*".
ORIGIN_JS = """
function makeOrigin(model) {
  const allowed = model.get("allowed_origins") || ["*"];
  const any = allowed.includes("*");
  // With exactly one configured parent we know where to send from the start.
  // Otherwise fall back to the referrer, which a strict Referrer-Policy may
  // have stripped - in that case we wait for a verified inbound message
  // rather than broadcasting to an unknown parent.
  let target = any ? "*" : (allowed.length === 1 ? allowed[0] : null);
  if (target === null) {
    try {
      const ref = document.referrer ? new URL(document.referrer).origin : null;
      if (ref && allowed.includes(ref)) target = ref;
    } catch (e) { /* opaque referrer */ }
  }
  return {
    accept(event) {
      if (event.source !== window.parent) return false;
      if (any) return true;
      if (!allowed.includes(event.origin)) {
        console.warn("osw-chatbot: dropped message from", event.origin);
        return false;
      }
      target = event.origin;
      return true;
    },
    send(msg) {
      if (target === null) {
        console.warn("osw-chatbot: no verified parent origin, message not sent");
        return;
      }
      window.parent.postMessage(msg, target);
    }
  };
}
"""


class ChatFrontendWidget(AnyWidgetComponent):
    """Bridge between one chat session and the wiki page embedding it.

    One instance per Panel session. Do not create this at module level: a
    param change is pushed to every Document the component is rendered in, so
    a shared instance broadcasts every tool call to every connected browser.
    """

    function_call = param.Dict()
    function_called = param.Dict()
    allowed_origins = param.List(default=["*"])

    _esm = ORIGIN_JS + """
    function render({ model, el }) {
      const org = makeOrigin(model);
      model.on("change:function_call", () => {
        org.send(model.get("function_call"));
      });
      window.addEventListener("message", (event) => {
        if (!org.accept(event)) return;
        if (event.data && event.data["type"] === "function_call_result") {
          model.set("function_called", event.data);
          model.save_changes();
        }
      });
    }
    export default { render };
    """
