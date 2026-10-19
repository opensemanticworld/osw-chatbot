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

    # Both directions carry the *whole* current set rather than one message, so
    # that a param update which coalesces with the next one cannot drop a call
    # or a reply. Outbound: every call still waiting for an answer. Inbound: the
    # recent replies, each tagged with a sequence number. A later value is
    # always a superset of what the other side has not processed yet, which is
    # what makes concurrent tool calls safe.
    function_calls = param.List(default=[])
    function_events = param.List(default=[])
    # Sequence ids the backend has handled, so the browser can forget them.
    processed_events = param.List(default=[])
    allowed_origins = param.List(default=["*"])

    _esm = ORIGIN_JS + """
    const EVENT_HISTORY = 50;   // far above any plausible number in flight

    function render({ model, el }) {
      const org = makeOrigin(model);
      const dispatched = new Set();
      let events = [];
      let seq = 0;
      // Panel can re-run render() without tearing the old one down, and the
      // backend remembers which events it has already handled for the life of
      // the session. A per-render token keeps a fresh closure's numbering from
      // colliding with the previous one's, which would make every reply look
      // like a duplicate and silently strand every tool call.
      const token = Math.random().toString(36).slice(2);

      model.on("change:function_calls", () => {
        const calls = model.get("function_calls") || [];
        const live = new Set(calls.map((c) => c.id));
        // Python drops a call once it is answered or gives up on it; forgetting
        // it here too keeps the set from growing for the life of the session.
        for (const id of [...dispatched]) if (!live.has(id)) dispatched.delete(id);
        for (const call of calls) {
          if (dispatched.has(call.id)) continue;   // already running
          dispatched.add(call.id);
          org.send(call);
        }
      });

      function emit(data) {
        events.push({ ...data, seq: token + ":" + (++seq) });
        if (events.length > EVENT_HISTORY) events = events.slice(-EVENT_HISTORY);
        model.set("function_events", [...events]);
        model.save_changes();
      }

      // The list is only cumulative so that a coalesced update cannot hide a
      // reply python has not seen yet. Once it confirms a reply, drop it:
      // results carry whole pages and data urls, and re-serialising those on
      // every later event is what froze the chat.
      model.on("change:processed_events", () => {
        const done = new Set(model.get("processed_events") || []);
        if (!done.size) return;
        events = events.filter((e) => !done.has(e.seq));
        // Deliberately no model.set here - python already has these, and
        // publishing the shorter list would just bounce back again.
      });

      const onMessage = (event) => {
        if (!org.accept(event)) return;
        const data = event.data;
        if (!data) return;
        // A progress message buys the call more time - the browser is alive,
        // it is just waiting for the user to answer a confirmation dialog.
        if (data["type"] === "function_call_result" ||
            data["type"] === "function_call_progress") {
          emit(data);
        }
      };
      window.addEventListener("message", onMessage);

      // Without this a re-render leaves the old listener attached: two
      // listeners then publish rival cumulative lists and replies get lost.
      return () => { window.removeEventListener("message", onMessage); };
    }
    export default { render };
    """
