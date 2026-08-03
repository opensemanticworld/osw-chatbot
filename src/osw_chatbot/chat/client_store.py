"""Key/value store living in the *browser*, used to persist the chat history.

The chatbot is embedded as a third-party iframe, and third-party
``localStorage`` is partitioned by top level site (Chrome, Firefox) or blocked
outright (Safari with "Prevent cross-site tracking", private mode). So the
primary backend is the *wiki page's* own first-party storage, reached over
postMessage; the extension namespaces the key per wiki user.

Backends, in order:
  "parent" - the wiki answered our chatbot_storage_get, it owns the storage
  "local"  - no answer (older extension); our own partitioned localStorage
  "none"   - storage unavailable/blocked; the chat works, without persistence

``ready`` flips to True in *all three* cases, so whoever waits for it is never
left hanging.
"""

import param
from panel.custom import AnyWidgetComponent

from osw_chatbot.chat.chat_panel_component import ORIGIN_JS


class ClientStoreWidget(AnyWidgetComponent):
    """One instance per Panel session (see ChatFrontendWidget for why)."""

    # server -> client
    store_key = param.String(default="history")
    write = param.String(default=None, allow_None=True)
    # client -> server
    value = param.String(default=None, allow_None=True)
    backend = param.String(default="none")
    ready = param.Boolean(default=False)

    allowed_origins = param.List(default=["*"])

    _esm = ORIGIN_JS + """
    const PARENT_TIMEOUT_MS = 1200;

    function localGet(key) {
      try { return window.localStorage.getItem(key); } catch (e) { return undefined; }
    }
    function localSet(key, val) {
      try {
        if (val === "") window.localStorage.removeItem(key);
        else window.localStorage.setItem(key, val);
        return true;
      } catch (e) {
        // SecurityError (ITP / cookies blocked) or QuotaExceededError
        console.warn("osw-chatbot: localStorage write failed", e && e.name);
        return false;
      }
    }

    function render({ model, el }) {
      const org = makeOrigin(model);
      const key = model.get("store_key");
      const reqId = "store-" + Math.random().toString(36).slice(2);
      let settled = false;

      function settle(backend, value) {
        if (settled) return;
        settled = true;
        model.set("backend", backend);
        model.set("value", value === undefined || value === null ? null : value);
        model.set("ready", true);
        model.save_changes();
      }

      window.addEventListener("message", (event) => {
        if (!org.accept(event)) return;
        const data = event.data;
        if (!data || data["type"] !== "chatbot_storage_result") return;
        if (data["id"] !== reqId) return;
        settle("parent", data["value"]);
      });

      // Ask the wiki page first; fall back to our own (partitioned) storage.
      org.send({ type: "chatbot_storage_get", id: reqId, key: key });
      setTimeout(() => {
        if (settled) return;
        const val = localGet(key);
        settle(val === undefined ? "none" : "local", val);
      }, PARENT_TIMEOUT_MS);

      model.on("change:write", () => {
        const val = model.get("write");
        if (val === null || val === undefined) return;
        const backend = model.get("backend");
        if (backend === "parent") {
          org.send({
            type: val === "" ? "chatbot_storage_remove" : "chatbot_storage_set",
            key: key,
            value: val
          });
        } else if (backend === "local") {
          if (!localSet(key, val)) {
            model.set("backend", "none");
            model.save_changes();
          }
        }
      });
    }
    export default { render };
    """
