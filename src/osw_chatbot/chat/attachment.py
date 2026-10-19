"""File attachments that stay in the browser.

A phone photo is 3-12 MB. Sending it to the server, only to send it back
down so the browser can upload it to the wiki, would cross the websocket
twice and run into bokeh's 20 MB message limit. So the file is **staged in
the iframe**: only metadata and a downscaled preview go to Python, and the
original is handed to the wiki page over postMessage when - and only when -
the assistant actually uploads it.

The preview doubles as the model's view of the image. ~1568px on the long
edge is what vision models want anyway, and it turns 12 MB into ~300 kB.
"""

import param
from panel.custom import AnyWidgetComponent

from osw_chatbot.chat.chat_panel_component import ORIGIN_JS


class AttachmentWidget(AnyWidgetComponent):
    """One instance per Panel session (see ChatFrontendWidget for why)."""

    # client -> server: [{id, name, mime, size, preview, text, error}]
    attachments = param.List(default=[])
    # server -> client: id to drop from staging, or "*" for all
    discard = param.String(default=None, allow_None=True)

    max_bytes = param.Integer(default=20 * 1024 * 1024)
    preview_max_edge = param.Integer(default=1568)
    text_preview_chars = param.Integer(default=20000)
    allowed_origins = param.List(default=["*"])

    # Scoped to the widget's shadow root, so it cannot leak into the chat.
    _stylesheets = ["""
    .osw-attachments {
      display: flex; align-items: center; flex-wrap: wrap; gap: 6px;
      padding: 2px 0 4px 0; font-size: 12px; line-height: 1.4;
    }
    .osw-attach-button {
      display: inline-flex; align-items: center; gap: 5px;
      background: #2a2a2a; color: #ddd; border: 1px solid #444;
      border-radius: 4px; padding: 3px 9px; cursor: pointer;
      font: inherit; font-size: 12px; line-height: 1.4;
    }
    .osw-attach-icon { font-size: 13px; line-height: 1; }
    .osw-attach-button:hover { background: #383838; }
    .osw-attachment-chips { display: flex; flex-wrap: wrap; gap: 4px; }
    .osw-chip {
      display: inline-flex; align-items: center; gap: 5px;
      background: #2f2f2f; color: #c8c8c8; border: 1px solid #3d3d3d;
      border-radius: 11px; padding: 1px 4px 1px 9px; max-width: 100%;
    }
    .osw-chip-name {
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 14ch;
    }
    .osw-chip-error { color: #ff8a80; }
    .osw-chip-remove {
      background: none; border: 0; color: #888; cursor: pointer;
      padding: 0 3px; font-size: 13px; line-height: 1;
    }
    .osw-chip-remove:hover { color: #ddd; }
    """]

    _esm = ORIGIN_JS + """
    const IMAGE = /^image\\//;
    const TEXTISH = /^text\\/|json|csv|xml|yaml|markdown/;

    function downscale(file, maxEdge) {
      return new Promise((resolve) => {
        const url = URL.createObjectURL(file);
        const img = new Image();
        img.onload = () => {
          const scale = Math.min(1, maxEdge / Math.max(img.width, img.height));
          const w = Math.round(img.width * scale), h = Math.round(img.height * scale);
          const canvas = document.createElement("canvas");
          canvas.width = w; canvas.height = h;
          canvas.getContext("2d").drawImage(img, 0, 0, w, h);
          URL.revokeObjectURL(url);
          try {
            // jpeg keeps photos small; png would undo the point for screenshots
            resolve(canvas.toDataURL("image/jpeg", 0.85));
          } catch (e) { resolve(null); }
        };
        img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
        img.src = url;
      });
    }

    function render({ model, el }) {
      const org = makeOrigin(model);
      const staged = new Map();

      const bar = document.createElement("div");
      bar.className = "osw-attachments";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "osw-attach-button";
      button.innerHTML = '<span class="osw-attach-icon">\\u{1F4CE}</span><span>Attach</span>';
      button.title = "Attach a file for the assistant to look at or upload";
      const input = document.createElement("input");
      input.type = "file";
      input.multiple = true;
      input.style.display = "none";
      const chips = document.createElement("span");
      chips.className = "osw-attachment-chips";
      bar.appendChild(button); bar.appendChild(input); bar.appendChild(chips);
      el.appendChild(bar);

      button.onclick = () => input.click();
      input.onchange = () => { addFiles(input.files); input.value = ""; };

      // Drag and drop anywhere in the chat. Capture phase on `document`, which
      // is what the wiki's own uploader uses - it runs before any element
      // handler and before anything that stops propagation on the way up.
      const onDragOver = (e) => { e.preventDefault(); };
      const onDrop = (e) => {
        if (!e.dataTransfer) return;
        // Snapshot synchronously: addFiles awaits, and by then the browser has
        // put the DataTransfer into protected mode and the list reads empty.
        const files = [...e.dataTransfer.files];
        if (!files.length) return;
        e.preventDefault();
        addFiles(files);
      };
      document.addEventListener("dragover", onDragOver, true);
      document.addEventListener("drop", onDrop, true);

      // Files dropped on the wiki page around the iframe never reach this
      // document at all - the gadget catches those and forwards them here.
      const onParentFiles = (event) => {
        if (!org.accept(event)) return;
        const data = event.data;
        if (!data || data.type !== "chatbot_attachment_drop") return;
        addFiles(data.files || []);
      };
      window.addEventListener("message", onParentFiles);

      function publish() {
        model.set("attachments", [...staged.values()].map(s => s.meta));
        model.save_changes();
        chips.textContent = "";
        for (const entry of staged.values()) {
          const meta = entry.meta;
          const chip = document.createElement("span");
          chip.className = "osw-chip";
          const name = document.createElement("span");
          name.className = "osw-chip-name" + (meta.error ? " osw-chip-error" : "");
          name.textContent = meta.name;
          name.title = meta.error ? meta.name + " - " + meta.error : meta.name;
          const remove = document.createElement("button");
          remove.className = "osw-chip-remove";
          remove.type = "button";
          remove.textContent = "×";
          remove.title = "Remove";
          remove.onclick = () => { staged.delete(meta.id); publish(); };
          chip.appendChild(name); chip.appendChild(remove);
          chips.appendChild(chip);
        }
      }

      async function addFiles(files) {
        for (const file of files) {
          const id = "att-" + Math.random().toString(36).slice(2);
          const meta = { id: id, name: file.name, mime: file.type || "", size: file.size };
          if (file.size > model.get("max_bytes")) {
            meta.error = "too large (" + Math.round(file.size / 1048576) + " MB)";
            staged.set(id, { file: null, meta: meta });
            publish();
            continue;
          }
          staged.set(id, { file: file, meta: meta });
          publish();                       // show it immediately, enrich after
          try {
            if (IMAGE.test(file.type)) {
              meta.preview = await downscale(file, model.get("preview_max_edge"));
            } else if (TEXTISH.test(file.type) || /\\.(txt|md|csv|json|ya?ml)$/i.test(file.name)) {
              meta.text = (await file.text()).slice(0, model.get("text_preview_chars"));
            }
          } catch (e) {
            meta.error = String(e && e.message || e);
          }
          publish();
        }
      }

      // The parent asks for the original when the assistant uploads it.
      const onBlobRequest = (event) => {
        if (!org.accept(event)) return;
        const data = event.data;
        if (!data || data.type !== "chatbot_attachment_get") return;
        const entry = staged.get(data.id);
        org.send({
          type: "chatbot_attachment_blob",
          id: data.id,
          requestId: data.requestId,
          name: entry && entry.meta.name,
          mime: entry && entry.meta.mime,
          blob: entry ? entry.file : null      // structured clone, no base64
        });
      };
      window.addEventListener("message", onBlobRequest);

      model.on("change:discard", () => {
        const id = model.get("discard");
        if (!id) return;
        if (id === "*") staged.clear(); else staged.delete(id);
        publish();
      });

      // Panel re-runs render() on a re-render. Without this the old listeners
      // stay on `document`/`window` holding a stale `staged` map, and a stale
      // blob handler can answer an upload request with null before the live
      // one gets to it.
      return () => {
        document.removeEventListener("dragover", onDragOver, true);
        document.removeEventListener("drop", onDrop, true);
        window.removeEventListener("message", onParentFiles);
        window.removeEventListener("message", onBlobRequest);
      };
    }
    export default { render };
    """
