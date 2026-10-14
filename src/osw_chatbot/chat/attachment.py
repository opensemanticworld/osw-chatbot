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
      button.textContent = "📎";
      button.title = "Attach a file";
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

      // drag and drop anywhere in the chat
      window.addEventListener("dragover", (e) => { e.preventDefault(); });
      window.addEventListener("drop", (e) => {
        if (!e.dataTransfer || !e.dataTransfer.files.length) return;
        e.preventDefault();
        addFiles(e.dataTransfer.files);
      });

      function publish() {
        model.set("attachments", [...staged.values()].map(s => s.meta));
        model.save_changes();
        chips.textContent = [...staged.values()].map(s => s.meta.name).join(", ");
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
      window.addEventListener("message", (event) => {
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
      });

      model.on("change:discard", () => {
        const id = model.get("discard");
        if (!id) return;
        if (id === "*") staged.clear(); else staged.delete(id);
        publish();
      });
    }
    export default { render };
    """
