"""Exercise the MwJson write path against a real wiki.

Complements the harness, which fakes the wiki side: this logs in with a browser
and drives mwjson.api directly, so it checks what MediaWiki actually does -
that a jsondata slot really is stored as JSON, that editing one slot leaves the
others alone, and which error codes come back when a write is refused.

    MW_ADMIN_PASS=... uv run python src/osw_chatbot/tests/harness/wiki_write_test.py

Set WIKI_URL/WIKI_USER to point it elsewhere. It creates and deletes its own
pages under Item:OSWchatbottest0001 and User:<user>/ChatbotTest.js.
"""
import json, os, sys
from playwright.sync_api import sync_playwright

WIKI = os.environ.get("WIKI_URL", "https://wiki-dev.open-semantic-lab.org/w/index.php")
USER = os.environ.get("WIKI_USER", "Admin")
PASS = os.environ["MW_ADMIN_PASS"]
TITLE = "Item:OSWchatbottest0001"

results = []
def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {detail}" if detail else ""))

with sync_playwright() as pw:
    page = pw.chromium.launch(headless=True).new_page()
    page.goto(f"{WIKI}?title=Special:UserLogin", timeout=60000)
    page.fill("#wpName1", USER)
    page.fill("#wpPassword1", PASS)
    page.press("#wpPassword1", "Enter")
    page.wait_for_timeout(4000)
    page.goto(f"{WIKI}?title=Main_Page", timeout=60000)
    user = page.evaluate("() => mw.config.get('wgUserName')")
    check("logged in", user == USER, f"wgUserName={user!r}, expected {USER!r}")
    if user != "Admin":
        sys.exit(1)

    page.goto(f"{WIKI}?title=Main_Page", timeout=60000)
    page.wait_for_function(
        "() => window.mw && mw.loader && typeof mw.loader.using === 'function'",
        timeout=60000)

    # 1. create a page with two slots through the same calls the tool makes
    res = page.evaluate("""async ([title]) => {
        await mw.loader.using(['ext.mwjson.api']);
        try {
            const p = await mwjson.api.getPage(title);
            const created = !p.exists;
            p.slots['main'] = "== Chatbot test ==\\n'''bold''' text";
            p.slots_changed['main'] = true;
            p.slots['jsondata'] = { label: [{ text: "Chatbot test", lang: "en" }] };
            p.slots_changed['jsondata'] = true;
            await mwjson.api.updatePage(p, { comment: "chatbot verification" });
            return { ok: true, created: created };
        } catch (e) { return { ok: false, code: e && e.code, info: e && (e.info || e.message) }; }
    }""", [TITLE])
    check("create a page with main + jsondata", res.get("ok"), json.dumps(res))

    # 2. read it back: both slots stored, jsondata as JSON
    back = page.evaluate("""async ([title]) => {
        const r = await fetch(mw.config.get('wgScriptPath') +
            '/api.php?action=query&format=json&prop=revisions&rvprop=content&rvslots=*&titles=' +
            encodeURIComponent(title));
        const d = await r.json();
        const pages = d.query.pages; const k = Object.keys(pages)[0];
        const slots = (pages[k].revisions || [{}])[0].slots || {};
        return Object.fromEntries(Object.entries(slots).map(([n, s]) => [n, s['*']]));
    }""", [TITLE])
    check("main slot stored as wikitext", "'''bold'''" in (back.get("main") or ""), str(back.get("main"))[:80])
    jd = back.get("jsondata")
    check("jsondata stored as JSON", bool(jd) and json.loads(jd).get("label"), str(jd)[:80])

    # 3. edit one slot only - the others must survive
    res = page.evaluate("""async ([title]) => {
        const p = await mwjson.api.getPage(title);
        p.slots['main'] = "== Changed ==";
        p.slots_changed['main'] = true;
        try { await mwjson.api.updatePage(p, { comment: "second edit" }); return { ok: true }; }
        catch (e) { return { ok: false, info: e && (e.info || e.message) }; }
    }""", [TITLE])
    back2 = page.evaluate("""async ([title]) => {
        const r = await fetch(mw.config.get('wgScriptPath') +
            '/api.php?action=query&format=json&prop=revisions&rvprop=content&rvslots=*&titles=' +
            encodeURIComponent(title));
        const d = await r.json(); const pages = d.query.pages; const k = Object.keys(pages)[0];
        const slots = (pages[k].revisions || [{}])[0].slots || {};
        return Object.fromEntries(Object.entries(slots).map(([n, s]) => [n, s['*']]));
    }""", [TITLE])
    check("editing one slot leaves the others intact",
          "Changed" in (back2.get("main") or "") and bool(back2.get("jsondata")),
          f"main={str(back2.get('main'))[:40]!r} jsondata present={bool(back2.get('jsondata'))}")

    # 4. a write Admin may not do: site JS needs editsitejs
    res = page.evaluate("""async () => {
        try {
            const p = await mwjson.api.getPage('MediaWiki:Common.js');
            p.slots['main'] = "// chatbot test";
            p.slots_changed['main'] = true;
            await mwjson.api.updatePage(p, { comment: "should fail" });
            return { ok: true };
        } catch (e) { return { ok: false, code: e && e.code, info: e && (e.info || e.message) }; }
    }""")
    check("site JS is refused by MediaWiki itself", res.get("ok") is False,
          json.dumps(res)[:160])

    # 5. a user-space script is allowed
    res = page.evaluate("""async ([title]) => {
        try {
            const p = await mwjson.api.getPage(title);
            p.slots['main'] = "// written by the chatbot verification";
            p.slots_changed['main'] = true;
            await mwjson.api.updatePage(p, { comment: "user script" });
            return { ok: true };
        } catch (e) { return { ok: false, code: e && e.code, info: e && (e.info || e.message) }; }
    }""", [f"User:{USER}/ChatbotTest.js"])
    check("a user-space script page can be written", res.get("ok"), json.dumps(res)[:160])

    # 6. upload a file the way upload_file does, and check it is a WikiFile entity
    res = page.evaluate("""async () => {
        await mw.loader.using(['ext.mwjson.api', 'ext.mwjson.util', 'ext.OpenSemanticLab.forms']);
        // 1x1 png
        const b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=";
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const uuid = mwjson.util.uuidv4();
        const target = mwjson.util.OswId(uuid) + ".png";
        const title = "File:" + target;
        const file = new File([new Blob([bytes], {type: "image/png"})], target, {type: "image/png"});
        try {
            let page;
            try { page = await mwjson.api.getFilePage(target); }
            catch (e) { page = await mwjson.api.getPage(title); }
            page.file = file; page.file.contentBlob = file; page.file.changed = true;
            page.slots['jsondata'] = {
                type: ["Category:OSW11a53cdfbdc24524bf8ac435cbf65d9d"],
                uuid: uuid, file: title,
                label: [{ text: "chatbot verification.png", lang: "en" }]
            };
            page.slots_changed['jsondata'] = true;
            page = await osl.util.postProcessPage(page, []);
            await mwjson.api.updatePage(page, { comment: "chatbot verification" });
            return { ok: true, title: title };
        } catch (e) { return { ok: false, code: e && e.code, info: e && (e.info || e.message) }; }
    }""")
    check("upload a file as a WikiFile entity", res.get("ok"), json.dumps(res)[:200])

    if res.get("ok"):
        file_title = res["title"]
        back = page.evaluate("""async ([title]) => {
            const r = await fetch(mw.config.get('wgScriptPath') +
                '/api.php?action=query&format=json&prop=revisions&rvprop=content&rvslots=*&titles=' +
                encodeURIComponent(title));
            const d = await r.json(); const pages = d.query.pages; const k = Object.keys(pages)[0];
            const slots = (pages[k].revisions || [{}])[0].slots || {};
            return {missing: 'missing' in pages[k],
                    slots: Object.fromEntries(Object.entries(slots).map(([n, s]) => [n, s['*']]))};
        }""", [file_title])
        jd = json.loads(back["slots"].get("jsondata") or "{}")
        check("the file page carries the WikiFile type and a self-reference",
              jd.get("type") == ["Category:OSW11a53cdfbdc24524bf8ac435cbf65d9d"]
              and jd.get("file") == file_title and bool(jd.get("label")),
              json.dumps(jd)[:160])
        check("the file page has the header/footer invokes",
              "#invoke:Entity|header" in (back["slots"].get("header") or ""),
              str(back["slots"].get("header"))[:60])
        page.evaluate("""async ([t]) => { try { await mwjson.api.deletePage(t, {comment:'cleanup'}); } catch (e) {} }""",
                      [file_title])

    # cleanup
    for t in [TITLE, f"User:{USER}/ChatbotTest.js"]:
        page.evaluate("""async ([t]) => { try { await mwjson.api.deletePage(t, {comment:'cleanup'}); } catch (e) {} }""", [t])

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
