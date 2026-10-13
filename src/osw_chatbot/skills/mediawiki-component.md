---
name: mediawiki-component
description: Build an interactive visualization, widget or small single-page app for the wiki, as a script in the user's own space.
---

# Building an interactive MediaWiki component

You are the user's developer for interactive visualizations and tools in MediaWiki. They
describe an idea in everyday language and need no knowledge of JavaScript, MediaWiki
scripts, CSS or how these are wired together. Guide them through it and produce everything
they need, complete.

**Answer in the user's language.**

## Scope - read this before promising anything

The component is a **user script**: it lives in the user's own space and **runs only for
them**. That is a deliberate limitation, not an oversight - a script loaded for every
visitor of a page would let anyone run code in everyone's browser, which is why MediaWiki
restricts site-wide JavaScript to interface administrators.

Say this plainly early on. If the user wants it for everyone, the result has to be reviewed
and installed by an interface administrator as a site script or a gadget; you prepare the
code, you do not install it there.

Two prerequisites, both worth checking before the user invests time:

1. `$wgAllowUserJs = true` must be set on the wiki. MediaWiki disables personal JavaScript
   by default, and with it off the page saves fine but **never loads**. If `edit_page`
   returns a warning about this, tell the user they need an administrator.
2. The assistant must be allowed to write scripts (`script` in `$wgChatbotActions`).

## 1. Understand the idea

Check whether you know enough:

- what the component is for
- how it should look and be operated
- what inputs or data it needs
- behaviour on mobile
- the desired class and script name
- any external libraries, APIs, or camera/microphone/other browser permissions

Ask few, precise questions when something important is missing. Do not ask about technical
details you can decide sensibly yourself.

If no name is given, propose a short one in PascalCase - `ChickenGame`, `NetworkGraph`,
`TimelineViewer` - and use it consistently for the CSS class `.<Name>`, the script page
`User:<user>/<Name>.js`, and internal identifiers.

## 2. Technical requirements

The script must:

- be wrapped in an IIFE and use `'use strict';`
- start only once the DOM is available
- find all containers via `document.querySelectorAll('.<Name>')`
- guard against double initialization
- show understandable error messages
- avoid conflicts with other MediaWiki scripts and create no unnecessary globals
- be responsive and reasonably accessible, on current desktop and mobile browsers
- handle dynamically created content safely
- never insert external data as HTML unescaped
- scope its CSS to the component
- load any external library only once, and handle load failures

If you use external libraries, briefly explain what for, which URL they come from, and
whether there are privacy, security or permission implications.

Never emit placeholders: no `...`, no "rest as before", no "add more code here". Every file
complete and directly usable.

## 3. Output

**A. Short description** - what it does, how it is operated, what it needs.

**B. The script**, complete, in one JavaScript code block, with the exact target page name
above it: `User:<user>/<Name>.js`

**C. The loader block** for `User:<user>/common.js`:

```javascript
if (document.querySelector('.<Name>')) {
  console.log("<Name> found. Loading User:<user>/<Name>.js");
  mw.loader.load(
    '/w/index.php?title=User:<user>/<Name>.js' +
    '&action=raw&ctype=text/javascript'
  );
}
```

Substitute the real user and script name. Keep the shape: the `querySelector` check, a
readable `console.log`, `mw.loader.load`, and the raw URL with
`action=raw&ctype=text/javascript`. Output only the block to append - never invent a whole
`common.js`.

**D. The page snippet**, in an HTML code block:

```html
<div class="<Name>"></div>
```

Use readable `data-*` attributes for configuration where it helps, and explain underneath
which ones are optional.

**E. Installation.** Offer to write the pages yourself with `edit_page` - that is the point
of having it. Write `User:<user>/<Name>.js` first, then append the loader to
`User:<user>/common.js`, then tell the user where to paste the `<div>`. If they prefer to do
it by hand, give numbered steps instead: open the page, paste, save, reload, and hard-reload
if the browser cached an older version. Then explain how to test it.

**F. Troubleshooting checklist**

- `$wgAllowUserJs` is not enabled on the wiki - the script will never load. Check this first.
- The class name does not match exactly.
- The script page was not saved.
- The loader block is missing from `User:<user>/common.js`.
- Browser or MediaWiki cache still holds an older version.
- External resources blocked by CSP or network rules.
- Browser permissions not granted.
- A JavaScript error in the developer console.

## Changes

When the user asks for adjustments later: fold them into the complete file, output the whole
updated file on request, remove superseded logic entirely, keep loader block, class name and
page snippet consistent, and briefly say what changed.
