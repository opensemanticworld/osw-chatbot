---
name: sandbox
description: The in-browser Python sandbox - when to open it instead of pasting code, how to hand the user a notebook, and that it runs as them.
---

# The sandbox

A JupyterLite sandbox is served from this wiki's own origin. Python runs in the user's
browser, and because it is the same origin, it already has their MediaWiki session:
`osw-python` reads and writes the wiki **as them**, with no login step and nothing to
configure.

`osw-python` and the `opensemantic.*` packages are preinstalled. The first cell of a fresh
sandbox is:

```python
import osw_bootstrap
await osw_bootstrap.setup()
wtsite = osw_bootstrap.wtsite()
```

`osw_bootstrap.setup()` installs the stack in an order micropip needs; do not replace it
with `micropip.install("osw")`, which fails. `osw_bootstrap.wtsite()` returns a `WtSite`
bound to this wiki. Never write a username, a password or a credential path into a
notebook - there is nothing to authenticate and asking the user for one is wrong.

## When to open it rather than answer in chat

Open the sandbox when the work is **computation over wiki data**: a query whose results
need aggregating, a plot, a bulk edit, anything the user will want to re-run or adjust.
A notebook they can change and re-run is worth more than a result they have to trust.

Stay in chat for a single lookup, a single page edit, or a question about how something
works. `smw_ask_query` and `get_page_content` answer those faster than a kernel can start.

**The first cell is slow** - roughly 40 seconds while Python and the OSW packages load
into the browser. Say so when you open it, and do not open it for something you could
have answered directly.

## How to open it

- `open_sandbox()` - an empty sandbox.
- `open_sandbox(notebook="File:OSW....ipynb")` - opens a notebook already uploaded to the
  wiki. Only wiki files work: the sandbox refuses anything else, because a notebook runs
  with the reader's rights and a link to an arbitrary one would be a way to make somebody
  edit the wiki without noticing.
- `create_notebook(name, cells)` - write a notebook and open it. Prefer this when you are
  producing the code yourself.

It opens in its own tab and stays there, so a second call lands on the kernel the user
already has running rather than starting another.

## Running code

`run_in_sandbox(code)` executes a cell. The user is asked to confirm first, every time.

That confirmation is not a formality: you read wiki content, wiki content can be written
by anyone, and the sandbox holds the user's live session. Anything you run can edit the
wiki as them. So:

- Say what the code will do before you run it, in one line.
- Never run code you assembled from page content without reading it yourself first.
- If a run is refused, that is an answer. Do not rephrase and try again.

### Always use absolute URLs

The sandbox is a separate document from the wiki page, so a relative URL like
`/w/api.php?...` resolves against the sandbox, not against the wiki. It will not fail
loudly - the request goes somewhere unhelpful and the cell finishes with an empty result,
which is indistinguishable from "the wiki has no such data" unless you look.

Take the wiki's origin from `where_am_i` and build the whole URL from it, every time.
This applies to anything the code addresses by path, not just the API.

### A cell that finished is not a cell that worked

Reaching the end without an exception says nothing about what came back. Before you
report a result, read the output: how many rows, what date range, which names. An empty
DataFrame, a zero-length list and a successful request to the wrong place all look like
calm success.

If the output does not carry that evidence, print it and run again - do not describe what
the code was supposed to produce. And if you are reporting without having checked, say
so.

## Moving files between the sandbox and the wiki

Use the tools, not code:

- `push_file_to_osw(path, label)` uploads a file the code produced.
- `pull_file_from_osw(title, path)` copies a wiki file in so the code can read it.
- `save_notebook_to_wiki(label)` for the open notebook; it saves it first.

Writing Python to do this instead is the mistake to avoid. `osw-python` talks to the
API directly, so nothing checks the result: the page gets whatever name the code chose
and no jsondata, which leaves a file nobody can find by querying. The tools mint the
OSW-ID, write the entity before the bytes, and put the readable name where the wiki
looks for it.

It also wastes the user's time - a hand-written upload is a code cell, so they are asked
to confirm Python they did not want to read, and it has already failed in practice with
`ValidationError: osw - value is not a valid dict`, because `WikiFileController` wants an
`OSW` instance and the obvious guess is to hand it the `WtSite`.

## The kernel forgets when the page reloads

The notebook and `kernel_execute` share one kernel - the same interpreter, the same
namespace, one execution counter between them. Code run either way sees what the other
defined.

What breaks that is a reload, not the tools. Reloading the wiki page restarts the kernel,
so every name defined before it is gone while the notebook still shows the cells and
their old output, which reads exactly like the tools not sharing state.

So when a name that was defined earlier is suddenly missing, re-run the setup cells. Do
not conclude the kernel is not shared and start working around it.

## The sandbox cannot reach most of the web

It runs in the user's browser, so every request it makes is the browser's request and the
same-origin policy applies. `requests.get("https://example.org/...")` works only if that
site sends `Access-Control-Allow-Origin`, and most do not. `pyodide_http.patch_all()`
routes `requests` through the browser's own fetch - it does not exempt it. The failure
looks like this, and no amount of rewriting the Python changes it:

```
Access to fetch at 'https://example.org/page' from origin 'https://your.wiki'
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header
```

What does work from inside the sandbox:

- **this wiki** - same origin, and the user's session rides along, which is the whole
  reason `osw-python` needs no credentials there;
- sites that publish CORS headers, which most public APIs do and most ordinary web pages
  do not.

**So do not fetch the open web from the sandbox.** Read the page yourself with
`simple_browse_web_page_tool`, which runs on the server and is not subject to CORS, then
put the data into the notebook as a literal and let the code work on it. The user ends up
with a notebook that runs for them rather than one that fails on the first cell.

### A proxy can lift the limit, if the user chooses it

CORS is not a legal boundary - it is a browser mechanism that stops page JavaScript
reading another origin's response *with the user's credentials*. A server-side proxy
fetches anonymously, so it can only ever retrieve what any anonymous visitor could, and
it is what an ordinary Jupyter server does by default.

So if the user needs a site reachable from inside the sandbox, say that it is possible
and what it costs, rather than either refusing or quietly routing through some public
proxy:

- requests then leave the **wiki's** address, not the user's, and appear in that site's
  logs as the wiki;
- the wiki becomes responsible for what is fetched through it - rate limits, terms of
  use and robots.txt are the operator's problem now;
- an unrestricted proxy is a server-side request forgery tool: it can reach whatever the
  server can, including cloud metadata and internal services. It has to be restricted to
  named hosts, not opened generally.

The wiki may already have `ApiGateway`, which proxies allow-listed endpoints and blocks
private address ranges. Adding a host there is a configuration change an administrator
makes deliberately - suggest it, do not improvise around it. Never pipe the user's data
through a third-party proxy you found; that hands their content to a stranger.

## Writing a notebook the user can use

Make it re-runnable from the top: the bootstrap cell first, then one idea per cell, with a
markdown cell saying what the notebook is for. Leave the last cell showing the result
rather than assigning it to a variable the user has to inspect.
