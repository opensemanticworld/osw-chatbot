---
name: osl-page-editing
description: How to write wiki page content - wikitext instead of Markdown, inline SMW queries and data tables, and the slot/schema conventions for Category, Property, Item and JsonSchema pages.
---

# Writing wiki pages

Read this before using `edit_page`. Three parts: markup, queries, and schema pages.

---

## 1. Wikitext, not Markdown

Wiki pages are **not** Markdown. Markdown written into a page renders literally - the reader
sees `## Heading` and `**bold**` as characters on screen. This is the single most common
mistake; check your output against this table before saving.

| Intent | Write this | Never this |
|---|---|---|
| heading | `== Heading ==`, `=== Sub ===` | `## Heading` |
| bold | `'''bold'''` | `**bold**` |
| italic | `''italic''` | `*italic*` |
| bullet list | `* item`, `** nested` | `- item` |
| numbered list | `# item` | `1. item` |
| internal link | `[[Item:OSW123…\|Label]]` | `[Label](/wiki/Item:OSW123…)` |
| external link | `[https://example.org Label]` | `[Label](https://example.org)` |
| code block | `<syntaxhighlight lang="python">…</syntaxhighlight>` | triple backticks |
| inline code | `<code>x</code>` | `` `x` `` |
| table | `{\| class="wikitable"` … `\|}` | pipe tables |
| definition list | `; term` then `: definition` | — |
| horizontal rule | `----` | `---` |

Further rules:

- A single newline does **not** break a line. A blank line starts a new paragraph; use
  `<br>` only where a hard break is genuinely needed.
- Level-1 `=` headings are reserved for the page title - start at `==`.
- Templates are `{{Name|param=value}}`; parser functions `{{#ask: …}}`, `{{#if: …}}`.
- Prefer wikitext to raw HTML wherever an equivalent exists.
- Signatures (`~~~~`) belong on talk pages only.

A wikitable:

```
{| class="wikitable sortable"
! Name !! Description
|-
| [[Item:OSW123…|Example]] || An example row
|-
| [[Item:OSW456…|Another]] || A second row
|}
```

---

## 2. Inline queries and data tables

Prefer a live query over a hand-written list: it stays correct as the wiki changes.

```
{{#ask: [[Category:OSW...]] [[HasSomeProperty::Value]]
 |?HasName=Name
 |?HasDescription=Description
 |format=datatables
 |limit=50
}}
```

**Conditions** - `[[Category:OSW…]]` restricts by category, `[[Property::Value]]` by property
value, `[[Property::like:*text*]]` matches substrings, `[[Property::!Value]]` negates, and
`OR` combines alternatives. A nested `[[Property::<q>[[Other::X]]</q>]]` is a subquery.
Inverse properties take a leading `-`, e.g. `[[-HasPart::Item:OSW…]]`.

**Printouts** - `|?Property=Label` adds a column with a heading. `|mainlabel=-` drops the page
column. `|limit=`, `|offset=`, `|sort=`, `|order=asc|desc` do what they say.

**Formats available on this wiki**: `table`, `broadtable`, `datatables`, `filtered`, `list`,
`plainlist`, `ul`, `ol`, `category`, `count`, `template`, plus charts (`d3chart`,
`googlecharts`, `jqplotchart`), `timeline`, `calendar`, `gallery`, `tree`.

Choosing:

- `format=table` - short, static lists (roughly under 20 rows).
- `format=datatables` - **the default for anything longer**: sortable, searchable, paginated.
- `format=filtered` - when the user should narrow results by facets.
- `format=count` - just the number.

Existence checks: prefer an `#ask` over `{{#ifexist:}}`, which is an expensive parser
function and aggressively cached:

```
{{#if: {{#ask: [[:Category:OSW...]] |?HasName= |mainlabel=- }} | (exists) | (missing) }}
```

Two limits worth knowing: `subclassof_range` does **not** include the category itself (add an
explicit `[[:Category:OSW…]]OR[[SubClassOf::Category:OSW…]]`), and inverse property chains
break past about three levels (`-SubClassOf.-SubClassOf.-SubClassOf`) - the wiki then reports
conditions it could not consider.

---

## 3. Entity and schema pages

OpenSemanticLab pages keep structured data in **slots** next to the wikitext.

**Which slots a page has**

| Page | Slots |
|---|---|
| `Item:`, `Property:` | `jsondata` (object), empty `main`, `header` = `{{#invoke:Entity\|header}}`, `footer` = `{{#invoke:Entity\|footer}}` |
| `Category:` | the same, plus `jsonschema` (object), and `schema_template` where used |
| `JsonSchema:` (legacy) | the schema lives in `main`, as JSON |

`edit_page` sets the header/footer invokes for you when you write `jsondata`. Without them an
entity renders blank.

> **Never write structured data to `main` on `Item:`/`Property:` pages.** All structured
> content lives in the `jsondata` slot; `main` stays empty wikitext (the invoke calls live in
> `header`/`footer`). When reading a page back via an API or script (e.g.
> `mwjson.api.getPage`), mutate only `page.slots.jsondata` and set only
> `page.slots_changed.jsondata = true`.

JSON in `main` is the mistake to watch for: the page then shows a wall of raw JSON, the
entity has no data as far as the wiki is concerned, and no SMW query will find it.

**Entities, not plain pages**

Anything that *is* a thing - a device, an event, a person, a project, a sample, a method -
belongs in the wiki as an instance of its category, never as a hand-written wikitext page.
The procedure:

1. `find_page_from_topic("device")` → the `Category:OSW…` the user means. Offer the choices
   if several fit rather than picking one silently.
2. `get_category_schema(category)` → the property names and which are required. The property
   holding a picture is **not** always called `image`; read it, do not assume.
3. Write `Item:OSW<uuid without dashes>` with a `jsondata` slot:

```json
{"type": ["Category:OSW..."], "uuid": "<uuid4>",
 "label": [{"text": "Fluke 87V", "lang": "en"}],
 "description": [{"text": "Handheld multimeter", "lang": "en"}],
 "image": "File:OSW....jpg"}
```

The only pages that are plain wikitext or source: the user's own `User:<name>/…` subpages,
`.js`/`.css` pages, and the main page.

**Write it yourself; use the form editor sparingly.** `edit_page` creates the instance
headlessly and can be repeated - ten keywords in a row are ten quiet writes.
`create_category_instance` opens a modal form for the user to review, which is right when
the category has required fields you cannot infer, and wrong for an entity whose only real
field is a label.

The editor returns as soon as it is open, so the user has as long as they want to check the
fields - but the page does not exist until they save, so never report it as created. **Only
one editor can be open at a time**: opening a second while the first is still open does not
work. Open one, say what to do with it, and either write the rest with `edit_page` or wait
for the user to save before offering the next.

**Reference fields are not free text**

A property with `"format": "autocomplete"` - usually carrying a `range`,
`subclassof_range` and/or an `options.autocomplete.query` - holds a **page title**, never a
label or a phrase:

```json
"status": "Item:OSWaa8d29404288446a9f3ec7afa4e2a512"     ✓
"status": "in progress"                                   ✗ nothing can resolve this
"manufacturer": "Item:OSW1234..."                         ✓
"manufacturer": "Fluke"                                   ✗
```

The format is `<Namespace>:OSW<uuid without hyphens>` - `Item:`, `Category:`, `Property:`,
`File:` as appropriate.

Always honour the constraint on the field; a title that exists but is out of range is just
as wrong as free text:

* `range: Category:OSW…` - the value must be an instance of that category.
* `subclassof_range: Category:OSW…` - the value must be a **subclass** of it. The category
  itself is not included (see the undocumented behaviours below).
* `options.autocomplete.query` - run that query with `smw_ask_query` and choose from its
  results. It is the definitive list of what the field accepts.

So: read the schema, resolve the value to a real page (`find_page_from_topic`,
`smw_ask_query`, `full_text_search`), and check it satisfies the constraint. If nothing
suitable exists, ask the user or create the referenced entity first - never invent an OSW-ID
and never fall back to a plain string.

**Files**

An uploaded file is an entity too - `upload_file` creates
`File:OSW<uuid>.<ext>` as an instance of `Category:OSW11a53cdfbdc24524bf8ac435cbf65d9d`
(Wiki File), whose jsondata carries `type`, `uuid`, a self-referencing `file` and a `label`
taken from the original filename. Put the returned title into the entity's image property as
a plain string; it renders through `Special:Redirect/file/`. Do **not** write
`[[File:…|thumb]]` into `main` - that is the plain-wiki reflex and leaves the entity without
a picture as far as the data model is concerned.

**Schema conventions**

- Give every inline sub-object (inside `items`, `properties`, `oneOf`) a meaningful, unique
  `title`. The Python code generator uses it as the class name; without it you get
  collisions like `LabelItem1`.
- Reuse shared definitions by `$ref` to a `JsonSchema:` page, e.g.
  `{"$ref": "/wiki/JsonSchema:Label?action=raw"}`. Keep custom keywords **inside** the
  referenced definition when they are the same everywhere; only when overriding do you put
  them next to the `$ref`, and then give it a distinct `title`.
- Inheritance is `allOf` with a `$ref` to the parent category's schema slot:
  `{"allOf": [{"$ref": "/wiki/Category:OSW...?action=raw&slot=jsonschema"}]}`.
- `propertyOrder` controls field position in the editor; lower comes first. Conventions:
  `type` -1010, `subclass_of` -1009, `label` -1008.
- Non-standard keywords (`eval_template`, `title*`, `description*`, `options`, `watch`,
  `dynamic_template`, `range`, `category`) are meaningful and round-trip through code
  generation - never strip them.
- `range`/`category` point an autocomplete field at a category and create a real dependency
  on that schema. `options.hidden: true` hides a deprecated field without removing it.
  `$comment` is preserved but ignored. `format: "table"` renders array items compactly.
- In schema templates use `{{{_current_subject_}}}`; `{{{_page_title}}}` is deprecated.

**Autocomplete queries in schemas** - three substitution forms, and they are not
interchangeable:

- `{{field}}` - the field exists directly in the page's jsondata.
- `$(watch_var)` - the value comes from a watch path, especially `root.*` paths.
- `{{$(watch_var)}}` - **required inside array item schemas**, where the watch uses a path
  relative to an `id` anchor; Handlebars alone cannot resolve those from the root context.

Inside array `items`, set `"id": "<anchor>"` to create a reference point so sibling fields of
the *same* element can be watched as `<anchor>.<field>`. Built-ins: `{{_user_input}}`,
`{{_user_input_normalized}}`, `{{_user_lang}}`.

**UUIDs and ids** - an OSW-ID is `"OSW"` followed by a UUID without dashes. Use a random
uuid4 for one-off entities; use uuid5 over a canonical string (e.g. an IRI) when the id must
be reproducible across wikis.

**Two warnings before you edit a schema**

1. Category and Property pages that come from a **page package** are overwritten on the next
   package update. A fix made only in the wiki will be silently lost. Say so, and point the
   user at the package repository for anything that should persist.
2. The `Category` namespace is often protected (`$wgNamespaceProtection`) and MwJson adds its
   own per-category edit rights. A `permissiondenied` here usually means the user genuinely
   lacks the right, not that something is broken - relay it rather than retrying.
