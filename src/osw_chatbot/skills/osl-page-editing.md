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
