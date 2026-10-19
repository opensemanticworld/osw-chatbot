"""Clean up the data used to prefill the category form editor.

The editor is prefilled by a second, schema-constrained LLM pass. That pass
sees the JSON Schema but not what OpenSemanticLab means by it, so it fills
reference fields the way a person would write them - as a name, or as a label
object - and the editor stores whatever it is given. The result is a field
whose declared `range` says "a page title" holding
`{"text": "Materials science", "lang": "en"}`, which resolves to nothing.

So: find the values that are supposed to be page titles, reduce them to the
plain text the model meant, and let the caller look each one up. Anything that
cannot be resolved is dropped rather than guessed at - an invented OSW-ID is
worse than an empty field, because it looks right.
"""

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

# <Namespace>:OSW<uuid without hyphens> - what a reference field must contain.
REFERENCE_TITLE = re.compile(r"^[A-Za-z][A-Za-z_ ]*:OSW[0-9a-fA-F]{32}$")

# Keys a label-shaped object uses for the text a human would read.
_TEXT_KEYS = ("text", "label", "name", "title")


def normalize_label(text: str) -> str:
    """Lowercase alphanumerics, matching the wiki's HasNormalizedLabel."""
    return re.sub(r"[^0-9a-z]", "", (text or "").lower())


def looks_like_reference(value: Any) -> bool:
    return isinstance(value, str) and bool(REFERENCE_TITLE.match(value.strip()))


def label_text(value: Any) -> Optional[str]:
    """The human-readable text inside whatever the model produced."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, dict):
        for key in _TEXT_KEYS:
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
            # label: [{"text": ...}] nested one level deeper
            if isinstance(found, list):
                nested = label_text(found[0]) if found else None
                if nested:
                    return nested
    return None


def is_reference_schema(subschema: Any) -> bool:
    """True when this schema describes a page title, not free text."""
    if not isinstance(subschema, dict):
        return False
    if subschema.get("format") == "autocomplete":
        return True
    if "range" in subschema or "subclassof_range" in subschema:
        return True
    options = subschema.get("options")
    if isinstance(options, dict) and isinstance(options.get("autocomplete"), dict):
        return "query" in options["autocomplete"]
    return False


def reference_constraint(subschema: Dict[str, Any]) -> Dict[str, Any]:
    """What the value has to satisfy, for the lookup and for the message."""
    constraint = {}
    for key in ("range", "subclassof_range"):
        if subschema.get(key):
            constraint[key] = subschema[key]
    options = subschema.get("options")
    if isinstance(options, dict) and isinstance(options.get("autocomplete"), dict):
        query = options["autocomplete"].get("query")
        if query:
            constraint["query"] = query
    return constraint


def find_references(
    data: Any, schema: Any, path: str = ""
) -> Iterator[Tuple[list, Any, str, Any, Dict[str, Any]]]:
    """Yield every value sitting in a reference field.

    Each item is (container, key, path, value, constraint), where container[key]
    is the value - so the caller can replace or delete it in place.
    """
    if not isinstance(schema, dict):
        return

    if isinstance(data, dict):
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return
        for key, value in list(data.items()):
            subschema = properties.get(key)
            if not isinstance(subschema, dict):
                continue
            here = f"{path}.{key}" if path else key
            if is_reference_schema(subschema):
                yield data, key, here, value, reference_constraint(subschema)
            elif subschema.get("type") == "array" or "items" in subschema:
                yield from find_references(value, subschema, here)
            else:
                yield from find_references(value, subschema, here)

    elif isinstance(data, list):
        items = schema.get("items")
        if not isinstance(items, dict):
            return
        for index, value in enumerate(data):
            here = f"{path}.{index}"
            if is_reference_schema(items):
                yield data, index, here, value, reference_constraint(items)
            else:
                yield from find_references(value, items, here)


def describe_constraint(constraint: Dict[str, Any]) -> str:
    if constraint.get("range"):
        return f"an instance of {constraint['range']}"
    if constraint.get("subclassof_range"):
        return f"a subclass of {constraint['subclassof_range']}"
    if constraint.get("query"):
        return f"a result of the query {constraint['query']}"
    return "a page title"


def drop(container: Any, key: Any) -> None:
    """Remove a value, whichever kind of container it sits in."""
    if isinstance(container, dict):
        container.pop(key, None)
    elif isinstance(container, list) and isinstance(key, int):
        if 0 <= key < len(container):
            container[key] = None


def prune_empty(data: Any) -> Any:
    """Drop the holes left by `drop`, so the editor sees no empty rows."""
    if isinstance(data, dict):
        return {k: prune_empty(v) for k, v in data.items() if v is not None}
    if isinstance(data, list):
        return [prune_empty(v) for v in data if v is not None]
    return data
