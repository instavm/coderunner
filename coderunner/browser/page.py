"""A page as compact text for a model: the accessibility tree rendered with numbered elements, and the
changes between two renderings."""

from __future__ import annotations

import difflib
import json
import urllib.parse
from typing import Any

INTERACTIVE = frozenset({
    "link", "button", "textbox", "searchbox", "combobox", "checkbox", "radio", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "option", "switch", "slider", "spinbutton", "listbox",
})
INLINE = frozenset({"strong", "emphasis", "mark", "subscript", "superscript", "time", "code", "abbr",
                    "insertion", "deletion"})
CHECKABLE = frozenset({"checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio"})
PAGE_CHARS = 12000          # page text returned at once; browser_find reaches the rest
SHOWN_REMOVED = 5           # removed lines listed per change; more are counted
MOSTLY_NEW = 0.5            # below this similarity an action's result is the whole page, not the changes


def ax_value(field: Any) -> Any:
    return field.get("value") if isinstance(field, dict) else None


def squash(text: str) -> str:
    return " ".join(text.split())


def cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters; browser_find searches all of the page]"


def short_url(url: str, page_url: str) -> str:
    target, page = urllib.parse.urlsplit(url), urllib.parse.urlsplit(page_url)
    if target.scheme in ("http", "https") and (target.scheme, target.netloc) == (page.scheme, page.netloc):
        url = urllib.parse.urlunsplit(("", "", target.path or "/", target.query, target.fragment))
    return url if len(url) <= 120 else url[:119] + "…"


def _element_line(ref: int, role: str, name: str, node: dict, props: dict, page_url: str, secret: bool) -> str:
    line = f"[{ref}] {role}"
    if name:
        line += " " + json.dumps(name[:200], ensure_ascii=False)
    value = ax_value(node.get("value"))
    if secret:
        line += " (password field: the owner fills it in, use handoff)"
    elif value not in (None, "") and str(value) != name:
        line += " value=" + json.dumps(str(value)[:200], ensure_ascii=False)
    checked = props.get("checked")
    if role in CHECKABLE and checked is not None:
        line += {"true": " checked", "mixed": " mixed"}.get(str(checked).lower(), " unchecked")
    if props.get("expanded") is not None:
        line += " expanded" if props["expanded"] in (True, "true") else " collapsed"
    for flag in ("selected", "disabled"):
        if props.get(flag) in (True, "true"):
            line += f" {flag}"
    if role == "link" and props.get("url"):
        line += " -> " + short_url(str(props["url"]), page_url)
    return line


def render_ax(nodes: list[dict], page_url: str = "", secret: frozenset | set = frozenset(),
              numbers: dict | None = None) -> tuple[str, dict]:
    """Compact text for an Accessibility.getFullAXTree result.

    Headings become "# name", interactive elements "[N] role "name" ...", and
    other text is joined per block and deduplicated. Returns (text, refs)
    where refs maps N to the element's backendDOMNodeId, role and name.

    numbers ({"next": N, "ids": {backendDOMNodeId: N}}) keeps an element's
    number across renderings of the same document, so a model can act on a
    number it saw earlier and changes between renderings stay small.
    """
    numbers = numbers if numbers is not None else {"next": 1, "ids": {}}
    by_id = {node["nodeId"]: node for node in nodes}
    roots = [node["nodeId"] for node in nodes if node.get("parentId") not in by_id]
    lines: list[str] = []
    refs: dict[int, dict] = {}
    seen: set[str] = set()
    parts: list[str] = []
    owner: list[Any] = [None]

    def flush() -> None:
        text = squash("".join(parts))
        parts.clear()
        owner[0] = None
        if text and text not in seen:
            seen.add(text)
            lines.append(text)

    # Iterative depth-first walk: (node id, text block owner, name that already covers the text).
    stack: list[tuple[str, Any, str]] = [(root, None, "") for root in reversed(roots)]
    while stack:
        node_id, block, covered = stack.pop()
        node = by_id.get(node_id)
        if node is None:
            continue
        children = node.get("childIds") or []
        role = str(ax_value(node.get("role")) or "")
        raw_name = str(ax_value(node.get("name")) or "")
        name = squash(raw_name)

        def descend(new_block: Any, new_covered: str) -> None:
            stack.extend((child, new_block, new_covered) for child in reversed(children))

        if node.get("ignored"):
            descend(block, covered)
        elif role == "StaticText":
            if name and not (covered and name in covered):
                if block != owner[0]:
                    flush()
                    owner[0] = block
                parts.append(raw_name)
        elif role == "LineBreak":
            flush()
        elif role == "InlineTextBox":
            continue
        elif role == "heading":
            flush()
            props = {p.get("name"): ax_value(p.get("value")) for p in node.get("properties", [])}
            if name:
                try:
                    level = min(max(int(props.get("level") or 1), 1), 6)
                except (TypeError, ValueError):
                    level = 1
                lines.append("#" * level + " " + name)
            descend(node_id, name)
        elif role in INTERACTIVE and node.get("backendDOMNodeId"):
            flush()
            props = {p.get("name"): ax_value(p.get("value")) for p in node.get("properties", [])}
            backend = node["backendDOMNodeId"]
            ref = numbers["ids"].get(backend)
            if ref is None or ref in refs:
                ref = numbers["ids"][backend] = numbers["next"]
                numbers["next"] += 1
            is_secret = backend in secret
            refs[ref] = {"backend": backend, "role": role, "name": name, "password": is_secret}
            lines.append(_element_line(ref, role, name, node, props, page_url, is_secret))
            if role not in ("textbox", "searchbox") and not props.get("editable"):
                descend(node_id, name)      # the value is already shown; skip editor internals
        elif role in INLINE:
            descend(block, covered)
        else:
            descend(node_id, covered)
    flush()
    return "\n".join(lines), refs


def page_changes(before: list[str], after: list[str]) -> str | None:
    """What changed between two renderings of a page: added lines (+), removed lines (-) and the line above
    each change for orientation. None when most of the page is new, so the whole page is worth reading."""
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    if matcher.ratio() < MOSTLY_NEW:
        return None
    out: list[str] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if j1 > 0:
            out.append("  " + after[j1 - 1])
        removed = before[i1:i2]
        out += ["- " + line for line in removed[:SHOWN_REMOVED]]
        if len(removed) > SHOWN_REMOVED:
            out.append(f"- … {len(removed) - SHOWN_REMOVED} more lines")
        out += ["+ " + line for line in after[j1:j2]]
    return "\n".join(out)
