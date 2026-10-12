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
IMAGES = frozenset({"image", "img"})     # their alternative text is read like text (an airline's logo)
# Containers whose name can be content: booking.com puts a flight's price in a region's name, with no text
# inside. The name is read like text, and text inside that only repeats it is skipped.
NAMED = frozenset({"region", "group", "article", "figure", "status", "alert", "note", "cell", "gridcell",
                   "listitem", "term", "definition", "math", "timer", "log"})
PAGE_CHARS = 12000          # page text returned at once; browser_find reaches the rest
SHOWN_REMOVED = 5           # removed lines listed per change; more are counted
CHANGES_SHARE = 0.7         # changes longer than this share of the page are sent as the whole page
SMALL_PAGE = 1500           # below this size a page's changes are always sent as changes
LINK_CHARS = 60             # link addresses are cut to this
FOOTERS = frozenset({"contentinfo"})     # a page's footer: shown as one line unless the whole page is asked for


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
    if len(url) <= LINK_CHARS:
        return url
    # Tracking parameters fill most long links; the path says where a link goes.
    path = url.split("?", 1)[0]
    return path + "?…" if "?" in url and len(path) < LINK_CHARS else url[:LINK_CHARS - 1] + "…"


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
              numbers: dict | None = None, whole: bool = False) -> tuple[str, dict]:
    """Compact text for an Accessibility.getFullAXTree result.

    Headings become "# name", interactive elements "[N] role "name" ...", images
    their alternative text, and other text is joined per block. Text equal to
    the line just above it (a label under its control, a paragraph shown twice)
    is dropped; repeated text elsewhere is kept, since in lists of results each
    entry's "Direct" or "2h 55m" belongs to it. Returns (text, refs) where refs
    maps N to the element's backendDOMNodeId, role and name.

    numbers ({"next": N, "ids": {backendDOMNodeId: N}}) keeps an element's
    number across renderings of the same document, so a model can act on a
    number it saw earlier and changes between renderings stay small.

    The footer (dozens of links on most sites) is one line unless whole is set.
    """
    numbers = numbers if numbers is not None else {"next": 1, "ids": {}}
    by_id = {node["nodeId"]: node for node in nodes}
    roots = [node["nodeId"] for node in nodes if node.get("parentId") not in by_id]
    lines: list[str] = []
    refs: dict[int, dict] = {}
    last = [""]                     # the text of the line above: an element's name or a text block
    parts: list[str] = []
    owner: list[Any] = [None]

    def emit(line: str, text: str) -> None:
        lines.append(line)
        last[0] = text

    def flush() -> None:
        text = squash("".join(parts))
        parts.clear()
        owner[0] = None
        if text and text != last[0]:
            emit(text, text)

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
        elif role in FOOTERS and not whole:
            flush()
            emit(f"[footer with {_count_controls(by_id, node_id)} links and buttons; browser_find searches it]", "")
        elif role == "StaticText" or (role in IMAGES and name):
            if name and not (covered and name in covered):
                if block != owner[0]:
                    flush()
                    owner[0] = block
                parts.append(raw_name if role == "StaticText" else f" {raw_name} ")
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
                emit("#" * level + " " + name, name)
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
            emit(_element_line(ref, role, name, node, props, page_url, is_secret), name)
            if role not in ("textbox", "searchbox") and not props.get("editable"):
                descend(node_id, name)      # the value is already shown; skip editor internals
        elif role in INLINE:
            descend(block, covered)
        elif role in NAMED and name and not (covered and name in covered):
            if owner[0] is not None:
                flush()
            owner[0] = node_id
            parts.append(f" {raw_name} ")
            descend(node_id, name)
        else:
            descend(node_id, covered)
    flush()
    return "\n".join(lines), refs


def _count_controls(by_id: dict, root: str) -> int:
    count, stack = 0, [root]
    while stack:
        node = by_id.get(stack.pop())
        if node is None:
            continue
        count += str(ax_value(node.get("role")) or "") in INTERACTIVE and not node.get("ignored")
        stack.extend(node.get("childIds") or [])
    return count


def page_changes(before: list[str], after: list[str]) -> str | None:
    """What changed between two renderings of a page: added lines (+), removed lines (-) and the line above
    each change for orientation. None when the changes are not much shorter than the page itself."""
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
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
    changes, page = "\n".join(out), "\n".join(after)
    return None if len(page) > SMALL_PAGE and len(changes) > CHANGES_SHARE * len(page) else changes
