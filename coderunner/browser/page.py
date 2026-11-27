"""A page as compact text for a model: the accessibility tree rendered with numbered elements, and the
changes between two renderings."""

from __future__ import annotations

import difflib
import json
import re
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
# Page text returned at once; the rest is a call away (more=true, start=<text>, browser_find). On a local
# planner every 1,000 tokens of page text cost about 2 s a step to read: 12,000 characters were 6 s.
PAGE_CHARS = 7000
SHOWN_REMOVED = 5           # removed lines listed per change; more are counted
CHANGES_SHARE = 0.7         # changes longer than this share of the page are sent as the whole page
SMALL_PAGE = 1500           # below this size a page's changes are always sent as changes
LINK_CHARS = 60             # link addresses are cut to this
FOOTERS = frozenset({"contentinfo"})     # a page's footer: shown as one line unless the whole page is asked for
# Landmarks whose lines render_ax records (marks): the main content, and menus that a long page shows as one line.
REGIONS = frozenset({"main", "navigation"})
MENU_KEPT = 400             # a menu shorter than this is shown as it is
HEAD_SHARE = 0.25           # of a long page's text, at most this share goes to what comes before its main content
NEAR_CHARS = 1500           # at most, for the lines around the element last used when they fall past the cut
LATER_CONTROLS = 400        # characters of controls (not links) past the cut, named after it
MORE = ("browser_snapshot more=true reads on, start=<text> reads from a later line, browser_find searches the whole "
        "page, browser_list reads a list of results one line per item; scrolling does not change this text")
DRAWN_TEXT = 300            # a pointer-cursor box with more text than this is a region, not a control
DRAWN_NAME = 100            # a drawn control's name, taken from its text, is cut to this
ELEMENT = re.compile(r'^\[(\d+)\] (\w+)(?: ("(?:[^"\\]|\\.)*"))?')


def ax_value(field: Any) -> Any:
    return field.get("value") if isinstance(field, dict) else None


def squash(text: str) -> str:
    return " ".join(text.split())


def cut(text: str, limit: int, changes: bool = False) -> str:
    if len(text) <= limit:
        return text
    if changes:
        return text[:limit] + f"\n… [{len(text) - limit} more characters of changes; browser_snapshot reads the page]"
    return text[:limit] + f"\n… [{len(text) - limit} more characters: {MORE}]"


def _element_name(line: str) -> str:
    match = ELEMENT.match(line)
    if not match or not match.group(3):
        return ""
    try:
        return str(json.loads(match.group(3)))
    except ValueError:
        return ""


def _menu_line(lines: list[str], name: str) -> str:
    """A long menu as one line: how many links it holds and the first few, so the model knows what is there."""
    names = [n for n in (_element_name(line) for line in lines) if n]
    first = ", ".join(n[:30] for n in names[:3]) + (", …" if len(names) > 3 else "")
    label = f" {json.dumps(name[:40], ensure_ascii=False)}" if name else ""
    return (f"[menu{label} with {sum(bool(ELEMENT.match(line)) for line in lines)} links and buttons"
            + (f" ({first})" if first else "") + "; browser_find searches it]")


def fit(lines: list[str], limit: int, marks: dict | None = None, near: int | None = None) -> tuple[str, int]:
    """A page's lines in at most about `limit` characters, the most useful first: long menus as one line each, a
    short share for what comes before the main content, then the main content, and the lines around line `near`
    (the element last used) when they would fall past the cut. Returns the text and the index of the line after
    the last one shown in order, where reading on (more=true) starts.

    Nothing is renumbered or forgotten: an element left out keeps its number, browser_find finds it, and the
    controls (not links) past the cut are named after it."""
    text = "\n".join(lines)
    if len(text) <= limit:
        return text, len(lines)
    marks = marks or {}
    menus: dict[int, tuple[int, str]] = {}
    for start, end, name in marks.get("navigation", []):
        size = sum(len(line) + 1 for line in lines[start:end])
        if size > MENU_KEPT and not (near is not None and start <= near < end) and \
                end > menus.get(start, (start, ""))[0]:
            menus[start] = (end, name)
    # Entries: (first line, line after it, text): a line of the page, or a menu shown as one line.
    entries: list[tuple[int, int, str]] = []
    longest = max(limit // 3, 200)          # a paragraph longer than this is cut, so it cannot take all the room
    i = 0
    while i < len(lines):
        if i in menus:
            end, name = menus[i]
            entries.append((i, end, _menu_line(lines[i:end], name)))
            i = end
        else:
            line = lines[i]
            if len(line) > longest:
                line = line[:longest] + " … [line cut; browser_find reads it whole]"
            entries.append((i, i + 1, line))
            i += 1
    sizes = [len(entry[2]) + 1 for entry in entries]
    if sum(sizes) <= limit:
        return "\n".join(entry[2] for entry in entries), len(lines)
    chosen: set[int] = set()

    def take(first: int, stop: int, room: float) -> tuple[int, int]:
        used, at = 0, first
        while at < stop and used + sizes[at] <= room:
            chosen.add(at)
            used += sizes[at]
            at += 1
        return at, used

    def position(line: int | None) -> int | None:
        if line is None:
            return None
        return next((k for k, entry in enumerate(entries) if entry[0] <= line < entry[1]), None)

    mains = sorted(marks.get("main", []), key=lambda m: m[0] - m[1])      # the largest first
    main = position(mains[0][0]) if mains and mains[0][1] - mains[0][0] > 0 else None
    focus = position(near)
    used = 0
    if main:
        _, used = take(0, main, limit * HEAD_SHARE)
    reserve = min(NEAR_CHARS, limit // 4) if focus is not None else 0
    end, more = take(main or 0, len(entries), limit - used - reserve)
    used += more
    if focus is not None and focus not in chosen:
        take(max(end, focus - 3), len(entries), limit - used)
    elif reserve:
        end, more = take(end, len(entries), limit - used)
        used += more
    out: list[str] = []
    previous = -1
    for k in sorted(chosen):
        if k != previous + 1:
            out.append(f"… [{entries[k][0] - entries[previous + 1][0]} lines left out]")
        out.append(entries[k][2])
        previous = k
    after = [entry for k, entry in enumerate(entries) if k > previous]
    if after:
        rest = sum(len(entry[2]) + 1 for entry in after)
        elements = sum(bool(ELEMENT.match(line)) for line in lines[after[0][0]:])
        out.append(f"… [{rest} more characters" + (f" with {elements} element{'s' * (elements != 1)}" if elements else "")
                   + f": {MORE}]")
        controls, size = [], 0
        for line in lines[after[0][0]:]:
            match = ELEMENT.match(line)
            if match and match.group(2) != "link":
                item = line.split(" -> ")[0][:100]
                if size + len(item) > LATER_CONTROLS:
                    break
                controls.append(item)
                size += len(item) + 2
        if controls:
            out.append("Controls further down: " + "; ".join(controls))
    return "\n".join(out), entries[end - 1][1] if end else 0


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


def _subtree_text(by_id: dict, root: str, limit: int) -> str:
    """The words inside an element as read out: its text and its pictures' alternative text, up to `limit`."""
    words: list[str] = []
    size, stack = 0, list(reversed(by_id[root].get("childIds") or []))
    while stack and size <= limit:
        node = by_id.get(stack.pop())
        if node is None:
            continue
        role = str(ax_value(node.get("role")) or "")
        if role == "StaticText" or role in IMAGES:
            text = squash(str(ax_value(node.get("name")) or ""))
            if text:
                words.append(text)
                size += len(text) + 1
            if role == "StaticText":
                continue
        stack.extend(reversed(node.get("childIds") or []))
    return " ".join(words)


def _drawn_control(by_id: dict, node: dict, props: dict, clickable: frozenset | set) -> str | None:
    """The role of a control the page draws without saying what it is, which the model could not act on: a box
    typed into that is not a form field (a chat's message box, contenteditable), or a box the page makes
    clickable with its own script and shows with a pointer (a support page's "Chat with us"). None for anything else.
    A clickable box holding links or buttons of its own is a card, whose controls are numbered instead."""
    if node.get("ignored"):
        return None
    if props.get("editable") in ("richtext", "plaintext") and props.get("focusable") in (True, "true"):
        return "textbox"
    if node.get("backendDOMNodeId") in clickable and not _count_controls(by_id, node["nodeId"]):
        return "button"
    return None


def render_ax(nodes: list[dict], page_url: str = "", secret: frozenset | set = frozenset(),
              numbers: dict | None = None, whole: bool = False, marks: dict | None = None,
              clickable: frozenset | set = frozenset()) -> tuple[str, dict]:
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

    marks, when given, is filled with the lines of the main content and of menus: {"main": [(first line, line
    after it, name)], "navigation": [...]}, which fit() uses to put the main content first.

    clickable holds the backendDOMNodeIds of boxes the page shows with a pointer cursor (the outermost of each):
    those without controls of their own are numbered as buttons, named by their text. Boxes typed into that are
    not form fields (contenteditable) are numbered as textboxes.
    """
    numbers = numbers if numbers is not None else {"next": 1, "ids": {}}
    by_id = {node["nodeId"]: node for node in nodes}
    roots = [node["nodeId"] for node in nodes if node.get("parentId") not in by_id]
    lines: list[str] = []
    refs: dict[int, dict] = {}
    last = [""]                     # the text of the line above: an element's name or a text block
    parts: list[str] = []
    owner: list[Any] = [None]

    def number(backend: int, role: str, name: str) -> int:
        ref = numbers["ids"].get(backend)
        if ref is None or ref in refs:
            ref = numbers["ids"][backend] = numbers["next"]
            numbers["next"] += 1
        refs[ref] = {"backend": backend, "role": role, "name": name, "password": backend in secret}
        return ref

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
        if node_id is None:                 # the end of a landmark (see REGIONS): block is (role, first line, name)
            flush()
            if marks is not None:
                marks.setdefault(block[0], []).append((block[1], len(lines), block[2]))
            continue
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
            ref = number(node["backendDOMNodeId"], role, name)
            if role == "link" and props.get("url"):
                refs[ref]["url"] = str(props["url"])
            emit(_element_line(ref, role, name, node, props, page_url, node["backendDOMNodeId"] in secret), name)
            if role not in ("textbox", "searchbox") and not props.get("editable"):
                descend(node_id, name)      # the value is already shown; skip editor internals
        elif node.get("backendDOMNodeId") and role not in ("StaticText", "InlineTextBox", "RootWebArea") and \
                (drawn := _drawn_control(by_id, node, {p.get("name"): ax_value(p.get("value"))
                                                       for p in node.get("properties", [])}, clickable)) and \
                (len(text := name or _subtree_text(by_id, node_id, DRAWN_TEXT)) <= DRAWN_TEXT or drawn == "textbox"):
            flush()
            text = text[:DRAWN_NAME]
            ref = number(node["backendDOMNodeId"], drawn, text)
            refs[ref]["drawn"] = True
            emit(_element_line(ref, drawn, text, node, {}, page_url, False), text)   # its text is its name
        elif role in INLINE:
            descend(block, covered)
        elif role in REGIONS:
            flush()
            stack.append((None, (role, len(lines), name), ""))     # popped after everything inside
            descend(node_id, covered)
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
