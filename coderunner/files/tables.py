"""Tables and Office files as text that search and the assistant can use.

A table (a CSV or TSV file, a spreadsheet's sheet) becomes one line saying what it holds (its rows, its columns,
the total of each column of numbers), then each row with its columns named ("Name: Ada; Amount: 120"). As flat
text, a row of numbers under long column titles matched no question. Excel and PowerPoint files are read from
their parts (zipped XML); an iWork file through the preview it carries (a PDF of the whole document in files
from before 2013, a picture of its first page since).
"""

from __future__ import annotations

import csv
import io
import pathlib
import re
import xml.etree.ElementTree as ET
import zipfile

NUMBER = re.compile(r"^\d+(\.\d+)?$")
CURRENCY = re.compile(r"^[\(\-−]?\s*([A-Z]{3}|[$€£¥₹])?\s*|\s*([A-Z]{3}|[$€£¥₹])?\)?$")
TITLE_CUT = 60


def number(cell: str) -> float | None:
    """A cell as a number: digits with grouping commas, a sign, a currency sign or code before or after."""
    text = cell.strip()
    if not text or len(text) >= 40:
        return None
    negative = text.startswith("(") and text.endswith(")")
    stripped = CURRENCY.sub("", text).replace(",", "")
    if not NUMBER.match(stripped):
        return None
    value = float(stripped)
    signed = text.startswith("-") or text.startswith("−") or negative
    return -value if signed else value


def _format(value: float) -> str:
    return str(int(value)) if value == int(value) and abs(value) < 1e15 else f"{value:.2f}"


class Table:
    def __init__(self, header: list[str], rows: list[list[str]]):
        self.header = header
        self.rows = rows

    @classmethod
    def from_rows(cls, rows: list[list[str]]) -> "Table | None":
        """From rows, the first naming the columns when it is not numbers. None with fewer than two columns or no
        rows under the header."""
        rows = [row for row in rows if any(cell.strip() for cell in row)]
        if len(rows) < 2 or max(len(r) for r in rows) < 2:
            return None
        width = max(len(r) for r in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        first = [cell.strip() for cell in padded[0]]
        if any(cell and number(cell) is None for cell in first):
            header = [cell or f"Column {i + 1}" for i, cell in enumerate(first)]
            return cls(header, padded[1:])
        return cls([f"Column {i + 1}" for i in range(width)], padded)

    @classmethod
    def parse(cls, text: str, separator: str = ",") -> "Table | None":
        """A delimited text table, with quoted cells as spreadsheets write them."""
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=separator))
        except csv.Error:
            return None
        return cls.from_rows(rows)

    def described(self, name: str) -> str:
        """The table as text: one line saying what it holds, then its rows with their columns named. A column
        title longer than a short phrase is cut in the rows (whole in the first line)."""
        totals = []
        for index, title in enumerate(self.header):
            cells = [row[index].strip() for row in self.rows if row[index].strip()]
            numbers = [number(c) for c in cells]
            if cells and len(cells) > 1 and all(n is not None for n in numbers):
                totals.append(f"{title} {_format(sum(numbers))}")
        summed = " Totals: " + "; ".join(totals) + "." if totals else ""
        lines = [f"{name}: a table of {len(self.rows)} rows and {len(self.header)} columns: "
                 f"{'; '.join(self.header)}.{summed}"]
        titles = [t if len(t) <= TITLE_CUT else t[:TITLE_CUT - 3] + "…" for t in self.header]
        for row in self.rows:
            pairs = [f"{title}: {value.strip()}" for title, value in zip(titles, row) if value.strip()]
            if pairs:
                lines.append("; ".join(pairs))
        return "\n".join(lines)


# ---- Office and iWork files --------------------------------------------------------

def _ordinal(name: str) -> int:
    digits = re.sub(r"\D", "", name)
    return int(digits) if digits else 0


def _strip(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _texts(data: bytes, element: str) -> list[str]:
    """The text of each `element` (its runs joined), in document order."""
    found: list[str] = []
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return found
    for node in root.iter():
        if _strip(node.tag) == element:
            found.append("".join(t for t in node.itertext()))
    return found


def slides(path: pathlib.Path) -> list[tuple[int, str]]:
    """Each slide's words, in order."""
    found: list[tuple[int, str]] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted((n for n in archive.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")),
                       key=_ordinal)
        for name in names:
            try:
                root = ET.fromstring(archive.read(name))
            except ET.ParseError:
                continue
            words = " ".join(node.text for node in root.iter() if _strip(node.tag) == "t" and node.text)
            if words.strip():
                found.append((_ordinal(name), words))
    return found


def _reference(text: str) -> tuple[int, int]:
    """"AB12" as column 27 (from 0) and row 12."""
    column, digits = 0, ""
    for character in text:
        if character.isalpha():
            column = column * 26 + (ord(character.upper()) - 64)
        elif character.isdigit():
            digits += character
    return max(column - 1, 0), int(digits or 0)


def sheets(path: pathlib.Path) -> list[tuple[str, Table]]:
    """Each sheet as a table, with its name."""
    found: list[tuple[str, Table]] = []
    with zipfile.ZipFile(path) as archive:
        names = sorted((n for n in archive.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")),
                       key=_ordinal)
        shared = _texts(archive.read("xl/sharedStrings.xml"), "si") if "xl/sharedStrings.xml" in archive.namelist() else []
        titles: list[str] = []
        if "xl/workbook.xml" in archive.namelist():
            try:
                titles = [node.get("name", "") for node in ET.fromstring(archive.read("xl/workbook.xml")).iter()
                          if _strip(node.tag) == "sheet"]
            except ET.ParseError:
                titles = []
        for index, name in enumerate(names):
            try:
                root = ET.fromstring(archive.read(name))
            except ET.ParseError:
                continue
            cells: dict[int, dict[int, str]] = {}
            for node in root.iter():
                if _strip(node.tag) != "c":
                    continue
                column, row = _reference(node.get("r", ""))
                kind = node.get("t", "")
                value = ""
                for child in node:
                    tag = _strip(child.tag)
                    if tag == "v" or (tag == "is"):
                        value = "".join(child.itertext())
                if kind == "s":
                    try:
                        value = shared[int(value)]
                    except (ValueError, IndexError):
                        value = ""
                if value:
                    cells.setdefault(row, {})[column] = value
            if not cells:
                continue
            width = max(max(columns) for columns in cells.values()) + 1
            rows = [[cells[r].get(c, "") for c in range(width)] for r in sorted(cells)]
            table = Table.from_rows(rows)
            if table is not None:
                found.append((titles[index] if index < len(titles) else f"Sheet {index + 1}", table))
    return found


def iwork_preview(path: pathlib.Path) -> tuple[str, bytes] | None:
    """An iWork file's preview: ("pdf", bytes) for a PDF of the whole document, ("image", bytes) for a picture
    of its first page, None when it has neither. A package is a folder; a newer file, a zip."""
    if path.is_dir():
        pdf, picture = path / "QuickLook" / "Preview.pdf", path / "preview.jpg"
        if pdf.is_file():
            return "pdf", pdf.read_bytes()
        if picture.is_file():
            return "image", picture.read_bytes()
        return None
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            if "QuickLook/Preview.pdf" in names:
                return "pdf", archive.read("QuickLook/Preview.pdf")
            if "preview.jpg" in names:
                return "image", archive.read("preview.jpg")
    except (zipfile.BadZipFile, OSError):
        return None
    return None
