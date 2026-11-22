"""Finding files by what they are about.

Hybrid search: meaning and words, fused by reciprocal rank (Cormack, Clarke & Büttcher, SIGIR 2009; k = 60),
with no weights or thresholds to tune.

Meaning: the question is embedded and compared with every piece (vectors.VectorIndex: binary codes, then exact
scores for each modality's closest pieces); a file ranks by its most similar piece. EmbeddingGemma 2's space is
aligned across media, so plain cosine similarity compares a passage with a picture or a sound directly
(STANDARDISE reads each score as standard deviations above its modality's average instead, the earlier rule).
Code passages are compared with the question in the model's code-retrieval form.

Words: BM25 (SQLite's full-text index) over the pieces holding *every* word of the question, as keyword search
conventionally means: it finds what an embedding cannot read (invoice numbers, serials, codes) and stays quiet
for a question in plain language, where matching any one word ("photo" in a file name) would outvote the
meaning. Names: the same over each file's or folder's own name and kind, so a name holding every word counts
once more ("copperleaf" opens the Copperleaf folder, not the notes inside that mention it).

The kind of file the question names ("photo of…", "the slides about…") is one more ranking by meaning over
files of that kind alone. Folders are results too: found by their names, by the mean similarity of the files
under them, and shown after the first of the results that gather in them (first, when the question asks for a
group of things). Exact copies of a file count once; a document saved twice and changed since, found by the
same passage, counts as a version of the first.

Measured on ori's bench corpus (389 files with noise, 108 file questions, 87 folder questions) and on the
owner's real folders: see VALIDATION.md.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from typing import Any

from .discover import content_fingerprint, in_cloud_only
from .extract import NAME_ONLY
from .kinds import is_code
from .vectors import VectorIndex, dot, unpack

RANKER = "hybrid"           # "hybrid": meaning, words and names fused by rank; "meaning": meaning plus the kind bonus
STANDARDISE = False         # z-score each similarity within its modality (the rule before 2026-11)
CODE_PROMPT = False         # compare code passages with the question in the model's code-retrieval form: under plain
                            # cosine every code passage then scored 0.75-0.80 against any question at all, which put
                            # a script above the garlic note, the penguin video and the zebra photo
K = 60.0                    # reciprocal rank fusion constant
DEPTH = 100                 # files taken from the meaning ranking into the fusion
CANDIDATES = 300            # pieces of each modality scored exactly after the binary first pass
HINT = 1.0                  # standard deviations added for the named kind (meaning ranker)
WORD_DEPTH = 200            # pieces read from the BM25 ranking
SMALL = 50                  # with fewer pieces than this, modalities cannot be standardised; raw similarity then
SNIPPET = 240
LONGEST_WORD = 64           # a run this long without a space is not a word (base64, a hash, a tracking URL)
MAX_COPIES = 5              # other places of a copied file named in its result
CLUSTER_SHARE = 0.25        # a folder is shown among the first ten when they are this share of its own files
COUNT_CACHE = 600.0         # seconds the per-folder file counts are kept
WORD = re.compile(r"\w+", re.UNICODE)
STOPWORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "from", "by", "with", "and", "or", "my", "me",
             "i", "is", "are", "was", "were", "it", "this", "that", "what", "where", "when", "which", "who",
             "about", "show", "find", "get", "some", "any", "all", "do", "did", "have", "has", "had"}
KIND_WORDS = {
    "photo": "image", "photos": "image", "picture": "image", "pictures": "image", "pic": "image", "pics": "image",
    "image": "image", "images": "image", "screenshot": "image", "screenshots": "image", "selfie": "image",
    "video": "video", "videos": "video", "clip": "video", "clips": "video", "movie": "video", "footage": "video",
    "audio": "audio", "song": "audio", "songs": "audio", "music": "audio", "track": "audio", "podcast": "audio",
    "recording": "audio", "voice": "audio", "pdf": "doc", "document": "doc", "documents": "doc", "letter": "doc",
    "contract": "doc", "note": "text", "notes": "text",
}
# Words that only say what kind of thing is wanted, left out of a name search that found nothing ("costa rica
# pics": no folder's name says "pics").
FILLER = {"photo", "photos", "pic", "pics", "picture", "pictures", "image", "images", "screenshot", "screenshots",
          "video", "videos", "clip", "clips", "movie", "movies", "recording", "recordings", "audio", "song", "songs",
          "doc", "docs", "document", "documents", "slide", "slides", "deck", "decks", "pdf", "pdfs", "spreadsheet",
          "spreadsheets", "file", "files", "folder", "folders", "stuff", "things", "my"}
DESCRIBING = {"that", "which", "who", "whose", "where", "when", "for", "with", "without", "in", "about", "to", "of",
              "from", "on", "at", "by", "showing", "shown", "containing"}
LEADING = {"a", "an", "the", "my", "our", "some", "any", "find", "show", "me"}
COLLECTIONS = {"stuff", "paperwork", "junk", "pile", "collection", "folder", "folders", "everything"}
_VIDEO = re.compile(r"\b(?:videos?|clips?|screen ?recordings?|screencasts?|movies?|footage)\b", re.I)
_PICTURE = re.compile(r"\b(?:photos?|photographs?|pictures?|pics?|images?|screenshots?|screen ?shots?|selfies?|logos?"
                      r"|diagrams?|drawings?|illustrations?|memes?)\b", re.I)
_SCREEN = re.compile(r"\bscreens?\b(?! ?(?:recordings?|casts?|captures? video))", re.I)
_DOCUMENT = re.compile(r"\b(?:slides?|slide ?decks?|decks?|presentations?|spreadsheets?|workbooks?|pdfs?)\b", re.I)
_CODE = re.compile(r"\b(?:scripts?|source(?: code)?|code|functions?|implementations?)\b", re.I)
_SCAN = re.compile(r"\b(?:scans?|scanned)\b", re.I)
_SOUND = re.compile(r"\b(?:voice (?:memos?|notes?|messages?)|audio|podcasts?|songs?|sound (?:clips?|recordings?)"
                    r"|recording of (?:a |my )?(?:call|talk|meeting|lecture))\b", re.I)
_MAIL = re.compile(r"\b(?:e-?mails?|mails?)\b", re.I)
_MESSAGE = re.compile(r"\b(?:texts?|text messages?|i?messages?|chats?|whatsapp)\b", re.I)
_NOTE = re.compile(r"\b(?:notes?)\b", re.I)
_EVENT = re.compile(r"\b(?:events?|meetings?|appointments?|calendar)\b", re.I)


def terms(query: str) -> list[str]:
    words = list(dict.fromkeys(w.lower() for w in WORD.findall(query)))
    kept = [w for w in words if w not in STOPWORDS]
    return kept or words


def kind_hints(query: str) -> set[str]:
    """Every kind a question mentions anywhere (the meaning ranker's bonus)."""
    return {KIND_WORDS[w] for w in WORD.findall(query.lower()) if w in KIND_WORDS}


def _head(query: str) -> list[str]:
    """The thing asked for: the words before the first that starts describing it ("photo of a zebra", "safari
    trip photos", "the clip that says"), leading articles dropped."""
    words = [w for w in re.split(r"[^\w]+", query.lower()) if w]
    while words and words[0] in LEADING:
        words.pop(0)
    head = []
    for word in words:
        if word in DESCRIBING:
            break
        head.append(word)
    return head


def named_kinds(query: str) -> set[str]:
    """The kinds of file a question asks for by name: a photo, a screenshot or a scan is a picture (a scan can
    be a document too), a video or a screen recording is a video, a voice memo is audio, slides are a document,
    a script is code. Only what is asked for counts, not what describes it ("a script that blurs names in
    screenshots" asks for code)."""
    head = " ".join(_head(query)[-2:])
    kinds: set[str] = set()
    if _VIDEO.search(head):
        kinds.add("video")
    if _PICTURE.search(head) or _SCREEN.search(head):
        kinds.add("image")
    if _SCAN.search(head):
        kinds.update({"image", "doc"})
    if _SOUND.search(head):
        kinds.add("audio")
    if _DOCUMENT.search(head):
        kinds.add("doc")
    if _CODE.search(head):
        kinds.add("code")
    if _MAIL.search(head):
        kinds.add("mail")
    if _MESSAGE.search(head):
        kinds.add("message")
    if _NOTE.search(head):
        kinds.update({"text", "note"})
    if _EVENT.search(head):
        kinds.add("event")
    return kinds


def asks_for_many(query: str) -> bool:
    """Whether a question asks for a group of things: its last word before any describing one is plural, or a
    word for a collection ("stuff", "paperwork"), and nothing narrows them down but what they are of. "Flower
    photos" and "photos of musical instruments" are a group; "notes for the call with the roaster" and "videos
    with birds in them" are particular files among others."""
    words = [w for w in re.split(r"[^\w]+", query.lower()) if w]
    if any(w != "of" and w in DESCRIBING for w in words):
        return False
    head = _head(query)
    if not head:
        return False
    last = head[-1]
    if last in COLLECTIONS:
        return True
    return len(last) >= 4 and last.endswith("s") and not last.endswith(("ss", "us", "is"))


def is_lookup(words: list[str]) -> bool:
    """Codes and numbers ("K7Q2", "4815162342") mean nothing to an embedding and everything to keywords."""
    return bool(words) and all(any(c.isdigit() for c in w) and (any(c.isalpha() for c in w) or len(w) >= 4)
                               for w in words)


def of_kind(path: str, kind: str, kinds: set[str]) -> bool:
    """Whether a file of `kind` (or code) is one of `kinds`."""
    return kind in kinds or ("code" in kinds and is_code(path))


def snippet(text: str, words: list[str]) -> str:
    """About SNIPPET characters around the first query word, on whole words; no base64 or hashes."""
    pieces = [w for w in text.split() if len(w) <= LONGEST_WORD]
    if not pieces:
        return ""
    lowered = [w.lower() for w in pieces]
    hit = next((i for i, w in enumerate(lowered) if any(q in w for q in words)), 0)
    first = max(0, hit - 8)
    length, last = 0, first
    while last < len(pieces) and length + len(pieces[last]) + 1 <= SNIPPET:
        length += len(pieces[last]) + 1
        last += 1
    body = " ".join(pieces[first:max(last, first + 1)])
    return ("…" if first > 0 else "") + body + ("…" if last < len(pieces) else "")


def edit_distance(a: str, b: str, most: int) -> int:
    """Levenshtein distance, given up past `most`."""
    row = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        nxt = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            nxt[j] = min(row[j] + 1, nxt[j - 1] + 1, row[j - 1] + (a[i - 1] != b[j - 1]))
        if min(nxt) > most:
            return most + 1
        row = nxt
    return row[len(b)]


def nearest_word(term: str, words: list[tuple[str, int]]) -> str | None:
    """The word nearest a misspelt one: starting with the same letter, one edit away (two for a word of eight
    letters or more), the one most names hold first. None for a word under five letters, or none near."""
    if len(term) < 5 or not term.isalpha():
        return None
    most = 2 if len(term) >= 8 else 1
    best: tuple[str, int, int] | None = None
    for word, count in words:
        if not word or word[0] != term[0] or abs(len(word) - len(term)) > most:
            continue
        distance = edit_distance(term, word, most)
        if 0 < distance <= most and (best is None or distance < best[1] or (distance == best[1] and count > best[2])):
            best = (word, distance, count)
    return best[0] if best else None


class Searcher:
    def __init__(self, store, vectors: VectorIndex, embedder=None):
        self.store = store
        self.vectors = vectors
        self.embedder = embedder
        self._fingerprints: dict[tuple, bytes] = {}
        self._fingerprint_lock = threading.Lock()
        self._counts: tuple[dict[str, int], float] | None = None
        self._name_words: tuple[list[tuple[str, int]], float] | None = None

    async def search(self, query: str, kind: str | None = None, limit: int = 8) -> list[dict]:
        query = (query or "").strip()
        if not query:
            return []
        limit = max(1, min(int(limit), 30))
        kinds = {kind} if kind else None
        words = terms(query)
        lookup = is_lookup(query.split())        # "HX-55821" as typed, not split into hx and 55821
        meaning, lexical, named = await asyncio.gather(
            self._meaning(query, kinds) if not lookup else asyncio.sleep(0, {}),
            asyncio.to_thread(self._words, words, kinds),
            asyncio.to_thread(self._names, words, kinds) if RANKER == "hybrid" else asyncio.sleep(0, []))
        paths = set(meaning) | {p for p, _ in lexical} | {p for p, _ in named}
        if not paths:
            return []
        files = {r["path"]: r for r in self._rows(
            "SELECT path, kind, size, holds FROM files WHERE path IN ({})", list(paths))}
        if kinds:
            meaning = {p: m for p, m in meaning.items() if files.get(p, {}).get("kind") in kinds}
            lexical = [(p, c) for p, c in lexical if files.get(p, {}).get("kind") in kinds]
            named = [(p, c) for p, c in named if files.get(p, {}).get("kind") in kinds]
        score: dict[str, float] = {}
        chosen: dict[str, int] = {}
        strongest: dict[str, float] = {}

        def add(path: str, value: float, chunk: int) -> None:
            score[path] = score.get(path, 0.0) + value
            if value > strongest.get(path, float("-inf")):
                strongest[path] = value
                chosen[path] = chunk

        if RANKER == "hybrid":
            best = {path: similarity for path, (similarity, _) in meaning.items()}
            if meaning:
                best = self._with_folder_contents(best, files)
            by_meaning = sorted(best, key=lambda p: (-best[p], p))
            for offset, path in enumerate(by_meaning):
                add(path, 1 / (K + offset + 1), meaning[path][1] if path in meaning else -1)
            asked = named_kinds(query)
            if asked and not lookup:
                # The kind the question names: the same ranking by meaning, of files of that kind only, as one more
                # ranking; what the names found leads it ("costa rica pics" is the Costa Rica folder of photos).
                by_name = [p for p, _ in named]
                candidates = by_name + [p for p in by_meaning[:DEPTH] if p not in set(by_name)]
                for offset, path in enumerate(p for p in candidates if self._is_kind(p, files, asked)):
                    add(path, 1 / (K + offset + 1), meaning[path][1] if path in meaning else
                        next((c for q, c in named if q == path), -1))
            for offset, (path, chunk) in enumerate(lexical):
                add(path, 1 / (K + offset + 1), chunk)
            for offset, (path, chunk) in enumerate(named):
                add(path, 1 / (K + offset + 1), chunk)
            ranked = sorted(score, key=lambda p: (-score[p], p))
            if not kinds or "folder" in kinds:
                ranked = self._with_clusters(ranked, files, asks_for_many(query))
        else:
            hints = kind_hints(query)
            for path, (z, chunk) in meaning.items():
                add(path, z + (HINT if files.get(path, {}).get("kind") in hints else 0.0), chunk)
            ranked = sorted(score, key=lambda p: (-score[p], p))
            # Then what only words find, in BM25 order: everything for a code, else files with no embedded piece.
            for path, chunk in lexical:
                if path not in chosen:
                    ranked.append(path)
                    chosen[path] = chunk
        ranked, copies, versions = await asyncio.to_thread(self._collapse, ranked, files, chosen, limit)
        return self._results(ranked, chosen, files, words, copies, versions)

    # ---- meaning ---------------------------------------------------------------

    async def _meaning(self, query: str, kinds: set[str] | None) -> dict[str, tuple[float, int]]:
        """Each file's best similarity (standardised within its piece's modality when STANDARDISE) and the piece
        that has it."""
        if self.embedder is None:
            return {}
        await self.vectors.ready()
        if not len(self.vectors):
            return {}
        code_prefix = getattr(self.embedder, "code_prefix", "") if CODE_PROMPT else ""
        if code_prefix:
            vector, code = await self.embedder.queries(query, [self.embedder.query_prefix, code_prefix])
        else:
            vector, code = await self.embedder.query(query), None
        chunks = self.vectors.nearest(vector, CANDIDATES, kinds)
        if code is not None:
            chunks = list(dict.fromkeys([*chunks, *self.vectors.nearest(code, CANDIDATES, kinds)]))
        if not chunks:
            return {}
        standardise = STANDARDISE and len(self.vectors) >= SMALL
        calibrations = {"text": self.vectors.calibration(vector)} if standardise else {}
        if standardise and code is not None:
            calibrations["code"] = self.vectors.calibration(code)
        best: dict[str, tuple[float, int]] = {}
        for row in self._rows("SELECT id, path, embedding, modality FROM file_chunks WHERE id IN ({})", chunks):
            stored = unpack(row["embedding"])
            if len(stored) != len(vector):
                continue
            modality = row["modality"] or "text"
            as_code = code is not None and modality == "text" and is_code(row["path"])
            similarity = dot(code if as_code else vector, stored)
            if standardise:
                means, spread = calibrations["code" if as_code else "text"]
                similarity = (similarity - means.get(modality, 0.0)) / spread
            if similarity > best.get(row["path"], (float("-inf"), 0))[0]:
                best[row["path"]] = (similarity, row["id"])
        return best

    def _with_folder_contents(self, best: dict[str, float], files: dict) -> dict[str, float]:
        """A folder is as close to the question as the files under it are on average: "safari trip photos" is the
        Kenya folder, every photo in it of a safari, though its name never says so. A file the meaning list did
        not reach counts as its lowest similarity, so a folder of mostly other things stays low; an average never
        passes the folder's best file, so a question about one file still finds the file first. Folders whose
        mean falls below the first DEPTH files by meaning are left out."""
        if not best:
            return best
        floor = min(best.values())
        found: dict[str, list[float]] = {}
        for path, value in best.items():
            if files.get(path, {}).get("kind") == "folder":
                continue
            folder = os.path.dirname(path)
            while folder and folder != "/":
                found.setdefault(folder, []).append(value)
                parent = os.path.dirname(folder)
                if parent == folder:
                    break
                folder = parent
        folders = {f: v for f, v in found.items() if len(v) >= 2}
        if not folders:
            return best
        counts = self._subtree_counts()
        scores = sorted(best.values(), reverse=True)
        least = scores[DEPTH - 1] if len(scores) > DEPTH else float("-inf")
        result = dict(best)
        for folder, values in folders.items():
            total = counts.get(folder)
            if total is None or total < len(values):
                continue
            mean = (sum(values) + (total - len(values)) * floor) / total
            if mean > result.get(folder, float("-inf")) and mean >= least:
                result[folder] = mean
                files.setdefault(folder, {"path": folder, "kind": "folder", "size": 0, "holds": ""})
        return result

    def _subtree_counts(self) -> dict[str, int]:
        """How many files each indexed folder holds at any depth, from one pass over the index once in COUNT_CACHE
        seconds."""
        now = time.monotonic()
        if self._counts is not None and now - self._counts[1] < COUNT_CACHE:
            return self._counts[0]
        counts: dict[str, int] = {}
        folders: set[str] = set()
        with self.store.reader() as db:
            for path, kind in db.execute("SELECT path, kind FROM files"):
                if kind == "folder":
                    folders.add(path)
                    continue
                folder = os.path.dirname(path)
                while folder and folder != "/":
                    counts[folder] = counts.get(folder, 0) + 1
                    parent = os.path.dirname(folder)
                    if parent == folder:
                        break
                    folder = parent
        counts = {f: n for f, n in counts.items() if f in folders}
        self._counts = (counts, now)
        return counts

    def _direct_counts(self, folders: list[str]) -> dict[str, int]:
        """How many files (not folders) each folder holds directly."""
        counts: dict[str, int] = {}
        for folder in folders:
            rows = self.store.query("SELECT path FROM files WHERE path >= ? AND path < ? AND kind != 'folder'",
                                    (folder + "/", folder + "0"))
            counts[folder] = sum(1 for r in rows if "/" not in r["path"][len(folder) + 1:])
        return counts

    def _is_kind(self, path: str, files: dict, asked: set[str]) -> bool:
        row = files.get(path) or {}
        if row.get("kind") == "folder":
            holds = set((row.get("holds") or "").split())
            return bool(holds & asked) or ("code" in asked and "code" in holds)
        return of_kind(path, row.get("kind", ""), asked)

    # ---- words and names -------------------------------------------------------------

    def _words(self, words: list[str], kinds: set[str] | None) -> list[tuple[str, int]]:
        """Files in BM25 order (SQLite FTS5) by their best piece holding every word, with that piece. Ranked inside
        the full-text index first and only then joined to the pieces: joined in one query, SQLite read the piece
        behind every match (6 s for a question whose words matched 130,000 of 332,000 pieces)."""
        if not words:
            return []
        joiner = " AND " if RANKER == "hybrid" else " OR "
        expression = joiner.join(f'"{w}"' for w in words[:16])
        depth = WORD_DEPTH * (4 if kinds else 1) if RANKER == "hybrid" else 1000
        ranked = [r["id"] for r in self.store.query(
            "SELECT rowid AS id FROM file_chunks_fts WHERE file_chunks_fts MATCH ? "
            "ORDER BY bm25(file_chunks_fts) LIMIT ?", (expression, depth))]
        owner = {r["id"]: r["path"] for r in self._rows("SELECT id, path FROM file_chunks WHERE id IN ({})", ranked)}
        found: dict[str, int] = {}
        for chunk in ranked:
            if chunk in owner:
                found.setdefault(owner[chunk], chunk)
        return list(found.items())

    def _names(self, words: list[str], kinds: set[str] | None) -> list[tuple[str, int]]:
        """Files and folders whose own name (and kind) holds every word, best BM25 first. When none does: folders
        named with the folder they are in ("audio downloads"); then the words that only say what kind of file is
        wanted left out ("costa rica pics"); then a misspelt word taken as the nearest name word ("personel")."""
        if not words:
            return []
        found = self._name_pieces(words, kinds)
        if not found and len(words) > 1:
            found = self._name_pieces(words, kinds, with_parent=True, folders_only=True)
        rest = words
        if not found:
            rest = [w for w in words if w not in FILLER]
            if rest and len(rest) < len(words):
                found = self._name_pieces(rest, kinds)
            else:
                rest = words
        if not found and rest:
            vocabulary = self._words_of_names()
            corrected = [w if self._name_holds(w) else (nearest_word(w, vocabulary) or w) for w in rest]
            if corrected != rest:
                found = self._name_pieces(corrected, kinds)
        return found

    def _name_pieces(self, words: list[str], kinds: set[str] | None, *, with_parent: bool = False,
                     folders_only: bool = False) -> list[tuple[str, int]]:
        expression = " AND ".join(f'"{w}"' for w in words[:16])
        depth = WORD_DEPTH * (4 if kinds else 1)
        if with_parent:
            rows = self.store.query(
                "SELECT c.id, c.path, f.kind, bm25(file_chunks_fts) AS rank FROM file_chunks_fts "
                "JOIN file_chunks c ON c.id = file_chunks_fts.rowid JOIN files f ON f.path = c.path "
                "WHERE file_chunks_fts MATCH ? AND c.modality = 'name' AND f.kind = 'folder' ORDER BY rank LIMIT ?",
                (expression, depth))
            return [(r["path"], r["id"]) for r in rows]
        rows = self.store.query(
            "SELECT rowid AS id, bm25(file_names) AS rank FROM file_names WHERE file_names MATCH ? ORDER BY rank LIMIT ?",
            (expression, depth))
        owner = {r["id"]: (r["path"], r["kind"]) for r in self._rows(
            "SELECT c.id, c.path, f.kind FROM file_chunks c JOIN files f ON f.path = c.path WHERE c.id IN ({})",
            [r["id"] for r in rows])}
        found: dict[str, int] = {}
        for row in rows:
            if row["id"] in owner:
                path, kind = owner[row["id"]]
                if folders_only and kind != "folder":
                    continue
                found.setdefault(path, row["id"])
        return list(found.items())

    def _name_holds(self, word: str) -> bool:
        return self.store.one("SELECT rowid FROM file_names WHERE file_names MATCH ? LIMIT 1", (f'"{word}"',)) is not None

    def _words_of_names(self) -> list[tuple[str, int]]:
        """The words of every name, with how many names hold each; read once in COUNT_CACHE seconds."""
        now = time.monotonic()
        if self._name_words is not None and now - self._name_words[1] < COUNT_CACHE:
            return self._name_words[0]
        try:
            self.store.execute("CREATE VIRTUAL TABLE IF NOT EXISTS temp.file_name_words USING fts5vocab(main, "
                               "'file_names', 'row')")
            words = [(r["term"], r["doc"]) for r in self.store.query("SELECT term, doc FROM temp.file_name_words")]
        except Exception:
            words = []
        self._name_words = (words, now)
        return words

    # ---- folders the results gather in ---------------------------------------------

    def _with_clusters(self, ordered: list[str], files: dict, many: bool) -> list[str]:
        """Each indexed folder holding at least two of the first ten results, a quarter of what it holds or more,
        moved up to just after the first of them (if it is not higher already), or ahead of them when the
        question asks for a group of things."""
        top = ordered[:10]
        counts: dict[str, int] = {}
        for path in top:
            if files.get(path, {}).get("kind") != "folder":
                folder = os.path.dirname(path)
                counts[folder] = counts.get(folder, 0) + 1
        clusters = [f for f, n in counts.items() if n >= 2]
        if not clusters:
            return ordered
        held = self._direct_counts(clusters)
        clusters = [f for f in clusters if counts[f] >= CLUSTER_SHARE * held.get(f, 0)]
        folders = {r["path"]: r for r in self._rows("SELECT path, kind, size, holds FROM files WHERE kind = 'folder' "
                                                     "AND path IN ({})", clusters)}
        if not folders:
            return ordered
        result = list(ordered)
        for folder in sorted(folders):
            files.setdefault(folder, folders[folder])
            inside = [i for i, p in enumerate(result) if os.path.dirname(p) == folder]
            if not inside:
                continue
            first = inside[0]
            place = next((i for i in inside if files.get(result[i], {}).get("kind") != "folder"), first) \
                if many else first + 1
            if folder in result:
                at = result.index(folder)
                if at <= place:
                    continue
                result.pop(at)
            result.insert(min(place, len(result)), folder)
        return result

    # ---- copies and versions -------------------------------------------------------------

    def _collapse(self, ranked: list[str], files: dict, chosen: dict[str, int], limit: int):
        """Copies count once, the best-ranked standing for the others: exact copies ("file (1).pdf", a second
        checkout of a repository), and a document saved or exported twice and changed since, found by the same
        passage, and a folder of the same name holding the same files. Shown once, the others under "copies" and
        "versions"."""
        kept: list[str] = []
        copies: dict[str, list[str]] = {}
        versions: dict[str, list[str]] = {}
        sizes: dict[int, list[str]] = {}
        named: dict[str, list[str]] = {}
        considered = ranked[:limit * 4]
        passages = {r["id"]: (r["part"], r["text"]) for r in self._rows(
            "SELECT id, part, text FROM file_chunks WHERE id IN ({})", [chosen[p] for p in considered if chosen.get(p, -1) >= 0])}

        def key(path: str) -> str:
            return files.get(path, {}).get("kind", "") + "/" + os.path.basename(path)

        def same_passage(a: str, b: str) -> bool:
            x, y = passages.get(chosen.get(a, -1)), passages.get(chosen.get(b, -1))
            return bool(x and y and x[1] and x == y)

        for path in ranked:
            row = files.get(path)
            if row is None:
                continue
            if row.get("kind") == "folder":
                twin = next((k for k in named.get(key(path), []) if self._same_folder(k, path)), None)
                if twin:
                    versions.setdefault(twin, []).append(path)
                elif len(kept) < limit:
                    kept.append(path)
                    named.setdefault(key(path), []).append(path)
                continue
            size = row.get("size")
            same_name = named.get(key(path), [])
            if len(kept) >= limit and size not in sizes and not same_name:
                continue
            twin = next((k for k in sizes.get(size, []) if self._identical(k, path)), None)
            if twin:
                copies.setdefault(twin, []).append(path)
            elif (twin := next((k for k in same_name if same_passage(k, path)), None)):
                versions.setdefault(twin, []).append(path)
            elif len(kept) < limit:
                kept.append(path)
                sizes.setdefault(size, []).append(path)
                named.setdefault(key(path), []).append(path)
        return kept, copies, versions

    def _same_folder(self, a: str, b: str) -> bool:
        """Two folders of one name holding as many files under them, and the same names directly in them."""
        counts = self._subtree_counts()
        if counts.get(a) != counts.get(b):
            return False

        def names(folder: str) -> set[str]:
            rows = self.store.query("SELECT path FROM files WHERE path >= ? AND path < ? LIMIT 5000",
                                    (folder + "/", folder + "0"))
            return {r["path"][len(folder) + 1:] for r in rows if "/" not in r["path"][len(folder) + 1:]}
        left = names(a)
        return bool(left) and left == names(b)

    def _identical(self, a: str, b: str) -> bool:
        return (fingerprint := self._fingerprint(a)) is not None and fingerprint == self._fingerprint(b)

    def _fingerprint(self, path: str) -> bytes | None:
        """The file's content fingerprint, cached by identity and change time (a file replaced with other bytes
        of the same size keeps its mtime after cp -p or an unzip; its inode or change time does not)."""
        try:
            stat = os.stat(path)
        except OSError:
            return None
        if in_cloud_only(stat):
            return None
        key = (path, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        with self._fingerprint_lock:
            known = self._fingerprints.get(key)
        if known is not None:
            return known
        fingerprint = content_fingerprint(path)
        if fingerprint is not None:
            with self._fingerprint_lock:              # searches run this in threads of their own
                self._fingerprints[key] = fingerprint
                while len(self._fingerprints) > 4096:
                    self._fingerprints.pop(next(iter(self._fingerprints)))
        return fingerprint

    # ---- rows and results --------------------------------------------------------------

    def _rows(self, sql: str, values: list[Any]) -> list[dict]:
        found: list[dict] = []
        for start in range(0, len(values), 500):               # SQLite caps the number of parameters
            part = values[start:start + 500]
            found += self.store.query(sql.format(",".join("?" * len(part))), part)
        return found

    def _results(self, ranked, chosen, files, words, copies, versions) -> list[dict]:
        pieces = {r["id"]: r for r in self._rows("SELECT id, part, text, loc, modality FROM file_chunks "
                                                 "WHERE id IN ({})", [chosen[p] for p in ranked if chosen.get(p, -1) >= 0])}
        results = []
        for path in ranked:
            piece = pieces.get(chosen.get(path, -1)) or {}
            part = piece.get("part") or ("folder" if files.get(path, {}).get("kind") == "folder" else "")
            found: dict[str, Any] = {"path": path, "kind": files.get(path, {}).get("kind", ""), "part": part}
            if piece.get("loc") is not None:
                place = "page" if part.startswith("page") else "line" if part.startswith("from line") else "at"
                found[place] = int(piece["loc"]) if place != "at" else round(piece["loc"], 1)
            if piece.get("modality") == "text" and piece.get("text"):
                found["snippet"] = snippet(piece["text"], words)
            if part.startswith(NAME_ONLY):
                found["in_icloud"] = True
            if copies.get(path):
                found["copies"] = copies[path][:MAX_COPIES]
            if versions.get(path):
                found["versions"] = versions[path][:MAX_COPIES]
            results.append(found)
        return results
