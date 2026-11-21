"""The owner's local files as something the assistant can find and use.

The owner chooses folders (`coderunner files add ~/Documents`). A background indexer embeds what is in
them with a multimodal embedding model: through Ollama, EmbeddingGemma 2 places names, text, pictures and
sound in one space, video is indexed as frames plus its sound, and text (passages, PDF pages, text read
from pictures) also goes into a keyword index. Nothing leaves the machine. File contents are the owner's
private data and may also be third-party text (downloads, attachments), so the tools that return them mark
the task as holding private data and as having read untrusted content.

  kinds     what kind of file a path is; how embedding models want text phrased
  discover  the shared folders, and what inside them may be indexed and read
  extract   a file's pieces: name, passages, pages, pictures, text in pictures, sound, transcripts
  embed     the embedding model (batched, kept loaded)
  vectors   half-precision vectors on disk, sign codes in memory, per-modality calibration
  search    meaning and words together, kind hints, weak results cut
  index     the indexer that ties these together and keeps the index current
  watch     file events (macOS FSEvents)
  power     waiting on a low battery or in Low Power Mode
"""

from .discover import FileAccessError, Folders, in_cloud_only, is_secret
from .embed import Embedder
from .extract import NAME_ONLY, Extractor, Media, Piece, Skipped
from .index import FileIndex
from .kinds import PROFILES, embedding_profile, kind_of
from .vectors import pack, unpack

__all__ = ["Embedder", "Extractor", "FileAccessError", "FileIndex", "Folders", "Media", "NAME_ONLY", "PROFILES",
           "Piece", "embedding_profile", "in_cloud_only", "is_secret", "kind_of", "pack", "unpack"]
