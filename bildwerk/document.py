"""Geöffnete Bilder mit Verlauf (Rückgängig/Wiederholen) und vorgemerkten Vorschauen."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from .engine import Change
from .imageio import ImageMeta
from .selection import Box

MAX_ENTRIES = 100


def _nbytes(im: Image.Image) -> int:
    return im.width * im.height * len(im.getbands())


@dataclass
class HistoryEntry:
    label: str
    kind: Literal["patch", "replace"]
    box: Box | None
    before: Image.Image
    after: Image.Image

    @property
    def nbytes(self) -> int:
        return _nbytes(self.before) + _nbytes(self.after)


@dataclass
class ChangeInfo:
    """Beschreibt für die Oberfläche, welcher Bereich neu geladen werden muss."""

    kind: Literal["none", "patch", "replace"] = "none"
    box: Box | None = None

    def merge(self, other: "ChangeInfo") -> "ChangeInfo":
        if self.kind == "none":
            return other
        if other.kind == "none":
            return self
        if "replace" in (self.kind, other.kind):
            return ChangeInfo("replace")
        a, b = self.box, other.box
        return ChangeInfo("patch", (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))

    def to_json(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind}
        if self.box is not None:
            data["box"] = list(self.box)
        return data


@dataclass
class Pending:
    """Berechnetes, aber noch nicht übernommenes Ergebnis (Vorschau, KI-Ergebnis)."""

    id: str
    version: int
    label: str
    change: Change
    key: str | None = None


class VersionConflict(RuntimeError):
    pass


class Document:
    def __init__(
        self,
        image: Image.Image,
        name: str,
        meta: ImageMeta | None = None,
        origin_label: str = "Geöffnet",
        history_limit: int = 768 * 1024 * 1024,
        saved: bool = True,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.image = image
        self.meta = meta or ImageMeta()
        self.base_label = origin_label
        self.entries: list[HistoryEntry] = []
        self.position = 0
        self.version = 1
        self.saved_version = 1 if saved else 0
        self.history_limit = history_limit
        self.pending: Pending | None = None
        self.lock = threading.RLock()

    # -- Verlauf -----------------------------------------------------------------
    def apply(self, change: Change, label: str, expected_version: int | None = None) -> ChangeInfo:
        with self.lock:
            if expected_version is not None and expected_version != self.version:
                raise VersionConflict("Das Bild wurde zwischenzeitlich geändert. Bitte erneut versuchen.")
            if change.kind == "patch":
                box = change.box
                before = self.image.crop(box)
                new_image = self.image.copy()
                new_image.paste(change.patch, box[:2])
                entry = HistoryEntry(label, "patch", box, before, change.patch)
                info = ChangeInfo("patch", box)
            else:
                entry = HistoryEntry(label, "replace", None, self.image, change.image)
                new_image = change.image
                info = ChangeInfo("replace")
            del self.entries[self.position :]
            self.entries.append(entry)
            self.position += 1
            self._set_image(new_image)
            self._trim()
            return info

    def undo(self) -> ChangeInfo:
        with self.lock:
            if self.position == 0:
                return ChangeInfo()
            entry = self.entries[self.position - 1]
            self.position -= 1
            return self._restore(entry, entry.before)

    def redo(self) -> ChangeInfo:
        with self.lock:
            if self.position >= len(self.entries):
                return ChangeInfo()
            entry = self.entries[self.position]
            self.position += 1
            return self._restore(entry, entry.after)

    def goto(self, position: int) -> ChangeInfo:
        with self.lock:
            position = max(0, min(len(self.entries), position))
            info = ChangeInfo()
            while self.position > position:
                info = info.merge(self.undo())
            while self.position < position:
                info = info.merge(self.redo())
            return info

    def _restore(self, entry: HistoryEntry, state: Image.Image) -> ChangeInfo:
        if entry.kind == "patch":
            new_image = self.image.copy()
            new_image.paste(state, entry.box[:2])
            self._set_image(new_image)
            return ChangeInfo("patch", entry.box)
        self._set_image(state)
        return ChangeInfo("replace")

    def _set_image(self, image: Image.Image) -> None:
        self.image = image
        self.version += 1
        self.pending = None

    def _trim(self) -> None:
        total = sum(e.nbytes for e in self.entries)
        while self.position > 1 and (total > self.history_limit or len(self.entries) > MAX_ENTRIES):
            total -= self.entries.pop(0).nbytes
            self.position -= 1
            self.base_label = "Ältere Schritte entfernt"

    # -- Vorschau ------------------------------------------------------------------
    def set_pending(self, change: Change, label: str, version: int, key: str | None = None) -> Pending | None:
        with self.lock:
            if version != self.version:
                return None
            self.pending = Pending(uuid.uuid4().hex[:12], version, label, change, key)
            return self.pending

    def commit_pending(self, pending_id: str) -> ChangeInfo:
        with self.lock:
            pending = self.pending
            if pending is None or pending.id != pending_id:
                raise VersionConflict("Diese Vorschau ist nicht mehr gültig.")
            return self.apply(pending.change, pending.label, expected_version=pending.version)

    def take_matching_pending(self, key: str) -> Pending | None:
        with self.lock:
            pending = self.pending
            if pending and pending.key == key and pending.version == self.version:
                return pending
            return None

    def discard_pending(self) -> None:
        with self.lock:
            self.pending = None

    # -- Darstellung -----------------------------------------------------------------
    def mark_saved(self) -> None:
        self.saved_version = self.version

    def info(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "name": self.name,
                "width": self.image.width,
                "height": self.image.height,
                "has_alpha": self.image.mode == "RGBA",
                "version": self.version,
                "dirty": self.version != self.saved_version,
                "source_format": self.meta.source_format,
                "history": {
                    "labels": [self.base_label] + [e.label for e in self.entries],
                    "position": self.position,
                },
            }


class DocumentStore:
    def __init__(self, history_limit: int = 768 * 1024 * 1024) -> None:
        self._docs: dict[str, Document] = {}
        self._lock = threading.Lock()
        self.history_limit = history_limit

    def add(self, image: Image.Image, name: str, meta: ImageMeta | None = None, **kwargs: Any) -> Document:
        doc = Document(image, name, meta, history_limit=self.history_limit, **kwargs)
        with self._lock:
            self._docs[doc.id] = doc
        return doc

    def get(self, doc_id: str) -> Document:
        with self._lock:
            return self._docs[doc_id]

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._docs.pop(doc_id, None)

    def all(self) -> list[Document]:
        with self._lock:
            return list(self._docs.values())
