"""Liest PyTorch-Gewichtsdateien (``.pth``) ohne PyTorch.

``torch.save`` erzeugt ein ZIP-Archiv mit einer Pickle-Datei (``data.pkl``), die auf
Rohdaten-Dateien (``data/0``, ``data/1`` …) verweist. Dieser Lader baut daraus NumPy-Arrays.
Er erlaubt nur die wenigen Klassen, die in Gewichtsdateien vorkommen – beliebiger Code
aus einer Datei kann so nicht ausgeführt werden.
"""

from __future__ import annotations

import collections
import io
import pickle
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


class WeightFileError(ValueError):
    """Die Gewichtsdatei ist beschädigt oder hat ein unbekanntes Format."""


_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "DoubleStorage": np.float64,
    "BFloat16Storage": "bfloat16",
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name: str) -> None:
        self.name = name
        self.dtype = _STORAGE_DTYPES[name]


def _rebuild_tensor(storage: np.ndarray, offset: int, size, stride, *_args) -> np.ndarray:
    size, stride = tuple(size), tuple(stride)
    if not size:
        return np.array(storage[offset])
    view = np.lib.stride_tricks.as_strided(
        storage[offset:], shape=size, strides=tuple(s * storage.itemsize for s in stride)
    )
    return np.ascontiguousarray(view)


def _rebuild_parameter(data: np.ndarray, *_args) -> np.ndarray:
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: BinaryIO, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file)
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _STORAGE_DTYPES:
            return _StorageType(name)
        if (module, name) == ("torch", "Size"):
            return tuple
        raise WeightFileError(f"Unerwarteter Inhalt in der Gewichtsdatei: {module}.{name}")

    def persistent_load(self, pid: Any) -> np.ndarray:
        if not (isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage"):
            raise WeightFileError("Unbekanntes Speicherformat in der Gewichtsdatei.")
        storage_type, key = pid[1], pid[2]
        raw = self._archive.read(f"{self._prefix}data/{key}")
        if storage_type.dtype == "bfloat16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            return bits.view(np.float32)
        return np.frombuffer(raw, dtype=np.dtype(storage_type.dtype).newbyteorder("<"))


def load(source: str | Path | bytes) -> Any:
    """Lädt den Inhalt einer ``.pth``-Datei (Tensoren werden zu NumPy-Arrays)."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(source) if isinstance(source, bytes) else source)
    except zipfile.BadZipFile as exc:
        raise WeightFileError("Die Datei ist keine gültige PyTorch-Gewichtsdatei (ZIP-Format erwartet).") from exc
    with archive:
        pkl = [n for n in archive.namelist() if n.endswith("data.pkl")]
        if not pkl:
            raise WeightFileError("In der Gewichtsdatei fehlt data.pkl.")
        prefix = pkl[0][: -len("data.pkl")]
        with archive.open(pkl[0]) as fh:
            try:
                return _Unpickler(io.BytesIO(fh.read()), archive, prefix).load()
            except WeightFileError:
                raise
            except Exception as exc:
                raise WeightFileError(f"Die Gewichtsdatei konnte nicht gelesen werden: {exc}") from exc


def load_state_dict(source: str | Path | bytes) -> dict[str, np.ndarray]:
    """Lädt die Gewichte (bei Real-ESRGAN unter ``params_ema`` bzw. ``params``)."""
    obj = load(source)
    if isinstance(obj, dict):
        for key in ("params_ema", "params", "state_dict"):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not all(isinstance(v, np.ndarray) for v in obj.values()):
        raise WeightFileError("Die Gewichtsdatei enthält keine Modellgewichte.")
    return {k: v.astype(np.float32) for k, v in obj.items()}
