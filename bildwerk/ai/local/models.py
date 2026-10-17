"""Verwaltung der lokalen Modelle: Herunterladen, Umwandeln, Laden."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image

from . import esrgan, runtime
from .torchfile import WeightFileError, load_state_dict

log = logging.getLogger(__name__)

RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class WeightFile:
    name: str
    url: str
    size_mb: float


@dataclass(frozen=True)
class LocalModel:
    id: str
    label: str
    description: str
    files: tuple[WeightFile, ...]
    arch: str  # "srvgg" oder "rrdb"
    blend: tuple[float, ...] | None = None  # Mischung mehrerer Gewichtssätze (Real-ESRGAN „DNI“)
    tile: int = 256  # Kachelgröße (Eingangspixel)
    pad: int = 24  # Rand je Kachel – größer bei Netzen mit großem Einzugsbereich

    @property
    def size_mb(self) -> float:
        return round(sum(f.size_mb for f in self.files), 1)


GENERAL = WeightFile("realesr-general-x4v3.pth", f"{RELEASES}/v0.2.5.0/realesr-general-x4v3.pth", 4.7)
GENERAL_WDN = WeightFile("realesr-general-wdn-x4v3.pth", f"{RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth", 4.7)
X4PLUS = WeightFile("RealESRGAN_x4plus.pth", f"{RELEASES}/v0.1.0/RealESRGAN_x4plus.pth", 63.9)

CATALOG: dict[str, LocalModel] = {
    m.id: m
    for m in (
        LocalModel(
            "general",
            "Fotos – schnell",
            "Real-ESRGAN General: schärft und entrauscht, sehr schnell.",
            (GENERAL, GENERAL_WDN),
            "srvgg",
            blend=(0.5, 0.5),  # wie die Voreinstellung des Originalprogramms (Rauschminderung 0,5)
            tile=384,
            pad=24,
        ),
        LocalModel(
            "x4plus",
            "Fotos – hohe Qualität",
            "Real-ESRGAN x4plus: kräftigere Rekonstruktion, deutlich langsamer.",
            (X4PLUS,),
            "rrdb",
            tile=192,
            pad=32,
        ),
    )
}


class ModelError(RuntimeError):
    """Benutzerverständlicher Fehler beim Laden oder Ausführen eines lokalen Modells."""


class ModelStore:
    """Lädt Gewichte bei Bedarf herunter und hält fertige Sitzungen im Speicher."""

    def __init__(self, directory: Path, client: httpx.Client | None = None) -> None:
        self.directory = directory
        self.client = client or httpx.Client(follow_redirects=True)
        self._lock = threading.Lock()
        self._file_locks: dict[str, threading.Lock] = {}
        self._sessions: dict[str, Any] = {}
        self._progress: dict[str, float] = {}
        self._errors: dict[str, str] = {}

    def path(self, weight: WeightFile) -> Path:
        return self.directory / weight.name

    def installed(self, model: LocalModel) -> bool:
        return all(self.path(f).is_file() for f in model.files)

    # -- Herunterladen ---------------------------------------------------------------
    def _download(self, weight: WeightFile, model_id: str, done_mb: float, total_mb: float) -> None:
        with self._lock:
            lock = self._file_locks.setdefault(weight.name, threading.Lock())
        with lock:
            target = self.path(weight)
            if target.is_file():
                return
            self.directory.mkdir(parents=True, exist_ok=True)
            part = target.with_suffix(".part")
            try:
                with self.client.stream("GET", weight.url, timeout=httpx.Timeout(30.0, read=120.0)) as response:
                    if response.status_code != 200:
                        raise ModelError(f"Download von {weight.name} fehlgeschlagen (HTTP {response.status_code}).")
                    loaded = 0
                    with part.open("wb") as fh:
                        for chunk in response.iter_bytes(1 << 20):
                            fh.write(chunk)
                            loaded += len(chunk)
                            self._progress[model_id] = min(0.99, (done_mb + loaded / 1e6) / total_mb)
                part.replace(target)
            except httpx.HTTPError as exc:
                part.unlink(missing_ok=True)
                raise ModelError(
                    f"Download von {weight.name} fehlgeschlagen: {exc}. Die Datei kann auch von Hand "
                    f"heruntergeladen ({weight.url}) und in {self.directory} abgelegt werden."
                ) from exc
            except BaseException:
                part.unlink(missing_ok=True)
                raise

    def ensure(self, model: LocalModel) -> None:
        """Lädt fehlende Gewichtsdateien des Modells herunter."""
        if self.installed(model):
            return
        self._errors.pop(model.id, None)
        self._progress[model.id] = 0.0
        total = sum(f.size_mb for f in model.files) or 1.0
        done = 0.0
        try:
            for weight in model.files:
                self._download(weight, model.id, done, total)
                done += weight.size_mb
        except ModelError as exc:
            self._errors[model.id] = str(exc)
            raise
        finally:
            self._progress.pop(model.id, None)

    def start_download(self, model: LocalModel) -> None:
        if self.installed(model) or model.id in self._progress:
            return
        self._progress[model.id] = 0.0

        def run() -> None:
            try:
                self.ensure(model)
            except ModelError:
                log.warning("Download von %s fehlgeschlagen", model.id, exc_info=True)
            finally:
                if self.installed(model):  # falls ein anderer Aufruf schneller fertig war
                    self._progress.pop(model.id, None)

        threading.Thread(target=run, daemon=True, name=f"download-{model.id}").start()

    def status(self) -> list[dict[str, Any]]:
        return [
            {
                "id": m.id,
                "label": m.label,
                "description": m.description,
                "size_mb": m.size_mb,
                "installed": self.installed(m),
                "progress": self._progress.get(m.id),
                "error": self._errors.get(m.id),
            }
            for m in CATALOG.values()
        ]

    # -- Ausführen -------------------------------------------------------------------
    def session(self, model: LocalModel):
        with self._lock:
            cached = self._sessions.get(model.id)
        if cached is not None:
            return cached
        self.ensure(model)
        try:
            states = [load_state_dict(self.path(f)) for f in model.files]
            state = esrgan.blend_states(states, model.blend) if model.blend else states[0]
            onnx_model = esrgan.srvgg_onnx(state) if model.arch == "srvgg" else esrgan.rrdbnet_onnx(state)
        except WeightFileError as exc:
            raise ModelError(f"{model.label}: {exc} Tipp: Die Datei in {self.directory} löschen, dann wird sie neu geladen.") from exc
        session = runtime.create_session(onnx_model)
        _self_test(session, model)
        with self._lock:
            self._sessions[model.id] = session
        return session

    def upscale(self, model: LocalModel, image: Image.Image) -> Image.Image:
        session = self.session(model)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        try:
            out = runtime.upscale_tiled(session, arr, scale=4, tile=model.tile, pad=model.pad)
        except Exception as exc:  # z. B. zu wenig Grafikspeicher
            raise ModelError(f"Das lokale Modell ist bei der Berechnung gescheitert: {exc}") from exc
        return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGB")

    def device(self) -> str:
        """Rechengerät der geladenen Modelle bzw. das voraussichtlich genutzte."""
        with self._lock:
            sessions = list(self._sessions.values())
        return runtime.device_label(sessions[0]) if sessions else runtime.expected_device()


def _self_test(session, model: LocalModel) -> None:
    """Kurzer Plausibilitätstest: ein Farbverlauf muss als ähnlicher Verlauf herauskommen."""
    yy, xx = np.mgrid[0:24, 0:24].astype(np.float32) / 23.0
    test = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    out = runtime.upscale_tiled(session, test, scale=4, tile=24, pad=4)
    reference = np.repeat(np.repeat(test, 4, axis=0), 4, axis=1)
    if not np.isfinite(out).all() or np.corrcoef(out.ravel(), reference.ravel())[0, 1] < 0.9:
        raise ModelError(f"{model.label}: Das Modell liefert unplausible Ergebnisse – ist die Gewichtsdatei beschädigt?")
