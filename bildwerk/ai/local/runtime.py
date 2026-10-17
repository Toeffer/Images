"""Ausführung der lokalen Modelle mit ONNX Runtime – auf der Grafikkarte, wenn möglich."""

from __future__ import annotations

import logging
import math
import os

import numpy as np

log = logging.getLogger(__name__)

DEVICE_LABELS = {
    "DmlExecutionProvider": "Grafikkarte (DirectML)",
    "CUDAExecutionProvider": "Grafikkarte (CUDA)",
    "CPUExecutionProvider": "Prozessor (CPU)",
}


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import onnx  # noqa: F401
    except ImportError:
        return False
    return True


def cpu_only() -> bool:
    """``BILDWERK_LOCAL_DEVICE=cpu`` erzwingt den Prozessor, z. B. bei Treiberproblemen."""
    return os.environ.get("BILDWERK_LOCAL_DEVICE", "").strip().lower() == "cpu"


def expected_device() -> str:
    try:
        import onnxruntime as ort
    except ImportError:
        return "nicht verfügbar"
    if cpu_only():
        return DEVICE_LABELS["CPUExecutionProvider"]
    for provider in ("DmlExecutionProvider", "CUDAExecutionProvider"):
        if provider in ort.get_available_providers():
            return DEVICE_LABELS[provider]
    return DEVICE_LABELS["CPUExecutionProvider"]


def _provider_candidates() -> list:
    import onnxruntime as ort

    available = [] if cpu_only() else ort.get_available_providers()
    candidates: list = []
    if "DmlExecutionProvider" in available:
        # Auf Laptops mit zwei Grafikchips die leistungsstarke Grafikkarte wählen
        candidates.append([("DmlExecutionProvider", {"performance_preference": "high_performance", "device_filter": "gpu"}), "CPUExecutionProvider"])
        candidates.append(["DmlExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" in available:
        candidates.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
    candidates.append(["CPUExecutionProvider"])
    return candidates


def create_session(model: bytes):
    """Erzeugt eine InferenceSession; fällt bei Problemen mit der Grafikkarte auf die CPU zurück."""
    import onnxruntime as ort

    last_error: Exception | None = None
    for providers in _provider_candidates():
        options = ort.SessionOptions()
        # Voraussetzung für DirectML laut ONNX-Runtime-Dokumentation
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        try:
            return ort.InferenceSession(model, sess_options=options, providers=providers)
        except Exception as exc:  # z. B. alter Treiber oder unbekannte Option
            log.warning("ONNX Runtime mit %s nicht möglich: %s", providers, exc)
            last_error = exc
    raise RuntimeError(f"ONNX Runtime konnte das Modell nicht laden: {last_error}")


def device_label(session) -> str:
    provider = session.get_providers()[0]
    return DEVICE_LABELS.get(provider, provider)


def upscale_tiled(session, image: np.ndarray, scale: int = 4, tile: int = 256, pad: int = 16) -> np.ndarray:
    """Vergrößert ``image`` (H×W×3, float32 in [0, 1]) kachelweise.

    Jede Kachel wird mit Rand (``pad``) aus der Umgebung gerechnet, damit keine Nähte
    entstehen. Alle Kacheln haben dieselbe Größe (am Bildrand gespiegelt), so muss die
    Grafikkarte den Graphen nicht für jede Größe neu vorbereiten.
    """
    height, width = image.shape[:2]
    tile_h, tile_w = min(tile, height), min(tile, width)
    pad_y, pad_x = min(pad, max(0, height - 1)), min(pad, max(0, width - 1))
    padded = np.pad(image, ((pad_y, pad_y + tile_h), (pad_x, pad_x + tile_w), (0, 0)), mode="reflect")
    out = np.empty((height * scale, width * scale, 3), dtype=np.float32)
    input_name = session.get_inputs()[0].name

    for ty in range(math.ceil(height / tile_h)):
        for tx in range(math.ceil(width / tile_w)):
            y0, x0 = ty * tile_h, tx * tile_w
            patch = padded[y0 : y0 + tile_h + 2 * pad_y, x0 : x0 + tile_w + 2 * pad_x]
            batch = np.ascontiguousarray(patch.transpose(2, 0, 1)[None], dtype=np.float32)
            result = session.run(None, {input_name: batch})[0][0].transpose(1, 2, 0)
            h = min(tile_h, height - y0)
            w = min(tile_w, width - x0)
            out[y0 * scale : (y0 + h) * scale, x0 * scale : (x0 + w) * scale] = result[
                pad_y * scale : (pad_y + h) * scale, pad_x * scale : (pad_x + w) * scale
            ]
    return np.clip(out, 0.0, 1.0)
