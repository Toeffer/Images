"""Lokale Bildgenerierung über ComfyUI (https://www.comfy.org).

Bildwerk spricht die HTTP-Schnittstelle einer laufenden ComfyUI-Installation an: Workflow
einreihen (``/prompt``), auf das Ergebnis warten (``/history``), Bilder abholen (``/view``).
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
from PIL import Image

from ..imageio import ImageLoadError, decode_image
from .base import GenerateOptions, Provider, ProviderError, closest_ratio, png_bytes
from .comfy_workflows import WorkflowError, WorkflowLibrary, checkpoint_generate, checkpoint_inpaint, fill

# ComfyUI Desktop nutzt 8000 (ist der belegt, 8001), die portable Version und Handinstallationen 8188
DEFAULT_URLS = ("http://127.0.0.1:8000", "http://127.0.0.1:8188", "http://127.0.0.1:8001")
# Bewährte Größen für SDXL-artige Modelle (ca. 1 Megapixel, Vielfache von 64)
SIZES = {
    "1:1": (1024, 1024),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "4:3": (1152, 896),
    "3:4": (896, 1152),
    "16:9": (1344, 768),
    "9:16": (768, 1344),
}
CHECKPOINT_PARAMS = (
    {"name": "steps", "label": "Schritte", "type": "int", "default": 25, "min": 1, "max": 100, "step": 1},
    {"name": "cfg", "label": "Prompt-Treue (CFG)", "type": "float", "default": 6.0, "min": 1.0, "max": 15.0, "step": 0.5},
)
EDIT_PARAMS = (
    {
        "name": "denoise", "label": "Stärke der Änderung", "type": "float", "default": 1.0, "min": 0.3, "max": 1.0,
        "step": 0.05, "slider": True, "help": "1 = Bereich ganz neu malen, kleiner = näher am Original bleiben",
    },
)

_probe_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_probe_lock = threading.Lock()
_active: dict[str, tuple[str, threading.Event, str | None]] = {}  # prompt_id -> (URL, Abbruch-Signal, Auftrag)
_cancelled: dict[str, float] = {}  # abgebrochene Aufträge der Oberfläche -> Zeitpunkt
_active_lock = threading.Lock()


def configured_url(context: dict[str, Any]) -> str | None:
    url = context["settings"].get("comfyui_url") or os.environ.get("COMFYUI_URL")
    return url.rstrip("/") if url else None


def probe(url: str, client: httpx.Client, connect_timeout: float = 1.0) -> dict[str, Any] | None:
    """Fragt ``/system_stats`` ab (kurz zwischengespeichert). ``None`` = nicht erreichbar."""
    now = time.monotonic()
    with _probe_lock:
        cached = _probe_cache.get(url)
        # Erfolge länger merken: ComfyUI antwortet unter Last manchmal träge
        if cached and now - cached[0] < (15.0 if cached[1] else 3.0):
            return cached[1]
    try:
        response = client.get(f"{url}/system_stats", timeout=httpx.Timeout(5.0, connect=connect_timeout))
        stats = response.json() if response.status_code == 200 else None
        if not (isinstance(stats, dict) and "system" in stats):
            stats = None
    except (httpx.HTTPError, ValueError):
        stats = None
    with _probe_lock:
        _probe_cache[url] = (now, stats)
    return stats


def forget_probes() -> None:
    """Verbindung beim nächsten Zugriff neu prüfen (z. B. nachdem ComfyUI gestartet wurde)."""
    with _probe_lock:
        _probe_cache.clear()


def find_server(context: dict[str, Any], client: httpx.Client) -> tuple[str, dict[str, Any] | None]:
    url = configured_url(context)
    if url:
        return url, probe(url, client)
    # Gleichzeitig und mit kurzer Frist prüfen: Unter Windows wird ein Verbindungsversuch zu einem
    # unbenutzten Port erst nach Sekunden abgewiesen, ein laufender Dienst auf diesem PC antwortet sofort.
    with ThreadPoolExecutor(len(DEFAULT_URLS)) as pool:
        results = list(pool.map(lambda candidate: probe(candidate, client, 0.5), DEFAULT_URLS))
    for candidate, stats in zip(DEFAULT_URLS, results):
        if stats:
            return candidate, stats
    return DEFAULT_URLS[0], None


def device_text(stats: dict[str, Any]) -> str:
    devices = stats.get("devices") or []
    if not devices:
        return "verbunden"
    device = devices[0]
    name = str(device.get("name", "")).split(" : ")[0].replace("cuda:0", "").strip() or device.get("type", "")
    vram = device.get("vram_total") or 0
    return f"{name} ({vram / 1024**3:.0f} GB)" if vram else name


def _stop(client: httpx.Client, url: str, prompt_id: str) -> None:
    for path, body in (("/interrupt", {"prompt_id": prompt_id}), ("/queue", {"delete": [prompt_id]})):
        try:
            client.post(f"{url}{path}", json=body, timeout=3)
        except httpx.HTTPError:
            pass


def cancel(client: httpx.Client, job_id: str | None = None) -> int:
    """Bricht einen Auftrag der Oberfläche ab – oder ohne ``job_id`` alle laufenden Aufträge von Bildwerk."""
    now = time.monotonic()
    with _active_lock:
        if job_id:
            # merken, falls der Auftrag noch gar nicht bei ComfyUI eingereiht ist
            for old in [j for j, t in _cancelled.items() if now - t > 600]:
                del _cancelled[old]
            _cancelled[job_id] = now
        jobs = [(pid, url, event) for pid, (url, event, job) in _active.items() if job_id is None or job == job_id]
    for prompt_id, url, event in jobs:
        event.set()
        _stop(client, url, prompt_id)
    return len(jobs)


def _check_cancelled(job_id: str | None) -> None:
    if job_id and job_id in _cancelled:
        raise ProviderError("Abgebrochen.", 499)


class ComfyUIProvider(Provider):
    id = "comfyui"
    name = "ComfyUI (lokal)"
    env_vars = ("COMFYUI_URL",)
    key_url = "https://www.comfy.org/download"
    capabilities = frozenset({"generate", "edit"})
    models = ()
    default_model = ""
    aspect_ratios = tuple(SIZES)
    supports_negative_prompt = True
    supports_seed = True
    max_count = 4
    timeout = 1800.0
    requires_key = False
    local = True
    enhance_via_edit = False  # zum Schärfen ist der lokale Real-ESRGAN-Anbieter besser geeignet
    note = "Nutzt die Modelle deiner ComfyUI-Installation."

    @property
    def library(self) -> WorkflowLibrary:
        return WorkflowLibrary(self.context["settings"].config_dir / "comfyui")

    @classmethod
    def availability(cls, context: dict[str, Any]) -> dict[str, Any]:
        client: httpx.Client = context["http"]
        url, stats = find_server(context, client)
        workflows = WorkflowLibrary(context["settings"].config_dir / "comfyui").list()
        base = {"comfy": {"url": url, "reachable": bool(stats), "workflows": [w.to_json() for w in workflows]}}
        if not stats:
            return {
                **base,
                "configured": False,
                "status": f"Nicht erreichbar unter {url} – ComfyUI starten oder die Adresse unter ⚙ Einstellungen prüfen.",
            }
        try:
            response = client.get(f"{url}/models/checkpoints", timeout=5)
            checkpoints = response.json() if response.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            checkpoints = []
        models = [
            {
                "id": f"wf:{w.id}",
                "label": f"{w.name} (Workflow)",
                "description": "Eigener Workflow: " + ("Generieren" if w.kind == "generate" else "Bearbeiten"),
                "capabilities": [w.kind],
            }
            for w in workflows
        ] + [
            {
                "id": f"ckpt:{name}",
                "label": name,
                "description": "Checkpoint-Modell (z. B. SDXL) – Generieren und Bearbeiten",
                "capabilities": ["generate", "edit"],
                "params": True,
            }
            for name in checkpoints
            if isinstance(name, str)
        ]
        default = next((m["id"] for m in models if "generate" in m["capabilities"]), "")
        status = f"Verbunden · {device_text(stats)}"
        if not models:
            status += " · noch keine Modelle gefunden"
        base["comfy"]["checkpoints"] = checkpoints
        return {
            **base,
            "configured": True,
            "status": status,
            "models": models,
            "default_model": default,
            "model_scope": "card",
            "checkpoint_params": list(CHECKPOINT_PARAMS),
            "edit_params": list(EDIT_PARAMS),
            "edit_negative_prompt": True,
        }

    # -- Verbindung ------------------------------------------------------------------------
    @property
    def url(self) -> str:
        url, stats = find_server(self.context, self.client)
        if not stats:
            raise ProviderError(f"ComfyUI ist unter {url} nicht erreichbar. Bitte ComfyUI starten.", 503)
        return url

    def _post_json(self, url: str, body: dict[str, Any]) -> httpx.Response:
        try:
            return self.client.post(url, json=body, timeout=30)
        except httpx.HTTPError as exc:
            raise ProviderError(f"Keine Verbindung zu ComfyUI: {exc}", 502) from exc

    def _queue(self, url: str, graph: dict[str, Any]) -> str:
        response = self._post_json(f"{url}/prompt", {"prompt": graph, "client_id": "bildwerk"})
        try:
            data = response.json()
        except ValueError:
            data = {}
        if response.status_code != 200 or "prompt_id" not in data:
            raise ProviderError(_prompt_error(data, response.status_code), 400)
        return data["prompt_id"]

    def _wait(self, url: str, prompt_id: str, job_id: str | None = None) -> dict[str, Any]:
        cancelled = threading.Event()
        with _active_lock:
            _active[prompt_id] = (url, cancelled, job_id)
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if cancelled.is_set():
                    raise ProviderError("Abgebrochen.", 499)
                if job_id and job_id in _cancelled:  # Abbruch kam, bevor der Auftrag eingetragen war
                    _stop(self.client, url, prompt_id)
                    raise ProviderError("Abgebrochen.", 499)
                try:
                    response = self.client.get(f"{url}/history/{prompt_id}", timeout=10)
                    entry = response.json().get(prompt_id) if response.status_code == 200 else None
                except (httpx.HTTPError, ValueError):
                    entry = None  # ComfyUI ist beschäftigt – erneut versuchen
                if entry:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        raise ProviderError(_execution_error(status), 500)
                    if status.get("completed", True) and entry.get("outputs"):
                        return entry["outputs"]
                cancelled.wait(0.5)
            raise ProviderError("ComfyUI hat nicht rechtzeitig geantwortet.", 504)
        finally:
            with _active_lock:
                _active.pop(prompt_id, None)

    def _images(self, url: str, outputs: dict[str, Any]) -> list[Image.Image]:
        images = []
        for output in outputs.values():
            for item in output.get("images") or []:
                params = {"filename": item["filename"], "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")}
                try:
                    response = self.client.get(f"{url}/view", params=params, timeout=60)
                    response.raise_for_status()
                    images.append(decode_image(response.content))
                except (httpx.HTTPError, ImageLoadError) as exc:
                    raise ProviderError(f"Ergebnisbild konnte nicht von ComfyUI geladen werden: {exc}") from exc
        if not images:
            raise ProviderError("ComfyUI hat kein Bild geliefert – speichert der Workflow ein Bild (Save Image)?")
        return images

    def _run(self, graph: dict[str, Any], job_id: str | None = None) -> list[Image.Image]:
        _check_cancelled(job_id)
        url = self.url
        return self._images(url, self._wait(url, self._queue(url, graph), job_id))

    def _upload(self, url: str, image: Image.Image) -> str:
        name = f"bildwerk_{uuid.uuid4().hex[:12]}.png"
        files = {"image": (name, png_bytes(image), "image/png")}
        try:
            response = self.client.post(f"{url}/upload/image", files=files, data={"type": "input", "overwrite": "true"}, timeout=60)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ProviderError(f"Bild konnte nicht an ComfyUI übertragen werden: {exc}") from exc
        return f"{data['subfolder']}/{data['name']}" if data.get("subfolder") else data["name"]

    # -- Schnittstelle ------------------------------------------------------------------------
    def _workflow(self, model: str, kind: str):
        try:
            workflow = self.library.get(model[3:])
        except WorkflowError as exc:
            raise ProviderError(str(exc), 400) from exc
        if workflow.kind != kind:
            what = "Generieren" if kind == "generate" else "Bearbeiten"
            raise ProviderError(f"Der Workflow „{workflow.name}“ eignet sich nicht zum {what}.", 400)
        return workflow

    @staticmethod
    def _seed(seed: int | None) -> int:
        return seed if seed is not None else random.randint(0, 2**32 - 1)

    def generate(self, opts: GenerateOptions) -> list[Image.Image]:
        model = opts.model or ""
        job_id = (opts.extra or {}).get("job_id")
        width, height = SIZES.get(opts.aspect_ratio, SIZES["1:1"])
        count = max(1, min(self.max_count, opts.count))
        seed = self._seed(opts.seed)
        if model.startswith("wf:"):
            workflow = self._workflow(model, "generate")
            info = workflow.info
            images: list[Image.Image] = []
            runs = 1 if info.batch else count
            for i in range(runs):
                graph = fill(workflow.graph, info, prompt=opts.prompt, negative=opts.negative_prompt, width=width,
                             height=height, seed=seed + i, count=count)
                images.extend(self._run(graph, job_id))
            return images[:count]
        if model.startswith("ckpt:"):
            extra = opts.extra or {}
            graph = checkpoint_generate(
                model[5:], opts.prompt, opts.negative_prompt or "", width, height, seed,
                int(extra.get("steps", 25)), float(extra.get("cfg", 6.0)), count,
            )
            return self._run(graph, job_id)[:count]
        raise ProviderError("Bitte in ComfyUI ein Modell auswählen (Checkpoint oder Workflow).", 400)

    def edit_size(self, aspect: float, model: str | None) -> tuple[int, int]:
        return SIZES[closest_ratio(aspect, tuple(SIZES))]

    def edit(self, image: Image.Image, mask: Image.Image, prompt: str, model: str | None, options: dict | None = None) -> Image.Image:
        model = model or ""
        options = options or {}
        job_id = options.get("job_id")
        _check_cancelled(job_id)
        url = self.url
        # Die Maske steckt im Alphakanal: durchsichtig = neu malen (so liest ComfyUIs „Load Image“ sie aus)
        upload = image.convert("RGB")
        upload.putalpha(Image.eval(mask.convert("L"), lambda v: 255 - v))
        image_name = self._upload(url, upload)
        seed = self._seed(options.get("seed"))
        negative = options.get("negative_prompt") or ""
        if model.startswith("wf:"):
            workflow = self._workflow(model, "edit")
            graph = fill(workflow.graph, workflow.info, prompt=prompt, negative=negative, width=image.width,
                         height=image.height, seed=seed, image_name=image_name)
        elif model.startswith("ckpt:"):
            graph = checkpoint_inpaint(
                model[5:], prompt, negative, image_name, seed, int(options.get("steps", 25)),
                float(options.get("cfg", 6.0)), float(options.get("denoise", 1.0)),
            )
        else:
            raise ProviderError("Bitte in ComfyUI ein Modell zum Bearbeiten auswählen.", 400)
        return self._run(graph, job_id)[0]


def _prompt_error(data: dict[str, Any], status: int) -> str:
    error = data.get("error") if isinstance(data, dict) else None
    message = error.get("message") if isinstance(error, dict) else str(error or f"HTTP {status}")
    details = []
    for node_id, node in (data.get("node_errors") or {}).items() if isinstance(data, dict) else []:
        for item in node.get("errors") or []:
            details.append(f"Knoten {node_id} ({node.get('class_type', '?')}): {item.get('details') or item.get('message')}")
    text = f"ComfyUI hat den Auftrag abgelehnt: {message}"
    if details:
        text += " – " + "; ".join(details[:3])
    if "does not exist" in str(message):
        text += " (Fehlt eine Erweiterung in ComfyUI? Im ComfyUI-Manager „Install Missing Custom Nodes“ verwenden.)"
    return text


def _execution_error(status: dict[str, Any]) -> str:
    for event, info in status.get("messages") or []:
        if event == "execution_error" and isinstance(info, dict):
            message = str(info.get("exception_message", "")).strip()
            hint = ""
            if "out of memory" in message.lower() or "allocation" in message.lower():
                hint = " Tipp: zu wenig Grafikspeicher – kleineres Format oder ein kleineres/quantisiertes Modell wählen."
            return f"ComfyUI-Fehler in {info.get('node_type', 'einem Knoten')}: {message[:300]}{hint}"
    return "ComfyUI hat die Bearbeitung mit einem Fehler beendet."
