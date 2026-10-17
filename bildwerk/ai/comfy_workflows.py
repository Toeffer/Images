"""ComfyUI-Workflows: eingebaute Vorlagen und das Auswerten exportierter Workflows.

Eigene Workflows exportiert man in ComfyUI über „Workflow → Export (API)“. Bildwerk
findet darin selbst, wo Beschreibung, Bildgröße, Seed und Eingangsbild hingehören. Wer es
eindeutig möchte, schreibt ``{{prompt}}`` (und optional ``{{negative}}``) in die Textfelder.
"""

from __future__ import annotations

import copy
import json
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

TEXT_KEYS = ("text", "prompt", "text_g", "text_l", "value", "string")
SEED_KEYS = ("seed", "noise_seed")
OUTPUT_CLASSES = ("SaveImage", "PreviewImage")
IMAGE_INPUT_CLASSES = ("LoadImage",)
PROMPT_PLACEHOLDER = "{{prompt}}"
NEGATIVE_PLACEHOLDER = "{{negative}}"


class WorkflowError(ValueError):
    """Der Workflow kann nicht verwendet werden (Meldung für Benutzer)."""


def _is_link(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass
class WorkflowInfo:
    kind: str  # "generate" oder "edit"
    prompt: tuple[str, str]
    negative: tuple[str, str] | None = None
    size_nodes: list[str] = field(default_factory=list)
    seeds: list[tuple[str, str]] = field(default_factory=list)
    batch: list[tuple[str, str]] = field(default_factory=list)
    image_node: str | None = None
    outputs: list[str] = field(default_factory=list)

    def summary(self, graph: dict[str, Any]) -> list[str]:
        def label(node_id: str) -> str:
            node = graph[node_id]
            title = (node.get("_meta") or {}).get("title") or node["class_type"]
            return f"Knoten {node_id} ({title})"

        lines = [f"Beschreibung: {label(self.prompt[0])}"]
        if self.negative:
            lines.append(f"Negativ: {label(self.negative[0])}")
        if self.size_nodes:
            lines.append("Bildgröße: " + ", ".join(label(n) for n in self.size_nodes))
        if self.image_node:
            lines.append(f"Eingangsbild: {label(self.image_node)}")
        if self.seeds:
            lines.append("Zufallswert: " + ", ".join(label(n) for n, _ in self.seeds))
        return lines


def _nodes(graph: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(graph, dict) or not graph:
        raise WorkflowError("Die Datei enthält keinen ComfyUI-Workflow.")
    if "nodes" in graph and "links" in graph:
        raise WorkflowError(
            "Das ist das normale ComfyUI-Speicherformat. Bitte in ComfyUI über „Workflow → Export (API)“ exportieren."
        )
    nodes = {
        str(nid): node
        for nid, node in graph.items()
        if isinstance(node, dict) and isinstance(node.get("class_type"), str) and isinstance(node.get("inputs"), dict)
    }
    if not nodes:
        raise WorkflowError("Die Datei enthält keinen ComfyUI-Workflow im API-Format.")
    return nodes


def _text_input(node: dict[str, Any]) -> str | None:
    for key in TEXT_KEYS:
        if isinstance(node["inputs"].get(key), str):
            return key
    return None


def _trace(nodes: dict[str, dict[str, Any]], input_name: str) -> tuple[str, str] | None:
    """Folgt dem Eingang ``input_name`` (z. B. „positive“) rückwärts bis zu einem Textfeld."""
    starts = []
    for node in nodes.values():
        value = node["inputs"].get(input_name)
        if _is_link(value) and (input_name != "conditioning" or "guider" in node["class_type"].lower()):
            starts.append(str(value[0]))
    for start in starts:
        queue, seen = deque([start]), set()
        while queue:
            nid = queue.popleft()
            if nid in seen or nid not in nodes:
                continue
            seen.add(nid)
            key = _text_input(nodes[nid])
            if key:
                return nid, key
            queue.extend(str(v[0]) for v in nodes[nid]["inputs"].values() if _is_link(v))
    return None


def analyze(graph: Any) -> WorkflowInfo:
    nodes = _nodes(graph)
    prompt = negative = None
    for nid, node in nodes.items():
        for key, value in node["inputs"].items():
            if isinstance(value, str) and PROMPT_PLACEHOLDER in value:
                prompt = (nid, key)
            elif isinstance(value, str) and NEGATIVE_PLACEHOLDER in value:
                negative = (nid, key)
    if prompt is None:
        prompt = _trace(nodes, "positive") or _trace(nodes, "conditioning")
    if prompt is None:
        text_nodes = [(nid, _text_input(n)) for nid, n in nodes.items() if _text_input(n) and "encode" in n["class_type"].lower()]
        prompt = text_nodes[0] if text_nodes else None
    if prompt is None:
        raise WorkflowError(
            "Kein Textfeld für die Beschreibung gefunden. Tipp: In ComfyUI {{prompt}} in das Textfeld schreiben und neu exportieren."
        )
    if negative is None:
        negative = _trace(nodes, "negative")
        if negative and negative[0] == prompt[0]:
            negative = None

    size_nodes = [
        nid
        for nid, n in nodes.items()
        if _is_number(n["inputs"].get("width")) and _is_number(n["inputs"].get("height")) and "latent" in n["class_type"].lower()
    ]
    seeds = [(nid, k) for nid, n in nodes.items() for k in SEED_KEYS if _is_number(n["inputs"].get(k))]
    batch = [(nid, "batch_size") for nid in size_nodes if _is_number(nodes[nid]["inputs"].get("batch_size"))]
    images = [nid for nid, n in nodes.items() if n["class_type"] in IMAGE_INPUT_CLASSES]
    outputs = [nid for nid, n in nodes.items() if n["class_type"] in OUTPUT_CLASSES or n["class_type"].startswith("SaveImage")]
    if not outputs:
        raise WorkflowError("Der Workflow speichert kein Bild – es fehlt ein „Save Image“-Knoten.")
    return WorkflowInfo(
        kind="edit" if images else "generate",
        prompt=prompt,
        negative=negative,
        size_nodes=size_nodes,
        seeds=seeds,
        batch=batch,
        image_node=images[0] if images else None,
        outputs=outputs,
    )


def fill(
    graph: dict[str, Any],
    info: WorkflowInfo,
    *,
    prompt: str,
    negative: str | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int = 0,
    count: int = 1,
    image_name: str | None = None,
) -> dict[str, Any]:
    """Setzt Beschreibung, Größe, Seed usw. in eine Kopie des Workflows ein."""
    graph = copy.deepcopy(graph)

    def put(target: tuple[str, str], text: str, placeholder: str) -> None:
        nid, key = target
        current = graph[nid]["inputs"][key]
        graph[nid]["inputs"][key] = current.replace(placeholder, text) if placeholder in current else text

    put(info.prompt, prompt, PROMPT_PLACEHOLDER)
    if info.negative:
        put(info.negative, negative or "", NEGATIVE_PLACEHOLDER)
    for nid in info.size_nodes:
        if width and height:
            graph[nid]["inputs"]["width"] = width
            graph[nid]["inputs"]["height"] = height
    for nid, key in info.seeds:
        graph[nid]["inputs"][key] = seed
    for nid, key in info.batch:
        graph[nid]["inputs"][key] = count
    if info.image_node and image_name:
        graph[info.image_node]["inputs"]["image"] = image_name
    return graph


# -- Eingebaute Vorlagen für Checkpoint-Modelle (SD 1.5, SDXL und Abkömmlinge) ------------------
def checkpoint_generate(
    ckpt: str, prompt: str, negative: str, width: int, height: int, seed: int, steps: int, cfg: float, count: int,
    sampler: str = "dpmpp_2m", scheduler: str = "karras",
) -> dict[str, Any]:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": count}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                "scheduler": scheduler, "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0], "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "Bildwerk"}},
    }


def checkpoint_inpaint(
    ckpt: str, prompt: str, negative: str, image_name: str, seed: int, steps: int, cfg: float, denoise: float,
    sampler: str = "dpmpp_2m", scheduler: str = "karras",
) -> dict[str, Any]:
    """Bearbeitet den maskierten Bereich – die Maske steckt im Alphakanal des Eingangsbilds."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {
            "class_type": "InpaintModelConditioning",
            "inputs": {"positive": ["2", 0], "negative": ["3", 0], "vae": ["1", 2], "pixels": ["4", 0], "mask": ["4", 1], "noise_mask": True},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                "scheduler": scheduler, "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2], "denoise": denoise,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "Bildwerk"}},
    }


# -- Gespeicherte Workflows -------------------------------------------------------------------
@dataclass
class StoredWorkflow:
    id: str
    name: str
    kind: str
    graph: dict[str, Any]

    @property
    def info(self) -> WorkflowInfo:
        return analyze(self.graph)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "summary": self.info.summary(self.graph)}


class WorkflowLibrary:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = threading.Lock()

    def list(self) -> list[StoredWorkflow]:
        if not self.directory.is_dir():
            return []
        result = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append(StoredWorkflow(path.stem, data["name"], data["kind"], data["graph"]))
            except (OSError, ValueError, KeyError):
                continue  # beschädigte Datei überspringen
        return result

    def get(self, workflow_id: str) -> StoredWorkflow:
        for workflow in self.list():
            if workflow.id == workflow_id:
                return workflow
        raise WorkflowError("Dieser Workflow ist nicht mehr vorhanden.")

    def add(self, name: str, graph: Any) -> StoredWorkflow:
        info = analyze(graph)
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "workflow"
        workflow = StoredWorkflow(f"{slug}-{uuid.uuid4().hex[:6]}", name.strip() or "Workflow", info.kind, graph)
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            payload = {"name": workflow.name, "kind": workflow.kind, "graph": graph}
            (self.directory / f"{workflow.id}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return workflow

    def remove(self, workflow_id: str) -> None:
        if not re.fullmatch(r"[a-z0-9-]+", workflow_id):
            raise WorkflowError("Ungültige Workflow-Kennung.")
        with self._lock:
            (self.directory / f"{workflow_id}.json").unlink(missing_ok=True)
