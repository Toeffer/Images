"""HTTP-Schnittstelle (FastAPI) zwischen Browser-Oberfläche und Bildverarbeitung."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .ai import GenerateOptions, ProviderError, ProviderRegistry
from .ai import comfyui_provider
from .ai.comfy_workflows import WorkflowError, WorkflowLibrary
from .ai.local.models import CATALOG as LOCAL_MODELS
from .ai.local.runtime import onnxruntime_available
from .ai.pipeline import ai_edit, ai_enhance
from .document import ChangeInfo, Document, DocumentStore, Pending, VersionConflict
from .engine import Change, NewImage, run_operation
from .imageio import EXPORT_FORMATS, HEIC_SUPPORT, ImageLoadError, ImageMeta, export_image, load_image, to_display_rgba
from .ops import OperationError, catalog, get_operation
from .selection import Selection
from .settings import Settings

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 300 * 1024 * 1024
LOCAL_HOSTS = ["127.0.0.1", "localhost", "::1"]


# -- Anfragemodelle -----------------------------------------------------------------------
class OpRequest(BaseModel):
    op: str
    params: dict[str, Any] = Field(default_factory=dict)
    selection: Selection | None = None


class CommitRequest(BaseModel):
    preview_id: str


class NewDocumentRequest(BaseModel):
    width: int = Field(ge=1, le=20000)
    height: int = Field(ge=1, le=20000)
    background: str = Field("#ffffff", pattern=r"^(#[0-9a-fA-F]{6}|transparent)$")
    name: str = Field("Unbenannt", min_length=1, max_length=200)


class RenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class AiOptions(BaseModel):
    """Zusatzeinstellungen lokaler Modelle (ComfyUI)."""

    steps: int | None = Field(None, ge=1, le=150)
    cfg: float | None = Field(None, ge=0, le=30)
    denoise: float | None = Field(None, ge=0.05, le=1)
    seed: int | None = Field(None, ge=0, le=4_294_967_294)
    negative_prompt: str | None = Field(None, max_length=4000)


# Kennung eines KI-Auftrags aus der Oberfläche, erlaubt gezieltes Abbrechen
JobId = Annotated[str | None, Field(max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]


class GenerateRequest(BaseModel):
    provider: str
    job_id: JobId = None
    prompt: str = Field(min_length=1, max_length=8000)
    model: str | None = None
    aspect_ratio: str = "1:1"
    quality: str | None = None
    resolution: str | None = None
    negative_prompt: str | None = Field(None, max_length=4000)
    seed: int | None = Field(None, ge=0, le=4_294_967_294)
    count: int = Field(1, ge=1, le=4)
    options: AiOptions = Field(default_factory=AiOptions)


class AiEditRequest(BaseModel):
    provider: str
    job_id: JobId = None
    prompt: str = Field(min_length=1, max_length=8000)
    model: str | None = None
    selection: Selection | None = None
    options: AiOptions = Field(default_factory=AiOptions)


class AiEnhanceRequest(BaseModel):
    provider: str
    model: str | None = None
    upscale_model: str | None = None
    selection: Selection | None = None
    new_document: bool = False


class CancelRequest(BaseModel):
    job_id: JobId = None


class ApiKeyRequest(BaseModel):
    api_key: str | None = Field(None, max_length=1000)


class ComfyUrlRequest(BaseModel):
    url: str | None = Field(None, max_length=300, pattern=r"^(https?://[^\s]+)?$")


# -- Hilfsfunktionen ---------------------------------------------------------------------
def _stem(name: str) -> str:
    stem = name.rsplit(".", 1)[0] if "." in name[1:] else name
    return stem.strip() or "Bild"


def _short(text: str, limit: int = 40) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pending_json(pending: Pending) -> dict[str, Any]:
    change = pending.change
    data: dict[str, Any] = {"preview_id": pending.id, "kind": change.kind, "label": pending.label}
    if change.kind == "patch":
        data["box"] = list(change.box)
    else:
        data["width"], data["height"] = change.image.size
    return data


def _op_key(op_id: str, params: dict[str, Any], selection: Selection | None) -> str:
    payload = {"op": op_id, "params": params, "selection": selection.model_dump(mode="json") if selection else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _options(options: AiOptions, job_id: str | None) -> dict[str, Any]:
    result = options.model_dump(exclude_none=True)
    if job_id:
        result["job_id"] = job_id
    return result


def _raw_response(image: Image.Image, meta: ImageMeta | None, version: int | None = None) -> Response:
    headers = {"X-Width": str(image.width), "X-Height": str(image.height), "Cache-Control": "no-store"}
    if version is not None:
        headers["X-Version"] = str(version)
    return Response(to_display_rgba(image, meta), media_type="application/octet-stream", headers=headers)


def create_app(
    settings: Settings | None = None,
    registry: ProviderRegistry | None = None,
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    settings = settings or Settings()
    registry = registry or ProviderRegistry(settings)
    history_mb = int(os.environ.get("BILDWERK_HISTORY_MB", "768"))
    store = DocumentStore(history_limit=history_mb * 1024 * 1024)

    app = FastAPI(title="Bildwerk", version=__version__, docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.settings = settings
    app.state.registry = registry
    app.state.store = store

    hosts = allowed_hosts or LOCAL_HOSTS
    if "*" not in hosts:
        # Schützt vor DNS-Rebinding: nur Anfragen an localhost werden angenommen.
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=hosts)

    @app.middleware("http")
    async def reject_foreign_origins(request: Request, call_next):
        # Fremde Webseiten dürfen keine verändernden Anfragen an Bildwerk schicken.
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            if urlsplit(origin).netloc != request.headers.get("host", ""):
                return JSONResponse({"detail": "Anfrage von einer fremden Webseite abgelehnt."}, status_code=403)
        return await call_next(request)

    @app.exception_handler(OperationError)
    async def _op_error(request: Request, exc: OperationError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ImageLoadError)
    async def _load_error(request: Request, exc: ImageLoadError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ProviderError)
    async def _provider_error(request: Request, exc: ProviderError):
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.exception_handler(VersionConflict)
    async def _conflict(request: Request, exc: VersionConflict):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(WorkflowError)
    async def _workflow_error(request: Request, exc: WorkflowError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    def get_doc(doc_id: str) -> Document:
        try:
            return store.get(doc_id)
        except KeyError:
            raise HTTPException(404, "Bild nicht gefunden – wurde Bildwerk zwischendurch neu gestartet?") from None

    def mutation(doc: Document, info: ChangeInfo, **extra: Any) -> dict[str, Any]:
        return {"document": doc.info(), "change": info.to_json(), **extra}

    def add_derived(doc: Document, image: Image.Image, label: str) -> Document:
        meta = ImageMeta(icc_profile=doc.meta.icc_profile)
        return store.add(image, f"{_stem(doc.name)} – {label}", meta, origin_label=label, saved=False)

    # -- Oberfläche ---------------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {
            "version": __version__,
            "categories": catalog(),
            "providers": registry.describe(),
            "export_formats": [
                {"id": k, "label": v[5], "ext": v[1], "alpha": v[3], "lossy": v[4]} for k, v in EXPORT_FORMATS.items()
            ],
            "heic": HEIC_SUPPORT,
        }

    # -- Dokumente -----------------------------------------------------------------------
    @app.get("/api/documents")
    def list_documents() -> dict[str, Any]:
        return {"documents": [d.info() for d in store.all()]}

    @app.post("/api/documents")
    def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Die Datei ist zu groß (max. 300 MB).")
        image, meta = load_image(data)
        doc = store.add(image, file.filename or "Bild", meta)
        return doc.info()

    @app.post("/api/documents/new")
    def new_document(req: NewDocumentRequest) -> dict[str, Any]:
        if req.width * req.height > 100_000_000:
            raise HTTPException(400, "Das Bild wäre zu groß (max. 100 Megapixel).")
        if req.background == "transparent":
            image = Image.new("RGBA", (req.width, req.height), (0, 0, 0, 0))
        else:
            image = Image.new("RGB", (req.width, req.height), req.background)
        doc = store.add(image, req.name, origin_label="Neues Bild", saved=False)
        return doc.info()

    @app.get("/api/documents/{doc_id}")
    def document_info(doc_id: str) -> dict[str, Any]:
        return get_doc(doc_id).info()

    @app.patch("/api/documents/{doc_id}")
    def rename(doc_id: str, req: RenameRequest) -> dict[str, Any]:
        doc = get_doc(doc_id)
        doc.name = req.name.strip()
        return doc.info()

    @app.delete("/api/documents/{doc_id}")
    def close(doc_id: str) -> dict[str, bool]:
        get_doc(doc_id)
        store.remove(doc_id)
        return {"ok": True}

    @app.get("/api/documents/{doc_id}/pixels")
    def pixels(
        doc_id: str,
        x: int = Query(0, ge=0),
        y: int = Query(0, ge=0),
        w: int | None = Query(None, ge=1),
        h: int | None = Query(None, ge=1),
    ) -> Response:
        doc = get_doc(doc_id)
        with doc.lock:
            image, version = doc.image, doc.version
        x0, y0 = min(x, image.width - 1), min(y, image.height - 1)
        x1 = image.width if w is None else min(image.width, x0 + w)
        y1 = image.height if h is None else min(image.height, y0 + h)
        if (x0, y0, x1, y1) != (0, 0, image.width, image.height):
            image = image.crop((x0, y0, x1, y1))
        return _raw_response(image, doc.meta, version)

    @app.get("/api/documents/{doc_id}/export")
    def export(doc_id: str, format: str = "png", quality: int = Query(92, ge=1, le=100)) -> Response:
        doc = get_doc(doc_id)
        if format not in EXPORT_FORMATS:
            raise HTTPException(400, f"Unbekanntes Format: {format}")
        with doc.lock:
            image, version = doc.image, doc.version
        data, ext, mime = export_image(image, doc.meta, format, quality)
        with doc.lock:
            if doc.version == version:
                doc.mark_saved()
        filename = f"{_stem(doc.name)}.{ext}"
        disposition = f"attachment; filename=\"{filename.encode('ascii', 'replace').decode()}\"; filename*=UTF-8''{quote(filename)}"
        return Response(data, media_type=mime, headers={"Content-Disposition": disposition, "Cache-Control": "no-store"})

    # -- Bearbeiten ------------------------------------------------------------------------
    @app.post("/api/documents/{doc_id}/preview")
    def preview(doc_id: str, req: OpRequest) -> dict[str, Any]:
        doc = get_doc(doc_id)
        op = get_operation(req.op)
        params = op.validate(req.params)
        with doc.lock:
            image, version = doc.image, doc.version
        result = run_operation(op, image, params, req.selection)
        if isinstance(result, NewImage):
            raise OperationError("Für diese Funktion gibt es keine Vorschau.")
        pending = doc.set_pending(result, op.label, version, _op_key(op.id, params, req.selection))
        if pending is None:
            raise VersionConflict("Das Bild wurde zwischenzeitlich geändert.")
        return _pending_json(pending)

    @app.get("/api/documents/{doc_id}/preview/{preview_id}/pixels")
    def preview_pixels(doc_id: str, preview_id: str) -> Response:
        doc = get_doc(doc_id)
        pending = doc.pending
        if pending is None or pending.id != preview_id:
            raise HTTPException(404, "Vorschau nicht mehr vorhanden.")
        return _raw_response(pending.change.result, doc.meta)

    @app.delete("/api/documents/{doc_id}/preview")
    def discard_preview(doc_id: str) -> dict[str, bool]:
        get_doc(doc_id).discard_pending()
        return {"ok": True}

    @app.post("/api/documents/{doc_id}/commit")
    def commit(doc_id: str, req: CommitRequest) -> dict[str, Any]:
        doc = get_doc(doc_id)
        return mutation(doc, doc.commit_pending(req.preview_id))

    @app.post("/api/documents/{doc_id}/apply")
    def apply(doc_id: str, req: OpRequest) -> dict[str, Any]:
        doc = get_doc(doc_id)
        op = get_operation(req.op)
        params = op.validate(req.params)
        pending = doc.take_matching_pending(_op_key(op.id, params, req.selection))
        if pending is not None:
            return mutation(doc, doc.commit_pending(pending.id))
        with doc.lock:
            image, version = doc.image, doc.version
        result = run_operation(op, image, params, req.selection)
        if isinstance(result, NewImage):
            new_doc = add_derived(doc, result.image, result.label)
            return mutation(doc, ChangeInfo(), new_document=new_doc.info())
        return mutation(doc, doc.apply(result, op.label, expected_version=version))

    @app.post("/api/documents/{doc_id}/undo")
    def undo(doc_id: str) -> dict[str, Any]:
        doc = get_doc(doc_id)
        return mutation(doc, doc.undo())

    @app.post("/api/documents/{doc_id}/redo")
    def redo(doc_id: str) -> dict[str, Any]:
        doc = get_doc(doc_id)
        return mutation(doc, doc.redo())

    @app.post("/api/documents/{doc_id}/history/{position}")
    def goto(doc_id: str, position: int) -> dict[str, Any]:
        doc = get_doc(doc_id)
        return mutation(doc, doc.goto(position))

    # -- KI --------------------------------------------------------------------------------
    @app.post("/api/ai/generate")
    def generate(req: GenerateRequest) -> dict[str, Any]:
        provider = registry.get(req.provider, "generate")
        opts = GenerateOptions(
            prompt=req.prompt,
            model=req.model,
            aspect_ratio=req.aspect_ratio,
            quality=req.quality,
            resolution=req.resolution,
            negative_prompt=req.negative_prompt or None,
            seed=req.seed,
            count=req.count,
            extra=_options(req.options, req.job_id),
        )
        images = provider.generate(opts)
        docs = []
        for i, image in enumerate(images, start=1):
            suffix = f" ({i})" if len(images) > 1 else ""
            docs.append(store.add(image, f"KI – {_short(req.prompt, 32)}{suffix}", origin_label="KI-generiert", saved=False))
        return {"documents": [d.info() for d in docs]}

    @app.post("/api/documents/{doc_id}/ai/edit")
    def ai_edit_route(doc_id: str, req: AiEditRequest) -> dict[str, Any]:
        doc = get_doc(doc_id)
        provider = registry.get(req.provider, "edit")
        with doc.lock:
            image, version = doc.image, doc.version
        change = ai_edit(provider, image, req.selection, req.prompt, req.model, options=_options(req.options, req.job_id))
        pending = doc.set_pending(change, f"KI: {_short(req.prompt, 28)}", version)
        if pending is None:
            raise VersionConflict("Das Bild wurde während der KI-Bearbeitung geändert. Bitte erneut versuchen.")
        return {"preview": _pending_json(pending)}

    @app.post("/api/documents/{doc_id}/ai/enhance")
    def ai_enhance_route(doc_id: str, req: AiEnhanceRequest) -> dict[str, Any]:
        doc = get_doc(doc_id)
        provider = registry.get(req.provider)
        with doc.lock:
            image, version = doc.image, doc.version
        result = ai_enhance(provider, image, req.selection, req.model, req.upscale_model, as_new=req.new_document)
        if isinstance(result, NewImage):
            return {"new_document": add_derived(doc, result.image, result.label).info()}
        pending = doc.set_pending(result, "KI-Schärfen", version)
        if pending is None:
            raise VersionConflict("Das Bild wurde während der KI-Bearbeitung geändert. Bitte erneut versuchen.")
        return {"preview": _pending_json(pending)}

    @app.get("/api/ai/providers")
    def providers(refresh: bool = False) -> dict[str, Any]:
        if refresh:
            comfyui_provider.forget_probes()
        return {"providers": registry.describe()}

    @app.post("/api/ai/cancel")
    def cancel_ai(req: CancelRequest | None = None) -> dict[str, int]:
        return {"cancelled": comfyui_provider.cancel(registry.client, req.job_id if req else None)}

    # -- Lokale KI ------------------------------------------------------------------------------
    local_store = registry.context["local_store"]
    workflows = WorkflowLibrary(settings.config_dir / "comfyui")

    def local_status() -> dict[str, Any]:
        return {
            "available": onnxruntime_available(),
            "device": local_store.device(),
            "models": local_store.status(),
            "directory": str(local_store.directory),
        }

    @app.get("/api/local")
    def get_local() -> dict[str, Any]:
        return local_status()

    @app.post("/api/local/models/{model_id}/download")
    def download_local_model(model_id: str) -> dict[str, Any]:
        if model_id not in LOCAL_MODELS:
            raise HTTPException(404, "Unbekanntes Modell.")
        local_store.start_download(LOCAL_MODELS[model_id])
        return local_status()

    def comfy_status() -> dict[str, Any]:
        info = comfyui_provider.ComfyUIProvider.availability(registry.context)
        return {
            **info["comfy"],
            "configured_url": settings.get("comfyui_url"),
            "status": info["status"],
            "checkpoints": info["comfy"].get("checkpoints", []),
        }

    @app.get("/api/comfyui")
    def get_comfyui(refresh: bool = False) -> dict[str, Any]:
        if refresh:
            comfyui_provider.forget_probes()
        return comfy_status()

    @app.put("/api/comfyui/url")
    def set_comfyui_url(req: ComfyUrlRequest) -> dict[str, Any]:
        settings.set("comfyui_url", (req.url or "").rstrip("/") or None)
        comfyui_provider.forget_probes()
        return comfy_status()

    @app.post("/api/comfyui/workflows")
    def add_workflow(file: UploadFile = File(...), name: str = Form("")) -> dict[str, Any]:
        raw = file.file.read(5 * 1024 * 1024 + 1)
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "Die Workflow-Datei ist zu groß.")
        try:
            graph = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise HTTPException(400, "Die Datei ist kein gültiges JSON.") from None
        title = name.strip() or _stem(file.filename or "Workflow")
        workflow = workflows.add(title, graph)
        return {"workflow": workflow.to_json(), **comfy_status()}

    @app.delete("/api/comfyui/workflows/{workflow_id}")
    def delete_workflow(workflow_id: str) -> dict[str, Any]:
        workflows.remove(workflow_id)
        return comfy_status()

    # -- Einstellungen -------------------------------------------------------------------------
    @app.get("/api/settings")
    def get_settings() -> dict[str, Any]:
        return {"providers": registry.describe(), "config_path": str(settings.path)}

    @app.put("/api/settings/keys/{provider_id}")
    def set_key(provider_id: str, req: ApiKeyRequest) -> dict[str, Any]:
        registry.key_status(provider_id)  # prüft, ob es den Anbieter gibt
        settings.set_api_key(provider_id, (req.api_key or "").strip() or None)
        return {"providers": registry.describe(), "config_path": str(settings.path)}

    return app
