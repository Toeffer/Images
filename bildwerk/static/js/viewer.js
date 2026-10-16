// Darstellung des Bildes mit Zoom/Verschieben, Vorschau-Ebene und Auswahl-Overlay.

import { paintShape, tracePath } from './selection.js';
import { clamp } from './util.js';

const ZOOM_STEPS = [0.02, 0.03, 0.05, 0.0625, 0.08, 0.1, 0.125, 1 / 6, 0.25, 1 / 3, 0.5, 2 / 3, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32];
const MIN_ZOOM = 0.01;
const MAX_ZOOM = 32;
const TINT = 'rgba(6, 8, 12, 0.42)';

export class Viewer extends EventTarget {
  constructor(viewport, mainCanvas, overlayCanvas) {
    super();
    this.viewport = viewport;
    this.main = mainCanvas;
    this.overlay = overlayCanvas;
    this.mainCtx = mainCanvas.getContext('2d');
    this.overlayCtx = overlayCanvas.getContext('2d');
    this.maskCanvas = document.createElement('canvas');
    this.tintCanvas = document.createElement('canvas');
    this.doc = null;
    this.scale = 1;
    this.ox = 0;
    this.oy = 0;
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.comparing = false; // true: Original statt Vorschau zeigen
    this.draft = null; // Form, die gerade gezogen wird: {shape, mode}
    this.cursor = null; // Pinselkreis: {x, y, r} in Bildkoordinaten
    this.ants = 0;
    this._tintDirty = true;
    this._needMain = false;
    this._needOverlay = false;
    this._frame = 0;

    new ResizeObserver(() => this.resize()).observe(viewport);
    viewport.addEventListener('wheel', (e) => this._onWheel(e), { passive: false });
    setInterval(() => {
      if (this.doc && !document.hidden && this._displayShapes().some((s) => s.mode === 'add')) {
        this.ants = (this.ants + 1) % 16;
        this.requestOverlay();
      }
    }, 120);
    this.resize();
  }

  // -- Koordinaten ----------------------------------------------------------------
  toImage(clientX, clientY) {
    const rect = this.viewport.getBoundingClientRect();
    return { x: (clientX - rect.left - this.ox) / this.scale, y: (clientY - rect.top - this.oy) / this.scale };
  }

  toScreen(x, y) {
    return [x * this.scale + this.ox, y * this.scale + this.oy];
  }

  // -- Dokument & Ansicht -----------------------------------------------------------
  setDocument(doc) {
    this.doc = doc;
    this.draft = null;
    this.cursor = null;
    this.comparing = false;
    if (doc?.view) {
      ({ scale: this.scale, ox: this.ox, oy: this.oy } = doc.view);
      this._changed(doc.view.fitted);
    } else if (doc) {
      this.fit();
    } else {
      this._changed();
    }
  }

  resize() {
    const rect = this.viewport.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    for (const canvas of [this.main, this.overlay, this.maskCanvas, this.tintCanvas]) {
      canvas.width = Math.max(1, Math.round(rect.width * this.dpr));
      canvas.height = Math.max(1, Math.round(rect.height * this.dpr));
    }
    if (this.doc && (!this.doc.view || this.doc.view.fitted)) this.fit();
    else this._changed();
  }

  fit() {
    const doc = this.doc;
    if (!doc || !this.width) return;
    const pad = 28;
    const s = Math.min((this.width - 2 * pad) / doc.width, (this.height - 2 * pad) / doc.height, 1);
    this.scale = clamp(s, MIN_ZOOM, MAX_ZOOM);
    this.ox = Math.round((this.width - doc.width * this.scale) / 2);
    this.oy = Math.round((this.height - doc.height * this.scale) / 2);
    this._changed(true);
  }

  zoomTo(scale, cx = this.width / 2, cy = this.height / 2) {
    if (!this.doc) return;
    const s = clamp(scale, MIN_ZOOM, MAX_ZOOM);
    const ix = (cx - this.ox) / this.scale;
    const iy = (cy - this.oy) / this.scale;
    this.scale = s;
    this.ox = Math.round(cx - ix * s);
    this.oy = Math.round(cy - iy * s);
    this._changed();
  }

  zoomStep(direction, cx, cy) {
    const current = this.scale;
    const next = direction > 0
      ? ZOOM_STEPS.find((z) => z > current * 1.001) ?? MAX_ZOOM
      : [...ZOOM_STEPS].reverse().find((z) => z < current / 1.001) ?? MIN_ZOOM;
    this.zoomTo(next, cx, cy);
  }

  panBy(dx, dy) {
    this.ox += dx;
    this.oy += dy;
    this._changed();
  }

  _changed(fitted = false) {
    if (this.doc) this.doc.view = { scale: this.scale, ox: this.ox, oy: this.oy, fitted };
    this.invalidateTint();
    this.requestRender();
    this.dispatchEvent(new Event('view'));
  }

  _onWheel(e) {
    if (!this.doc) return;
    e.preventDefault();
    const rect = this.viewport.getBoundingClientRect();
    if (e.shiftKey && !e.ctrlKey) {
      this.panBy(-(e.deltaY || e.deltaX), 0);
      return;
    }
    let dy = e.deltaY;
    if (e.deltaMode === 1) dy *= 16;
    else if (e.deltaMode === 2) dy *= 400;
    const factor = Math.exp(-clamp(dy, -200, 200) * (e.ctrlKey ? 0.01 : 0.0016));
    this.zoomTo(this.scale * factor, e.clientX - rect.left, e.clientY - rect.top);
  }

  // -- Zeichnen -------------------------------------------------------------------------
  invalidateTint() {
    this._tintDirty = true;
    this.requestOverlay();
  }

  requestRender() {
    this._needMain = true;
    this.requestOverlay();
  }

  requestOverlay() {
    this._needOverlay = true;
    if (!this._frame) {
      this._frame = requestAnimationFrame(() => {
        this._frame = 0;
        if (this._needMain) this._renderMain();
        if (this._needOverlay) this._renderOverlay();
        this._needMain = this._needOverlay = false;
      });
    }
  }

  _checker() {
    if (!this._pattern) {
      const tile = document.createElement('canvas');
      tile.width = tile.height = 16;
      const g = tile.getContext('2d');
      g.fillStyle = '#d9dbe0';
      g.fillRect(0, 0, 16, 16);
      g.fillStyle = '#b9bcc4';
      g.fillRect(0, 0, 8, 8);
      g.fillRect(8, 8, 8, 8);
      this._pattern = this.mainCtx.createPattern(tile, 'repeat');
    }
    return this._pattern;
  }

  _renderMain() {
    const ctx = this.mainCtx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.main.width, this.main.height);
    const doc = this.doc;
    if (!doc || !doc.canvas.width) return;

    const preview = this.comparing ? null : doc.preview;
    let rect = [0, 0, doc.width, doc.height];
    if (preview?.kind === 'replace') {
      const { width: pw, height: ph } = preview.canvas;
      rect = [(doc.width - pw) / 2, (doc.height - ph) / 2, pw, ph];
    }
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const [sx, sy] = this.toScreen(rect[0], rect[1]);
    ctx.fillStyle = this._checker();
    ctx.fillRect(sx, sy, rect[2] * this.scale, rect[3] * this.scale);

    ctx.save();
    ctx.translate(this.ox, this.oy);
    ctx.scale(this.scale, this.scale);
    ctx.imageSmoothingEnabled = this.scale < 2;
    ctx.imageSmoothingQuality = 'high';
    if (preview?.kind === 'replace') {
      ctx.drawImage(preview.canvas, rect[0], rect[1]);
    } else {
      ctx.drawImage(doc.canvas, 0, 0);
      if (preview) ctx.drawImage(preview.canvas, preview.box[0], preview.box[1]);
    }
    ctx.restore();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.strokeRect(sx - 0.5, sy - 0.5, rect[2] * this.scale + 1, rect[3] * this.scale + 1);
  }

  _displayShapes() {
    const shapes = this.doc?.selection.shapes ?? [];
    const d = this.draft;
    if (!d) return shapes;
    if (d.mode === 'new') return [{ ...d.shape, mode: 'add' }];
    return [...shapes, { ...d.shape, mode: d.mode }];
  }

  _buildTint(shapes) {
    const { dpr, scale, ox, oy, doc } = this;
    const m = this.maskCanvas.getContext('2d');
    m.setTransform(1, 0, 0, 1, 0, 0);
    m.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    m.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
    m.fillStyle = m.strokeStyle = '#fff';
    const identity = (x, y) => [x, y];
    for (const s of shapes) {
      m.globalCompositeOperation = s.mode === 'add' ? 'source-over' : 'destination-out';
      paintShape(m, s, identity, 1);
    }
    m.globalCompositeOperation = 'source-over';

    const t = this.tintCanvas.getContext('2d');
    t.setTransform(1, 0, 0, 1, 0, 0);
    t.clearRect(0, 0, this.tintCanvas.width, this.tintCanvas.height);
    t.fillStyle = TINT;
    t.fillRect(dpr * ox, dpr * oy, dpr * doc.width * scale, dpr * doc.height * scale);
    t.globalCompositeOperation = 'destination-out';
    const feather = doc.selection.feather;
    if (feather > 0) t.filter = `blur(${(feather / 2) * scale * dpr}px)`;
    t.drawImage(this.maskCanvas, 0, 0);
    t.filter = 'none';
    t.globalCompositeOperation = 'source-over';
    this._tintDirty = false;
  }

  _renderOverlay() {
    const ctx = this.overlayCtx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    if (!this.doc) return;

    const shapes = this._displayShapes();
    if (shapes.some((s) => s.mode === 'add') && !this.comparing) {
      if (this._tintDirty) this._buildTint(shapes);
      ctx.drawImage(this.tintCanvas, 0, 0);
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const map = (x, y) => this.toScreen(x, y);
      ctx.lineWidth = 1;
      for (const s of shapes) {
        if (s.type === 'brush') continue;
        tracePath(ctx, s, map, this.scale);
        ctx.setLineDash([]);
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.stroke();
        ctx.setLineDash([4, 4]);
        ctx.lineDashOffset = -this.ants;
        ctx.strokeStyle = s.mode === 'add' ? '#ffffff' : '#ff9a9a';
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    if (this.cursor) {
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const [cx, cy] = this.toScreen(this.cursor.x, this.cursor.y);
      const r = Math.max(2, this.cursor.r * this.scale);
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.lineWidth = 3;
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.stroke();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = '#fff';
      ctx.stroke();
    }
  }
}
