// Werkzeuge: Hand (Verschieben), Rechteck, Ellipse, Lasso und Pinsel-Auswahl.

import { $, $$, storage } from './util.js';

const MAX_POINTS = 19000;
const SELECTION_TOOLS = ['rect', 'ellipse', 'lasso', 'brush'];

function decimate(points) {
  if (points.length <= MAX_POINTS) return points;
  const step = Math.ceil(points.length / MAX_POINTS);
  return points.filter((_, i) => i % step === 0 || i === points.length - 1);
}

export class Tools {
  constructor(app) {
    this.app = app;
    this.viewer = app.viewer;
    this.tool = storage.get('tool', 'rect');
    this.mode = 'new';
    this.brushSize = storage.get('brushSize', 40);
    this.spaceDown = false;
    this.drag = null;

    const surface = this.viewer.overlay;
    surface.addEventListener('pointerdown', (e) => this._down(e));
    surface.addEventListener('pointermove', (e) => this._move(e));
    surface.addEventListener('pointerup', (e) => this._up(e));
    surface.addEventListener('pointercancel', () => this._cancel());
    surface.addEventListener('pointerleave', () => {
      if (this.viewer.cursor) {
        this.viewer.cursor = null;
        this.viewer.requestOverlay();
      }
      app.status.cursor(null);
    });
    surface.addEventListener('contextmenu', (e) => e.preventDefault());

    for (const button of $$('#toolbar [data-tool]')) {
      button.addEventListener('click', () => this.setTool(button.dataset.tool));
    }
    for (const button of $$('#sel-mode [data-mode]')) {
      button.addEventListener('click', () => this.setMode(button.dataset.mode));
    }
    const brush = $('#brush-size');
    brush.value = this.brushSize;
    brush.addEventListener('input', () => this.setBrushSize(Number(brush.value)));
    this.setBrushSize(this.brushSize);
    this.setTool(this.tool);
  }

  setTool(tool) {
    this.tool = tool;
    storage.set('tool', tool);
    for (const button of $$('#toolbar [data-tool]')) button.classList.toggle('active', button.dataset.tool === tool);
    $('#tool-options [data-for="brush"]').hidden = tool !== 'brush';
    this._updateCursor();
  }

  setMode(mode) {
    this.mode = mode;
    for (const button of $$('#sel-mode [data-mode]')) button.classList.toggle('active', button.dataset.mode === mode);
  }

  setBrushSize(size) {
    this.brushSize = Math.max(2, Math.min(400, Math.round(size)));
    storage.set('brushSize', this.brushSize);
    $('#brush-size').value = this.brushSize;
    $('#brush-size-value').textContent = `${this.brushSize} px`;
    if (this.viewer.cursor) {
      this.viewer.cursor.r = this.brushSize / 2;
      this.viewer.requestOverlay();
    }
  }

  setSpace(down) {
    this.spaceDown = down;
    this._updateCursor();
  }

  _updateCursor() {
    let cursor = 'crosshair';
    if (this.drag?.type === 'pan') cursor = 'grabbing';
    else if (this.spaceDown || this.tool === 'move') cursor = 'grab';
    else if (this.tool === 'brush') cursor = 'brush';
    if (cursor === 'crosshair') delete this.viewer.viewport.dataset.cursor;
    else this.viewer.viewport.dataset.cursor = cursor;
  }

  _modeFor(event) {
    if (event.shiftKey) return 'add';
    if (event.altKey) return 'subtract';
    return this.mode;
  }

  _down(e) {
    const doc = this.app.docs.active;
    if (!doc || this.drag) return;
    const panning = e.button === 1 || (e.button === 0 && (this.spaceDown || this.tool === 'move'));
    if (!panning && (e.button !== 0 || doc.blocked)) return;
    e.preventDefault();
    this.viewer.overlay.setPointerCapture(e.pointerId);

    if (panning) {
      this.drag = { type: 'pan', x: e.clientX, y: e.clientY };
      this._updateCursor();
      return;
    }
    const p = this.viewer.toImage(e.clientX, e.clientY);
    const mode = this._modeFor(e);
    if (this.tool === 'rect' || this.tool === 'ellipse') {
      this.drag = { type: this.tool, start: p, mode, sx: e.clientX, sy: e.clientY };
    } else if (this.tool === 'lasso') {
      this.drag = { type: 'lasso', points: [[p.x, p.y]], mode, sx: e.clientX, sy: e.clientY };
    } else if (this.tool === 'brush') {
      this.drag = { type: 'brush', points: [[p.x, p.y]], mode, size: this.brushSize };
    }
    this._updateDraft(e);
  }

  _move(e) {
    const doc = this.app.docs.active;
    if (!doc) return;
    const p = this.viewer.toImage(e.clientX, e.clientY);
    this.app.status.cursor(p);
    if (this.tool === 'brush' && !this.spaceDown && this.drag?.type !== 'pan') {
      this.viewer.cursor = { x: p.x, y: p.y, r: this.brushSize / 2 };
      this.viewer.requestOverlay();
    } else if (this.viewer.cursor) {
      this.viewer.cursor = null;
      this.viewer.requestOverlay();
    }

    const drag = this.drag;
    if (!drag) return;
    if (drag.type === 'pan') {
      this.viewer.panBy(e.clientX - drag.x, e.clientY - drag.y);
      drag.x = e.clientX;
      drag.y = e.clientY;
      return;
    }
    if (drag.type === 'lasso') {
      const last = drag.last || [drag.sx, drag.sy];
      if (Math.hypot(e.clientX - last[0], e.clientY - last[1]) >= 2) {
        drag.points.push([p.x, p.y]);
        drag.last = [e.clientX, e.clientY];
      }
    } else if (drag.type === 'brush') {
      const [lx, ly] = drag.points[drag.points.length - 1];
      if (Math.hypot(p.x - lx, p.y - ly) >= Math.max(0.75, drag.size / 8)) drag.points.push([p.x, p.y]);
    } else {
      drag.end = p;
    }
    this._updateDraft(e);
  }

  _rectFrom(drag, constrain) {
    const doc = this.app.docs.active;
    const a = drag.start;
    const b = drag.end || a;
    let w = b.x - a.x;
    let h = b.y - a.y;
    if (constrain) {
      const side = Math.max(Math.abs(w), Math.abs(h));
      w = Math.sign(w || 1) * side;
      h = Math.sign(h || 1) * side;
    }
    let x0 = Math.min(a.x, a.x + w), y0 = Math.min(a.y, a.y + h);
    let x1 = Math.max(a.x, a.x + w), y1 = Math.max(a.y, a.y + h);
    if (drag.type === 'rect') {
      // Rechtecke pixelgenau und innerhalb des Bildes
      x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
      x1 = Math.min(doc.width, Math.round(x1)); y1 = Math.min(doc.height, Math.round(y1));
    }
    return { type: drag.type, x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
  }

  _shapeOf(drag, event) {
    if (drag.type === 'rect' || drag.type === 'ellipse') return this._rectFrom(drag, event?.shiftKey && drag.mode !== 'add');
    if (drag.type === 'lasso') return drag.points.length >= 3 ? { type: 'polygon', points: decimate(drag.points) } : null;
    return { type: 'brush', points: decimate(drag.points), size: drag.size };
  }

  _updateDraft(event) {
    const shape = this.drag && this.drag.type !== 'pan' ? this._shapeOf(this.drag, event) : null;
    this.viewer.draft = shape ? { shape, mode: this.drag.mode } : null;
    this.viewer.invalidateTint();
  }

  _up(e) {
    const drag = this.drag;
    if (!drag) return;
    this.drag = null;
    this.viewer.overlay.releasePointerCapture?.(e.pointerId);
    this._updateCursor();
    if (drag.type === 'pan') return;

    const doc = this.app.docs.active;
    const shape = this._shapeOf(drag, e);
    this.viewer.draft = null;
    const moved = drag.type === 'brush' || Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 3;
    const tooSmall = !shape || (shape.type !== 'polygon' && shape.type !== 'brush' && (shape.w < 1 || shape.h < 1));
    if (!moved || tooSmall) {
      // Einfacher Klick mit neuer Auswahl: Auswahl aufheben
      if (drag.mode === 'new') this.app.setSelection(doc, []);
      else this.viewer.invalidateTint();
      return;
    }
    const entry = { ...shape, mode: drag.mode === 'subtract' ? 'subtract' : 'add' };
    const shapes = drag.mode === 'new' ? [entry] : [...doc.selection.shapes, entry];
    this.app.setSelection(doc, shapes);
  }

  _cancel() {
    this.drag = null;
    this.viewer.draft = null;
    this.viewer.invalidateTint();
    this._updateCursor();
  }

  isSelectionTool() {
    return SELECTION_TOOLS.includes(this.tool);
  }
}
