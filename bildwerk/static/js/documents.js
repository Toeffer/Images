// Geöffnete Bilder auf Browser-Seite: Pixelkopie für die Anzeige, Auswahl, Ansicht, Vorschau.

import { api } from './api.js';
import { Selection } from './selection.js';

export class ClientDoc {
  constructor(info) {
    this.info = info;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.selection = new Selection();
    this.view = null;
    this.preview = null; // {id, kind, box, canvas, source: 'op'|'ai', label}
    this.queue = Promise.resolve();
    this.blocked = false; // während einer KI-Bearbeitung gesperrt
  }

  get id() { return this.info.id; }
  get name() { return this.info.name; }
  get width() { return this.info.width; }
  get height() { return this.info.height; }
  get history() { return this.info.history; }
  get canUndo() { return this.info.history.position > 0; }
  get canRedo() { return this.info.history.position < this.info.history.labels.length - 1; }
}

export function imageDataToCanvas(data) {
  const canvas = document.createElement('canvas');
  canvas.width = data.width;
  canvas.height = data.height;
  canvas.getContext('2d').putImageData(data, 0, 0);
  return canvas;
}

export class DocumentManager {
  constructor(emit) {
    this.emit = emit;
    this.docs = [];
    this.active = null;
  }

  get(id) {
    return this.docs.find((d) => d.id === id) || null;
  }

  async open(info, { activate = true } = {}) {
    let doc = this.get(info.id);
    if (!doc) {
      doc = new ClientDoc(info);
      await this._loadFull(doc);
      this.docs.push(doc);
      this.emit('list');
    }
    if (activate) this.activate(doc.id);
    return doc;
  }

  activate(id) {
    const doc = this.get(id);
    if (doc === this.active) return;
    this.active = doc;
    this.emit('active', doc);
  }

  /** Führt Aufgaben je Dokument nacheinander aus, damit Pixel-Updates nicht durcheinandergeraten. */
  run(doc, task) {
    const result = doc.queue.then(task);
    doc.queue = result.catch(() => {});
    return result;
  }

  async _loadFull(doc) {
    const data = await api.pixels(doc.id);
    doc.canvas.width = data.width;
    doc.canvas.height = data.height;
    doc.ctx.putImageData(data, 0, 0);
  }

  async _loadRegion(doc, box) {
    const data = await api.pixels(doc.id, box);
    doc.ctx.putImageData(data, box[0], box[1]);
  }

  /** Übernimmt eine Server-Antwort ({document, change, new_document?}). */
  async applyResult(doc, result) {
    const { document: info, change } = result;
    const sizeChanged = info.width !== doc.width || info.height !== doc.height;
    doc.info = info;
    if (change.kind !== 'none') doc.preview = null;
    if (change.kind === 'patch') await this._loadRegion(doc, change.box);
    else if (change.kind === 'replace') await this._loadFull(doc);
    if (sizeChanged) {
      doc.selection = new Selection([], doc.selection.feather);
      doc.view = null;
    }
    this.emit('updated', doc, { sizeChanged });
    if (result.new_document) await this.open(result.new_document);
  }

  updateInfo(doc, info) {
    doc.info = info;
    this.emit('updated', doc, { sizeChanged: false });
    this.emit('list');
  }

  async close(doc) {
    try {
      await api.close(doc.id);
    } catch {
      /* Server kennt das Bild evtl. schon nicht mehr */
    }
    const index = this.docs.indexOf(doc);
    this.docs.splice(index, 1);
    this.emit('list');
    if (this.active === doc) {
      const next = this.docs[Math.min(index, this.docs.length - 1)];
      this.active = null;
      if (next) this.activate(next.id);
      else this.emit('active', null);
    }
  }

  /** Nach einem Neuladen der Seite: noch offene Bilder vom Server holen. */
  async restore() {
    const { documents } = await api.documents();
    for (const info of documents) await this.open(info, { activate: false });
    if (this.docs.length) this.activate(this.docs[this.docs.length - 1].id);
  }
}
