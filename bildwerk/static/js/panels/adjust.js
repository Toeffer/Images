// Reiter „Bearbeiten“: Liste der Operationen, Regler und Live-Vorschau.
// Die Regler werden aus den Parameterbeschreibungen des Servers erzeugt.

import { api } from '../api.js';
import { imageDataToCanvas } from '../documents.js';
import { clamp, debounce, el, errorMessage, icon, showError, storage } from '../util.js';

export class AdjustPanel {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.categories = app.info.categories;
    this.ops = new Map(this.categories.flatMap((c) => c.operations.map((op) => [op.id, op])));
    this.current = null;
    this.values = {};
    this.previewOn = storage.get('preview', true);
    this.previewToken = 0;
    this.inflight = false;
    this.queued = false;
    this.applying = false;
    this.paused = false; // nach „Anwenden“ keine erneute Vorschau, bis sich etwas ändert
    this.previewDoc = null;
    this.schedulePreview = debounce(() => this._requestPreview(), 160);

    app.on('active', () => this._onActiveChanged());
    app.on('updated', ({ doc }) => {
      if (doc !== app.docs.active) return;
      this._refreshScope();
      if (this.current && !this.applying) this._maybePreview();
    });
    app.on('selection', ({ doc }) => {
      if (doc === app.docs.active && this.current) {
        this.paused = false;
        this._refreshScope();
        this._maybePreview();
      }
    });
    app.on('preview', ({ doc }) => {
      if (doc === app.docs.active) this._refreshScope();
    });
    this.renderList();
  }

  // -- Übersicht ---------------------------------------------------------------------
  renderList() {
    this.current = null;
    const blocks = this.categories.map((cat) => [
      el('h3', {}, cat.label),
      el('div', { class: 'op-list' }, cat.operations.map((op) =>
        el('button', { class: 'op-item', title: op.description || op.label, onclick: () => this.open(op.id), dataset: { op: op.id } },
          el('span', {}, op.label),
          op.shortcut ? el('kbd', {}, op.shortcut) : null))),
    ]);
    this.root.replaceChildren(
      ...blocks.flat(),
      el('p', { class: 'hint' }, 'Tipp: Wähle vorher einen Bereich aus – dann wirken Anpassungen und Filter nur dort, mit weicher Kante, wenn eingestellt.'),
    );
  }

  open(opId) {
    const op = this.ops.get(opId);
    if (!op) return;
    this._clearPreview();
    this.current = op;
    this.paused = false;
    this.values = this._defaults(op);
    this.renderEditor();
    this._maybePreview();
  }

  close() {
    this._clearPreview();
    this.renderList();
  }

  /** Esc: Editor schließen. Gibt true zurück, wenn die Taste verbraucht wurde. */
  handleEscape() {
    if (!this.current) return false;
    this.close();
    return true;
  }

  isEditing() {
    return Boolean(this.current);
  }

  _defaults(op) {
    const doc = this.app.docs.active;
    const remembered = storage.get(`op.${op.id}`, {});
    const values = {};
    for (const p of op.params) {
      if (p.default_from && doc) values[p.name] = p.default_from === 'width' ? doc.width : doc.height;
      else if (p.name in remembered) values[p.name] = remembered[p.name];
      else values[p.name] = p.default;
    }
    return values;
  }

  // -- Editor --------------------------------------------------------------------------
  renderEditor() {
    const op = this.current;
    this.fields = new Map();
    this.scopeEl = el('div', { class: 'scope' });
    this.statusEl = el('div', { class: 'hint', 'aria-live': 'polite' });
    this.applyBtn = el('button', { class: 'btn primary', onclick: () => this.apply() }, icon('check'), 'Anwenden');
    const reset = op.params.length
      ? el('button', { class: 'btn', title: 'Alle Regler auf Standard', onclick: () => this.reset() }, 'Zurücksetzen')
      : null;
    const previewToggle = op.preview
      ? el('label', { class: 'check' },
          el('input', {
            type: 'checkbox',
            checked: this.previewOn,
            onchange: (e) => {
              this.previewOn = e.target.checked;
              storage.set('preview', this.previewOn);
              this._maybePreview();
            },
          }),
          'Live-Vorschau')
      : null;

    this.root.replaceChildren(
      el('div', { class: 'editor-head' },
        el('button', { class: 'icon-btn', title: 'Zurück zur Übersicht (Esc)', onclick: () => this.close() }, icon('chevron-left')),
        el('h2', {}, op.label)),
      op.description ? el('p', { class: 'editor-desc' }, op.description) : null,
      this.scopeEl,
      ...op.params.map((p) => this._field(p)),
      previewToggle,
      el('div', { class: 'editor-actions' }, reset, this.applyBtn),
      this.statusEl,
    );
    this._refreshVisibility();
    this._refreshScope();
  }

  _field(p) {
    const value = this.values[p.name];
    let control;
    let head = el('label', {}, p.label);
    if (p.type === 'float' || p.type === 'int') {
      const step = p.step ?? (p.type === 'int' ? 1 : 0.1);
      const number = el('input', { type: 'number', min: p.min, max: p.max, step, value, 'aria-label': p.label });
      const wide = p.default_from || (p.max - p.min) > 5000;
      const range = wide ? null : el('input', { type: 'range', min: p.min, max: p.max, step, value, 'aria-label': p.label });
      const commit = (raw) => {
        let v = Number(raw);
        if (!Number.isFinite(v)) v = p.default;
        v = clamp(v, p.min ?? -Infinity, p.max ?? Infinity);
        if (p.type === 'int') v = Math.round(v);
        number.value = v;
        if (range) range.value = v;
        this._set(p.name, v);
      };
      number.addEventListener('change', () => commit(number.value));
      if (range) {
        range.addEventListener('input', () => {
          number.value = range.value;
          this._set(p.name, Number(range.value));
        });
        range.addEventListener('dblclick', () => commit(p.default));
      }
      head = el('div', { class: 'field-head' }, el('label', {}, p.label), el('span', { class: 'field-value' }, number, p.unit ? el('span', {}, p.unit) : null));
      control = range;
    } else if (p.type === 'bool') {
      const box = el('input', { type: 'checkbox', checked: Boolean(value), onchange: (e) => this._set(p.name, e.target.checked) });
      const wrap = el('div', { class: 'field', dataset: { param: p.name } },
        el('label', { class: 'check' }, box, p.label),
        p.help ? el('div', { class: 'field-help' }, p.help) : null);
      this.fields.set(p.name, wrap);
      return wrap;
    } else if (p.type === 'choice') {
      control = el('select', { 'aria-label': p.label, onchange: (e) => this._set(p.name, e.target.value) },
        p.choices.map((c) => el('option', { value: c.value, selected: c.value === value }, c.label)));
    } else if (p.type === 'color') {
      control = el('input', { type: 'color', value, oninput: (e) => this._set(p.name, e.target.value) });
    }
    const wrap = el('div', { class: 'field', dataset: { param: p.name } }, head, control, p.help ? el('div', { class: 'field-help' }, p.help) : null);
    this.fields.set(p.name, wrap);
    return wrap;
  }

  _set(name, value) {
    this.values[name] = value;
    this.paused = false;
    const op = this.current;
    const remember = Object.fromEntries(op.params.filter((p) => !p.default_from).map((p) => [p.name, this.values[p.name]]));
    storage.set(`op.${op.id}`, remember);
    this._refreshVisibility();
    this._maybePreview();
  }

  reset() {
    const op = this.current;
    storage.set(`op.${op.id}`, {});
    this.paused = false;
    this.values = this._defaults(op);
    this.renderEditor();
    this._maybePreview();
  }

  _refreshVisibility() {
    for (const p of this.current?.params ?? []) {
      if (!p.show_if) continue;
      const visible = Object.entries(p.show_if).every(([key, allowed]) => allowed.includes(this.values[key]));
      this.fields.get(p.name).hidden = !visible;
    }
  }

  _refreshScope() {
    const op = this.current;
    if (!op || !this.scopeEl) return;
    const doc = this.app.docs.active;
    const hasSel = doc && !doc.selection.empty;
    let text;
    let warn = false;
    if (!doc) {
      text = 'Öffne zuerst ein Bild.';
      warn = true;
    } else if (doc.preview?.source === 'ai') {
      text = 'Bitte zuerst das KI-Ergebnis übernehmen oder verwerfen.';
      warn = true;
    } else if (op.requires_selection && !hasSel) {
      text = 'Bitte zuerst einen Bereich auswählen.';
      warn = true;
    } else if (op.kind === 'filter') {
      text = hasSel ? 'Wirkt nur auf die Auswahl.' : 'Wirkt auf das ganze Bild (keine Auswahl).';
    } else if (op.kind === 'new_document') {
      text = `${hasSel ? 'Nimmt den ausgewählten Bereich.' : 'Nimmt das ganze Bild.'} Das Ergebnis öffnet sich als neues Bild.`;
    } else {
      text = op.id === 'crop' ? 'Schneidet auf das Rechteck um die Auswahl zu.' : 'Wirkt auf das ganze Bild.';
    }
    this.scopeEl.replaceChildren(icon(warn ? 'warning' : 'info'), el('span', {}, text));
    this.scopeEl.classList.toggle('warn', warn);
    this.applyBtn.disabled = warn || this.applying;
  }

  _setStatus(text, isError = false) {
    if (!this.statusEl) return;
    this.statusEl.textContent = text || '';
    this.statusEl.style.color = isError ? 'var(--danger)' : '';
  }

  // -- Vorschau --------------------------------------------------------------------------
  _previewAllowed() {
    const doc = this.app.docs.active;
    const op = this.current;
    return Boolean(
      doc && op?.preview && this.previewOn && !this.paused && doc.preview?.source !== 'ai' && !doc.blocked &&
      !(op.requires_selection && doc.selection.empty),
    );
  }

  _maybePreview() {
    if (this._previewAllowed()) this.schedulePreview();
    else this._clearPreview();
  }

  async _requestPreview() {
    if (!this._previewAllowed()) return;
    const doc = this.app.docs.active;
    const op = this.current;
    const token = ++this.previewToken;
    if (this.inflight) {
      this.queued = true;
      return;
    }
    this.inflight = true;
    this.previewDoc = doc;
    const slow = setTimeout(() => this._setStatus('Vorschau wird berechnet …'), 300);
    try {
      const request = { op: op.id, params: { ...this.values }, selection: doc.selection.toJSON() };
      const result = await api.preview(doc.id, request);
      const fresh = () => token === this.previewToken && doc === this.app.docs.active && op === this.current;
      if (fresh()) {
        const data = await api.previewPixels(doc.id, result.preview_id);
        if (fresh()) {
          doc.preview = { ...result, id: result.preview_id, canvas: imageDataToCanvas(data), source: 'op' };
          this.app.emit('preview', { doc });
          this._setStatus('');
        }
      }
    } catch (err) {
      if (token === this.previewToken) this._setStatus(errorMessage(err), true);
    } finally {
      clearTimeout(slow);
      this.inflight = false;
      if (this.queued) {
        this.queued = false;
        this._requestPreview();
      }
    }
  }

  _clearPreview() {
    this.previewToken++;
    this.queued = false;
    this.schedulePreview.cancel();
    this._setStatus('');
    for (const doc of new Set([this.previewDoc, this.app.docs.active])) {
      if (doc?.preview?.source === 'op') {
        doc.preview = null;
        this.app.emit('preview', { doc });
        api.discardPreview(doc.id).catch(() => {});
      }
    }
  }

  _onActiveChanged() {
    this._clearPreview();
    this.paused = false;
    if (!this.current) return;
    for (const p of this.current.params) {
      const doc = this.app.docs.active;
      if (p.default_from && doc) this.values[p.name] = p.default_from === 'width' ? doc.width : doc.height;
    }
    this.renderEditor();
    this._maybePreview();
  }

  // -- Anwenden --------------------------------------------------------------------------
  async apply() {
    const doc = this.app.docs.active;
    const op = this.current;
    if (!doc || !op || this.applying || this.applyBtn.disabled) return;
    this.previewToken++;
    this.schedulePreview.cancel();
    this.applying = true;
    this._refreshScope();
    const request = { op: op.id, params: { ...this.values }, selection: doc.selection.toJSON() };
    try {
      await this.app.mutate(doc, () => api.apply(doc.id, request), `${op.label} …`);
      this.paused = op.kind === 'filter';
      this._setStatus(this.paused ? 'Angewendet. Für eine neue Vorschau Regler oder Auswahl ändern.' : '');
    } catch (err) {
      showError(err);
    } finally {
      this.applying = false;
      if (this.current === op) {
        if (op.params.some((p) => p.default_from)) this._onActiveChanged();
        else this._refreshScope();
      }
    }
  }
}
