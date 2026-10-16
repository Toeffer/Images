// Reiter „KI“: Bilder generieren, Auswahl per Text bearbeiten, KI-Schärfen.

import { api } from '../api.js';
import { imageDataToCanvas } from '../documents.js';
import { el, icon, showError, storage, toast } from '../util.js';

const CAPABILITY_LABELS = { generate: 'Bilder generieren', edit: 'Bereiche bearbeiten', upscale: 'Hochskalieren' };

export class AiPanel {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.providers = app.info.providers;
    this.providerId = storage.get('ai.provider');
    this.models = storage.get('ai.models', {});
    this.options = storage.get('ai.options', {});
    this.drafts = storage.get('ai.drafts', { generate: '', negative: '', edit: '' });
    this.generating = null;
    this.lastRun = null;
    if (!this.providers.some((p) => p.id === this.providerId)) {
      this.providerId = (this.providers.find((p) => p.configured) || this.providers[0])?.id;
    }

    app.on('providers', ({ providers }) => {
      this.providers = providers;
      if (!this.provider?.configured) {
        const configured = providers.find((p) => p.configured);
        if (configured) this.providerId = configured.id;
      }
      this.render();
    });
    app.on('active', () => this._refreshDocState());
    app.on('selection', ({ doc }) => doc === app.docs.active && this._refreshDocState());
    app.on('updated', ({ doc }) => doc === app.docs.active && this._refreshDocState());
    this.render();
  }

  get provider() {
    return this.providers.find((p) => p.id === this.providerId) || null;
  }

  get model() {
    const p = this.provider;
    const stored = this.models[p.id];
    return p.models.some((m) => m.id === stored) ? stored : p.default_model;
  }

  _option(key, fallback) {
    return this.options[`${this.providerId}.${key}`] ?? fallback;
  }

  _setOption(key, value) {
    this.options[`${this.providerId}.${key}`] = value;
    storage.set('ai.options', this.options);
  }

  _draft(key, value) {
    this.drafts[key] = value;
    storage.set('ai.drafts', this.drafts);
  }

  focusPrompt() {
    this.genPrompt?.focus();
  }

  // -- Aufbau ---------------------------------------------------------------------------
  render() {
    const p = this.provider;
    const anyConfigured = this.providers.some((x) => x.configured);
    this.genPrompt = this.genButton = this.seedInput = null;
    this.editPrompt = this.editScope = this.editButton = null;
    this.enhanceNew = this.enhanceScope = this.enhanceButton = null;
    const parts = [];

    if (!anyConfigured) {
      parts.push(el('div', { class: 'callout' },
        el('strong', {}, 'Noch kein KI-Anbieter eingerichtet.'),
        el('div', {}, 'Trage einen API-Schlüssel von OpenAI, Google Gemini oder Stability AI ein, um Bilder zu generieren oder per KI zu bearbeiten.'),
        el('button', { class: 'btn small', onclick: () => this.app.commands.settings() }, icon('key'), 'Einstellungen öffnen')));
    }

    // Anbieter & Modell
    const providerSelect = el('select', {
      'aria-label': 'KI-Anbieter',
      onchange: (e) => {
        this.providerId = e.target.value;
        storage.set('ai.provider', this.providerId);
        this.render();
      },
    }, this.providers.map((x) => el('option', { value: x.id, selected: x.id === this.providerId }, x.configured ? x.name : `${x.name} – Schlüssel fehlt`)));
    const modelSelect = el('select', {
      'aria-label': 'Modell',
      onchange: (e) => {
        this.models[p.id] = e.target.value;
        storage.set('ai.models', this.models);
        this._renderModelHint();
      },
    }, p.models.map((m) => el('option', { value: m.id, selected: m.id === this.model }, m.label)));
    this.modelHint = el('div', { class: 'hint' });
    parts.push(el('div', { class: 'card' },
      el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Anbieter')), providerSelect),
      el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Modell')), modelSelect, this.modelHint),
      el('div', { class: 'ai-status' },
        el('span', { class: `dot${p.configured ? ' ok' : ''}` }),
        p.configured ? `Bereit · kann ${p.capabilities.map((c) => CAPABILITY_LABELS[c]).join(', ')}` : 'API-Schlüssel fehlt',
        p.configured ? null : el('button', { class: 'btn small', onclick: () => this.app.commands.settings() }, 'Eintragen'))));

    if (p.capabilities.includes('generate')) parts.push(this._generateCard(p));
    if (p.capabilities.includes('edit')) parts.push(this._editCard(p));
    if (p.capabilities.includes('edit') || p.capabilities.includes('upscale')) parts.push(this._enhanceCard(p));
    parts.push(el('p', { class: 'hint' }, 'KI-Anfragen werden an den gewählten Anbieter gesendet und dort abgerechnet. Ergebnisse können vom Original abweichen – prüfe sie vor dem Übernehmen.'));

    this.root.replaceChildren(...parts);
    this._renderModelHint();
    this._refreshDocState();
  }

  _renderModelHint() {
    const p = this.provider;
    const m = p.models.find((x) => x.id === this.model);
    this.modelHint.textContent = [m?.description, p.note].filter(Boolean).join(' · ');
  }

  _select(label, key, choices, fallback) {
    const current = this._option(key, fallback);
    return el('div', { class: 'field' },
      el('div', { class: 'field-head' }, el('label', {}, label)),
      el('select', { 'aria-label': label, onchange: (e) => this._setOption(key, e.target.value) },
        choices.map((c) => el('option', { value: c.value, selected: c.value === current }, c.label))));
  }

  _generateCard(p) {
    this.genPrompt = el('textarea', {
      rows: 4,
      placeholder: 'Beschreibe das Bild, z. B. „Ein Leuchtturm an einer Steilküste bei Sonnenuntergang, Ölgemälde“',
      oninput: (e) => this._draft('generate', e.target.value),
      onkeydown: (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) this.generate();
      },
    });
    this.genPrompt.value = this.drafts.generate || '';
    const negative = p.supports_negative_prompt
      ? el('div', { class: 'field' },
          el('div', { class: 'field-head' }, el('label', {}, 'Nicht im Bild (optional)')),
          el('input', {
            type: 'text',
            value: this.drafts.negative || '',
            placeholder: 'z. B. Text, Wasserzeichen, unscharf',
            oninput: (e) => this._draft('negative', e.target.value),
          }))
      : null;
    const ratios = p.aspect_ratios.map((r) => ({ value: r, label: ratioLabel(r) }));
    const extra = p.qualities.length
      ? this._select('Qualität', 'quality', p.qualities, 'auto')
      : p.resolutions.length ? this._select('Auflösung', 'resolution', p.resolutions, p.resolutions[0].value) : null;
    const counts = Array.from({ length: p.max_count }, (_, i) => ({ value: String(i + 1), label: String(i + 1) }));
    this.seedInput = p.supports_seed
      ? el('input', { type: 'number', min: 0, max: 4294967294, placeholder: 'zufällig', value: this._option('seed', '') ?? '', onchange: (e) => this._setOption('seed', e.target.value) })
      : null;
    this.genButton = el('button', { class: 'btn ai block', onclick: () => this.generate() }, icon('sparkles'), 'Generieren');
    return el('div', { class: 'card' },
      el('h4', {}, icon('sparkles'), 'Bild generieren'),
      el('div', { class: 'field' }, this.genPrompt),
      negative,
      el('div', { class: 'row-inline' }, this._select('Format', 'aspect', ratios, '1:1'), extra),
      el('div', { class: 'row-inline' },
        this._select('Anzahl', 'count', counts, '1'),
        this.seedInput ? el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Seed')), this.seedInput) : null),
      this.genButton);
  }

  _editCard() {
    this.editPrompt = el('textarea', {
      rows: 3,
      placeholder: 'Was soll im ausgewählten Bereich passieren? z. B. „Ersetze den Himmel durch einen Sternenhimmel“ oder „Entferne die Person“',
      oninput: (e) => this._draft('edit', e.target.value),
      onkeydown: (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) this.edit();
      },
    });
    this.editPrompt.value = this.drafts.edit || '';
    this.editScope = el('div', { class: 'scope' });
    this.editButton = el('button', { class: 'btn ai block', onclick: () => this.edit() }, icon('sparkles'), 'Auswahl bearbeiten');
    return el('div', { class: 'card' },
      el('h4', {}, icon('brush'), 'Auswahl mit KI bearbeiten'),
      el('div', { class: 'field' }, this.editPrompt),
      this.editScope,
      this.editButton);
  }

  _enhanceCard(p) {
    const upscale = p.capabilities.includes('upscale') && p.upscale_models.length
      ? this._select('Verfahren', 'upscale_model', p.upscale_models.map((m) => ({ value: m.id, label: m.label })), p.upscale_models[0].id)
      : null;
    this.enhanceNew = el('input', { type: 'checkbox', checked: Boolean(this._option('enhance_new', false)), onchange: (e) => this._setOption('enhance_new', e.target.checked) });
    this.enhanceScope = el('div', { class: 'scope' });
    this.enhanceButton = el('button', { class: 'btn ai block', onclick: () => this.enhance() }, icon('sparkles'), 'KI-Schärfen');
    return el('div', { class: 'card' },
      el('h4', {}, icon('zoom-in'), 'KI-Schärfen & Vergrößern'),
      el('p', { class: 'hint' }, 'Stellt unscharfe oder verrauschte Bildausschnitte mit KI wieder her. Wähle dafür am besten einen überschaubaren Bereich aus, z. B. ein Gesicht oder ein Schild. Als neues Bild wird der Ausschnitt zusätzlich vergrößert.'),
      upscale,
      el('label', { class: 'check' }, this.enhanceNew, 'Als neues, vergrößertes Bild öffnen'),
      this.enhanceScope,
      this.enhanceButton);
  }

  _refreshDocState() {
    const doc = this.app.docs.active;
    const p = this.provider;
    const hasSel = doc && !doc.selection.empty;
    if (this.genButton && !this.generating) this.genButton.disabled = !p.configured;
    const scopeText = !doc ? 'Öffne zuerst ein Bild.' : hasSel ? 'Wirkt auf die Auswahl.' : 'Keine Auswahl – das ganze Bild wird verwendet.';
    for (const [scope, button] of [[this.editScope, this.editButton], [this.enhanceScope, this.enhanceButton]]) {
      if (!scope || !button) continue;
      scope.replaceChildren(icon(doc ? 'info' : 'warning'), el('span', {}, scopeText));
      scope.classList.toggle('warn', !doc);
      button.disabled = !doc || !p.configured || doc.blocked;
    }
  }

  // -- Aktionen ------------------------------------------------------------------------------
  async generate() {
    const p = this.provider;
    if (this.generating) {
      this.generating.abort();
      return;
    }
    const prompt = this.genPrompt.value.trim();
    if (!prompt) {
      toast('Bitte beschreibe zuerst, was auf dem Bild zu sehen sein soll.');
      this.genPrompt.focus();
      return;
    }
    const seed = this.seedInput?.value ? Number(this.seedInput.value) : null;
    const request = {
      provider: p.id,
      model: this.model,
      prompt,
      negative_prompt: p.supports_negative_prompt ? (this.drafts.negative || '').trim() || null : null,
      aspect_ratio: this._option('aspect', '1:1'),
      quality: p.qualities.length ? this._option('quality', 'auto') : null,
      resolution: p.resolutions.length ? this._option('resolution', p.resolutions[0].value) : null,
      count: Number(this._option('count', '1')),
      seed,
    };
    if (!p.aspect_ratios.includes(request.aspect_ratio)) request.aspect_ratio = p.aspect_ratios[0];

    const controller = new AbortController();
    this.generating = controller;
    const start = Date.now();
    const label = () => `Generiere … ${Math.round((Date.now() - start) / 1000)} s (klicken zum Abbrechen)`;
    this.genButton.replaceChildren(el('span', { class: 'spinner tiny' }), label());
    const timer = setInterval(() => this.genButton.replaceChildren(el('span', { class: 'spinner tiny' }), label()), 1000);
    try {
      const result = await api.generate(request, controller.signal);
      for (const info of result.documents) await this.app.docs.open(info);
      toast(result.documents.length === 1 ? 'Bild erzeugt.' : `${result.documents.length} Bilder erzeugt.`, 'success', 2500);
    } catch (err) {
      showError(err);
    } finally {
      clearInterval(timer);
      this.generating = null;
      this.genButton.replaceChildren(icon('sparkles'), 'Generieren');
      this._refreshDocState();
    }
  }

  edit() {
    const doc = this.app.docs.active;
    const prompt = this.editPrompt.value.trim();
    if (!doc) return;
    if (!prompt) {
      toast('Bitte beschreibe, was im Bereich passieren soll.');
      this.editPrompt.focus();
      return;
    }
    const request = { provider: this.providerId, model: this.model, prompt, selection: doc.selection.toJSON() };
    this._runOnDoc(doc, 'KI bearbeitet den Bereich …', (signal) => api.aiEdit(doc.id, request, signal));
  }

  enhance() {
    const doc = this.app.docs.active;
    if (!doc) return;
    const p = this.provider;
    const request = {
      provider: p.id,
      model: this.model,
      upscale_model: p.upscale_models.length ? this._option('upscale_model', p.upscale_models[0].id) : null,
      selection: doc.selection.toJSON(),
      new_document: this.enhanceNew.checked,
    };
    this._runOnDoc(doc, 'KI schärft nach …', (signal) => api.aiEnhance(doc.id, request, signal));
  }

  retry() {
    if (this.lastRun && this.app.docs.get(this.lastRun.doc.id)) this._runOnDoc(...this.lastRun.args);
  }

  async _runOnDoc(doc, text, call) {
    if (doc.blocked) return;
    const controller = new AbortController();
    this.lastRun = { doc, args: [doc, text, call] };
    this.app.setTask(doc, { text, start: Date.now(), cancel: () => controller.abort() });
    this._refreshDocState();
    try {
      const result = await call(controller.signal);
      if (result.new_document) {
        await this.app.docs.open(result.new_document);
        toast('Ergebnis als neues Bild geöffnet.', 'success', 2500);
      } else if (result.preview) {
        const data = await api.previewPixels(doc.id, result.preview.preview_id, controller.signal);
        doc.preview = { ...result.preview, id: result.preview.preview_id, canvas: imageDataToCanvas(data), source: 'ai' };
        this.app.emit('preview', { doc });
      }
    } catch (err) {
      showError(err);
    } finally {
      this.app.setTask(doc, null);
      this._refreshDocState();
    }
  }
}

function ratioLabel(ratio) {
  const [w, h] = ratio.split(':').map(Number);
  const kind = w === h ? 'Quadrat' : w > h ? 'Querformat' : 'Hochformat';
  return `${ratio.replace(':', ' : ')} · ${kind}`;
}
