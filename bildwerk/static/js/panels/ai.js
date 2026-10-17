// Reiter „KI“: Bilder generieren, Auswahl per Text bearbeiten, KI-Schärfen.

import { api } from '../api.js';
import { imageDataToCanvas } from '../documents.js';
import { clamp, el, formatNumber, icon, showError, storage, toast } from '../util.js';

const CAPABILITY_LABELS = { generate: 'Bilder generieren', edit: 'Bereiche bearbeiten', upscale: 'Hochskalieren' };

/** Startanbieter: am liebsten einer, der Bilder generieren kann. */
function preferredProvider(providers) {
  return providers.find((p) => p.configured && p.capabilities.includes('generate'))
    || providers.find((p) => p.configured)
    || providers[0];
}

function providerLabel(p) {
  if (p.configured) return p.name;
  return p.requires_key ? `${p.name} – Schlüssel fehlt` : `${p.name} – nicht verfügbar`;
}

/** Kennung eines Auftrags, damit sich genau dieser abbrechen lässt. */
const newJobId = () => `job-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export class AiPanel {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    this.providers = app.info.providers;
    this.providerId = storage.get('ai.provider');
    this.models = storage.get('ai.models', {});
    this.options = storage.get('ai.options', {});
    this.drafts = { generate: '', negative: '', edit: '', editNegative: '', ...storage.get('ai.drafts', {}) };
    this.generating = null;
    this.lastRun = null;
    this.refreshedAt = Date.now();
    if (!this.providers.some((p) => p.id === this.providerId)) this.providerId = preferredProvider(this.providers)?.id;

    app.on('providers', ({ providers }) => {
      this.providers = providers;
      this.refreshedAt = Date.now();
      const current = this.provider;
      // Nur weiterwechseln, wenn ein Schlüssel fehlt – ein lokaler Dienst kann gleich wieder laufen
      if (!current || (!current.configured && current.requires_key)) {
        const next = preferredProvider(providers);
        if (!current || next?.configured) this.providerId = next?.id;
      }
      this.render();
    });
    app.on('panel', ({ id }) => {
      if (id === 'ai' && Date.now() - this.refreshedAt > 5000) this.refresh(true);
    });
    app.on('active', () => this._refreshDocState());
    app.on('selection', ({ doc }) => doc === app.docs.active && this._refreshDocState());
    app.on('updated', ({ doc }) => doc === app.docs.active && this._refreshDocState());
    this.render();
  }

  get provider() {
    return this.providers.find((p) => p.id === this.providerId) || null;
  }

  /** Modelle des Anbieters für eine Aufgabe; ohne Angabe gelten sie für alle Aufgaben. */
  _models(p, task) {
    return p.models.filter((m) => !m.capabilities || m.capabilities.includes(task));
  }

  _modelKey(p, task) {
    return p.model_scope === 'card' ? `${p.id}.${task}` : p.id;
  }

  _model(p, task) {
    const models = this._models(p, task);
    const stored = this.models[this._modelKey(p, task)];
    if (models.some((m) => m.id === stored)) return stored;
    if (models.some((m) => m.id === p.default_model)) return p.default_model;
    return models[0]?.id ?? (p.default_model || null);
  }

  _modelInfo(p, task) {
    const id = this._model(p, task);
    return this._models(p, task).find((m) => m.id === id) || null;
  }

  _option(key, fallback) {
    return this.options[`${this.providerId}.${key}`] ?? fallback;
  }

  _setOption(key, value) {
    this.options[`${this.providerId}.${key}`] = value;
    storage.set('ai.options', this.options);
  }

  /** Gespeicherte Auswahl – oder der Standard, falls es die Auswahl nicht mehr gibt. */
  _choice(key, values, fallback) {
    const current = this._option(key, fallback);
    return values.includes(current) ? current : fallback;
  }

  _draft(key, value) {
    this.drafts[key] = value;
    storage.set('ai.drafts', this.drafts);
  }

  focusPrompt() {
    if (!this.provider?.capabilities.includes('generate')) {
      // z. B. „Lokal auf diesem PC“ kann nur schärfen – zu einem Anbieter wechseln, der generiert
      const next = this.providers.find((p) => p.configured && p.capabilities.includes('generate'))
        || this.providers.find((p) => p.capabilities.includes('generate'));
      if (next) {
        this.providerId = next.id;
        this.render();
      }
    }
    this.genPrompt?.focus();
  }

  /** Fragt den Zustand der Anbieter neu ab (z. B. nachdem ComfyUI gestartet wurde). */
  async refresh(quiet = false) {
    try {
      const { providers } = await api.aiProviders(!quiet);
      this.app.emit('providers', { providers });
    } catch (err) {
      if (!quiet) showError(err);
    }
  }

  // -- Aufbau ---------------------------------------------------------------------------
  render() {
    const p = this.provider;
    this.genPrompt = this.genButton = this.seedInput = null;
    this.editPrompt = this.editScope = this.editButton = null;
    this.enhanceNew = this.enhanceScope = this.enhanceButton = null;
    this.genReady = this.editReady = true;
    const parts = [];

    if (!this.providers.some((x) => x.configured && x.capabilities.includes('generate'))) {
      parts.push(el('div', { class: 'callout' },
        el('strong', {}, 'Zum Generieren ist noch nichts eingerichtet.'),
        el('div', {}, 'Starte ComfyUI auf diesem PC (kostenlos, siehe Anleitung) oder trage einen API-Schlüssel von OpenAI, Google Gemini oder Stability AI ein. Das KI-Schärfen funktioniert auch ohne.'),
        el('button', { class: 'btn small', onclick: () => this.app.commands.settings() }, icon('settings'), 'Einstellungen öffnen')));
    }

    // Anbieter & Modell
    const providerSelect = el('select', {
      'aria-label': 'KI-Anbieter',
      onchange: (e) => {
        this.providerId = e.target.value;
        storage.set('ai.provider', this.providerId);
        this.render();
      },
    }, [['Lokal & kostenlos', true], ['Cloud (API-Schlüssel)', false]].map(([label, local]) => {
      const group = this.providers.filter((x) => Boolean(x.local) === local);
      return group.length
        ? el('optgroup', { label }, group.map((x) => el('option', { value: x.id, selected: x.id === this.providerId }, providerLabel(x))))
        : null;
    }));
    const providerCard = el('div', { class: 'card' },
      el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Anbieter')), providerSelect));
    if (p.model_scope !== 'card' && p.models.length) {
      this.modelHint = el('div', { class: 'hint' });
      const modelSelect = this._modelSelect(p, 'generate', () => this._renderModelHint());
      providerCard.append(el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Modell')), modelSelect, this.modelHint));
    } else {
      this.modelHint = null;
      if (p.note) providerCard.append(el('p', { class: 'hint' }, p.note));
    }
    providerCard.append(this._status(p));
    parts.push(providerCard);

    if (p.capabilities.includes('generate')) parts.push(this._generateCard(p));
    if (p.capabilities.includes('edit')) parts.push(this._editCard(p));
    if (p.enhance) parts.push(this._enhanceCard(p));
    parts.push(el('p', { class: 'hint' }, p.local
      ? 'Läuft auf deinem PC: kostenlos, und deine Bilder verlassen den Rechner nicht. Ergebnisse können vom Original abweichen – prüfe sie vor dem Übernehmen.'
      : 'KI-Anfragen werden an den gewählten Anbieter gesendet und dort abgerechnet. Ergebnisse können vom Original abweichen – prüfe sie vor dem Übernehmen.'));

    this.root.replaceChildren(...parts);
    this._renderModelHint();
    this._refreshDocState();
  }

  _status(p) {
    let text;
    let action = null;
    if (p.requires_key) {
      text = p.configured ? `Bereit · kann ${p.capabilities.map((c) => CAPABILITY_LABELS[c]).join(', ')}` : 'API-Schlüssel fehlt';
      if (!p.configured) action = el('button', { class: 'btn small', onclick: () => this.app.commands.settings() }, 'Eintragen');
    } else {
      text = p.status || (p.configured ? 'Bereit' : 'Nicht verfügbar');
      action = el('button', { class: 'icon-btn small', title: 'Erneut prüfen', 'aria-label': 'Erneut prüfen', onclick: () => this.refresh() }, icon('refresh'));
    }
    return el('div', { class: 'ai-status' }, el('span', { class: `dot${p.configured ? ' ok' : ''}` }), el('span', { class: 'grow' }, text), action);
  }

  _renderModelHint() {
    if (!this.modelHint) return;
    const p = this.provider;
    this.modelHint.textContent = [this._modelInfo(p, 'generate')?.description, p.note].filter(Boolean).join(' · ');
  }

  _modelSelect(p, task, onChange) {
    const current = this._model(p, task);
    return el('select', {
      'aria-label': 'Modell',
      onchange: (e) => {
        this.models[this._modelKey(p, task)] = e.target.value;
        storage.set('ai.models', this.models);
        onChange?.();
      },
    }, this._models(p, task).map((m) => el('option', { value: m.id, selected: m.id === current }, m.label)));
  }

  /** Modellauswahl innerhalb einer Karte (bei Anbietern mit Modellen je Aufgabe, z. B. ComfyUI). */
  _cardModel(p, task, params) {
    if (p.model_scope !== 'card' || !p.configured) return { field: null, params: null };
    const paramsBox = el('div');
    if (!this._models(p, task).length) {
      if (task === 'generate') this.genReady = false;
      else this.editReady = false;
      const text = task === 'generate'
        ? 'Keine Modelle gefunden. Lege ein Checkpoint-Modell (z. B. SDXL) in den ComfyUI-Ordner models/checkpoints oder importiere unter ⚙ Einstellungen einen Workflow (z. B. Z-Image Turbo).'
        : 'Kein Modell zum Bearbeiten gefunden – dafür eignet sich ein Checkpoint-Modell (z. B. SDXL) oder ein importierter Bearbeitungs-Workflow.';
      return { field: el('div', { class: 'scope warn' }, icon('warning'), el('span', {}, text)), params: null };
    }
    const hint = el('div', { class: 'hint' });
    const update = () => {
      const m = this._modelInfo(p, task);
      hint.textContent = m?.description || '';
      paramsBox.replaceChildren(...(m?.params ? this._paramFields(params) : []));
    };
    const field = el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Modell')), this._modelSelect(p, task, update), hint);
    update();
    return { field, params: paramsBox };
  }

  _paramFields(params) {
    const numbers = [];
    const sliders = [];
    for (const param of params) {
      const value = Number(this._option(param.name, param.default));
      const set = (v) => {
        if (Number.isFinite(v)) this._setOption(param.name, clamp(param.type === 'int' ? Math.round(v) : v, param.min, param.max));
      };
      if (param.slider) {
        const out = el('span', { class: 'field-value' }, formatNumber(value, param.step));
        const range = el('input', {
          type: 'range', min: param.min, max: param.max, step: param.step, value, 'aria-label': param.label,
          oninput: (e) => {
            out.textContent = formatNumber(Number(e.target.value), param.step);
            set(Number(e.target.value));
          },
        });
        sliders.push(el('div', { class: 'field' },
          el('div', { class: 'field-head' }, el('label', {}, param.label), out), range,
          param.help ? el('div', { class: 'field-help' }, param.help) : null));
      } else {
        numbers.push(el('div', { class: 'field' },
          el('div', { class: 'field-head' }, el('label', {}, param.label)),
          el('input', { type: 'number', min: param.min, max: param.max, step: param.step, value, 'aria-label': param.label, onchange: (e) => set(Number(e.target.value)) })));
      }
    }
    return [numbers.length ? el('div', { class: 'row-inline' }, numbers) : null, ...sliders];
  }

  _paramValues(params) {
    const result = {};
    for (const param of params || []) {
      const value = Number(this._option(param.name, param.default));
      if (Number.isFinite(value)) result[param.name] = clamp(param.type === 'int' ? Math.round(value) : value, param.min, param.max);
    }
    return result;
  }

  _select(label, key, choices, fallback) {
    const current = this._choice(key, choices.map((c) => c.value), fallback);
    return el('div', { class: 'field' },
      el('div', { class: 'field-head' }, el('label', {}, label)),
      el('select', { 'aria-label': label, onchange: (e) => this._setOption(key, e.target.value) },
        choices.map((c) => el('option', { value: c.value, selected: c.value === current }, c.label))));
  }

  _negativeField(key, placeholder) {
    return el('div', { class: 'field' },
      el('div', { class: 'field-head' }, el('label', {}, 'Nicht im Bild (optional)')),
      el('input', { type: 'text', value: this.drafts[key] || '', placeholder, oninput: (e) => this._draft(key, e.target.value) }));
  }

  _generateCard(p) {
    const model = this._cardModel(p, 'generate', p.checkpoint_params || []);
    this.genPrompt = el('textarea', {
      rows: 4,
      placeholder: 'Beschreibe das Bild, z. B. „Ein Leuchtturm an einer Steilküste bei Sonnenuntergang, Ölgemälde“',
      oninput: (e) => this._draft('generate', e.target.value),
      onkeydown: (e) => {
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) this.generate();
      },
    });
    this.genPrompt.value = this.drafts.generate || '';
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
      model.field,
      el('div', { class: 'field' }, this.genPrompt),
      p.supports_negative_prompt ? this._negativeField('negative', 'z. B. Text, Wasserzeichen, unscharf') : null,
      el('div', { class: 'row-inline' }, this._select('Format', 'aspect', ratios, defaultRatio(p)), extra),
      el('div', { class: 'row-inline' },
        this._select('Anzahl', 'count', counts, '1'),
        this.seedInput ? el('div', { class: 'field' }, el('div', { class: 'field-head' }, el('label', {}, 'Seed')), this.seedInput) : null),
      model.params,
      this.genButton);
  }

  _editCard(p) {
    const model = this._cardModel(p, 'edit', [...(p.checkpoint_params || []), ...(p.edit_params || [])]);
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
      model.field,
      el('div', { class: 'field' }, this.editPrompt),
      p.edit_negative_prompt ? this._negativeField('editNegative', 'z. B. Text, verschwommen') : null,
      model.params,
      this.editScope,
      this.editButton);
  }

  _upscaleModel(p) {
    const ids = p.upscale_models.map((m) => m.id);
    return ids.length ? this._choice('upscale_model', ids, ids[0]) : null;
  }

  _enhanceCard(p) {
    let upscale = null;
    if (p.capabilities.includes('upscale') && p.upscale_models.length) {
      const hint = el('div', { class: 'hint' });
      const update = () => {
        hint.textContent = p.upscale_models.find((m) => m.id === this._upscaleModel(p))?.description || '';
      };
      const current = this._upscaleModel(p);
      upscale = el('div', { class: 'field' },
        el('div', { class: 'field-head' }, el('label', {}, 'Verfahren')),
        el('select', {
          'aria-label': 'Verfahren',
          onchange: (e) => {
            this._setOption('upscale_model', e.target.value);
            update();
          },
        }, p.upscale_models.map((m) => el('option', { value: m.id, selected: m.id === current }, m.label))),
        hint);
      update();
    }
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
    if (this.genButton && !this.generating) this.genButton.disabled = !p.configured || !this.genReady;
    const scopeText = !doc ? 'Öffne zuerst ein Bild.' : hasSel ? 'Wirkt auf die Auswahl.' : 'Keine Auswahl – das ganze Bild wird verwendet.';
    for (const [scope, button, ready] of [[this.editScope, this.editButton, this.editReady], [this.enhanceScope, this.enhanceButton, true]]) {
      if (!scope || !button) continue;
      scope.replaceChildren(icon(doc ? 'info' : 'warning'), el('span', {}, scopeText));
      scope.classList.toggle('warn', !doc);
      button.disabled = !doc || !p.configured || !ready || doc.blocked;
    }
  }

  // -- Aktionen ------------------------------------------------------------------------------
  async generate() {
    const p = this.provider;
    if (this.generating) {
      this.generating.cancel();
      return;
    }
    if (!this.genButton || this.genButton.disabled) return;
    const prompt = this.genPrompt.value.trim();
    if (!prompt) {
      toast('Bitte beschreibe zuerst, was auf dem Bild zu sehen sein soll.');
      this.genPrompt.focus();
      return;
    }
    const seed = this.seedInput?.value ? Number(this.seedInput.value) : null;
    const jobId = newJobId();
    const request = {
      provider: p.id,
      job_id: jobId,
      model: this._model(p, 'generate'),
      prompt,
      negative_prompt: p.supports_negative_prompt ? (this.drafts.negative || '').trim() || null : null,
      aspect_ratio: this._choice('aspect', p.aspect_ratios, defaultRatio(p)),
      quality: p.qualities.length ? this._choice('quality', p.qualities.map((q) => q.value), 'auto') : null,
      resolution: p.resolutions.length ? this._choice('resolution', p.resolutions.map((r) => r.value), p.resolutions[0].value) : null,
      count: Number(this._option('count', '1')),
      seed,
      options: this._modelInfo(p, 'generate')?.params ? this._paramValues(p.checkpoint_params) : {},
    };

    const controller = new AbortController();
    this.generating = {
      cancel: () => {
        controller.abort();
        if (p.local) api.cancelAi(jobId).catch(() => {});
      },
    };
    const start = Date.now();
    const label = () => `Generiere … ${Math.round((Date.now() - start) / 1000)} s (klicken zum Abbrechen)`;
    this.genButton.replaceChildren(el('span', { class: 'spinner tiny' }), label());
    const timer = setInterval(() => this.genButton?.replaceChildren(el('span', { class: 'spinner tiny' }), label()), 1000);
    try {
      const result = await api.generate(request, controller.signal);
      for (const info of result.documents) await this.app.docs.open(info);
      toast(result.documents.length === 1 ? 'Bild erzeugt.' : `${result.documents.length} Bilder erzeugt.`, 'success', 2500);
    } catch (err) {
      showError(err);
    } finally {
      clearInterval(timer);
      this.generating = null;
      this.genButton?.replaceChildren(icon('sparkles'), 'Generieren');
      this._refreshDocState();
    }
  }

  edit() {
    const doc = this.app.docs.active;
    const p = this.provider;
    if (!doc || !this.editPrompt) return;
    const prompt = this.editPrompt.value.trim();
    if (!prompt) {
      toast('Bitte beschreibe, was im Bereich passieren soll.');
      this.editPrompt.focus();
      return;
    }
    const options = this._modelInfo(p, 'edit')?.params
      ? this._paramValues([...(p.checkpoint_params || []), ...(p.edit_params || [])])
      : {};
    const negative = (this.drafts.editNegative || '').trim();
    if (p.edit_negative_prompt && negative) options.negative_prompt = negative;
    const request = { provider: p.id, model: this._model(p, 'edit'), prompt, selection: doc.selection.toJSON(), options };
    this._runOnDoc(doc, 'KI bearbeitet den Bereich …', (signal, jobId) => api.aiEdit(doc.id, { ...request, job_id: jobId }, signal), { cancellable: p.local });
  }

  enhance() {
    const doc = this.app.docs.active;
    if (!doc) return;
    const p = this.provider;
    const upscaleModel = this._upscaleModel(p);
    const request = {
      provider: p.id,
      model: this._model(p, 'edit'),
      upscale_model: upscaleModel,
      selection: doc.selection.toJSON(),
      new_document: this.enhanceNew.checked,
    };
    // Beim ersten Mal lädt der lokale Anbieter das Modell herunter – den Fortschritt anzeigen
    const spec = p.upscale_models.find((m) => m.id === upscaleModel);
    const download = p.local && spec?.installed === false ? spec : null;
    this._runOnDoc(doc, download ? `Lade „${download.label}“ (einmalig) …` : 'KI schärft nach …', (signal) => api.aiEnhance(doc.id, request, signal), {
      watch: download ? (update) => watchDownload(download, update) : null,
      after: download ? () => this.refresh(true) : null,
    });
  }

  retry() {
    if (this.lastRun && this.app.docs.get(this.lastRun.doc.id)) this._runOnDoc(...this.lastRun.args);
  }

  async _runOnDoc(doc, text, call, extra = {}) {
    if (doc.blocked) return;
    const { cancellable = false, watch = null, after = null } = extra;
    const controller = new AbortController();
    const jobId = newJobId();
    this.lastRun = { doc, args: [doc, text, call, extra] };
    const task = {
      text,
      start: Date.now(),
      cancel: () => {
        controller.abort();
        if (cancellable) api.cancelAi(jobId).catch(() => {});
      },
    };
    this.app.setTask(doc, task);
    this._refreshDocState();
    const stopWatch = watch?.((message) => {
      task.text = message;
      if (doc.task === task) this.app.setTask(doc, task);
    });
    try {
      const result = await call(controller.signal, jobId);
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
      stopWatch?.();
      this.app.setTask(doc, null);
      this._refreshDocState();
      after?.();
    }
  }
}

/** Fragt während des ersten Aufrufs regelmäßig den Download-Fortschritt eines lokalen Modells ab. */
function watchDownload(model, update) {
  let stopped = false;
  let timer = null;
  const poll = async () => {
    try {
      const status = await api.local();
      const entry = status.models.find((m) => m.id === model.id);
      if (stopped) return;
      if (entry?.progress != null) update(`Lade „${model.label}“ (einmalig) … ${Math.round(entry.progress * 100)} %`);
      else if (entry?.installed) update('KI schärft nach …');
    } catch {
      /* nächster Versuch */
    }
    if (!stopped) timer = setTimeout(poll, 700);
  };
  timer = setTimeout(poll, 300);
  return () => {
    stopped = true;
    clearTimeout(timer);
  };
}

const defaultRatio = (p) => (p.aspect_ratios.includes('1:1') ? '1:1' : p.aspect_ratios[0]);

function ratioLabel(ratio) {
  const [w, h] = ratio.split(':').map(Number);
  const kind = w === h ? 'Quadrat' : w > h ? 'Querformat' : 'Hochformat';
  return `${ratio.replace(':', ' : ')} · ${kind}`;
}
