// Dialoge: Neues Bild, Speichern, Einstellungen, Bestätigung.

import { api } from './api.js';
import { $, el, icon, showError, storage, toast } from './util.js';

function stem(name) {
  const i = name.lastIndexOf('.');
  return (i > 0 ? name.slice(0, i) : name).trim() || 'Bild';
}

/** Öffnet einen <dialog> und liefert dessen returnValue, sobald er geschlossen wird. */
function show(dialog) {
  return new Promise((resolve) => {
    dialog.returnValue = '';
    dialog.addEventListener('close', () => resolve(dialog.returnValue), { once: true });
    dialog.showModal();
  });
}

export async function confirmDialog(title, text, okLabel = 'OK') {
  const dialog = $('#dlg-confirm');
  dialog.querySelector('[data-title]').textContent = title;
  dialog.querySelector('[data-text]').textContent = text;
  dialog.querySelector('[data-ok]').textContent = okLabel;
  return (await show(dialog)) === 'ok';
}

export async function openNewDialog(app) {
  const dialog = $('#dlg-new');
  const form = dialog.querySelector('form');
  form.preset.onchange = () => {
    if (form.preset.value === 'custom') return;
    const [w, h] = form.preset.value.split('x');
    form.width.value = w;
    form.height.value = h;
  };
  form.width.oninput = form.height.oninput = () => {
    const key = `${form.width.value}x${form.height.value}`;
    form.preset.value = [...form.preset.options].some((o) => o.value === key) ? key : 'custom';
  };
  form.color.oninput = () => {
    form.querySelector('input[name=bg][value=custom]').checked = true;
  };
  if ((await show(dialog)) !== 'ok') return;
  const choice = form.querySelector('input[name=bg]:checked').value;
  try {
    const info = await api.newDocument({
      width: Number(form.width.value),
      height: Number(form.height.value),
      background: choice === 'custom' ? form.color.value : choice,
      name: 'Unbenannt',
    });
    await app.docs.open(info);
  } catch (err) {
    showError(err);
  }
}

function defaultFormat(doc, formats) {
  const remembered = storage.get('export.format');
  const bySource = { JPEG: 'jpeg', MPO: 'jpeg', PNG: 'png', WEBP: 'webp', AVIF: 'avif', TIFF: 'tiff' }[doc.info.source_format];
  const wanted = doc.info.has_alpha ? 'png' : bySource || remembered || 'png';
  return formats.find((f) => f.id === wanted) ? wanted : 'png';
}

export async function openExportDialog(app, doc) {
  const dialog = $('#dlg-export');
  const form = dialog.querySelector('form');
  const formats = app.info.export_formats;
  if (!form.format.options.length) {
    for (const f of formats) form.format.append(el('option', { value: f.id }, `${f.label} (.${f.ext})`));
  }
  form.filename.value = stem(doc.name);
  form.format.value = defaultFormat(doc, formats);
  form.quality.value = storage.get('export.quality', 92);
  const update = () => {
    const f = formats.find((x) => x.id === form.format.value);
    dialog.querySelector('[data-quality]').hidden = !f.lossy;
    dialog.querySelector('[data-quality-value]').textContent = form.quality.value;
    dialog.querySelector('[data-alpha-hint]').hidden = !(doc.info.has_alpha && !f.alpha);
  };
  form.format.onchange = update;
  form.quality.oninput = update;
  update();
  if ((await show(dialog)) !== 'ok') return;

  const format = formats.find((f) => f.id === form.format.value);
  const quality = Number(form.quality.value);
  storage.set('export.format', format.id);
  storage.set('export.quality', quality);
  const filename = `${stem(form.filename.value)}.${format.ext}`;
  await saveDocument(app, doc, format, quality, filename);
}

async function saveDocument(app, doc, format, quality, filename) {
  let handle = null;
  if (window.showSaveFilePicker) {
    try {
      handle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: format.label, accept: { [mimeOf(format)]: [`.${format.ext}`] } }],
      });
    } catch (err) {
      if (err.name === 'AbortError') return;
      handle = null; // z. B. ohne Benutzeraktion – dann normaler Download
    }
  }
  const stop = app.status.busy('Speichere …');
  try {
    const blob = await api.export(doc.id, format.id, quality);
    if (handle) {
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
    } else {
      const url = URL.createObjectURL(blob);
      const link = el('a', { href: url, download: filename });
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 30_000);
    }
    app.docs.updateInfo(doc, await api.document(doc.id));
    toast(`Gespeichert: ${handle?.name || filename}`, 'success', 2500);
  } catch (err) {
    showError(err);
  } finally {
    stop();
  }
}

function mimeOf(format) {
  return { png: 'image/png', jpeg: 'image/jpeg', webp: 'image/webp', avif: 'image/avif', tiff: 'image/tiff' }[format.id];
}

const CAPABILITIES = { generate: 'Bilder generieren', edit: 'Bereiche bearbeiten', upscale: 'KI-Hochskalieren' };

export async function openSettingsDialog(app) {
  const dialog = $('#dlg-settings');
  const cloudBox = $('#settings-providers');
  const localBox = $('#settings-local');
  let data;
  let local = null;
  let comfy = null;
  try {
    [data, local, comfy] = await Promise.all([api.settings(), api.local().catch(() => null), api.comfyui().catch(() => null)]);
  } catch (err) {
    showError(err);
    return;
  }
  let open = true;
  let pollTimer = null;

  // Das KI-Panel soll Änderungen sofort sehen
  const refreshProviders = async () => {
    try {
      const { providers } = await api.aiProviders();
      data.providers = providers;
      app.emit('providers', { providers });
    } catch {
      /* egal – beim nächsten Öffnen */
    }
  };

  // -- Lokale KI ------------------------------------------------------------------------------
  const pollDownloads = () => {
    clearTimeout(pollTimer);
    if (!open || !local?.models.some((m) => m.progress != null)) return;
    pollTimer = setTimeout(async () => {
      try {
        const before = local.models.filter((m) => m.installed).length;
        local = await api.local();
        renderLocal();
        if (local.models.filter((m) => m.installed).length !== before) refreshProviders();
      } catch {
        /* nächster Versuch */
      }
      pollDownloads();
    }, 700);
  };

  const download = async (model) => {
    try {
      local = await api.downloadLocalModel(model.id);
      renderLocal();
      pollDownloads();
    } catch (err) {
      showError(err);
    }
  };

  const localCard = () => {
    if (!local) return null;
    const rows = local.models.map((m) => {
      let state;
      if (m.installed) state = el('span', { class: 'badge ok' }, icon('check'), 'Geladen');
      else if (m.progress != null) state = el('span', { class: 'badge' }, el('span', { class: 'spinner tiny' }), `${Math.round(m.progress * 100)} %`);
      else state = el('button', { type: 'button', class: 'btn small', disabled: !local.available, onclick: () => download(m) }, icon('download'), `Laden (${formatSize(m.size_mb)})`);
      return el('div', { class: 'model-row' },
        el('div', { class: 'model-info' },
          el('strong', {}, m.label),
          el('div', { class: 'hint' }, m.description),
          m.error ? el('div', { class: 'error-text' }, m.error) : null),
        state);
    });
    return el('div', { class: 'provider-card' },
      el('div', { class: 'head' },
        el('span', { class: `dot${local.available ? ' ok' : ''}` }),
        el('strong', {}, 'KI-Schärfen (Real-ESRGAN)')),
      el('p', { class: 'hint' }, local.available
        ? `Rechnet auf: ${local.device}. Die Modelle werden beim ersten Schärfen automatisch geladen – oder hier schon vorab.`
        : 'Das Paket „onnxruntime“ fehlt. start.bat bzw. start.sh erneut ausführen; klappt die Installation nicht, Python 3.12 bis 3.14 verwenden (für ganz neue Python-Versionen gibt es ONNX Runtime oft noch nicht).'),
      rows,
      el('div', { class: 'hint small path' }, `Modellordner: ${local.directory}`));
  };

  // -- ComfyUI --------------------------------------------------------------------------------
  const comfyAction = async (call, message) => {
    try {
      const result = await call();
      comfy = result;
      renderLocal();
      refreshProviders();
      if (message) toast(typeof message === 'function' ? message(result) : message, 'success', 2500);
    } catch (err) {
      showError(err);
    }
  };

  const comfyCard = () => {
    if (!comfy) return null;
    const urlInput = el('input', {
      type: 'text',
      value: comfy.configured_url || '',
      spellcheck: 'false',
      placeholder: `automatisch (${comfy.url})`,
      'aria-label': 'Adresse von ComfyUI',
      onkeydown: (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          saveUrl();
        }
      },
    });
    const saveUrl = () => comfyAction(() => api.setComfyUrl(urlInput.value.trim() || null),
      (result) => (result.reachable ? 'ComfyUI ist verbunden.' : 'Adresse gespeichert – ComfyUI ist dort aber nicht erreichbar.'));
    const fileInput = el('input', {
      type: 'file',
      accept: '.json,application/json',
      hidden: true,
      onchange: (e) => {
        const file = e.target.files[0];
        e.target.value = '';
        if (file) comfyAction(() => api.addWorkflow(file), (result) => `Workflow „${result.workflow.name}“ importiert.`);
      },
    });
    const workflows = comfy.workflows.map((w) => el('div', { class: 'model-row' },
      el('div', { class: 'model-info' },
        el('strong', {}, w.name),
        el('div', { class: 'hint' }, `${w.kind === 'generate' ? 'Generieren' : 'Bearbeiten'} · ${w.summary.join(' · ')}`)),
      el('button', {
        type: 'button',
        class: 'btn small',
        title: 'Workflow entfernen',
        'aria-label': `Workflow ${w.name} entfernen`,
        onclick: () => comfyAction(() => api.deleteWorkflow(w.id)),
      }, icon('x'))));
    let models = null;
    if (comfy.reachable) {
      models = comfy.checkpoints.length
        ? `Gefundene Checkpoint-Modelle: ${comfy.checkpoints.join(', ')}`
        : 'Keine Checkpoint-Modelle gefunden (Ordner ComfyUI/models/checkpoints). Modelle wie Z-Image oder FLUX nutzt du über einen importierten Workflow.';
    }
    const status = comfy.reachable
      ? comfy.status
      : `Nicht erreichbar unter ${comfy.url}. Starte ComfyUI – Bildwerk sucht es automatisch unter 127.0.0.1:8000, :8001 und :8188. Läuft es woanders, trage hier die Adresse ein.`;
    return el('div', { class: 'provider-card' },
      el('div', { class: 'head' },
        el('span', { class: `dot${comfy.reachable ? ' ok' : ''}` }),
        el('strong', {}, 'ComfyUI – Generieren & Bearbeiten'),
        el('button', { type: 'button', class: 'icon-btn small', title: 'Erneut prüfen', 'aria-label': 'Verbindung erneut prüfen', onclick: () => comfyAction(() => api.comfyui(true)) }, icon('refresh'))),
      el('p', { class: 'hint' }, status),
      el('div', { class: 'key-row' },
        urlInput,
        el('button', { type: 'button', class: 'btn', onclick: saveUrl }, 'Übernehmen')),
      models ? el('p', { class: 'hint' }, models) : null,
      el('div', { class: 'section-head' },
        el('strong', {}, 'Eigene Workflows'),
        el('button', { type: 'button', class: 'btn small', onclick: () => fileInput.click() }, icon('file-plus'), 'Importieren …'),
        fileInput),
      workflows.length ? workflows : el('p', { class: 'hint' }, 'In ComfyUI über „Workflow → Export (API)“ speichern und hier importieren. Bildwerk setzt Beschreibung, Format und Seed selbst ein.'));
  };

  const renderLocal = () => {
    localBox.replaceChildren(...[localCard(), comfyCard()].filter(Boolean));
  };

  // -- Cloud-Anbieter ---------------------------------------------------------------------------
  const renderCloud = () => {
    cloudBox.replaceChildren(...data.providers.filter((p) => p.requires_key).map((p) => providerCard(p)));
    $('#settings-path').textContent = `Gespeichert in: ${data.config_path}`;
  };

  const save = async (provider, key) => {
    try {
      data = await api.saveKey(provider.id, key);
      app.emit('providers', { providers: data.providers });
      renderCloud();
      toast(key ? `Schlüssel für ${provider.name} gespeichert.` : `Schlüssel für ${provider.name} entfernt.`, 'success', 2500);
    } catch (err) {
      showError(err);
    }
  };

  const providerCard = (p) => {
    let status = 'Nicht eingerichtet';
    if (p.key_source === 'settings') status = `Eingerichtet ${p.key_hint || ''}`;
    else if (p.key_source === 'env') status = `Aus Umgebungsvariable ${p.env_vars.join(' / ')}`;
    const input = el('input', {
      type: 'password',
      autocomplete: 'off',
      spellcheck: 'false',
      placeholder: p.configured ? 'Neuen Schlüssel einfügen, um ihn zu ersetzen' : 'API-Schlüssel einfügen',
      'aria-label': `API-Schlüssel für ${p.name}`,
      onkeydown: (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          if (input.value.trim()) save(p, input.value.trim());
        }
      },
    });
    return el('div', { class: 'provider-card' },
      el('div', { class: 'head' },
        el('span', { class: `dot${p.configured ? ' ok' : ''}` }),
        el('strong', {}, p.name),
        el('span', { class: 'hint' }, status)),
      el('div', { class: 'key-row' },
        input,
        el('button', { type: 'button', class: 'btn', onclick: () => input.value.trim() && save(p, input.value.trim()) }, 'Speichern'),
        p.key_source === 'settings'
          ? el('button', { type: 'button', class: 'btn', title: 'Gespeicherten Schlüssel löschen', onclick: () => save(p, null) }, icon('x'))
          : null),
      el('div', { class: 'hint' },
        `Kann: ${p.capabilities.map((c) => CAPABILITIES[c]).join(', ')}. `,
        el('a', { href: p.key_url, target: '_blank', rel: 'noopener noreferrer' }, 'Schlüssel erstellen ↗')));
  };

  renderLocal();
  renderCloud();
  pollDownloads();
  await show(dialog);
  open = false;
  clearTimeout(pollTimer);
}

function formatSize(mb) {
  return `${mb.toLocaleString('de-DE', { maximumFractionDigits: 1 })} MB`;
}
