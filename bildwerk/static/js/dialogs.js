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
  const box = $('#settings-providers');
  let data;
  try {
    data = await api.settings();
  } catch (err) {
    showError(err);
    return;
  }

  const render = () => {
    box.replaceChildren(...data.providers.map((p) => providerCard(p)));
    $('#settings-path').textContent = `Gespeichert in: ${data.config_path}`;
  };

  const save = async (provider, key) => {
    try {
      data = await api.saveKey(provider.id, key);
      app.emit('providers', { providers: data.providers });
      render();
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

  render();
  await show(dialog);
}
