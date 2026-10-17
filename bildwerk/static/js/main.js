// Einstiegspunkt: verbindet Anzeige, Werkzeuge, Seitenleiste und Befehle.

import { api } from './api.js';
import { confirmDialog, openExportDialog, openNewDialog, openSettingsDialog } from './dialogs.js';
import { DocumentManager } from './documents.js';
import { AdjustPanel } from './panels/adjust.js';
import { AiPanel } from './panels/ai.js';
import { HistoryPanel } from './panels/history.js';
import { fullRect, invertShapes, Selection } from './selection.js';
import { Tools } from './tools.js';
import { $, $$, el, hydrateIcons, icon, isTyping, showError, storage, toast } from './util.js';
import { Viewer } from './viewer.js';

const events = new EventTarget();
const app = {
  info: null,
  on(type, handler) {
    events.addEventListener(type, (e) => handler(e.detail || {}));
  },
  emit(type, detail = {}) {
    events.dispatchEvent(new CustomEvent(type, { detail }));
  },
};
window.bildwerk = app; // zum Ausprobieren in der Browser-Konsole

// -- Statuszeile -------------------------------------------------------------------------
function createStatus() {
  let busyCount = 0;
  return {
    cursor(p) {
      const doc = app.docs.active;
      const inside = p && doc && p.x >= 0 && p.y >= 0 && p.x < doc.width && p.y < doc.height;
      $('#status-cursor').textContent = inside ? `x ${Math.floor(p.x)}   y ${Math.floor(p.y)}` : '';
    },
    busy(text) {
      busyCount++;
      const timer = setTimeout(() => {
        $('#status-busy-text').textContent = text;
        $('#status-busy').hidden = false;
      }, 250);
      return () => {
        clearTimeout(timer);
        busyCount = Math.max(0, busyCount - 1);
        if (!busyCount) $('#status-busy').hidden = true;
      };
    },
  };
}

function refreshStatus() {
  const doc = app.docs.active;
  $('#status-size').textContent = doc ? `${doc.width} × ${doc.height} px` : '';
  const b = doc?.selection.bounds(doc.width, doc.height);
  $('#status-selection').textContent = b ? `Auswahl: ${b[2] - b[0]} × ${b[3] - b[1]} px` : '';
  const zoom = app.viewer.scale * 100;
  const zoomText = doc ? `${zoom.toLocaleString('de-DE', { maximumFractionDigits: zoom < 10 ? 1 : 0 })} %` : '';
  $('#status-zoom').textContent = zoomText;
  $('#zoom-label').textContent = zoomText || '100 %';
}

function refreshChrome() {
  const doc = app.docs.active;
  for (const button of $$('[data-needs-doc]')) button.disabled = !doc;
  if (doc) {
    $('[data-command="undo"]').disabled = !doc.canUndo;
    $('[data-command="redo"]').disabled = !doc.canRedo;
  }
  $('#welcome').hidden = Boolean(doc);
  document.title = doc ? `${doc.name} – Bildwerk` : 'Bildwerk';
  const feather = doc ? doc.selection.feather : storage.get('feather', 0);
  $('#feather').value = feather;
  $('#feather-value').textContent = `${feather} px`;
  refreshStatus();
  refreshResultBar();
  refreshTask();
}

// -- Auswahl & Aufgaben ---------------------------------------------------------------------
app.setSelection = (doc, shapes, feather = doc.selection.feather) => {
  doc.selection = new Selection(shapes, feather);
  if (doc === app.docs.active) app.viewer.invalidateTint();
  app.emit('selection', { doc });
};

/** Sperrt ein Dokument während einer langen (KI-)Aufgabe und zeigt den Fortschritt. */
app.setTask = (doc, task) => {
  doc.task = task;
  doc.blocked = Boolean(task);
  if (doc === app.docs.active) refreshTask();
};

let taskTimer = null;
function refreshTask() {
  const task = app.docs.active?.task;
  $('#busy-overlay').hidden = !task;
  clearInterval(taskTimer);
  if (!task) return;
  $('#busy-text').textContent = task.text;
  const cancel = $('#busy-cancel');
  cancel.hidden = !task.cancel;
  cancel.onclick = () => task.cancel?.();
  const tick = () => {
    $('#busy-time').textContent = `${Math.round((Date.now() - task.start) / 1000)} s`;
  };
  tick();
  taskTimer = setInterval(tick, 500);
}

/** Führt eine verändernde Anfrage aus und übernimmt das Ergebnis in die Anzeige. */
app.mutate = (doc, call, text = 'Bitte warten …') =>
  app.docs.run(doc, async () => {
    const stop = app.status.busy(text);
    try {
      const result = await call();
      await app.docs.applyResult(doc, result);
      return result;
    } finally {
      stop();
    }
  });

function aiResultPending(doc) {
  if (doc?.preview?.source !== 'ai') return false;
  toast('Bitte zuerst das KI-Ergebnis übernehmen oder verwerfen.');
  return true;
}

// -- Befehle -----------------------------------------------------------------------------------
function withDoc(fn) {
  const doc = app.docs.active;
  if (doc) return fn(doc);
  return undefined;
}

function historyStep(doc, request) {
  if (doc.blocked || aiResultPending(doc)) return;
  app.mutate(doc, request, 'Verlauf …').catch(showError);
}

async function openFiles(files) {
  const images = [...files].filter((f) => f.type.startsWith('image/') || /\.(heic|heif|avif|tiff?|webp)$/i.test(f.name));
  if (!images.length) {
    if (files.length) toast('Das scheint keine Bilddatei zu sein.', 'error');
    return;
  }
  for (const file of images) {
    const stop = app.status.busy(`Öffne ${file.name} …`);
    try {
      await app.docs.open(await api.upload(file));
    } catch (err) {
      showError(err);
    } finally {
      stop();
    }
  }
}

async function copySelection(doc) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    toast('Kopieren in die Zwischenablage wird von diesem Browser nicht unterstützt.');
    return;
  }
  const b = doc.selection.bounds(doc.width, doc.height) || [0, 0, doc.width, doc.height];
  const canvas = el('canvas', { width: b[2] - b[0], height: b[3] - b[1] });
  canvas.getContext('2d').drawImage(doc.canvas, b[0], b[1], canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
  try {
    const blob = new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]);
    toast('In die Zwischenablage kopiert.', 'success', 2000);
  } catch (err) {
    showError(err);
  }
}

function showPanel(id) {
  for (const button of $$('.sidebar-tabs [data-panel]')) button.classList.toggle('active', button.dataset.panel === id);
  for (const panel of $$('.sidebar .panel')) panel.hidden = panel.id !== `panel-${id}`;
  storage.set('panel', id);
  app.emit('panel', { id });
  const appEl = $('#app');
  if (appEl.classList.contains('sidebar-hidden')) appEl.classList.remove('sidebar-hidden');
  if (matchMedia('(max-width: 900px)').matches) appEl.classList.add('sidebar-open');
}

function toggleSidebar() {
  const appEl = $('#app');
  if (matchMedia('(max-width: 900px)').matches) appEl.classList.toggle('sidebar-open');
  else appEl.classList.toggle('sidebar-hidden');
  setTimeout(() => app.viewer.resize(), 0);
}

const commands = {
  new: () => openNewDialog(app),
  open: () => $('#file-input').click(),
  export: () => withDoc((doc) => !aiResultPending(doc) && openExportDialog(app, doc)),
  undo: () => withDoc((doc) => historyStep(doc, () => api.undo(doc.id))),
  redo: () => withDoc((doc) => historyStep(doc, () => api.redo(doc.id))),
  historyGoto: (doc, position) => historyStep(doc, () => api.history(doc.id, position)),
  'zoom-in': () => app.viewer.zoomStep(1),
  'zoom-out': () => app.viewer.zoomStep(-1),
  'zoom-fit': () => app.viewer.fit(),
  'zoom-100': () => app.viewer.zoomTo(1),
  'select-all': () => withDoc((doc) => app.setSelection(doc, [fullRect(doc.width, doc.height)])),
  deselect: () => withDoc((doc) => app.setSelection(doc, [])),
  'invert-selection': () => withDoc((doc) => app.setSelection(doc, invertShapes(doc.selection.shapes, doc.width, doc.height))),
  copy: () => withDoc(copySelection),
  settings: () => openSettingsDialog(app),
  'toggle-sidebar': toggleSidebar,
  'ai-generate': () => {
    showPanel('ai');
    app.panels.ai.focusPrompt();
  },
};
app.commands = commands;

// -- Tabs & Ergebnisleiste ----------------------------------------------------------------------
async function closeDoc(doc) {
  if (doc.blocked) {
    toast('Bitte warten, bis die KI-Bearbeitung fertig ist.');
    return;
  }
  if (doc.info.dirty && !(await confirmDialog('Bild schließen?', `„${doc.name}“ enthält ungespeicherte Änderungen. Trotzdem schließen?`, 'Schließen'))) return;
  await app.docs.close(doc);
}

function renderTabs() {
  const active = app.docs.active;
  $('#tabs').replaceChildren(...app.docs.docs.map((doc) =>
    el('div', {
      class: `tab${doc === active ? ' active' : ''}`,
      role: 'tab',
      title: `${doc.name} – ${doc.width} × ${doc.height} px`,
      onclick: () => app.docs.activate(doc.id),
      onauxclick: (e) => e.button === 1 && closeDoc(doc),
    },
    doc.info.dirty ? el('span', { class: 'dirty', title: 'Ungespeicherte Änderungen' }) : null,
    el('span', { class: 'tab-name' }, doc.name),
    el('button', { class: 'tab-close', title: 'Schließen', onclick: (e) => { e.stopPropagation(); closeDoc(doc); } }, icon('x')))));
}

function refreshResultBar() {
  const doc = app.docs.active;
  const show = doc?.preview?.source === 'ai';
  $('#result-bar').hidden = !show;
  if (show) $('#result-label').textContent = doc.preview.label || 'KI-Ergebnis';
}

function setupResultBar() {
  $('#result-accept').addEventListener('click', () => withDoc((doc) => {
    if (doc.preview?.source !== 'ai') return;
    const previewId = doc.preview.id;
    app.mutate(doc, () => api.commit(doc.id, previewId), 'Übernehme KI-Ergebnis …').catch(showError);
  }));
  $('#result-discard').addEventListener('click', () => withDoc((doc) => {
    doc.preview = null;
    api.discardPreview(doc.id).catch(() => {});
    app.emit('preview', { doc });
  }));
  $('#result-retry').addEventListener('click', () => app.panels.ai.retry());
  const compare = $('#result-compare');
  const setCompare = (on) => {
    app.viewer.comparing = on;
    app.viewer.requestRender();
  };
  compare.addEventListener('pointerdown', () => setCompare(true));
  for (const type of ['pointerup', 'pointerleave', 'pointercancel']) compare.addEventListener(type, () => setCompare(false));
}

// -- Eingaben ---------------------------------------------------------------------------------
function setupKeyboard() {
  window.addEventListener('keydown', (e) => {
    if (document.querySelector('dialog[open]')) return;
    const mod = e.ctrlKey || e.metaKey;
    const key = e.key.toLowerCase();
    if (isTyping(e)) {
      if (mod && key === 's') {
        e.preventDefault();
        commands.export();
      }
      return;
    }
    if (e.code === 'Space') {
      e.preventDefault();
      if (!e.repeat) app.tools.setSpace(true);
      if (document.activeElement instanceof HTMLButtonElement) document.activeElement.blur();
      return;
    }
    if (e.key === 'Escape') {
      if (app.tools.drag) return;
      if (app.panels.adjust.handleEscape()) return;
      commands.deselect();
      return;
    }
    const shortcuts = mod
      ? { z: e.shiftKey ? 'redo' : 'undo', y: 'redo', o: 'open', s: 'export', a: 'select-all', d: 'deselect', c: 'copy', 0: 'zoom-fit', 1: 'zoom-100', '+': 'zoom-in', '=': 'zoom-in', '-': 'zoom-out', i: e.shiftKey ? 'invert-selection' : null }
      : { '+': 'zoom-in', '=': 'zoom-in', '-': 'zoom-out', 0: 'zoom-fit', 1: 'zoom-100', n: 'new' };
    const command = shortcuts[key];
    if (command) {
      e.preventDefault();
      commands[command]();
      return;
    }
    if (mod || e.altKey) return;
    const tools = { v: 'move', h: 'move', m: 'rect', e: 'ellipse', l: 'lasso', b: 'brush' };
    if (tools[key]) {
      app.tools.setTool(tools[key]);
      return;
    }
    if (key === '[' || key === ']') {
      app.tools.setBrushSize(app.tools.brushSize * (key === '[' ? 0.85 : 1.18));
      return;
    }
    if (e.key === 'Enter' && app.panels.adjust.isEditing()) {
      app.panels.adjust.apply();
      return;
    }
    const op = app.info.categories.flatMap((c) => c.operations).find((o) => o.shortcut?.toLowerCase() === key);
    if (op && app.docs.active) {
      showPanel('adjust');
      app.panels.adjust.open(op.id);
    }
  });
  window.addEventListener('keyup', (e) => {
    if (e.code === 'Space') app.tools.setSpace(false);
  });
  window.addEventListener('blur', () => app.tools.setSpace(false));
}

function setupDropAndPaste() {
  const hint = $('#drop-hint');
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    depth++;
    hint.hidden = false;
  });
  window.addEventListener('dragleave', (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) hint.hidden = true;
  });
  window.addEventListener('dragover', (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener('drop', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    hint.hidden = true;
    openFiles(e.dataTransfer.files);
  });
  window.addEventListener('paste', (e) => {
    if (isTyping(e)) return;
    const files = [...(e.clipboardData?.items || [])]
      .filter((item) => item.kind === 'file' && item.type.startsWith('image/'))
      .map((item) => item.getAsFile())
      .filter(Boolean)
      .map((file) => new File([file], file.name && file.name !== 'image.png' ? file.name : 'Eingefügt.png', { type: file.type }));
    if (files.length) {
      e.preventDefault();
      openFiles(files);
    }
  });
  const input = $('#file-input');
  input.addEventListener('change', () => {
    openFiles(input.files);
    input.value = '';
  });
}

function setupChrome() {
  for (const button of $$('[data-command]')) {
    button.addEventListener('click', () => commands[button.dataset.command]?.());
  }
  for (const button of $$('.sidebar-tabs [data-panel]')) {
    button.addEventListener('click', () => showPanel(button.dataset.panel));
  }
  showPanel(storage.get('panel', 'adjust'));
  if (matchMedia('(max-width: 900px)').matches) $('#app').classList.remove('sidebar-open');

  const feather = $('#feather');
  feather.addEventListener('input', () => {
    const value = Number(feather.value);
    $('#feather-value').textContent = `${value} px`;
    storage.set('feather', value);
    withDoc((doc) => app.setSelection(doc, doc.selection.shapes, value));
  });
}

// -- Start -------------------------------------------------------------------------------------
async function main() {
  hydrateIcons();
  try {
    app.info = await api.info();
  } catch (err) {
    $('#welcome .welcome-card p').textContent = 'Die Verbindung zum Bildwerk-Server ist fehlgeschlagen. Bitte das Programm neu starten.';
    showError(err);
    return;
  }

  app.viewer = new Viewer($('#viewport'), $('#canvas-main'), $('#canvas-overlay'));
  app.docs = new DocumentManager((type, doc, extra = {}) => app.emit(type, { doc, ...extra }));
  app.status = createStatus();
  app.tools = new Tools(app);
  app.panels = {
    adjust: new AdjustPanel(app, $('#panel-adjust')),
    ai: new AiPanel(app, $('#panel-ai')),
    history: new HistoryPanel(app, $('#panel-history')),
  };

  app.on('active', ({ doc }) => {
    if (doc && !doc.selection.shapes.length) doc.selection.feather = storage.get('feather', 0);
    app.viewer.setDocument(doc);
    renderTabs();
    refreshChrome();
  });
  app.on('updated', ({ doc, sizeChanged }) => {
    if (doc === app.docs.active) {
      if (sizeChanged) app.viewer.setDocument(doc);
      else app.viewer.requestRender();
      refreshChrome();
    }
    renderTabs();
  });
  app.on('list', renderTabs);
  app.on('selection', ({ doc }) => doc === app.docs.active && refreshStatus());
  app.on('preview', ({ doc }) => {
    if (doc !== app.docs.active) return;
    app.viewer.requestRender();
    refreshResultBar();
  });
  app.viewer.addEventListener('view', refreshStatus);

  setupChrome();
  setupKeyboard();
  setupDropAndPaste();
  setupResultBar();
  refreshChrome();

  try {
    await app.docs.restore();
  } catch (err) {
    showError(err);
  }
  refreshChrome();
}

main();
