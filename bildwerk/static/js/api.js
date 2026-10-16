// Zugriff auf die Bildwerk-Schnittstelle des lokalen Servers.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function check(response) {
  if (response.ok) return response;
  let message = `Fehler ${response.status}`;
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') message = data.detail;
    else if (Array.isArray(data.detail)) message = data.detail.map((d) => d.msg).join('; ');
  } catch {
    /* keine JSON-Antwort */
  }
  throw new ApiError(message, response.status);
}

async function json(method, url, body, signal) {
  const init = { method, signal, headers: {} };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  const response = await check(await fetch(url, init));
  return response.json();
}

/** Lädt Rohpixel (RGBA) als ImageData. */
async function pixels(url, signal) {
  const response = await check(await fetch(url, { signal, cache: 'no-store' }));
  const width = Number(response.headers.get('X-Width'));
  const height = Number(response.headers.get('X-Height'));
  const buffer = await response.arrayBuffer();
  return new ImageData(new Uint8ClampedArray(buffer), width, height);
}

const doc = (id, path = '') => `api/documents/${encodeURIComponent(id)}${path}`;

export const api = {
  info: () => json('GET', 'api/info'),
  documents: () => json('GET', 'api/documents'),
  async upload(file, name) {
    const form = new FormData();
    form.append('file', file, name || file.name || 'Bild');
    const response = await check(await fetch('api/documents', { method: 'POST', body: form }));
    return response.json();
  },
  newDocument: (options) => json('POST', 'api/documents/new', options),
  rename: (id, name) => json('PATCH', doc(id), { name }),
  close: (id) => json('DELETE', doc(id)),
  document: (id) => json('GET', doc(id)),
  pixels(id, box, signal) {
    const query = box ? `?x=${box[0]}&y=${box[1]}&w=${box[2] - box[0]}&h=${box[3] - box[1]}` : '';
    return pixels(doc(id, `/pixels${query}`), signal);
  },
  preview: (id, request, signal) => json('POST', doc(id, '/preview'), request, signal),
  previewPixels: (id, previewId, signal) => pixels(doc(id, `/preview/${previewId}/pixels`), signal),
  discardPreview: (id) => json('DELETE', doc(id, '/preview')),
  commit: (id, previewId) => json('POST', doc(id, '/commit'), { preview_id: previewId }),
  apply: (id, request) => json('POST', doc(id, '/apply'), request),
  undo: (id) => json('POST', doc(id, '/undo')),
  redo: (id) => json('POST', doc(id, '/redo')),
  history: (id, position) => json('POST', doc(id, `/history/${position}`)),
  async export(id, format, quality) {
    const response = await check(await fetch(doc(id, `/export?format=${format}&quality=${quality}`)));
    return response.blob();
  },
  generate: (request, signal) => json('POST', 'api/ai/generate', request, signal),
  aiEdit: (id, request, signal) => json('POST', doc(id, '/ai/edit'), request, signal),
  aiEnhance: (id, request, signal) => json('POST', doc(id, '/ai/enhance'), request, signal),
  settings: () => json('GET', 'api/settings'),
  saveKey: (provider, apiKey) => json('PUT', `api/settings/keys/${encodeURIComponent(provider)}`, { api_key: apiKey }),
};
