// Reiter „Verlauf“: alle Bearbeitungsschritte, per Klick erreichbar.

import { el } from '../util.js';

export class HistoryPanel {
  constructor(app, root) {
    this.app = app;
    this.root = root;
    app.on('active', () => this.render());
    app.on('updated', ({ doc }) => doc === app.docs.active && this.render());
    this.render();
  }

  render() {
    const doc = this.app.docs.active;
    if (!doc) {
      this.root.replaceChildren(el('p', { class: 'empty-note' }, 'Kein Bild geöffnet.'));
      return;
    }
    const { labels, position } = doc.history;
    const items = labels.map((label, i) => {
      const state = i === position ? 'current' : i > position ? 'future' : '';
      return el('li', { class: state },
        el('button', { onclick: () => this.app.commands.historyGoto(doc, i), title: i > position ? 'Wiederherstellen' : 'Zu diesem Schritt zurück' },
          el('span', { class: 'num' }, i === 0 ? '·' : String(i)),
          el('span', {}, label)));
    });
    this.root.replaceChildren(
      el('h3', {}, 'Bearbeitungsschritte'),
      el('ol', { class: 'history-list' }, items),
      el('p', { class: 'hint' }, 'Klicke auf einen Schritt, um dorthin zurückzukehren. Strg+Z macht rückgängig, Strg+Y stellt wieder her.'),
    );
  }
}
