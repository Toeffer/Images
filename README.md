# Bildwerk

Ein kleines, erweiterbares Bildbearbeitungsprogramm, das lokal auf deinem Rechner läuft und im Browser bedient wird. Du kannst damit **Bildausschnitte auswählen und gezielt nachschärfen**, Fotos korrigieren und – mit einem API-Schlüssel – **Bilder per KI generieren oder bearbeiten lassen**.

![Bildwerk: Ausgewählter Bildbereich mit „Unschärfe entfernen“ und Live-Vorschau](docs/screenshot.webp)

## Funktionen

**Auswählen**
- Rechteck, Ellipse, Lasso (frei) und Pinsel (Bereich übermalen)
- Auswahlen kombinieren: hinzufügen (Umschalt) oder abziehen (Alt), umkehren, alles auswählen
- Weiche Kante, damit Bearbeitungen unsichtbar ins Bild übergehen

**Nachschärfen und verbessern** – wirkt auf die Auswahl oder das ganze Bild
- *Schärfen* (Unscharf maskieren) mit Stärke, Radius und Schwellwert
- *Unschärfe entfernen*: rekonstruiert leicht verwackelte oder unscharfe Bereiche (Richardson-Lucy-Dekonvolution mit Rauschschutz)
- *Klarheit* (lokaler Kontrast), *Rauschen reduzieren* (kantenerhaltend), *Weichzeichnen*, *Verpixeln* (z. B. für Gesichter oder Kennzeichen)
- *Detail-Zoom*: Ausschnitt vergrößern und nachschärfen, öffnet sich als neues Bild

**Korrigieren und Transformieren**
- Helligkeit & Kontrast, Tonwerte, Farbe (Sättigung, Farbtemperatur, Tönung), Auto-Kontrast, Schwarzweiß, Sepia, Invertieren
- Zuschneiden, Größe ändern, Drehen (90° und frei/Begradigen), Spiegeln
- Füllen und Löschen (transparent)

**KI** (optional, mit API-Schlüssel eines Anbieters)
- *Bild generieren*: aus einer Textbeschreibung, mit Format, Qualität und Anzahl
- *Auswahl mit KI bearbeiten*: z. B. „Ersetze den Himmel durch einen Sternenhimmel“ oder „Entferne die Person“ – nur der ausgewählte Bereich ändert sich
- *KI-Schärfen & Vergrößern*: stellt unscharfe Bildausschnitte wieder her, im Bild oder als vergrößertes neues Bild
- Ergebnisse erscheinen erst als Vorschau mit *Vorher*, *Nochmal*, *Verwerfen* und *Übernehmen*

**Komfort**
- Live-Vorschau, Rückgängig/Wiederholen und ein Verlauf aller Schritte
- Mehrere Bilder in Tabs, Zoomen mit dem Mausrad, Verschieben mit gedrückter Leertaste
- Öffnen per Dialog, Drag & Drop oder Einfügen aus der Zwischenablage (Strg+V)
- Speichern als PNG, JPEG, WebP, AVIF oder TIFF – EXIF-Daten und Farbprofil bleiben erhalten, Handyfotos werden automatisch richtig gedreht

## Installation und Start

Du brauchst **Python 3.10 oder neuer** ([python.org/downloads](https://www.python.org/downloads/)). Unter Windows bei der Installation „Add python.exe to PATH“ anhaken.

**Am einfachsten:**

- **Windows:** Doppelklick auf `start.bat`
- **macOS/Linux:** im Terminal `./start.sh`

Beim ersten Start wird automatisch eine eigene Python-Umgebung (`.venv`) eingerichtet. Danach öffnet sich Bildwerk im Browser (Standard: <http://127.0.0.1:8000>). Beenden mit **Strg+C** im Terminalfenster.

**Oder von Hand:**

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -e .
bildwerk            # oder: python -m bildwerk
```

Nützliche Optionen: `bildwerk --port 9000`, `bildwerk --no-browser`. Mit `--host 0.0.0.0` ist Bildwerk auch von anderen Geräten im Heimnetz erreichbar (nur in vertrauenswürdigen Netzen verwenden).

**iPhone-Fotos (HEIC)** lassen sich nach `pip install -e ".[heic]"` öffnen.

> Die geöffneten Bilder liegen im Arbeitsspeicher des Bildwerk-Programms. Ein Neuladen der Seite ist kein Problem, beim Beenden von Bildwerk gehen ungespeicherte Änderungen aber verloren – also vorher speichern.

## KI einrichten

Für die KI-Funktionen brauchst du einen API-Schlüssel bei **mindestens einem** Anbieter. Die Nutzung wird vom Anbieter pro Bild abgerechnet.

| Anbieter | Kann | Schlüssel erstellen |
|---|---|---|
| OpenAI (GPT Image) | Generieren, Bereiche bearbeiten, KI-Schärfen | <https://platform.openai.com/api-keys> |
| Google Gemini („Nano Banana“) | Generieren, Bereiche bearbeiten, KI-Schärfen | <https://aistudio.google.com/apikey> |
| Stability AI | Generieren, Bereiche bearbeiten, KI-Hochskalieren (4×) | <https://platform.stability.ai/account/keys> |

Den Schlüssel trägst du in Bildwerk unter **⚙ Einstellungen** ein. Er wird nur lokal gespeichert (Windows: `%APPDATA%\Bildwerk`, macOS: `~/Library/Application Support/Bildwerk`, Linux: `~/.config/bildwerk`). Alternativ kannst du `.env.example` nach `.env` kopieren und die Schlüssel dort eintragen, oder die Umgebungsvariablen `OPENAI_API_KEY`, `GEMINI_API_KEY` bzw. `STABILITY_API_KEY` setzen.

Im Reiter **KI** wählst du Anbieter und Modell. Die Modellnamen ändern sich bei den Anbietern regelmäßig – die Liste steht jeweils oben in `bildwerk/ai/<anbieter>_provider.py` und lässt sich dort leicht anpassen. Manche OpenAI-Modelle setzen eine verifizierte Organisation im OpenAI-Konto voraus.

## So geht's

**Bildausschnitt nachschärfen**
1. Bild öffnen (oder in das Fenster ziehen).
2. Mit Rechteck (M), Lasso (L) oder Pinsel (B) den Bereich auswählen; bei Bedarf „Weiche Kante“ erhöhen.
3. Rechts unter *Bearbeiten* → *Schärfen* oder *Unschärfe entfernen* wählen. Die Live-Vorschau zeigt das Ergebnis sofort; zum Vergleich ins Bild zoomen (Mausrad, Taste 1 für 100 %).
4. *Anwenden* – und mit Strg+S speichern.

**Bild generieren:** Reiter *KI* → Beschreibung eingeben → *Generieren*. Das Ergebnis öffnet sich als neues Bild.

**Bereich per KI ändern:** Bereich auswählen → Reiter *KI* → unter *Auswahl mit KI bearbeiten* beschreiben, was passieren soll → *Auswahl bearbeiten* → Ergebnis prüfen und *Übernehmen*.

**KI-Schärfen:** Einen überschaubaren Bereich wählen (z. B. ein Gesicht oder ein Schild) → Reiter *KI* → *KI-Schärfen*. Mit „Als neues, vergrößertes Bild öffnen“ entsteht ein hochaufgelöster Detail-Zoom.

### Tastenkürzel

| Taste | Funktion | Taste | Funktion |
|---|---|---|---|
| Strg+O | Öffnen | Strg+S | Speichern |
| Strg+Z | Rückgängig | Strg+Y / Strg+Umschalt+Z | Wiederholen |
| Strg+A | Alles auswählen | Strg+D / Esc | Auswahl aufheben |
| Strg+Umschalt+I | Auswahl umkehren | Strg+C | Auswahl kopieren |
| V oder Leertaste | Verschieben (Hand) | M / E / L / B | Rechteck / Ellipse / Lasso / Pinsel |
| [ und ] | Pinsel kleiner / größer | + / − / 0 / 1 | Zoom: rein / raus / einpassen / 100 % |
| S | Schärfen | C | Zuschneiden |
| N | Neues Bild | Enter | Geöffnete Funktion anwenden |

Beim Auswählen: **Umschalt** gedrückt halten fügt hinzu, **Alt** zieht ab.

## Selbst erweitern

Bildwerk ist so gebaut, dass neue Funktionen wenig Code brauchen.

**Neue Bildoperation** – die Regler in der Oberfläche entstehen automatisch aus den Parametern. Zum Beispiel in `bildwerk/ops/filters.py`:

```python
from PIL import ImageOps

@operation(
    "posterize", "Tontrennung", "filter",
    params=[Param("bits", "Bits pro Kanal", "int", 3, min=1, max=8)],
    description="Reduziert die Anzahl der Farbstufen.",
)
def posterize(img, ctx, *, bits):
    return ImageOps.posterize(img, bits)
```

Nach einem Neustart erscheint „Tontrennung“ in der Liste – inklusive Live-Vorschau, Auswahl-Unterstützung und Rückgängig.

**Neuer KI-Anbieter** – eine Unterklasse von `Provider` (siehe `bildwerk/ai/base.py`) mit den Methoden `generate`, `edit` und/oder `upscale`, eingetragen in `PROVIDER_CLASSES` in `bildwerk/ai/__init__.py`. Ausschnitt, Maske und das nahtlose Einmischen des Ergebnisses übernimmt `bildwerk/ai/pipeline.py`.

## Projektaufbau

```
bildwerk/
  __main__.py        Start (Befehl „bildwerk“), öffnet den Browser
  server.py          HTTP-Schnittstelle (FastAPI)
  document.py        geöffnete Bilder, Verlauf, Vorschauen
  engine.py          führt Operationen aus – auf Wunsch nur in der Auswahl
  selection.py       Auswahlformen → Masken mit weicher Kante
  imageio.py         Laden, Farbprofile, EXIF, Export
  settings.py        Einstellungen und API-Schlüssel
  ops/               alle Bildoperationen (Anpassungen, Filter, Transformationen)
  ai/                KI-Anbieter (OpenAI, Gemini, Stability) und Pipeline
  static/            Oberfläche (HTML, CSS, JavaScript ohne Build-Schritt)
tests/               automatische Tests (pytest, Browser-Tests mit Playwright)
```

## Entwicklung und Tests

```bash
pip install -e ".[dev]"
pytest -m "not e2e"                  # schnelle Tests
python -m playwright install chromium
pytest -m e2e                        # Browser-Tests der Oberfläche
```

Die KI-Anbieter werden in den Tests mit simulierten Antworten geprüft – es entstehen dabei keine Kosten. Bei jedem Push laufen die Tests per GitHub Actions unter Windows, macOS und Linux.

## Datenschutz und Sicherheit

- Bildwerk läuft nur auf deinem Rechner und ist standardmäßig nur dort erreichbar (`127.0.0.1`).
- Bilder verlassen den Rechner nur, wenn du eine KI-Funktion nutzt – dann wird der betroffene Ausschnitt an den gewählten Anbieter gesendet.
- API-Schlüssel werden nie an die Oberfläche zurückgegeben (nur die letzten vier Zeichen zur Kontrolle). Anfragen von fremden Webseiten an Bildwerk werden blockiert.
