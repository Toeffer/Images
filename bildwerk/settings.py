"""Einstellungen und API-Schlüssel.

Schlüssel werden in einer JSON-Datei im Benutzerverzeichnis gespeichert
(Windows: ``%APPDATA%\\Bildwerk``, macOS: ``~/Library/Application Support/Bildwerk``,
Linux: ``~/.config/bildwerk``). Alternativ können sie über Umgebungsvariablen oder
eine ``.env``-Datei gesetzt werden. In der Oberfläche gespeicherte Schlüssel haben Vorrang.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)


def default_config_dir() -> Path:
    override = os.environ.get("BILDWERK_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "Bildwerk"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Bildwerk"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "bildwerk"


def load_dotenv(path: Path) -> None:
    """Liest ``KEY=VALUE``-Zeilen aus einer .env-Datei (bestehende Variablen bleiben)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


class Settings:
    def __init__(self, config_dir: Path | None = None, environ: Mapping[str, str] | None = None) -> None:
        self.path = (config_dir or default_config_dir()) / "settings.json"
        self._environ = os.environ if environ is None else environ
        self._lock = threading.Lock()
        self._data = self._load()

    @property
    def config_dir(self) -> Path:
        """Ordner für Einstellungen, lokale Modelle und ComfyUI-Workflows."""
        return self.path.parent

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            log.warning("Einstellungen in %s konnten nicht gelesen werden", self.path, exc_info=True)
            return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)  # nur für den eigenen Benutzer lesbar
        except OSError:
            pass
        os.replace(tmp, self.path)

    def api_key(self, provider_id: str, env_vars: tuple[str, ...]) -> tuple[str | None, str | None]:
        """Liefert (Schlüssel, Quelle) mit Quelle ``"settings"``, ``"env"`` oder ``None``."""
        with self._lock:
            stored = self._data.get("api_keys", {}).get(provider_id)
        if stored:
            return stored, "settings"
        for name in env_vars:
            value = self._environ.get(name, "").strip()
            if value:
                return value, "env"
        return None, None

    def set_api_key(self, provider_id: str, key: str | None) -> None:
        with self._lock:
            keys = self._data.setdefault("api_keys", {})
            if key:
                keys[provider_id] = key.strip()
            else:
                keys.pop(provider_id, None)
            self._save()

    def get(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get("preferences", {}).get(name, default)

    def set(self, name: str, value: Any) -> None:
        with self._lock:
            self._data.setdefault("preferences", {})[name] = value
            self._save()
