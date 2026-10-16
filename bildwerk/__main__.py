"""Startet Bildwerk: ``python -m bildwerk`` oder einfach ``bildwerk``."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__
from .settings import load_dotenv


def _find_free_port(host: str, start: int, attempts: int = 20) -> int:
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
                return port
            except OSError:
                continue  # belegt oder gesperrt – nächsten Port probieren
    raise SystemExit(f"Kein freier Port zwischen {start} und {start + attempts - 1} gefunden.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="bildwerk", description="Bildwerk – Bildbearbeitung im Browser")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse (Standard: 127.0.0.1, nur dieser Rechner)")
    parser.add_argument("--port", type=int, default=8000, help="Port (Standard: 8000, sonst der nächste freie)")
    parser.add_argument("--no-browser", action="store_true", help="Browser nicht automatisch öffnen")
    parser.add_argument("--version", action="version", version=f"Bildwerk {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    import uvicorn

    from .server import LOCAL_HOSTS, create_app

    port = _find_free_port(args.host, args.port)
    local = args.host in LOCAL_HOSTS
    app = create_app(allowed_hosts=None if local else ["*"])
    shown_host = args.host if local else "127.0.0.1"
    if ":" in shown_host:
        shown_host = f"[{shown_host}]"
    url = f"http://{shown_host}:{port}/"

    print(f"\n  Bildwerk {__version__} läuft unter {url}")
    if not local:
        print("  Achtung: Bildwerk ist auch für andere Geräte im Netzwerk erreichbar.")
    print("  Beenden mit Strg+C.\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    try:
        uvicorn.run(app, host=args.host, port=port, log_level="warning", access_log=False)
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
