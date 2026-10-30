"""``python -m src.web [--config CONFIG] [--port PORT]``: serve the web interface on
``web.host:web.port`` and open it in the browser (``web.open_browser``).

The server must start only under ``__main__``: reconstructions run in spawned child
processes, which import this module as ``__mp_main__``.
"""

from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.config import WebConfig, config_from_args
from src.web.app import create_app


def build(argv: list[str] | None = None) -> tuple[FastAPI, WebConfig]:
    """The app and its ``web`` settings, from the command line."""
    parser = argparse.ArgumentParser(prog="python -m src.web", description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--port", dest="web.port", type=int, help="web.port")
    config = config_from_args(parser.parse_args(argv))
    return create_app(config), config.web


def main(argv: list[str] | None = None) -> None:
    app, web = build(argv)
    server = uvicorn.Server(uvicorn.Config(app, host=web.host, port=web.port))
    if web.open_browser:
        url = f"http://{web.host}:{web.port}/"
        threading.Thread(target=_open_when_started, args=(server, url), daemon=True).start()
    server.run()


def _open_when_started(server: uvicorn.Server, url: str) -> None:
    """Open ``url`` once the server is listening (not if it stops first)."""
    while not server.started:
        if server.should_exit:
            return
        time.sleep(0.05)
    webbrowser.open(url)


if __name__ == "__main__":
    main()
