"""Единая точка входа: поднимает FastAPI в фоне и запускает Streamlit.

Используется для деплоя там, где есть только одна точка входа
(например, Streamlit Cloud: Main file path = app/serve.py).

Схема:
    app/serve.py  ── импортирует ──> app/backend_api.py (FastAPI, :8000)
                  └── subprocess    └──> streamlit run app/streamlit_app.py (:8501)

Streamlit-фронтенд обращается к бэкенду через HTTP (переменная BACKEND_URL,
по умолчанию http://localhost:8000), так что в DevTools виден реальный
POST /predict вместо вызова в обход сети.

Запуск локально:
    streamlit run app/serve.py --server.port 8501
или (если хочется явно):
    python -m app.serve
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
import uvicorn

HOST = os.environ.get("BACKEND_HOST", "0.0.0.0")
PORT = int(os.environ.get("BACKEND_PORT", "8000"))
BACKEND_URL = f"http://localhost:{PORT}"


def _port_is_free(host: str, port: int) -> bool:
    """Проверяет, свободен ли порт: если FastAPI уже подняли где-то ещё
    (отдельным uvicorn-процессом), не стартуем второй экземпляр."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.connect((host, port))
        except OSError:
            return True
        return False


def _wait_for_backend(url: str, timeout_s: float = 15.0) -> None:
    """Блокирующее ожидание готовности /health."""
    deadline = time.monotonic() + timeout_s
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=1.0)
            if r.status_code == 200:
                print(f"[serve] backend ready at {url}", flush=True)
                return
        except Exception as exc:
            last_exc = exc
        time.sleep(0.2)
    raise RuntimeError(
        f"Backend at {url} did not become ready within {timeout_s}s "
        f"(last error: {last_exc})"
    )


def _start_backend() -> None:
    """Запустить uvicorn в фоновом потоке того же процесса."""
    from app.backend_api import app as fastapi_app

    config = uvicorn.Config(
        fastapi_app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("BACKEND_LOG_LEVEL", "info"),
        # На Streamlit Cloud нельзя переиспользовать reload/workers
        reload=False,
        workers=1,
    )
    server = uvicorn.Server(config)

    import threading
    t = threading.Thread(target=server.run, name="uvicorn", daemon=True)
    t.start()
    _wait_for_backend(BACKEND_URL)


def _start_streamlit() -> int:
    """Запустить streamlit в основном потоке (subprocess streamlit run)."""
    streamlit_app = Path(__file__).resolve().parent / "streamlit_app.py"
    port = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(streamlit_app),
        "--server.port", port,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print(f"[serve] launching: {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> int:
    if _port_is_free("127.0.0.1", PORT):
        _start_backend()
    else:
        print(f"[serve] port {PORT} already in use, assuming backend is up",
              flush=True)
        _wait_for_backend(BACKEND_URL)

    return _start_streamlit()


if __name__ == "__main__":
    sys.exit(main())