import os
import threading
import time
import webbrowser

import uvicorn

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def run_server() -> None:
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="warning")


def open_ui() -> None:
    try:
        import webview  # type: ignore

        webview.create_window(
            "SNGSLUISGUZMAN - AI",
            URL,
            width=1200,
            height=800,
            min_size=(900, 650),
            resizable=True,
        )
        webview.start()
    except Exception:
        webbrowser.open(URL)
        print(f"pywebview indisponible, ouverture navigateur: {URL}")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(1.2)

    open_ui()
