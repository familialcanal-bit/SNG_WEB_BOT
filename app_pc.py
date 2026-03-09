import os
import time
import threading

import uvicorn
import webview


HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def run_server() -> None:
    uvicorn.run(
        "server:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="warning",
    )


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(1.2)

    webview.create_window(
        "SNGSLUISGUZMAN - AI",
        URL,
        width=1200,
        height=800,
        min_size=(900, 650),
        resizable=True,
    )
    webview.start()
