import os
import time
import threading
import webview
import uvicorn

def run_server():
    # Lance ton FastAPI depuis server.py -> app
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        log_level="warning"
    )

import webview

if __name__ == "__main__":
    webview.create_window(
        "SNGSLUISGUZMAN - AI",
        "http://127.0.0.1:8000",
        width=1200,
        height=800
    )
    webview.start()

    # Petite attente pour laisser le serveur démarrer
    time.sleep(1.2)

    # Ouvre la fenêtre PC (WebView)
    webview.create_window(
        "SNGSLUISGUZMAN - AI",
        "http://127.0.0.1:8000",
        width=1200,
        height=800,
        min_size=(900, 650),
    )
    webview.start()
import webview


