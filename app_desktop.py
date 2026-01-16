import os
import time
import threading
import webview
import uvicorn

def run_server():
    # Lance ton FastAPI app = server:app
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="warning",
    )

if __name__ == "__main__":
    # (Optionnel) s'assure d'être dans le dossier du projet
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    # Petite pause pour laisser le serveur démarrer
    time.sleep(1.0)

    webview.create_window(
        "SNGSLUISGUZMAN",
        "http://127.0.0.1:8000",
        width=1200,
        height=800,
        resizable=True,
    )
    webview.start()
