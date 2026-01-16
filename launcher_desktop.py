import sys
import time
import subprocess
from pathlib import Path
import webview

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

def main():
    project_dir = Path(__file__).resolve().parent
    python = sys.executable

    # Lance le serveur FastAPI
    subprocess.Popen(
        [python, "-m", "uvicorn", "server:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(project_dir)
    )

    # Attendre démarrage serveur
    time.sleep(3)

    # Ouvre fenêtre desktop
    webview.create_window(
        "SNG WEB BOT",
        URL,
        width=1100,
        height=700
    )
    webview.start()

if __name__ == "__main__":
    main()
