import subprocess
import sys
import time
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    python = sys.executable

    subprocess.Popen(
        [python, "-m", "uvicorn", "server:app", "--host", HOST, "--port", str(PORT)],
        cwd=str(project_dir),
    )

    time.sleep(2.5)

    try:
        import webview  # type: ignore

        webview.create_window("SNG WEB BOT", URL, width=1100, height=700)
        webview.start()
    except Exception:
        webbrowser.open(URL)
        print(f"pywebview indisponible, ouverture navigateur: {URL}")


if __name__ == "__main__":
    main()
