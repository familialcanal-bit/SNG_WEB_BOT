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


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    time.sleep(1.2)
    webbrowser.open(URL)
    print(f"Interface ouverte: {URL}")

    while True:
        time.sleep(3600)
