import webview
import threading
import subprocess
import time


def run_nextjs():
    subprocess.run(["yarn", "dev"], cwd="../steg-frontend/")


if __name__ == "__main__":
    threading.Thread(target=run_nextjs, daemon=True).start()

    time.sleep(3)

    window = webview.create_window(
        "Audio Steganography",
        "https://localhost:3000",
        width=1200,
        height=800,
        min_size=(800, 600),
    )
    webview.start()
