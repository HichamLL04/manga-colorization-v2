"""Exercise the real local HTTP editor against the supplied checkpoint."""
import base64
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from PIL import Image
import pytest


def test_editor_http_generates_color_and_preserves_locked_pixel(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "v2generator.zip"
    if not checkpoint.exists():
        pytest.skip("Supply the original v2 checkpoint for HTTP integration")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mangacolor", "--threads", "2", "edit",
             "--checkpoint", str(checkpoint), "--port", str(port)],
            cwd=root, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while True:
                try:
                    with urllib.request.urlopen(endpoint + "/capabilities", timeout=1) as response:
                        capabilities = json.load(response)
                    break
                except (urllib.error.URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() >= deadline:
                        pytest.fail("Local editor failed to start")
                    time.sleep(.1)
            assert capabilities["palette_trained"] is False
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                assert "Colorear página" in response.read().decode("utf-8")

            def encoded(image):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

            image = Image.new("RGB", (64, 64), (20, 20, 20))
            hints = Image.new("RGBA", image.size)
            hints.putpixel((20, 20), (40, 80, 220, 255))
            payload = json.dumps({"image": encoded(image), "hints": encoded(hints), "size": 64,
                                  "lock_hints": True, "preserve_ink": True}).encode()
            request = urllib.request.Request(endpoint + "/colorize", data=payload,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            output = Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[1])))
            assert output.size == image.size
            assert output.getpixel((20, 20)) == (40, 80, 220)
            assert result["report"]["locked_regions"] is True
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
