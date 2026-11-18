"""Small loopback-only editor; no external web service or runtime dependency."""
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.resources import files

from PIL import Image

from .inference import ColorizationPipeline, parse_palette


def decode_image(value):
    if not isinstance(value, str) or not value.startswith("data:image/png;base64,"):
        raise ValueError("Se esperaba una imagen PNG del editor.")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    image = Image.open(io.BytesIO(raw))
    if image.width * image.height > 25_000_000:
        raise ValueError("La imagen supera 25 millones de píxeles.")
    image.load()
    return image


def serve(checkpoint, device="cpu", port=7860):
    pipeline = ColorizationPipeline(checkpoint, device)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, content_type="application/json"):
            data = value.encode("utf-8") if isinstance(value, str) else json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, files("mangacolor").joinpath("web/editor.html").read_text(encoding="utf-8"), "text/html")
            elif self.path == "/capabilities":
                meta = dict(pipeline.metadata)
                meta.update({
                    "reconstruction_v1": True,
                    "guided_upsampling": True,
                    "gamut_mapping": True,
                })
                self.reply(200, meta)
            else:
                self.reply(404, {"error": "Ruta no encontrada"})

        def do_POST(self):
            if self.path != "/colorize":
                self.reply(404, {"error": "Ruta no encontrada"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 40_000_000:
                    raise ValueError("Tamaño de petición inválido.")
                payload = json.loads(self.rfile.read(length))
                image = decode_image(payload["image"]).convert("RGB")
                hints = decode_image(payload["hints"]).convert("RGBA")
                export_size = payload.get("export_size", "original")
                reconstruct_mode = payload.get("reconstruct_mode", "guided")
                result, report = pipeline.colorize(
                    image, hints, parse_palette(payload.get("palette", [])),
                    int(payload.get("size", 576)), payload.get("palette_mode", "hints"),
                    bool(payload.get("lock_hints", False)), bool(payload.get("preserve_ink", False)),
                    reconstruct_mode=reconstruct_mode, export_size=export_size)
                buffer = io.BytesIO()
                result.save(buffer, format="PNG")
                self.reply(200, {"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
                                 "report": report})
            except (ValueError, KeyError, OSError, RuntimeError) as exc:
                self.reply(400, {"error": str(exc)})

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Editor local: http://127.0.0.1:{port} (Ctrl+C para salir)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
