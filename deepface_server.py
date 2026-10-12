"""Optional DeepFace helper for Eye: age, gender and emotion for the face crops the page sends.

Listens on this PC only (127.0.0.1:5005). Setup once: pip install -r requirements.txt
Then: python deepface_server.py (start.bat does this for you when DeepFace is installed).
"""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from deepface import DeepFace

PORT, MAX_BYTES = 5005, 2_000_000  # a 224 px JPEG crop is ~20 KB; anything near the cap is not ours
ACTIONS = ["age", "gender", "emotion"]


def analyze(img):
    # The page sends an already cropped face, so DeepFace's own face detector is skipped.
    return DeepFace.analyze(img_path=img, actions=ACTIONS, detector_backend="skip", enforce_detection=False, silent=True)[0]


class Handler(BaseHTTPRequestHandler):
    def headers_out(self, code, length=0):
        self.send_response(code)
        self.send_header("Access-Control-Allow-Origin", "*")  # the page comes from another port (or from Vercel)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def reply(self, code, body):
        data = json.dumps(body).encode()
        self.headers_out(code, len(data))
        self.wfile.write(data)

    def do_OPTIONS(self):  # CORS preflight
        self.headers_out(204)

    def do_POST(self):
        if self.path != "/analyze":
            return self.reply(404, {"error": "not found"})
        size = int(self.headers.get("Content-Length") or 0)
        if not 0 < size <= MAX_BYTES:
            return self.reply(413, {"error": "image missing or too large"})
        try:
            img = json.loads(self.rfile.read(size)).get("img")
        except (ValueError, AttributeError):
            return self.reply(400, {"error": "body must be JSON {img: data URL}"})
        if not isinstance(img, str) or not img.startswith("data:image/"):
            return self.reply(400, {"error": "img must be a data:image/... URL"})
        try:
            r = analyze(img)
        except Exception as e:  # bad image data: tell the page, keep serving
            return self.reply(422, {"error": str(e)[:200]})
        self.reply(200, {"age": int(r["age"]), "gender": r["dominant_gender"], "emotion": r["dominant_emotion"]})

    def log_message(self, *args):  # one line per request would flood the console
        pass


if __name__ == "__main__":
    print("Loading DeepFace models (the first run downloads about 1 GB)...")
    analyze(np.zeros((224, 224, 3), np.uint8))  # warm-up: download and load now, not on the first face
    print(f"DeepFace ready on http://127.0.0.1:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
