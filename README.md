# Eye

Real-time body, face and object recognition in the browser, with a futuristic HUD.

- **Body:** up to 4 people, each limb tracked with joint angles; posture (standing, squat, T-pose, hands up), walking, jumping, squat and jump counters.
- **Face:** eyes, irises, eyebrows and mouth; expression (smiling, threatening, surprised, sad, neutral), eyes open/closed/winking, mouth open/closed.
- **People counter:** a box and a stable ID for every person in view, with "in frame" and "total seen" counts. IDs survive brief occlusions, and one-frame false detections aren't counted.
- **Objects:** 80 everyday object types (cup, phone, laptop, bottle, chair…) with YOLO26 nano, the model used by [SharpAI/DeepCamera](https://github.com/SharpAI/DeepCamera). It runs on the graphics card through WebGPU when the browser supports it, otherwise on the processor.
- **Objects beyond YOLO's 80:** an ImageNet classifier (1000 categories, e.g. ballpoint pen, fountain pen, screwdriver, remote) looks at each hand and at the centre of the view, shown in green as "Left hand", "Right hand" and "In view". It works best when the object fills a good part of the picture. Results are averaged over time, and clothing and animal categories are ignored. It runs in a separate thread so it never slows the video.
- **Video files:** analyze a video from your device instead of the webcam (**Video file** button).
- **Face catalog:** the **Capture faces** button saves a snapshot of every face in view (up to 4), grouped by person ID with time and expression. Click a snapshot to download it. Snapshots stay in the browser's memory and are lost on reload.
- **Age, gender and emotion (optional, local only):** with [DeepFace](https://github.com/serengil/deepface) running on your PC, the Face panel and each catalog snapshot also show estimated age, gender and emotion. See [DeepFace](#deepface-optional) below.
- **Recording:** saves the body and face panels as a `.webm` video.
- **Body / Face / Objects switches:** pause the analyses you don't need. On slower graphics cards, pausing Body and Face makes object detection about twice as fast.

The people counter is a browser port of the idea behind [scimone/PeopleDetector](https://github.com/scimone/PeopleDetector) (YOLOv5 + motpy in Python), rewritten from scratch.

Everything runs on the viewer's device with [MediaPipe](https://ai.google.dev/edge/mediapipe) and [ONNX Runtime Web](https://onnxruntime.ai). No video is uploaded anywhere. The optional DeepFace helper receives face snapshots, but it runs on your own PC (127.0.0.1) and nothing leaves it.

## Run locally

Requires [Node.js](https://nodejs.org). Double-click `start.bat` (Windows) or run:

```
node server.mjs
```

Then open http://localhost:8080. Opening `index.html` directly doesn't work, because browsers block it.

## DeepFace (optional)

DeepFace is a Python library, so it runs as a small separate helper next to the page. Without it everything else works the same, and the Face panel shows "DeepFace offline".

1. Install [Python](https://www.python.org/downloads/) 3.11 or 3.12 and tick **Add python.exe to PATH**.
2. In this folder run `pip install -r requirements.txt` (TensorFlow included, a few hundred MB).
3. Double-click `start.bat`: it opens the DeepFace window too. Or run `python deepface_server.py` yourself.

The first start downloads the age, gender and emotion models (about 1 GB) and takes a few minutes; the page connects within 10 seconds of the helper saying "ready". It only works when Eye runs on this PC, not on the Vercel site.

## Deploy

It's a static site with no build step. Push the folder to GitHub, then on [vercel.com](https://vercel.com) choose **Add New → Project**, import the repository and press **Deploy**. Leave every setting at its default.

The camera only works over HTTPS, which Vercel provides.

## Test

```
node classify.test.mjs
```

## License

EYE is released under the [GNU AGPL-3.0](LICENSE), because it includes the YOLO26 model weights.

- `models/yolo26n.onnx`: Ultralytics YOLO26 nano, © Ultralytics, [AGPL-3.0](https://www.ultralytics.com/license). Taken from [SharpAI/DeepCamera](https://github.com/SharpAI/DeepCamera) (MIT).
- Body, face and image-classifier models: Google MediaPipe, Apache 2.0, loaded from Google's servers.
