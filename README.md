# Eye

Real-time body, face and object recognition in the browser, with a futuristic HUD.

- **Body:** up to 4 people, each limb tracked with joint angles; posture (standing, squat, T-pose, hands up), walking, jumping, squat and jump counters.
- **Hands:** up to 4 hands, each with its 21 finger points and a state: fist, holding something (with the object's name when the classifier recognizes it), open hand, pointing, thumbs up/down, victory, I love you. A curled hand counts as a fist when the fingertips close onto the palm, and as holding when they stay apart around an object.
- **Face:** eyes, irises, eyebrows and mouth; expression (smiling, threatening, surprised, sad, neutral), eyes open/closed/winking, mouth open/closed.
- **People counter:** a box and a stable ID for every person in view, with "in frame" and "total seen" counts. IDs survive brief occlusions, and one-frame false detections aren't counted.
- **Objects:** 80 everyday object types (cup, phone, laptop, bottle, chair…) with YOLO26 nano, the model used by [SharpAI/DeepCamera](https://github.com/SharpAI/DeepCamera). It runs on the graphics card through WebGPU when the browser supports it, otherwise on the processor.
- **Objects beyond YOLO's 80:** an ImageNet classifier (1000 categories, e.g. ballpoint pen, fountain pen, screwdriver, remote) looks at each hand and at the centre of the view, shown in green as "Left hand", "Right hand" and "In view". It works best when the object fills a good part of the picture. Results are averaged over time, and clothing and animal categories are ignored. It runs in a separate thread so it never slows the video.
- **Video files:** analyze a video from your device instead of the webcam (**Video file** button).
- **Face catalog:** the **Capture faces** button saves a snapshot of every face in view (up to 4), grouped by person ID with time and expression. Click a snapshot to download it. Snapshots stay in the browser's memory and are lost on reload.
- **Age, gender and emotion:** estimated for the face in the Face panel (about once a second) and for every catalog snapshot, with the small TensorFlow.js models of [face-api](https://github.com/vladmandic/face-api). Treat age and gender as rough guesses.
- **Recording:** saves the body and face panels as a `.webm` video.
- **Body / Face / Objects switches:** pause the analyses you don't need. On slower graphics cards, pausing Body and Face makes object detection about twice as fast.

The people counter is a browser port of the idea behind [scimone/PeopleDetector](https://github.com/scimone/PeopleDetector) (YOLOv5 + motpy in Python), rewritten from scratch.

Everything runs on the viewer's device with [MediaPipe](https://ai.google.dev/edge/mediapipe) and [ONNX Runtime Web](https://onnxruntime.ai). No video is uploaded anywhere.

## Run locally

Requires [Node.js](https://nodejs.org). Double-click `start.bat` (Windows) or run:

```
node server.mjs
```

Then open http://localhost:8080. Opening `index.html` directly doesn't work, because browsers block it.

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
