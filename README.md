# Eye

Real-time body, face and object recognition in the browser, with a futuristic HUD.

- **Body:** up to 4 people, each limb tracked with joint angles; posture (standing, squat, T-pose, hands up), walking, jumping, squat and jump counters.
- **Face:** eyes, irises, eyebrows and mouth; expression (smiling, threatening, surprised, sad, neutral), eyes open/closed/winking, mouth open/closed.
- **People counter:** a box and a stable ID for every person in view, with "in frame" and "total seen" counts. IDs survive brief occlusions, and one-frame false detections aren't counted.
- **Objects:** 80 everyday object types (cup, phone, laptop, bottle, chair…).
- **Video files:** analyze a video from your device instead of the webcam (**Video file** button).
- **Recording:** saves the body and face panels as a `.webm` video.

The people counter is a browser port of the idea behind [scimone/PeopleDetector](https://github.com/scimone/PeopleDetector) (YOLOv5 + motpy in Python), rewritten from scratch.

Everything runs on the viewer's device with [MediaPipe](https://ai.google.dev/edge/mediapipe). No video is uploaded anywhere.

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
