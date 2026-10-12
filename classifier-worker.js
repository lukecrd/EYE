// ImageNet classifier (1000 categories: ballpoint pen, fountain pen, remote, screwdriver…) for the objects YOLO's
// 80 classes miss. Runs in this worker so a slow CPU never freezes the video. CPU delegate: MediaPipe's
// graphics-card path can't run this compressed (int8) model.
const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const ready = (async () => {
  const { ImageClassifier, FilesetResolver } = await import(CDN);
  const vision = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  return ImageClassifier.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite2/int8/1/efficientnet_lite2.tflite", delegate: "CPU" },
    runningMode: "IMAGE", maxResults: 15, // extra results so some survive the held-object filter
  });
})();

// Message in: { side, bitmap } (a 224×224 crop). Message out: { side, categories, ms } or { side, error }.
onmessage = async ({ data: { side, bitmap } }) => {
  try {
    const classifier = await ready, t0 = performance.now();
    const categories = classifier.classify(bitmap).classifications[0].categories.map(({ categoryName, index, score }) => ({ categoryName, index, score }));
    postMessage({ side, categories, ms: performance.now() - t0 });
  } catch (e) {
    postMessage({ side, error: String(e?.message ?? e) });
  } finally {
    bitmap.close();
  }
};
