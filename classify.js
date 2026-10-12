// Pure movement logic over MediaPipe Pose landmarks (33 points, normalized, y grows downward).
export const L = { nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lHip: 23, rHip: 24, lKn: 25, rKn: 26, lAn: 27, rAn: 28 };

// Angle at b (degrees) formed by a-b-c.
export function angle(a, b, c) {
  const ab = Math.atan2(a.y - b.y, a.x - b.x), cb = Math.atan2(c.y - b.y, c.x - b.x);
  let d = Math.abs((ab - cb) * 180 / Math.PI);
  return d > 180 ? 360 - d : d;
}

const center = p => ({ x: (p[L.lHip].x + p[L.rHip].x) / 2, y: (p[L.lHip].y + p[L.rHip].y) / 2 });
// Shoulder-to-hip height: the body's own scale, so thresholds work at any distance from the camera.
const torso = p => Math.abs(center(p).y - (p[L.lSh].y + p[L.rSh].y) / 2) || 1e-6;

// Returns a posture label for one frame.
export function classifyPose(p) {
  const knee = (angle(p[L.lHip], p[L.lKn], p[L.lAn]) + angle(p[L.rHip], p[L.rKn], p[L.rAn])) / 2;
  const elbow = Math.min(angle(p[L.lSh], p[L.lEl], p[L.lWr]), angle(p[L.rSh], p[L.rEl], p[L.rWr]));
  const t = torso(p);
  const wristsAtShoulder = Math.abs(p[L.lWr].y - p[L.lSh].y) < 0.25 * t && Math.abs(p[L.rWr].y - p[L.rSh].y) < 0.25 * t;

  if (p[L.lWr].y < p[L.nose].y && p[L.rWr].y < p[L.nose].y) return "Hands up";
  if (wristsAtShoulder && elbow > 150) return "T-pose";
  if (knee < 110) return "Squat";
  if (knee > 160) return "Standing";
  return "Bending";
}

// Movement over time from a short history [{t, lm}] (~1.5 s): "Jumping", "Walking" or null.
export function classifyMotion(hist) {
  if (hist.length < 5) return null;
  const cur = hist.at(-1).lm, t = torso(cur);
  // Jump: both feet clearly above the lowest ankle level seen recently (the floor).
  const floor = Math.max(...hist.map(h => (h.lm[L.lAn].y + h.lm[L.rAn].y) / 2));
  if (Math.max(cur[L.lAn].y, cur[L.rAn].y) < floor - 0.15 * t) return "Jumping";
  // Walk: the left-right ankle offset swings back and forth (x when seen from the side, y when facing the camera).
  for (const axis of ["x", "y"]) {
    const s = hist.map(h => (h.lm[L.lAn][axis] - h.lm[L.rAn][axis]) / torso(h.lm));
    const max = Math.max(...s), min = Math.min(...s), mean = (max + min) / 2, band = (max - min) / 4;
    if (max - min < 0.3) continue;
    let side = 0, flips = 0;
    for (const v of s) {
      if (Math.abs(v - mean) <= band) continue; // dead band so jitter around the middle isn't a step
      if (side && Math.sign(v - mean) !== side) flips++;
      side = Math.sign(v - mean);
    }
    if (flips >= 2) return "Walking";
  }
  return null;
}

export const LIMBS = [
  { name: "Left arm", ids: [L.lSh, L.lEl, L.lWr] },
  { name: "Right arm", ids: [L.rSh, L.rEl, L.rWr] },
  { name: "Left leg", ids: [L.lHip, L.lKn, L.lAn] },
  { name: "Right leg", ids: [L.rHip, L.rKn, L.rAn] },
];

// Per-limb status: visible?, bend angle at elbow/knee, and Raised / Extended / Bent.
export function limbStates(p) {
  return LIMBS.map(({ name, ids: [a, b, c] }) => {
    const visible = [a, b, c].every(i => (p[i].visibility ?? 1) > 0.5);
    const deg = Math.round(angle(p[a], p[b], p[c]));
    const raised = name.endsWith("arm") ? p[c].y < p[a].y : p[b].y < p[a].y; // wrist above shoulder / knee above hip
    const state = !visible ? "Not visible" : raised ? "Raised" : deg > 150 ? "Extended" : "Bent";
    return { name, ids: [a, b, c], visible, deg, state };
  });
}

// Facial expression from MediaPipe face blendshapes ({name: score 0..1}).
// shortcut: hand-tuned weights; train a classifier on labeled faces if accuracy matters.
export function classifyExpression(s) {
  const g = k => s[k] ?? 0, both = k => (g(k + "Left") + g(k + "Right")) / 2, clamp = v => Math.min(1, Math.max(0, v));
  const smile = both("mouthSmile");
  const scores = {
    Smiling: clamp(smile * 1.2),
    Threatening: clamp((both("browDown") * 1.5 + both("noseSneer") + both("eyeSquint") * 0.5 + both("mouthPress") * 0.5) * (1 - smile)),
    Surprised: clamp((g("browInnerUp") + both("eyeWide") + g("jawOpen")) / 1.5),
    Sad: clamp((both("mouthFrown") * 1.5 + g("browInnerUp") * 0.5) * (1 - smile)),
  };
  const [top, score] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0];
  const [l, r] = [g("eyeBlinkLeft") > 0.5, g("eyeBlinkRight") > 0.5];
  return {
    label: score > 0.45 ? top : "Neutral", scores,
    eyes: l && r ? "Closed" : l || r ? "Winking" : "Open",
    mouth: g("jawOpen") > 0.3 ? "Open" : "Closed",
  };
}

// The 80 COCO classes, in YOLO26 output order.
export const COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"];

// YOLO26 output → detections in video pixels, shaped like MediaPipe's so the drawing code is shared.
// logits: [300 × 80] raw class scores; boxes: [300 × 4] center x, center y, width, height as fractions of the letterboxed input.
// The model is NMS-free (one query per object), so no overlap filtering is needed.
export function decodeYolo(logits, boxes, { size = 640, scale, padX, padY, threshold = 0.45 }) {
  const out = [], K = COCO.length;
  for (let q = 0; q < boxes.length / 4; q++) {
    let bi = 0;
    for (let k = 1; k < K; k++) if (logits[q * K + k] > logits[q * K + bi]) bi = k;
    const score = 1 / (1 + Math.exp(-logits[q * K + bi]));
    if (score < threshold) continue;
    const [cx, cy, w, h] = [0, 1, 2, 3].map(i => boxes[q * 4 + i] * size);
    out.push({
      boundingBox: { originX: (cx - w / 2 - padX) / scale, originY: (cy - h / 2 - padY) / scale, width: w / scale, height: h / scale },
      categories: [{ categoryName: COCO[bi], score }],
    });
  }
  return out;
}

// Groups MediaPipe object detections by name: [{name, count, best}] most frequent first.
export function summarizeObjects(detections) {
  const by = {};
  for (const { categoryName: name, score } of detections.map(d => d.categories[0])) {
    by[name] ??= { name, count: 0, best: 0 };
    by[name].count++; by[name].best = Math.max(by[name].best, score);
  }
  return Object.values(by).sort((a, b) => b.count - a.count || b.best - a.best);
}

// Overlap of two {x, y, w, h} boxes, 0..1.
export function iou(a, b) {
  const x = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)), y = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  return x * y / (a.w * a.h + b.w * b.h - x * y || 1);
}

// People counter: stable ids for person boxes across detection steps (same idea as motpy in scimone/PeopleDetector).
// A track gets an id only after minHits detections (drops one-off false positives) and keeps it through maxStale missed steps (brief occlusions).
// shortcut: greedy IoU matching without motion prediction; add a Kalman filter if fast movers lose their id.
export function makeBoxTracker({ minIou = 0.3, maxStale = 10, minHits = 3 } = {}) {
  let tracks = [], nextId = 1;
  const step = boxes => {
    const free = new Set(tracks);
    for (const b of [...boxes].sort((p, q) => q.score - p.score)) {
      let best = null, bv = minIou;
      for (const t of free) { const v = iou(t.box, b); if (v > bv) { bv = v; best = t; } }
      if (best) free.delete(best); else tracks.push(best = { hits: 0 });
      Object.assign(best, { box: b, score: b.score, hits: best.hits + 1, stale: 0 });
      if (!best.id && best.hits >= minHits) best.id = nextId++;
    }
    for (const t of free) t.stale++;
    tracks = tracks.filter(t => t.stale <= maxStale);
    return tracks.filter(t => t.id && t.stale === 0); // confirmed people seen in this step
  };
  step.total = () => nextId - 1;
  return step;
}

// Mean landmark displacement between two frames (normalized units).
export function motion(prev, cur) {
  if (!prev) return 0;
  let s = 0;
  for (let i = 0; i < cur.length; i++) s += Math.hypot(cur[i].x - prev[i].x, cur[i].y - prev[i].y);
  return s / cur.length;
}

// Counts reps: one rep = entering `active`, then reaching `rest` (any other label if rest is omitted).
export function makeRepCounter(active, rest) {
  let down = false, reps = 0;
  return label => {
    if (label === active) down = true;
    else if (down && (rest === undefined || label === rest)) { down = false; reps++; }
    return reps;
  };
}

// Keeps a stable id per person across frames by matching hip centers to last known positions.
// shortcut: greedy nearest match, can swap ids when people cross paths; upgrade to Hungarian matching if that matters.
export function makeTracker(maxDist = 0.2, ttl = 1000) {
  let tracks = [], nextId = 1;
  return (poses, now) => {
    const free = new Set(tracks);
    const out = poses.map(lm => {
      const c = center(lm);
      let best = null, bd = maxDist;
      for (const tr of free) { const d = Math.hypot(c.x - tr.c.x, c.y - tr.c.y); if (d < bd) { bd = d; best = tr; } }
      if (best) free.delete(best); else best = { id: nextId++ };
      Object.assign(best, { c, seen: now, lm });
      return best;
    });
    tracks = [...out, ...[...free].filter(tr => now - tr.seen < ttl)]; // briefly hidden people keep their id
    return out;
  };
}
