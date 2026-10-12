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

// The tracked box containing pixel (x, y); the smallest wins when boxes overlap (the nearer, tighter fit).
export function boxAt(tracks, x, y) {
  let best = null;
  for (const t of tracks) {
    const { x: bx, y: by, w, h } = t.box;
    if (x >= bx && x <= bx + w && y >= by && y <= by + h && (!best || w * h < best.box.w * best.box.h)) best = t;
  }
  return best;
}

// Head signature: 3D distances between rigid face landmarks (eye corners, nose, cheekbones, brows, forehead) divided by
// the outer eye span, so it holds at any distance and head turn. Mouth and jaw are left out: they move with expressions.
const RIGID = [33, 133, 362, 263, 1, 168, 2, 98, 327, 234, 454, 10, 70, 300];
export function faceSignature(lm, W, H) {
  const p = RIGID.map(i => [lm[i].x * W, lm[i].y * H, (lm[i].z ?? 0) * W]); // MediaPipe z is on the same scale as x
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]), unit = d(p[0], p[3]) || 1e-6, out = [];
  for (let i = 0; i < p.length; i++) for (let j = i + 1; j < p.length; j++) out.push(d(p[i], p[j]) / unit);
  return out;
}
// Mean relative difference of two signatures: 0 = same head geometry.
export const sigDiff = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]) / ((v + b[i]) / 2 || 1), 0) / a.length;

// Head re-identification: person box track ids → stable person ids. A new box collects `samples` head signatures, then takes
// the id of the closest known head (someone walking back in) or becomes a new person. Known heads also answer for faces with no box.
// shortcut: landmark geometry separates a handful of people in one session, not strangers in a crowd; swap in a face-embedding model (e.g. ArcFace ONNX) for that.
export function makeHeadIds({ maxDiff = 0.06, samples = 8 } = {}) {
  const heads = new Map(), alias = new Map(), pending = new Map();
  const best = (sig, skip = new Set()) => {
    let id = null, bd = maxDiff;
    for (const [pid, h] of heads) if (!skip.has(pid)) { const v = sigDiff(sig, h.sig); if (v < bd) { bd = v; id = pid; } }
    return id;
  };
  const learn = (pid, sig) => { // running mean, capped so a head keeps adapting to light and pose
    const h = heads.get(pid) ?? { sig, n: 0 };
    h.n = Math.min(h.n + 1, 50); h.sig = h.sig.map((v, i) => v + (sig[i] - v) / h.n);
    heads.set(pid, h);
  };
  return {
    id: boxId => alias.get(boxId) ?? boxId,
    // busy: person ids on the other boxes in view, since one person can't be in two places.
    observe(boxId, sig, busy) {
      if (alias.has(boxId)) return learn(alias.get(boxId), sig);
      const got = pending.get(boxId) ?? [];
      pending.set(boxId, got); got.push(sig);
      if (got.length < samples) return;
      pending.delete(boxId);
      const mean = got[0].map((_, i) => got.reduce((s, g) => s + g[i], 0) / got.length), pid = best(mean, busy) ?? boxId;
      alias.set(boxId, pid); learn(pid, mean);
    },
    match: sig => best(sig),
    merged: () => [...alias].filter(([b, p]) => b !== p).length, // box tracks that turned out to be someone already seen
  };
}

// Square crop around normalized face landmarks, in video pixels, padded so the whole head fits.
export function faceCrop(lm, W, H, pad = 1.4) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const q of lm) { x0 = Math.min(x0, q.x); y0 = Math.min(y0, q.y); x1 = Math.max(x1, q.x); y1 = Math.max(y1, q.y); }
  const size = Math.max((x1 - x0) * W, (y1 - y0) * H) * pad;
  return { sx: (x0 + x1) / 2 * W - size / 2, sy: (y0 + y1) / 2 * H - size / 2, size, cx: (x0 + x1) / 2 * W, cy: (y0 + y1) / 2 * H };
}

// Square crop around one hand from pose landmarks, in video pixels, for classifying what the hand holds.
// Centered past the wrist toward the index knuckle; sized from the forearm so it scales with distance from the camera.
const HAND = { left: [L.lEl, L.lWr, 19], right: [L.rEl, L.rWr, 20] };
export function handCrop(p, side, W, H) {
  const [e, w, i] = HAND[side].map(k => ({ x: p[k].x * W, y: p[k].y * H }));
  const fore = Math.hypot(w.x - e.x, w.y - e.y), hand = Math.hypot(i.x - w.x, i.y - w.y);
  const cx = w.x + (i.x - w.x) * 0.8, cy = w.y + (i.y - w.y) * 0.8, size = Math.max(48, fore * 1.65, hand * 6); // wide enough for a pen sticking out of the fist
  return { sx: cx - size / 2, sy: cy - size / 2, size, visible: (p[HAND[side][1]].visibility ?? 1) > 0.5 };
}

// ImageNet results that can be a held or shown object: man-made things and food (indices 398–969, 998–999),
// minus clothing and body-worn items the classifier reports whenever a hand is in front of the body.
const WORN = new Set(["abaya", "academic gown", "apron", "bow tie", "brassiere", "breastplate", "bulletproof vest", "cardigan", "chain mail",
  "cloak", "cowboy hat", "cuirass", "fur coat", "gown", "hoopskirt", "jean", "jersey", "kimono", "lab coat", "maillot", "military uniform",
  "miniskirt", "mortarboard", "overskirt", "pajama", "poncho", "sarong", "suit", "sweatshirt", "swimming trunks", "trench coat", "vestment",
  "velvet", "wig", "Windsor tie", "wool", "stole", "bonnet", "sombrero", "shower cap", "Band Aid", "neck brace", "bib", "feather boa",
  "mask", "ski mask", "seat belt", "pickelhaube"]);
export const heldCandidates = categories =>
  categories.filter(c => ((c.index >= 398 && c.index <= 969) || c.index >= 998) && !WORN.has(c.categoryName));

// Running average of classifier scores ({label: score}): a real object keeps scoring the same label,
// one-frame noise fades out. Returns the new map and its best label.
export function smoothScores(prev, categories, alpha = 0.35) {
  const next = {};
  for (const [k, v] of Object.entries(prev)) if (v * (1 - alpha) > 0.01) next[k] = v * (1 - alpha);
  for (const { categoryName, score } of categories) next[categoryName] = (next[categoryName] ?? 0) + alpha * score;
  let label = null, score = 0;
  for (const [k, v] of Object.entries(next)) if (v > score) { label = k; score = v; }
  return { scores: next, label, score };
}

// Face catalog: captures grouped by person id, most recently seen person first, newest capture first.
export function groupCaptures(captures) {
  const by = new Map();
  for (const c of [...captures].sort((a, b) => b.t - a.t)) {
    if (!by.has(c.id)) by.set(c.id, []);
    by.get(c.id).push(c);
  }
  return [...by].map(([id, items]) => ({ id, items }));
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
