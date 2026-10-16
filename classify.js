// Pure movement logic over MediaPipe Pose landmarks (33 points, normalized, y grows downward).
export const L = { nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16, lHip: 23, rHip: 24, lKn: 25, rKn: 26, lAn: 27, rAn: 28 };

// Angle at b (degrees) formed by a-b-c.
export function angle(a, b, c) {
  const ab = Math.atan2(a.y - b.y, a.x - b.x), cb = Math.atan2(c.y - b.y, c.x - b.x);
  let d = Math.abs((ab - cb) * 180 / Math.PI);
  return d > 180 ? 360 - d : d;
}
// Angle at b in 3D (degrees): a limb pointing at the camera keeps its true bend, where the 2D angle collapses.
const angle3 = (a, b, c) => {
  const u = [a.x - b.x, a.y - b.y, a.z - b.z], v = [c.x - b.x, c.y - b.y, c.z - b.z];
  return Math.acos(Math.max(-1, Math.min(1, (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v) || 1e-9)))) * 180 / Math.PI;
};
// Joint angle at b: from world landmarks w (3D, meters) when given, else from the 2D image landmarks p.
const bend = (p, w, a, b, c) => w ? angle3(w[a], w[b], w[c]) : angle(p[a], p[b], p[c]);

// One Euro filter over a landmark array (x, y, z of each point; visibility passes through), t in ms.
// Still points are smoothed hard (no jitter), fast ones barely (no lag): the cutoff rises with speed.
// shortcut: minCutoff/beta are hand-picked for normalized coords at ~30 fps; tune on the camera if joints lag or still shake.
export function makeOneEuro({ minCutoff = 1, beta = 5, dCutoff = 1 } = {}) {
  let prev = null, dx = null, last = 0;
  const a = (cut, dt) => 1 / (1 + 1 / (2 * Math.PI * cut * dt));
  return (lm, t) => {
    const dt = (t - last) / 1000;
    if (!prev || prev.length !== lm.length) { last = t; dx = lm.map(() => ({ x: 0, y: 0, z: 0 })); return prev = lm.map(q => ({ ...q })); }
    if (dt <= 0) return prev; // same frame again
    last = t;
    return prev = lm.map((q, i) => {
      const o = { ...q };
      for (const k of ["x", "y", "z"]) if (k in q) {
        dx[i][k] += a(dCutoff, dt) * ((q[k] - prev[i][k]) / dt - dx[i][k]);
        o[k] = prev[i][k] + a(minCutoff + beta * Math.abs(dx[i][k]), dt) * (q[k] - prev[i][k]);
      }
      return o;
    });
  };
}

const center = p => ({ x: (p[L.lHip].x + p[L.rHip].x) / 2, y: (p[L.lHip].y + p[L.rHip].y) / 2 });
// Shoulder-to-hip height: the body's own scale, so thresholds work at any distance from the camera.
const torso = p => Math.abs(center(p).y - (p[L.lSh].y + p[L.rSh].y) / 2) || 1e-6;
const seen = (p, ...ids) => ids.every(i => (p[i].visibility ?? 1) > 0.5);

// Returns a posture label for one frame. p: image landmarks (positions on screen); w: optional world landmarks (joint angles).
export function classifyPose(p, w) {
  const knee = (bend(p, w, L.lHip, L.lKn, L.lAn) + bend(p, w, L.rHip, L.rKn, L.rAn)) / 2;
  const elbow = Math.min(bend(p, w, L.lSh, L.lEl, L.lWr), bend(p, w, L.rSh, L.rEl, L.rWr));
  const t = torso(p);
  const wristsAtShoulder = Math.abs(p[L.lWr].y - p[L.lSh].y) < 0.25 * t && Math.abs(p[L.rWr].y - p[L.rSh].y) < 0.25 * t;

  if (p[L.lWr].y < p[L.nose].y && p[L.rWr].y < p[L.nose].y) return "Hands up";
  if (wristsAtShoulder && elbow > 150) return "T-pose";
  // MediaPipe guesses knees and ankles out of frame (seated at a desk), so leg postures need the legs really seen.
  if (!seen(p, L.lHip, L.rHip, L.lKn, L.rKn, L.lAn, L.rAn)) return "Upper body";
  if (knee < 110) return "Squat";
  if (knee > 160) return "Standing";
  return "Bending";
}

const median = a => { const s = [...a].sort((p, q) => p - q); return s[s.length >> 1]; };
// Jump: for the last `frames` frames, feet visible and both feet and hips clearly above their usual level (the median of
// the window, so one noisy frame can't move the floor), at the same body size (walking away also moves everything up).
// MediaPipe guesses ankles even when the feet are out of frame, so without the visibility check any wiggle at a desk was a jump.
// Thresholds, exported so the debug overlay can show each measurement next to the value it's compared with.
export const JUMP = { rise: 0.25, frames: 3, minScale: 0.85 }; // rise in torso lengths
export const WALK = { legs: 0.8, travel: 0.3, grow: 1.15 }; // share of frames with legs seen; hip travel in torso lengths; size ratio
// Per-frame jump measures against the window's usual level: rise of feet and hips in torso lengths, body size ratio, feet/hips seen.
function jumpStats(hist) {
  const feet = p => Math.max(p[L.lAn].y, p[L.rAn].y), hips = p => center(p).y;
  const floor = median(hist.map(h => feet(h.lm))), hip = median(hist.map(h => hips(h.lm))), size = median(hist.map(h => torso(h.lm)));
  return lm => ({ feet: (floor - feet(lm)) / size, hips: (hip - hips(lm)) / size, scale: torso(lm) / size, seen: seen(lm, L.lAn, L.rAn, L.lHip, L.rHip) });
}
function isJumping(hist) {
  if (hist.length < JUMP.frames + 3) return false;
  const at = jumpStats(hist);
  return hist.slice(-JUMP.frames).every(({ lm }) => { const s = at(lm); return s.seen && s.feet > JUMP.rise && s.hips > JUMP.rise && s.scale > JUMP.minScale; });
}
// Walking gate measures: share of frames with knees and ankles seen, sideways hip travel (torso lengths), size change ratio.
function walkStats(hist) {
  const hx = hist.map(h => center(h.lm).x), sizes = hist.map(h => torso(h.lm));
  return { legs: hist.filter(h => seen(h.lm, L.lKn, L.rKn, L.lAn, L.rAn)).length / hist.length,
    travel: (Math.max(...hx) - Math.min(...hx)) / median(sizes), grow: Math.max(...sizes) / Math.min(...sizes) };
}
// What jump and walk detection measure on the latest frame, for the debug overlay; null until there's enough history.
export function motionStats(hist) {
  return hist.length < 5 ? null : { jump: jumpStats(hist)(hist.at(-1).lm), walk: walkStats(hist) };
}

// Movement over time from a short history [{t, lm}] (~1.5 s): "Jumping", "Walking" or null.
export function classifyMotion(hist) {
  if (hist.length < 5) return null;
  if (isJumping(hist)) return "Jumping";
  // Walking needs legs in view (MediaPipe guesses hidden ankles, so sitting at a desk looked like walking) and a body that goes somewhere:
  // hips travel sideways or the body grows/shrinks (walking toward/away). Legs swinging under a chair don't move the hips.
  // shortcut: walking on the spot (treadmill) isn't recognized; add a knee-lift check if that's needed.
  const w = walkStats(hist);
  if (w.legs < WALK.legs || (w.travel < WALK.travel && w.grow < WALK.grow)) return null;
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

// Per-limb status: visible?, bend angle at elbow/knee (3D from world landmarks w when given), and Raised / Extended / Bent.
export function limbStates(p, w) {
  return LIMBS.map(({ name, ids: [a, b, c] }) => {
    const visible = seen(p, a, b, c);
    const deg = Math.round(bend(p, w, a, b, c));
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
// Groups: each track is matched where it's heading (constant velocity), and pairs are taken best overlap first over all
// tracks and boxes, so two people crossing or standing shoulder to shoulder don't swap ids.
// shortcut: global greedy, not Hungarian, and no Kalman filter; upgrade if dense crowds still swap ids.
// Someone walking through a door can move more than their own width between two scans (slow CPU detection), so the boxes
// don't overlap at all and every scan looked like a new person that never got confirmed or counted. Leftover boxes are
// then matched by distance (within maxJump box widths, similar height) to a track seen in the previous scan, ranked below any overlap.
// shortcut: a person entering right where another left in the previous scan inherits that id; fine for a door, not for a turnstile.
export function makeBoxTracker({ minIou = 0.3, maxStale = 10, minHits = 3, maxJump = 2 } = {}) {
  let tracks = [], nextId = 1;
  const ahead = t => ({ ...t.box, x: t.box.x + t.vx * (t.stale + 1), y: t.box.y + t.vy * (t.stale + 1) });
  const near = (a, b) => {
    const d = Math.hypot(a.x + a.w / 2 - b.x - b.w / 2, a.y + a.h / 2 - b.y - b.h / 2) / Math.max(a.w, b.w), r = b.h / a.h;
    return d < maxJump && r > 2 / 3 && r < 1.5 ? minIou * (1 - d / maxJump) : 0; // always below minIou: overlaps win
  };
  const step = boxes => {
    const pairs = [];
    for (const t of tracks) {
      const p = ahead(t);
      boxes.forEach((b, i) => {
        const v = Math.max(iou(p, b), iou(t.box, b)); // last box too: someone who stopped while hidden isn't predicted away
        if (v > minIou) pairs.push([v, t, i]);
        else if (t.stale === 0) { const n = near(p, b); if (n > 0) pairs.push([n, t, i]); }
      });
    }
    const free = new Set(tracks), done = new Set();
    for (const [, t, i] of pairs.sort((p, q) => q[0] - p[0])) {
      if (!free.has(t) || done.has(i)) continue;
      free.delete(t); done.add(i);
      const b = boxes[i];
      t.vx = 0.5 * t.vx + 0.5 * (b.x - t.box.x) / (t.stale + 1); t.vy = 0.5 * t.vy + 0.5 * (b.y - t.box.y) / (t.stale + 1);
      Object.assign(t, { box: b, score: b.score, hits: t.hits + 1, stale: 0 });
    }
    boxes.forEach((b, i) => { if (!done.has(i)) tracks.push({ box: b, score: b.score, hits: 1, stale: 0, vx: 0, vy: 0 }); });
    for (const t of tracks) if (!t.id && t.hits >= minHits) t.id = nextId++;
    for (const t of free) t.stale++;
    tracks = tracks.filter(t => t.stale <= maxStale);
    return tracks.filter(t => t.id && t.stale === 0); // confirmed people seen in this step
  };
  step.total = () => nextId - 1;
  return step;
}

// Where a body part sits in a person box, as a fraction of its height from the top.
export const ANCHOR = { head: 0.12, hips: 0.55 };
// One person box per point ({x, y} pixels), or null: among the boxes containing the point, the one whose anchor (e.g. the head
// spot) is closest, in box sizes. In a group the face of the person in front also lies inside the box of the one behind, and
// "smallest box" picked the wrong one. Each box goes to at most one point (closest pairs first), since a box is one person.
export function assignToBoxes(points, tracks, anchor = ANCHOR.hips) {
  const pairs = [];
  points.forEach((p, i) => tracks.forEach(t => {
    const { x, y, w, h } = t.box;
    if (p.x >= x && p.x <= x + w && p.y >= y && p.y <= y + h) pairs.push([Math.hypot((p.x - x - w / 2) / w, (p.y - y - anchor * h) / h), i, t]);
  }));
  const out = points.map(() => null), used = new Set();
  for (const [, i, t] of pairs.sort((a, b) => a[0] - b[0])) if (!out[i] && !used.has(t)) { out[i] = t; used.add(t); }
  return out;
}
export const boxAt = (tracks, x, y, anchor) => assignToBoxes([{ x, y }], tracks, anchor)[0];

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
export const HEAD_MAX_DIFF = 0.06; // at or above this signature difference, two heads count as different people
export function makeHeadIds({ maxDiff = HEAD_MAX_DIFF, samples = 8 } = {}) {
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
    // Closest known head however far, for the debug overlay: {id, diff}, or null before anyone is known.
    nearest(sig) {
      let id = null, diff = Infinity;
      for (const [pid, h] of heads) { const v = sigDiff(sig, h.sig); if (v < diff) { diff = v; id = pid; } }
      return id === null ? null : { id, diff };
    },
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

// Which person's hand it is: the nearest pose wrist, as {pose index, side} (pose sides are the person's own left/right).
// MediaPipe's handedness label isn't used: it assumes a mirrored image, and video files aren't mirrored.
export function assignHands(hands, poses, maxDist = 0.12) {
  return hands.map(h => {
    let best = null, bd = maxDist;
    poses.forEach((p, pose) => {
      for (const side of ["left", "right"]) {
        const w = p[side === "left" ? L.lWr : L.rWr], d = Math.hypot(w.x - h[0].x, w.y - h[0].y);
        if (d < bd) { bd = d; best = { pose, side }; }
      }
    });
    return best;
  });
}

// Hand state from MediaPipe hand world landmarks (21 points in meters, so thresholds hold at any distance from the camera).
// A curled hand is a fist when the fingertips close onto the palm, and holding something when they stay apart, wrapped
// around it; `holding` (an object seen in that hand) also tells a pen pinch from an empty "OK" sign.
// shortcut: GRIP.fistGap is a hand-tuned average adult hand; calibrate on real hands if fists read as holding or the reverse.
const FINGERS = [[5, 6, 8], [9, 10, 12], [13, 14, 16], [17, 18, 20]]; // MCP, PIP and tip of index…pinky
export const GRIP = { fistGap: 0.045, curl: 120, pinch: 0.03 }; // m, degrees at the middle joint, m
const GESTURES = { Open_Palm: "Open hand", Pointing_Up: "Pointing", Thumb_Up: "Thumbs up", Thumb_Down: "Thumbs down", Victory: "Victory", ILoveYou: "I love you" };
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
// What handPose measures (also shown by the debug overlay): curled fingers, mean fingertip-to-palm gap and thumb-index gap in meters.
export function handStats(w) {
  const palm = [0, 5, 9, 13, 17].reduce((s, i) => ({ x: s.x + w[i].x / 5, y: s.y + w[i].y / 5, z: s.z + w[i].z / 5 }), { x: 0, y: 0, z: 0 });
  return { curled: FINGERS.filter(([m, p, t]) => angle3(w[m], w[p], w[t]) < GRIP.curl).length,
    gap: FINGERS.reduce((s, [, , t]) => s + d3(w[t], palm), 0) / FINGERS.length, pinch: d3(w[4], w[8]) };
}
export function handPose(w, { gesture = "None", holding = false } = {}) {
  const { curled, gap, pinch } = handStats(w), named = GESTURES[gesture];
  if (named && !holding) return named; // thumbs up and pointing are curled hands too, so the recognizer's names come first
  if (curled >= 3) return gap < GRIP.fistGap && !holding ? "Fist" : "Holding";
  if (holding && (pinch < GRIP.pinch || curled >= 2)) return "Holding"; // pinch grip (pen) or a loose grip
  return named ?? (curled === 0 ? "Open hand" : "Relaxed");
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

// Per-person age, gender and emotion that settle instead of jumping each second. Age and gender are means over the last
// ~maxN readings (gender as the probability of "male"); emotion follows an exponential average of the expression scores,
// slower than one reading but still able to change. r: {age, gender, genderScore, scores: {emotion: 0..1}}.
export function makeTraitAverager({ maxN = 30, alpha = 0.3 } = {}) {
  const by = new Map();
  return (id, r) => {
    const p = by.get(id) ?? { n: 0, age: 0, male: 0, emo: {} };
    p.n = Math.min(p.n + 1, maxN);
    p.age += (r.age - p.age) / p.n;
    p.male += ((r.gender === "male" ? r.genderScore : 1 - r.genderScore) - p.male) / p.n;
    for (const [k, v] of Object.entries(r.scores)) p.emo[k] = p.emo[k] === undefined ? v : p.emo[k] + alpha * (v - p.emo[k]);
    by.set(id, p);
    const [emotion] = Object.entries(p.emo).sort((a, b) => b[1] - a[1])[0] ?? [r.emotion];
    return { age: Math.round(p.age), gender: p.male >= 0.5 ? "male" : "female", genderScore: Math.max(p.male, 1 - p.male), emotion, n: p.n };
  };
}

// Event log: turns noisy per-frame states ({key: value}) into change events [{key, from, to}]. A value must hold for `hold` ms
// before it's reported (one-frame flickers and the brief id a box has before head re-identification never show up),
// and a key missing for `gone` ms reports to: null (left the view, disappeared).
export function makeStateLog({ hold = 1000, gone = 3000 } = {}) {
  const keys = new Map();
  return (states, now) => {
    const events = [];
    for (const [key, value] of Object.entries(states)) {
      const k = keys.get(key) ?? { shown: null, cand: null, since: now };
      if (value !== k.cand) { k.cand = value; k.since = now; }
      k.seen = now; keys.set(key, k);
      if (k.cand !== k.shown && now - k.since >= hold) { events.push({ key, from: k.shown, to: k.cand }); k.shown = k.cand; }
    }
    for (const [key, k] of keys) if (now - k.seen >= gone) {
      if (k.shown !== null) events.push({ key, from: k.shown, to: null });
      keys.delete(key);
    }
    return events;
  };
}

// Rows of values → CSV text (quotes doubled, every field quoted so commas and line breaks are safe).
export const toCsv = rows => rows.map(r => r.map(v => `"${String(v ?? "").replace(/"/g, '""')}"`).join(",")).join("\r\n");

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
