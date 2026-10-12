import assert from "node:assert/strict";
import { angle, classifyPose, classifyMotion, motion, makeRepCounter, makeTracker, limbStates, classifyExpression, summarizeObjects, iou, makeBoxTracker, decodeYolo, COCO, L } from "./classify.js";

// Build a 33-point skeleton from a few named joints.
const body = j => Array.from({ length: 33 }, (_, i) => j[Object.keys(L).find(k => L[k] === i)] ?? { x: 0.5, y: 0.5 });
const base = { nose: { x: .5, y: .1 }, lSh: { x: .4, y: .2 }, rSh: { x: .6, y: .2 }, lHip: { x: .45, y: .5 }, rHip: { x: .55, y: .5 },
  lKn: { x: .45, y: .7 }, rKn: { x: .55, y: .7 }, lAn: { x: .45, y: .9 }, rAn: { x: .55, y: .9 },
  lEl: { x: .38, y: .35 }, rEl: { x: .62, y: .35 }, lWr: { x: .38, y: .48 }, rWr: { x: .62, y: .48 } };
const shift = (j, dx, dy) => Object.fromEntries(Object.entries(j).map(([k, v]) => [k, { x: v.x + dx, y: v.y + dy }]));

// Posture
assert.equal(Math.round(angle({ x: 1, y: 0 }, { x: 0, y: 0 }, { x: 0, y: 1 })), 90);
assert.equal(classifyPose(body(base)), "Standing");
assert.equal(classifyPose(body({ ...base, lWr: { x: .4, y: 0 }, rWr: { x: .6, y: 0 } })), "Hands up");
assert.equal(classifyPose(body({ ...base, lEl: { x: .3, y: .2 }, rEl: { x: .7, y: .2 }, lWr: { x: .2, y: .2 }, rWr: { x: .8, y: .2 } })), "T-pose");
const squat = body({ ...base, lKn: { x: .3, y: .6 }, rKn: { x: .7, y: .6 }, lAn: { x: .45, y: .7 }, rAn: { x: .55, y: .7 } });
assert.equal(classifyPose(squat), "Squat");
assert.equal(motion(null, squat), 0);
assert.ok(motion(body(base), squat) > 0);

// Motion over time
const hist = frames => frames.map((lm, i) => ({ t: i * 100, lm }));
const still = Array(10).fill(body(base));
assert.equal(classifyMotion(hist(still)), null);
assert.equal(classifyMotion(hist(still.slice(0, 3))), null); // too little history
assert.equal(classifyMotion(hist([...still.slice(0, 6), body(shift(base, 0, -.1))])), "Jumping");
// Side view walking: ankles swap front/back three times.
const step = d => body({ ...base, lAn: { x: .5 + d, y: .9 }, rAn: { x: .5 - d, y: .9 } });
assert.equal(classifyMotion(hist([step(.1), step(.1), step(-.1), step(-.1), step(.1), step(.1), step(-.1)])), "Walking");
// Small jitter is not walking.
assert.equal(classifyMotion(hist([step(.05), step(.04), step(.05), step(.04), step(.05), step(.04)])), null);

// Rep counters
const squats = makeRepCounter("Squat", "Standing");
["Standing", "Squat", "Bending", "Standing", "Standing", "Squat", "Standing"].forEach(squats);
assert.equal(squats("Standing"), 2);
const jumps = makeRepCounter("Jumping");
[null, "Jumping", "Jumping", null, "Walking", "Jumping"].forEach(jumps);
assert.equal(jumps(null), 2);

// Tracker: ids stay with the same person when order changes, and survive a brief disappearance.
const track = makeTracker();
const a = body(shift(base, -.3, 0)), b = body(shift(base, .3, 0));
const [p1, p2] = track([a, b], 0);
assert.deepEqual(track([b, a], 33).map(p => p.id), [p2.id, p1.id]);
assert.deepEqual(track([a], 500).map(p => p.id), [p1.id]);
assert.deepEqual(track([a, b], 900).map(p => p.id), [p1.id, p2.id]);
track([a], 2000); // b gone longer than ttl
assert.equal(track([a, b], 2033)[1].id, 3);
// Limbs
const limbs = limbStates(body(base));
assert.deepEqual(limbs.map(l => l.state), ["Extended", "Extended", "Extended", "Extended"]);
assert.equal(limbStates(body({ ...base, lWr: { x: .4, y: 0 } }))[0].state, "Raised");
assert.equal(limbStates(squat)[2].state, "Bent");
const hidden = body(base); hidden[L.rWr] = { ...hidden[L.rWr], visibility: 0.1 };
assert.equal(limbStates(hidden)[1].state, "Not visible");

// Face expression
assert.deepEqual(classifyExpression({}), { label: "Neutral", scores: { Smiling: 0, Threatening: 0, Surprised: 0, Sad: 0 }, eyes: "Open", mouth: "Closed" });
assert.equal(classifyExpression({ mouthSmileLeft: .8, mouthSmileRight: .8 }).label, "Smiling");
assert.equal(classifyExpression({ browDownLeft: .5, browDownRight: .5, noseSneerLeft: .3, noseSneerRight: .3 }).label, "Threatening");
assert.equal(classifyExpression({ browInnerUp: .7, eyeWideLeft: .6, eyeWideRight: .6, jawOpen: .5 }).label, "Surprised");
assert.equal(classifyExpression({ jawOpen: .5 }).label, "Neutral"); // talking isn't surprise
const wink = classifyExpression({ eyeBlinkLeft: .9, jawOpen: .5 });
assert.equal(wink.eyes, "Winking"); assert.equal(wink.mouth, "Open");
// Objects
const det = (name, score) => ({ categories: [{ categoryName: name, score }] });
assert.deepEqual(summarizeObjects([det("cup", .6), det("laptop", .9), det("cup", .8)]), [{ name: "cup", count: 2, best: .8 }, { name: "laptop", count: 1, best: .9 }]);
assert.deepEqual(summarizeObjects([]), []);
// People counter
assert.equal(iou({ x: 0, y: 0, w: 10, h: 10 }, { x: 0, y: 0, w: 10, h: 10 }), 1);
assert.equal(iou({ x: 0, y: 0, w: 10, h: 10 }, { x: 20, y: 0, w: 10, h: 10 }), 0);
const ppl = makeBoxTracker(), A = { x: 0, y: 0, w: 100, h: 200, score: .9 }, B = { x: 300, y: 0, w: 100, h: 200, score: .8 };
assert.deepEqual(ppl([A, B]), []); assert.deepEqual(ppl([A, B]), []); // not confirmed yet
assert.deepEqual(ppl([{ ...A, x: 10 }, B]).map(t => t.id), [1, 2]); // A moved a bit, same person
for (let i = 0; i < 5; i++) ppl([B]); // A hidden for 5 steps
assert.deepEqual(ppl([A, B]).map(t => t.id).sort(), [1, 2]); // A keeps its id
assert.deepEqual(ppl([{ x: 600, y: 0, w: 100, h: 200, score: .9 }]), []); // one-off detection isn't counted
for (let i = 0; i < 11; i++) ppl([]);
ppl([A]); ppl([A]); assert.deepEqual(ppl([A]).map(t => t.id), [3]); // gone too long: new person
assert.equal(ppl.total(), 3);
// YOLO decoding: one confident cup in a 1280x720 frame letterboxed to 640 (scale .5, 140px bars top and bottom)
const logits = new Float32Array(2 * 80).fill(-10), qboxes = new Float32Array([.5, .5, .25, .25, .1, .1, .1, .1]);
logits[COCO.indexOf("cup")] = 3; // query 0 → cup, sigmoid ≈ .95; query 1 stays below threshold
const [cup, ...rest] = decodeYolo(logits, qboxes, { scale: .5, padX: 0, padY: 140 });
assert.equal(rest.length, 0);
assert.equal(cup.categories[0].categoryName, "cup"); assert.ok(cup.categories[0].score > .9);
assert.deepEqual(cup.boundingBox, { originX: 480, originY: 200, width: 320, height: 320 });
console.log("all checks passed");
