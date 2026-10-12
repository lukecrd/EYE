import assert from "node:assert/strict";
import { angle, classifyPose, classifyMotion, motion, makeRepCounter, makeTracker, limbStates, classifyExpression, summarizeObjects, iou, makeBoxTracker, decodeYolo, COCO, boxAt, faceCrop, groupCaptures, handCrop, smoothScores, heldCandidates, faceSignature, sigDiff, makeHeadIds, handPose, assignHands, makeStateLog, toCsv, makeTraitAverager, motionStats, handStats, JUMP, L } from "./classify.js";

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
const up = body(shift(base, 0, -.1)); // whole body 0.33 torso higher
assert.equal(classifyMotion(hist([...still.slice(0, 6), up, up, up])), "Jumping");
assert.equal(classifyMotion(hist([...still.slice(0, 6), up])), null); // one noisy frame is not a jump
const noFeet = up.map((q, i) => i === L.lAn || i === L.rAn ? { ...q, visibility: .1 } : q);
assert.equal(classifyMotion(hist([...still.slice(0, 6), noFeet, noFeet, noFeet])), null); // feet out of frame (webcam at a desk)
const feetOnly = body({ ...base, lAn: { x: .45, y: .8 }, rAn: { x: .55, y: .8 } });
assert.equal(classifyMotion(hist([...still.slice(0, 6), feetOnly, feetOnly, feetOnly])), null); // feet up but hips still: not a jump
const away = body(shift(base, 0, -.1)).map(q => ({ x: q.x, y: .1 + (q.y - .1) * .7 })); // smaller and higher: walking away
assert.equal(classifyMotion(hist([...still.slice(0, 6), away, away, away])), null);
// Side view walking: ankles swap front/back three times while the body moves across the view.
const step = (d, dx = 0) => body(shift({ ...base, lAn: { x: .5 + d, y: .9 }, rAn: { x: .5 - d, y: .9 } }, dx, 0));
const swings = [.1, .1, -.1, -.1, .1, .1, -.1];
assert.equal(classifyMotion(hist(swings.map((d, i) => step(d, i * .02)))), "Walking");
assert.equal(classifyMotion(hist(swings.map(d => step(d)))), null); // legs swinging, hips in place: seated, not walking
const legsHidden = f => f.map((q, i) => [L.lKn, L.rKn, L.lAn, L.rAn].includes(i) ? { ...q, visibility: .2 } : q);
assert.equal(classifyMotion(hist(swings.map((d, i) => legsHidden(step(d, i * .02))))), null); // legs out of frame (desk): guessed ankles don't count
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
// Face catalog helpers
const big = { id: 1, box: { x: 0, y: 0, w: 500, h: 500 } }, small = { id: 2, box: { x: 100, y: 100, w: 100, h: 200 } };
assert.equal(boxAt([big, small], 150, 150).id, 2); // inside both: the tighter box
assert.equal(boxAt([big, small], 400, 400).id, 1);
assert.equal(boxAt([big, small], 600, 600), null);
assert.deepEqual(Object.values(faceCrop([{ x: .4, y: .2 }, { x: .6, y: .5 }], 1000, 500, 1)).map(Math.round), [400, 75, 200, 500, 175]); // sx, sy, size, cx, cy
assert.deepEqual(groupCaptures([{ id: 1, t: 1 }, { id: 2, t: 2 }, { id: 1, t: 3 }]).map(g => [g.id, g.items.map(c => c.t)]), [[1, [3, 1]], [2, [2]]]);
// Held objects
const arm = body({ ...base, lEl: { x: .3, y: .4 }, lWr: { x: .3, y: .5 } }); arm[19] = { x: .3, y: .55 };
const hc = handCrop(arm, "left", 1000, 1000);
assert.deepEqual([hc.sx + hc.size / 2, hc.sy + hc.size / 2].map(Math.round), [300, 540]); // centered past the wrist toward the knuckles
assert.equal(Math.round(hc.size), 300); // 6 hand lengths (50px) beats 1.65 forearms (100px)
assert.ok(hc.visible);
let sm = { scores: {} };
for (let i = 0; i < 6; i++) sm = smoothScores(sm.scores, [{ categoryName: "ballpoint", score: .5 }, { categoryName: i % 2 ? "nail" : "flute", score: .3 }]);
assert.equal(sm.label, "ballpoint"); assert.ok(sm.score > .4 && sm.score < .5); // steady label wins, flickering ones stay low
for (let i = 0; i < 20; i++) sm = smoothScores(sm.scores, []);
assert.equal(sm.label, null); // fades out once nothing is seen
const cat = (categoryName, index) => ({ categoryName, index, score: .3 });
assert.deepEqual(heldCandidates([cat("tench", 0), cat("ballpoint", 418), cat("breastplate", 461), cat("orange", 950), cat("alp", 970), cat("toilet tissue", 999)]).map(c => c.categoryName),
  ["ballpoint", "orange", "toilet tissue"]); // no animals, clothing or scenery
// Head re-identification
const head = (k = 1) => Array.from({ length: 478 }, (_, i) => ({ x: .5 + Math.sin(i) * .05 * (i % 7 === 3 ? k : 1), y: .4 + Math.cos(i * 1.3) * .06, z: Math.sin(i * .7) * .02 }));
const sa = faceSignature(head(), 1000, 1000), far = faceSignature(head().map(q => ({ x: .5 + (q.x - .5) / 2, y: .4 + (q.y - .4) / 2, z: q.z / 2 })), 1000, 1000);
assert.ok(sigDiff(sa, far) < 1e-9); // same head at half the size: same signature
const sb = faceSignature(head(1.6), 1000, 1000);
assert.ok(sigDiff(sa, sb) > .06); // different proportions: a different head
const ids = makeHeadIds({ samples: 2 });
ids.observe(1, sa, new Set()); assert.equal(ids.id(1), 1); ids.observe(1, sa, new Set());
ids.observe(2, sb, new Set([1])); ids.observe(2, sb, new Set([1])); assert.equal(ids.id(2), 2); // new person
ids.observe(5, sa, new Set([2])); ids.observe(5, sa, new Set([2])); assert.equal(ids.id(5), 1); // person 1 walks back in as box 5
assert.equal(ids.merged(), 1);
ids.observe(6, sa, new Set([1])); ids.observe(6, sa, new Set([1])); assert.equal(ids.id(6), 6); // look-alike while person 1 is in view: not merged
assert.equal(ids.match(sb), 2); // head alone, no body box
// Hands: world landmarks in meters, wrist at the origin, fingers pointing up (-y), palm facing the camera (+z).
const hand = finger => {
  const w = Array.from({ length: 21 }, () => ({ x: 0, y: 0, z: 0 }));
  [-.03, -.01, .01, .03].forEach((x, f) => { const [m, p, d, t] = [5, 6, 7, 8].map(i => i + 4 * f); Object.assign(w, finger(x, m, p, d, t)); });
  w[1] = { x: -.03, y: -.02, z: 0 }; w[2] = { x: -.05, y: -.04, z: 0 }; w[3] = { x: -.06, y: -.06, z: 0 }; w[4] = { x: -.07, y: -.08, z: 0 }; // thumb out to the side
  return w;
};
const openHand = hand((x, m, p, d, t) => ({ [m]: { x, y: -.08, z: 0 }, [p]: { x, y: -.11, z: 0 }, [d]: { x, y: -.13, z: 0 }, [t]: { x, y: -.15, z: 0 } }));
const fist = hand((x, m, p, d, t) => ({ [m]: { x, y: -.08, z: 0 }, [p]: { x, y: -.1, z: .02 }, [d]: { x, y: -.08, z: .03 }, [t]: { x, y: -.05, z: .02 } }));
const grip = hand((x, m, p, d, t) => ({ [m]: { x, y: -.08, z: 0 }, [p]: { x, y: -.1, z: .03 }, [d]: { x, y: -.09, z: .06 }, [t]: { x, y: -.07, z: .07 } })); // around a ~7 cm bottle
assert.equal(handPose(openHand), "Open hand");
assert.equal(handPose(fist), "Fist");
assert.equal(handPose(grip), "Holding"); // curled but the fingertips stay away from the palm
assert.equal(handPose(fist, { holding: true }), "Holding"); // tight grip on something thin the classifier sees
assert.equal(handPose(fist, { gesture: "Thumb_Up" }), "Thumbs up");
assert.equal(handPose(openHand, { gesture: "Closed_Fist" }), "Open hand"); // the recognizer's fist is decided by the geometry
const pinch = openHand.map((q, i) => i === 4 ? { ...openHand[8], x: openHand[8].x - .01 } : q);
assert.equal(handPose(pinch, { holding: true }), "Holding"); // pen between thumb and index
assert.equal(handPose(openHand, { holding: true }), "Open hand"); // object in front of an open hand is not held
const person = body(base);
assert.deepEqual(assignHands([[{ x: .62, y: .5 }], [{ x: .38, y: .47 }], [{ x: .9, y: .9 }]], [person]), [{ pose: 0, side: "right" }, { pose: 0, side: "left" }, null]);
// Event log
const ev = makeStateLog({ hold: 1000, gone: 3000 });
assert.deepEqual(ev({ "person:1": "in" }, 0), []);
assert.deepEqual(ev({ "person:1": "in" }, 1000), [{ key: "person:1", from: null, to: "in" }]); // held 1 s: entered
assert.deepEqual(ev({ "person:1": "in", "hand:1:left": "Fist" }, 1100), []);
assert.deepEqual(ev({ "person:1": "in", "hand:1:left": "Open hand" }, 1200), []); // flicker: Fist never held, never reported
assert.deepEqual(ev({ "person:1": "in", "hand:1:left": "Open hand" }, 2200), [{ key: "hand:1:left", from: null, to: "Open hand" }]);
assert.deepEqual(ev({ "person:1": "in", "person:9": "in" }, 2500), []); // id 9 shows briefly (before re-identification)…
assert.deepEqual(ev({ "person:1": "in" }, 4000), []); // …and is gone before it's held: never logged
assert.deepEqual(ev({}, 5300), [{ key: "hand:1:left", from: "Open hand", to: null }]); // hand unseen 3 s
assert.deepEqual(ev({}, 7000), [{ key: "person:1", from: "in", to: null }]); // left
assert.equal(toCsv([["time", "event"], ["12:00", `said "hi", left`]]), `"time","event"\r\n"12:00","said ""hi"", left"`); // quotes doubled, commas kept inside the field
// Per-person traits average out
const avg = makeTraitAverager();
const read = (age, gender, genderScore, happy) => ({ age, gender, genderScore, scores: { happy, neutral: 1 - happy } });
avg(1, read(30, "male", .9, .9));
let tr = avg(1, read(50, "female", .6, .9)); // one odd reading
assert.equal(tr.age, 40); assert.equal(tr.gender, "male"); assert.equal(tr.n, 2); // P(male) = (.9 + .4) / 2
for (let i = 0; i < 10; i++) tr = avg(1, read(34, "male", .9, .1));
assert.ok(tr.age > 33 && tr.age < 37); assert.equal(tr.emotion, "neutral"); // emotion follows, not stuck on the first one
assert.equal(avg(2, read(20, "female", .8, .1)).age, 20); // other people are separate
// Debug measures agree with the decisions
const ms = motionStats(hist([...still.slice(0, 6), up, up, up]));
assert.ok(ms.jump.feet > JUMP.rise && ms.jump.hips > JUMP.rise && ms.jump.seen); assert.equal(motionStats(hist(still.slice(0, 3))), null);
const hs = handStats(fist); assert.equal(hs.curled, 4); assert.ok(hs.gap < .045);
console.log("all checks passed");
