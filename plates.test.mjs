import assert from "node:assert/strict";
import { decodePlateBoxes, readPlates, makePlateList, OCR } from "./plates.js";

// Detector rows: a sure plate and a weak one, in a 1280×720 frame letterboxed to 384 (scale .3, 54 px bars top and bottom).
const boxes = decodePlateBoxes([0, 100, 154, 160, 169, 0, .9, 0, 10, 60, 20, 70, 0, .1], { scale: .3, padX: 0, padY: 54 });
assert.equal(boxes.length, 1);
assert.deepEqual(Object.values(boxes[0]).map(v => Math.round(v * 10) / 10), [333.3, 333.3, 200, 50, .9]);

// OCR: "AB123CD" then padding; the weakest letter sets the confidence.
const probsFor = (text, weak = {}) => {
  const K = OCR.alphabet.length, p = new Float32Array(OCR.slots * K);
  for (let s = 0; s < OCR.slots; s++) {
    const ch = text[s] ?? OCR.pad, c = weak[s] ?? .95;
    p.fill((1 - c) / (K - 1), s * K, (s + 1) * K); p[s * K + OCR.alphabet.indexOf(ch)] = c;
  }
  return p;
};
const both = new Float32Array([...probsFor("AB123CD", { 2: .5 }), ...probsFor("")]);
const [plate, empty] = readPlates(both);
assert.equal(plate.text, "AB123CD"); assert.equal(plate.conf, .5);
assert.deepEqual(empty, { text: "", conf: 0 });

// Plate list: needs 3 close readings, ignores unsure or short ones, and counts a later return as a new pass.
const list = makePlateList();
assert.equal(list.read("AB123CD", .9, 0), null);
assert.equal(list.read("AB123CD", .3, 100), null); // unsure: not a vote
assert.equal(list.read("AB1", .9, 200), null); // too short
assert.equal(list.read("AB123CD", .9, 300), null);
assert.equal(list.read("AB123CD", .8, 400).passes, 1); // third vote: in the list
assert.equal(list.read("AB123CD", .9, 5000).passes, 1); // same pass
assert.equal(list.read("AB123CD", .9, 100000).passes, 2); // back after a minute
list.read("XY999ZZ", .9, 0); list.read("XY999ZZ", .9, 20000); list.read("XY999ZZ", .9, 40000);
assert.deepEqual(list.list().map(e => e.plate), ["AB123CD"]); // votes 20 s apart: a stray misread, not a plate
list.clear(); list.load([{ plate: "ZZ000AA", first: 1, last: 2, passes: 1, conf: .9 }]);
assert.equal(list.read("ZZ000AA", .9, 3).plate, "ZZ000AA"); // a saved plate is recognized at once
console.log("all plate checks passed");
