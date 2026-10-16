// Pure logic for the license plate page (plates.html): detector output, plate reading, and the list of confirmed plates.

// Plate detector (open-image-models yolo-v9-t-384, end2end): rows of [batch, x1, y1, x2, y2, class, score] in letterboxed
// input pixels → boxes in video pixels.
export function decodePlateBoxes(rows, { scale, padX, padY, threshold = 0.4 }) {
  const out = [];
  for (let i = 0; i + 7 <= rows.length; i += 7) {
    const [, x1, y1, x2, y2, , score] = rows.slice(i, i + 7);
    if (score < threshold) continue;
    out.push({ x: (x1 - padX) / scale, y: (y1 - padY) / scale, w: (x2 - x1) / scale, h: (y2 - y1) / scale, score });
  }
  return out;
}

// Plate OCR (fast-plate-ocr cct-xs-v2-global): per plate, `slots` character positions with a probability for each
// alphabet symbol; "_" pads short plates. conf is the weakest character's probability, so one unsure letter lowers it.
export const OCR = { slots: 10, alphabet: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_", pad: "_" };
export function readPlates(probs, { slots, alphabet, pad } = OCR) {
  const K = alphabet.length, out = [];
  for (let p = 0; p * slots * K < probs.length; p++) {
    let text = "", conf = 1;
    for (let s = 0; s < slots; s++) {
      let bi = 0;
      const at = (p * slots + s) * K;
      for (let k = 1; k < K; k++) if (probs[at + k] > probs[at + bi]) bi = k;
      if (alphabet[bi] === pad) continue;
      text += alphabet[bi]; conf = Math.min(conf, probs[at + bi]);
    }
    out.push({ text, conf: text ? conf : 0 });
  }
  return out;
}

// The plate list. A reading counts only when it's sure enough and long enough, and a plate enters the list after
// `votes` identical readings, so a misread letter from one blurry frame never shows up. A plate seen again after
// `gap` ms counts as a new pass (the car came back).
export function makePlateList({ votes = 3, window = 10000, minConf = 0.6, minLen = 4, gap = 60000 } = {}) {
  const hits = new Map(), plates = new Map();
  return {
    // Returns the plate entry when this reading added or updated it, else null.
    read(text, conf, t) {
      if (conf < minConf || text.length < minLen) return null;
      const known = plates.get(text);
      if (known) {
        if (t - known.last > gap) known.passes++;
        known.last = t; known.conf = Math.max(known.conf, conf);
        return known;
      }
      const h = hits.get(text), n = h && t - h.t < window ? h.n + 1 : 1; // votes must come close together, not over hours
      hits.set(text, { n, t });
      if (n < votes) return null;
      hits.delete(text);
      const entry = { plate: text, first: t, last: t, passes: 1, conf };
      plates.set(text, entry);
      return entry;
    },
    list: () => [...plates.values()].sort((a, b) => b.last - a.last),
    load(entries) { for (const e of entries) plates.set(e.plate, { ...e }); },
    clear() { plates.clear(); hits.clear(); },
  };
}
