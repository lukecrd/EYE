// Zero-dependency static server: camera access needs http://localhost, and ES modules don't load from file://.
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const root = import.meta.dirname, types = { ".html": "text/html", ".js": "text/javascript", ".onnx": "application/octet-stream" };
http.createServer(async (req, res) => {
  const path = normalize(join(root, req.url === "/" ? "index.html" : decodeURIComponent(req.url.split("?")[0])));
  if (!path.startsWith(root) || !types[extname(path)]) return res.writeHead(404).end(); // serve only our .html/.js/.onnx
  // The plate page is cross-origin isolated so its models run on several CPU cores (same headers as vercel.json).
  const isolate = path.endsWith("plates.html") ? { "Cross-Origin-Opener-Policy": "same-origin", "Cross-Origin-Embedder-Policy": "credentialless" } : {};
  try { res.writeHead(200, { "Content-Type": types[extname(path)], ...isolate }).end(await readFile(path)); }
  catch { res.writeHead(404).end(); }
}).listen(process.env.PORT || 8080, "127.0.0.1", function () { console.log(`Open http://localhost:${this.address().port}`); });
