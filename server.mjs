// Zero-dependency static server: camera access needs http://localhost, and ES modules don't load from file://.
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const root = import.meta.dirname, types = { ".html": "text/html", ".js": "text/javascript" };
http.createServer(async (req, res) => {
  const path = normalize(join(root, req.url === "/" ? "index.html" : decodeURIComponent(req.url.split("?")[0])));
  if (!path.startsWith(root) || !types[extname(path)]) return res.writeHead(404).end(); // serve only our .html/.js
  try { res.writeHead(200, { "Content-Type": types[extname(path)] }).end(await readFile(path)); }
  catch { res.writeHead(404).end(); }
}).listen(8080, "127.0.0.1", () => console.log("Open http://localhost:8080"));
