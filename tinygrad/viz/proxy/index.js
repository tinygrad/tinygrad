// in browser, max stack depth is hardcoded
// the DSP schedule graph is very deep, so you need to proxy the layout calculation through this extra server to VIZ it:
// npm install
// node --stack-size=65500 index.js

const { createCanvas } = require("canvas");
const dagre = require("dagre");

// ** copied from worker.js

const NODE_PADDING = 10;
const LINE_HEIGHT = 14;
const canvas = createCanvas(0, 0);
const ctx = canvas.getContext("2d");
ctx.font = `${LINE_HEIGHT}px sans-serif`;

function getTextDims(text) {
  let [maxWidth, height] = [0, 0];
  for (line of text.split("\n")) {
    const { width } = ctx.measureText(line);
    if (width > maxWidth) maxWidth = width;
    height += LINE_HEIGHT;
  }
  return [maxWidth, height];
}

function calculateLayout(graph) {
  const g = new dagre.graphlib.Graph({ compound: true });
  g.setGraph({ rankdir: "LR" }).setDefaultEdgeLabel(function() { return {}; });
  for (const [k, {label, src, color}] of Object.entries(graph)) {
    // adjust node dims by label size + add padding
    const [labelWidth, labelHeight] = getTextDims(label);
    g.setNode(k, {label, color, width:labelWidth+NODE_PADDING*2, height:labelHeight+NODE_PADDING*2, padding:NODE_PADDING});
    const edgeCounts = {}
    for (const s of src) {
      edgeCounts[s] = (edgeCounts[s] || 0)+1;
    }
    for (const s of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? edgeCounts[s] : null });
  }
  dagre.layout(g);
  return dagre.graphlib.json.write(g);
}

// ** HTTP stuff

const http = require("http");
const fs = require("fs");
const path = require("path");

const PORT = 8080;
const DATA_DIR = path.join(__dirname);

const server = http.createServer((req, res) => {
  res.setHeader("Access-Control-Allow-Origin", '*');
  if (req.method === "OPTIONS") {
    res.writeHead(204, { "Access-Control-Allow-Methods": "OPTIONS, POST", "Access-Control-Allow-Headers": "Content-Type" });
    res.end();
    return;
  }
  console.log(`${new Date()} proxy request`)
  let body = "";
  req.on("data", chunk => body += chunk);
  req.on("end", () => {
    try {
      const graph = JSON.parse(body);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(calculateLayout(graph)));
    } catch (e) {
      res.writeHead(400);
      res.end("invalid request body");
      return;
    }
  });
});

server.listen(PORT, () => {
  console.log(`dagre layout server started at http://localhost:${PORT}`);
});
