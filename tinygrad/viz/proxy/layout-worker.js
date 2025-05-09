const { createCanvas } = require("canvas");
const dagre = require("dagre");

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

const { parentPort, workerData } = require('worker_threads');
parentPort.postMessage(calculateLayout(workerData));
