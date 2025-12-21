const NODE_PADDING = 10;
const LINE_HEIGHT = 14;
const canvas = new OffscreenCanvas(0, 0);
const ctx = canvas.getContext("2d");
ctx.font = `350 ${LINE_HEIGHT}px sans-serif`;

onmessage = (e) => {
  const { data, opts } = e.data;
  let g = new dagre.graphlib.Graph({ compound: true }).setDefaultEdgeLabel(function() { return {}; });
  g = data.graph != null ? layoutUOp(g, data, opts) : layoutCfg(g, data, opts);
  postMessage(dagre.graphlib.json.write(g));
  self.close();
}

const layoutCfg = ({ blocks, paths }) => {
  g.setGraph({ rankdir:"TD" });
  return g;
}

const layoutUOp = (g, { graph, changed }, opts) => {
  g.setGraph({ rankdir:"LR" });
  if (changed?.length) g.setNode("changed", {label:"", labelWidth:0, labelHeight:0, className:"overlay"});
  for (let [k, {label, src, ref, ...rest }] of Object.entries(graph)) {
    // adjust node dims by label size (excluding escape codes) + add padding
    let [width, height] = [0, 0];
    for (line of label.replace(/\u001B\[(?:K|.*?m)/g, "").split("\n")) {
      width = Math.max(width, ctx.measureText(line).width);
      height += LINE_HEIGHT;
    }
    g.setNode(k, {width:width+NODE_PADDING*2, height:height+NODE_PADDING*2, label, labelHeight:height, labelWidth:width, ref, id:k, ...rest});
    // add edges
    const edgeCounts = {}
    for (const [_, s] of src) edgeCounts[s] = (edgeCounts[s] || 0)+1;
    for (const [port, s] of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? {type:"tag", text:edgeCounts[s]} : {type:"port", text:port}});
    if (changed?.includes(parseInt(k))) g.setParent(k, "changed");
  }
  // optionally hide nodes from the layuot
  if (!opts.showIndexing) {
    for (const n of g.nodes()) {
      const node = g.node(n);
      if (node.label.includes("dtypes.index")) g.removeNode(n);
    }
  }
  dagre.layout(g);
  // remove changed overlay if it's empty
  if (!g.node("changed")?.width) g.removeNode("changed");
  return g;
}
