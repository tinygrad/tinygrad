const NODE_PADDING = 10;
const rectDims = (lw, lh) => ({ width:lw+NODE_PADDING*2, height:lh+NODE_PADDING*2, labelWidth:lw, labelHeight:lh });

const canvas = new OffscreenCanvas(0, 0);
const ctx = canvas.getContext("2d");

onmessage = (e) => {
  const { data, opts } = e.data;
  const g = new dagre.graphlib.Graph({ compound: true }).setDefaultEdgeLabel(function() { return {}; });
  (data.blocks != null ? layoutCfg : layoutUOp)(g, data, opts);
  postMessage(dagre.graphlib.json.write(g));
  self.close();
}

const layoutCfg = (g, { blocks, paths, pc_table, counters, colors }) => {
  const lineHeight = 16;
  g.setGraph({ rankdir:"TD", font:"monospace", lh:lineHeight });
  ctx.font = `350 ${lineHeight}px ${g.graph().font}`;
  // basic blocks render the assembly in nodes
  let maxColor = 0;
  for (const [lead, members] of Object.entries(blocks)) {
    let [width, height, label] = [0, 0, []];
    for (const m of members) {
      const text = pc_table[m][0];
      if (counters != null) {
        const num = counters[m]?.hit_count || 0;
        if (num > maxColor) maxColor = num;
        label.push([{st:text, color:num}]);
      } else { const [inst, ...operands] = text.split(" "); label.push([{st:inst+" ", color:"#7aa2f7"}, {st:operands.join(" "), color:"#9aa5ce"}]); }
      width = Math.max(width, ctx.measureText(text).width);
      height += lineHeight;
    }
    g.setNode(lead, { ...rectDims(width, height), label, id:lead, color:"#1a1b26" });
  }
  g.graph().colorDomain = [0, maxColor];
  // paths become edges between basic blocks
  for (const [lead, value] of Object.entries(paths)) {
    for (const [id, color] of Object.entries(value)) g.setEdge(lead, id, {label:{type:"port", text:""}, color:colors[color]});
  }
  dagre.layout(g);
}

const layoutUOp = (g, { graph, change, is_deps_graph, device_count, devices }, opts) => {
  const lineHeight = 14;
  // use top-to-bottom layout with tighter spacing for dependency graphs
  if (is_deps_graph) {
    g.setGraph({ rankdir: "TB", font:"sans-serif", lh:lineHeight, ranksep: 25, nodesep: 20, ranker: "tight-tree" });
  } else {
    g.setGraph({ rankdir: "LR", font:"sans-serif", lh:lineHeight });
  }
  ctx.font = `350 ${lineHeight}px ${g.graph().font}`;
  if (change?.length) g.setNode("overlay", {label:"", labelWidth:0, labelHeight:0, className:"overlay"});

  // create device cluster nodes for dependency graphs
  if (is_deps_graph && device_count) {
    for (let i = 0; i < device_count; i++) {
      const label = devices?.[i] || `Device ${i}`;
      g.setNode(`device_${i}`, {label, labelWidth:0, labelHeight:0, className:"device-cluster", clusterLabelPos: "top", color:"rgba(42, 45, 58, 0.3)"});
    }
  }

  for (const [k, {label, src, ref, color, tag, device_col }] of Object.entries(graph)) {
    // adjust node dims by label size (excluding escape codes) + add padding
    let [width, height] = [0, 0];
    for (line of label.replace(/\u001B\[(?:K|.*?m)/g, "").split("\n")) {
      width = Math.max(width, ctx.measureText(line).width);
      height += lineHeight;
    }
    g.setNode(k, {...rectDims(width, height), label, ref, id:k, color, tag, device_col});
    // add edges
    const edgeCounts = {};
    for (const [_, s] of src) edgeCounts[s] = (edgeCounts[s] || 0)+1;
    for (const [port, s] of src) g.setEdge(s, k, { label: edgeCounts[s] > 1 ? {type:"tag", text:edgeCounts[s]} : {type:"port", text:port}});
    if (change?.includes(parseInt(k))) g.setParent(k, "overlay");
    else if (is_deps_graph && device_col != null) g.setParent(k, `device_${device_col}`);
  }
  // optionally hide nodes from the layuot
  if (opts && !opts.showIndexing) {
    for (const n of g.nodes()) {
      const node = g.node(n);
      if (node.label.includes("dtypes.index")) g.removeNode(n);
    }
  }
  dagre.layout(g);
  // remove overlay node if it's empty
  if (!g.node("overlay")?.width) g.removeNode("overlay");
}
