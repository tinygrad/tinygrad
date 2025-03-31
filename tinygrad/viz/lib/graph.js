function intersectRect(r1, r2) {
  const dx = r2.x-r1.x;
  const dy = r2.y-r1.y;
  if (dx === 0 && dy === 0) throw new Error("Invalid node coordinates, rects must not overlap");
  const scaleX = dx !== 0 ? (r1.width/2)/Math.abs(dx) : Infinity;
  const scaleY = dy !== 0 ? (r1.height/2)/Math.abs(dy) : Infinity;
  const scale = Math.min(scaleX, scaleY);
  return {x:r1.x+dx*scale, y:r1.y+dy*scale};
}

let [workerUrl, worker, timeout] = [null, null, null];
window.renderGraph = async function(graph, additions, name) {
  if (name === "View Memory Graph") {
    return renderMemoryGraph(graph);
  }
  d3.select("#bars").html("");

  // ** start calculating the new layout (non-blocking)
  if (worker == null) {
    const resp = await Promise.all(["/assets/dagrejs.github.io/project/dagre/latest/dagre.min.js","/lib/worker.js"].map(u => fetch(u)));
    workerUrl = URL.createObjectURL(new Blob([(await Promise.all(resp.map((r) => r.text()))).join("\n")], { type: "application/javascript" }));
    worker = new Worker(workerUrl);
  } else {
    worker.terminate();
    worker = new Worker(workerUrl);
  }
  if (timeout != null) clearTimeout(timeout);
  const progressMessage = document.querySelector(".progress-message");
  timeout = setTimeout(() => {
    progressMessage.style.display = "block";
  }, 2000);
  worker.postMessage({graph, additions});

  worker.onmessage = (e) => {
    progressMessage.style.display = "none";
    clearTimeout(timeout);
    const g = dagre.graphlib.json.read(e.data);
    // ** draw nodes
    const nodeRender = d3.select("#nodes");
    const nodes = nodeRender.selectAll("g").data(g.nodes().map(id => g.node(id)), d => d).join("g")
      .attr("transform", d => `translate(${d.x},${d.y})`);
    nodes.selectAll("rect").data(d => [d]).join("rect").attr("width", d => d.width).attr("height", d => d.height).attr("fill", d => d.color)
      .attr("x", d => -d.width/2).attr("y", d => -d.height/2).attr("style", d => d.style);
    // +labels
    nodes.selectAll("g.label").data(d => [d]).join("g").attr("class", "label").attr("transform", d => {
      const x = (d.width-d.padding*2)/2;
      const y = (d.height-d.padding*2)/2;
      return `translate(-${x}, -${y})`;
     }).selectAll("text").data(d => [d.label.split("\n")]).join("text").selectAll("tspan").data(d => d).join("tspan").text(d => d).attr("x", "1")
       .attr("dy", 14).attr("xml:space", "preserve");

    // ** draw edges
    const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
    const edgeRender = d3.select("#edges");
    edgeRender.selectAll("path.edgePath").data(g.edges()).join("path").attr("class", "edgePath").attr("d", (e) => {
      const edge = g.edge(e);
      const points = edge.points.slice(1, edge.points.length-1);
      points.unshift(intersectRect(g.node(e.v), points[0]));
      points.push(intersectRect(g.node(e.w), points[points.length-1]));
      return line(points);
    }).attr("marker-end", "url(#arrowhead)");
  };
}


DTYPE_SIZE = {"bool": 1, "char": 1, "uchar": 1, "short": 2, "ushort": 2, "int": 4, "uint": 4,
              "long": 8, "ulong": 8, "half": 2, "bfloat": 2, "float": 4, "double": 8}
function getBuffer(e) {
  const [_, size, dtype, device, num] = e.label.split("\n");
  return {nbytes:size*DTYPE_SIZE[dtype.split("dtypes.")[1]], dtype, device:device.split(" ")[1], num:parseInt(num.split(" ")[1])};
}
function pluralize(num, name, alt=null) {
  return num === 1 ? `${num} ${name}` : `${num} ${alt ?? name+'s'}`
}

function renderMemoryGraph(graph) {
  // ** construct alloc/free traces
  // we can map reads/writes from the kernel graph
  const actions = [];
  const children = new Map(); // {buffer: [...assign]}
  for (const [k,v] of Object.entries(graph)) {
    if (!v.label.startsWith("ASSIGN")) continue;
    actions.push({ op: "write", buffer: v.src[0] });
    for (const ks of graph[v.src[1]].src) {
      const node = graph[ks];
      const s = node.label.startsWith("ASSIGN") ? node.src[0] : ks;
      if (!children.has(s)) children.set(s, []);
      children.get(s).push(v);
      if (s !== v.src[0]) actions.push({ op: "read", buffer: s });
    }
  }
  const prealloc = new Set();
  const traces = [];
  for (const a of actions) {
    // a buffer is allocated immediately before the first write
    // TODO: we don't know the buffer is preallocated if there's only an assign in the graph
    if (a.op === "write") {
      traces.push({ type: "alloc", buffer: a.buffer });
    }
    else {
      if (traces.find(t => t.buffer === a.buffer && t.type === "alloc") == null) {
        prealloc.add(a.buffer);
      }
      else if (a === actions.findLast(({ buffer }) => buffer === a.buffer)) {
        traces.push({type: "free", buffer: a.buffer });
      }
    }
  }
  // ** get coordinates and layout for each buffer
  const ret = {};
  let timestep = 0; // x
  let memUsed = 0; // y
  for (const id of prealloc) {
    const buf = getBuffer(graph[id]);
    ret[id] = { x: [timestep], y: [memUsed], buf, id };
    memUsed += buf.nbytes;
  }
  let peak = memUsed;
  const liveBufs = [...prealloc];
  for (const t of traces) {
    const buf = getBuffer(graph[t.buffer]);
    const idx = liveBufs.findLastIndex(b => t.buffer === b);
    // alloc
    if (idx === -1) {
      liveBufs.push(t.buffer);
      ret[t.buffer] = { x: [timestep], y: [memUsed], buf, id: t.buffer };
      memUsed += buf.nbytes;
      peak = Math.max(memUsed, peak);
      timestep += 1;
    } // free
    else {
      memUsed -= buf.nbytes;
      timestep += 1;
      const removed = ret[liveBufs.splice(idx, 1)[0]];
      removed.x.push(timestep);
      removed.y.push(removed.y.at(-1));
      if (idx < liveBufs.length) {
        for (let j=idx; j<liveBufs.length; j++) {
          const b = ret[liveBufs[j]];
          b.x.push(timestep, timestep);
          b.y.push(b.y.at(-1), b.y.at(-1)-buf.nbytes);
        }
      }
    }
  }
  for (const id of liveBufs) {
    const b = ret[id];
    b.x.push(timestep);
    b.y.push(b.y.at(-1));
  }
  // ** render traces
  const render = d3.select("#bars");
  const yscale = d3.scaleLinear().domain([0, peak]).range([576, 0]);
  const xscale = d3.scaleLinear().domain([0, timestep]).range([0, 1024]);
  const xaxis = d3.axisBottom(xscale);
  const axesGroup = render.append("g").attr("id", "axes");
  const nbytes_format = (d) => d3.format(".3~s")(d)+"B";
  axesGroup.append("g").call(d3.axisLeft(yscale).tickFormat(nbytes_format));
  axesGroup.append("g").attr("transform", `translate(0, ${yscale.range()[0]})`).call(d3.axisBottom(xscale).tickFormat(() => ""));
  const polygonGroup = render.append("g").attr("id", "polygons");
  const colors = ["7aa2f7", "ff9e64", "f7768e", "2ac3de", "7dcfff", "1abc9c", "9ece6a", "e0af68", "bb9af7", "9d7cd8", "ff007c"];
  const polygons = polygonGroup.selectAll("polygon").data(Object.values(ret)).join("polygon").attr("points", (d) => {
    const xs = d.x.map(t => xscale(t));
    const y1 = d.y.map(t => yscale(t));
    const y2 = d.y.map(t => yscale(t+d.buf.nbytes));
    const p0 = xs.map((x, i) => `${x},${y1[i]}`);
    const p1 = xs.map((x, i) => `${x},${y2[i]}`).reverse();
    return `${p0.join(' ')} ${p1.join(' ')}`;
  }).attr("fill", d => `#${colors[d.buf.num % colors.length]}`).on("mouseover", (e, { id, buf, x }) => {
    d3.select(e.currentTarget).attr("stroke", "rgba(26, 27, 38, 0.8)").attr("stroke-width", 0.8);
    const metadata = document.querySelector(".container.metadata");
    document.getElementById("current-buf")?.remove();
    const { num, dtype, nbytes, ...rest } = buf;
    let label = `<BUFFER n${num} ${dtype} ${nbytes_format(nbytes)}>\nalive for ${pluralize(x[x.length-1]-x[0], 'timestep')}`;
    label += '\n'+Object.entries(rest).map(([k, v]) => `${k}=${v}`).join('\n');
    const buf_children = children.get(id);
    if (buf_children) {
      label += `\n${pluralize(buf_children.length, 'child', 'children')}\n`;
      label += buf_children.map((c,i) => `[${i+1}] `+graph[c.src[1]].label.split("\n")[1]).join("\n");
    }
    metadata.appendChild(Object.assign(document.createElement("pre"), { innerText: label, id: "current-buf", className: "wrap" }));
  }).on("mouseout", (e, _) => {
    d3.select(e.currentTarget).attr("stroke", null).attr("stroke-width", null);
    document.getElementById("current-buf")?.remove()
  });
  // TODO: add the toposort graph here
  document.querySelector(".progress-message").style.display = "none";
  d3.select("#nodes").html("");
  d3.select("#edges").html("");
}
