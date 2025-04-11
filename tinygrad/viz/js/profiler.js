const colors = ["7aa2f7", "ff9e64", "f7768e", "2ac3de", "7dcfff", "1abc9c", "9ece6a", "e0af68", "bb9af7", "9d7cd8", "ff007c"];

const formatTime = (ms) => {
  if (ms<=1e3) return `${ms}us`;
  if (ms<=1e6) return `${(ms*1e-3).toFixed(2)}ms`;
  return `${(ms*1e-6).toFixed(2)}s`;
}

async function main() {
  const { traceEvents } = await (await fetch("/get_profile")).json();
  const root = createChild("div.root", document.querySelector("body"));
  const list = createChild("div.list", root);
  const data = [];
  const nameColors = {}; // event names get a unique color
  const procNames = {};
  for (const e of traceEvents) {
    if (e.name === "process_name") {
      const proc = createChild(`div.proc-${e.pid}`, list);
      createChild("p.process-name", proc).textContent = e.args.name;
      procNames[e.pid] = e.args.name;
    }
    else if (e.name === "thread_name") {
      const thread = createChild(`div.thread-${e.pid}-${e.tid}`, `proc-${e.pid}`);
      createChild("p.thread-name", thread).textContent = e.args.name;
    }
    else if (e.ph === "X") {
      const thread = document.getElementById(`thread-${e.pid}-${e.tid}`);
      if (!(e.name in nameColors)) nameColors[e.name] = colors[data.length%(colors.length-1)];
      data.push({ ...e, y:rect(thread).y, color:`#${nameColors[e.name]}`, proc:procNames[e.pid] });
    }
  }
  // render graph
  const svg = d3.select(root).append("svg").attr("width", "100%");
  const { y, width } = rect(svg.node()); // global coordinates
  const render = svg.append("g").attr("transform", `translate(0, ${y})`);
  const timestamps = data.map(t => t.ts);
  const st = Math.min(...timestamps);
  const timeScale = d3.scaleLinear().domain([0, Math.max(...timestamps)-st]).range([y, width]);
  const timeAxis = render.append("g").call(d3.axisTop(timeScale).tickFormat(formatTime));
  list.style = `margin-top: ${rect(timeAxis.node()).bottom}px;`;
  // rescale time based coordinates to fit screen
  for (e of data) {
    e.st = e.ts-st;
    e.x = timeScale(e.st);
    e.width = timeScale(e.dur);
  }
  render.selectAll("rect").data(data).join("rect").attr("fill", d => d.color).attr("x", d => d.x).attr("y", d => d.y).attr("width", d => d.width)
    .attr("height", 20);
  render.call(d3.brush().on("end", (e) => {
    if (!e.selection) return renderTable({ data });
    const [[x0, y0], [x1, y1]] = e.selection;
    const newData = data.filter(d => d.x>=x0 && d.x<=x1 && d.y>=y0 && d.y<=y1);
    renderTable({ data: newData });
  }));
  createChild("div.table-root", root);
  renderTable({ data });
}

const rect = (e) => e.getBoundingClientRect();

const createChild = (es, p) => {
  const parts = es.split(".", 2);
  if (typeof p === "string") p = document.getElementById(p);
  const ret = p.appendChild(document.createElement(parts[0]));
  if (parts.length !== 1) ret.id = parts[1];
  return ret;
}

const columnNames = {"name":"Name", "st":"Start Time", "dur":"Duration", "proc":"Process"};
const tableState = {data:null, sortBy:null, asc:true};
function renderTable(newState) {
  const { data, sortBy, asc } = Object.assign(tableState, newState);
  const root = document.getElementById("table-root");
  root.innerHTML = "";
  const table = createChild("table", root);
  const thead = createChild("tr", createChild("thead", table));
  for (const [k,v] of Object.entries(columnNames)) {
    const th = createChild(`th.${k}`, thead);
    th.innerText = v;
    th.onclick = (e) => renderTable(k === sortBy ? { asc:!asc } : { sortBy:k, asc:true });
  }
  if (sortBy != null) {
    data.sort((a, b) => asc ? a[sortBy]-b[sortBy] : b[sortBy]-a[sortBy]); // inplace sort
    document.getElementById(sortBy).className = asc ? "sorted-asc" : "sorted-desc";
  }
  const tbody = createChild("tbody", table);
  for (const d of data) {
    const row = createChild("tr", tbody);
    for (const k of Object.keys(columnNames)) {
      let formatted = typeof d[k] === "string" ? d[k] : formatTime(d[k]);
      createChild("td", row).innerText = formatted;
    }
  }
}

main()
