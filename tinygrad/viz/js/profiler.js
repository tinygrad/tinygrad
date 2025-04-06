let profiles = null;
let [procMap, threadMap] = [{}, {}];
const colors = ["7aa2f7", "ff9e64", "f7768e", "2ac3de", "7dcfff", "1abc9c", "9ece6a", "e0af68", "bb9af7", "9d7cd8", "ff007c"];
const data = [];

async function main() {
  // ** fetch + processing
  if (profiles == null) {
    profiles = {};
    const { traceEvents } = await (await fetch("/get_profile")).json();
    for (const t of traceEvents) {
      if (t.ph === "M" && t.name === "process_name") {
        procMap[t.pid] = t;
        threadMap[t.pid] = {};
        profiles[t.pid] = {};
      } else if (t.ph === "M" && t.name === "thread_name") {
        threadMap[t.pid][t.tid] = t;
        profiles[t.pid][t.tid] = [];
      } else if (t.ph === "X") {
        profiles[t.pid][t.tid].push(t);
      }
    }
  }
  // ** render
  const PADDING = 20;
  const PADDING_SM = 4;
  const TICK_SIZE = 6;
  const HEIGHT = 16; // TODO: make this meaningful
  const root = createChild("div", document.querySelector("body"));
  root.style = `display:flex; width:100%; height: 100%; padding: ${PADDING}px;`
  // PID/threads list
  const list = createChild("div", root);
  list.style = `margin-top: ${PADDING+TICK_SIZE}px;`
  const nameColors = {};
  let maxTimestamp = null;
  let minTimestamp = null;
  for (const [pid, threads] of Object.entries(profiles)) {
    if (Object.values(threads).every((t) => t.length === 0)) continue;
    const proc = createChild("div", list);
    const procName = proc.appendChild(document.createElement("p"));
    procName.textContent = procMap[pid].args.name;
    procName.style = `background: #0f1018; padding: ${PADDING_SM}px; border-radius: 2px;`;
    for (const [tid, events] of Object.entries(threads)) {
      thread = proc.appendChild(document.createElement("div"));
      thread.textContent = threadMap[pid][tid].args.name;
      thread.style = `padding: ${PADDING_SM}px;`
      const y = rect(thread).y-HEIGHT;
      for (const [i,e] of events.entries()) {
        if (!(e.name in nameColors)) nameColors[e.name] = colors[i%colors.length];
        data.push({ ...e, y, height:HEIGHT, color:`#${nameColors[e.name]}` });
      }
      const lastEvent = events[events.length-1];
      minTimestamp = minTimestamp == null ? events[0].ts : Math.min(events[0].ts, minTimestamp);
      maxTimestamp = maxTimestamp == null ? lastEvent.ts+lastEvent.dur : Math.max(lastEvent.ts+lastEvent.dur, maxTimestamp);
    }
  }
  // timeline graph
  const svg = d3.select(root).append("svg").attr("width", "100%");
  const render = svg.append("g").attr("id", "render");
  const timeScale = d3.scaleLinear().domain([0, maxTimestamp-minTimestamp]).range([PADDING, rect(root).width-rect(list).width-PADDING*2-8]);
  const timeAxis = render.append("g").call(d3.axisTop(timeScale).tickFormat(formatTime).tickSize(TICK_SIZE))
    .attr("transform", `translate(0, ${PADDING+TICK_SIZE})`);
  // rescale time-based coordinates
  for (e of data) {
    e.rts = e.ts-minTimestamp;
    e.x = timeScale(e.rts);
    e.width = timeScale(e.dur);
  }
  render.selectAll("rect").data(data).join("rect").attr("fill", d => d.color).attr("x", d => d.x).attr("y", d => d.y)
    .attr("width", d => d.width).attr("height", d => d.height);
  // info table
  const info = createChild("div", root);
  info.id = "table-root";
  const { width, height } = rect(render.node());
  const INFO_HEIGHT = rect(root).height-height-PADDING*2;
  info.style = `position: absolute; width: 100%; height: ${INFO_HEIGHT}px; background: #0f1018; bottom: 0; left: 0; padding: ${PADDING}px;`;
  renderTable(data);
  render.call(d3.brush().extent([[0, 0], [width+PADDING, height+PADDING]]).on("end", (e) => {
    if (!e.selection) renderTable(data);
    const [[x0, y0], [x1, y1]] = e.selection;
    renderTable(data.filter(d => (d.x+d.width)>=x0 && d.x<=x1 && (d.y+d.height)>=y0 && d.y<=y1))
  }));
}

const formatTime = (ms) => {
  if (ms<1e2) return `${Math.round(ms,2)}us`;
  if (ms<1e6) return `${Math.round(ms*1e-3,2)}ms`;
  return `${Math.round(ms*1e-6,2)}s`;
}

const rect = (e) => e.getBoundingClientRect();

const createChild = (es, p) => p.appendChild(document.createElement(es));

const columns = {"Name":"name", "Start Time":"rts", "Duration":"dur", "Process":"pid"};
let selectedData = null;
function renderTable(newData) {
  selectedData = newData.sort((a, b) => b.dur-a.dur);
  const root = document.getElementById("table-root");
  root.innerHTML = "";
  createChild("p", root).innerText = `${Intl.NumberFormat('en-US').format(selectedData.length)} traces`
  const table = createChild("table", root);
  const thead = createChild("tr", createChild("thead", table));
  const tbody = createChild("tbody", table);
  for (const k of Object.keys(columns)) {
    const th = createChild("th", thead);
    th.innerText = k;
  }
  for (const d of selectedData) {
    const tr = createChild("tr", tbody);
    createChild("td", tr).innerText = d.name;
    createChild("td", tr).innerText = d.rts;
    createChild("td", tr).innerText = formatTime(d.dur);
    createChild("td", tr).innerText = procMap[d.pid].args.name;
  }
}

main();
