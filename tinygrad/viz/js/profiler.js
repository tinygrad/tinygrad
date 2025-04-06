let profiles = null;
const colors = ["7aa2f7", "ff9e64", "f7768e", "2ac3de", "7dcfff", "1abc9c", "9ece6a", "e0af68", "bb9af7", "9d7cd8", "ff007c"];
async function main() {
  // fetch
  if (profiles == null) {
    procMap = {};
    threadMap = {};
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
  // render
  const HEIGHT = 16; // TODO: make this meaningful
  const PADDING = 20;
  const TICK_SIZE = 6;
  const root = document.querySelector("body").appendChild(document.createElement("div"));
  root.style = `display:flex; width:100%; height: 100%; padding: ${PADDING}px;`
  const list = root.appendChild(document.createElement("div"));
  list.style = `margin-top: ${PADDING+TICK_SIZE}px;`
  const data = [];
  const nameColors = {};
  let maxTimestamp = null;
  let minTimestamp = null;
  for (const [pid, threads] of Object.entries(profiles)) {
    if (Object.values(threads).every((t) => t.length === 0)) continue;
    const proc = list.appendChild(document.createElement("div"));
    const procName = proc.appendChild(document.createElement("p"));
    procName.textContent = procMap[pid].args.name;
    procName.style = "background: #0f1018; padding: 4px; border-radius: 2px;";
    for (const [tid, events] of Object.entries(threads)) {
      thread = proc.appendChild(document.createElement("div"));
      thread.textContent = threadMap[pid][tid].args.name;
      thread.style = "padding: 4px 4px;"
      const y = rect(thread).y-HEIGHT;
      for (const [i,e] of events.entries()) {
        if (!(e.name in nameColors)) nameColors[e.name] = colors[i%colors.length];
        data.push({ ts:e.ts, width:e.dur, y, height:HEIGHT, name:e.name, color:`#${nameColors[e.name]}` });
      }
      minTimestamp = minTimestamp == null ? events[0].ts : Math.min(events[0].ts, minTimestamp);
      maxTimestamp = maxTimestamp == null ? events[events.length-1].ts : Math.max(events[events.length-1].ts, maxTimestamp);
    }
  }
  for (e of data) {
    e.x = e.ts-minTimestamp;
  }
  // graph
  const svg = d3.select(root).append("svg").attr("width", "100%").attr("height", "100%");
  const render = svg.append("g");
  const x = d3.scaleLinear().domain([0, maxTimestamp-minTimestamp]).range([PADDING, rect(root).width-rect(list).width-PADDING]);;
  const time = render.append("g").call(d3.axisTop(x).tickFormat(formatTime).tickSize(TICK_SIZE))
    .attr("transform", `translate(0, ${PADDING+TICK_SIZE})`);
  render.selectAll("rect").data(data).join("rect").attr("fill", d => d.color).attr("x", d => x(d.x)).attr("y", d => d.y)
    .attr("width", d => x(d.width)).attr("height", d => d.height);
}

const rect = (e) => e.getBoundingClientRect();

const formatTime = (ms) => {
  if (ms<1e2) return `${Math.round(ms,2)}us`;
  if (ms<1e6) return `${Math.round(ms*1e-3,2)}ms`;
  return `${Math.round(ms*1e-6,2)}s`;
}

main();
