// in browser, max stack depth is hardcoded
// the DSP schedule graph is very deep, so you need to proxy the layout calculation through this extra server to VIZ it:
// npm install
// node --stack-size=65500 ./tinygrad/viz/proxy/index.js

const { Worker } = require('worker_threads');

function runLayout(graph) {
  return new Promise((resolve, reject) => {
    const worker = new Worker('./tinygrad/viz/proxy/layout-worker.js', { workerData: graph });
    worker.on('message', resolve);
    worker.on('error', reject);
    worker.on('exit', code => {
      if (code !== 0) reject(new Error(`Worker stopped with code ${code}`));
    });
  });
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
  req.on("end", async () => {
    try {
      const graph = JSON.parse(body);
      const layout = await runLayout(graph);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(layout));
    } catch (e) {
      console.log("ERR", e);
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: "invalid request body" }));
      return;
    }
  });
});

server.listen(PORT, () => {
  console.log(`dagre layout server started at http://localhost:${PORT}`);
});
