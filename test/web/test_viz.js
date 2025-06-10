const { spawn } = require("child_process");
const puppeteer = require("puppeteer");

async function main() {
  // ** start viz server
  const proc = spawn("python", ["-u", "-c", "from tinygrad import Tensor; Tensor.arange(4).realize()"], { env: { ...process.env, VIZ:"1" },
                      stdio: ["inherit", "pipe", "inherit"]});
  await new Promise(resolve => proc.stdout.on("data", r => {
    if (r.includes("ready")) resolve();
  }));

  // ** run browser tests
  let browser;
  try {
    browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    const res = await page.goto("http://localhost:8000", { waitUntil:"domcontentloaded" });
    if (res.status() !== 200) throw new Error("Failed to load page");
    const scheduleSelector = await page.waitForSelector("ul");
    scheduleSelector.click();
    await page.waitForSelector("rect");
    const nodes = await page.evaluate(() => document.querySelectorAll("#nodes > g").length);
    const edges = await page.evaluate(() => document.querySelectorAll("#edges > path").length);
    if (!nodes || !edges) {
      throw new Error("VIZ didn't render a graph")
    }
  } finally {
    // ** cleanups
    if (browser) await browser.close();
    proc.kill();
  }
}

main();
