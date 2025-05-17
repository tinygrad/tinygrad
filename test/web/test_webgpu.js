const disableTimeout = process.argv.includes('--disableTimeout');
const puppeteer = require("puppeteer");
const { spawn } = require("child_process");
const child = spawn("python", ["-m", "http.server", "8000"], { shell: true, detached: true });

async function timeout(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

function cleanup(err) {
  console.log("cleaning up");
  try {
    process.kill(-child.pid);
  } catch (error) {
    console.error("Error while killing process group:", error);
  }
  if (err != null) {
    console.error(err);
    process.exit(1);
  }
  process.exit(0);
}

async function waitForText(selector, text) {
  let n = 0;
  let ready = false;
  while (n < 30) {
    const res = await selector.evaluate(el => el.textContent);
    console.log(`waiting for text ${text} got ${res}`);
    if (res == text) {
      ready = true;
      break;
    }
    await timeout(1000);
    n += 1;
  }
  return ready;
}

async function runTest() {
  const browser = await puppeteer.launch({
    headless: false,
    args: ["--enable-unsafe-webgpu", "--remote-debugging-port=9222"],
  });
  const page = await browser.newPage();

  page
    .on("console", (message) =>
      console.log(`message from console ${message.text()}`),
    )
    .on("pageerror", ({ message }) =>
      console.log(`error from page ${message}`),
    );

  const res = await page.goto("http://localhost:8000/examples/webgpu/efficientnet/index.html");
  if (res.status() !== 200) throw new Error("Failed to load page");

  const textSelector = await page.waitForSelector("#result");
  const buttonSelector = await page.waitForSelector("input[type=button]");
  const ready = await waitForText(textSelector, "ready");
  if (!ready) throw new Error("Failed to load page");

  await buttonSelector.evaluate((e) => e.click());
  const done = await waitForText(textSelector, "hen");
  if (!done) throw new Error("failed to get hen");

  cleanup(null);
}

runTest().catch((err) => cleanup(err));
