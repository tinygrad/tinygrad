import os, threading, unittest
from contextlib import asynccontextmanager
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
if "BROWSER_TESTS" in os.environ:
  from playwright.async_api import async_playwright
else:
  async_playwright = None

class HTTPServerThread:
  def __init__(self, host="localhost", port=0, directory=None):
    if directory:
      def handler(*args, **kwargs): SimpleHTTPRequestHandler(*args, directory=directory, **kwargs)
    else: handler = SimpleHTTPRequestHandler
    self._server = TCPServer((host, port), handler)
    self.host, self.port = self._server.server_address
    self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
  def start(self):
    self._thread.start()
    print(f"HTTP server running at http://{self.host}:{self.port}/")
  def stop(self):
    self._server.shutdown()
    self._server.server_close()
    self._thread.join()

@unittest.skipUnless("BROWSER_TESTS" in os.environ, "browser tests need playwright dependency")
class TestBrowserModels(unittest.IsolatedAsyncioTestCase):
  @classmethod
  def setUpClass(cls):
    cls.http = HTTPServerThread(directory=os.getcwd())
    cls.http.start()
  @classmethod
  def tearDownClass(cls):
    cls.http.stop()

  @asynccontextmanager
  async def browser_page(self, extra_args=None):
    # Launches a Chromium browser + new page, yields the page. At end, shuts everything down
    args = ["--enable-unsafe-webgpu"]
    if extra_args:
        args = extra_args  # e.g. ["--enable-features=Vulkan", "--enable-unsafe-webgpu"]
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=False, args=args)
    page = await browser.new_page()
    try:
        yield page
    finally:
        await browser.close()
        await pw.stop()

  async def test_efficientnet(self):
    async with self.browser_page() as page:
      resp = await page.goto(f"http://{self.http.host}:{self.http.port}/examples/webgpu/efficientnet/index.html")
      self.assertIsNotNone(resp)
      self.assertEqual(resp.status, 200)
      await page.wait_for_function("() => document.querySelector('#result').textContent.trim() === 'ready'", timeout=30_000)
      await page.click("input[type=button]")
      await page.wait_for_function("() => document.querySelector('#result').textContent.trim() === 'hen'", timeout=60_000)

  async def test_yolov8(self):
    async with self.browser_page() as page:
      resp = await page.goto(f"http://{self.http.host}:{self.http.port}/examples/webgpu/yolov8/?VALIDATE=1")
      self.assertIsNotNone(resp)
      self.assertEqual(resp.status, 200)
      await page.wait_for_selector("#validate-output:not(:empty)", timeout=30_000)
      raw = await page.text_content("#validate-output")
      text = raw.replace("\r\n", "\n").strip()
      self.assertEqual(text, "label: bird")

  async def test_tinychat(self):
    async with self.browser_page(extra_args=["--enable-features=Vulkan", "--enable-unsafe-webgpu"]) as page:
      # VALIDATE=1 fixes the random seeds and counter to 0
      resp = await page.goto(f"http://{self.http.host}:{self.http.port}/examples/tinychat/tinychat-browser/index.html?VALIDATE=1")
      self.assertIsNotNone(resp)
      self.assertEqual(resp.status, 200)
      await page.wait_for_selector("textarea#input-form", timeout=30_000)
      await page.fill("textarea#input-form", "yo")
      await page.press("textarea#input-form", "Enter")
      await page.wait_for_function("() => document.querySelectorAll('.message-role-assistant').length > 0", timeout=10_000)
      await page.wait_for_selector("textarea#input-form:enabled", timeout=10_000)
      last = await page.inner_text(".message-role-assistant:last-child")
      # NOTE: relies on random seeds staying constant; TODO: set random seeds manually
      self.assertEqual(last.strip(), "What's up?")

if __name__ == "__main__":
  unittest.main()
