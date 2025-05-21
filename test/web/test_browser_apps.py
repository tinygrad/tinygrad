import os, threading, unittest
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from playwright.async_api import async_playwright

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

  async def test_efficientnet(self):
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=False, args=["--enable-unsafe-webgpu"])
      page = await browser.new_page()
      url = f"http://{self.http.host}:{self.http.port}/examples/webgpu/efficientnet/index.html"
      resp = await page.goto(url)
      self.assertIsNotNone(resp)
      self.assertEqual(resp.status, 200)
      await page.wait_for_function("() => document.querySelector('#result').textContent.trim() === 'ready'", timeout=30_000)
      await page.click("input[type=button]")
      await page.wait_for_function("() => document.querySelector('#result').textContent.trim() === 'hen'", timeout=60_000)
      await browser.close()

  async def test_tinychat(self):
    async with async_playwright() as p:
      browser = await p.chromium.launch(headless=False, args=["--enable-features=Vulkan", "--enable-unsafe-webgpu"])
      page = await browser.new_page()
      url = f"http://{self.http.host}:{self.http.port}/examples/tinychat/tinychat-browser/index.html"
      resp = await page.goto(url)
      self.assertIsNotNone(resp)
      self.assertEqual(resp.status, 200)
      await page.wait_for_selector("textarea#input-form", timeout=30_000)
      await page.fill("textarea#input-form", "hi")
      await page.press("textarea#input-form", "Enter")
      await page.wait_for_function("() => document.querySelectorAll('.message-role-assistant').length > 0", timeout=10_000)
      await page.wait_for_selector("textarea#input-form:enabled", timeout=10_000)
      last = await page.inner_text(".message-role-assistant:last-child")
      # NOTE: relies on random seeds staying constant; TODO: set random seeds manually
      self.assertEqual(last.strip(), "How can I help you today?")
      await browser.close()

if __name__ == "__main__":
  unittest.main()
