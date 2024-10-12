from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class WebHandler(BaseHTTPRequestHandler):
  def setup(self):
    super().setup()
    print(f"connection established with {self.client_address}, socket: {self.connection.fileno()}")

  def do_GET(self):
    print(f"Received GET request for {self.path}")
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()
    if self.path == "/":
      self.get_page()

  def get_page(self):
    with open("extra/webgpu/index.html", "rb") as f:
      self.wfile.write(f.read())
      print("done")

class Web:
  def __init__(self):
    self.server = HTTPServer(('', 3000), WebHandler)
  def start(self):
    print("listening on port 3000")
    self.server.serve_forever()
  def shutdown(self):
    print("Shutting down server")
    self.server.shutdown()
    self.server.server_close()
    print("Server stopped")


web = Web()
server_thread = threading.Thread(target=web.start)
server_thread.start()

try:
  while True:
    pass
except KeyboardInterrupt:
  web.shutdown()
  server_thread.join()