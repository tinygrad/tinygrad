import struct
import socketserver
from base64 import b64encode
from hashlib import sha1
from email.parser import Parser
from io import StringIO
import multiprocessing
import socket
import time
import threading

server_ready = False


class WebSocketsHandler(socketserver.StreamRequestHandler):
  magic = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'

  def setup(self):
    print("setup")
    socketserver.StreamRequestHandler.setup(self)
    print("connection established", self.client_address)
    self.handshake_done = False

  def handle(self):
    print("request object", self.request)
    return
    # while True:
    #   if not self.handshake_done:
    #     self.handshake()
    #   else:

  def read_next_message(self):
    length = self.rfile.read(2)[1] & 127
    if length == 126:
      length = struct.unpack(">H", self.rfile.read(2))[0]
    elif length == 127:
      length = struct.unpack(">Q", self.rfile.read(8))[0]
    masks = [byte for byte in self.rfile.read(4)]
    decoded = ""
    for char in self.rfile.read(length):
        decoded += chr(char ^ masks[len(decoded) % 4])
    self.on_message(decoded)

  def send_message(self, message):
    self.request.send(bytes([129]))
    length = len(message)
    if length <= 125:
      self.request.send(bytes([length]))
    elif 126 <= length <= 65535:
      self.request.send(bytes([126]))
      self.request.send(struct.pack(">H", length))
    else:
      self.request.send(bytes([127]))
      self.request.send(struct.pack(">Q", length))
    self.request.send(message.encode())

  def handshake(self):
    data = self.request.recv(1024).strip().decode()
    headers = Parser().parsestr(data.split('\r\n', 1)[1])
    if headers.get("Upgrade", None) != "websocket":
      return
    print('Handshaking...')
    key = headers['Sec-WebSocket-Key']
    digest = b64encode(sha1((key + self.magic).encode()).digest()).decode()
    response = 'HTTP/1.1 101 Switching Protocols\r\n'
    response += 'Upgrade: websocket\r\n'
    response += 'Connection: Upgrade\r\n'
    response += f'Sec-WebSocket-Accept: {digest}\r\n\r\n'
    self.handshake_done = self.request.send(response.encode())
    global server_ready
    server_ready = True

  def on_message(self, message):
    print("Received message", message)

HOST, PORT = "localhost", 8766
def start_server():
  server = socketserver.TCPServer((HOST, PORT), WebSocketsHandler)
  print("Serving")
  server.serve_forever()


class WebDevice:
  def __init__(self):
    self.sock = None
    self.server = socketserver.TCPServer((HOST, PORT), WebSocketsHandler)
    self.process = threading.Thread(target=self.server.serve_forever)
    self.process.daemon = True
    self.process.start()
    print("server started")
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.connected = False

  def connect(self):
    if not self.connected:
      self.sock.connect((HOST, PORT))
      self.connected = True
    self.sock.sendall(bytes("HELLO\n", "utf-8"))
    print("sent")

  def __del__(self):
    print('closing')
    if self.sock:
      self.sock.close()
    print('socks closed')

if __name__ == "__main__":
  try:
    a = WebDevice()
    time.sleep(1)
    a.connect()
    time.sleep(10)
  finally:
    print('shutdown')
    a.server.shutdown()