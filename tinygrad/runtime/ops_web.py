import struct
from base64 import b64encode
from hashlib import sha1
from email.parser import Parser
import io
import socket
import threading
from typing import Literal

HOST, PORT = "localhost", 8766


def declineExtraConnection(socket: socket.socket):
  while True:
    req, addr = socket.accept()
    print(f"Address {addr} attempted to connect, turning down")
    try:
      req.sendall(b"Only one connection allowed\n")
    finally:
      req.close()

def handshake(req: socket.socket):
  ws_hash = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
  data = req.recv(1024).strip().decode()
  headers = Parser().parsestr(data.split("\r\n", 1)[1])
  if headers.get("Upgrade", None) != "websocket":
    return
  print("Handshaking...")
  key = headers["Sec-WebSocket-Key"]
  digest = b64encode(sha1((key + ws_hash).encode()).digest()).decode()
  response = "HTTP/1.1 101 Switching Protocols\r\n"
  response += "Upgrade: websocket\r\n"
  response += "Connection: Upgrade\r\n"
  response += f"Sec-WebSocket-Accept: {digest}\r\n\r\n"
  req.send(response.encode())

def read(rfile: io.BufferedReader):
  length = rfile.read(2)[1] & 127
  if length == 126:
    length = struct.unpack(">H", rfile.read(2))[0]
  elif length == 127:
    length = struct.unpack(">Q", rfile.read(8))[0]
  masks = [byte for byte in rfile.read(4)]
  decoded = ""
  for char in rfile.read(length):
    decoded += chr(char ^ masks[len(decoded) % 4])
  return decoded

def write(req: socket.socket, message: bytes, mode: Literal['b', 't']):
  TEXT = bytes([129])
  BINARY = bytes([130])
  req.send(BINARY if mode == 'b' else TEXT)
  length = len(message)
  if length <= 125:
    req.send(bytes([length]))
  elif 126 <= length <= 65535:
    req.send(bytes([126]))
    req.send(struct.pack(">H", length))
  else:
    req.send(bytes([127]))
    req.send(struct.pack(">Q", length))
  req.send(message if mode == 'b' else message.encode('utf-8'))

class WebDevice:
  def __init__(self, address):
    self.socket: socket.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    self.socket.bind(address)
    self.socket.listen(1)
    print("Waiting for browser")
    req, address = self.socket.accept()
    print("Browser connected from", address)
    t = threading.Thread(target=declineExtraConnection, args=(self.socket,))
    t.daemon = True
    t.start()
    handshake(req)
    self.rfile = req.makefile("rb", -1)
    self.req = req

  def send(self, msg, mode: Literal['t', 'b']):
    write(self.req, msg, mode)
    response = read(self.rfile)
    return response


if __name__ == "__main__":
  a = WebDevice((HOST, PORT))
  print("Creating buffer on browser")
  response = a.send('{"receiver": "device", "method": "createBuffer"}', 't')
  print("response", response)