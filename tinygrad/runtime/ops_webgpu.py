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
import os, subprocess, pathlib, ctypes, tempfile, functools
from typing import List, Any, Tuple, Optional, cast, TypeVar
from tinygrad.helpers import prod, getenv, DEBUG
from tinygrad.device import Compiled, Compiler, CompileError, LRUAllocator
from tinygrad.renderer.cstyle import MetalRenderer
import traceback
from enum import Enum

server_ready = False


class WebSocketsHandler(socketserver.StreamRequestHandler):
  def __init__(self, request, client_address, server):
    self.request = request
    self.client_address = client_address
    self.server = server
    self.setup()
    try:
        self.handle()
    finally:
        self.finish()
  magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

  def setup(self):
    socketserver.StreamRequestHandler.setup(self)
    print("connection established", self.client_address)
    self.handshake_done = False

  def handle(self):
    while True:
      if not self.handshake_done:
        self.handshake()
      else:
        self.read_next_message()

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


  def msg(self, message):
    TEXT = bytes([129])
    BINARY = bytes([130])
    self.request.send(BINARY)
    length = len(message)
    if length <= 125:
      self.request.send(bytes([length]))
    elif 126 <= length <= 65535:
      self.request.send(bytes([126]))
      self.request.send(struct.pack(">H", length))
    else:
      self.request.send(bytes([127]))
      self.request.send(struct.pack(">Q", length))
    self.request.send(message)

  def handshake(self):
    data = self.request.recv(1024).strip().decode()
    headers = Parser().parsestr(data.split("\r\n", 1)[1])
    if headers.get("Upgrade", None) != "websocket":
      return
    print("Handshaking...")
    key = headers["Sec-WebSocket-Key"]
    digest = b64encode(sha1((key + self.magic).encode()).digest()).decode()
    response = "HTTP/1.1 101 Switching Protocols\r\n"
    response += "Upgrade: websocket\r\n"
    response += "Connection: Upgrade\r\n"
    response += f"Sec-WebSocket-Accept: {digest}\r\n\r\n"
    self.handshake_done = self.request.send(response.encode())
    global server_ready
    server_ready = True

  def on_message(self, message):
    print(message)


class Server(socketserver.TCPServer):
  allow_reuse_address = True

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.ws = None

  def finish_request(self, request, client_address):
    try:
      self.ws = self.RequestHandlerClass.__new__(self.RequestHandlerClass)
      self.ws.__init__(request, client_address, self)
    except Exception:
      print("Error ocurred in websocket, likely that the client disconnected you should restart the program")
      print(traceback.format_exc())


HOST, PORT = "localhost", 8766

server = Server((HOST, PORT), WebSocketsHandler)

class Methods:
  createShaderModule: 0
  createBuffer: 1
  createBufferInit: 2
  createComputePipeline: 3

class WebGPUAllocator:
  def _alloc(self, size: int):
    pass

class WebDevice:
  def __init__(self):
    self.sock = None
    self.process = threading.Thread(target=server.serve_forever)
    self.process.daemon = True
    self.process.start()
    self.server = server
    while server.ws is None:
      print("Waiting for browser connect")
      time.sleep(3)
    self.device = self.server.ws
    print("Browser ready")


if __name__ == "__main__":
  a = WebDevice()
  while True:
    if a.device:
      text = input('Hit enter to send HELLO')
      a.device.msg(b'HELLO')
