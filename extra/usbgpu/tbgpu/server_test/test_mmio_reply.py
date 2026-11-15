import glob, os, socket, struct, subprocess, sys, tempfile, time, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = os.getenv("TINYGPU_TEST_SERVER_SOURCE", str(HERE / "../installer/Shared/server.c"))
REQUEST, RESPONSE = "<BIIQQQ", "<BQQ"
PATTERN = bytes((i * 7 + 3) & 255 for i in range(4096))


def build_server(binary:Path):
  compilers = [["xcrun", "clang"]]
  clt = "/Library/Developer/CommandLineTools"
  if sdks := sorted(glob.glob(f"{clt}/SDKs/MacOSX*.sdk")):
    compilers.append([f"{clt}/usr/bin/clang", "-isysroot", sdks[-1]])
  errors = []
  for compiler in compilers:
    commands = [
      [*compiler, "-O2", "-include", str(HERE / "mock_mmio.h"), "-c", SERVER, "-o", str(binary) + ".o"],
      [*compiler, "-O2", str(binary) + ".o", *(str(HERE / name) for name in ("mock_iokit.c", "mock_main.c", "mock_mmio.c")),
       "-o", str(binary)],
    ]
    for command in commands:
      result = subprocess.run(command, capture_output=True, text=True)
      if result.returncode:
        errors.append(result.stderr)
        break
    else:
      return
  raise RuntimeError("cannot build the isolated server:\n" + "\n".join(errors))


@unittest.skipUnless(sys.platform == "darwin", "TinyGPU server uses macOS headers")
class TestMMIOReply(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.directory = tempfile.TemporaryDirectory(prefix="tgmmio", dir="/tmp")
    cls.binary = Path(cls.directory.name) / "server"
    build_server(cls.binary)

  @classmethod
  def tearDownClass(cls):
    cls.directory.cleanup()

  def start(self, **env):
    self.path = str(Path(self.directory.name) / f"s{time.monotonic_ns()}")
    process = subprocess.Popen([str(self.binary), self.path], env=dict(os.environ, **env),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    def stop():
      if process.poll() is None:
        process.kill()
      process.wait(timeout=3)
    self.addCleanup(stop)
    for _ in range(200):
      if process.poll() is not None:
        self.fail("isolated server exited before listening")
      try:
        return self.connect()
      except (FileNotFoundError, ConnectionRefusedError):
        time.sleep(0.01)
    self.fail("isolated server did not listen")

  @staticmethod
  def receive(connection, size):
    data = bytearray()
    while len(data) < size:
      block = connection.recv(size - len(data))
      if not block:
        raise ConnectionError("reply ended before its declared length")
      data.extend(block)
    return bytes(data)

  def connect(self):
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    self.addCleanup(connection.close)
    connection.settimeout(2)
    connection.connect(self.path)
    connection.sendall(struct.pack(REQUEST, 3, 0, 0, 0, 2, 0))
    self.assertEqual(struct.unpack(RESPONSE, self.receive(connection, 17))[:2], (0, 0x1002))
    connection.sendall(struct.pack(REQUEST, 1, 0, 0, 0, 0, 0))
    self.assertEqual(struct.unpack(RESPONSE, self.receive(connection, 17))[0], 0)
    return connection

  def write_pattern(self, connection):
    connection.sendall(struct.pack(REQUEST, 7, 0, 0, 0, len(PATTERN), 0) + PATTERN)

  def read(self, connection, size):
    connection.sendall(struct.pack(REQUEST, 6, 0, 0, 0, size, 0))
    self.assertEqual(struct.unpack(RESPONSE, self.receive(connection, 17)), (0, size, 0))
    return self.receive(connection, size)

  def test_short_replies(self):
    for limit in (1, 3, 17, 19):
      with self.subTest(limit=limit):
        connection = self.start(MOCK_MMIO_LIMIT=str(limit))
        self.write_pattern(connection)
        self.assertEqual(self.read(connection, len(PATTERN)), PATTERN)
        self.assertEqual(self.read(connection, 31), PATTERN[:31])
        connection.close()

  def test_interrupted_reply(self):
    connection = self.start(MOCK_MMIO_EINTR="1", MOCK_MMIO_LIMIT="19")
    self.write_pattern(connection)
    self.assertEqual(self.read(connection, len(PATTERN)), PATTERN)
    self.assertEqual(self.read(connection, 4), PATTERN[:4])

  def check_failed_reply(self, **env):
    connection = self.start(**env)
    self.write_pattern(connection)
    connection.sendall(struct.pack(REQUEST, 6, 0, 0, 0, len(PATTERN), 0))
    self.assertEqual(connection.recv(1), b"")
    replacement = self.connect()
    self.assertEqual(self.read(replacement, 4), PATTERN[:4])

  def test_zero_send_disconnects_client(self):
    self.check_failed_reply(MOCK_MMIO_ZERO="1")

  def test_failed_send_disconnects_client(self):
    self.check_failed_reply(MOCK_MMIO_ERROR="1")

  def test_peer_shutdown_does_not_stop_server(self):
    connection = self.start(MOCK_MMIO_WAIT_CLOSE="1")
    self.write_pattern(connection)
    connection.sendall(struct.pack(REQUEST, 6, 0, 0, 0, len(PATTERN), 0))
    connection.close()
    replacement = self.connect()
    self.assertEqual(self.read(replacement, 4), PATTERN[:4])

  def test_empty_and_small_replies(self):
    connection = self.start()
    self.write_pattern(connection)
    for size in (0, 1, 3, 4, 31, 4096):
      with self.subTest(size=size):
        self.assertEqual(self.read(connection, size), PATTERN[:size])


if __name__ == "__main__":
  unittest.main()
