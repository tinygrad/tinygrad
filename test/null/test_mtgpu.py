import ctypes, struct, unittest
from tinygrad.runtime.support.memory import MMIOInterface
from tinygrad.runtime.support.mt.mtdev import MTSMC, MTDev, MT_IPC_EVENT_GET_FW_VERSION, MT_IPC_EVENT_SET_POWER_STATE, _Ring, _ipc_header


def make_mmio():
  buf = (ctypes.c_ubyte * 0x84000)()
  for ctrl, data in ((0x30000, 0x30018), (0x30218, 0x30230)):
    struct.pack_into('<IIIIII', buf, ctrl, 0, 0, 0, 0x200, data, 0x7f)
  mmio = MMIOInterface(ctypes.addressof(buf), len(buf))
  mmio.owner = buf  # keep the backing allocation alive for the lifetime of the view
  return mmio


class MockSMC(MTSMC):
  def __init__(self, mmio):
    super().__init__(mmio)
    self.requests = []
    self.response_error = 0

  def _kick(self, event_id, target, msg_id):
    header, payload, next_rd = self.tx.read_message()
    self._w32(self.tx.ctrl, next_rd)
    self.requests.append((event_id, target, msg_id, payload))
    if event_id == MT_IPC_EVENT_GET_FW_VERSION:
      response = b'rtos\0\0\0\0' + struct.pack('<I', 20107) + b'vbios\0\0\0' + struct.pack('<I', 10606)
    else: response = bytes(12)
    header = _ipc_header(event_id, len(response) // 4, msg_id=msg_id, msg_type=1) | (self.response_error << 53)
    self.rx.write(struct.pack('<Q', header) + response)


class TestMTSMC(unittest.TestCase):
  def test_version_and_pstate(self):
    smc = MockSMC(make_mmio())
    self.assertEqual(smc.rtos_version(), [('rtos', 20107), ('vbios', 10606)])
    smc.set_pstate(0)
    self.assertEqual([x[:3] for x in smc.requests], [(MT_IPC_EVENT_GET_FW_VERSION, 0, 0), (MT_IPC_EVENT_SET_POWER_STATE, 0, 1)])
    self.assertEqual(smc.requests[1][3], struct.pack('<III', 0, 0, 0))

  def test_ring_wrap(self):
    smc = MockSMC(make_mmio())
    for ring in (smc.tx, smc.rx):
      smc._w32(ring.ctrl, 0x1f8 * 2)
      smc._w32(ring.ctrl + 4, 0x1f8 * 2)
    self.assertEqual(smc.rtos_version(), [('rtos', 20107), ('vbios', 10606)])
    self.assertEqual(smc._r32(smc.tx.ctrl), (108 * 2) | 1)
    self.assertEqual(smc._r32(smc.tx.ctrl + 4), (108 * 2) | 1)
    self.assertEqual(smc._r32(smc.rx.ctrl), (24 * 2) | 1)

  def test_ipc_error(self):
    smc = MockSMC(make_mmio())
    smc.response_error = 1
    with self.assertRaisesRegex(RuntimeError, 'failed with error 1'): smc.set_pstate(0)

  def test_bad_ring(self):
    mmio = make_mmio()
    smc = MTSMC(mmio)
    smc._w32(0x30010, 0x3001c)
    with self.assertRaisesRegex(RuntimeError, 'data pointer'): _Ring(smc, 0x30000)


class TestMTDev(unittest.TestCase):
  def test_bvnc(self):
    dev = MTDev.__new__(MTDev)
    dev.core_ids = [0x0023000406600017]
    self.assertEqual(dev.bvnc, (35, 4, 1632, 23))


if __name__ == '__main__': unittest.main()
