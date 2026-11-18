import dataclasses, struct, unittest
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.ccb import MTCCB, MTResponse

class _Dev:
  def __init__(self): self.smc = self
  def _write(self, off, data): self.doorbell = (off, data)

class TestMTCCB(unittest.TestCase):
  def test_submission_and_response(self):
    dev, mem = _Dev(), bytearray(0x30000)
    q = MTCCB(dev, mem, 0x10000, 0x20000)
    q.submit(0x8000420000, 0x19250, 0x30000000)
    ci = mtfw.MTFW_CCB_ITEM.from_buffer_copy(mem[0x10000:0x10000+mtfw.MTFW_CCB_ITEM.SIZE])
    self.assertEqual(ci.submissionCmd, 104)
    self.assertEqual(ci.ccbItemToken, 0x103)
    self.assertEqual(ci.ccbItemPara.submissionPara.submissionAddr, 0x8000420000)
    self.assertEqual(ci.ccbItemPara.submissionPara.submissionSize, 0x19250)
    self.assertEqual(ci.ccbItemPara.submissionPara.pageTableRootAddr, 0x30000000)
    self.assertEqual(struct.unpack_from('<Q', mem, 0x20000)[0], 0x90000000 + 0x10000)
    self.assertEqual(struct.unpack_from('<I', mem, 0x24000)[0], 1)
    self.assertEqual(dev.doorbell, (0xc00b00, struct.pack('<I', 3)))

    response = MTResponse(0, 0, 0x103, 0x4d544e4f50, 0, 0, 0, 0)
    mem[0x21000:0x21030] = struct.pack('<IIQQQIIQ', response.flags, response.value, response.token, response.private,
      response.start_timestamp, response.run_time, response.info, response.ext_info)
    mem[0x24020:0x24024] = struct.pack('<I', 1)
    self.assertEqual(q.wait(0.01), response)
    self.assertEqual(struct.unpack_from('<I', mem, 0x24028)[0], 1)

    q.submit(0x8000420000, 0x19250, 0x30000000)
    ci = mtfw.MTFW_CCB_ITEM.from_buffer_copy(mem[0x10000:0x10000+mtfw.MTFW_CCB_ITEM.SIZE])
    self.assertEqual(ci.ccbItemToken, 0x104)
    self.assertEqual(struct.unpack_from('<Q', mem, 0x20008)[0], 0x90000000 + 0x10000)
    self.assertEqual(struct.unpack_from('<I', mem, 0x24000)[0], 2)
    response2 = dataclasses.replace(response, token=0x104)
    mem[0x21030:0x21060] = struct.pack('<IIQQQIIQ', response2.flags, response2.value, response2.token, response2.private,
      response2.start_timestamp, response2.run_time, response2.info, response2.ext_info)
    mem[0x24020:0x24024] = struct.pack('<I', 2)
    self.assertEqual(q.wait(0.01), response2)
    self.assertEqual(struct.unpack_from('<I', mem, 0x24028)[0], 2)

  def test_ring_wrap(self):
    dev, mem = _Dev(), bytearray(0x30000)
    q = MTCCB(dev, mem, 0x10000, 0x20000)
    q.wr_idx, q.seq, q.response_rd = 0xff, 0xff, 0xff
    q.submit(0x8000420000, 0x19250, 0x30000000)
    self.assertEqual(struct.unpack_from('<Q', mem, 0x207f8)[0], 0x90000000 + 0x10000)
    self.assertEqual(struct.unpack_from('<I', mem, 0x24000)[0], 0)
    response = MTResponse(0, 0, 0x202, 0x4d544e4f50, 0, 0, 0, 0)
    mem[0x23fd0:0x24000] = struct.pack('<IIQQQIIQ', response.flags, response.value, response.token, response.private,
      response.start_timestamp, response.run_time, response.info, response.ext_info)
    mem[0x24020:0x24024] = struct.pack('<I', 0)
    self.assertEqual(q.wait(0.01), response)
    self.assertEqual(struct.unpack_from('<I', mem, 0x24028)[0], 0)

  def test_response_size(self):
    with self.assertRaises(ValueError): MTResponse.from_bytes(bytes(4))

if __name__ == '__main__': unittest.main()
