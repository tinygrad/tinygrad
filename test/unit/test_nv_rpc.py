import ctypes, types, unittest

from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.nv.ip import NVRpcQueue


class TestNVRpcQueueRecords(unittest.TestCase):
  def setUp(self):
    self.queue = object.__new__(NVRpcQueue)
    self.queue.tx = types.SimpleNamespace(msgSize=0x1000, msgCount=4)
    self.storage = bytearray(self.queue.tx.msgSize * self.queue.tx.msgCount)
    self.queue.queue_mv = memoryview(self.storage)

  def make_record(self, sequence:int, function:int, payload:bytes) -> bytes:
    hdr = nv.rpc_message_header_v(signature=nv.NV_VGPU_MSG_SIGNATURE_VALID, function=function,
                                  length=ctypes.sizeof(nv.rpc_message_header_v) + len(payload))
    elem_count = (ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) + hdr.length + 0xFFF) // 0x1000
    elem = nv.GSP_MSG_QUEUE_ELEMENT(elemCount=elem_count, seqNum=sequence)
    elem.checkSum = self.queue._checksum(bytes(elem) + bytes(hdr) + payload)
    return bytes(elem) + bytes(hdr) + payload

  def write_record(self, slot:int, raw:bytes):
    off = slot * self.queue.tx.msgSize
    first = min(len(raw), len(self.storage) - off)
    self.storage[off:off+first] = raw[:first]
    self.storage[:len(raw)-first] = raw[first:]

  def test_reads_checksum_valid_record_across_ring_wrap(self):
    payload = bytes((x & 0xFF) for x in range(0x1000))
    self.write_record(3, self.make_record(7, 0x1234, payload))

    elem, hdr, raw = self.queue._read_record(3)

    self.assertEqual(elem.elemCount, 2)
    self.assertEqual(hdr.function, 0x1234)
    self.assertEqual(raw[ctypes.sizeof(nv.GSP_MSG_QUEUE_ELEMENT) + ctypes.sizeof(nv.rpc_message_header_v):], payload)

  def test_read_resp_returns_exact_payload_and_advances_by_element_count(self):
    payload = bytes((x & 0xFF) for x in range(0x1000))
    self.write_record(3, self.make_record(9, 0x1234, payload))
    self.queue.rx_view = [3]
    write_index = getattr(nv.msgqTxHeader, 'writePtr').offset // 4
    self.queue.tx_view = [0] * (write_index + 1)
    self.queue.tx_view[write_index] = 1
    self.queue.gsp = types.SimpleNamespace(nvdev=types.SimpleNamespace(pci_dev=types.SimpleNamespace(), is_err_state=False, devfmt="test"))

    self.assertEqual(list(self.queue.read_resp()), [(0x1234, payload)])
    self.assertEqual(self.queue.rx_view[0], 1)

if __name__ == "__main__": unittest.main()
