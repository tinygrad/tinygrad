from __future__ import annotations
import dataclasses, struct, time
from tinygrad.runtime.autogen import mtfw
from tinygrad.runtime.support.mt.mtdev import MTDev

@dataclasses.dataclass(frozen=True)
class MTResponse:
  flags: int
  value: int
  token: int
  private: int
  start_timestamp: int
  run_time: int
  info: int
  ext_info: int

  @classmethod
  def from_bytes(cls, data:bytes) -> MTResponse:
    if len(data) != 0x30: raise ValueError("Invalid MT CCB response size")
    return cls(*struct.unpack('<IIQQQIIQ', data))

class MTCCB:
  def __init__(self, dev:MTDev, mem, item_off:int, queue_off:int, ccb_type:int=3, meta_base:int=0x90000000):
    self.dev, self.mem, self.item_off, self.queue_off = dev, mem, item_off, queue_off
    self.ccb_type, self.meta_base, self.wr_idx = ccb_type, meta_base, 0

  def _publish(self, ci) -> None:
    self.mem[self.item_off:self.item_off+ci.SIZE] = bytes(ci)
    self.mem[self.queue_off:self.queue_off+8] = struct.pack('<Q', self.meta_base + self.item_off)
    self.wr_idx += 1
    self.mem[self.queue_off+0x4000:self.queue_off+0x4004] = struct.pack('<I', self.wr_idx)
    self.dev.smc._write(0xc00b00, struct.pack('<I', self.ccb_type))

  def submit(self, submission_addr:int, submission_size:int, root_pa:int, generation:int=1, command:int=104,
             token:int=0x103, private:int=0x4d544e4f50, queue_id:int=0) -> None:
    ci = mtfw.MTFW_CCB_ITEM()
    ci.submissionCmd, ci.ccbItemToken, ci.hostPrivateData, ci.processId = command, token, private, 1
    sp = ci.ccbItemPara.submissionPara
    sp.pageTableRootAddr, sp.pageTableGeneration = root_pa, generation
    sp.submissionQueueId, sp.submissionAddr, sp.submissionSize = queue_id, submission_addr, submission_size
    sp.qSempOutofOrder = 1
    self._publish(ci)

  def flush_mmu(self, root_pa:int, generation:int=1) -> None:
    ci = mtfw.MTFW_CCB_ITEM()
    ci.submissionCmd, ci.ccbItemToken, ci.hostPrivateData, ci.processId = 13, 0x103, 0x4d544e4f50, 1
    sp = ci.ccbItemPara.invalidTlbCfg
    sp.pageTableRootAddr, sp.pageTableGeneration = root_pa, generation
    self._publish(ci)

  def response_index(self) -> int:
    return struct.unpack('<I', self.mem[self.queue_off+0x4020:self.queue_off+0x4024])[0]

  def wait(self, timeout:float=5.0) -> MTResponse:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
      if (wr := self.response_index()) != 0:
        off = self.queue_off + 0x1000 + (wr - 1) * 0x30
        response = MTResponse.from_bytes(bytes(self.mem[off:off+0x30]))
        self.mem[self.queue_off+0x4028:self.queue_off+0x402c] = struct.pack('<I', wr)
        return response
      time.sleep(0.001)
    raise TimeoutError(f"MT CCB {self.ccb_type} response timed out")
