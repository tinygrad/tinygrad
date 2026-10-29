from __future__ import annotations
import struct
from tinygrad.helpers import wait_cond, round_up, DEBUG
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.system import PCIDevice, System

# controller registers, admin and io opcodes (nvm express base spec)
CAP, CC, CSTS, AQA, ASQ, ACQ, DOORBELLS = 0x0, 0x14, 0x1c, 0x24, 0x28, 0x30, 0x1000
OP_CREATE_SQ, OP_CREATE_CQ, OP_IDENTIFY, OP_SET_FEATURES, OP_WRITE, OP_READ = 0x01, 0x05, 0x06, 0x09, 0x01, 0x02
FEAT_NUM_QUEUES = 0x07
PAGE = 0x1000
SQE = struct.Struct("<IIQQQQIII12x")

class NVMeQueue:
  """a submission and a completion queue in host memory, polled. the command id is the submission slot"""
  def __init__(self, dev:NVMeDev, qid:int, entries:int):
    self.dev, self.qid, self.entries, self.tail, self.db_tail, self.head, self.phase, self.inflight = dev, qid, entries, 0, 0, 0, 1, 0
    mem, paddrs = dev.pci_dev.alloc_sysmem((sq_size:=round_up(entries * 64, PAGE)) + entries * 16)
    self.sq, self.cq = mem.view(0, sq_size).mv, mem.view(sq_size, entries * 16, fmt='I').mv
    self.sq_addr, self.cq_addr = paddrs[0], paddrs[sq_size // PAGE]
    self.sq_db, self.cq_db = (DOORBELLS + 2 * qid * dev.db_stride) // 4, (DOORBELLS + (2 * qid + 1) * dev.db_stride) // 4

  def push(self, opcode:int, nsid:int=0, prp1:int=0, prp2:int=0, cdw10:int=0, cdw11:int=0, cdw12:int=0) -> int:
    assert self.inflight < self.entries - 1, f"nvme {self.dev.pci_dev.pcibus}: queue {self.qid} is full"
    SQE.pack_into(self.sq, (cid:=self.tail) * 64, opcode | cid << 16, nsid, 0, 0, prp1, prp2, cdw10, cdw11, cdw12)
    self.tail, self.inflight = (cid + 1) % self.entries, self.inflight + 1
    return cid

  # the controller wedges a queue on doorbell writes that don't move it, so both doorbells are written only on change
  def kick(self):
    if self.tail == self.db_tail: return
    System.memory_barrier()
    self.dev.regs[self.sq_db] = self.db_tail = self.tail

  def reap(self) -> int:
    n = 0
    while ((dw3:=self.cq[self.head * 4 + 3]) >> 16 & 1) == self.phase:
      if dw3 >> 17: raise RuntimeError(f"nvme {self.dev.pci_dev.pcibus}: queue {self.qid} command {dw3 & 0xffff} failed, status {dw3 >> 17:#x}")
      self.head, n = self.head + 1, n + 1
      if self.head == self.entries: self.head, self.phase = 0, self.phase ^ 1
    if n: self.dev.regs[self.cq_db], self.inflight = self.head, self.inflight - n
    return n

  def wait(self, timeout_ms:int=10000):
    wait_cond(lambda: (self.reap(), self.inflight)[1], value=0, timeout_ms=timeout_ms, msg=f"nvme queue {self.qid}")

class PRPList:
  """prps for any page aligned range of a buffer (pages: its 4k iovas), from lists built once. list page j holds the entries of pages
  256j..256j+511, so the at most 256 entries of a transfer sit in one list page and lists never chain: a chained list is ambiguous,
  the controller takes the last slot of a page as data when a transfer happens to end there"""
  def __init__(self, dev:NVMeDev, pages:list[int]):
    assert dev.max_xfer <= 256 * PAGE, "a transfer must fit one list page"
    self.pages, n = pages, max(1, (len(pages) + 255) // 256)
    mem, self.addrs = dev.pci_dev.alloc_sysmem(n * PAGE)
    entries = mem.view(fmt='Q').mv
    for j in range(n):
      chunk = pages[256 * j:256 * j + 512]
      entries[512 * j:512 * j + len(chunk)] = memoryview(struct.pack(f"<{len(chunk)}Q", *chunk)).cast('Q')

  def __call__(self, off:int, nbytes:int) -> tuple[int, int]:
    """prp1, prp2 for nbytes at byte offset off of the buffer (off page aligned)"""
    i, n = off // PAGE, (nbytes + PAGE - 1) // PAGE
    if n <= 2: return self.pages[i], self.pages[i + 1] if n == 2 else 0
    return self.pages[i], self.addrs[(i + 1) // 256] + (i + 1) % 256 * 8

class NVMeDev:
  def __init__(self, pci_dev:PCIDevice, entries:int=1024, queues:int=1):
    self.pci_dev = pci_dev
    pci_dev.write_config(pci.PCI_COMMAND, pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MEMORY | pci.PCI_COMMAND_MASTER, 2)
    self.regs = pci_dev.map_bar(0, size=0x2000, fmt='I')
    cap = self.regs[CAP // 4] | self.regs[CAP // 4 + 1] << 32
    self.db_stride, timeout_ms, entries = 4 << (cap >> 32 & 0xf), (cap >> 24 & 0xff) * 500, min(entries, (cap & 0xffff) + 1)

    # reset, then enable with the admin queue: 16 byte completions, 64 byte submissions, 4k pages
    self.regs[CC // 4] = 0
    wait_cond(lambda: self.regs[CSTS // 4] & 1, value=0, timeout_ms=timeout_ms, msg="nvme disable")
    self.admin_q = NVMeQueue(self, 0, 64)
    self.regs[AQA // 4] = 63 << 16 | 63
    self.regs[ASQ // 4], self.regs[ASQ // 4 + 1] = self.admin_q.sq_addr & 0xffffffff, self.admin_q.sq_addr >> 32
    self.regs[ACQ // 4], self.regs[ACQ // 4 + 1] = self.admin_q.cq_addr & 0xffffffff, self.admin_q.cq_addr >> 32
    self.regs[CC // 4] = 4 << 20 | 6 << 16 | 1
    wait_cond(lambda: self.regs[CSTS // 4] & 3, value=1, timeout_ms=timeout_ms, msg="nvme enable")

    ident, ident_addr = pci_dev.alloc_sysmem(PAGE)
    self.admin(OP_IDENTIFY, prp1=ident_addr[0], cdw10=1)
    self.max_xfer = PAGE << ident.mv[77] if ident.mv[77] else 1 << 20
    self.admin(OP_IDENTIFY, nsid=1, prp1=ident_addr[0], cdw10=0)
    self.lba_shift = ident.mv[128 + 4 * (ident.mv[26] & 0xf) + 2]
    self.size = struct.unpack_from("<Q", ident.mv, 0)[0] << self.lba_shift

    # io queues 1..queues, each its own completion queue. the controller takes their commands round robin
    self.admin(OP_SET_FEATURES, cdw10=FEAT_NUM_QUEUES, cdw11=(queues - 1) << 16 | (queues - 1))
    self.io_qs = [NVMeQueue(self, qid, entries) for qid in range(1, queues + 1)]
    for q in self.io_qs:
      self.admin(OP_CREATE_CQ, prp1=q.cq_addr, cdw10=(entries - 1) << 16 | q.qid, cdw11=1)
      self.admin(OP_CREATE_SQ, prp1=q.sq_addr, cdw10=(entries - 1) << 16 | q.qid, cdw11=q.qid << 16 | 1)
    self.io_q = self.io_qs[0]
    if DEBUG >= 2: print(f"nvme {pci_dev.pcibus}: {self.size / 1e9:.0f} GB, lba {1 << self.lba_shift}, max transfer {self.max_xfer >> 10} KB")

  def admin(self, opcode:int, **kwargs):
    self.admin_q.push(opcode, **kwargs)
    self.admin_q.kick()
    self.admin_q.wait()

  def io(self, opcode:int, lba:int, nbytes:int, prp1:int, prp2:int) -> int:
    """queue a read or write of nbytes at lba (in device blocks), returns its cid. call io_q.kick() to submit and io_q.reap() to complete"""
    return self.io_q.push(opcode, 1, prp1, prp2, lba & 0xffffffff, lba >> 32, (nbytes >> self.lba_shift) - 1)

  def fini(self): self.regs[CC // 4] = 0
