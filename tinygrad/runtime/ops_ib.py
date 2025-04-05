from __future__ import annotations
import ctypes, struct, threading, socket, time, tinygrad.runtime.autogen.ib as ib
from tinygrad.device import Compiled, Allocator
from tinygrad.helpers import DEBUG, mv_address, getenv

PORT_ID = 1
GID_ID = 3
DISCOVERY_PORT = 31337
RDMA = getenv('RDMA', 'PUSH')
assert RDMA in {'PUSH', 'PULL'}

def checkz(x):
  assert x == 0, ctypes.get_errno()

class DMABuf:
  def __init__(self, iova:int, size:int, base_iova:int|None=None, base_size:int|None=None, dmabuf_fd:int|None=None):
    self.iova, self.size, self.dmabuf_fd = iova, size, dmabuf_fd
    self.base_iova, self.base_size = base_iova if base_iova is not None else iova, base_size if base_size is not None else size

class IBDiscovery:
  def __init__(self):
    self.pending: dict[str, tuple[bytes, int]] = {}
    self.thread = threading.Thread(target=self.server_thread, daemon=True)
    self.thread.start()
  def exchange(self, remote:str, gid:bytes, qid:int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((remote, DISCOVERY_PORT))
    sock.setblocking(False)

    self.pending[remote] = (gid, qid)
    while True:
      try:
        sock.send(b'\x00')
        rcv = sock.recv(20)
        while remote in self.pending: time.sleep(0.02)
        return (rcv[:16], struct.unpack('<I', rcv[16:])[0])
      except (BlockingIOError, ConnectionRefusedError): time.sleep(0.05)
  def server_thread(self):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', DISCOVERY_PORT))

    while True:
      _, (raddr, rport) = sock.recvfrom(1)
      if raddr in self.pending:
        sock.sendto(self.pending[raddr][0] + struct.pack('<I', self.pending[raddr][1]), (raddr, rport))
        del self.pending[raddr]

class IBCtx:
  def __init__(self, dev):
    self.ctx = ib.ibv_open_device(dev)
    self.pd = ib.ibv_alloc_pd(self.ctx)
    self.mrs = {}
    self.hmv = memoryview(bytearray(b'\x00'*0x1000))
    self.hmr = self.reg(DMABuf(mv_address(self.hmv), len(self.hmv)))
  def reg(self, dmabuf:DMABuf):
    if dmabuf.base_iova not in self.mrs:
      if dmabuf.dmabuf_fd is not None:
        mr = ib.ibv_reg_dmabuf_mr(self.pd, 0, dmabuf.base_size, dmabuf.base_iova, dmabuf.dmabuf_fd,
                                                          ib.IBV_ACCESS_LOCAL_WRITE | ib.IBV_ACCESS_REMOTE_READ | ib.IBV_ACCESS_REMOTE_WRITE)
      else:
        mr = ib.ibv_reg_mr_iova2(self.pd, ctypes.c_void_p(dmabuf.base_iova), dmabuf.base_size, dmabuf.base_iova,
                                                         ib.IBV_ACCESS_LOCAL_WRITE | ib.IBV_ACCESS_REMOTE_READ | ib.IBV_ACCESS_REMOTE_WRITE)
      if mr is None: raise RuntimeError(f"Couldn't register memory region for {dmabuf}")
      if DEBUG>=1: print(f"IB: Registred memory region {dmabuf.base_iova:#x}")
      self.mrs[dmabuf.base_iova] = mr
    return self.mrs[dmabuf.base_iova]
  def dereg(self, base_iova:int):
    ib.ibv_dereg_mr(self.mrs[base_iova])
    del self.mrs[base_iova]
    if DEBUG>=1: print(f"IB: Deregistred memory region {base_iova:#x}")
  def __del__(self):
    for base_iova in self.mrs: self.dereg(base_iova)
    self.mrs.clear()
    ib.ibv_dealloc_pd(self.pd)
    ib.ibv_close_device(self.ctx)

class IBConn:
  def __init__(self, ctx:IBCtx, discovery:IBDiscovery, remote:str):
    self.ctx = ctx
    self.comp_channel = ib.ibv_create_comp_channel(self.ctx.ctx)
    self.cq = ib.ibv_create_cq(self.ctx.ctx, 100, None, self.comp_channel, 0)

    qp_init_attrs_cap = ib.struct_ibv_qp_cap(max_send_wr=64, max_recv_wr=64, max_send_sge=16, max_recv_sge=16)
    qp_init_attrs = ib.struct_ibv_qp_init_attr(send_cq=self.cq, recv_cq=self.cq, cap=qp_init_attrs_cap, qp_type=ib.IBV_QPT_RC)
    self.qp = ib.ibv_create_qp(self.ctx.pd, qp_init_attrs)
    checkz(ib.ibv_query_gid(self.ctx.ctx, PORT_ID, GID_ID, ctypes.byref(gidu:=ib.union_ibv_gid())))
    self.gid, self.qid = bytes(gidu.raw), self.qp.contents.qp_num

    # INIT
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_INIT, port_num=PORT_ID, qp_access_flags=ib.IBV_ACCESS_REMOTE_WRITE | ib.IBV_ACCESS_REMOTE_READ)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_PKEY_INDEX | ib.IBV_QP_PORT | ib.IBV_QP_ACCESS_FLAGS))

    self.remote_gid, self.remote_qid = discovery.exchange(remote, self.gid, self.qid)

    # RTR
    qp_ah_attr_grh = ib.struct_ibv_global_route(hop_limit=1, dgid=ib.union_ibv_gid(raw=(ctypes.c_ubyte * 16)(*self.remote_gid)), sgid_index=GID_ID)
    qp_ah_attr = ib.struct_ibv_ah_attr(is_global=1, port_num=PORT_ID, grh=qp_ah_attr_grh)
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_RTR, path_mtu=ib.IBV_MTU_4096, dest_qp_num=self.remote_qid, rq_psn=0, max_dest_rd_atomic=1,
                                min_rnr_timer=12, ah_attr=qp_ah_attr)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_AV | ib.IBV_QP_PATH_MTU | ib.IBV_QP_DEST_QPN | ib.IBV_QP_RQ_PSN | \
                            ib.IBV_QP_MAX_DEST_RD_ATOMIC | ib.IBV_QP_MIN_RNR_TIMER))

    # RTS
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_RTS, timeout=14, retry_cnt=7, rnr_retry=7, sq_psn=0, max_rd_atomic=1)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_TIMEOUT | ib.IBV_QP_RETRY_CNT | ib.IBV_QP_RNR_RETRY | ib.IBV_QP_SQ_PSN | \
                            ib.IBV_QP_MAX_QP_RD_ATOMIC))
  def __del__(self):
    ib.ibv_destroy_qp(self.qp)
    ib.ibv_destroy_cq(self.cq)
    ib.ibv_destroy_comp_channel(self.comp_channel)
  def wait_cq(self, wr_id: int):
    wc = ib.struct_ibv_wc()
    while True:
      if (r:=self.ctx.ctx.contents.ops.poll_cq(self.cq, 1, ctypes.byref(wc))) != 0:
        assert r == 1 and wc.wr_id == wr_id and wc.status == 0, f'{r} {wc.wr_id} {wc.status}'
        break
  def recv(self, size:int):
    assert size <= 0x1000
    sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
    rwr = ib.struct_ibv_recv_wr(wr_id=1337, sg_list=ctypes.pointer(sgl), num_sge=1)
    checkz(self.ctx.ctx.contents.ops.post_recv(self.qp, rwr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_recv_wr)())))
    self.wait_cq(1337)
    return bytes(self.ctx.hmv[:size])
  def send(self, msg:bytes):
    assert len(msg) <= 0x1000
    self.ctx.hmv[:len(msg)] = msg
    sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
    swr = ib.struct_ibv_send_wr(wr_id=1338, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_SEND, send_flags=ib.IBV_SEND_SIGNALED)
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(1338)
  def rdma_read(self, dst_iova: int, src_iova: int, size:int, dst_key:int, src_key:int):
    sgl = ib.struct_ibv_sge(addr=dst_iova, length=size, lkey=dst_key)
    swr = ib.struct_ibv_send_wr(wr_id=1339, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_RDMA_READ, send_flags=ib.IBV_SEND_SIGNALED,
                                wr=ib.union_ibv_send_wr_wr(rdma=ib.struct_ibv_send_wr_1_rdma(remote_addr=src_iova, rkey=src_key)))
    start = time.perf_counter()
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(1339)
    end = time.perf_counter()
    if DEBUG>=1: print(f"IB: pulled {size/1024/1024} MB ({(size*8/1024/1024/1024)/(end-start):.1f} Gb/s)")
  def rdma_write(self, dst_iova: int, src_iova: int, size:int, dst_key:int, src_key:int):
    sgl = ib.struct_ibv_sge(addr=src_iova, length=size, lkey=src_key)
    swr = ib.struct_ibv_send_wr(wr_id=1340, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_RDMA_WRITE, send_flags=ib.IBV_SEND_SIGNALED,
                                wr=ib.union_ibv_send_wr_wr(rdma=ib.struct_ibv_send_wr_1_rdma(remote_addr=dst_iova, rkey=dst_key)))
    start = time.perf_counter()
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(1340)
    end = time.perf_counter()
    if DEBUG>=1: print(f"IB: pushed {size/1024/1024} MB ({(size*8/1024/1024/1024)/(end-start):.1f} Gb/s)")

class IBAllocator(Allocator):
  def __init__(self, dev:IBDevice): self.dev = dev
  def _alloc(self, size, options): pass
  def dma_copyin(self, src:DMABuf):
    mr = self.dev.ctx.reg(src)
    if RDMA == 'PULL':
      self.dev.conn.send(struct.pack('<QI', src.iova, mr.contents.rkey))
      assert self.dev.conn.recv(4) == b'done'
    elif RDMA == 'PUSH':
      dst_iova, dst_rkey = struct.unpack('<QI', self.dev.conn.recv(12))
      self.dev.conn.rdma_write(dst_iova, src.iova, src.size, dst_rkey, mr.contents.lkey)
      self.dev.conn.send(b'done')
    else: raise RuntimeError(RDMA)
  def dma_copyout(self, dst:DMABuf):
    mr = self.dev.ctx.reg(dst)
    if RDMA == 'PULL':
      src_iova, src_rkey = struct.unpack('<QI', self.dev.conn.recv(12))
      self.dev.conn.rdma_read(dst.iova, src_iova, dst.size, mr.contents.lkey, src_rkey)
      self.dev.conn.send(b'done')
    elif RDMA == 'PUSH':
      self.dev.conn.send(struct.pack('<QI', dst.iova, mr.contents.rkey))
      assert self.dev.conn.recv(4) == b'done'
    else: raise RuntimeError(RDMA)

class IBDevice(Compiled):
  dev_list = None
  discovery = None
  ctxs: dict[int, IBCtx] = {}

  def __init__(self, device:str):
    print(f'Opening {device}')
    # Enumerate devices on first open
    if IBDevice.dev_list is None:
      devs = ib.ibv_get_device_list(ctypes.byref(ndevs:=ctypes.c_int32()))
      IBDevice.dev_list = [devs[i] for i in range(ndevs.value)]
    # Start discoveery server if not started
    if IBDevice.discovery is None: IBDevice.discovery = IBDiscovery()

    _, idx, remote = device.split(':')
    IBDevice.ctxs[int(idx)] = self.ctx = IBCtx(IBDevice.dev_list[int(idx)]) if int(idx) not in IBDevice.ctxs else IBDevice.ctxs[int(idx)]
    self.conn = IBConn(self.ctx, IBDevice.discovery, remote)

    super().__init__(device, IBAllocator(self), None, None, None)

  @staticmethod
  def dereg_all(base_iova:int):
    for ctx in IBDevice.ctxs.values():
      if base_iova in ctx.mrs: ctx.dereg(base_iova)

