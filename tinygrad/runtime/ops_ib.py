from __future__ import annotations
import ctypes, struct, threading, socket, time, tinygrad.runtime.autogen.ib as ib
from tinygrad.device import Compiled, Allocator, DMABuf
from tinygrad.helpers import DEBUG, mv_address

PORT_ID = 1
GID_ID = 3
DISCOVERY_PORT = 31337

def checkz(x):
  assert x == 0, ctypes.get_errno()

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
      if DEBUG>=2: print(f"Registred {dmabuf.base_iova:#x}:{dmabuf.base_iova+dmabuf.base_size-1:#x}")
      if dmabuf.dmabuf_fd is not None:
        mr = ib.ibv_reg_dmabuf_mr(self.pd, _dmabuf_offset:=0, dmabuf.base_size, dmabuf.base_iova, dmabuf.dmabuf_fd,
                                  ib.IBV_ACCESS_LOCAL_WRITE | ib.IBV_ACCESS_REMOTE_READ | ib.IBV_ACCESS_REMOTE_WRITE)
      else:
        mr = ib.ibv_reg_mr_iova2(self.pd, ctypes.c_void_p(dmabuf.base_iova), dmabuf.base_size, dmabuf.base_iova,
                                 ib.IBV_ACCESS_LOCAL_WRITE | ib.IBV_ACCESS_REMOTE_READ | ib.IBV_ACCESS_REMOTE_WRITE)
      if mr is None: raise RuntimeError(f"Couldn't register memory region for {dmabuf}")
      self.mrs[dmabuf.base_iova] = mr
    return self.mrs[dmabuf.base_iova]
  def dereg(self, base_iova:int):
    if DEBUG>=2: print(f"Deregistred {base_iova:#x}")
    ib.ibv_dereg_mr(self.mrs[base_iova])
    del self.mrs[base_iova]
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
    return self.ctx.hmv[:size]
  def send(self, msg:bytes):
    assert len(msg) <= 0x1000
    self.ctx.hmv[:len(msg)] = msg
    sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
    swr = ib.struct_ibv_send_wr(wr_id=1338, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_SEND, send_flags=ib.IBV_SEND_SIGNALED)
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(1338)
  def rdma_write(self, dst_iova: int, src_iova: int, size:int, dst_key:int, src_key:int):
    sgl = ib.struct_ibv_sge(addr=src_iova, length=size, lkey=src_key)
    swr = ib.struct_ibv_send_wr(wr_id=1339, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_RDMA_WRITE, send_flags=ib.IBV_SEND_SIGNALED,
                                wr=ib.union_ibv_send_wr_wr(rdma=ib.struct_ibv_send_wr_1_rdma(remote_addr=dst_iova, rkey=dst_key)))
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(1339)

class IBAllocator(Allocator):
  def __init__(self, dev:IBDevice): self.dev = dev
  def _alloc(self, size, options): pass
  def dma_send(self, src:DMABuf):
    mr = self.dev.ctx.reg(src)
    msg = self.dev.conn.recv(0x1000)
    dst_iova, dst_size, dst_rkey, vid_len = struct.unpack('<QQII', bytes(msg[:24]))
    vid = msg[24:24+vid_len]
    assert self.dev.vid == vid, f"{self.dev.vid!r} != {vid!r}"
    assert src.size == dst_size, f"{src.size} != {dst_size}"
    if DEBUG>=2: print(f"RDMA WRITE {src.iova:#x}:{src.iova+src.size-1:#x} => {dst_iova:#x}:{dst_iova+src.size-1:#x}")
    self.dev.conn.rdma_write(dst_iova, src.iova, src.size, dst_rkey, mr.contents.lkey)
    self.dev.conn.send(b'done')
  def dma_recv(self, dst:DMABuf):
    mr = self.dev.ctx.reg(dst)
    self.dev.conn.send(struct.pack('<QQII', dst.iova, dst.size, mr.contents.rkey, len(self.dev.vid))+self.dev.vid)
    assert self.dev.conn.recv(4) == b'done'

class IBDevice(Compiled):
  dev_list = None
  discovery = None
  ctxs: dict[int, IBCtx] = {}
  conns: dict[tuple[int, str], IBConn] = {}

  def __init__(self, device:str):
    # Enumerate devices on first open
    if IBDevice.dev_list is None:
      devs = ib.ibv_get_device_list(ctypes.byref(ndevs:=ctypes.c_int32()))
      IBDevice.dev_list = [devs[i] for i in range(ndevs.value)]
    # Start discoveery server if not started
    if IBDevice.discovery is None: IBDevice.discovery = IBDiscovery()

    _, dev = device.split(':')
    idx, remote, name = dev.split('/')
    IBDevice.ctxs[int(idx)] = self.ctx = IBCtx(IBDevice.dev_list[int(idx)]) if int(idx) not in IBDevice.ctxs else IBDevice.ctxs[int(idx)]
    IBDevice.conns[(int(idx), remote)] = self.conn = IBDevice.conns[(int(idx), remote)] if (int(idx), remote) in IBDevice.conns else IBConn(self.ctx, IBDevice.discovery, remote) # noqa: E501
    self.vid = name.encode()

    super().__init__(device, IBAllocator(self), None, None, None)

  def dma_dereg(self, base_iova:int):
    if base_iova in self.ctx.mrs: self.ctx.dereg(base_iova)
