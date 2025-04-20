import ctypes, struct, threading, socket, resource, time, tinygrad.runtime.autogen.ib as ib
from tinygrad.device import Compiled, Allocator, DMABuf
from tinygrad.helpers import DEBUG, mv_address

# Default soft fd limit is 1024, which is not enough, set soft to hard (maximum allowed by the os)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
if DEBUG>=2: print(f"IB: Increased fd limit from {soft} to {hard}")

def checkz(x, ret=None):
  assert x == 0, f'{x} != 0 (errno {ctypes.get_errno()})'
  return ret

class IBDiscovery:
  PORT = 31337

  def __init__(self):
    self.pending: dict[str, tuple[bytes, int]] = {}
    self.thread = threading.Thread(target=self.server_thread, daemon=True)
    self.thread.start()

  def exchange(self, remote:str, gid:bytes, qid:int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((remote, IBDiscovery.PORT))
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
    sock.bind(('0.0.0.0', IBDiscovery.PORT))

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

  def __del__(self):
    for base_iova in self.mrs: self.dereg(base_iova)
    self.mrs.clear()
    ib.ibv_dealloc_pd(self.pd)
    ib.ibv_close_device(self.ctx)

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

class IBConn:
  def __init__(self, ctx:IBCtx, discovery:IBDiscovery, remote:str, port_idx:int=1, gid_idx:int=3):
    self.ctx = ctx
    # Get our GID. It's kind of like an ip/mac address in infiniband. In RoCEv2 it's just an encoded ip address
    # Port index is a physical port on a card, indexing staring from one. A port can have multiple GIDs, indexing staring from zero
    self.gid = checkz(ib.ibv_query_gid(self.ctx.ctx, port_idx, gid_idx, ctypes.byref(gidu:=ib.union_ibv_gid())), bytes(gidu.raw))

    # Create Completion Channel. It is a file descriptor that kernel sends us notifications through, not a thing in infiniband spec, just linux-ism
    self.comp_channel = ib.ibv_create_comp_channel(self.ctx.ctx)
    # Create Completion Queue. When a Work Request with signaled flag is completed a Completion Queue Entry is pushed onto this queue
    self.cq = ib.ibv_create_cq(self.ctx.ctx, _capacity:=256, _cq_context:=None, self.comp_channel, _comp_vector:=0)
    self.pending_wrids: set[int] = set()
    self.last_wrid = 0

    # Create Queue Pair. It's the closest thing to a socket in infiniband with QP id being the closest thing to a port, except we can't choose it
    qp_init_attrs_cap = ib.struct_ibv_qp_cap(max_send_wr=64, max_recv_wr=64, max_send_sge=16, max_recv_sge=16)
    qp_init_attrs = ib.struct_ibv_qp_init_attr(send_cq=self.cq, recv_cq=self.cq, cap=qp_init_attrs_cap, qp_type=ib.IBV_QPT_RC) # Reliable Connection
    self.qp = ib.ibv_create_qp(self.ctx.pd, qp_init_attrs)

    # An important thing about QPs is their state, when a new QP is created it's in the RESET state, in order to bring up a connection we have to go
    # through INIT, Ready To Receive, Ready To Send. A good docs on QP state machine: https://www.rdmamojo.com/2012/05/05/qp-state-machine/

    # INIT
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_INIT, port_num=port_idx, qp_access_flags=ib.IBV_ACCESS_REMOTE_WRITE | ib.IBV_ACCESS_REMOTE_READ)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_PORT | ib.IBV_QP_ACCESS_FLAGS | ib.IBV_QP_PKEY_INDEX))

    # Exchange GID and QP id with remote. At least in RoCEv2 gid can be guessed from remote's ip, QP id can't.
    self.remote_gid, self.remote_qp_id = discovery.exchange(remote, self.gid, self.qp.contents.qp_num)

    # RTR
    qp_ah_attr_grh = ib.struct_ibv_global_route(hop_limit=1, dgid=ib.union_ibv_gid(raw=(ctypes.c_ubyte * 16)(*self.remote_gid)), sgid_index=gid_idx)
    qp_ah_attr = ib.struct_ibv_ah_attr(is_global=1, port_num=port_idx, grh=qp_ah_attr_grh)
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_RTR, path_mtu=ib.IBV_MTU_4096, dest_qp_num=self.remote_qp_id, rq_psn=0, max_dest_rd_atomic=1,
                                min_rnr_timer=12, ah_attr=qp_ah_attr)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_PATH_MTU | ib.IBV_QP_DEST_QPN | ib.IBV_QP_RQ_PSN | \
                                          ib.IBV_QP_MAX_DEST_RD_ATOMIC | ib.IBV_QP_MIN_RNR_TIMER | ib.IBV_QP_AV))

    # RTS
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_RTS, timeout=14, retry_cnt=7, rnr_retry=7, sq_psn=0, max_rd_atomic=1)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_TIMEOUT | ib.IBV_QP_RETRY_CNT | ib.IBV_QP_RNR_RETRY | ib.IBV_QP_SQ_PSN | \
                                          ib.IBV_QP_MAX_QP_RD_ATOMIC))

  def __del__(self):
    self.wait_cq()
    ib.ibv_destroy_qp(self.qp)
    ib.ibv_destroy_cq(self.cq)
    ib.ibv_destroy_comp_channel(self.comp_channel)

  def next_wrid(self):
    self.last_wrid += 1 # wc_id is uint64, this will never overflow
    self.pending_wrids.add(self.last_wrid)
    return self.last_wrid

  def wait_cq(self, wr_id: int|None=None):
    while (wr_id in self.pending_wrids) if wr_id is not None else self.pending_wrids:
      if self.ctx.ctx.contents.ops.poll_cq(self.cq, _num_entries:=1, ctypes.byref(wc:=ib.struct_ibv_wc())):
        if wc.status != ib.IBV_WC_SUCCESS:
          raise RuntimeError(f'Work Request completed with error: wr_id={wc.wr_id} status={ib.ibv_wc_status__enumvalues.get(wc.status, wc.status)}')
        self.pending_wrids.remove(wc.wr_id)

  # TODO: proper async for graph
  def recv(self, size:int):
    assert size <= 0x1000
    wrid = self.next_wrid()
    sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
    rwr = ib.struct_ibv_recv_wr(wr_id=wrid, sg_list=ctypes.pointer(sgl), num_sge=1)
    checkz(self.ctx.ctx.contents.ops.post_recv(self.qp, rwr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_recv_wr)())))
    self.wait_cq(wrid)
    return self.ctx.hmv[:size]

  def send(self, msg:bytes):
    assert len(msg) <= 0x1000
    self.ctx.hmv[:len(msg)] = msg
    wrid = self.next_wrid()
    sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
    swr = ib.struct_ibv_send_wr(wr_id=wrid, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_SEND, send_flags=ib.IBV_SEND_SIGNALED)
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(wrid)

  def rdma_write(self, dst_iova: int, src_iova: int, size:int, dst_key:int, src_key:int):
    wrid = self.next_wrid()
    sgl = ib.struct_ibv_sge(addr=src_iova, length=size, lkey=src_key)
    swr = ib.struct_ibv_send_wr(wr_id=wrid, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_RDMA_WRITE, send_flags=ib.IBV_SEND_SIGNALED,
                                wr=ib.union_ibv_send_wr_wr(rdma=ib.struct_ibv_send_wr_1_rdma(remote_addr=dst_iova, rkey=dst_key)))
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
    self.wait_cq(wrid)

class IBAllocator(Allocator):
  def __init__(self, dev):
    self.dev = dev

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
