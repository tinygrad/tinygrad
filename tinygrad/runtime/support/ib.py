from __future__ import annotations
import resource, ctypes, weakref, itertools, tinygrad.runtime.autogen.ib as ib
from typing import Iterator
from weakref import WeakKeyDictionary
from tinygrad.device import Buffer, DMACPURef, DMAFdRef
from tinygrad.helpers import getenv, round_up, DEBUG

DEFAULT_PORT, DEFAULT_GID = getenv("DEFAULT_PORT", 1), getenv("DEFAULT_GID", 3)
IOVA_ALIGN = resource.getpagesize()

# Default soft fd limit is 1024, which is not enough, set soft to hard (maximum allowed by the os)
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
if DEBUG>=2: print(f"IB: Increased fd limit from {soft} to {hard}")

def checkz(x, ret=None):
  assert x == 0, f'{x} != 0 (errno {ctypes.get_errno()})'
  return ret

class IBCtx:
  def __init__(self, idx:int):
    # Open the device (aka Host Channel Adapter in ib-speak)
    devs = ib.ibv_get_device_list(ctypes.byref(ndevs:=ctypes.c_int32()))
    if idx >= ndevs.value: raise IndexError(f"{idx} > {ndevs.value}")
    self.ctx = ib.ibv_open_device(devs[idx])
    ib.ibv_free_device_list(devs)

    # HACK: remove this (and all usage of `ctx.contents.ops`) when clang2py can deal with `static inline` wrapper-functions
    self.vctx = ctypes.cast(ctypes.addressof(self.ctx.contents) - ib.struct_verbs_context.context.offset, ctypes.POINTER(ib.struct_verbs_context))

    # Get attributes. Something like port_attr.max_msg_sz sound like it might requre taking the min of host's and remote's attributes if they differ
    self.device_attr = checkz(ib.ibv_query_device(self.ctx, ctypes.byref(da:=ib.struct_ibv_device_attr())), da)
    self.port_attr = checkz(self.vctx.contents.query_port(self.ctx, DEFAULT_PORT, ctypes.byref(pa:=ib.struct_ibv_port_attr()), ctypes.sizeof(pa)), pa)
    self.gid_attr = checkz(ib.ibv_query_gid(self.ctx, DEFAULT_PORT, DEFAULT_GID, ctypes.byref(ga:=ib.union_ibv_gid())), ga)

    # Allocate protection domain
    self.pd = ib.ibv_alloc_pd(self.ctx)
    self.next_iova: int = IOVA_ALIGN # don't start at zero (nullptr)

    # weakref(buf) => (iova, mr, mr_dealloc). mr_dealloc is kept here to avoid double freeing mrs that are deallocated in __del__
    self.mrs: WeakKeyDictionary[Buffer, tuple[int, ctypes._Pointer[ib.struct_ibv_mr], weakref.finalize]] = WeakKeyDictionary()

  def __del__(self):
    # must deallocate all mrs in protection domain before deallocating the protection domain
    if hasattr(self, "mrs"): [fin() for _,_,fin in self.mrs.values()]
    if hasattr(self, "pd"): ib.ibv_dealloc_pd(self.pd)
    if hasattr(self, "ctx"): ib.ibv_close_device(self.ctx)

  def alloc_iova(self, size:int, required_offset:int):
    iova = round_up(self.next_iova - required_offset, IOVA_ALIGN) + required_offset
    self.next_iova = iova + size
    return iova

  def reg(self, buf:Buffer) -> tuple[int, ctypes._Pointer[ib.struct_ibv_mr]]:
    buf = buf.base
    if buf not in self.mrs:
      if buf.nbytes > self.device_attr.max_mr_size: raise RuntimeError(f"Buffer too big: {buf.nbytes:#x} > {self.device_attr.max_mr_size:#x}")
      if len(self.mrs) >= self.device_attr.max_mr: raise RuntimeError(f"Out of memory region cap: {len(self.mrs)} >= {self.device_attr.max_mr}")
      # Local read is implied (but still have to create the memory region, except for short sends/writes with IBV_SEND_INLINE that are inlined by cpu)
      mr_flags = ib.IBV_ACCESS_LOCAL_WRITE | ib.IBV_ACCESS_REMOTE_READ | ib.IBV_ACCESS_REMOTE_WRITE
      match (dmaref:=buf.as_dmaref()):
        case DMACPURef():
          iova = self.alloc_iova(dmaref.size, dmaref.addr % IOVA_ALIGN)
          mr = ib.ibv_reg_mr_iova2(self.pd, ctypes.c_void_p(dmaref.addr), dmaref.size, iova, mr_flags)
        case DMAFdRef():
          iova = self.alloc_iova(dmaref.size, dmaref.offset % IOVA_ALIGN)
          mr = ib.ibv_reg_dmabuf_mr(self.pd, dmaref.offset, dmaref.size, iova, dmaref.fd, mr_flags)
        case _: raise RuntimeError(f"Unknown type of dma ref: {dmaref}")
      if not mr: raise RuntimeError(f"Couldn't register memory region for {buf} {dmaref} (errno={ctypes.get_errno()})")
      self.mrs[buf] = (iova, mr, weakref.finalize(buf, ib.ibv_dereg_mr, mr))
    return self.mrs[buf][0:2]

class IBConn:
  def __init__(self, ctx:IBCtx):
    self.ctx = ctx

    # Create Completion Channel. It is a file descriptor that kernel sends notifications through, not a thing in infiniband spec, just linux-ism
    self.comp_channel = ib.ibv_create_comp_channel(self.ctx.ctx)
    # Create Completion Queue. When a Work Request with signaled flag is completed a Completion Queue Entry is pushed onto this queue
    self.cq = ib.ibv_create_cq(self.ctx.ctx, _capacity:=256, _cq_context:=None, self.comp_channel, _comp_vector:=0)
    self.pending_wrids: set[int] = set()
    self.wrid_num: Iterator[int] = itertools.count(0) # wc_id is uint64, this will never overflow

    # Create Queue Pair. It's the closest thing to a socket in infiniband with QP num being the closest thing to a port, except it's allocated by hca
    qp_init_attrs_cap = ib.struct_ibv_qp_cap(max_send_wr=64, max_recv_wr=64, max_send_sge=8, max_recv_sge=8, max_inline_data=64)
    qp_init_attrs = ib.struct_ibv_qp_init_attr(send_cq=self.cq, recv_cq=self.cq, cap=qp_init_attrs_cap, qp_type=ib.IBV_QPT_RC) # Reliable Connection
    self.qp = ib.ibv_create_qp(self.ctx.pd, ctypes.byref(qp_init_attrs))
    self.qp_cap = qp_init_attrs.cap

    # The most important thing about QPs is their state, when a new QP is created it's in the RESET state, before it can be properly used it has to go
    # through Init, Ready To Receive, Ready To Send. A good docs on QP state machine: https://www.rdmamojo.com/2012/05/05/qp-state-machine/

    # INIT
    qp_access_flags = ib.IBV_ACCESS_REMOTE_WRITE | ib.IBV_ACCESS_REMOTE_READ
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_INIT, port_num=DEFAULT_PORT, qp_access_flags=qp_access_flags)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_PORT | ib.IBV_QP_ACCESS_FLAGS | ib.IBV_QP_PKEY_INDEX))

    self.gid, self.qp_num = bytes(self.ctx.gid_attr.raw), self.qp.contents.qp_num

  # Exchange GID and QP num with remote. At least in RoCEv2 gid can be guessed from remote's ip, QP num can't.

  def connect(self, remote_gid:bytes, remote_qp_num:int):
    # RTR
    qp_ah_attr_grh = ib.struct_ibv_global_route(hop_limit=1, dgid=ib.union_ibv_gid(raw=(ctypes.c_ubyte * 16)(*remote_gid)), sgid_index=DEFAULT_GID)
    qp_ah_attr = ib.struct_ibv_ah_attr(is_global=1, port_num=DEFAULT_PORT, grh=qp_ah_attr_grh)
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_RTR, path_mtu=ib.IBV_MTU_4096, dest_qp_num=remote_qp_num, rq_psn=0, max_dest_rd_atomic=1,
                                min_rnr_timer=12, ah_attr=qp_ah_attr)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_PATH_MTU | ib.IBV_QP_DEST_QPN | ib.IBV_QP_RQ_PSN | \
                                          ib.IBV_QP_MAX_DEST_RD_ATOMIC | ib.IBV_QP_MIN_RNR_TIMER | ib.IBV_QP_AV))

    # RTS
    qpa = ib.struct_ibv_qp_attr(qp_state=ib.IBV_QPS_RTS, timeout=14, retry_cnt=7, rnr_retry=7, sq_psn=0, max_rd_atomic=1)
    checkz(ib.ibv_modify_qp(self.qp, qpa, ib.IBV_QP_STATE | ib.IBV_QP_TIMEOUT | ib.IBV_QP_RETRY_CNT | ib.IBV_QP_RNR_RETRY | ib.IBV_QP_SQ_PSN | \
                                          ib.IBV_QP_MAX_QP_RD_ATOMIC))

  def __del__(self):
    self.wait_cq() # need to wait for **everything** to complete before it's safe to dealloc queues and stuff
    ib.ibv_destroy_qp(self.qp)
    ib.ibv_destroy_cq(self.cq)
    ib.ibv_destroy_comp_channel(self.comp_channel)

  def next_wrid(self):
    self.pending_wrids.add(wrid:=next(self.wrid_num))
    return wrid

  def wait_cq(self, wr_id: int|None=None):
    while (wr_id in self.pending_wrids) if wr_id is not None else self.pending_wrids:
      if self.ctx.ctx.contents.ops.poll_cq(self.cq, _num_entries:=1, ctypes.byref(wc:=ib.struct_ibv_wc())):
        if wc.status != ib.IBV_WC_SUCCESS:
          raise RuntimeError(f'Work Request completed with error: wr_id={wc.wr_id} status={ib.ibv_wc_status__enumvalues.get(wc.status, wc.status)}')
        self.pending_wrids.remove(wc.wr_id)

  # def recv(self, size:int):
  #   assert size <= 0x1000
  #   wrid = self.next_wrid()
  #   sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
  #   rwr = ib.struct_ibv_recv_wr(wr_id=wrid, sg_list=ctypes.pointer(sgl), num_sge=1)
  #   checkz(self.ctx.ctx.contents.ops.post_recv(self.qp, rwr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_recv_wr)())))
  #   self.wait_cq(wrid)
  #   return self.ctx.hmv[:size]

  # def send(self, msg:bytes):
  #   assert len(msg) <= 0x1000
  #   self.ctx.hmv[:len(msg)] = msg
  #   wrid = self.next_wrid()
  #   sgl = ib.struct_ibv_sge(addr=mv_address(self.ctx.hmv), length=len(self.ctx.hmv), lkey=self.ctx.hmr.contents.lkey)
  #   swr = ib.struct_ibv_send_wr(wr_id=wrid, sg_list=ctypes.pointer(sgl), num_sge=1, opcode=ib.IBV_WR_SEND, send_flags=ib.IBV_SEND_SIGNALED)
  #   checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.pointer(ctypes.POINTER(ib.struct_ibv_send_wr)())))
  #   self.wait_cq(wrid)

  # TODO: amd slow read issues might or might not be resolved by writing through multiple QPs in parallel

  def rdma_write(self, dst_iova: int, src_iova: int, size:int, dst_key:int, src_key:int):
    swr, wr_id = None, self.next_wrid()
    # Message size limit (max 2GB per ib spec, 1GB on tinybox mellanoxes) applies to both scatter-gather entries and entire wrs
    for off in reversed(range(0, size, self.ctx.port_attr.max_msg_sz) or [0]): # create at least one if zero-sized
      # Scatter-Gather Entry for local memory
      sge = ctypes.pointer(ib.struct_ibv_sge(addr=src_iova+off, length=min(size-off, self.ctx.port_attr.max_msg_sz), lkey=src_key))
      # RDMA struct for remote memory
      wr = ib.union_ibv_send_wr_wr(rdma=ib.struct_ibv_send_wr_1_rdma(remote_addr=dst_iova+off, rkey=dst_key))
      # Signal (with chosen work request id) if it's the last wr (first in the loop since it goes from end to start for elegant chaining)
      cwrid, flags = (wr_id, ib.IBV_SEND_SIGNALED) if swr is None else (0, 0)
      # Create Send Request
      swr = ctypes.pointer(ib.struct_ibv_send_wr(opcode=ib.IBV_WR_RDMA_WRITE, sg_list=sge, num_sge=1, wr=wr, wr_id=cwrid, send_flags=flags, next=swr))
    # The swr can be freed when this returns, the memory that sge points to can be unmapped after work completion is retrieved from cq
    checkz(self.ctx.ctx.contents.ops.post_send(self.qp, swr, ctypes.byref(_bad_wr:=ctypes.POINTER(ib.struct_ibv_send_wr)())))
    # TODO: async
    self.wait_cq(wr_id)
