# RDNA4 instructions that are not in the autogen
from extra.assembly.amd.autogen.rdna4.ins import *
from extra.assembly.amd.dsl import RawImm, NULL

def ds_load_b64(vdst, addr, offset1=0):
  return DS(op=DSOp.DS_LOAD_B64, vdst=vdst, addr=addr, offset0=0, offset1=offset1)
def ds_load_b128(vdst, addr, offset0=0, offset1=0):
  return DS(op=DSOp.DS_LOAD_B128, vdst=vdst, addr=addr, offset0=offset0, offset1=offset1)
def ds_store_b32(addr, data0, offset1=0):
  return DS(op=DSOp.DS_STORE_B32, addr=addr, data0=data0, offset0=0, offset1=offset1)
def ds_store_b64(addr, data0, offset1=0):
  return DS(op=DSOp.DS_STORE_B64, addr=addr, data0=data0, offset0=0, offset1=offset1)
def ds_store_b128(addr, data0, offset0=0, offset1=0):
  return DS(op=DSOp.DS_STORE_B128, addr=addr, data0=data0, offset0=offset0, offset1=offset1)
def buffer_load_b32(vdata, vaddr, rsrc):
  return VBUFFER(op=VBUFFEROp.BUFFER_LOAD_B32, vdata=vdata, vaddr=vaddr, rsrc=RawImm(rsrc), soffset=NULL, offen=1, format=1)
def buffer_load_b128(vdata, vaddr, rsrc):
  return VBUFFER(op=VBUFFEROp.BUFFER_LOAD_B128, vdata=vdata, vaddr=vaddr, rsrc=RawImm(rsrc), soffset=NULL, offen=1, format=1)
def buffer_store_b64(vdata, vaddr, rsrc):
  return VBUFFER(op=VBUFFEROp.BUFFER_STORE_B64, vdata=vdata, vaddr=vaddr, rsrc=rsrc, soffset=NULL, offen=1, format=1)
def s_barrier_signal():
  return SOP1(op=SOP1Op.S_BARRIER_SIGNAL, sdst=RawImm(0), ssrc0=RawImm(193))
# VOP3P WMMA with correct defaults (opsel_hi=3, opsel_hi2=1 -> combined opsel_hi=7)
def v_wmma_f32_16x16x16_f16(vdst, src0, src1, src2):
  return VOP3P(op=VOP3POp.V_WMMA_F32_16X16X16_F16, vdst=vdst, src0=src0, src1=src1, src2=src2, opsel_hi=3, opsel_hi2=1)
