import pathlib, sys
import os
import os, ctypes, pathlib, re, fcntl, functools, mmap, struct, tempfile, hashlib, subprocess, time, array
import numpy
from tinygrad.helpers import to_mv, from_mv
from tinygrad.helpers import init_c_var, to_char_p_p, from_mv, OSX, DEBUG, getenv

# if getenv("IOCTL"): import extra.qcom_gpu_driver.opencl_ioctl # noqa: F401

def ptr(buff):
  return ctypes.addressof(from_mv(memoryview(buff)))

SHADER = b'Q\x00\x00\x00\x01@$0P\x00\x00\x00\x00@$ \x00\x00\x00\x00\x02@U \x00\x00\x00\x00\x12@D \x00\x00\x00@\x13@D \x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x04 \x03\x000B\x02\x00\x08 \x04\x000B\x02\x00\x0c \x05\x000B\x02\x00\x10 \x06\x000B\x02\x00\x14 \x07\x000B\x02\x00\x18 \x08\x000B\x02\x00\x1c \n\x000B\x02\x00  \x02\x000B\x03\x00\x00\x00\x11@\x15 \x04\x00\x00\x00\x0f@\x15 \x05\x00\x00\x00\r@\x15 \x06\x00\x00\x00\x0b@\x15 \x07\x00\x00\x00\t@\x15 \x08\x00\x00\x00\x07@\x15 \n\x00\x00\x00\x05@\x15 \x02\x00\x00\x00\x03@\x15 P\x10\x11\x00\x02\x00\x10BP\x10\x0f\x00\x04\x00\x10BP\x10\r\x00\x06\x00\x10BP\x10\x0b\x00\x08\x00\x10BP\x10\t\x00\n\x00\x10BP\x10\x07\x00\x0c\x00\x10BP\x10\x05\x00\x0e\x00\x10BP\x10\x03\x00\x10\x00\x10B\x02\x00P\x10\x14\x00\x90B\x11\x00\x1f \x11\x00\xf0F\x04\x00P\x10\x15\x00\x90B\x0f\x00\x1f \x0f\x00\xf0F\x06\x00P\x10\x16\x00\x90B\r\x00\x1f \r\x00\xf0F\x08\x00P\x10\x17\x00\x90B\x0b\x00\x1f \x0b\x00\xf0F\n\x00P\x10\x18\x00\x90B\t\x00\x1f \x19\x00\xf0F\x0c\x00P\x10\x1a\x00\x90B\x07\x00\x1f \x1b\x00\xf0F\x0e\x00P\x10\x1c\x00\x90B\x05\x00\x1f \x1d\x00\xf0F\x10\x00P\x10\x1e\x00\x90B\x03\x00\x1f \x1f\x00\xf0FQ\x10\x14@\x03\x80\x88gQ\x10\x15@\x05\x80\x87gQ\x10\x16@\x07\x80\x86gQ\x10\x17@\t\x80\x85gQ\x10\x18@\x0b\x80\x8cgQ\x10\x1a@\r\x80\x8dgQ\x10\x1c@\x0f\x80\x8egQ\x10\x1e@\x11\x80\x8fg&\x00\x80\x01\x00\x01\xc2\xc0$\x00\x80\x01\x00\x05\xc2\xc0$\x00\x80\x01\x00\t\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x80\x01\x00\r\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x00&\x00\x80\x01\x00\x11\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x80\x01\x00\x15\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x80\x01\x00\x19\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x00$\x00\x80\x01\x00\x1d\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x00&\x00\x80\x01\x00!\xc2\xc0\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
CONSTANTS = bytes((ctypes.c_ubyte * 512) ())


def patch_constants(offset: int, value: int):
  packed_value = struct.pack('<Q', value)
  return CONSTANTS[:offset] + packed_value + CONSTANTS[offset + len(packed_value):]

from extra.qcom_gpu_driver import msm_kgsl
def ioctls_from_header():
  hdr = (pathlib.Path(__file__).parent.parent.parent / "extra/qcom_gpu_driver/msm_kgsl.h").read_text().replace("\\\n", "")
  pattern = r'#define\s+(IOCTL_KGSL_[A-Z0-9_]+)\s+_IOWR?\(KGSL_IOC_TYPE,\s+(0x[0-9a-fA-F]+),\s+struct\s([A-Za-z0-9_]+)\)'
  matches = re.findall(pattern, hdr, re.MULTILINE)
  return {int(nr, 0x10):(name, getattr(msm_kgsl, "struct_"+sname)) for name, nr, sname in matches}

nrs = ioctls_from_header()

def parity(val: int):
  val ^= val >> 16
  val ^= val >> 8
  val ^= val >> 4
  val &= 0xf
  return (~0x6996 >> val) & 1

def pkt7_hdr(opcode: int, cnt: int):
  return 0x70000000 | cnt & 0x3FFF | parity(cnt) << 15 | (opcode & 0x7F) << 16 | parity(opcode) << 23

def pkt4_hdr(reg: int, cnt: int):
   return 0x40000000 | cnt & 0x3FFF | parity(cnt) << 7 | (reg & 0x3FFFF) << 8 | parity(reg) << 27

def cp_load_state6_frag(dstoff, typ, src, block):
  return dstoff & 0x3FFF | typ 


class CMDBuffer():
  def __init__(self):
    self.q = []

  def push(self, opcode=None, reg=None, vals = []):
    if opcode: self.q += [pkt7_hdr(opcode, len(vals)), *vals]
    if reg: self.q += [pkt4_hdr(reg, len(vals)), *vals]
    return self

  def get(self):
    return array.array('I', self.q)

class Qcom():
  def __init__(self):
    self.fd = os.open('/dev/kgsl-3d0', os.O_RDWR)

  def _ioctl(self, nr, arg):
    ret = fcntl.ioctl(self.fd, (3 << 30) | (ctypes.sizeof(arg) & 0x1FFF) << 16 | 0x9 << 8 | (nr & 0xFF), arg)
    if ret != 0: raise RuntimeError(f"ioctl returned {ret}")
    return ret
  
  def alloc(self, size: int, flags: int, mapped:bool=False, id:bool=False):
    gpuobj_alloc = msm_kgsl.struct_kgsl_gpuobj_alloc(size=size, flags=flags)
    self._ioctl(0x45, gpuobj_alloc) # IOCTL_KGSL_GPUOBJ_ALLOC
    if not gpuobj_alloc.mmapsize:
      raise RuntimeError(f"failed to alloc, mmap_size={gpuobj_alloc.mmapsize}")
    gpuobj_info = msm_kgsl.struct_kgsl_gpuobj_info(id=gpuobj_alloc.id)
    self._ioctl(0x47, gpuobj_info) # IOCTL_KGSL_GPUOBJ_INFO
    if not mapped:
      return gpuobj_info.id, gpuobj_info.gpuaddr
    return gpuobj_info.id, gpuobj_info.gpuaddr, mmap.mmap(self.fd, gpuobj_alloc.mmapsize, offset=gpuobj_info.id * 0x1000)

  def submit_cmds(self, ctx_id: int, *cmds: CMDBuffer):
    objs = (msm_kgsl.struct_kgsl_command_object * len(cmds))()
    for i, cmd in enumerate(cmds):
      cmdbytes = cmd.get()
      _, addr, buff = self.alloc((len(cmdbytes) * 4 + 4096 - 1) // 4096 * 4096, 0xC0A00, mapped=True)
      ctypes.memmove(ptr(buff), ptr(cmdbytes), len(cmdbytes) * 4)
      objs[i].gpuaddr = addr
      objs[i].size = len(cmdbytes) * 4
      objs[i].flags = 0x00000001

    submit_req = msm_kgsl.struct_kgsl_gpu_command()
    submit_req.flags = 0x0 # 0x11
    submit_req.cmdlist = ctypes.addressof(objs)
    submit_req.cmdsize = ctypes.sizeof(msm_kgsl.struct_kgsl_command_object)
    submit_req.numcmds = len(cmds)
    submit_req.context_id = ctx_id

    self._ioctl(0x4A, submit_req)

  def test(self):
    print('****** create eye Tensor and copy it to the GPU ******')
    _, tensor_gpuaddr, tensor_buff = self.alloc(0x144000, 0xC0C0E00, mapped=True)
    matrix_bytes = bytearray(numpy.eye(576, dtype=numpy.float32).flatten().tobytes())
    ctypes.memmove(ptr(tensor_buff), ptr(matrix_bytes), len(matrix_bytes))

    print('****** initial values ******')
    print(f"values = {struct.unpack('f' * 9, tensor_buff[0:9*4])}")

    print('****** alloc and fill shader ******')
    _, shader_gpuaddr, shader_buff = self.alloc(0x1000, 0x10C0A00, mapped=True)
    ctypes.memmove(ptr(shader_buff), ptr(bytearray(SHADER)), len(SHADER))

    print('****** alloc and fill consts ******')
    _, consts_gpuaddr, consts_buff = self.alloc(0x100000, 0xC0A00, mapped=True)
    CONSTANTS = patch_constants(0x140, tensor_gpuaddr)
    ctypes.memmove(ptr(consts_buff), ptr(bytearray(CONSTANTS)), len(CONSTANTS))

    print('****** fill cmd buff 1 ******')
    # Setting up GPU for the compute shader mode...
    buff1 = CMDBuffer()
    buff1.push(opcode=0x26) # CP_WAIT_FOR_IDLE
    buff1.push(opcode=0x65, vals=[0x8]) # CP_SET_MARKER = RM6_COMPUTE
    buff1.push(reg=0xb983, vals=[0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc]) # HLSQ_CONTROL_2_REG
    buff1.push(reg=0xbb08, vals=[0x60]) # HLSQ_INVALIDATE_CMD
    buff1.push(reg=0xbb08, vals=[0]) # HLSQ_INVALIDATE_CMD
    buff1.push(reg=0xa9ba, vals=[0x80]) # SP_CS_TEX_COUNT
    buff1.push(reg=0xa9c0, vals=[0]) # unknown
    buff1.push(reg=0xa9d0, vals=[0x250000,5,0x240000,5]) #unknown
    buff1.push(reg=0xaa00, vals=[0x40]) #SP_CS_IBO_COUNT
    buff1.push(reg=0xaa31, vals=[0x0]) #unknown
    buff1.push(reg=0xab00, vals=[0x2]) # ISAMMODE = ISAMMODE_CL
    buff1.push(opcode=0x26) # CP_WAIT_FOR_IDLE
    buff1.push(reg=0xae0f, vals=[0x3f]) # SP_PERFCTR_ENABLE
    buff1.push(reg=0xb309, vals=[9]) # SP_TP_MODE_CNTL=ISAMMODE|UNK3
    buff1.push(opcode=0x26) # CP_WAIT_FOR_IDLE
    buff1.push(reg=0xb600, vals=[0]) # TPL1_DBG_ECO_CNTL

    print('****** fill cmd buff 2 ******')
    # Preparing GPU for the specific shader...
    _, private_gpuaddr = self.alloc(0x202000, 0xC0F00)
    cr = msm_kgsl.struct_kgsl_drawctxt_create(flags=(2<<20) | 0x10 | 0x2)
    self._ioctl(0x13, cr)
    context_id = cr.drawctxt_id

    buff2 = CMDBuffer()
    # self.cmdbuff.push(opcode=0x3e, vals=[0x40000980,0x23e028,5]) # CP_REG_TO_MEM, reg=2432 cnt=0 b64=1 accum=0 dest=0x50023e028
    # self.cmdbuff.push(opcode=0x46, vals=[0x31]) # CP_EVENT_WRITE7
    buff2.push(reg=0x26) # CP_WAIT_FOR_IDLE
    buff2.push(reg=0x26) # CP_WAIT_FOR_IDLE
    buff2.push(reg=0xe12, vals=[0x10000000])
    buff2.push(reg=0xb987, vals=[0x140]) # HLSQ_CS_CNTL
    buff2.push(reg=0xb990, vals=[0xf01f,0x48,0,0x480,0,1,0,0xccc0cf,0x2fc,9,0x48,1]) # local size, global size, dimension
    buff2.push(reg=0x26) # CP_WAIT_FOR_IDLE
    buff2.push(reg=0xae03, vals=[0x20]) # SP_CHICKEN_BITS
    buff2.push(reg=0xa9b0, vals=[0x100402, 0x41, 0, 0, shader_gpuaddr & 0xffffffff, (shader_gpuaddr >> 32) & 0xffffffff, 0, private_gpuaddr & 0xffffffff, (private_gpuaddr >> 32) & 0xffffffff, 0x101])
    buff2.push(reg=0xa9bb, vals=[0x100]) # SP_CS_CONFIG
    buff2.push(reg=0xa9bd, vals=[0]) # SP_CS_PVT_MEM_HW_STACK_OFFSET
    buff2.push(opcode=0x34, vals=[0x40364000, consts_gpuaddr & 0xffffffff, (consts_gpuaddr >> 32) & 0xffffffff]) # CP_LOAD_STATE6_FRAG, load constants
    buff2.push(opcode=0x34, vals=[0xf60000,shader_gpuaddr & 0xffffffff, (shader_gpuaddr >> 32) & 0xffffffff]) # CP_LOAD_STATE6_FRAG, load shader
    buff2.push(reg=0xa9bc, vals=[13]) # SP_CS_INSTRLEN
    buff2.push(opcode=0x31, vals=[0]) # CP_RUN_OPENCL
    # self.cmdbuff.push(opcode=0x46, vals=[4,0x460000,5,1]) # CP_EVENT_WRITE7
    # self.cmdbuff.push(opcode=0x46, vals=[0x31]) # CP_EVENT_WRITE7
    buff2.push(opcode=0x26) # CP_WAIT_FOR_IDLE
    # self.cmdbuff.push(opcode=0x3e, vals=[0x40000980,0x23e030,5]) # CP_REG_TO_MEM, reg=2432 cnt=0 b64=1 accum=0 dest=0x50023e030

    print('****** submit commands to GPU ******')
    self.submit_cmds(context_id, buff1, buff2)

    print('****** result values ******')
    time.sleep(2)
    print(f"values = {struct.unpack('f' * 9, tensor_buff[0:9*4])}")


if __name__ == '__main__':
  qcom = Qcom()
  qcom.test()
