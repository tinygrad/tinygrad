from __future__ import annotations
import platform, itertools
import numpy as np
import pyopencl as cl  # type: ignore
from typing import Optional, List
from tinygrad.helpers import DEBUG, getenv, prod, ImageDType
from tinygrad.ops import Compiled
from tinygrad.runtime.lib import RawBufferCopyInOut
from tinygrad.codegen.cstyle import CStyleCodegen, CStyleLanguage

OSX = platform.system() == "Darwin"
OSX_TIMING_RATIO = (125/3) if OSX else 1.0   # see test/external_osx_profiling.py to determine this ratio. it's in like GPU clocks or something
FLOAT16 = getenv("FLOAT16", 0)

class _CL:
  def __init__(self):
    devices: List[cl.Device] = sum([x.get_devices(device_type=cl.device_type.GPU) for x in cl.get_platforms()], [])
    if len(devices) == 0: devices = sum([x.get_devices(device_type=cl.device_type.CPU) for x in cl.get_platforms()], []) # settle for CPU
    if len(devices) > 1 or DEBUG >= 1: print(f"using {devices[getenv('CL_DEVICE', 0)]}")
    self.cl_ctx: cl.Context = cl.Context(devices=[devices[getenv("CL_DEVICE", 0)]])
    self.cl_queue: cl.CommandQueue = cl.CommandQueue(self.cl_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)  # this is an in-order command queue
CL = _CL()

# TODO: merge CLImage in here
class CLBuffer(RawBufferCopyInOut):
  def __init__(self, size, dtype):
    if isinstance(dtype, ImageDType):
      fmt = cl.ImageFormat(cl.channel_order.RGBA, {2: cl.channel_type.HALF_FLOAT, 4: cl.channel_type.FLOAT}[dtype.itemsize])
      buf = cl.Image(CL.cl_ctx, cl.mem_flags.READ_WRITE, fmt, shape=(dtype.shape[1], dtype.shape[0]))
      assert size == prod(dtype.shape), f"image size mismatch {size} != {dtype.shape}"
      # NOTE: the memory is a bit off here due to padding, it's buf.row_pitch * buf.height * 4 * dtype.itemsize
    else:
      buf = cl.Buffer(CL.cl_ctx, cl.mem_flags.READ_WRITE, size * dtype.itemsize)
    super().__init__(size, dtype, buf)
  def _copyin(self, x:np.ndarray):
    assert not self.dtype.name.startswith("image"), f"can't copyin images {self.dtype}"
    cl.enqueue_copy(CL.cl_queue, self._buf, x, is_blocking=False)
  def _copyout(self, x:np.ndarray):
    assert not self.dtype.name.startswith("image"), f"can't copyout images {self.dtype}"
    cl.enqueue_copy(CL.cl_queue, x, self._buf, is_blocking=True)

class CLProgram:
  def __init__(self, name:str, prg:str, binary=False, argdtypes=None):
    self.name, self.argdtypes, self.clprogram = name, argdtypes, cl.Program(CL.cl_ctx, CL.cl_ctx.devices, [prg]) if binary else cl.Program(CL.cl_ctx, prg)  # type: ignore
    try:
      self._clprg = self.clprogram.build()
    except cl.RuntimeError as e:
      if DEBUG >= 3: print("FAILED TO BUILD", prg)
      raise e
    self.clprg = self._clprg.__getattr__(name)
    if DEBUG >= 5 and not OSX:
      binary = self.clprogram.get_info(cl.program_info.BINARIES)[0]
      if 'Adreno' in CL.cl_ctx.devices[0].name:
        from disassemblers.adreno import disasm
        disasm(binary)
      else:
        # print the PTX for NVIDIA. TODO: probably broken for everything else
        print(binary.decode('utf-8'))
    if self.argdtypes is not None: self.clprg.set_scalar_arg_dtypes(self.argdtypes)

  @staticmethod
  def max_work_group_size(): return CL.cl_ctx.devices[0].max_work_group_size

  def __call__(self, global_size, local_size, *bufs, wait=False) -> Optional[float]:
    e = self.clprg(CL.cl_queue, global_size, local_size, *[x._buf if isinstance(x, CLBuffer) else x for x in bufs])
    if wait:
      CL.cl_queue.finish()
      return ((e.profile.end - e.profile.start) * OSX_TIMING_RATIO) * 1e-9
    return None

class CLCodegen(CStyleCodegen):
  lang = CStyleLanguage(
    kernel_prefix = "__kernel", buffer_prefix = "__global ", smem_prefix = "__local ",
    half_prekernel = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable",
    barrier = "barrier(CLK_LOCAL_MEM_FENCE);", float4 = "(float4)",
    gid = [f'get_global_id({i})' for i in range(3)], lid = [f'get_local_id({i})' for i in range(3)], uses_vload=True)

  # ******************** GPU simplifiers ********************

  def required_optimizations(self, early_only=False):
    for buf_index,buf in enumerate(self.bufs):
      upcast_strides = [self.sts[buf_index].strides[i] for i in self.upcast_in_mid_reduce_axes]
      if (not early_only or buf in self.earlybufs) and isinstance(self.bufs[buf_index].dtype, ImageDType) and not (self.can_float4(buf_index) or (buf not in self.earlybufs and (1 in upcast_strides))):
        axes = [i for i,x in enumerate(self.sts[buf_index].strides) if x == 1]
        assert len(axes) == 1, f"wrong number of stride 1 axis : {axes} on buf_index {buf_index}, {self.sts[buf_index]}"
        assert self.sts[buf_index].shape[axes[0]]%4 == 0, f"axis:{axes[0]} in buffer {buf_index} is not a multiple of 4, {self.sts[buf_index].shape}"
        self.shift_to(axes[0], 4)
        self.upcast()
        assert self.can_float4(buf_index)

  def hand_coded_optimizations(self):
    # if there's images in the earlybufs, we have to make an axis the 4 loading one
    self.required_optimizations(early_only=True)

    # simplify (sets first_reduce)
    self.simplify_ones()

    # are we grouping? (requires local shape support)
    if not self.can_float4(0) and self.first_reduce <= 2 and self.first_reduce + 1 <= self.shape_len and prod(self.sts[0].shape[:self.first_reduce]) <= 2048:
      # TODO: use 1024 if it's allowed in a smarter way
      for sz in (([256, 16]) if prod(self.sts[0].shape[:self.first_reduce]) <= 32 else [16]):
        if all([st.shape[self.first_reduce] % sz == 0 or st.shape[self.first_reduce] == 1 for st in self.sts]):
          self.shift_to(self.first_reduce, sz, top=True, insert_before=self.first_reduce + len(self.group_for_reduce))
          self.group_for_reduce.append(sz)
          break

    # are we upcasting in mid reduce? (only for images)
    if self.bufs[0].dtype.name.startswith('image') and not self.can_float4(0) and self.group_for_reduce and self.first_reduce <= 2 and prod(self.sts[0].shape) > 1:
      axes = [i for i,x in enumerate(self.sts[0].strides) if x == 1]
      assert len(axes) == 1, f"wrong number of stride 1 axis : {axes}"
      if self.sts[0].shape[axes[0]]%4 == 0:
        self.upcast_in_mid_reduce_axes.append(self.first_reduce + len(self.group_for_reduce))
        self.shift_to(axes[0], 4, insert_before=self.first_reduce + len(self.group_for_reduce))   # insert at the end of the grouped axis
        self.group_for_reduce.append(4)

    # now do everything required
    self.required_optimizations()

    # simplify (sets first_reduce)
    self.simplify_ones()

    # use more opencl indexing if the output buffer is an image and we have room
    if self.bufs[0].dtype.name.startswith('image') and self.first_reduce+len(self.group_for_reduce) < 3:
      base_shape = self.bufs[0].dtype.shape
      if (base_shape[0]*base_shape[1]) % self.sts[0].shape[0] == 0 and self.sts[0].shape[0]//base_shape[0] != 0:
        if DEBUG >= 4: print("split opencl", base_shape, self.sts[0].shape)
        self.reshape_and_permute(lambda x: [base_shape[0], x[0]//base_shape[0]]+list(x[1:]), None)
        self.simplify_ones()

    # no more opt if we are grouping
    if self.group_for_reduce: return

    # **** below this line need to be optional and benchmarked ****

    # potentially do more upcasts of non reduce axes based on a heuristic
    while prod(self.sts[0].shape[:self.first_reduce]) >= 1024:
      xb_choices = []
      for axis, upcast_amount in itertools.product(range(self.first_reduce), [3,4]):   # consider all the non reduce axes, and a 3 or 4 reduce
        # if it mods, and some buffer has stride 0 on axis while having no stride 0 in the buftoken
        if self.full_shape[axis]%upcast_amount == 0 and any(self.sts[buf_index].strides[axis] == 0 and not any(x[1] == 0 for x in self.upcasted_axis(buf_index)) for buf_index in range(len(self.sts))):
          xb_choices.append((sum(st.strides[axis]>0 for st in self.sts), sum(st.strides[axis] for st in self.sts), axis, upcast_amount))
      if len(xb_choices):
        xb_choices = sorted(xb_choices)
        if DEBUG >= 4: print(f"float4 merging axis : {xb_choices}")
        self.shift_to(xb_choices[0][2], amount=xb_choices[0][3])
        self.upcast()
        self.simplify_ones()
      else:
        break

    # if last dim <= 5 and it's a reduce dim, upcast the reduce (loop unrolling). no simplify needed since it's just an upcast. NOTE: careful, this has broken VALIDHACKS
    if self.first_reduce < (self.shape_len-self.upcasted) and self.full_unupcasted_shape[-1] <= 5 and (len(self.offsets(self.full_buf_index)) <= 4 or not any(r for _,_,r in self.upcasted_axis(self.full_buf_index))):
      self.upcast()

GPUBuffer = Compiled(CLBuffer, CLCodegen, CLProgram, CL.cl_queue.finish)
