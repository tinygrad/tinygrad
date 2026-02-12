import math, pathlib
from tinygrad import Device, Tensor
from tinygrad.helpers import Context
from tinygrad.runtime.support.compiler_amd import HIPCCCompiler
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.autogen import libc
import numpy as np

if __name__ == "__main__":
  code = (pathlib.Path(__file__).parent / "fa_fwd_causal.cpp").read_text()
  device = Device["AMD"]
  kitten_args = [f"-I{(pathlib.Path(__file__).parent / 'include').as_posix()}", "-std=c++20"]
  lib = HIPCCCompiler(device.compiler.arch, kitten_args).compile(code)

  # extract kernel name from ELF symbol table
  _, sections, _ = elf_loader(lib)
  symtab_sh = next(sh for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB)
  strtab_sh = sections[symtab_sh.header.sh_link]
  syms = (libc.Elf64_Sym * (symtab_sh.header.sh_size // symtab_sh.header.sh_entsize)).from_buffer_copy(symtab_sh.content)
  kernel_name = next(strtab_sh.content[s.st_name:strtab_sh.content.index(b'\x00', s.st_name)].decode()
                     for s in syms if libc.ELF64_ST_TYPE(s.st_info) == libc.STT_FUNC and s.st_name != 0)
  print("kernel name", kernel_name)

  prg = device.runtime(kernel_name, lib)

  dynamic_smem = 160000
  prg.group_segment_size = max(prg.group_segment_size, dynamic_smem)
  lds_size = ((prg.group_segment_size + 511) // 512) & 0x1FF
  prg.rsrc2 = (prg.rsrc2 & ~(0x1FF << 15)) | (lds_size << 15)

  B, N, H, H_KV, D = 16, 8192, 32, 8, 128
  q = Tensor.randn(B, N, H, D, device="AMD", dtype="bfloat16").contiguous()
  k = Tensor.randn(B, N, H_KV, D, device="AMD", dtype="bfloat16").contiguous()
  v = Tensor.randn(B, N, H_KV, D, device="AMD", dtype="bfloat16").contiguous()
  out = Tensor.empty(B, N, H, D, device="AMD", dtype="bfloat16").contiguous()
  L_vec = Tensor.empty(B, H, 1, N, device="AMD", dtype="float32").contiguous()
  Tensor.realize(q, k, v, out, L_vec)

  Q_BLOCK_SIZE = 32
  NUM_WARPS = 8
  NUM_THREADS = 64 * NUM_WARPS

  gsz = (H, (math.ceil((N // Q_BLOCK_SIZE) / NUM_WARPS)), B)
  lsz = (NUM_THREADS, 1, 1)
  for _ in range(5):
    et = prg(out.uop.buffer.ensure_allocated()._buf, q.uop.buffer._buf, k.uop.buffer._buf, v.uop.buffer._buf, L_vec.uop.buffer.ensure_allocated()._buf,
             global_size=gsz, local_size=lsz, wait=True)

    attn_flops = 2 * B * H * N * N * D + \
                 4 * B * H * N * N + \
                 2 * B * H * N * N * D
    print(f"{attn_flops/(et*1e9):2f} GFLOPS")

  with Context(DEBUG=2):
    ref = q.transpose(1,2).scaled_dot_product_attention(k.transpose(1,2), v.transpose(1,2), is_causal=True, enable_gqa=True).transpose(1,2)

  ref_np, out_np = ref.float().numpy(), out.float().numpy()
  np.testing.assert_allclose(ref_np, out_np, atol=2e-2, rtol=1e-2)
