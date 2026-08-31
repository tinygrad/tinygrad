from test.mockgpu.nv.emu import ptx_run  # noqa: F401

class PythonRemu:
  """Python RDNA3/RDNA4 emulator wrapper used by mockgpu."""
  valid_mem_ranges: set[tuple[int, int]] = set()
  rsrc2: int = 0x19c  # Default: USER_SGPR_COUNT=14, enable X and Y workgroup IDs
  scratch_size: int = 0  # private_segment_fixed_size from kernel descriptor
  arch: str = "rdna3"  # Architecture: rdna3 or rdna4
  user_data: list[int] = []  # All COMPUTE_USER_DATA registers (loaded into s[0:N])

  def run_asm(self, lib: int, lib_sz: int, gx: int, gy: int, gz: int, lx: int, ly: int, lz: int, args_ptr: int) -> int:
    from test.mockgpu.amd.emu import run_asm
    return run_asm(lib, lib_sz, gx, gy, gz, lx, ly, lz, args_ptr, self.rsrc2, self.scratch_size, self.arch, self.user_data)
