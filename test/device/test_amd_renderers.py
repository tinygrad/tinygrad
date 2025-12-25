import unittest
from tinygrad import Device
from tinygrad.device import CompileError
if Device.DEFAULT == "AMD":
  from tinygrad.renderer.cstyle import AMDJITRenderer, AMDCCRenderer, HIPJITRenderer
  from tinygrad.renderer.llvmir import AMDLLVMJITRenderer

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDJITRenderer(unittest.TestCase):
  def test_renderer_has_compiler(self):
    renderer = AMDJITRenderer(Device[Device.DEFAULT].arch)
    self.assertIsNotNone(renderer.compiler)
    self.assertEqual(renderer.arch, Device[Device.DEFAULT].arch)
    self.assertEqual(renderer.device, "AMD")

  def test_compile(self):
    renderer = AMDJITRenderer(Device[Device.DEFAULT].arch)
    src = '''extern "C" __attribute__((global)) void __attribute__((amdgpu_flat_work_group_size(1, 1))) test() {}'''
    lib = renderer.compiler.compile(src)
    self.assertIsInstance(lib, bytes)
    self.assertGreater(len(lib), 0)

  def test_compile_error(self):
    renderer = AMDJITRenderer(Device[Device.DEFAULT].arch)
    with self.assertRaises(CompileError):
      renderer.compiler.compile("invalid code")

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDCCRenderer(unittest.TestCase):
  def test_renderer_has_compiler(self):
    renderer = AMDCCRenderer(Device[Device.DEFAULT].arch)
    self.assertIsNotNone(renderer.compiler)
    self.assertEqual(renderer.arch, Device[Device.DEFAULT].arch)
    self.assertEqual(renderer.device, "AMD")

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestHIPJITRenderer(unittest.TestCase):
  def test_renderer_has_compiler(self):
    renderer = HIPJITRenderer(Device[Device.DEFAULT].arch)
    self.assertIsNotNone(renderer.compiler)
    self.assertEqual(renderer.arch, Device[Device.DEFAULT].arch)
    self.assertEqual(renderer.device, "HIP")

@unittest.skipUnless(Device.DEFAULT == "AMD", "Runs only on AMD")
class TestAMDLLVMJITRenderer(unittest.TestCase):
  def test_renderer_has_compiler(self):
    renderer = AMDLLVMJITRenderer(Device[Device.DEFAULT].arch)
    self.assertIsNotNone(renderer.compiler)
    self.assertEqual(renderer.arch, Device[Device.DEFAULT].arch)

  def test_compile(self):
    renderer = AMDLLVMJITRenderer(Device[Device.DEFAULT].arch)
    src = '''
; https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/imm.ll
define amdgpu_kernel void @i64_imm_inline_lo(ptr addrspace(1) %out) {
entry:
  store i64 1311768464867721221, ptr addrspace(1) %out ; 0x1234567800000005
  ret void
}
    '''
    lib = renderer.compiler.compile(src)
    self.assertIsInstance(lib, bytes)
    self.assertGreater(len(lib), 0)

  def test_compile_error(self):
    renderer = AMDLLVMJITRenderer(Device[Device.DEFAULT].arch)
    src = """
@local_temp0 = internal unnamed_addr addrspace(3) global [{N} x float*] undef, align 16
define amdgpu_kernel void @test(float* noalias align 32 %data0, half* noalias align 32 %data1, float* noalias align 32 %data2) #0
{{
  %local_temp0 = addrspacecast [{N} x float*] addrspace(3)* @local_temp0 to [{N} x float*]*
  %v178 = getelementptr inbounds float, float* %local_temp0, i32 1
  %v133 = getelementptr inbounds float, float* %data2, i32 1
  %v134 = load float, float* %v133
  store float %v134, float* %v178
  ret void
}}
"""
    # Valid compile
    renderer.compiler.compile(src.format(N=65536//8))
    # Exceeds local memory limit
    with self.assertRaises(CompileError):
      renderer.compiler.compile(src.format(N=65536//8+1))

if __name__ == '__main__':
  unittest.main()
