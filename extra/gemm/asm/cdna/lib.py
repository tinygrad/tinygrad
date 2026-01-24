import pathlib
from extra.gemm.asm.cdna.gemm import asm_gemm_kernel

_, lib = asm_gemm_kernel(4096).to_asm()
print(len(lib))
with open(pathlib.Path(__file__).parent/"lib", "wb") as f: f.write(lib)
