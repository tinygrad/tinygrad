from typing import List, Optional, Tuple
from tinygrad.codegen.linearizer import UOp

def uops_to_llvm_ir(function_name:str, uops:List[UOp]) -> Tuple[str, Optional[List[int]], Optional[List[int]]]:
  return "", [], []