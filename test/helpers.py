import sys, unittest
import numpy as np
from tinygrad import Tensor, Device, dtypes
from tinygrad.codegen.uops import UOp
from tinygrad.tensor import _to_np_dtype
from tinygrad.engine.realize import Runner
from tinygrad.dtype import DType
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import Context, CI, OSX, getenv
from typing import Callable, List, Optional
from tinygrad.codegen.uops import UOpGraph, UOps, UOp, constant_folder, PatternMatcher, UPat
from collections import defaultdict

def derandomize_model(model):
  with Context(GRAPH=0):
    for p in get_parameters(model):
      p.lazydata = Tensor.empty(p.shape, device=p.device, dtype=p.dtype).lazydata
      p.realize()

def assert_jit_cache_len(fxn, expected_len):
  assert len(fxn.jit_cache) > 0
  # until we have a better way of typing the prg in ExecItem
  if issubclass(type(fxn.jit_cache[0].prg), Runner) and not type(fxn.jit_cache[0].prg).__name__.endswith('Graph'):
    assert len(fxn.jit_cache) == expected_len
  else:
    assert len(fxn.jit_cache) == 1
    # until we have a better way of typing the prg in ExecItem
    assert type(fxn.jit_cache[0].prg).__name__.endswith('Graph')
    assert len(fxn.jit_cache[0].prg.jit_cache) == expected_len

def is_dtype_supported(dtype: DType, device: str = Device.DEFAULT):
  if dtype == dtypes.bfloat16:
    # NOTE: this requires bf16 buffer support
    return device in {"AMD"} or (device in {"CUDA", "NV"} and not CI and not getenv("PTX"))
  if device in ["WEBGPU", "WEBGL"]: return dtype in [dtypes.float, dtypes.int32, dtypes.uint32]
  # for CI GPU and OSX, cl_khr_fp16 isn't supported
  # for CI LLVM, it segfaults because it can't link to the casting function
  # CUDACPU architecture is sm_35 but we need at least sm_70 to run fp16 ALUs
  # PYTHON supports half memoryview in 3.12+ https://github.com/python/cpython/issues/90751
  if dtype == dtypes.half:
    if device == "GPU": return not CI and not OSX
    if device in ["LLVM", "CUDA", "NV"]: return not CI
    if device == "PYTHON": return sys.version_info >= (3, 12)
  if dtype == dtypes.float64: return device != "METAL" and not (OSX and device == "GPU")
  return True

def rand_for_dtype(dt:DType, size:int):
  if dtypes.is_unsigned(dt):
    return np.random.randint(0, 100, size=size, dtype=_to_np_dtype(dt))
  elif dtypes.is_int(dt):
    return np.random.randint(-100, 100, size=size, dtype=_to_np_dtype(dt))
  elif dt == dtypes.bool:
    return np.random.choice([True, False], size=size)
  return np.random.uniform(-10, 10, size=size).astype(_to_np_dtype(dt))

class NodeVisitCounter:
  def __init__(self) -> None:
    self.counter = defaultdict(int)

  def __getitem__(self, key: UOp) -> int:
    return self.counter[key]

  def __setitem__(self, key: UOp, value: int) -> None:
    self.counter[key] = value
  
  @property
  def total(self):
    return sum(self.counter.values())

  def __repr__(self):
    display = "Visit Counter\n"
    for k,v in self.counter.items():
      display += f"   {k.op}: {v}\n"
    display += f"Total: {self.total}\n"
    return display


  def colored_count(self, uop: UOp) -> str:
    count = self[uop]
    if count == 0: return ""
    if count < 5: return f"\033[32m<{count}>\033[0m"
    if count < 10: return f"\033[33m<{count}>\033[0m"
    return f"\033[31m<{count}>\033[0m"


def print_uop_tree(uop: UOp, node_visit_counter: Optional[NodeVisitCounter] = None, _print=True) -> str:
  """
  Args:
    uop (UOp): the uop to print
    node_visit_counter (NodeVisitCounter): the node visit counter to use
    _print (bool): whether to print the result to screen
  """
  printable = ''
  def recursively_print(uop: UOp, depth=0, has_siblings=False, last_in_siblings=False, parents_branch: List[bool] = []):
    """
    Args:
      has_siblings (bool): whether the current node has more than one child
      last_in_siblings (bool): whether the current node is the last child of its parent
      parents_branch (List[bool]): whether the current line should display a branch symbol ┃ at the earlier depth position for the boolean index position
    """
    nonlocal printable
    uop_name = f"{uop.op}"
    uop_name += f" {uop.arg}"
    depth_fill = '  '
    depth_fill_if_parent_has_siblings = '┃ '
    depth_space = ''
    for i in range(depth):
      if parents_branch[i]:
        depth_space += depth_fill_if_parent_has_siblings
      else:
        depth_space += depth_fill
    branch_symbol_at_this_level = '┗' if (not has_siblings) or last_in_siblings else '┣'
    visit_count = node_visit_counter.colored_count(uop) if node_visit_counter else ''
    printable += f"{depth_space}{branch_symbol_at_this_level}{uop_name}{visit_count}\n"
    for i, src in enumerate(uop.src):
      has_more_than_one_child = len(uop.src) > 1
      _last_in_siblings = i == len(uop.src) - 1
      this_branch = has_siblings and not last_in_siblings
      recursively_print(src, depth + 1, has_more_than_one_child, _last_in_siblings, parents_branch + [this_branch])
  recursively_print(uop)
  printable += '\n'
  if _print:
    print(printable)
  return printable

def compare_uop_tree(uop1: UOp, uop2: UOp):
  def recursively_compare(uop1: UOp, uop2: UOp):
    if uop1.op != uop2.op:
      return False, f"op mismatch: {uop1.op} != {uop2.op}"
    if uop1.dtype != uop2.dtype:
      return False, f"dtype mismatch: {uop1.dtype} != {uop2.dtype}"
    if uop1.arg != uop2.arg:
      return False, f"arg mismatch: {uop1.arg} != {uop2.arg}"
    if len(uop1.src) != len(uop2.src):
      return False, f"src length mismatch: {len(uop1.src)} != {len(uop2.src)}"
    for s1, s2 in zip(uop1.src, uop2.src):
      result, reason = recursively_compare(s1, s2)
      if not result: return False, reason
    return True, ''
  return recursively_compare(uop1, uop2)

def create_uop_linearize_and_compare_bottomup_topdown(uop_factory: Callable[[], UOp]):
  def create_graph():
    uop_output = uop_factory()
    graph = UOpGraph([uop_output])
    graph.nodes = {}
    return uop_output, graph
  uop1, graph1 = create_graph()
  uop2, graph2 = create_graph()
  uop1_rewritten = graph1.graph_rewrite(uop1, constant_folder)
  uop2_rewritten = graph2.graph_rewrite_bottomup_no_backtrack(uop2, constant_folder)
  result, reason = compare_uop_tree(uop1_rewritten, uop2_rewritten)
  reason_and_tree = reason + '\nUOp1 (topdown): \n' + print_uop_tree(uop1_rewritten, _print=False)
  reason_and_tree += 'UOp2 (bottomup): \n' + print_uop_tree(uop2_rewritten, _print=False)
  return result, reason_and_tree

def create_compare_old_pattern_topdown_vs_new_pattern_bottomup(
  old_pattern_matcher: PatternMatcher, 
  new_pattern_matcher: PatternMatcher, 
  uop_factory: Callable[[], UOp]
):
  def create_graph():
    uop_output = uop_factory()
    graph = UOpGraph([uop_output])
    graph.nodes = {}
    return uop_output, graph
  old_uop, old_graph = create_graph()
  old_uop_rewritten = old_graph.graph_rewrite(old_uop, old_pattern_matcher)
  new_uop, new_graph = create_graph()
  new_uop_rewritten = new_graph.graph_rewrite_bottomup_no_backtrack(new_uop, new_pattern_matcher)
  result, reason = compare_uop_tree(old_uop_rewritten, new_uop_rewritten)
  reason_and_tree = reason + '\nUOp1 (topdown, old patterns): \n' + print_uop_tree(old_uop_rewritten, _print=False)
  reason_and_tree += 'UOp2 (bottomup, new Patterns): \n' + print_uop_tree(new_uop_rewritten, _print=False)
  return result, reason_and_tree

class TestUOps(unittest.TestCase):
  def setup_and_assert_uop_graph_equal(self, uop_factory: Callable[[], UOp]):
    result, reason = create_uop_linearize_and_compare_bottomup_topdown(uop_factory)
    self.assertTrue(result, reason)

  def assert_equiv_uops(self, uop1:UOp, uop2:UOp):
    # NOTE: direct UOps __eq__ is comparing object reference, use this function to compare two uops
    self.assertIs(uop1.op, uop2.op)
    self.assertEqual(uop1.dtype, uop2.dtype)
    self.assertEqual(uop1.arg, uop2.arg)
    self.assertEqual(len(uop1.src), len(uop2.src))
    for s1, s2 in zip(uop1.src, uop2.src): self.assert_equiv_uops(s1, s2)

  def setup_and_assert_old_pattern_topdown_vs_new_pattern_bottomup(
      self, 
      old_pattern_matcher: PatternMatcher, 
      new_pattern_matcher: PatternMatcher, 
      uop_factory: Callable[[], UOp]):
    result, reason = create_compare_old_pattern_topdown_vs_new_pattern_bottomup(
      old_pattern_matcher,
      new_pattern_matcher,
      uop_factory
    )
    self.assertTrue(result, reason)