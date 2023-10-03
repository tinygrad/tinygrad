from typing import Callable, List, Tuple, Any, Dict, cast, Union, Optional
from collections import defaultdict
import functools, itertools
from tinygrad.helpers import DEBUG, DType, merge_dicts
from tinygrad.ops import RawBuffer, Device, BasicBatchExecutor
from tinygrad.tensor import Tensor
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

JIT_SUPPORTED_DEVICE = ["GPU", "CLANG", "METAL", "CUDA", "HIP", "WEBGPU", "LLVM"]

class TinyJit:
  def __init__(self, fxn:Callable):
    self.fxn: Callable = fxn
    self.cnt: int = 0
    self.jit_cache: List[Tuple[Any, List[Optional[RawBuffer]], Dict[Variable, int]]] = []
    self.ret: Any = None
    self.input_replace: Dict[Tuple[int, int], Tuple[Union[int, str], ShapeTracker, DType]]= {}   # (kernel_number, buffer_number) -> (input_name, expected_shapetracker, expected_type)
    self.batch_executor: Any = None
    self.updatable_entries: Dict[int, List[int]] = defaultdict(list) # (kernel_number) -> list(argument id). These are buffers from input + variables.

    self.input_repl = []
    self.compiled_cache = []

  # add support for instance methods
  def __get__(self, obj, objtype): return functools.partial(self.__call__, obj)

  def __call__(self, *args, **kwargs) -> Any:
    if Device.DEFAULT not in JIT_SUPPORTED_DEVICE: return self.fxn(*args, **kwargs)  # only jit on supported device
    # NOTE: this cast is needed since although we know realize will create a ".realized" RawBuffer, the type checker doesn't
    input_rawbuffers: Dict[Union[int, str], Tuple[RawBuffer, ShapeTracker]] = {cast(Union[int, str], k):v.realize().lazydata for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
    assert len(input_rawbuffers) != 0, "no inputs to JIT"
    assert len(set(input_rawbuffers.values())) == len(input_rawbuffers), "duplicate inputs to JIT"
    if self.cnt >= 2:
      # try: var_vals: Dict[Variable, int] = kwargs["jit_ctx"]
      # except KeyError: var_vals = merge_dicts([arg.lazydata.var_vals for arg in args if arg.__class__ is Tensor])
      
      # if len(var_vals) > 1: var_vals = dict(sorted(var_vals.items(), key=lambda kv: kv[0].key))

      self.input_repl = {k:v.realize().lazydata for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
      for k,lb in input_rawbuffers.items():
        self.input_repl[k].realized = lb.realize().realized

      # for (j,i),(input_name, expected_st, expected_type) in self.input_replace.items():
      #   assert input_rawbuffers[input_name][0].dtype == expected_type, f"type mismatch in JIT, {input_rawbuffers[input_name][0].dtype} != {expected_type}"
      #   # NOTE: if we pass jit_ctx instead of using reshape to update the var_vals, we cannot compare the shapetracker directly
      #   if "jit_ctx" not in kwargs: assert input_rawbuffers[input_name][1].views == expected_st.views, f"ShapeTracker.views mismatch in JIT, {input_rawbuffers[input_name][1].views} != {expected_st.views}"
      #   self.jit_cache[j][1][i] = input_rawbuffers[input_name][0]

      # for j in self.updatable_entries.keys():
      #   for k in self.jit_cache[j][2].keys():
      #     try: self.jit_cache[j][2][k] = var_vals[k]
      #     except KeyError: pass

      from tinygrad.realize import run_compiled
      run_compiled([x for x in self.compiled_cache])

      # self.batch_executor.exec(self.jit_cache, self.updatable_entries)
      # for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 1:
      
      # New jit starts here
      SchedCollector.start()
      self.ret = self.fxn(*args, **kwargs)
      self.compiled_cache = SchedCollector.finish()
      
      # Compiled is some LB info and other.
      # from tinygrad.realize import run_compiled
      # run_compiled(self.compiled_cache)
      self.input_repl = {k:v.realize().lazydata for k,v in itertools.chain(enumerate(args), kwargs.items()) if v.__class__ is Tensor}
      # for out,prog,buffers in self.compiled_cache:
        # for buf in bu
      
      # CacheCollector.start()
      # self.ret = self.fxn(*args, **kwargs)
      # self.jit_cache = CacheCollector.finish()
      # assert len(self.jit_cache) != 0, "didn't JIT anything!"
      if DEBUG >= 1: print(f"JIT captured {len(self.jit_cache)} kernels with {len(input_rawbuffers)} inputs")

      # get the inputs for replacement
      # for j_,cache in enumerate(self.jit_cache): # type: Tuple[int, Tuple[Callable, List[Optional[RawBuffer]], Dict[Variable, int]]]
      #   for i,a in enumerate(cache[1]):
      #     if a in [v[0] for v in input_rawbuffers.values()]:
      #       self.input_replace[(j_,i)] = [(k, v[1], v[0].dtype) for k,v in input_rawbuffers.items() if v[0] == a][0]
      #       self.updatable_entries[j_].append(i)
      #   for i in range(len(cache[2])): self.updatable_entries[j_].append(len(cache[1])+i)
        #if prg.local_size is None: prg.local_size = prg.optimize_local_size(args, preserve_output=True)  # the JIT can optimize local
      # assert set([x[0] for x in self.input_replace.values()]) == set(input_rawbuffers.keys()), "some input tensors not found"
      # self.batch_executor = self.jit_cache[0][0].batch_exec(self.jit_cache) if hasattr(self.jit_cache[0][0], 'batch_exec') else BasicBatchExecutor(self.jit_cache)
      # for (j,i) in self.input_replace.keys(): self.jit_cache[j][1][i] = None
    elif self.cnt == 0:
      self.ret = self.fxn(*args, **kwargs)
    self.cnt += 1
    return self.ret

class _SchedCollector:
  def __init__(self): self.cache = None
  def start(self): self.cache = []
  def add(self, cc): self.cache.extend(cc) if self.cache is not None else None
  def finish(self):
    cache_result, self.cache = self.cache, None
    return cache_result
SchedCollector = _SchedCollector()

class _CacheCollector:
  def __init__(self): self.cache = None
  def start(self): self.cache = []
  def add(self, prg, rawbufs, var_vals): self.cache.append((prg, rawbufs, var_vals)) if self.cache is not None else None
  def finish(self):
    cache_result, self.cache = self.cache, None
    return cache_result
CacheCollector = _CacheCollector()
