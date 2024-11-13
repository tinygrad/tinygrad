from tinygrad.tensor import Tensor

from tinygrad.tensor import Tensor
a = Tensor([1,2])
b = Tensor([3,4])
res = a.dot(b).numpy()
print(res)
''' 
extra/dsp/run.py:15: error: Argument 1 to "append" of "list" has incompatible type "tuple[Any, Any]"; expected "str"  [arg-type]
extra/dsp/run.py:109: error: Module has no attribute "memcpy"  [attr-defined]
extra/dsp/run.py:110: error: Module has no attribute "memcpy"  [attr-defined]
extra/dsp/run.py:113: error: Incompatible types in assignment (expression has type "CDLL", variable has type Module)  [assignment]
extra/dsp/run.py:114: error: Module has no attribute "ioctl"  [attr-defined]
tinygrad/dtype.py:53: error: Argument 1 to "PtrDType" has incompatible type "*tuple[int | PtrDType | Any, ...]"; expected "int"  [arg-type]
tinygrad/dtype.py:53: error: Argument 1 to "PtrDType" has incompatible type "*tuple[int | PtrDType | Any, ...]"; expected "str"  [arg-type]
tinygrad/dtype.py:53: error: Argument 1 to "PtrDType" has incompatible type "*tuple[int | PtrDType | Any, ...]"; expected "str | None"  [arg-type]
tinygrad/dtype.py:53: error: Argument 1 to "PtrDType" has incompatible type "*tuple[int | PtrDType | Any, ...]"; expected "DType | None"  [arg-type]
tinygrad/dtype.py:53: error: Argument 1 to "PtrDType" has incompatible type "*tuple[int | PtrDType | Any, ...]"; expected "DType"  [arg-type]
tinygrad/dtype.py:53: error: Argument 1 to "PtrDType" has incompatible type "*tuple[int | PtrDType | Any, ...]"; expected "bool"  [arg-type]
tinygrad/runtime/ops_python.py:71: error: No overload variant of "cast" of "memoryview" matches argument type "str"  [call-overload]
tinygrad/runtime/ops_python.py:71: note: Possible overload variants:
tinygrad/runtime/ops_python.py:71: note:     def cast(self, format: Literal['c', '@c'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bytes]
tinygrad/runtime/ops_python.py:71: note:     def cast(self, format: Literal['f', '@f', 'd', '@d'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[float]
tinygrad/runtime/ops_python.py:71: note:     def cast(self, format: Literal['?'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bool]
tinygrad/runtime/ops_python.py:71: note:     def cast(self, format: Literal['b', 'B', '@b', '@B', 'h', 'H', '@h', '@H', 'i', 'I', '@i', '@I', 'l', 'L', '@l', '@L', 'q', 'Q', '@q', '@Q', 'P', '@P'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[int]
tinygrad/runtime/ops_python.py:75: error: No overload variant of "cast" of "memoryview" matches argument type "str"  [call-overload]
tinygrad/runtime/ops_python.py:75: note: Possible overload variants:
tinygrad/runtime/ops_python.py:75: note:     def cast(self, format: Literal['c', '@c'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bytes]
tinygrad/runtime/ops_python.py:75: note:     def cast(self, format: Literal['f', '@f', 'd', '@d'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[float]
tinygrad/runtime/ops_python.py:75: note:     def cast(self, format: Literal['?'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bool]
tinygrad/runtime/ops_python.py:75: note:     def cast(self, format: Literal['b', 'B', '@b', '@B', 'h', 'H', '@h', '@H', 'i', 'I', '@i', '@I', 'l', 'L', '@l', '@L', 'q', 'Q', '@q', '@Q', 'P', '@P'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[int]
tinygrad/multi.py:110: error: Argument 3 to "to_sharded" has incompatible type "Any | None"; expected "tuple[tuple[int, int], ...]"  [arg-type]
tinygrad/multi.py:111: error: Argument 2 to "to_sharded" has incompatible type "Any | None"; expected "int"  [arg-type]
tinygrad/multi.py:111: error: Argument 3 to "to_sharded" has incompatible type "Any | None"; expected "tuple[tuple[int, int], ...]"  [arg-type]
tinygrad/tensor.py:280: error: No overload variant of "cast" of "memoryview" matches argument types "str", "tuple[int, ...]"  [call-overload]
tinygrad/tensor.py:280: note: Possible overload variants:
tinygrad/tensor.py:280: note:     def cast(self, format: Literal['c', '@c'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bytes]
tinygrad/tensor.py:280: note:     def cast(self, format: Literal['f', '@f', 'd', '@d'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[float]
tinygrad/tensor.py:280: note:     def cast(self, format: Literal['?'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bool]
tinygrad/tensor.py:280: note:     def cast(self, format: Literal['b', 'B', '@b', '@B', 'h', 'H', '@h', '@H', 'i', 'I', '@i', '@I', 'l', 'L', '@l', '@L', 'q', 'Q', '@q', '@Q', 'P', '@P'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[int]
tinygrad/tensor.py:293: error: No overload variant of "cast" of "memoryview" matches argument type "str"  [call-overload]
tinygrad/tensor.py:293: note: Possible overload variants:
tinygrad/tensor.py:293: note:     def cast(self, format: Literal['c', '@c'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bytes]
tinygrad/tensor.py:293: note:     def cast(self, format: Literal['f', '@f', 'd', '@d'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[float]
tinygrad/tensor.py:293: note:     def cast(self, format: Literal['?'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[bool]
tinygrad/tensor.py:293: note:     def cast(self, format: Literal['b', 'B', '@b', '@B', 'h', 'H', '@h', '@H', 'i', 'I', '@i', '@I', 'l', 'L', '@l', '@L', 'q', 'Q', '@q', '@Q', 'P', '@P'], shape: list[int] | tuple[int, ...] = ...) -> memoryview[int]
test/external/fuzz_schedule.py:6: error: Module "tinygrad.helpers" has no attribute "MULTIOUTPUT"  [attr-defined]
test/external/fuzz_schedule.py:8: error: Module "tinygrad.engine.schedule" has no attribute "LBScheduleItem"; maybe "ScheduleItem"?  [attr-defined]
test/external/fuzz_schedule.py:8: error: Module "tinygrad.engine.schedule" has no attribute "_graph_schedule"  [attr-defined]
test/external/fuzz_schedule.py:41: error: Missing positional argument "assign_preloads" in call to "ScheduleItem"  [call-arg]
test/external/fuzz_schedule.py:63: error: Missing positional argument "assign_preloads" in call to "ScheduleItem"  [call-arg]
tinygrad/nn/state.py:19: error: Incompatible return value type (got "tuple[Tensor | Any, float | int | Any, Any]", expected "tuple[Tensor, int, Any]")  [return-value]
tinygrad/nn/state.py:19: error: Slice index must be an integer, SupportsIndex or None  [misc]
extra/mcts_search.py:133: error: Argument 3 to "_time_program" has incompatible type "dict[UOp, float | int]"; expected "dict[UOp, int]"  [arg-type]
tinygrad/runtime/graph/hcq.py:55: error: Invalid index type "Any | Compiled" for "dict[HCQCompiled, HWComputeQueue]"; expected type "HCQCompiled"  [index]
tinygrad/runtime/graph/hcq.py:55: error: Argument 1 to "setdefault" of "MutableMapping" has incompatible type "Any | Compiled"; expected "HCQCompiled"  [arg-type]
tinygrad/runtime/graph/hcq.py:55: error: Item "Compiled" of "Any | Compiled" has no attribute "hw_copy_queue_t"  [union-attr]
tinygrad/runtime/graph/hcq.py:56: error: Item "Compiled" of "Any | Compiled" has no attribute "signal_t"  [union-attr]
tinygrad/runtime/graph/hcq.py:85: error: Incompatible types in assignment (expression has type "tuple[Any | Compiled, HWComputeQueue | HWCopyQueue, list[tuple[HCQSignal, int]], list[tuple[HCQSignal, Any]], HCQSignal, int | None]", target has type "tuple[HCQCompiled, HWCommandQueue, list[Any], list[Any], HCQSignal, int | None]")  [assignment]
tinygrad/runtime/graph/hcq.py:97: error: Argument 1 to "append" of "list" has incompatible type "tuple[tuple[int, bool], tuple[int, bool], Any | Compiled, Any | str, bool, list[int], dict[str, object] | None]"; expected "tuple[tuple[int, bool], tuple[int, bool], HCQCompiled, str, bool, list[int], dict[Any, Any] | None]"  [arg-type]
tinygrad/runtime/graph/hcq.py:111: error: Incompatible types in assignment (expression has type "HWCommandQueue", variable has type "HWComputeQueue | HWCopyQueue")  [assignment]
extra/mockgpu/nv/nvdriver.py:94: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:112: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:126: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:142: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:146: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:148: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:167: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:173: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
extra/mockgpu/nv/nvdriver.py:179: error: Argument 1 to "from_address" of "_CData" has incompatible type "Any | None"; expected "int"  [arg-type]
tinygrad/nn/__init__.py:261: error: Need type annotation for "weight"  [var-annotated]
tinygrad/nn/__init__.py:261: error: Need type annotation for "bias"  [var-annotated]
tinygrad/nn/__init__.py:343: error: Need type annotation for "bias_ih"  [var-annotated]
tinygrad/nn/__init__.py:343: error: Need type annotation for "bias_hh"  [var-annotated]
Found 46 errors in 11 files (checked 73 source files)

'''