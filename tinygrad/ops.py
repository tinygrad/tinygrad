from __future__ import annotations
import functools
import time
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Union,
    Type,
    Tuple,
    Any,
    List,
    Optional,
    Dict,
    Callable,
    cast,
)
from tinygrad.helpers import ansilen, prod, DEBUG, getenv, GlobalCounters, DType, colored
from tinygrad.shape.shapetracker import MovementOps
from tinygrad.runtime.lib import RawBuffer, RawConst

if TYPE_CHECKING:
    from tinygrad.lazy import LazyBuffer

Op = Union[Enum, Type[Enum]]
OpType = Type[Enum]

class LazyOp:
    __slots__ = ("op", "src", "arg", "buffers")
    op: Op
    src: Tuple[Union[LazyOp, LazyBuffer], ...]
    arg: Any
    buffers: Tuple[LazyBuffer, ...]

    def __init__(
        self, op: Op, src: Tuple[Union[LazyOp, LazyBuffer], ...], arg: Any = None
    ) -> None:
        self.op = op
        self.src = src
        self.arg = arg
        self.buffers = tuple(y for x in src for y in x.buffers) if hasattr(src, "buffers") else ()

    def __repr__(self) -> str:
        return f"LazyOp(op={self.op}, src={self.src}, arg={self.arg})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LazyOp):
            return False
        return (
            self.op == other.op and
            self.src == other.src and
            self.arg == other.arg
        )

    def __hash__(self) -> int:
        return hash((self.op, self.src, self.arg))

    @property
    def key(self) -> Tuple[Any, ...]:
        return (
            self.op,
            tuple(x.key if hasattr(x, "key") else x for x in self.src),
            self.arg.key if hasattr(self.arg, "key") else self.arg,
        )

    def map_buffers(self, real_srcs: Dict[Any, Any]) -> LazyOp:
        return LazyOp(
            self.op,
            tuple(y.map_buffers(real_srcs) if hasattr(y, "map_buffers") else y for y in self.src),
            self.arg,
        )

    def get_lazyops(self) -> List[LazyOp]:
        lazyops = [self]
        for item in self.src:
            if isinstance(item, LazyOp):
                lazyops.extend(item.get_lazyops())
        return lazyops

class Interpreted:
    def __init__(
        self,
        buffer: Type[Any],
        fxn_for_op: Dict[Op, Callable],
        from_lazybuffer: Callable = lambda x: x.realized,
        to_underlying: Callable = lambda x: x._buf,
        from_underlying: Optional[Callable] = None,
    ) -> None:
        self.buffer = buffer
        self.fxn_for_op = fxn_for_op
        self.from_lazybuffer = from_lazybuffer
        self.from_underlying = buffer if from_underlying is None else from_underlying
        self.to_underlying = to_underlying
        self.synchronize = lambda: None
        self.codegen = None

    def exec_ast(self, ast: LazyOp, output=None, context=None, **kwargs):
        if (
            isinstance(ast.src[0], LazyOp)
            and ast.op == FusedOps.MULACC
            and ast.src[0].op == BinaryOps.MUL
        ):
            ast = LazyOp(FusedOps.MULACC, cast(LazyOp, ast.src[0]).src, ast.arg)
        created_context = context is None
        if context is None:
            context = dict()
        if not created_context and ast in context:
            return context[ast]
        srcs = [
            self.exec_ast(cast(LazyOp, x), context=context, **kwargs)
            if isinstance(x, LazyOp)
            else self.from_lazybuffer(x)
            for x in ast.src
        ]
        ret = self.from_underlying(
            self.fxn_for_op[ast.op](*[self.to_underlying(x) for x in srcs], ast.arg)
        )
        if not created_context:
            context[ast] = ret
        if output is not None and output.output_buffer is not None:
            assert (
                output.output_buffer.size == ret.size
                and output.output_buffer.dtype == ret.dtype
            )
            output.output_buffer._buf = ret._buf
            return output.output_buffer
        else:
            return ret

class FlopCounter:
    def __init__(self, tup: Tuple[Tuple[int, ...], DType, int]) -> None:
        self.shape, self.dtype, self.flops, self._buf = tup

    def consume_flops(self) -> int:
        ret = self.flops
        self.flops = 0
        return ret

def get_lazyop_info(ast: LazyOp) -> FlopCounter:
    from tinygrad.lazy import elementwise_op

    shape_fxn_for_op = {
        UnaryOps.CAST: lambda self, dtype: (
            self.shape,
            dtype,
            self.consume_flops(),
        ),
        **{
            op: lambda self: (
                self.shape,
                self.dtype,
                self.consume_flops() + prod(self.shape),
            )
            for op in UnaryOps
            if op != UnaryOps.CAST
        },
        **{
            op: lambda self, y: (
                self.shape,
                max(self.dtype, y.dtype),
                self.consume_flops() + y.consume_flops() + prod(self.shape),
            )
            for op in BinaryOps
        },
        **{
            op: functools.partial(
                lambda mop, self, arg: (
                    ShapeTracker(self.shape)
                    .movement_op(mop, arg)
                    .shape,
                    self.dtype,
                    self.consume_flops(),
                ),
                op,
            )
            for op in MovementOps
        },
    }

    interpreted_flop_counter = Interpreted(
        FlopCounter,
        shape_fxn_for_op,
        lambda x: FlopCounter((x.shape, x.dtype, 0)),
        lambda x: x,
    )

    return interpreted_flop_counter.exec_ast(ast)

class ASTRunner:
    def __init__(
        self,
        name: str,
        prg: str,
        global_size: Optional[List[int]] = None,
        local_size: Optional[List[int]] = None,
        op_estimate: int = 0,
        mem_estimate: int = 0,
        display_name: Optional[str] = None,
        runtime_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if DEBUG >= 4 and (runtime_args is None or 'binary' not in runtime_args):
            print(prg)
        self.name = name
        self.prg = prg
        self.global_size = global_size
        self.local_size = local_size
        self.op_estimate = op_estimate
        self.mem_estimate = mem_estimate
        self.display_name = display_name
        self.runtime_args = runtime_args if runtime_args is not None else {}

    def build(self, runtime) -> ASTRunner:
        self.clprg = runtime(self.name, self.prg, **self.runtime_args)
        return self

    def exec(self, bufs: List[RawBuffer]) -> Optional[float]:
        rawbufs = [
            x.realized for x in bufs if x.realized is not None and x.realized.__class__ is not RawConst
        ]
        if GlobalCounters.cache is not None:
            GlobalCounters.cache.append((self, rawbufs))
        return self(rawbufs)

    def __call__(
        self, rawbufs: List[RawBuffer], jit: bool = False, force_wait: bool = False
    ) -> Optional[float]:
        if et := self.clprg(
            (self.global_size + [1] * (3 - len(self.global_size)))
            if self.global_size is not None
            else None,
            (self.local_size + [1] * (3 - len(self.local_size)))
            if self.local_size is not None
            else None,
            *rawbufs,
            wait=force_wait or DEBUG >= 1,
        ):
            GlobalCounters.time_sum_s += et
        if DEBUG >= 2:
            print(
                f"{colored(f'*** {GlobalCounters.kernel_count:4d}', 'magenta' if jit else None)} {(self.display_name+' '*(29-ansilen(self.display_name))) if self.display_name is not None else self.name:26s} arg {len(rawbufs):3d} sz {str(self.global_size):18s} {str(self.local_size):12s} OPs {int(self.op_estimate/1e6):6d}M/{GlobalCounters.global_ops/1e9:7.2f}G  mem {GlobalCounters.mem_used/1e9:5.2f} GB "
                + (
                    str()
                    if et is None
                    else f"tm {et*1e6:9.2f}us/{GlobalCounters.time_sum_s*1e3:9.2f}ms ({self.op_estimate/((et or 1e-20)*1e9):8.2f} GFLOPS, {self.mem_estimate/((et or 1e-20)*1e9):7.2f} GB/s)"
                )
            )
        GlobalCounters.kernel_count += 1
        GlobalCounters.global_ops += self.op_estimate
        GlobalCounters.global_mem += self.mem_estimate
        if getenv("EARLY_STOPPING") and GlobalCounters.kernel_count == getenv("EARLY_STOPPING"):
            exit(0)
        return et

class Compiled:
    def __init__(
        self,
        buffer: Type[RawBuffer],
        codegen,
        runtime,
        synchronize: Callable = lambda: None,
    ) -> None:
        self.buffer = buffer
        self.codegen = codegen
        self.runtime = runtime
        self.synchronize = synchronize
        self.method_cache: Dict[Tuple[Any, ...], ASTRunner] = {}

    def exec_ast(self, ast: LazyOp, output: Any, **kwargs) -> Any:
        if ast.op in MovementOps and isinstance(ast.src[0], LazyOp) and ast.src[0].realized:
            return ast.src[0].realized

        output.realized = output.output_buffer
        if output.realized:
            if output.realized.__class__ is RawConst:
                output.realized = None
            for a in ast.buffers:
                if a.realized == output.realized and not a.st.contiguous:
                    output.realized = None
                    break

        if not output.realized:
            output.realized = self.buffer(prod(output.shape), output.dtype, **kwargs)

        k = self.codegen(ast, output)

        if hasattr(k, "key") and getenv("ENABLE_METHOD_CACHE", 1):
            if k.key not in self.method_cache:
                self.method_cache[k.key] = k.codegen().build(self.runtime)
            elif DEBUG >= 5:
                print(f"method cache hit : {k.key}")
            prg = self.method_cache[k.key]
        else:
            prg = k.codegen().build(self.runtime)

        if prg.name == getenv("PRINT_PRG", ""):
            print(prg.prg)

        prg.exec(k.bufs)
        return output.realized


