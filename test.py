from tinygrad import dtypes
from tinygrad.codegen.rewriter import get_late_rewrite_patterns, pm_render, full_graph_rewrite
from tinygrad.ops import UOp, Ops, symbolic_simple, graph_rewrite
from tinygrad.codegen.transcendental import xexp2


def test1():
    # this UOp is created after the transcendental rewrite rules replace exp2
    # this UOp is passed into xexp2
    d = UOp(Ops.MUL, dtypes.float.vec(3), arg=None, src=(
    UOp(Ops.VECTORIZE, dtypes.float.vec(3), arg=None, src=(
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
        UOp(Ops.INDEX, dtypes.float.ptr(56448), arg=None, src=(
            x3:=UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(56448), arg=1, src=()),
            x4:=UOp(Ops.ADD, dtypes.int, arg=None, src=(
            UOp(Ops.MUL, dtypes.int, arg=None, src=(
                UOp(Ops.RANGE, dtypes.int, arg=0, src=(
                x7:=UOp(Ops.CONST, dtypes.int, arg=0, src=()),
                UOp(Ops.CONST, dtypes.int, arg=384, src=()),)),
                UOp(Ops.CONST, dtypes.int, arg=147, src=()),)),
            UOp(Ops.RANGE, dtypes.int, arg=1, src=(
                x7,
                x11:=UOp(Ops.CONST, dtypes.int, arg=49, src=()),)),)),)),)),
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
        UOp(Ops.INDEX, dtypes.float.ptr(56448), arg=None, src=(
            x3,
            UOp(Ops.ADD, dtypes.int, arg=None, src=(
            x4,
            x11,)),)),)),
        UOp(Ops.LOAD, dtypes.float, arg=None, src=(
        UOp(Ops.INDEX, dtypes.float.ptr(56448), arg=None, src=(
            x3,
            UOp(Ops.ADD, dtypes.int, arg=None, src=(
            x4,
            UOp(Ops.CONST, dtypes.int, arg=98, src=()),)),)),)),)),
    UOp(Ops.VECTORIZE, dtypes.float.vec(3), arg=None, src=(
        x20:=UOp(Ops.CONST, dtypes.float, arg=-1.4426950408889634, src=()),
        x20,
        x20,)),))

    out = xexp2(d)
    print(out)


def test2():
    sink = UOp(Ops.EXP2, dtypes.float.vec(3), arg=None, src=(UOp(Ops.CONST, dtypes.float.vec(3), arg=3),))
    sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(())+pm_render)
    print(sink)

    sink = UOp(Ops.LOG2, dtypes.float.vec(3), arg=None, src=(UOp(Ops.CONST, dtypes.float.vec(3), arg=3),))
    sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(())+pm_render)
    print(sink)

    sink = UOp(Ops.SIN, dtypes.float.vec(3), arg=None, src=(UOp(Ops.CONST, dtypes.float.vec(3), arg=3),))
    sink = graph_rewrite(sink, symbolic_simple+get_late_rewrite_patterns(())+pm_render)
    print(sink)

def test3():
    sink = UOp(Ops.EXP2, dtypes.float.vec(3), arg=None, src=(UOp(Ops.CONST, dtypes.float.vec(3), arg=3),)).sink()
    sink = full_graph_rewrite(sink).src[0]
    print(sink)


if __name__ == '__main__':
    test1()
    test2()
    test3()
