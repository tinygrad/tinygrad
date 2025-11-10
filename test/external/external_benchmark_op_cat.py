# ruff: noqa: E501 E712
from tinygrad import dtypes, Device
from tinygrad.uop.ops import UOp, AxisType, Ops, KernelInfo
from tinygrad.codegen import full_rewrite
from tinygrad.renderer import ProgramSpec
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import dedup
from tinygrad.device import Buffer
from tinygrad.dtype import ImageDType, Invalid

c0 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1576), (), 0)
c2 = UOp.range(1576, 20, AxisType.LOOP)
c5 = c2<55
c6 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 16, 4)), (), 1)
c8 = UOp.range(16, 0, AxisType.REDUCE)
c11 = UOp.range(4, 1, AxisType.REDUCE)
c14 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((14, 64, 4)), (), 2)
c25 = c5.where((c2%4*4+c11+c8*16+c2//4*256), UOp.const(dtypes.index, Invalid))
c27 = c6.index((c8*4+c11))*c14.index(c25)
c29 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(55), (), 3)
c30 = c5.where(c2, UOp.const(dtypes.index, Invalid))
c34 = c5.where((c27.reduce(c8, c11, arg=Ops.ADD)+c29.index(c30)), UOp.const(dtypes.float, 0.0))
c38 = c2<87
c39 = (c5!=True)&c38
c40 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 8, 4)), (), 4)
c42 = UOp.range(8, 2, AxisType.REDUCE)
c44 = UOp.range(4, 3, AxisType.REDUCE)
c47 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((8, 32, 4)), (), 5)
c49 = c2+1
c51 = c49%4*4
c57 = c49//4*128
c61 = c39.where((c51+c44+c42*16+c57+-1792), UOp.const(dtypes.index, Invalid))
c63 = c40.index((c42*4+c44))*c47.index(c61)
c65 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(32), (), 6)
c68 = c39.where((c2+-55), UOp.const(dtypes.index, Invalid))
c71 = c39.where((c63.reduce(c42, c44, arg=Ops.ADD)+c65.index(c68)), UOp.const(dtypes.float, 0.0))
c75 = c2<99
c76 = (c38!=True)&c75
c77 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 8, 4)), (), 7)
c78 = UOp.range(8, 4, AxisType.REDUCE)
c80 = UOp.range(4, 5, AxisType.REDUCE)
c83 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((3, 32, 4)), (), 8)
c90 = c76.where((c51+c80+c78*16+c57+-2816), UOp.const(dtypes.index, Invalid))
c92 = c77.index((c78*4+c80))*c83.index(c90)
c94 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(12), (), 9)
c97 = c76.where((c2+-87), UOp.const(dtypes.index, Invalid))
c100 = c76.where((c92.reduce(c78, c80, arg=Ops.ADD)+c94.index(c97)), UOp.const(dtypes.float, 0.0))
c104 = c2<105
c105 = (c75!=True)&c104
c106 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 8, 4)), (), 10)
c107 = UOp.range(8, 6, AxisType.REDUCE)
c109 = UOp.range(4, 7, AxisType.REDUCE)
c112 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((2, 32, 4)), (), 11)
c119 = c105.where((c51+c109+c107*16+c57+-3200), UOp.const(dtypes.index, Invalid))
c121 = c106.index((c107*4+c109))*c112.index(c119)
c123 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(6), (), 12)
c126 = c105.where((c2+-99), UOp.const(dtypes.index, Invalid))
c129 = c105.where((c121.reduce(c107, c109, arg=Ops.ADD)+c123.index(c126)), UOp.const(dtypes.float, 0.0))
c133 = c2<117
c134 = (c104!=True)&c133
c135 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 8, 4)), (), 13)
c136 = UOp.range(8, 8, AxisType.REDUCE)
c138 = UOp.range(4, 9, AxisType.REDUCE)
c141 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((3, 32, 4)), (), 14)
c143 = c2+3
c145 = c143%4*4
c149 = c143//4
c150 = c149*128
c154 = c134.where((c145+c138+c136*16+c150+-3456), UOp.const(dtypes.index, Invalid))
c156 = c135.index((c136*4+c138))*c141.index(c154)
c158 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(12), (), 15)
c161 = c134.where((c2+-105), UOp.const(dtypes.index, Invalid))
c164 = c134.where((c156.reduce(c136, c138, arg=Ops.ADD)+c158.index(c161)), UOp.const(dtypes.float, 0.0))
c168 = c2<645
c169 = (c133!=True)&c168
c170 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 16, 4)), (), 16)
c171 = UOp.range(16, 10, AxisType.REDUCE)
c173 = UOp.range(4, 11, AxisType.REDUCE)
c176 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((132, 64, 4)), (), 17)
c180 = c149*256
c184 = c169.where((c145+c173+c171*16+c180+-7680), UOp.const(dtypes.index, Invalid))
c186 = c170.index((c171*4+c173))*c176.index(c184)
c188 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(528), (), 18)
c191 = c169.where((c2+-117), UOp.const(dtypes.index, Invalid))
c194 = c169.where((c186.reduce(c171, c173, arg=Ops.ADD)+c188.index(c191)), UOp.const(dtypes.float, 0.0))
c198 = c2<653
c199 = (c168!=True)&c198
c200 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 4, 4)), (), 19)
c201 = UOp.range(4, 12, AxisType.REDUCE)
c203 = UOp.range(4, 13, AxisType.REDUCE)
c206 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((2, 16, 4)), (), 20)
c215 = c199.where((c145+c203+c201*16+c149*64+-10368), UOp.const(dtypes.index, Invalid))
c217 = c200.index((c201*4+c203))*c206.index(c215)
c219 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(8), (), 21)
c222 = c199.where((c2+-645), UOp.const(dtypes.index, Invalid))
c225 = c199.where((c217.reduce(c201, c203, arg=Ops.ADD)+c219.index(c222)), UOp.const(dtypes.float, 0.0))
c229 = c2<917
c230 = (c198!=True)&c229
c231 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 8, 4)), (), 22)
c232 = UOp.range(8, 14, AxisType.REDUCE)
c234 = UOp.range(4, 15, AxisType.REDUCE)
c237 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((66, 32, 4)), (), 23)
c244 = c230.where((c145+c234+c232*16+c150+-20992), UOp.const(dtypes.index, Invalid))
c246 = c231.index((c232*4+c234))*c237.index(c244)
c248 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(264), (), 24)
c251 = c230.where((c2+-653), UOp.const(dtypes.index, Invalid))
c254 = c230.where((c246.reduce(c232, c234, arg=Ops.ADD)+c248.index(c251)), UOp.const(dtypes.float, 0.0))
c258 = c2<1061
c259 = (c229!=True)&c258
c260 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 16, 4)), (), 25)
c261 = UOp.range(16, 16, AxisType.REDUCE)
c263 = UOp.range(4, 17, AxisType.REDUCE)
c266 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((36, 64, 4)), (), 26)
c273 = c259.where((c145+c263+c261*16+c180+-58880), UOp.const(dtypes.index, Invalid))
c275 = c260.index((c261*4+c263))*c266.index(c273)
c277 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(144), (), 27)
c280 = c259.where((c2+-917), UOp.const(dtypes.index, Invalid))
c283 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(144), (), 28)
c286 = c259.where(((c275.reduce(c261, c263, arg=Ops.ADD)+c277.index(c280))*c283.index(c280)), UOp.const(dtypes.float, 0.0))
c290 = c2<1064
c291 = (c258!=True)&c290
c292 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 4, 4)), (), 29)
c293 = UOp.range(4, 18, AxisType.REDUCE)
c295 = UOp.range(4, 19, AxisType.REDUCE)
c298 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 16, 4)), (), 30)
c305 = c291.where((c2*4+c295+c293*16+-4244), UOp.const(dtypes.index, Invalid))
c307 = c292.index((c293*4+c295))*c298.index(c305)
c309 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(3), (), 31)
c312 = c291.where((c2+-1061), UOp.const(dtypes.index, Invalid))
c315 = c291.where((c307.reduce(c293, c295, arg=Ops.ADD)+c309.index(c312)), UOp.const(dtypes.float, 0.0))
c317 = UOp(Ops.DEFINE_GLOBAL, dtypes.imageh((1, 128, 4)), (), 32)
c321 = (c290!=True).where((c2+-1064), UOp.const(dtypes.index, Invalid))
c323 = UOp(Ops.DEFINE_GLOBAL, dtypes.float.ptr(1), (), 33)
c328 = c290.where(UOp.const(dtypes.float, 0.0), (c317.index(c321)*c323.index(UOp.const(dtypes.index, 0)).reciprocal()))
c329 = c34+c71+c100+c129+c164+c194+c225+c254+c286+c315+c328
c331 = c0.index(c2, ptr=True).store(c329).end(c2)
ast = c331.sink(arg=KernelInfo(name="cat", opts_to_apply=None))

compiler = Device.default.compiler
renderer = Device.default.renderer
allocator = Device.default.allocator

uops = full_rewrite(ast, renderer)
src = renderer.render(uops)

# NOLOCALS=1 IMAGE=2 DEV=CL
lib = compiler.compile(src)

ps = ProgramSpec("cat", src, Device.DEFAULT, ast, uops)
# print(ps.src)
# print(ps.applied_opts)
# NOTE: this is faster with no GROUP and with NOLOCALS
# (Opt(op=OptOps.UPCAST, axis=1, arg=4), Opt(op=OptOps.UNROLL, axis=19, arg=4), Opt(op=OptOps.UNROLL, axis=17, arg=4), Opt(op=OptOps.UNROLL, axis=15, arg=4), Opt(op=OptOps.UNROLL, axis=13, arg=4), Opt(op=OptOps.UNROLL, axis=11, arg=4), Opt(op=OptOps.UNROLL, axis=9, arg=4), Opt(op=OptOps.UNROLL, axis=7, arg=4), Opt(op=OptOps.UNROLL, axis=5, arg=4), Opt(op=OptOps.UNROLL, axis=3, arg=4), Opt(op=OptOps.UNROLL, axis=1, arg=4), Opt(op=OptOps.NOLOCALS, axis=None, arg=None))
cr = CompiledRunner(ps, precompiled=lib)

gs = sorted(dedup([u for u in ast.toposort() if u.op is Ops.DEFINE_GLOBAL]), key=lambda u: u.arg)
print(len(gs))
print([g.dtype for g in gs])

bufs = [Buffer(ps.device, g.size, g.dtype if isinstance(g.dtype, ImageDType) else g.dtype._base).ensure_allocated() for g in gs]

t = cr(bufs, wait=True)
print(f"{t*1e6:.2f} us")