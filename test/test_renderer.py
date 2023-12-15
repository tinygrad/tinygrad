from tinygrad.helpers import dtypes
from tinygrad.ops import BinaryOps, UnaryOps
from tinygrad.renderer.cstyle import HIPLanguage

class TestHIPRenderer():
  def test_code_for_op(self):
    lang = HIPLanguage()

    assert lang.code_for_op[UnaryOps.EXP2]("a", dtypes.half) == "hexp2((half)a)"
    assert lang.code_for_op[UnaryOps.EXP2]("a", dtypes.float) == "exp2(a)"
    assert lang.code_for_op[UnaryOps.EXP2]("a", dtypes.float.vec(4)) == "float4(exp2(a.x), exp2(a.y), exp2(a.z), exp2(a.w))"

    assert lang.code_for_op[UnaryOps.SIN]("a", dtypes.half) == "hsin((half)a)"
    assert lang.code_for_op[UnaryOps.SIN]("a", dtypes.float) == "sin(a)"
    assert lang.code_for_op[UnaryOps.SIN]("a", dtypes.float.vec(4)) == "float4(sin(a.x), sin(a.y), sin(a.z), sin(a.w))"

    assert lang.code_for_op[UnaryOps.NEG]("a", dtypes.half) == "(-(half)a)"
    assert lang.code_for_op[UnaryOps.NEG]("a", dtypes.bool) == "(!a)"
    assert lang.code_for_op[UnaryOps.NEG]("a", dtypes.float) == "(-a)"

    assert lang.code_for_op[BinaryOps.ADD]("a", "b", dtypes.float) == "(a+b)"
    assert lang.code_for_op[BinaryOps.ADD]("a", "b", dtypes.half) == "__hadd((half)a,(half)b)"

    assert lang.code_for_op[BinaryOps.SUB]("a", "b", dtypes.float) == "(a-b)"
    assert lang.code_for_op[BinaryOps.SUB]("a", "b", dtypes.half) == "__hsub((half)a,(half)b)"

    assert lang.code_for_op[BinaryOps.MUL]("a", "b", dtypes.float) == "(a*b)"
    assert lang.code_for_op[BinaryOps.MUL]("a", "b", dtypes.half) == "__hmul((half)a,(half)b)"

    assert lang.code_for_op[BinaryOps.DIV]("a", "b", dtypes.float) == "(a/b)"
    assert lang.code_for_op[BinaryOps.DIV]("a", "b", dtypes.half) == "__hdiv((half)a,(half)b)"

    assert lang.code_for_op[BinaryOps.MAX]("a", "b", dtypes.float) == "max(a,b)"
    assert lang.code_for_op[BinaryOps.MAX]("a", "b", dtypes.half) == "__hgt((half)a,(half)b)?(half)a:(half)b"

    assert lang.code_for_op[BinaryOps.MOD]("a", "b", dtypes.float) == "a%b"
    assert lang.code_for_op[BinaryOps.MOD]("a", "b", dtypes.half) == "__hsub(a, __hmul(b, __float2half(floorf(__half2float(a) / __half2float(b)))))"
