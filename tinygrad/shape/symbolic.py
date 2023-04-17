from __future__ import annotations
import math, itertools, functools
from typing import List, Dict, Callable, Type, Union
from tinygrad.helpers import partition, all_same

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code output is agnostic, and will never have negative numbers in div or mod

class Node:
  b: int
  min: int
  max: int
  def render(self, ops=None, ctx=None) -> str:
    if ops is None: ops = render_python
    assert isinstance(self, (Variable, NumNode)) or self.min != self.max
    return ops[type(self)](self, ops, ctx)
  @functools.cached_property
  def key(self) -> str: return self.render(ctx="DEBUG")
  def __repr__(self): return "<"+self.key+">"
  def __eq__(self, other:object) -> bool:
    if not isinstance(other, Node): return NotImplemented
    return self.key == other.key
  def __neg__(self): return self*-1
  def __add__(self, b:Union[Node, int]): return Variable.sum([self, b if isinstance(b, Node) else Variable.num(b)])
  def __sub__(self, b:Union[Node, int]): return self+-b
  def __ge__(self, b:int): return create_opnode(GeNode, self, b)
  def __lt__(self, b:int): return create_opnode(LtNode, self, b)
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self

    # this is a hack to make div work with boolean nodes. TODO: make generic
    if isinstance(self, GeNode): return (self.a*b) >= (self.b*b)
    if isinstance(self, LtNode): return (self.a*b) < (self.b*b)
    if isinstance(self, AndNode): return Variable.ands([x*b for x in self.nodes])

    if isinstance(self, MulNode): return self.a*(self.b*b) # two muls is one mul
    if isinstance(self, SumNode): return Variable.sum([x*b for x in self.nodes]) # distribute mul into sum
    return create_opnode(MulNode, self, b)

  # *** complex ops ***

  def __floordiv__(self, b:int, factoring_allowed=True):
    assert b != 0
    if b < 0: return (self//-b)*-1
    if b == 1: return self

    # this is a hack to make div work with boolean nodes. TODO: make generic
    if isinstance(self, GeNode): return (self.a//b) >= (self.b//b)
    if isinstance(self, LtNode): return (self.a//b) < (self.b//b)
    if isinstance(self, AndNode): return Variable.ands([x//b for x in self.nodes])

    if isinstance(self, ModNode) and self.b % b == 0: return (self.a//b) % (self.b//b) # put the div inside mod
    if isinstance(self, DivNode): return self.a//(self.b*b) # two divs is one div
    if isinstance(self, MulNode) and self.b % b == 0: return self.a*(self.b//b)
    if isinstance(self, MulNode) and b % self.b == 0 and self.b > 0: return self.a//(b//self.b) # NOTE: mod negative isn't handled right
    if isinstance(self, SumNode) and factoring_allowed:
      factors, tmp_nofactor = partition(self.nodes, lambda x: (isinstance(x, (MulNode, NumNode))) and x.b%b == 0)
      nofactor = []
      # ugh, i doubt this is universally right
      for x in tmp_nofactor:
        if isinstance(x, NumNode):
          if (x.b%b) != x.b:
            factors.append(Variable.num(x.b - (x.b%b)))  # python does floor division
          nofactor.append(Variable.num(x.b%b))
        else:
          nofactor.append(x)
      gcd = [math.gcd(x.b, b) if isinstance(x, (MulNode, NumNode)) else None for x in nofactor]
      if len(factors) > 0:
        # these don't have to be the same, just having a common factor
        if len(gcd) > 0 and all_same(gcd) and gcd[0] is not None and gcd[0] > 1:
          nofactor_term = Variable.sum([(x.a * (x.b//gcd[0])) if isinstance(x, MulNode) else Variable.num(x.b//gcd[0]) for x in nofactor])//(b//gcd[0])
        else:
          nofactor_term = Variable.sum(nofactor)//b
        return Variable.sum([(x.a * (x.b//b)) if isinstance(x, MulNode) else Variable.num(x.b//b) for x in factors] + [nofactor_term])
      else:
        muls = [x.b for x in nofactor if isinstance(x, MulNode)]
        for m in muls:
          if m > 1 and b%m == 0:
            return (self//m)//(b//m)
    # the numerator of div is not allowed to be negative
    if self.min < 0:
      offset = self.min//b
      # factor out an "offset" to make the numerator positive. don't allowing factoring again
      return (self + -offset*b).__floordiv__(b, factoring_allowed=False) + offset
    return create_opnode(DivNode, self, b)

  def __mod__(self, b:int):
    assert b > 0
    if b == 1: return NumNode(0)
    if isinstance(self, SumNode):
      new_nodes = []
      for x in self.nodes:
        if isinstance(x, NumNode): new_nodes.append(Variable.num(x.b%b))
        elif isinstance(x, MulNode): new_nodes.append(x.a * (x.b%b))
        else: new_nodes.append(x)
      a = Variable.sum(new_nodes)
    elif isinstance(self, MulNode):
      a = self.a * (self.b%b)
    else:
      a = self
    if a.min >= 0 and a.max < b: return a
    if a.min < 0: return (a - ((a.min//b)*b)) % b
    return create_opnode(ModNode, a, b)

  @staticmethod
  def num(num:int) -> Node: return NumNode(num)

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    # expand any sums inside one sum
    if any([isinstance(x, SumNode) for x in nodes]):
      nodes, sum_nodes = partition(nodes, lambda x: not isinstance(x, SumNode))
      for x in sum_nodes: nodes += x.nodes
      return Variable.sum(nodes)

    # combine any numbers inside a sum
    nodes, num_nodes = partition(nodes, lambda x: not isinstance(x, NumNode))
    nodes.append(NumNode(sum([x.b for x in num_nodes])))

    # combine any MulNodes that factorize (big hack sticking the MulNode(x, 1) on things)
    # TODO: this is slow!
    nodes, mul_nodes = partition(nodes, lambda x: not isinstance(x, MulNode))
    mul_nodes += [MulNode(x, 1) for x in nodes]
    mul_nodes = sorted(mul_nodes, key=lambda x: x.a.render()) # group by equality (ugh, uses render!)
    new_nodes = [k * sum(x.b for x in g) for k, g in itertools.groupby(mul_nodes, key=lambda x: x.a)]
    nodes = [x if not isinstance(x, MulNode) or x.b != 1 else x.a for x in new_nodes]

    # filter 0s
    nodes = [x for x in nodes if x.min != 0 or x.max != 0]
    return create_rednode(SumNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(0))

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if any((x.min == 0 and x.max == 0) for x in nodes): return NumNode(0)

    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return create_rednode(AndNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

class Variable(Node):
  def __new__(cls, expr:str, nmin:int, nmax:int):
    assert nmin >= 0 and nmin <= nmax
    if nmin == nmax: return NumNode(nmin)
    return super().__new__(cls)

  def __init__(self, expr:str, nmin:int, nmax:int):
    self.expr, self.min, self.max = expr, nmin, nmax

class NumNode(Node):
  def __init__(self, num:int):
    self.b, self.min, self.max = num, num, num

def create_node(ret:Node):
  assert ret.min <= ret.max, f"min greater than max! {ret.min} {ret.max} when creating {type(ret)} {ret}"
  if ret.min == ret.max: return NumNode(ret.min)
  return ret

class OpNode(Node):
  def __init__(self, a:Node, b:int): self.a, self.b = a, b

class GeNode(OpNode): pass
class LtNode(OpNode): pass
class MulNode(OpNode): pass
class DivNode(OpNode): pass
class ModNode(OpNode): pass

def create_opnode(typ:Type[OpNode], a:Node, b:int):
  ret = typ(a, b)
  if typ == GeNode: ret.min, ret.max = int(a.min >= b), int(a.max >= b)
  elif typ == LtNode: ret.min, ret.max = int(a.max < b), int(a.min < b)
  elif typ == MulNode: ret.min, ret.max = (a.min*b, a.max*b) if b >= 0 else (a.max*b, a.min*b)
  elif typ == DivNode:
    assert a.min >= 0
    ret.min, ret.max = a.min//b, a.max//b
  elif typ == ModNode:
    assert a.min >= 0
    ret.min, ret.max = (0, b-1) if a.max - a.min >= b or (a.min != a.max and a.min%b >= a.max%b) else (a.min%b, a.max%b)
  return create_node(ret)

class RedNode(Node):
  def __init__(self, nodes:List[Node]): self.nodes = nodes

class SumNode(RedNode): pass
class AndNode(RedNode): pass

def create_rednode(typ:Type[RedNode], nodes:List[Node]):
  ret = typ(nodes)
  if typ == SumNode: ret.min, ret.max = (sum([x.min for x in nodes]), sum([x.max for x in nodes]))
  elif typ == AndNode: ret.min, ret.max = (min([x.min for x in nodes]), max([x.max for x in nodes]))
  return create_node(ret)

render_python: Dict[Type, Callable] = {
  Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}]" if ctx == "DEBUG" else f"{self.expr}",
  NumNode: lambda self,ops,ctx: f"{self.b}",
  MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{self.b})",
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  GeNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}>={self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{self.b})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({' and '.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"
}