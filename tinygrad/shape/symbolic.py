from __future__ import annotations
import math, itertools
from typing import List, Dict, Callable, Type, Union
from tinygrad.helpers import partition, all_same

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code is outputs is agnostic

#def divn(x, a): return x//a if isinstance(x, Node) else int(x/a) 
#def modn(x, a): return x%a if isinstance(x, Node) else (-((-x)%a) if x < 0 else x%a)

#
def divn(x, a): return x//a
def modn(x, a): return x%a

class Node:
  b: int
  min: int
  max: int
  def __repr__(self): return "<"+self.render()+">"   # only use this for debugging!
  def render(self, ops=None, ctx=None):
    if ops is None: ops = render_python
    if self.min == self.max and type(self) != NumNode: return NumNode(self.min).render(ops, ctx)
    return ops[type(self)](self, ops, ctx)
  def __add__(self, b:Union[Node, int]): return Variable.sum([self, b if isinstance(b, Node) else Variable.num(b)])
  def __sub__(self, b:int): return self+-b
  def __ge__(self, b:int): return GeNode(self, b)
  def __lt__(self, b:int): return LtNode(self, b)
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self
    elif self.min == self.max: return NumNode(self.min*b)
    if isinstance(self, MulNode): return MulNode(self.a, self.b*b)
    # distribute mul into sum
    if isinstance(self, SumNode): return Variable.sum([x*b for x in self.nodes])
    return MulNode(self, b)

  # *** complex ops ***

  def __floordiv__(self, b:int):
    assert b > 0
    if b == 1: return self
    if isinstance(self, MulNode) and self.b % b == 0: return self.a*(self.b//b)
    if isinstance(self, MulNode) and b % self.b == 0: return self.a//(b//self.b)
    if isinstance(self, SumNode):
      factors, tmp_nofactor = partition(self.nodes, lambda x: (isinstance(x, (MulNode, NumNode))) and x.b%b == 0)
      nofactor = []
      # ugh, i doubt this is universally right
      for x in tmp_nofactor:
        if isinstance(x, NumNode):
          if modn(x.b, b) != x.b:
            factors.append(Variable.num(x.b - modn(x.b, b)))  # python does floor division
          nofactor.append(Variable.num(modn(x.b, b)))
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
    if self.min < 0:
      offset = self.min//b
      return (self+offset*b)//b - offset
    return DivNode(self, b)

  def __mod__(self, b:int):
    assert b > 0
    if b == 1: return NumNode(0)
    if isinstance(self, SumNode):
      new_nodes = []
      for x in self.nodes:
        if isinstance(x, NumNode): new_nodes.append(Variable.num(modn(x.b, b)))
        elif isinstance(x, MulNode): new_nodes.append(x.a * modn(x.b, b))
        else: new_nodes.append(x)
      a = Variable.sum(new_nodes)
    elif isinstance(self, MulNode):
      a = self.a * modn(self.b, b)
    else:
      a = self
    if a.min >= 0 and a.max < b: return a
    if a.min == a.max: return Variable.num(modn(a.min, b))
    if a.min < 0: return (a + ((a.min//b)*b)) % b
    return ModNode(a, b)

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

    # combine any MulNodes that factorize
    nodes, mul_nodes = partition(nodes, lambda x: not isinstance(x, MulNode))
    # group by equality (don't use render!)
    mul_nodes = sorted(mul_nodes, key=lambda x: x.a.render())
    new_mul_nodes = []
    for k, g in itertools.groupby(mul_nodes, key=lambda x: x.a):
      new_mul_nodes.append(k * sum(x.b for x in g))

    nodes += new_mul_nodes

    # filter 0s
    nodes = [x for x in nodes if x.min != 0 or x.max != 0]
    return SumNode(nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(0))

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if any((x.min == 0 and x.max == 0) for x in nodes): return NumNode(0)
    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return AndNode(nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

class Variable(Node):
  def __init__(self, expr:str, nmin:int, nmax:int):
    assert nmin >= 0 and nmin <= nmax
    self.expr, self.min, self.max = expr, nmin, nmax
  def __eq__(self, b:Variable): return isinstance(b, type(self)) and self.expr == b.expr

class NumNode(Node):
  def __init__(self, num:int):
    self.b, self.min, self.max = num, num, num
  def __eq__(self, b:NumNode): return isinstance(b, type(self)) and self.b == b.b

class OpNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = self.minmax(a,b)
  minmax = staticmethod(lambda a,b: (1//0, 1//0))
  def __eq__(self, b:OpNode): return isinstance(b, type(self)) and self.a == b.a and self.b == b.b

class RedNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
    self.min, self.max = self.minmax(nodes)
  minmax = staticmethod(lambda nodes: (1//0, 1//0))
  def __eq__(self, b:OpNode): return isinstance(b, type(self)) and self.nodes == b.nodes  # TODO: order doesn't matter

# operation nodes

class GeNode(OpNode): minmax = staticmethod(lambda a,b: (int(a.min >= b), int(a.max >= b)))
class LtNode(OpNode): minmax = staticmethod(lambda a,b: (int(a.max < b), int(a.min < b)))
class MulNode(OpNode): minmax = staticmethod(lambda a,b: (a.min*b, a.max*b))
class DivNode(OpNode):
  @staticmethod
  def minmax(a, b):
    assert a.min >= 0
    return a.min//b, a.max//b

# you have 3 included ranges
# range 1 from a.min -> max1 (smaller than a mod)
# range 2 from max1  -> min2
# range 3 from min2  -> a.max (smaller than a mod)
class ModNode(OpNode):
  @staticmethod
  def minmax(a, b):
    assert a.min >= 0
    max1, min2 = math.ceil(a.min/b)*b, math.floor(a.max/b)*b
    if max1 < min2: return (0, b-1)             # range 2 is the full distance
    if max1 > min2: return (a.min%b, a.max%b)   # range 2 doesn't exist, a.min -> a.max is smaller than a mod
    if a.min == max1: return (0, a.max%b)       # range 3 is the only valid
    return (0, b-1)                             # range 1 and 3 are valid

# reduce nodes

class SumNode(RedNode): minmax = staticmethod(lambda nodes: (sum([x.min for x in nodes]), sum([x.max for x in nodes])))
class AndNode(RedNode): minmax = staticmethod(lambda nodes: (min([x.min for x in nodes]), max([x.max for x in nodes])))

render_python : Dict[Type, Callable] = {
  Variable: lambda self,ops,ctx: f"{self.expr}",
  NumNode: lambda self,ops,ctx: f"{self.b}",
  MulNode: lambda self,ops,ctx: f"({self.a.render(ops)}*{self.b})",
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops)}%{self.b})",
  GeNode: lambda self,ops,ctx: f"({self.a.render(ops)}>={self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops)}<{self.b})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({'&&'.join(sorted([x.render(ops) for x in self.nodes]))})"
}