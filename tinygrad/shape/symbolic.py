from __future__ import annotations
import math
from typing import List, Dict, Callable, Type
from tinygrad.helpers import partition, modn, all_same

class Node:
  b: int
  min: int
  max: int
  def render(self, ops=None, ctx=None):
    if ops is None: ops = render_python
    if self.min == self.max and type(self) != NumNode: return NumNode(self.min).render(ops, ctx)
    return ops[type(self)](self, ops, ctx)
  def __add__(self, b:int): return Variable.sum([self, Variable.num(b)])
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self
    return MulNode(self, b)
  def __floordiv__(self, b:int):
    assert b != 0
    if b == 1: return self
    if isinstance(self, MulNode) and self.b%b == 0: return self.a*(self.b//b)
    if isinstance(self, MulNode) and b%self.b == 0: return self.a//(b//self.b)
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
    return DivNode(self, b)
  def __mod__(self, b:int):
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
    return ModNode(a, b)
  def __ge__(self, b:int):
    if self.max < b: return Variable.num(0)
    if self.min >= b: return Variable.num(1)
    return GeNode(self, b)
  def __lt__(self, b:int):
    if self.max < b: return Variable.num(1)
    if self.min >= b: return Variable.num(0)
    return LtNode(self, b)

  @staticmethod
  def num(num:int) -> Node:
    return NumNode(num)

  @staticmethod
  def sum(nodes:List[Node]) -> Node:
    nodes, num_nodes = partition(nodes, lambda x: not isinstance(x, NumNode))
    num_sum = sum([x.b for x in num_nodes])
    # TODO: this is broken due to something with negatives mods
    if num_sum > 0: nodes.append(NumNode(num_sum))
    else: nodes += [NumNode(x.b) for x in num_nodes if x.b != 0]

    if any([isinstance(x, SumNode) for x in nodes]):
      nodes, sum_nodes = partition(nodes, lambda x: not isinstance(x, SumNode))
      for x in sum_nodes: nodes += x.nodes
      return Variable.sum(nodes)
    nodes = [x for x in nodes if x.min != 0 or x.max != 0]
    if len(nodes) == 0: return NumNode(0)
    elif len(nodes) == 1: return nodes[0]
    return SumNode(nodes)

  @staticmethod
  def ands(nodes:List[Node]) -> Node:
    if any((x.min == 0 and x.max == 0) for x in nodes): return NumNode(0)
    nodes = [x for x in nodes if x.min != x.max]
    if len(nodes) == 0: return NumNode(1)
    elif len(nodes) == 1: return nodes[0]
    return AndNode(nodes)

# 4 basic node types

class Variable(Node):
  def __init__(self, expr:str, nmin:int, nmax:int):
    self.expr, self.min, self.max = expr, nmin, nmax

class NumNode(Node):
  def __init__(self, num:int):
    self.b, self.min, self.max = num, num, num

class OpNode(Node):
  def __init__(self, a:Node, b:int):
    self.a, self.b = a, b
    self.min, self.max = self.minmax(a,b)
  minmax = staticmethod(lambda a,b: (1//0, 1//0))

class RedNode(Node):
  def __init__(self, nodes:List[Node]):
    self.nodes = nodes
    self.min, self.max = self.minmax(nodes)
  minmax = staticmethod(lambda nodes: (1//0, 1//0))

# operation nodes

class MulNode(OpNode): minmax = staticmethod(lambda a,b: (a.min*b, a.max*b))
class DivNode(OpNode): minmax = staticmethod(lambda a,b: (int(a.min/b), int(a.max/b)))
# TODO: next three could be better
class ModNode(OpNode): minmax = staticmethod(lambda a,b: (min(max(a.min,-b+1),0),max(min(a.max,b-1),0)))
class GeNode(OpNode): minmax = staticmethod(lambda a,b: (0,1))
class LtNode(OpNode): minmax = staticmethod(lambda a,b: (0,1))

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
  SumNode: lambda self,ops,ctx: f"({'+'.join([x.render(ops) for x in self.nodes])})",
  AndNode: lambda self,ops,ctx: f"({'&&'.join([x.render(ops) for x in self.nodes])})"
}