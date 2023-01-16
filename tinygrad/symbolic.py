from __future__ import annotations
from typing import List
from tinygrad.helpers import partition

class Variable:
  def __init__(self, expr:str, min:int, max:int):
    self.expr, self.min, self.max = expr, min, max
  def __str__(self):
    if self.min == self.max: return str(self.min)  # this is universal
    return self.expr
  def __add__(self, b:int): return AddNode(self, b)
  def __mul__(self, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return self
    return MulNode(self, b)
  def __floordiv__(self, b:int):
    assert b != 0
    if b == 1: return self
    if isinstance(self, SumNode) and all((isinstance(x, MulNode) or isinstance(x, NumNode)) for x in self.nodes):
      factors, nofactor = partition(self.nodes, lambda x: x.b%b == 0)
      if len(factors) > 0: return Variable.sum([(x.a * (x.b//b)) if isinstance(x, MulNode) else Variable.num(x.b//b) for x in factors] + [Variable.sum(nofactor)//b])
    return DivNode(self, b)
  def __mod__(self, b:int):
    if b == 1: return NumNode(0)
    if isinstance(self, SumNode):
      a = Variable.sum([x for x in self.nodes if not (isinstance(x, MulNode) or isinstance(x, NumNode)) or (x.b%b != 0)])
    else:
      a = self
    if a.min >= 0 and a.max < b: return a
    return ModNode(a, b)
  def __ge__(self, b:int): return GeNode(self, b)
  def __lt__(self, b:int): return LtNode(self, b)

  @staticmethod
  def num(num:int) -> Variable:
    return NumNode(num)

  @staticmethod
  def sum(nodes:List[Variable]) -> Variable:
    nodes = [x for x in nodes if x.min != 0 or x.max != 0]
    if len(nodes) == 0: return NumNode(0)
    elif len(nodes) == 1: return nodes[0]
    return SumNode(nodes)

  @staticmethod
  def ands(nodes:List[Variable]) -> Variable:
    if len(nodes) == 0: return NumNode(1)
    elif len(nodes) == 1: return nodes[0]
    return AndNode(nodes)

class NumNode(Variable):
  def __init__(self, num:int):
    self.expr, self.b, self.min, self.max = str(num), num, num, num

class AddNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min+b, a.max+b
  @property
  def expr(self):
    return f"({self.a}+{self.b})" if self.b != 0 else str(self.a)

class MulNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min*b, a.max*b
  @property
  def expr(self):
    return f"({self.a}*{self.b})"

class DivNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min//b, a.max//b
  @property
  def expr(self):
    return f"({self.a}//{self.b})"

class ModNode(Variable):
  def __init__(self, a:Variable, b:int):
    if isinstance(a, SumNode):
      a = Variable.sum([x for x in a.nodes if not (isinstance(x, MulNode) or isinstance(x, NumNode)) or (x.b%b != 0)])
    self.a, self.b = a, b
    self.min, self.max = min(a.min, 0), max(a.max, b)
  @property
  def expr(self):
    assert self.a != self
    return f"({self.a}%{self.b})"

class GeNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = 0, 1
  @property
  def expr(self):
    return f"({self.a}>={self.b})"

class LtNode(Variable):
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = 0, 1
  @property
  def expr(self):
    return f"({self.a}<{self.b})"

# reduce nodes

class SumNode(Variable):
  def __init__(self, nodes:List[Variable]):
    self.nodes = nodes
    self.min, self.max = sum([x.min for x in nodes]), sum([x.max for x in nodes])
  @property
  def expr(self):
    return f"({'+'.join([str(x) for x in self.nodes])})"

class AndNode(Variable):
  def __init__(self, nodes:List[Variable]):
    self.nodes = nodes
    self.min, self.max = min([x.min for x in nodes]), max([x.max for x in nodes])
  @property
  def expr(self):
    return f"({'&&'.join([str(x) for x in self.nodes])})"
