from typing import List, Optional
from tinygrad.helpers import partition

class Node:
  pass

class Variable(Node):
  def __init__(self, expr:str, min:int, max:int):
    self.expr, self.min, self.max = expr, min, max
  def __str__(self):
    if self.min == self.max: return str(self.min)  # this is universal
    return self.expr
  def __add__(self, num:int): return AddNode(self, num)
  def __mul__(self, num:int): return MulNode(self, num)
  def __floordiv__(self, num:int): return DivNode(self, num)
  def __mod__(self, num:int): return ModNode(self, num)
  def __ge__(self, num:int): return GeNode(self, num)
  def __lt__(self, num:int): return LtNode(self, num)

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
  def __new__(cls, a:Variable, b:int):
    if b == 0: return NumNode(0)
    elif b == 1: return a
    return super().__new__(cls)
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min*b, a.max*b
  @property
  def expr(self):
    return f"({self.a}*{self.b})"

class DivNode(Variable):
  def __new__(cls, a:Variable, b:int):
    assert b != 0
    if b == 1: return a
    #if isinstance(a, MulNode) and a.b%b == 0: return MulNode(a.a, a.b//b)
    if isinstance(a, SumNode) and all((isinstance(x, MulNode) or isinstance(x, NumNode)) for x in a.nodes):
      factors, nofactor = partition(a.nodes, lambda x: x.b%b == 0)
      if len(factors) > 0: return SumNode([MulNode(x.a, x.b//b) if isinstance(x, MulNode) else NumNode(x.b//b) for x in factors] + [SumNode(nofactor)//b])
    return super().__new__(cls)
  def __init__(self, a:Variable, b:int):
    self.a, self.b = a, b
    self.min, self.max = a.min//b, a.max//b
  @property
  def expr(self):
    return f"({self.a}//{self.b})"

class ModNode(Variable):
  # TODO: why is this broken?
  def __new__(cls, a:Variable, b:int):
    if b == 1: return NumNode(0)
    # TODO: unduplicate this
    if isinstance(a, SumNode):
      a = SumNode([x for x in a.nodes if not (isinstance(x, MulNode) or isinstance(x, NumNode)) or (x.b%b != 0)])
    if a.min >= 0 and a.max < b: return a
    return super().__new__(cls)
  def __init__(self, a:Variable, b:int):
    if isinstance(a, SumNode):
      a = SumNode([x for x in a.nodes if not (isinstance(x, MulNode) or isinstance(x, NumNode)) or (x.b%b != 0)])
    self.a, self.b = a, b
    self.min, self.max = min(a.min, 0), max(a.max, b)
  @property
  def expr(self):
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
    return f"({'+'.join([str(x) for x in self.nodes if str(x) != '0'])})"

class AndNode(Variable):
  def __init__(self, nodes:List[Variable]):
    self.nodes = nodes
    self.min, self.max = min([x.min for x in nodes]), max([x.max for x in nodes])
  @property
  def expr(self):
    return f"({'&&'.join([str(x) for x in self.nodes])})"

